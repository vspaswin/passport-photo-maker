"""Strict quality checks for Indian passport / VFS-style photos.

Goal: only accept source photos that are likely to produce a submittable
result, and re-check the final framed output before offering downloads.

Automated checks cannot legally guarantee government acceptance, but we
refuse conversion when any high-risk defect is detected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .specs import PhotoSpec


@dataclass
class ValidationIssue:
    code: str
    message: str
    how_to_fix: str
    severity: str = "error"  # only "error" for strict mode

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ValidationReport:
    passed: bool
    stage: str  # "source" | "output" | "full"
    issues: List[ValidationIssue] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "issues": [i.to_dict() for i in self.issues],
            "checks": self.checks,
        }


class PhotoValidationError(Exception):
    """Raised when a photo fails strict passport validation."""

    def __init__(self, report: ValidationReport, message: Optional[str] = None):
        self.report = report
        self.message = message or _default_message(report)
        super().__init__(self.message)


def _default_message(report: ValidationReport) -> str:
    n = len(report.issues)
    if n == 0:
        return "Photo failed validation."
    if n == 1:
        return f"Photo rejected: {report.issues[0].message}"
    return f"Photo rejected with {n} issues. Fix them and try again."


def _load_cascades():
    base = Path(cv2.data.haarcascades)
    face = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
    eye_tree = cv2.CascadeClassifier(str(base / "haarcascade_eye_tree_eyeglasses.xml"))
    return face, eye, eye_tree


def _to_bgr_gray(im: Image.Image) -> Tuple[np.ndarray, np.ndarray, int, int]:
    rgb = np.array(im.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    return bgr, gray_eq, w, h


def _detect_faces(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    face_cascade, _, _ = _load_cascades()
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
        )
    if len(faces) == 0:
        return []
    # Sort largest first
    boxes = [tuple(map(int, f)) for f in faces]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    # Drop nested / overlapping duplicates (common Haar false doubles)
    cleaned: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        duplicate = False
        for cx, cy, cw, ch in cleaned:
            # IoU-ish: if centers are close relative to size, keep larger only
            if abs((x + w / 2) - (cx + cw / 2)) < 0.35 * max(w, cw) and abs(
                (y + h / 2) - (cy + ch / 2)
            ) < 0.35 * max(h, ch):
                duplicate = True
                break
        if not duplicate:
            cleaned.append(box)
    return cleaned


def _detect_eyes(
    gray: np.ndarray, face: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    _, eye_cascade, eye_tree = _load_cascades()
    x, y, w, h = face
    # Eyes are in upper portion of face box
    roi_y = y + int(0.12 * h)
    roi_h = int(0.55 * h)
    roi = gray[roi_y : y + roi_h, x : x + w]
    if roi.size == 0:
        return []

    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))
    if len(eyes) < 2:
        eyes = eye_tree.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(16, 16))

    out = []
    for ex, ey, ew, eh in eyes:
        out.append((x + int(ex), roi_y + int(ey), int(ew), int(eh)))
    # Prefer two largest
    out.sort(key=lambda b: b[2] * b[3], reverse=True)
    return out[:4]


def _laplacian_var(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


def _face_brightness_stats(gray: np.ndarray, face: Tuple[int, int, int, int]) -> Dict[str, float]:
    x, y, w, h = face
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(roi.mean()), "std": float(roi.std())}


def _estimate_underexposed_overexposed(gray: np.ndarray, face: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = face
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return 1.0, 1.0
    dark = float((roi < 40).mean())
    bright = float((roi > 235).mean())
    return dark, bright


def _skin_ratio(bgr: np.ndarray, face: Tuple[int, int, int, int]) -> float:
    """Rough skin presence inside face box — inclusive of a wide range of tones."""
    x, y, w, h = face
    # Use central face (avoid hair/background)
    x0 = x + int(0.15 * w)
    x1 = x + int(0.85 * w)
    y0 = y + int(0.20 * h)
    y1 = y + int(0.85 * h)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    # Wider Cr/Cb ranges than classic OpenCV demo (better for diverse skin tones)
    lower = np.array([0, 125, 70], dtype=np.uint8)
    upper = np.array([255, 180, 140], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    # Also accept HSV skin-ish warm tones
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([30, 200, 255]))
    hsv_mask2 = cv2.inRange(hsv, np.array([150, 15, 40]), np.array([180, 200, 255]))
    combined = cv2.bitwise_or(mask, cv2.bitwise_or(hsv_mask, hsv_mask2))
    return float(combined.mean() / 255.0)


def _white_clothing_risk(bgr: np.ndarray, face: Tuple[int, int, int, int], img_h: int) -> float:
    """Fraction of very bright low-chroma pixels in torso band under chin."""
    x, y, w, h = face
    # Band below face
    y0 = min(img_h - 1, y + h)
    y1 = min(img_h, y + int(h * 1.8))
    x0 = max(0, x - int(0.1 * w))
    x1 = min(bgr.shape[1], x + w + int(0.1 * w))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    mn = rgb.min(axis=2)
    mx = rgb.max(axis=2)
    lum = rgb.mean(axis=2)
    chroma = mx - mn
    whiteish = (lum >= 200) & (chroma <= 25)
    return float(whiteish.mean())


def validate_source_photo(
    im: Image.Image,
    spec: PhotoSpec,
    require_bg_removal: bool = True,
) -> ValidationReport:
    """Strict 'as-is' check: is this already a good passport candidate?

    For conversion eligibility use :func:`validate_source_convertible` instead.
    """
    return validate_source_as_is(im, spec, require_bg_removal=require_bg_removal)


def validate_source_as_is(
    im: Image.Image,
    spec: PhotoSpec,
    require_bg_removal: bool = True,
) -> ValidationReport:
    """Strict validation of the *current* photo against passport expectations."""
    issues: List[ValidationIssue] = []
    checks: Dict[str, Any] = {}

    w, h = im.size
    checks["image_width"] = w
    checks["image_height"] = h
    checks["megapixels"] = round((w * h) / 1_000_000, 2)

    # --- Resolution ---
    min_side = min(w, h)
    if min_side < 400:
        issues.append(
            ValidationIssue(
                code="resolution_too_low",
                message="Image resolution is too low for a sharp passport print.",
                how_to_fix="Use a higher-resolution photo (at least ~800×800, ideally phone camera full quality).",
            )
        )
    checks["min_side_px"] = min_side

    bgr, gray, gw, gh = _to_bgr_gray(im)

    # --- Face count ---
    faces = _detect_faces(gray)
    checks["face_count"] = len(faces)

    if len(faces) == 0:
        issues.append(
            ValidationIssue(
                code="no_face",
                message="No face detected.",
                how_to_fix="Use a clear front-facing photo of one person, looking at the camera, with the face fully visible.",
            )
        )
        return ValidationReport(
            passed=False, stage="source_as_is", issues=issues, checks=checks
        )

    if len(faces) > 1:
        # Only flag secondary faces that are large (real second person), not Haar noise
        primary = faces[0]
        p_area = primary[2] * primary[3]
        significant = [
            f for f in faces[1:] if (f[2] * f[3]) > 0.45 * p_area
        ]
        checks["secondary_significant_faces"] = len(significant)
        if significant:
            issues.append(
                ValidationIssue(
                    code="multiple_faces",
                    message=f"Multiple faces detected ({1 + len(significant)} significant).",
                    how_to_fix="Photograph only one person. No other people in the frame.",
                )
            )

    face = faces[0]
    fx, fy, fw, fh = face
    face_area_ratio = (fw * fh) / float(w * h)
    checks["face_box"] = {"x": fx, "y": fy, "w": fw, "h": fh}
    checks["face_area_ratio"] = round(face_area_ratio, 4)

    # Face should be reasonably large in the frame for quality crop
    if face_area_ratio < 0.03:
        issues.append(
            ValidationIssue(
                code="face_too_small",
                message="Face is too small in the photo.",
                how_to_fix="Move closer so the head and shoulders fill most of the frame (passport-style distance).",
            )
        )
    if face_area_ratio > 0.65:
        issues.append(
            ValidationIssue(
                code="face_too_close",
                message="Face fills almost the entire frame; top of head or chin may be cropped.",
                how_to_fix="Step back slightly so full head (hair to chin) and some shoulders are visible.",
            )
        )

    # Face not cut off at image borders
    margin = 0.02
    if fx < w * margin or fy < h * margin or (fx + fw) > w * (1 - margin) or (fy + fh) > h * (1 - margin * 0.5):
        # Top of hair often above face box — only flag strong bottom/side clipping
        side_clip = fx <= 2 or (fx + fw) >= w - 2
        bottom_clip = (fy + fh) >= h - 2
        if side_clip or bottom_clip:
            issues.append(
                ValidationIssue(
                    code="face_clipped",
                    message="Face appears cut off at the edge of the photo.",
                    how_to_fix="Center yourself with space around the head; do not crop the chin, ears, or top of head.",
                )
            )

    # Horizontal centering (rough)
    face_cx = fx + fw / 2
    offset = abs(face_cx - w / 2) / w
    checks["face_horizontal_offset"] = round(offset, 3)
    if offset > 0.28:
        issues.append(
            ValidationIssue(
                code="face_not_centered",
                message="Face is not reasonably centered in the frame.",
                how_to_fix="Stand centered in front of the camera and look straight ahead.",
            )
        )

    # --- Blur ---
    face_roi = gray[fy : fy + fh, fx : fx + fw]
    blur = _laplacian_var(face_roi)
    checks["sharpness_laplacian_var"] = round(blur, 1)
    # Laplacian variance on face ROI — fail only clearly unusable blur.
    # (AI-smoothed / soft beauty-filter faces often score ~18–30 but are still printable.)
    if blur < 12:
        issues.append(
            ValidationIssue(
                code="too_blurry",
                message="Photo is too blurry / out of focus.",
                how_to_fix="Hold the camera steady, use good light, tap to focus on the face, and retake. Avoid heavy beauty filters.",
            )
        )

    # --- Exposure ---
    bright = _face_brightness_stats(gray, face)
    checks["face_brightness_mean"] = round(bright["mean"], 1)
    checks["face_brightness_std"] = round(bright["std"], 1)
    dark_frac, bright_frac = _estimate_underexposed_overexposed(gray, face)
    checks["face_dark_fraction"] = round(dark_frac, 3)
    checks["face_bright_fraction"] = round(bright_frac, 3)

    # Mean luminance varies a lot by skin tone; only fail extreme underexposure.
    if bright["mean"] < 32:
        issues.append(
            ValidationIssue(
                code="too_dark",
                message="Face is too dark (underexposed).",
                how_to_fix="Use even front lighting (face a window or soft lamp). Avoid strong backlight.",
            )
        )
    if bright["mean"] > 220:
        issues.append(
            ValidationIssue(
                code="too_bright",
                message="Face is overexposed / washed out.",
                how_to_fix="Reduce harsh light or flash; use softer, even lighting.",
            )
        )
    if dark_frac > 0.40:
        issues.append(
            ValidationIssue(
                code="uneven_shadows",
                message="Large dark regions on the face (shadows).",
                how_to_fix="Use even lighting on both sides of the face; avoid side-only light.",
            )
        )
    if bright_frac > 0.35:
        issues.append(
            ValidationIssue(
                code="hotspots",
                message="Overexposed hotspots on the face.",
                how_to_fix="Avoid direct flash glare; use diffused light.",
            )
        )

    # --- Skin / person plausibility ---
    skin = _skin_ratio(bgr, face)
    checks["skin_ratio"] = round(skin, 3)
    # Low bar: only reject obvious non-faces (walls, objects). Diverse skin tones vary widely.
    if skin < 0.03:
        issues.append(
            ValidationIssue(
                code="face_not_personlike",
                message="Detected region does not look like a natural face / skin tones.",
                how_to_fix="Upload a real colour photo of a person (not a drawing, filter-heavy, or non-face image).",
            )
        )

    # --- Eyes open / visible ---
    eyes = _detect_eyes(gray, face)
    checks["eye_count"] = len(eyes)

    if len(eyes) < 2:
        issues.append(
            ValidationIssue(
                code="eyes_not_detected",
                message="Could not clearly detect both open eyes.",
                how_to_fix="Look straight at the camera with both eyes open. Remove sunglasses / heavy glare. Keep hair off the eyes.",
            )
        )
    else:
        # Sort left/right by x
        e_sorted = sorted(eyes[:2], key=lambda e: e[0])
        left, right = e_sorted[0], e_sorted[1]
        # Eyes roughly same height (head not tilted too much)
        ly = left[1] + left[3] / 2
        ry = right[1] + right[3] / 2
        eye_tilt = abs(ly - ry) / max(fh, 1)
        checks["eye_tilt_ratio"] = round(eye_tilt, 3)
        if eye_tilt > 0.12:
            issues.append(
                ValidationIssue(
                    code="head_tilted",
                    message="Head appears tilted (eyes not level).",
                    how_to_fix="Keep your head straight — both eyes on a horizontal line, face the camera directly.",
                )
            )

        # Horizontal separation should be reasonable (not profile)
        lx = left[0] + left[2] / 2
        rx = right[0] + right[2] / 2
        eye_sep = abs(rx - lx) / max(fw, 1)
        checks["eye_separation_ratio"] = round(eye_sep, 3)
        if eye_sep < 0.22:
            issues.append(
                ValidationIssue(
                    code="not_frontal",
                    message="Face may not be fully frontal (eyes too close together in the frame).",
                    how_to_fix="Face the camera directly (not a side or three-quarter pose).",
                )
            )

        # Dark glasses heuristic: very dark mean inside eye boxes
        dark_eyes = 0
        for ex, ey, ew, eh in eyes[:2]:
            eroi = gray[ey : ey + eh, ex : ex + ew]
            if eroi.size and float(eroi.mean()) < 35:
                dark_eyes += 1
        checks["dark_eye_boxes"] = dark_eyes
        if dark_eyes >= 2:
            issues.append(
                ValidationIssue(
                    code="possible_dark_glasses",
                    message="Eyes look covered by dark / tinted glasses.",
                    how_to_fix="Remove tinted or dark glasses. Clear glasses only if no glare (prefer none).",
                )
            )

    # --- Mouth roughly present (lower face not missing) ---
    # Contrast in lower third of face
    lower = face_roi[int(fh * 0.55) :, :]
    checks["lower_face_std"] = round(float(lower.std()) if lower.size else 0.0, 1)
    if lower.size and float(lower.std()) < 8:
        issues.append(
            ValidationIssue(
                code="lower_face_unclear",
                message="Lower face / chin area is unclear.",
                how_to_fix="Ensure chin and mouth are visible, uncovered, and in focus.",
            )
        )

    # --- White clothing risk (VFS: not pure white attire) ---
    white_cloth = _white_clothing_risk(bgr, face, h)
    checks["white_clothing_ratio"] = round(white_cloth, 3)
    if white_cloth > 0.70:
        issues.append(
            ValidationIssue(
                code="white_clothing",
                message="Clothing appears mostly white (often rejected against a white background).",
                how_to_fix="Wear a medium-coloured top (e.g. blue, not pure white).",
            )
        )

    # --- Colour photo (not grayscale) ---
    rgb = np.array(im.convert("RGB")).astype(np.float32)
    # Mean channel differences
    colourfulness = float(
        np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))
        + np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))
    )
    checks["colourfulness"] = round(colourfulness, 2)
    if colourfulness < 4.0:
        issues.append(
            ValidationIssue(
                code="not_colour",
                message="Photo appears black-and-white / lacking colour.",
                how_to_fix="Upload a colour photograph.",
            )
        )

    # Background removal is required for Indian passport digital pipeline
    checks["require_bg_removal"] = require_bg_removal

    # As-is photos should already have a mostly white background in the corners
    rgb_bg = np.array(im.convert("RGB"))
    s = max(8, min(w, h) // 12)
    corners = [
        rgb_bg[:s, :s],
        rgb_bg[:s, -s:],
        rgb_bg[-s:, :s],
        rgb_bg[-s:, -s:],
    ]
    corner_means = [float(c.mean()) for c in corners]
    checks["as_is_corner_brightness"] = [round(v, 1) for v in corner_means]
    if any(m < 220 for m in corner_means):
        issues.append(
            ValidationIssue(
                code="background_not_passport_ready",
                message="Background is not plain white enough for passport use as-is.",
                how_to_fix="Use Convert — the app can replace the background if the face is clear — or retake on a white backdrop.",
            )
        )

    passed = len(issues) == 0
    return ValidationReport(passed=passed, stage="source_as_is", issues=issues, checks=checks)


def validate_source_convertible(
    im: Image.Image,
    spec: PhotoSpec,
) -> ValidationReport:
    """Lighter check: can our converter likely produce a valid passport photo?

    Allows imperfect backgrounds, framing, and mild size issues that crop +
    white-bg replacement can fix. Still blocks unusable inputs (no face, multi
    person, extreme blur, closed eyes, etc.).
    """
    issues: List[ValidationIssue] = []
    checks: Dict[str, Any] = {"mode": "convertible"}

    w, h = im.size
    checks["image_width"] = w
    checks["image_height"] = h
    min_side = min(w, h)
    checks["min_side_px"] = min_side

    # Slightly looser resolution (crop still needs detail)
    if min_side < 300:
        issues.append(
            ValidationIssue(
                code="resolution_too_low",
                message="Image resolution is too low to produce a sharp passport print.",
                how_to_fix="Use a higher-resolution phone photo (full quality, not a tiny thumbnail).",
            )
        )

    bgr, gray, _, _ = _to_bgr_gray(im)
    faces = _detect_faces(gray)
    checks["face_count"] = len(faces)

    if len(faces) == 0:
        issues.append(
            ValidationIssue(
                code="no_face",
                message="No face detected — cannot convert to a passport photo.",
                how_to_fix="Use a clear front-facing photo of one person looking at the camera.",
            )
        )
        return ValidationReport(
            passed=False, stage="source_convertible", issues=issues, checks=checks
        )

    primary = faces[0]
    p_area = primary[2] * primary[3]
    significant = [f for f in faces[1:] if (f[2] * f[3]) > 0.45 * p_area]
    checks["secondary_significant_faces"] = len(significant)
    if significant:
        issues.append(
            ValidationIssue(
                code="multiple_faces",
                message="Multiple people detected — converter needs a single-person photo.",
                how_to_fix="Photograph only one person (not a contact sheet or group photo).",
            )
        )

    fx, fy, fw, fh = primary
    face_area_ratio = (fw * fh) / float(w * h)
    checks["face_area_ratio"] = round(face_area_ratio, 4)
    # Much looser than as-is: room portraits are often small in frame
    if face_area_ratio < 0.012:
        issues.append(
            ValidationIssue(
                code="face_too_small",
                message="Face is too small / far away to convert reliably.",
                how_to_fix="Move closer so the head and shoulders are clearly visible.",
            )
        )

    face_roi = gray[fy : fy + fh, fx : fx + fw]
    blur = _laplacian_var(face_roi)
    checks["sharpness_laplacian_var"] = round(blur, 1)
    if blur < 10:
        issues.append(
            ValidationIssue(
                code="too_blurry",
                message="Photo is too blurry to produce an acceptable passport print.",
                how_to_fix="Retake in focus with the camera steady and good light.",
            )
        )

    bright = _face_brightness_stats(gray, primary)
    checks["face_brightness_mean"] = round(bright["mean"], 1)
    if bright["mean"] < 25:
        issues.append(
            ValidationIssue(
                code="too_dark",
                message="Face is far too dark to convert well.",
                how_to_fix="Retake with even front lighting.",
            )
        )
    if bright["mean"] > 230:
        issues.append(
            ValidationIssue(
                code="too_bright",
                message="Face is overexposed / washed out.",
                how_to_fix="Retake with softer light (no harsh flash).",
            )
        )

    skin = _skin_ratio(bgr, primary)
    checks["skin_ratio"] = round(skin, 3)
    if skin < 0.02:
        issues.append(
            ValidationIssue(
                code="face_not_personlike",
                message="Detected region does not look like a real person.",
                how_to_fix="Upload a real colour photo of a person.",
            )
        )

    eyes = _detect_eyes(gray, primary)
    checks["eye_count"] = len(eyes)
    if len(eyes) < 2:
        issues.append(
            ValidationIssue(
                code="eyes_not_detected",
                message="Both open eyes are not clearly visible — cannot make a valid passport photo.",
                how_to_fix="Look at the camera with both eyes open; remove dark glasses; keep hair off the eyes.",
            )
        )
    else:
        e_sorted = sorted(eyes[:2], key=lambda e: e[0])
        left, right = e_sorted[0], e_sorted[1]
        ly = left[1] + left[3] / 2
        ry = right[1] + right[3] / 2
        eye_tilt = abs(ly - ry) / max(fh, 1)
        checks["eye_tilt_ratio"] = round(eye_tilt, 3)
        if eye_tilt > 0.18:
            issues.append(
                ValidationIssue(
                    code="head_tilted",
                    message="Head is tilted too much for a passport photo.",
                    how_to_fix="Keep your head straight and face the camera.",
                )
            )
        dark_eyes = 0
        for ex, ey, ew, eh in eyes[:2]:
            eroi = gray[ey : ey + eh, ex : ex + ew]
            if eroi.size and float(eroi.mean()) < 35:
                dark_eyes += 1
        if dark_eyes >= 2:
            issues.append(
                ValidationIssue(
                    code="possible_dark_glasses",
                    message="Dark / tinted glasses detected.",
                    how_to_fix="Remove tinted glasses before converting.",
                )
            )

    white_cloth = _white_clothing_risk(bgr, primary, h)
    checks["white_clothing_ratio"] = round(white_cloth, 3)
    if white_cloth > 0.75:
        issues.append(
            ValidationIssue(
                code="white_clothing",
                message="Clothing looks pure white (often rejected on a white background).",
                how_to_fix="Wear a medium-coloured top (not pure white).",
            )
        )

    rgb = np.array(im.convert("RGB")).astype(np.float32)
    colourfulness = float(
        np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))
        + np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))
    )
    checks["colourfulness"] = round(colourfulness, 2)
    if colourfulness < 4.0:
        issues.append(
            ValidationIssue(
                code="not_colour",
                message="Photo appears black-and-white.",
                how_to_fix="Upload a colour photograph.",
            )
        )

    # Note: messy backgrounds are OK for convertible — converter replaces them
    checks["background_can_be_replaced"] = True

    passed = len(issues) == 0
    return ValidationReport(
        passed=passed, stage="source_convertible", issues=issues, checks=checks
    )


def assess_photo(im: Image.Image, spec: PhotoSpec) -> Dict[str, Any]:
    """Run both as-is and convertible checks; recommend next step."""
    as_is = validate_source_as_is(im, spec)
    convertible = validate_source_convertible(im, spec)

    if as_is.passed:
        recommendation = "already_ok"
        summary = (
            "This photo already passes automated passport checks. "
            "You can still Convert to get standard print/upload files."
        )
    elif convertible.passed:
        recommendation = "convertible"
        summary = (
            "This photo is not passport-ready as-is, but it looks convertible. "
            "Use Convert — the app will fix background/framing and re-validate."
        )
    else:
        recommendation = "retake"
        summary = (
            "This photo cannot be fixed automatically. "
            "Retake using the suggested fixes below."
        )

    return {
        "recommendation": recommendation,
        "summary": summary,
        "as_is": as_is.to_dict(),
        "convertible": convertible.to_dict(),
        "can_check_only_pass": as_is.passed,
        "can_convert": convertible.passed,
    }


def validate_output_photo(
    framed: Image.Image,
    spec: PhotoSpec,
) -> ValidationReport:
    """Validate the final 2×2 framed result before offering downloads."""
    issues: List[ValidationIssue] = []
    checks: Dict[str, Any] = {}

    w, h = framed.size
    checks["output_size"] = [w, h]
    if w != h:
        issues.append(
            ValidationIssue(
                code="not_square",
                message="Output photo is not square.",
                how_to_fix="Internal error — please retry. If it persists, report a bug.",
            )
        )

    bgr, gray, _, _ = _to_bgr_gray(framed)
    faces = _detect_faces(gray)
    checks["output_face_count"] = len(faces)

    if len(faces) == 0:
        issues.append(
            ValidationIssue(
                code="output_no_face",
                message="No face found in the processed passport photo.",
                how_to_fix="Retake with a clearer front-facing portrait and try again.",
            )
        )
        return ValidationReport(passed=False, stage="output", issues=issues, checks=checks)

    if len(faces) > 1:
        primary = faces[0]
        p_area = primary[2] * primary[3]
        if any((f[2] * f[3]) > 0.3 * p_area for f in faces[1:]):
            issues.append(
                ValidationIssue(
                    code="output_multiple_faces",
                    message="More than one face visible in the final photo.",
                    how_to_fix="Use a photo with only one person.",
                )
            )

    face = faces[0]
    fx, fy, fw, fh = face
    # Head height estimate: extend face box upward for hair (~0.4*h) 
    top = max(0, fy - int(0.4 * fh))
    chin = min(h - 1, fy + int(1.05 * fh))
    head_frac = (chin - top) / float(h)
    checks["output_head_height_frac"] = round(head_frac, 3)
    checks["output_head_height_in"] = round(head_frac * spec.photo_inches[1], 3)

    # Allow slight tolerance beyond target band after detection noise
    min_h = spec.head_height_min * 0.90
    max_h = spec.head_height_max * 1.12
    if head_frac < min_h or head_frac > max_h:
        issues.append(
            ValidationIssue(
                code="output_head_size",
                message=(
                    f"Final head size looks off for passport rules "
                    f"(~{head_frac * spec.photo_inches[1]:.2f}\" vs allowed "
                    f"{spec.head_height_min * spec.photo_inches[1]:.2f}–"
                    f"{spec.head_height_max * spec.photo_inches[1]:.2f}\")."
                ),
                how_to_fix="Retake with head fully visible (hair to chin) and shoulders; avoid extreme close-ups.",
            )
        )

    # Eyes still visible in output
    eyes = _detect_eyes(gray, face)
    checks["output_eye_count"] = len(eyes)
    if len(eyes) < 2:
        issues.append(
            ValidationIssue(
                code="output_eyes",
                message="Both eyes are not clearly visible in the final photo.",
                how_to_fix="Retake looking at the camera with eyes open and hair off the face.",
            )
        )

    # Background must be predominantly pure white outside the subject
    rgb = np.array(framed.convert("RGB"))
    # Corner patches
    s = max(8, w // 12)
    corners = [
        rgb[:s, :s],
        rgb[:s, -s:],
        rgb[-s:, :s],
        rgb[-s:, -s:],
    ]
    corner_means = [float(c.mean()) for c in corners]
    corner_chroma = [float(c.max(axis=2).mean() - c.min(axis=2).mean()) for c in corners]
    checks["corner_brightness"] = [round(v, 1) for v in corner_means]
    checks["corner_chroma"] = [round(v, 1) for v in corner_chroma]

    # Corners must be near-white (subject shouldn't sit in a corner)
    if any(m < 235 for m in corner_means) or any(c > 20 for c in corner_chroma):
        issues.append(
            ValidationIssue(
                code="background_not_white",
                message="Background is not clean plain white (shadows or leftover scene).",
                how_to_fix="Retake against a plain light wall with even lighting, or ensure background removal can isolate you cleanly.",
            )
        )

    # Top border band (less likely to include shoulders than bottom)
    band = max(6, w // 20)
    top_border = rgb[:band].reshape(-1, 3)
    pure_white_top = float((top_border.min(axis=1) >= 245).mean())
    checks["top_border_pure_white_frac"] = round(pure_white_top, 3)
    if pure_white_top < 0.85:
        issues.append(
            ValidationIssue(
                code="background_dirty",
                message="Top of photo is not a clean white background.",
                how_to_fix="Retake with clear space above the head and even lighting on the background.",
            )
        )

    # Sharpness on output face
    ox, oy, ow, oh = face
    out_blur = _laplacian_var(gray[oy : oy + oh, ox : ox + ow])
    checks["output_sharpness"] = round(out_blur, 1)
    if out_blur < 12:
        issues.append(
            ValidationIssue(
                code="output_blurry",
                message="Final photo is not sharp enough.",
                how_to_fix="Start from a sharper original photo.",
            )
        )

    passed = len(issues) == 0
    return ValidationReport(passed=passed, stage="output", issues=issues, checks=checks)


def merge_reports(*reports: ValidationReport) -> ValidationReport:
    issues: List[ValidationIssue] = []
    checks: Dict[str, Any] = {}
    for r in reports:
        issues.extend(r.issues)
        checks[r.stage] = r.checks
    return ValidationReport(
        passed=all(r.passed for r in reports) and len(issues) == 0,
        stage="full",
        issues=issues,
        checks=checks,
    )
