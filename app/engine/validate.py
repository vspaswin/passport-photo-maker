"""Passport photo QC: one FaceAnalysis, two source policies, one output check.

Automated checks reduce reject risk; they are not a government guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .face import FaceAnalysis, analyze_image, detect_eyes, detect_faces, laplacian_var, to_bgr_gray
from .specs import PhotoSpec


@dataclass
class ValidationIssue:
    code: str
    message: str
    how_to_fix: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ValidationReport:
    passed: bool
    stage: str
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


# ---------------------------------------------------------------------------
# Source policies (thresholds only — rules are shared)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePolicy:
    stage: str
    min_side: int
    min_face_area: float
    max_face_area: Optional[float]
    check_centering: bool
    max_center_offset: float
    check_clip: bool
    min_laplacian: float
    min_brightness: float
    max_brightness: float
    max_dark_frac: Optional[float]
    max_bright_frac: Optional[float]
    min_skin: float
    max_eye_tilt: Optional[float]
    min_eye_sep: Optional[float]
    max_white_clothing: float
    min_colourfulness: float
    check_corners_white: bool
    min_corner_brightness: float
    check_lower_face: bool
    min_lower_face_std: float


# Previous as-is / convertible thresholds preserved.
AS_IS = SourcePolicy(
    stage="source_as_is",
    min_side=400,
    min_face_area=0.03,
    max_face_area=0.65,
    check_centering=True,
    max_center_offset=0.28,
    check_clip=True,
    min_laplacian=12.0,
    min_brightness=32.0,
    max_brightness=220.0,
    max_dark_frac=0.40,
    max_bright_frac=0.35,
    min_skin=0.03,
    max_eye_tilt=0.12,
    min_eye_sep=0.22,
    max_white_clothing=0.70,
    min_colourfulness=4.0,
    check_corners_white=True,
    min_corner_brightness=220.0,
    check_lower_face=True,
    min_lower_face_std=8.0,
)

CONVERTIBLE = SourcePolicy(
    stage="source_convertible",
    min_side=300,
    min_face_area=0.012,
    max_face_area=None,
    check_centering=False,
    max_center_offset=1.0,
    check_clip=False,
    min_laplacian=10.0,
    min_brightness=25.0,
    max_brightness=230.0,
    max_dark_frac=None,
    max_bright_frac=None,
    min_skin=0.02,
    max_eye_tilt=0.18,
    min_eye_sep=None,
    max_white_clothing=0.75,
    min_colourfulness=4.0,
    check_corners_white=False,
    min_corner_brightness=0.0,
    check_lower_face=False,
    min_lower_face_std=0.0,
)


def apply_source_policy(analysis: FaceAnalysis, policy: SourcePolicy) -> ValidationReport:
    """Map one FaceAnalysis through a threshold policy → issues."""
    issues: List[ValidationIssue] = []
    checks = analysis.to_checks()
    checks["mode"] = policy.stage
    if not policy.check_corners_white:
        checks["background_can_be_replaced"] = True

    def add(code: str, message: str, fix: str) -> None:
        issues.append(ValidationIssue(code=code, message=message, how_to_fix=fix))

    if analysis.min_side < policy.min_side:
        add(
            "resolution_too_low",
            "Image resolution is too low for a sharp passport print."
            if policy.stage == "source_as_is"
            else "Image resolution is too low to produce a sharp passport print.",
            "Use a higher-resolution phone photo (full quality, not a tiny thumbnail).",
        )

    if analysis.face_count == 0:
        add(
            "no_face",
            "No face detected."
            if policy.stage == "source_as_is"
            else "No face detected — cannot convert to a passport photo.",
            "Use a clear front-facing photo of one person looking at the camera.",
        )
        return ValidationReport(
            passed=False, stage=policy.stage, issues=issues, checks=checks
        )

    if analysis.secondary_significant > 0:
        n = 1 + analysis.secondary_significant
        add(
            "multiple_faces",
            f"Multiple faces detected ({n} significant)."
            if policy.stage == "source_as_is"
            else "Multiple people detected — converter needs a single-person photo.",
            "Photograph only one person (not a contact sheet or group photo).",
        )

    if analysis.face_area_ratio < policy.min_face_area:
        add(
            "face_too_small",
            "Face is too small in the photo."
            if policy.stage == "source_as_is"
            else "Face is too small / far away to convert reliably.",
            "Move closer so the head and shoulders are clearly visible.",
        )

    if policy.max_face_area is not None and analysis.face_area_ratio > policy.max_face_area:
        add(
            "face_too_close",
            "Face fills almost the entire frame; top of head or chin may be cropped.",
            "Step back slightly so full head (hair to chin) and some shoulders are visible.",
        )

    if policy.check_clip and (analysis.side_clipped or analysis.bottom_clipped):
        add(
            "face_clipped",
            "Face appears cut off at the edge of the photo.",
            "Center yourself with space around the head; do not crop the chin, ears, or top of head.",
        )

    if policy.check_centering and analysis.face_horizontal_offset > policy.max_center_offset:
        add(
            "face_not_centered",
            "Face is not reasonably centered in the frame.",
            "Stand centered in front of the camera and look straight ahead.",
        )

    if analysis.sharpness_laplacian_var < policy.min_laplacian:
        add(
            "too_blurry",
            "Photo is too blurry / out of focus."
            if policy.stage == "source_as_is"
            else "Photo is too blurry to produce an acceptable passport print.",
            "Hold the camera steady, use good light, tap to focus on the face, and retake.",
        )

    if analysis.face_brightness_mean < policy.min_brightness:
        add(
            "too_dark",
            "Face is too dark (underexposed)."
            if policy.stage == "source_as_is"
            else "Face is far too dark to convert well.",
            "Use even front lighting (face a window or soft lamp). Avoid strong backlight.",
        )
    if analysis.face_brightness_mean > policy.max_brightness:
        add(
            "too_bright",
            "Face is overexposed / washed out.",
            "Reduce harsh light or flash; use softer, even lighting.",
        )
    if policy.max_dark_frac is not None and analysis.face_dark_fraction > policy.max_dark_frac:
        add(
            "uneven_shadows",
            "Large dark regions on the face (shadows).",
            "Use even lighting on both sides of the face; avoid side-only light.",
        )
    if policy.max_bright_frac is not None and analysis.face_bright_fraction > policy.max_bright_frac:
        add(
            "hotspots",
            "Overexposed hotspots on the face.",
            "Avoid direct flash glare; use diffused light.",
        )

    if analysis.skin_ratio < policy.min_skin:
        add(
            "face_not_personlike",
            "Detected region does not look like a natural face / skin tones."
            if policy.stage == "source_as_is"
            else "Detected region does not look like a real person.",
            "Upload a real colour photo of a person.",
        )

    if analysis.eye_count < 2:
        add(
            "eyes_not_detected",
            "Could not clearly detect both open eyes."
            if policy.stage == "source_as_is"
            else "Both open eyes are not clearly visible — cannot make a valid passport photo.",
            "Look straight at the camera with both eyes open. Remove sunglasses; keep hair off the eyes.",
        )
    else:
        if (
            policy.max_eye_tilt is not None
            and analysis.eye_tilt_ratio is not None
            and analysis.eye_tilt_ratio > policy.max_eye_tilt
        ):
            add(
                "head_tilted",
                "Head appears tilted (eyes not level)."
                if policy.stage == "source_as_is"
                else "Head is tilted too much for a passport photo.",
                "Keep your head straight — both eyes on a horizontal line, face the camera directly.",
            )
        if (
            policy.min_eye_sep is not None
            and analysis.eye_separation_ratio is not None
            and analysis.eye_separation_ratio < policy.min_eye_sep
        ):
            add(
                "not_frontal",
                "Face may not be fully frontal (eyes too close together in the frame).",
                "Face the camera directly (not a side or three-quarter pose).",
            )
        if analysis.dark_eye_boxes >= 2:
            add(
                "possible_dark_glasses",
                "Eyes look covered by dark / tinted glasses."
                if policy.stage == "source_as_is"
                else "Dark / tinted glasses detected.",
                "Remove tinted or dark glasses.",
            )

    if policy.check_lower_face and analysis.lower_face_std < policy.min_lower_face_std:
        add(
            "lower_face_unclear",
            "Lower face / chin area is unclear.",
            "Ensure chin and mouth are visible, uncovered, and in focus.",
        )

    if analysis.white_clothing_ratio > policy.max_white_clothing:
        add(
            "white_clothing",
            "Clothing appears mostly white (often rejected against a white background)."
            if policy.stage == "source_as_is"
            else "Clothing looks pure white (often rejected on a white background).",
            "Wear a medium-coloured top (e.g. blue, not pure white).",
        )

    if analysis.colourfulness < policy.min_colourfulness:
        add(
            "not_colour",
            "Photo appears black-and-white / lacking colour."
            if policy.stage == "source_as_is"
            else "Photo appears black-and-white.",
            "Upload a colour photograph.",
        )

    if policy.check_corners_white and analysis.corner_brightness:
        if any(m < policy.min_corner_brightness for m in analysis.corner_brightness):
            add(
                "background_not_passport_ready",
                "Background is not plain white enough for passport use as-is.",
                "Use Convert — the app can replace the background if the face is clear — or retake on a white backdrop.",
            )

    return ValidationReport(
        passed=len(issues) == 0,
        stage=policy.stage,
        issues=issues,
        checks=checks,
    )


def validate_source_as_is(
    im: Image.Image,
    spec: PhotoSpec,
    require_bg_removal: bool = True,
    analysis: Optional[FaceAnalysis] = None,
) -> ValidationReport:
    """Strict as-is check (already passport-ready?)."""
    a = analysis or analyze_image(im)
    report = apply_source_policy(a, AS_IS)
    report.checks["require_bg_removal"] = require_bg_removal
    return report


def validate_source_convertible(
    im: Image.Image,
    spec: PhotoSpec,
    analysis: Optional[FaceAnalysis] = None,
) -> ValidationReport:
    """Lighter check: can Convert likely produce a valid passport photo?"""
    a = analysis or analyze_image(im)
    return apply_source_policy(a, CONVERTIBLE)


def validate_source_photo(
    im: Image.Image,
    spec: PhotoSpec,
    require_bg_removal: bool = True,
) -> ValidationReport:
    """Alias for as-is validation (backwards compatible)."""
    return validate_source_as_is(im, spec, require_bg_removal=require_bg_removal)


def assess_photo(im: Image.Image, spec: PhotoSpec) -> Dict[str, Any]:
    """One analysis, both policies — recommend next step."""
    analysis = analyze_image(im)
    as_is = validate_source_as_is(im, spec, analysis=analysis)
    convertible = validate_source_convertible(im, spec, analysis=analysis)

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


def validate_output_photo(framed: Image.Image, spec: PhotoSpec) -> ValidationReport:
    """Validate final 2×2 framed result before offering downloads."""
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

    bgr, gray = to_bgr_gray(framed)
    faces = detect_faces(gray, bgr=bgr)
    checks["output_face_count"] = len(faces)

    if not faces:
        issues.append(
            ValidationIssue(
                code="output_no_face",
                message="No face found in the processed passport photo.",
                how_to_fix="Retake with a clearer front-facing portrait and try again.",
            )
        )
        return ValidationReport(passed=False, stage="output", issues=issues, checks=checks)

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

    fx, fy, fw, fh = primary
    top = max(0, fy - int(0.4 * fh))
    chin = min(h - 1, fy + int(1.05 * fh))
    head_frac = (chin - top) / float(h)
    checks["output_head_height_frac"] = round(head_frac, 3)
    checks["output_head_height_in"] = round(head_frac * spec.photo_inches[1], 3)

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

    eyes = detect_eyes(gray, primary)
    checks["output_eye_count"] = len(eyes)
    if len(eyes) < 2:
        issues.append(
            ValidationIssue(
                code="output_eyes",
                message="Both eyes are not clearly visible in the final photo.",
                how_to_fix="Retake looking at the camera with eyes open and hair off the face.",
            )
        )

    rgb = np.array(framed.convert("RGB"))
    s = max(8, w // 12)
    corners = [rgb[:s, :s], rgb[:s, -s:], rgb[-s:, :s], rgb[-s:, -s:]]
    corner_means = [float(c.mean()) for c in corners]
    corner_chroma = [
        float(c.max(axis=2).mean() - c.min(axis=2).mean()) for c in corners
    ]
    checks["corner_brightness"] = [round(v, 1) for v in corner_means]
    checks["corner_chroma"] = [round(v, 1) for v in corner_chroma]

    if any(m < 235 for m in corner_means) or any(c > 20 for c in corner_chroma):
        issues.append(
            ValidationIssue(
                code="background_not_white",
                message="Background is not clean plain white (shadows or leftover scene).",
                how_to_fix="Retake against a plain light wall with even lighting, or ensure background removal can isolate you cleanly.",
            )
        )

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

    out_blur = laplacian_var(gray[fy : fy + fh, fx : fx + fw])
    checks["output_sharpness"] = round(out_blur, 1)
    if out_blur < 12:
        issues.append(
            ValidationIssue(
                code="output_blurry",
                message="Final photo is not sharp enough.",
                how_to_fix="Start from a sharper original photo.",
            )
        )

    return ValidationReport(
        passed=len(issues) == 0, stage="output", issues=issues, checks=checks
    )


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
