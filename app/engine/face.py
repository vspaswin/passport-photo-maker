"""Shared face detection and metric extraction (single Haar stack)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class FaceBox:
    """Primary face with estimated passport landmarks (for framing)."""

    x: int
    y: int
    w: int
    h: int
    top_of_head: int
    chin: int
    eye_y: int
    center_x: int


@dataclass
class FaceAnalysis:
    """One-pass measurements used by as-is / convertible / framing."""

    image_w: int
    image_h: int
    min_side: int
    face_count: int
    faces: List[BBox] = field(default_factory=list)
    primary: Optional[BBox] = None
    secondary_significant: int = 0
    face_area_ratio: float = 0.0
    face_horizontal_offset: float = 0.0
    side_clipped: bool = False
    bottom_clipped: bool = False
    sharpness_laplacian_var: float = 0.0
    face_brightness_mean: float = 0.0
    face_brightness_std: float = 0.0
    face_dark_fraction: float = 0.0
    face_bright_fraction: float = 0.0
    skin_ratio: float = 0.0
    eye_count: int = 0
    eye_tilt_ratio: Optional[float] = None
    eye_separation_ratio: Optional[float] = None
    dark_eye_boxes: int = 0
    lower_face_std: float = 0.0
    white_clothing_ratio: float = 0.0
    colourfulness: float = 0.0
    corner_brightness: List[float] = field(default_factory=list)

    def to_checks(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "image_width": self.image_w,
            "image_height": self.image_h,
            "megapixels": round((self.image_w * self.image_h) / 1_000_000, 2),
            "min_side_px": self.min_side,
            "face_count": self.face_count,
            "secondary_significant_faces": self.secondary_significant,
            "face_area_ratio": round(self.face_area_ratio, 4),
            "face_horizontal_offset": round(self.face_horizontal_offset, 3),
            "sharpness_laplacian_var": round(self.sharpness_laplacian_var, 1),
            "face_brightness_mean": round(self.face_brightness_mean, 1),
            "face_brightness_std": round(self.face_brightness_std, 1),
            "face_dark_fraction": round(self.face_dark_fraction, 3),
            "face_bright_fraction": round(self.face_bright_fraction, 3),
            "skin_ratio": round(self.skin_ratio, 3),
            "eye_count": self.eye_count,
            "dark_eye_boxes": self.dark_eye_boxes,
            "lower_face_std": round(self.lower_face_std, 1),
            "white_clothing_ratio": round(self.white_clothing_ratio, 3),
            "colourfulness": round(self.colourfulness, 2),
            "corner_brightness": [round(v, 1) for v in self.corner_brightness],
        }
        if self.primary:
            x, y, w, h = self.primary
            d["face_box"] = {"x": x, "y": y, "w": w, "h": h}
        if self.eye_tilt_ratio is not None:
            d["eye_tilt_ratio"] = round(self.eye_tilt_ratio, 3)
        if self.eye_separation_ratio is not None:
            d["eye_separation_ratio"] = round(self.eye_separation_ratio, 3)
        return d

    def to_face_box(self, image_h: Optional[int] = None) -> Optional[FaceBox]:
        if not self.primary:
            return None
        x, y, w, h = self.primary
        ih = image_h if image_h is not None else self.image_h
        top_of_head = max(0, int(y - 0.45 * h))
        chin = min(ih - 1, int(y + h * 1.05))
        eye_y = int(y + 0.38 * h)
        return FaceBox(
            x=x,
            y=y,
            w=w,
            h=h,
            top_of_head=top_of_head,
            chin=chin,
            eye_y=eye_y,
            center_x=int(x + w / 2),
        )


_cascades = None


def _load_cascades():
    global _cascades
    if _cascades is None:
        base = Path(cv2.data.haarcascades)
        _cascades = (
            cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(base / "haarcascade_eye.xml")),
            cv2.CascadeClassifier(str(base / "haarcascade_eye_tree_eyeglasses.xml")),
        )
    return _cascades


def to_bgr_gray(im: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.array(im.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return bgr, gray


def _detect_faces_mediapipe(bgr: np.ndarray) -> List[BBox]:
    """Optional MediaPipe face detector (more robust than Haar)."""
    try:
        from app.core.config import get_settings

        if not get_settings().use_mediapipe:
            return []
    except Exception:
        pass
    try:
        import mediapipe as mp
    except ImportError:
        return []

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes: List[BBox] = []
    try:
        # mediapipe 0.10+ solutions still work on many installs
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as detector:
            result = detector.process(rgb)
            if not result.detections:
                return []
            for det in result.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = min(w - x, int(bb.width * w))
                bh = min(h - y, int(bb.height * h))
                if bw > 20 and bh > 20:
                    boxes.append((x, y, bw, bh))
    except Exception:
        return []
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def _dedupe_faces(boxes: List[BBox]) -> List[BBox]:
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    cleaned: List[BBox] = []
    for box in boxes:
        x, y, w, h = box
        duplicate = False
        for cx, cy, cw, ch in cleaned:
            if abs((x + w / 2) - (cx + cw / 2)) < 0.35 * max(w, cw) and abs(
                (y + h / 2) - (cy + ch / 2)
            ) < 0.35 * max(h, ch):
                duplicate = True
                break
        if not duplicate:
            cleaned.append(box)
    return cleaned


def detect_faces(gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> List[BBox]:
    # Prefer MediaPipe when available and BGR provided
    if bgr is not None:
        mp_boxes = _detect_faces_mediapipe(bgr)
        if mp_boxes:
            return _dedupe_faces(mp_boxes)

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
    boxes = [tuple(map(int, f)) for f in faces]
    return _dedupe_faces(boxes)


def detect_eyes(gray: np.ndarray, face: BBox) -> List[BBox]:
    _, eye_cascade, eye_tree = _load_cascades()
    x, y, w, h = face
    roi_y = y + int(0.12 * h)
    roi_h = int(0.55 * h)
    roi = gray[roi_y : y + roi_h, x : x + w]
    if roi.size == 0:
        return []
    eyes = eye_cascade.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18)
    )
    if len(eyes) < 2:
        eyes = eye_tree.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=4, minSize=(16, 16)
        )
    out = [
        (x + int(ex), roi_y + int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eyes
    ]
    out.sort(key=lambda b: b[2] * b[3], reverse=True)
    return out[:4]


def laplacian_var(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


def _skin_ratio(bgr: np.ndarray, face: BBox) -> float:
    x, y, w, h = face
    x0, x1 = x + int(0.15 * w), x + int(0.85 * w)
    y0, y1 = y + int(0.20 * h), y + int(0.85 * h)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb, np.array([0, 125, 70], dtype=np.uint8), np.array([255, 180, 140], dtype=np.uint8)
    )
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([30, 200, 255]))
    hsv_mask2 = cv2.inRange(hsv, np.array([150, 15, 40]), np.array([180, 200, 255]))
    combined = cv2.bitwise_or(mask, cv2.bitwise_or(hsv_mask, hsv_mask2))
    return float(combined.mean() / 255.0)


def _white_clothing_ratio(bgr: np.ndarray, face: BBox, img_h: int) -> float:
    x, y, w, h = face
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
    mn, mx = rgb.min(axis=2), rgb.max(axis=2)
    lum, chroma = rgb.mean(axis=2), mx - mn
    return float(((lum >= 200) & (chroma <= 25)).mean())


def analyze_image(im: Image.Image) -> FaceAnalysis:
    """Run face/eye detection and quality metrics once."""
    w, h = im.size
    bgr, gray = to_bgr_gray(im)
    faces = detect_faces(gray, bgr=bgr)

    analysis = FaceAnalysis(
        image_w=w,
        image_h=h,
        min_side=min(w, h),
        face_count=len(faces),
        faces=faces,
    )

    # Corner brightness (as-is white-bg hint)
    rgb = np.array(im.convert("RGB"))
    s = max(8, min(w, h) // 12)
    corners = [rgb[:s, :s], rgb[:s, -s:], rgb[-s:, :s], rgb[-s:, -s:]]
    analysis.corner_brightness = [float(c.mean()) for c in corners]

    colourfulness = float(
        np.mean(np.abs(rgb[:, :, 0].astype(np.float32) - rgb[:, :, 1]))
        + np.mean(np.abs(rgb[:, :, 1].astype(np.float32) - rgb[:, :, 2]))
    )
    analysis.colourfulness = colourfulness

    if not faces:
        return analysis

    primary = faces[0]
    analysis.primary = primary
    p_area = primary[2] * primary[3]
    analysis.secondary_significant = sum(
        1 for f in faces[1:] if (f[2] * f[3]) > 0.45 * p_area
    )
    fx, fy, fw, fh = primary
    analysis.face_area_ratio = (fw * fh) / float(w * h)
    analysis.face_horizontal_offset = abs((fx + fw / 2) - w / 2) / w
    analysis.side_clipped = fx <= 2 or (fx + fw) >= w - 2
    analysis.bottom_clipped = (fy + fh) >= h - 2

    face_roi = gray[fy : fy + fh, fx : fx + fw]
    analysis.sharpness_laplacian_var = laplacian_var(face_roi)
    if face_roi.size:
        analysis.face_brightness_mean = float(face_roi.mean())
        analysis.face_brightness_std = float(face_roi.std())
        analysis.face_dark_fraction = float((face_roi < 40).mean())
        analysis.face_bright_fraction = float((face_roi > 235).mean())
        lower = face_roi[int(fh * 0.55) :, :]
        analysis.lower_face_std = float(lower.std()) if lower.size else 0.0

    analysis.skin_ratio = _skin_ratio(bgr, primary)
    analysis.white_clothing_ratio = _white_clothing_ratio(bgr, primary, h)

    eyes = detect_eyes(gray, primary)
    analysis.eye_count = len(eyes)
    if len(eyes) >= 2:
        e_sorted = sorted(eyes[:2], key=lambda e: e[0])
        left, right = e_sorted[0], e_sorted[1]
        ly = left[1] + left[3] / 2
        ry = right[1] + right[3] / 2
        analysis.eye_tilt_ratio = abs(ly - ry) / max(fh, 1)
        lx = left[0] + left[2] / 2
        rx = right[0] + right[2] / 2
        analysis.eye_separation_ratio = abs(rx - lx) / max(fw, 1)
        dark = 0
        for ex, ey, ew, eh in eyes[:2]:
            eroi = gray[ey : ey + eh, ex : ex + ew]
            if eroi.size and float(eroi.mean()) < 35:
                dark += 1
        analysis.dark_eye_boxes = dark

    return analysis


def detect_primary_face(im: Image.Image) -> Optional[FaceBox]:
    """Primary face with framing landmarks (used by process)."""
    return analyze_image(im).to_face_box()


def fallback_face(im: Image.Image) -> FaceBox:
    """Center-biased estimate when detector fails (unsafe / non-strict only)."""
    w, h = im.size
    box_h = int(h * 0.45)
    box_w = int(box_h * 0.75)
    cx = w // 2
    top = int(h * 0.08)
    chin = top + box_h
    return FaceBox(
        x=cx - box_w // 2,
        y=top + int(0.35 * box_h),
        w=box_w,
        h=int(box_h * 0.7),
        top_of_head=top,
        chin=chin,
        eye_y=top + int(box_h * 0.42),
        center_x=cx,
    )
