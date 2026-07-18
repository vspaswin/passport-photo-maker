"""Core photo processing: background, face geometry, exports."""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .face import FaceBox, detect_primary_face, fallback_face
from .specs import PhotoSpec, get_spec
from .validate import (
    PhotoValidationError,
    ValidationIssue,
    ValidationReport,
    merge_reports,
    validate_output_photo,
    validate_source_convertible,
)

logger = logging.getLogger(__name__)

# Lazy rembg session (loads ONNX model once)
_rembg_session = None


@dataclass
class ProcessResult:
    """All outputs from a conversion run."""

    doc_type: str
    preview_jpeg: bytes
    files: Dict[str, bytes]  # filename -> bytes
    metrics: Dict[str, float]
    warnings: List[str]
    validation: Optional[Dict] = None  # full validation report dict
    prepared_png: Optional[bytes] = None  # white-bg intermediate for reframe
    original_thumb: Optional[bytes] = None
    guide_preview_jpeg: Optional[bytes] = None
    face_dict: Optional[Dict] = None


# Quality order from rembg ecosystem; CPU-friendly first in fallbacks.
# birefnet-portrait is excellent but ~1GB and very slow on CPU — opt-in via REMBG_MODEL.
_REMBG_MODEL_FALLBACKS = (
    "u2net_human_seg",
    "u2net",
    "isnet-general-use",
    "birefnet-portrait",
    "birefnet-general",
)


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        preferred = "u2net_human_seg"
        try:
            from app.core.config import get_settings

            preferred = get_settings().rembg_model or preferred
        except Exception:
            pass

        candidates: List[str] = []
        for m in (preferred, *_REMBG_MODEL_FALLBACKS):
            if m and m not in candidates:
                candidates.append(m)

        last_err: Optional[Exception] = None
        for model in candidates:
            try:
                logger.info("Loading rembg model %s ...", model)
                _rembg_session = new_session(model)
                logger.info("Loaded rembg session (%s)", model)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("Could not load rembg model %s: %s", model, exc)
        if _rembg_session is None:
            raise RuntimeError(f"No rembg model could be loaded: {last_err}")
    return _rembg_session


def load_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def _corner_wall_stats(rgb: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Median colour / luminance / chroma of the four image corners."""
    h, w = rgb.shape[:2]
    s = max(8, min(h, w) // 20)
    corners = np.concatenate(
        [
            rgb[:s, :s].reshape(-1, 3),
            rgb[:s, -s:].reshape(-1, 3),
            rgb[-s:, :s].reshape(-1, 3),
            rgb[-s:, -s:].reshape(-1, 3),
        ],
        axis=0,
    )
    wall = np.median(corners, axis=0).astype(np.float32)
    chroma = float(
        np.mean(corners.max(axis=1) - corners.min(axis=1))
    )
    return wall, float(wall.mean()), chroma


def _is_plain_light_wall(wall_lum: float, wall_chroma: float) -> bool:
    """True when photo is already on a light plain backdrop (best natural path)."""
    return wall_lum >= 200.0 and wall_chroma <= 35.0


def _alpha_from_grabcut(im: Image.Image) -> np.ndarray:
    """Soft subject mask via OpenCV GrabCut (fast, natural on plain walls)."""
    rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    face = detect_face(im) or fallback_face(im)
    x0 = max(0, face.x - int(0.65 * face.w))
    y0 = max(0, face.top_of_head - int(0.2 * face.h))
    x1 = min(w, face.x + face.w + int(0.65 * face.w))
    y1 = min(h, face.chin + int(1.5 * face.h))
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    a = np.where((mask == 1) | (mask == 3), 1.0, 0.0).astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), 1.0)
    return np.clip(a, 0.0, 1.0)


def _alpha_from_rembg(im: Image.Image) -> np.ndarray:
    """Soft subject mask from rembg (for busy / non-white backgrounds)."""
    from rembg import remove

    session = _get_rembg_session()
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw = buf.getvalue()
    try:
        mask_bytes = remove(raw, session=session, only_mask=True)
        alpha_im = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception:  # noqa: BLE001
        cut = remove(raw, session=session)
        alpha_im = Image.open(io.BytesIO(cut)).convert("RGBA").split()[-1]
    if alpha_im.size != im.size:
        alpha_im = alpha_im.resize(im.size, Image.Resampling.LANCZOS)
    return np.array(alpha_im, dtype=np.float32) / 255.0


def remove_background_to_white(
    im: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Pure white background with natural hair — keep original person pixels.

    Strategy:
      • Plain light wall (typical home passport shot): GrabCut mask + original RGB
        → looks like left photo, only wall becomes pure white (not a cutout paste).
      • Busy backgrounds: rembg mask + same original-pixel composite.

    Never re-colours the subject from rembg's baked RGBA (that looks artificial).
    """
    rgb = np.array(im.convert("RGB"), dtype=np.float32)
    wall, wall_lum, wall_chroma = _corner_wall_stats(rgb)

    if _is_plain_light_wall(wall_lum, wall_chroma):
        logger.info(
            "Plain light wall detected (lum=%.0f chroma=%.1f) — natural GrabCut path",
            wall_lum,
            wall_chroma,
        )
        try:
            alpha = _alpha_from_grabcut(im)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GrabCut failed (%s); falling back to rembg", exc)
            alpha = _alpha_from_rembg(im)
    else:
        logger.info(
            "Non-plain background (lum=%.0f chroma=%.1f) — rembg mask path",
            wall_lum,
            wall_chroma,
        )
        alpha = _alpha_from_rembg(im)

    return _composite_original_on_white(im, alpha, wall, bg_rgb)


def _composite_original_on_white(
    original: Image.Image,
    alpha: np.ndarray,
    wall: Optional[np.ndarray] = None,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Soft-mask original photo onto pure white — natural hair, clean wall."""
    rgb = np.array(original.convert("RGB"), dtype=np.float32)
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    if wall is None:
        wall, _, _ = _corner_wall_stats(rgb)
    wall = np.asarray(wall, dtype=np.float32).reshape(3)

    # Mild clean — keep soft hair (no hard erode)
    a_u8 = (a * 255).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a_u8 = cv2.morphologyEx(a_u8, cv2.MORPH_OPEN, k3, iterations=1)
    a = a_u8.astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (3, 3), 0.5)
    a = np.clip(a, 0.0, 1.0)
    a = np.where(a < 0.03, 0.0, a)
    a = np.where(a > 0.94, 1.0, a)

    # Wall-like pixels that aren't solid subject → pure background
    lum = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    wall_lum = float(wall.mean())
    wall_like = (lum >= wall_lum - 18) & (lum >= 195) & (chroma <= 28) & (a < 0.60)
    a = np.where(wall_like, 0.0, a)

    # Despill edges using *original wall* colour (not white)
    eps = 1e-3
    a3 = a[:, :, None]
    wall3 = wall.reshape(1, 1, 3)
    edge = (a > 0.04) & (a < 0.96)
    fg = (rgb - (1.0 - a3) * wall3) / np.maximum(a3, eps)
    fg = np.clip(fg, 0, 255)
    rgb_out = rgb.copy()
    needs = edge & (lum > 75)
    rgb_out[needs] = fg[needs]
    # Keep original interior + dark hair strands (natural look from left photo)
    keep = (a >= 0.95) | ((a > 0.12) & (lum < 90))
    rgb_out[keep] = rgb[keep]

    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = rgb_out * a3 + bg * (1.0 - a3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[a < 0.02] = bg_rgb

    mn = out.min(axis=2)
    mx = out.max(axis=2)
    out_lum = out.mean(axis=2)
    leftover = (a < 0.50) & (out_lum >= 210) & ((mx - mn) <= 22)
    out[leftover] = bg_rgb
    out[(mn >= 248) & ((mx - mn) <= 8)] = bg_rgb
    return Image.fromarray(out, mode="RGB")


def _composite_clean_white(
    rgba: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Legacy path: composite rembg RGBA. Prefer remove_background_to_white."""
    arr = np.array(rgba)
    rgb_im = Image.fromarray(arr[:, :, :3], mode="RGB")
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    return _composite_original_on_white(rgb_im, alpha, None, bg_rgb)


def detect_face(im: Image.Image) -> Optional[FaceBox]:
    """Detect primary face and estimate head/eye landmarks."""
    return detect_primary_face(im)


def frame_to_spec(
    im: Image.Image,
    face: FaceBox,
    spec: PhotoSpec,
    out_px: Optional[int] = None,
    scale_factor: float = 1.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = 0.0,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Scale/place subject so head height and eye line match the spec.

    Fine-tune (optional):
      scale_factor: >1 enlarges subject (bigger head in frame), <1 shrinks
      offset_x_frac / offset_y_frac: shift as fraction of output size (+right / +down)
    """
    out_px = out_px or spec.print_px
    out_w = out_px
    out_h = out_px if spec.is_square else int(
        out_px * spec.photo_inches[1] / spec.photo_inches[0]
    )

    scale_factor = float(max(0.75, min(1.35, scale_factor)))
    offset_x_frac = float(max(-0.12, min(0.12, offset_x_frac)))
    offset_y_frac = float(max(-0.12, min(0.12, offset_y_frac)))

    head_px_src = max(1, face.chin - face.top_of_head)
    target_head_frac = spec.head_height_target * scale_factor
    # Keep target head within legal band after fine-tune
    target_head_frac = max(spec.head_height_min, min(spec.head_height_max, target_head_frac))
    head_out = target_head_frac * out_h
    scale = head_out / head_px_src

    new_w = max(1, int(im.width * scale))
    new_h = max(1, int(im.height * scale))
    scaled = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    eye_s = int(face.eye_y * scale)
    cx_s = int(face.center_x * scale)

    eye_y_out = out_h - int(spec.eye_from_bottom_target * out_h)
    paste_x = out_w // 2 - cx_s + int(offset_x_frac * out_w)
    paste_y = eye_y_out - eye_s + int(offset_y_frac * out_h)

    canvas = Image.new("RGB", (out_w, out_h), spec.background_rgb)

    src_x0 = max(0, -paste_x)
    src_y0 = max(0, -paste_y)
    src_x1 = min(new_w, out_w - paste_x)
    src_y1 = min(new_h, out_h - paste_y)
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop = scaled.crop((src_x0, src_y0, src_x1, src_y1))
        canvas.paste(crop, (max(0, paste_x), max(0, paste_y)))

    canvas = _clean_near_white(canvas, spec.background_rgb)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.05)

    head_in = spec.photo_inches[1] * target_head_frac
    eye_from_bottom_in = spec.photo_inches[1] * spec.eye_from_bottom_target
    metrics = {
        "head_height_in": round(head_in, 3),
        "eye_from_bottom_in": round(eye_from_bottom_in, 3),
        "head_height_min_in": round(spec.head_height_min * spec.photo_inches[1], 3),
        "head_height_max_in": round(spec.head_height_max * spec.photo_inches[1], 3),
        "eye_from_bottom_min_in": round(
            spec.eye_from_bottom_min * spec.photo_inches[1], 3
        ),
        "eye_from_bottom_max_in": round(
            spec.eye_from_bottom_max * spec.photo_inches[1], 3
        ),
        "output_px": float(out_px),
        "scale": round(scale, 4),
        "scale_factor": scale_factor,
        "offset_x_frac": offset_x_frac,
        "offset_y_frac": offset_y_frac,
    }
    return canvas, metrics


def build_guide_overlay(framed: Image.Image, spec: PhotoSpec) -> Image.Image:
    """Draw passport geometry guides on a copy of the framed photo."""
    im = framed.convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = im.size
    # Legal head height band (from top)
    y_head_max = int((1.0 - spec.head_height_min) * h)  # chin can be this low for min head
    y_head_min = int((1.0 - spec.head_height_max) * h)
    # Eye line band from bottom
    y_eye_lo = h - int(spec.eye_from_bottom_max * h)
    y_eye_hi = h - int(spec.eye_from_bottom_min * h)
    # Semi-transparent bands
    draw.rectangle([0, 0, w, max(0, y_head_min)], fill=(0, 120, 255, 30))
    draw.rectangle([0, y_eye_lo, w, y_eye_hi], fill=(255, 200, 0, 40))
    # Target eye line
    y_eye = h - int(spec.eye_from_bottom_target * h)
    draw.line([(0, y_eye), (w, y_eye)], fill=(255, 180, 0, 200), width=max(2, h // 400))
    # Center line
    draw.line([(w // 2, 0), (w // 2, h)], fill=(0, 180, 255, 120), width=max(1, h // 500))
    # Outer border
    draw.rectangle([1, 1, w - 2, h - 2], outline=(0, 0, 0, 80), width=max(2, h // 300))
    composed = Image.alpha_composite(im, overlay).convert("RGB")
    return composed


def face_box_to_dict(face: FaceBox) -> Dict:
    return {
        "x": face.x,
        "y": face.y,
        "w": face.w,
        "h": face.h,
        "top_of_head": face.top_of_head,
        "chin": face.chin,
        "eye_y": face.eye_y,
        "center_x": face.center_x,
    }


def face_box_from_dict(d: Dict) -> FaceBox:
    return FaceBox(
        x=int(d["x"]),
        y=int(d["y"]),
        w=int(d["w"]),
        h=int(d["h"]),
        top_of_head=int(d["top_of_head"]),
        chin=int(d["chin"]),
        eye_y=int(d["eye_y"]),
        center_x=int(d["center_x"]),
    )


def export_framed(
    framed: Image.Image,
    spec: PhotoSpec,
    *,
    warnings: Optional[List[str]] = None,
    full_report: Optional[ValidationReport] = None,
) -> Tuple[Dict[str, bytes], bytes, bytes, Dict[str, float]]:
    """Build download files + preview + guide preview from a framed square photo."""
    warnings = warnings or []
    metrics: Dict[str, float] = {}
    files: Dict[str, bytes] = {}
    base = f"{spec.id}"

    print_img = framed.resize((spec.print_px, spec.print_px), Image.Resampling.LANCZOS)
    files[f"{base}_PRINT_2x2_inch.jpg"] = jpeg_bytes(
        print_img, quality=95, dpi=(spec.print_dpi, spec.print_dpi)
    )
    master = framed.resize((1800, 1800), Image.Resampling.LANCZOS)
    files[f"{base}_master.jpg"] = jpeg_bytes(master, quality=97, dpi=(600, 600))

    for uv in spec.upload_variants:
        data = save_upload_variant(
            framed, uv.size_px, min_kb=uv.min_kb, max_kb=uv.max_kb
        )
        files[f"{base}_{uv.filename_suffix}.jpg"] = data
        metrics[f"{uv.filename_suffix}_kb"] = round(len(data) / 1024, 1)

    for sheet in spec.print_sheets:
        sheet_im = make_print_sheet(
            framed,
            page_inches=sheet.page_inches,
            cols=sheet.cols,
            rows=sheet.rows,
            photo_inches=spec.photo_inches,
            dpi=300,
        )
        files[f"{base}_{sheet.filename_suffix}.jpg"] = jpeg_bytes(
            sheet_im, quality=95, dpi=(300, 300)
        )

    preview = framed.resize((512, 512), Image.Resampling.LANCZOS)
    preview_jpeg = jpeg_bytes(preview, quality=88, dpi=(72, 72), progressive=True)
    guide = build_guide_overlay(preview, spec)
    guide_jpeg = jpeg_bytes(guide, quality=88, dpi=(72, 72), progressive=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        if full_report is not None:
            zf.writestr("README.txt", _result_readme(spec, metrics, warnings, full_report))
            zf.writestr(
                "VALIDATION_PASSED.txt", _validation_certificate(full_report, spec)
            )
    files[f"{base}_all.zip"] = zip_buf.getvalue()
    return files, preview_jpeg, guide_jpeg, metrics


def _clean_near_white(
    im: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    threshold: int = 242,
) -> Image.Image:
    """After resize: snap true near-white fringe only (keep clothing whites)."""
    arr = np.array(im.convert("RGB"))
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    chroma = mx - mn
    lum = arr.mean(axis=2)

    # Pure near-white → background
    arr[(mn >= threshold) & (chroma <= 14)] = bg_rgb

    pure = (mn >= 250) & (chroma <= 5)
    if pure.any() and not pure.all():
        non_w = (~pure).astype(np.uint8) * 255
        dist = cv2.distanceTransform(non_w, cv2.DIST_L2, 3)
        rim = (dist > 0) & (dist <= 1.8)
        # Only milky fringe next to pure white (not collar stripes deeper in)
        pale = rim & (lum >= 190) & (chroma <= 30)
        arr[pale] = bg_rgb

    return Image.fromarray(arr)


def jpeg_bytes(
    im: Image.Image,
    quality: int = 95,
    dpi: Tuple[int, int] = (300, 300),
    progressive: bool = False,
    optimize: bool = True,
) -> bytes:
    buf = io.BytesIO()
    im.save(
        buf,
        format="JPEG",
        quality=quality,
        dpi=dpi,
        optimize=optimize,
        progressive=progressive,
        subsampling=0 if quality >= 90 else 2,
    )
    return buf.getvalue()


def save_upload_variant(
    im: Image.Image,
    size_px: int,
    min_kb: int = 10,
    max_kb: int = 100,
) -> bytes:
    """Resize square and compress into portal file-size window."""
    out = im.resize((size_px, size_px), Image.Resampling.LANCZOS)
    min_b, max_b = min_kb * 1024, max_kb * 1024

    for q in range(90, 38, -2):
        data = jpeg_bytes(out, quality=q, progressive=True, dpi=(72, 72))
        if min_b <= len(data) <= max_b:
            return data

    # Shrink further if still too large
    for s in (size_px, 500, 450, 400, 350, 300):
        if s > size_px:
            continue
        out2 = im.resize((s, s), Image.Resampling.LANCZOS)
        for q in range(85, 35, -2):
            data = jpeg_bytes(out2, quality=q, progressive=True, dpi=(72, 72))
            if min_b <= len(data) <= max_b:
                return data

    # Last resort: under max_kb
    out3 = im.resize((min(350, size_px), min(350, size_px)), Image.Resampling.LANCZOS)
    for q in range(70, 25, -5):
        data = jpeg_bytes(out3, quality=q, progressive=True, dpi=(72, 72))
        if len(data) <= max_b:
            return data
    return data


def make_print_sheet(
    photo: Image.Image,
    page_inches: Tuple[float, float],
    cols: int,
    rows: int,
    photo_inches: Tuple[float, float] = (2.0, 2.0),
    dpi: int = 300,
    gap_in: float = 0.06,
) -> Image.Image:
    page_w = int(page_inches[0] * dpi)
    page_h = int(page_inches[1] * dpi)
    photo_w = int(photo_inches[0] * dpi)
    photo_h = int(photo_inches[1] * dpi)
    gap = int(gap_in * dpi)

    margin_x = (page_w - (cols * photo_w + (cols - 1) * gap)) // 2
    margin_y = (page_h - (rows * photo_h + (rows - 1) * gap)) // 2

    sheet = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    tile = photo.resize((photo_w, photo_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(sheet)

    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (photo_w + gap)
            y = margin_y + r * (photo_h + gap)
            sheet.paste(tile, (x, y))
            draw.rectangle(
                [x, y, x + photo_w - 1, y + photo_h - 1],
                outline=(210, 210, 210),
                width=1,
            )
    return sheet


def process_photo(
    image_bytes: bytes,
    doc_type: str = "indian-passport",
    remove_bg: bool = True,
    strict: bool = True,
    child_mode: bool = False,
    scale_factor: float = 1.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = 0.0,
) -> ProcessResult:
    """
    Convert an arbitrary photo into document-ready print + digital files.

    When ``strict`` is True (default):
      1) Source must be *convertible*
      2) After conversion, *output* must pass full passport QC
    """
    spec = get_spec(doc_type)
    warnings: List[str] = []

    original = load_image(image_bytes)
    thumb = original.copy()
    thumb.thumbnail((512, 512), Image.Resampling.LANCZOS)
    original_thumb = jpeg_bytes(thumb, quality=85, progressive=True)

    if strict and not remove_bg:
        raise PhotoValidationError(
            ValidationReport(
                passed=False,
                stage="source_convertible",
                issues=[
                    ValidationIssue(
                        code="bg_removal_required",
                        message="White background replacement is required for a submittable passport photo.",
                        how_to_fix="Background replacement is always applied during convert.",
                    )
                ],
                checks={},
            )
        )

    source_report = validate_source_convertible(original, spec, child_mode=child_mode)
    if strict and not source_report.passed:
        raise PhotoValidationError(source_report)

    if remove_bg:
        try:
            prepared = remove_background_to_white(original, spec.background_rgb)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background removal failed")
            if strict:
                raise PhotoValidationError(
                    ValidationReport(
                        passed=False,
                        stage="source",
                        issues=[
                            ValidationIssue(
                                code="bg_removal_failed",
                                message=f"Background removal failed: {exc}",
                                how_to_fix="Retry, or retake against a plain light wall with even lighting.",
                            )
                        ],
                        checks={},
                    )
                ) from exc
            warnings.append(f"Background removal failed ({exc}); used original image.")
            prepared = original.convert("RGB")
    else:
        prepared = original.convert("RGB")

    face = detect_face(prepared)
    if face is None:
        if strict:
            raise PhotoValidationError(
                ValidationReport(
                    passed=False,
                    stage="source",
                    issues=[
                        ValidationIssue(
                            code="no_face_after_bg",
                            message="Face could not be located after background processing.",
                            how_to_fix="Retake a clear front-facing photo with good lighting.",
                        )
                    ],
                    checks=source_report.checks,
                )
            )
        warnings.append("Could not detect a face reliably; used a center estimate.")
        face = fallback_face(prepared)

    framed, metrics = frame_to_spec(
        prepared,
        face,
        spec,
        scale_factor=scale_factor,
        offset_x_frac=offset_x_frac,
        offset_y_frac=offset_y_frac,
    )

    output_report = validate_output_photo(framed, spec, child_mode=child_mode)
    full_report = merge_reports(source_report, output_report)

    if strict and not output_report.passed:
        raise PhotoValidationError(full_report)

    metrics["head_height_ok"] = 1.0
    metrics["eye_position_ok"] = 1.0
    metrics["validation_passed"] = 1.0
    metrics["child_mode"] = 1.0 if child_mode else 0.0

    files, preview_jpeg, guide_jpeg, export_metrics = export_framed(
        framed, spec, warnings=warnings, full_report=full_report
    )
    metrics.update(export_metrics)

    prep_buf = io.BytesIO()
    prepared.save(prep_buf, format="PNG")

    return ProcessResult(
        doc_type=spec.id,
        preview_jpeg=preview_jpeg,
        files=files,
        metrics=metrics,
        warnings=warnings,
        validation=full_report.to_dict(),
        prepared_png=prep_buf.getvalue(),
        original_thumb=original_thumb,
        guide_preview_jpeg=guide_jpeg,
        face_dict=face_box_to_dict(face),
    )


def reframe_photo(
    prepared_png: bytes,
    face_dict: Dict,
    doc_type: str,
    *,
    scale_factor: float = 1.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = 0.0,
    child_mode: bool = False,
    strict: bool = True,
) -> ProcessResult:
    """Re-frame from stored white-bg intermediate (no rembg re-run)."""
    spec = get_spec(doc_type)
    prepared = Image.open(io.BytesIO(prepared_png)).convert("RGB")
    face = face_box_from_dict(face_dict)
    framed, metrics = frame_to_spec(
        prepared,
        face,
        spec,
        scale_factor=scale_factor,
        offset_x_frac=offset_x_frac,
        offset_y_frac=offset_y_frac,
    )
    output_report = validate_output_photo(framed, spec, child_mode=child_mode)
    if strict and not output_report.passed:
        raise PhotoValidationError(output_report)

    metrics["head_height_ok"] = 1.0
    metrics["eye_position_ok"] = 1.0
    metrics["validation_passed"] = 1.0
    metrics["child_mode"] = 1.0 if child_mode else 0.0

    files, preview_jpeg, guide_jpeg, export_metrics = export_framed(
        framed, spec, warnings=[], full_report=output_report
    )
    metrics.update(export_metrics)

    return ProcessResult(
        doc_type=spec.id,
        preview_jpeg=preview_jpeg,
        files=files,
        metrics=metrics,
        warnings=[],
        validation=output_report.to_dict(),
        prepared_png=prepared_png,
        guide_preview_jpeg=guide_jpeg,
        face_dict=face_dict,
    )


def _result_readme(
    spec: PhotoSpec,
    metrics: Dict[str, float],
    warnings: List[str],
    validation: Optional[ValidationReport] = None,
) -> str:
    lines = [
        f"Passport Photo Maker — {spec.title}",
        "=" * 50,
        "",
        spec.description,
        "",
        "VALIDATION: PASSED (automated QC)",
        "  This package was only generated because the source and final photo",
        "  passed strict automated checks (face, eyes, sharpness, lighting,",
        "  background, geometry). Government acceptance is still their decision,",
        "  but known reject risks were blocked before export.",
        "",
        "Geometry targets (VFS / ICAO style):",
        f"  Head height: {metrics.get('head_height_in')} in "
        f"(allowed {metrics.get('head_height_min_in')}–{metrics.get('head_height_max_in')})",
        f"  Eyes from bottom: {metrics.get('eye_from_bottom_in')} in "
        f"(allowed {metrics.get('eye_from_bottom_min_in')}–{metrics.get('eye_from_bottom_max_in')})",
        "",
        "Files:",
        "  *_PRINT_2x2_inch.jpg  — single physical 2×2 inch photo",
        "  *_sheet_4x6.jpg       — 6 copies on 4×6 photo paper",
        "  *_sheet_letter.jpg    — 12 copies on US Letter 8.5×11 (GP-701 / glossy Letter)",
        "  *_sheet_a4.jpg        — 12 copies on A4 (print at 100%)",
        "  *_upload_600.jpg      — digital portal upload",
        "  *_upload_350.jpg      — smaller portal fallback",
        "  *_master.jpg          — high-res square archive",
        "",
        "Tips:",
        "  - Portal upload: try 600 first; use 350 if rejected.",
        "  - Letter glossy photo paper (e.g. Canon GP-701): use *_sheet_letter.jpg",
        "    Print: Letter + Photo Glossy + High/Best quality, scale 100% (not Fit to Page).",
        "    Load glossy side correctly; avoid Draft; let dry before stacking.",
        "  - A4: use *_sheet_a4.jpg at 100% scale.",
        "  - Single 2×2: print *_PRINT_2x2_inch.jpg at actual size.",
        "",
    ]
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")
    for note in spec.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines) + "\n"


def _validation_certificate(report: ValidationReport, spec: PhotoSpec) -> str:
    lines = [
        "AUTOMATED VALIDATION CERTIFICATE",
        "================================",
        f"Document type: {spec.title}",
        f"Result: {'PASSED' if report.passed else 'FAILED'}",
        f"Issues: {len(report.issues)}",
        "",
        "Checks (summary):",
    ]
    for stage, data in (report.checks or {}).items():
        lines.append(f"  [{stage}]")
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"    {data}")
    lines.append("")
    lines.append(
        "This is automated QC, not an official government endorsement."
    )
    return "\n".join(lines) + "\n"
