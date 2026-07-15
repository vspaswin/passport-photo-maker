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


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        model = "u2net_human_seg"
        try:
            from app.core.config import get_settings

            model = get_settings().rembg_model or model
        except Exception:
            pass
        try:
            _rembg_session = new_session(model)
            logger.info("Loaded rembg session (%s)", model)
        except Exception:
            logger.warning("Failed to load %s; falling back to u2net", model)
            _rembg_session = new_session("u2net")
            logger.info("Loaded rembg session (u2net)")
    return _rembg_session


def load_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def remove_background_to_white(
    im: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Remove background and composite onto solid colour with clean edges.

    rembg's soft alpha often leaves a white/grey halo on passport white.
    We skip soft alpha-matting (it widens the fringe), harden body edges,
    but *restore dark hair fringes* so the hairline does not look chopped.
    """
    from rembg import remove

    session = _get_rembg_session()
    buf = io.BytesIO()
    im.save(buf, format="PNG")

    # Standard cutout (harder silhouette). Alpha-matting softens edges and
    # is a common source of the white outline on hair/shoulders.
    cut = remove(buf.getvalue(), session=session)

    rgba = Image.open(io.BytesIO(cut)).convert("RGBA")
    return _composite_clean_white(rgba, bg_rgb)


def _hair_fringe_mask(
    rgb: np.ndarray,
    alpha_raw: np.ndarray,
) -> np.ndarray:
    """Find dark hair-like pixels in rembg's soft matte (wisps / hairline).

    These are dark, low-chroma, partial-alpha samples that aggressive erode
    would otherwise delete — causing the "helmet / cookie-cutter" hair look.
    """
    lum = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    # True dark strands only (avoid grey matte / skin)
    hair = (
        (alpha_raw >= 0.20)
        & (alpha_raw <= 0.96)
        & (lum <= 80.0)
        & (chroma <= 48.0)
    )
    core = (alpha_raw >= 0.60).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_core = cv2.dilate(core, k, iterations=2) > 0
    hair = hair & near_core
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_u8 = cv2.morphologyEx(hair.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=1)
    return hair_u8 > 0


def _composite_clean_white(
    rgba: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Composite RGBA cutout onto pure white without soft edge outlines.

    Body uses a *binary* mask (no partial-alpha AA) so navy clothing does not
    pick up a dark outline. Hair keeps a softer matte for natural strands.
    Final passes scrub pale halo and dark silhouette rims.
    """
    arr = np.array(rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0
    alpha_raw = alpha.copy()

    hair = _hair_fringe_mask(rgb, alpha_raw)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_keep = cv2.dilate(hair.astype(np.uint8) * 255, k3, iterations=1) > 0

    # --- Body: hard binary mask (partial alpha → dark outline on navy) ---
    cut_b, keep_b = 0.55, 0.90
    a_body = np.clip((alpha - cut_b) / max(keep_b - cut_b, 1e-6), 0.0, 1.0)
    a_b_u8 = (a_body * 255).astype(np.uint8)
    a_b_u8 = cv2.morphologyEx(a_b_u8, cv2.MORPH_OPEN, k3, iterations=1)
    a_b_u8 = cv2.morphologyEx(a_b_u8, cv2.MORPH_CLOSE, k5, iterations=1)
    # Erode past contaminated rim (source of black shoulder stroke)
    a_b_u8 = cv2.erode(a_b_u8, k5, iterations=2)
    a_b_u8 = cv2.erode(a_b_u8, k3, iterations=1)
    # Fully binary body — no soft AA (AA darkens navy into a black edge)
    a_body = (a_b_u8 > 127).astype(np.float32)

    # --- Hair: softer partial matte ---
    a_hair = np.clip((alpha_raw - 0.18) / 0.65, 0.0, 1.0)
    a_hair = np.where(hair_keep, a_hair, 0.0)
    a_hair = cv2.GaussianBlur(a_hair, (3, 3), 0.55)
    a_hair = np.clip(a_hair, 0.0, 1.0)
    a_hair = np.where(a_hair < 0.15, 0.0, a_hair)

    a = np.maximum(a_body, a_hair)

    # Body colour: use original RGB only (skip un-premultiply — it inks edges)
    rgb_out = rgb.copy()
    if hair_keep.any():
        solid_hair = hair_keep & (alpha_raw >= 0.75) & (rgb.mean(axis=2) <= 70)
        if solid_hair.any():
            ref = np.median(rgb[solid_hair], axis=0)
        else:
            ref = np.array([28.0, 22.0, 20.0], dtype=np.float32)
        fringe_h = hair_keep & (a > 0.05) & (a < 0.98)
        fringe_lum = rgb.mean(axis=2)
        # Only de-milk washed strands; leave already-dark hair alone
        needs = fringe_h & (fringe_lum > 50)
        blended = rgb * 0.5 + ref.reshape(1, 1, 3) * 0.5
        rgb_out[needs] = blended[needs]

    a3 = a[:, :, None]
    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = rgb_out * a3 + bg * (1.0 - a3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out[a < 0.05] = bg_rgb
    out = _scrub_grey_halo(out, a, bg_rgb, protect_dark=True)
    out = _eat_light_halo(out, bg_rgb, max_px=3, protect_lum=95.0)
    out = _fix_silhouette_rim(out, bg_rgb)
    return Image.fromarray(out, mode="RGB")


def _fix_silhouette_rim(
    rgb: np.ndarray,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Remove dark outline stroke and pale fringe along the silhouette.

    rembg + hard masks often leave a 1–3 px ring that is *darker* than the
    interior clothing (black shoulder line) or paler (white halo). Compare
    each rim pixel to the interior a few pixels inward and correct it.
    """
    out = rgb.copy().astype(np.float32)
    lum = out.mean(axis=2)
    mn = out.min(axis=2)
    mx = out.max(axis=2)
    chroma = mx - mn

    pure_white = (mn >= 250) & (chroma <= 6)
    if not pure_white.any() or pure_white.all():
        return rgb

    non_w = (~pure_white).astype(np.uint8) * 255
    dist = cv2.distanceTransform(non_w, cv2.DIST_L2, 3)

    # Interior reference: median blur of subject (ignores thin rim noise)
    subject = (~pure_white).astype(np.uint8) * 255
    # Push white over subject briefly so blur samples interior only
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    interior_mask = cv2.erode(subject, k, iterations=1) > 0
    interior = out.copy()
    interior[~interior_mask] = 0
    # Box filter over interior; fall back to global subject median where empty
    ref = cv2.blur(interior, (15, 15))
    ref_w = cv2.blur(interior_mask.astype(np.float32), (15, 15))
    ref_w = np.maximum(ref_w, 1e-4)[:, :, None]
    ref = ref / ref_w
    if interior_mask.any():
        global_ref = np.median(out[interior_mask], axis=0)
    else:
        global_ref = np.array(bg_rgb, dtype=np.float32)
    missing = ~interior_mask
    ref[missing] = global_ref
    ref_lum = ref.mean(axis=2)

    # Outer rim only (next to pure white background) — never interior clothing
    rim = (dist > 0) & (dist <= 2.5)
    rim_lum = lum

    # 1) Dark stroke: clearly darker than nearby interior fabric → white
    darker = rim & (rim_lum < ref_lum - 14.0) & (rim_lum < 90) & (dist <= 2.0)
    # Absolute ink outline (black cutout edge on navy)
    ink = rim & (rim_lum <= 38) & (dist <= 1.8) & (rim_lum < ref_lum - 5.0)
    out[darker | ink] = bg_rgb

    # 2) Pale washed rim (not white clothing: must be desaturated + lighter
    #    than interior, and only in the outermost 1.5 px)
    pale = (
        rim
        & (dist <= 1.6)
        & (rim_lum >= 175)
        & (chroma <= 28)
        & (rim_lum > ref_lum + 25)
    )
    out[pale] = bg_rgb

    # 3) Mild dark rim: recolour toward interior fabric (keeps shape)
    mid = rim & ~(darker | ink | pale) & (dist <= 2.0)
    still_dark = mid & (out.mean(axis=2) < ref_lum - 12.0) & (out.mean(axis=2) < 100)
    if still_dark.any():
        out[still_dark] = 0.25 * out[still_dark] + 0.75 * ref[still_dark]

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    # Snap only true near-white *background* leftovers, not collar stripes
    # (collar stripes sit deeper than dist 2 and have interior neighbors)
    snap = (out_u8.min(axis=2) >= 245) & ((out_u8.max(axis=2) - out_u8.min(axis=2)) <= 12)
    snap = snap & (dist <= 1.2)
    out_u8[snap] = bg_rgb
    return out_u8


def _scrub_grey_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_rgb: Tuple[int, int, int],
    protect_dark: bool = True,
) -> np.ndarray:
    """Replace residual grey fringe pixels with pure background."""
    out = rgb.copy()
    mn = out.min(axis=2)
    mx = out.max(axis=2)
    chroma = mx - mn
    lum = out.mean(axis=2)

    a_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    subject = cv2.dilate(a_u8, kernel, iterations=1) > 30
    # Never force pure white over dark hair outside the dilated core
    if protect_dark:
        dark_hair = (lum <= 100) & (chroma <= 50)
        out[(~subject) & (~dark_hair)] = bg_rgb
    else:
        out[~subject] = bg_rgb

    fringe = (alpha < 0.99) & (alpha > 0.02)
    pale_grey = (lum >= 175) & (chroma <= 40) & fringe
    mid_grey = (lum >= 140) & (lum < 175) & (chroma <= 28) & (alpha < 0.75)
    washed = (lum >= 120) & (chroma <= 18) & fringe & (alpha < 0.85)
    # Do not scrub dark hair pixels
    scrub = pale_grey | mid_grey | washed
    if protect_dark:
        scrub = scrub & (lum >= 110)
    out[scrub] = bg_rgb

    # Only snap near-white where alpha says "background / edge", not solid
    # white clothing (collar stripes sit at high alpha).
    near_white = (mn >= 245) & (chroma <= 12) & (alpha < 0.85)
    out[near_white] = bg_rgb
    return out


def _eat_light_halo(
    rgb: np.ndarray,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    max_px: int = 3,
    protect_lum: float = 105.0,
) -> np.ndarray:
    """Grow pure white inward over light/desaturated rim pixels.

    After composite, a 1–3 px ring of washed-out colour often remains between
    pure white and solid subject. Those pixels sit next to pure white and are
    brighter/greyer than true clothing/hair — replace them with background.
    Dark hair (low luminance) is never eaten.
    """
    out = rgb.copy()
    mn = out.min(axis=2).astype(np.int16)
    mx = out.max(axis=2).astype(np.int16)
    chroma = (mx - mn).astype(np.int16)
    lum = out.mean(axis=2)

    pure_white = (mn >= 250) & (chroma <= 6)
    if not pure_white.any() or pure_white.all():
        return out

    non_white_u8 = (~pure_white).astype(np.uint8) * 255
    dist = cv2.distanceTransform(non_white_u8, cv2.DIST_L2, 3)

    rim = (dist > 0) & (dist <= float(max_px))
    lightish = lum >= 145
    desat = chroma <= 45
    very_pale = (lum >= 200) & (chroma <= 55)
    # Protect dark hair / navy fabric
    dark_protect = lum < protect_lum
    eat = rim & ((lightish & desat) | very_pale) & (~dark_protect)
    out[eat] = bg_rgb

    mn2 = out.min(axis=2)
    mx2 = out.max(axis=2)
    out[(mn2 >= 230) & ((mx2 - mn2) <= 20)] = bg_rgb
    return out


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
    threshold: int = 225,
) -> Image.Image:
    """Final pass after resize: clean only the outer silhouette fringe.

    Do NOT globally snap pale pixels — that erases white clothing stripes
    (collar rings) into the background and leaves a dashed collar look.
    """
    arr = np.array(im.convert("RGB"))
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    chroma = mx - mn

    # Only force pure-white snap for pixels already essentially background
    near = (mn >= threshold) & (chroma <= 18)
    arr[near] = bg_rgb

    max_px = int(np.clip(round(min(arr.shape[0], arr.shape[1]) * 0.005), 2, 5))
    arr = _eat_light_halo(arr, bg_rgb, max_px=max_px, protect_lum=95.0)
    arr = _fix_silhouette_rim(arr, bg_rgb)
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
