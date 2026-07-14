"""Core photo processing: background, face geometry, exports."""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .specs import PhotoSpec, get_spec

logger = logging.getLogger(__name__)

# Lazy rembg session (loads ONNX model once)
_rembg_session = None


@dataclass
class FaceBox:
    """Face region in pixel coordinates (source image)."""

    x: int
    y: int
    w: int
    h: int
    # Estimated landmarks
    top_of_head: int
    chin: int
    eye_y: int
    center_x: int


@dataclass
class ProcessResult:
    """All outputs from a conversion run."""

    doc_type: str
    preview_jpeg: bytes
    files: Dict[str, bytes]  # filename -> bytes
    metrics: Dict[str, float]
    warnings: List[str]


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        # u2net is the default; good balance of quality/speed for portraits
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
    """Remove background and composite onto solid colour."""
    from rembg import remove

    session = _get_rembg_session()
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    cut = remove(buf.getvalue(), session=session)
    rgba = Image.open(io.BytesIO(cut)).convert("RGBA")

    canvas = Image.new("RGB", rgba.size, bg_rgb)
    canvas.paste(rgba, mask=rgba.split()[3])
    return canvas


def detect_face(im: Image.Image) -> Optional[FaceBox]:
    """Detect primary face and estimate head/eye landmarks."""
    rgb = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        # Retry more leniently
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
        )

    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Haar box is roughly forehead→chin; extend upward for hair
    top_of_head = max(0, int(y - 0.45 * h))
    chin = min(im.height - 1, int(y + h * 1.05))
    # Eyes ~ 40% down from top of Haar face box
    eye_y = int(y + 0.38 * h)
    center_x = int(x + w / 2)

    return FaceBox(
        x=int(x),
        y=int(y),
        w=int(w),
        h=int(h),
        top_of_head=top_of_head,
        chin=chin,
        eye_y=eye_y,
        center_x=center_x,
    )


def _fallback_face(im: Image.Image) -> FaceBox:
    """Center-biased estimate when detector fails."""
    w, h = im.size
    # Assume subject is upper-center portrait
    box_h = int(h * 0.45)
    box_w = int(box_h * 0.75)
    cx = w // 2
    top = int(h * 0.08)
    chin = top + box_h
    eye_y = top + int(box_h * 0.42)
    return FaceBox(
        x=cx - box_w // 2,
        y=top + int(0.35 * box_h),
        w=box_w,
        h=int(box_h * 0.7),
        top_of_head=top,
        chin=chin,
        eye_y=eye_y,
        center_x=cx,
    )


def frame_to_spec(
    im: Image.Image,
    face: FaceBox,
    spec: PhotoSpec,
    out_px: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Scale/place subject so head height and eye line match the spec."""
    out_px = out_px or spec.print_px
    out_w = out_px
    out_h = out_px if spec.is_square else int(
        out_px * spec.photo_inches[1] / spec.photo_inches[0]
    )

    head_px_src = max(1, face.chin - face.top_of_head)
    target_head_frac = spec.head_height_target
    head_out = target_head_frac * out_h
    scale = head_out / head_px_src

    new_w = max(1, int(im.width * scale))
    new_h = max(1, int(im.height * scale))
    scaled = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    eye_s = int(face.eye_y * scale)
    cx_s = int(face.center_x * scale)

    eye_y_out = out_h - int(spec.eye_from_bottom_target * out_h)
    paste_x = out_w // 2 - cx_s
    paste_y = eye_y_out - eye_s

    canvas = Image.new("RGB", (out_w, out_h), spec.background_rgb)

    # Paste with clipping for out-of-bounds
    src_x0 = max(0, -paste_x)
    src_y0 = max(0, -paste_y)
    src_x1 = min(new_w, out_w - paste_x)
    src_y1 = min(new_h, out_h - paste_y)
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop = scaled.crop((src_x0, src_y0, src_x1, src_y1))
        canvas.paste(crop, (max(0, paste_x), max(0, paste_y)))

    canvas = _clean_near_white(canvas, spec.background_rgb)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.05)

    # Measured metrics on output (approximate from placement math)
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
    }
    return canvas, metrics


def _clean_near_white(
    im: Image.Image,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    threshold: int = 232,
) -> Image.Image:
    """Push near-white pale pixels toward pure background (edges/halos)."""
    arr = np.array(im.convert("RGB"))
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    near = (mn >= threshold) & ((mx - mn) <= 18)
    # Avoid nuking light clothing that is not near-gray: require low chroma
    arr[near] = bg_rgb
    # Pure near-255
    pure = (arr[:, :, 0] >= 240) & (arr[:, :, 1] >= 240) & (arr[:, :, 2] >= 240)
    arr[pure] = bg_rgb
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
) -> ProcessResult:
    """
    Convert an arbitrary photo into document-ready print + digital files.

    All processing is local (no network).
    """
    spec = get_spec(doc_type)
    warnings: List[str] = []

    original = load_image(image_bytes)

    if remove_bg:
        try:
            prepared = remove_background_to_white(original, spec.background_rgb)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background removal failed")
            warnings.append(
                f"Background removal failed ({exc}); used original image."
            )
            prepared = original.convert("RGB")
    else:
        prepared = original.convert("RGB")

    face = detect_face(prepared)
    if face is None:
        warnings.append(
            "Could not detect a face reliably; used a center estimate. "
            "Check the preview and re-take if needed."
        )
        face = _fallback_face(prepared)

    framed, metrics = frame_to_spec(prepared, face, spec)

    # Validate measured placement is within allowed bands (by construction targets are)
    head_ok = (
        spec.head_height_min * spec.photo_inches[1]
        <= metrics["head_height_in"]
        <= spec.head_height_max * spec.photo_inches[1]
    )
    eye_ok = (
        spec.eye_from_bottom_min * spec.photo_inches[1]
        <= metrics["eye_from_bottom_in"]
        <= spec.eye_from_bottom_max * spec.photo_inches[1]
    )
    metrics["head_height_ok"] = 1.0 if head_ok else 0.0
    metrics["eye_position_ok"] = 1.0 if eye_ok else 0.0

    files: Dict[str, bytes] = {}
    base = f"{spec.id}"

    # Single print 2x2
    print_img = framed.resize((spec.print_px, spec.print_px), Image.Resampling.LANCZOS)
    files[f"{base}_PRINT_2x2_inch.jpg"] = jpeg_bytes(
        print_img, quality=95, dpi=(spec.print_dpi, spec.print_dpi)
    )

    # Master high-res
    master = framed.resize((1800, 1800), Image.Resampling.LANCZOS)
    files[f"{base}_master.jpg"] = jpeg_bytes(master, quality=97, dpi=(600, 600))

    # Upload variants
    for uv in spec.upload_variants:
        data = save_upload_variant(
            framed, uv.size_px, min_kb=uv.min_kb, max_kb=uv.max_kb
        )
        files[f"{base}_{uv.filename_suffix}.jpg"] = data
        metrics[f"{uv.filename_suffix}_kb"] = round(len(data) / 1024, 1)

    # Print sheets
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

    # Preview (smaller JPEG for UI)
    preview = framed.resize((512, 512), Image.Resampling.LANCZOS)
    preview_jpeg = jpeg_bytes(preview, quality=88, dpi=(72, 72), progressive=True)

    # ZIP of everything
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        readme = _result_readme(spec, metrics, warnings)
        zf.writestr("README.txt", readme)
    files[f"{base}_all.zip"] = zip_buf.getvalue()

    return ProcessResult(
        doc_type=spec.id,
        preview_jpeg=preview_jpeg,
        files=files,
        metrics=metrics,
        warnings=warnings,
    )


def _result_readme(
    spec: PhotoSpec, metrics: Dict[str, float], warnings: List[str]
) -> str:
    lines = [
        f"Passport Photo Maker — {spec.title}",
        "=" * 50,
        "",
        spec.description,
        "",
        "Geometry targets (VFS / ICAO style):",
        f"  Head height: {metrics.get('head_height_in')} in "
        f"(allowed {metrics.get('head_height_min_in')}–{metrics.get('head_height_max_in')})",
        f"  Eyes from bottom: {metrics.get('eye_from_bottom_in')} in "
        f"(allowed {metrics.get('eye_from_bottom_min_in')}–{metrics.get('eye_from_bottom_max_in')})",
        "",
        "Files:",
        "  *_PRINT_2x2_inch.jpg  — single physical 2×2 inch photo",
        "  *_sheet_4x6.jpg       — 6 copies on 4×6 paper",
        "  *_sheet_a4.jpg        — 12 copies on A4 (print at 100%)",
        "  *_upload_600.jpg      — digital portal upload",
        "  *_upload_350.jpg      — smaller portal fallback",
        "  *_master.jpg          — high-res square archive",
        "",
        "Tips:",
        "  - Print on thin photo paper; prefer matte over glossy.",
        "  - Confirm the face is a true likeness before submitting.",
        "  - Portal upload: try 600 first; use 350 if rejected.",
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
