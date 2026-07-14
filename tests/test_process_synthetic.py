"""Smoke test framing without requiring rembg model download."""

import io

from PIL import Image, ImageDraw

from app.engine.process import detect_face, frame_to_spec, load_image, make_print_sheet
from app.engine.specs import get_spec


def _synthetic_portrait() -> bytes:
    """Simple face-like oval on white for OpenCV cascade (may or may not detect)."""
    im = Image.new("RGB", (800, 1000), (255, 255, 255))
    d = ImageDraw.Draw(im)
    # shoulders
    d.ellipse([200, 650, 600, 1100], fill=(30, 60, 140))
    # head
    d.ellipse([260, 180, 540, 520], fill=(180, 130, 100))
    # hair
    d.ellipse([250, 140, 550, 320], fill=(30, 25, 20))
    # eyes
    d.ellipse([320, 300, 360, 340], fill=(40, 30, 25))
    d.ellipse([440, 300, 480, 340], fill=(40, 30, 25))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_frame_to_spec_with_fallback():
    from app.engine.face import fallback_face

    data = _synthetic_portrait()
    im = load_image(data)
    face = detect_face(im) or fallback_face(im)
    spec = get_spec("indian-passport")
    framed, metrics = frame_to_spec(im, face, spec, out_px=600)
    assert framed.size == (600, 600)
    assert metrics["head_height_in"] > 0
    assert metrics["eye_from_bottom_in"] > 0


def test_print_sheet_size():
    photo = Image.new("RGB", (400, 400), (200, 100, 80))
    sheet = make_print_sheet(
        photo, page_inches=(4.0, 6.0), cols=2, rows=3, photo_inches=(2.0, 2.0), dpi=100
    )
    assert sheet.size == (400, 600)


def test_letter_print_sheet_size():
    """US Letter 8.5×11 at 100 dpi → 850×1100 px; 3×4 grid of 2×2 photos."""
    photo = Image.new("RGB", (400, 400), (200, 100, 80))
    sheet = make_print_sheet(
        photo,
        page_inches=(8.5, 11.0),
        cols=3,
        rows=4,
        photo_inches=(2.0, 2.0),
        dpi=100,
    )
    assert sheet.size == (850, 1100)
