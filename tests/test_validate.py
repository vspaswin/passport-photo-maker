"""Tests for strict passport validation."""

import io

from PIL import Image, ImageDraw

from app.engine.specs import get_spec
from app.engine.validate import PhotoValidationError, validate_source_photo
from app.engine.process import process_photo


def _blank_image(size=(800, 1000), color=(255, 255, 255)):
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _tiny_image():
    im = Image.new("RGB", (100, 100), (120, 100, 90))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def test_blank_image_fails_validation():
    im = Image.new("RGB", (800, 1000), (255, 255, 255))
    report = validate_source_photo(im, get_spec("indian-passport"))
    assert report.passed is False
    codes = {i.code for i in report.issues}
    assert "no_face" in codes


def test_tiny_image_fails_resolution_or_face():
    im = Image.open(io.BytesIO(_tiny_image()))
    report = validate_source_photo(im, get_spec("indian-passport"))
    assert report.passed is False
    codes = {i.code for i in report.issues}
    assert "resolution_too_low" in codes or "no_face" in codes


def test_process_photo_strict_raises_on_blank():
    """No face → rejected before any downloads (no rembg needed)."""
    try:
        process_photo(_blank_image(), strict=True, remove_bg=True)
        assert False, "expected PhotoValidationError"
    except PhotoValidationError as exc:
        assert exc.report.passed is False
        codes = {i.code for i in exc.report.issues}
        assert "no_face" in codes


def test_process_photo_requires_bg_removal_in_strict():
    try:
        process_photo(_blank_image(), strict=True, remove_bg=False)
        assert False, "expected PhotoValidationError"
    except PhotoValidationError as exc:
        codes = {i.code for i in exc.report.issues}
        assert "bg_removal_required" in codes
