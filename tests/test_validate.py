"""Tests for as-is vs convertible validation and convert path."""

import io

from PIL import Image

from app.engine.specs import get_spec
from app.engine.validate import (
    PhotoValidationError,
    assess_photo,
    validate_source_as_is,
    validate_source_convertible,
    validate_source_photo,
)
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


def test_blank_fails_as_is_and_convertible():
    im = Image.new("RGB", (800, 1000), (255, 255, 255))
    spec = get_spec("indian-passport")
    as_is = validate_source_as_is(im, spec)
    conv = validate_source_convertible(im, spec)
    assert as_is.passed is False
    assert conv.passed is False
    assert any(i.code == "no_face" for i in conv.issues)


def test_validate_source_photo_aliases_as_is():
    im = Image.new("RGB", (800, 1000), (255, 255, 255))
    spec = get_spec("indian-passport")
    a = validate_source_photo(im, spec)
    b = validate_source_as_is(im, spec)
    assert a.passed == b.passed
    assert a.stage == "source_as_is"


def test_tiny_image_fails_convertible():
    im = Image.open(io.BytesIO(_tiny_image()))
    report = validate_source_convertible(im, get_spec("indian-passport"))
    assert report.passed is False


def test_assess_photo_retake_on_blank():
    im = Image.new("RGB", (800, 1000), (255, 255, 255))
    a = assess_photo(im, get_spec("indian-passport"))
    assert a["recommendation"] == "retake"
    assert a["can_convert"] is False


def test_process_photo_strict_raises_on_blank():
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
