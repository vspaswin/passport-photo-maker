from app.engine.specs import DOCUMENT_TYPES, get_spec, list_document_types


def test_indian_passport_registered():
    assert "indian-passport" in DOCUMENT_TYPES
    spec = get_spec("indian-passport")
    assert spec.photo_inches == (2.0, 2.0)
    assert spec.head_height_min < spec.head_height_target < spec.head_height_max
    assert spec.eye_from_bottom_min < spec.eye_from_bottom_target < spec.eye_from_bottom_max
    assert len(spec.upload_variants) >= 1
    assert len(spec.print_sheets) >= 1
    suffixes = {s.filename_suffix for s in spec.print_sheets}
    assert "sheet_letter" in suffixes
    letter = next(s for s in spec.print_sheets if s.filename_suffix == "sheet_letter")
    assert letter.page_inches == (8.5, 11.0)
    assert letter.cols * letter.rows == 12


def test_passport_seva_35x45_registered():
    assert "passport-seva-35x45" in DOCUMENT_TYPES
    spec = get_spec("passport-seva-35x45")
    assert not spec.is_square
    assert not spec.require_square
    assert abs(spec.photo_mm[0] - 35.0) < 0.01
    assert abs(spec.photo_mm[1] - 45.0) < 0.01
    w, h = spec.output_size()
    assert h > w
    seva = next(v for v in spec.upload_variants if v.filename_suffix == "seva_630x810")
    assert seva.pixel_size() == (630, 810)
    assert seva.max_kb == 250
    assert seva.exact_pixels is True


def test_list_document_types():
    items = list_document_types()
    ids = {i["id"] for i in items}
    assert "indian-passport" in ids
    assert "passport-seva-35x45" in ids
