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


def test_list_document_types():
    items = list_document_types()
    assert any(i["id"] == "indian-passport" for i in items)
