"""Document photo specifications.

Indian passport geometry follows VFS Global / MEA photo specification
for Passport / Visa / OCI (ISO/ICAO-aligned):
  https://visa.vfsglobal.com/one-pager/india/united-states-of-america/passport-services/pdf/photo-specifiation.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class UploadVariant:
    """Digital file intended for online portals."""

    name: str
    filename_suffix: str
    size_px: int
    min_kb: int = 10
    max_kb: int = 100
    description: str = ""


@dataclass(frozen=True)
class PrintSheet:
    """Multi-copy sheet for physical printing."""

    name: str
    filename_suffix: str
    page_inches: Tuple[float, float]  # width, height
    cols: int
    rows: int
    description: str = ""


@dataclass(frozen=True)
class PhotoSpec:
    """Geometry and export rules for one document type."""

    id: str
    title: str
    description: str
    # Physical photo size
    photo_inches: Tuple[float, float] = (2.0, 2.0)
    # Head height (top of hair → chin) as fraction of photo height
    head_height_min: float = 1.0 / 2.0  # 1.0 in on 2 in
    head_height_max: float = 1.375 / 2.0  # 1.375 in on 2 in
    head_height_target: float = 1.22 / 2.0
    # Eye line from bottom of photo as fraction of height
    eye_from_bottom_min: float = 1.125 / 2.0
    eye_from_bottom_max: float = (1.0 + 1.0 / 3.0) / 2.0
    eye_from_bottom_target: float = 1.22 / 2.0
    background_rgb: Tuple[int, int, int] = (255, 255, 255)
    print_dpi: int = 600
    print_px: int = 1200  # 2 in @ 600 dpi
    upload_variants: Tuple[UploadVariant, ...] = field(default_factory=tuple)
    print_sheets: Tuple[PrintSheet, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_square(self) -> bool:
        return abs(self.photo_inches[0] - self.photo_inches[1]) < 1e-6


INDIAN_PASSPORT = PhotoSpec(
    id="indian-passport",
    title="Indian Passport / Visa / OCI",
    description=(
        "2×2 inch colour photo with plain white background. "
        "Full frontal face, eyes open, natural expression. "
        "Based on VFS Global / MEA photo specification (ISO/ICAO)."
    ),
    photo_inches=(2.0, 2.0),
    head_height_min=1.0 / 2.0,
    head_height_max=1.375 / 2.0,
    head_height_target=1.20 / 2.0,
    eye_from_bottom_min=1.125 / 2.0,
    eye_from_bottom_max=(1.0 + 1.0 / 3.0) / 2.0,
    eye_from_bottom_target=1.22 / 2.0,
    background_rgb=(255, 255, 255),
    print_dpi=600,
    print_px=1200,
    upload_variants=(
        UploadVariant(
            name="Portal upload (600×600)",
            filename_suffix="upload_600",
            size_px=600,
            min_kb=10,
            max_kb=100,
            description="Primary digital file for most online portals (~10–100 KB JPEG).",
        ),
        UploadVariant(
            name="Portal upload (350×350)",
            filename_suffix="upload_350",
            size_px=350,
            min_kb=10,
            max_kb=100,
            description="Fallback smaller file if the portal rejects larger dimensions.",
        ),
    ),
    print_sheets=(
        PrintSheet(
            name="4×6 photo sheet",
            filename_suffix="sheet_4x6",
            page_inches=(4.0, 6.0),
            cols=2,
            rows=3,
            description="Six 2×2 photos on standard 4×6 photo paper.",
        ),
        PrintSheet(
            name="Letter print sheet (8.5×11)",
            filename_suffix="sheet_letter",
            page_inches=(8.5, 11.0),
            cols=3,
            rows=4,
            description=(
                "Twelve true 2×2 photos on US Letter (8.5×11). "
                "Use for Canon GP-701 / glossy Letter photo paper: print at 100% scale, "
                "Paper=Letter, Type=Photo Glossy, Quality=High/Best."
            ),
        ),
        PrintSheet(
            name="A4 print sheet",
            filename_suffix="sheet_a4",
            page_inches=(8.27, 11.69),
            cols=3,
            rows=4,
            description="Twelve 2×2 photos on A4 (print at 100% scale).",
        ),
    ),
    notes=(
        "Print on photo paper when possible (e.g. Canon GP-701 glossy Letter); continuous-tone quality.",
        "Letter glossy: use *_sheet_letter.jpg at 100% / Actual size — not Fit to Page.",
        "Do not heavily beautify or distort the face (true likeness required).",
        "Wear coloured clothing (not pure white); avoid busy patterns.",
        "Children under 10: face/eye geometry may be slightly relaxed.",
    ),
)

DOCUMENT_TYPES: Dict[str, PhotoSpec] = {
    INDIAN_PASSPORT.id: INDIAN_PASSPORT,
}


def get_spec(doc_type: str) -> PhotoSpec:
    key = (doc_type or "").strip().lower()
    if key not in DOCUMENT_TYPES:
        known = ", ".join(sorted(DOCUMENT_TYPES))
        raise KeyError(f"Unknown document type '{doc_type}'. Known: {known}")
    return DOCUMENT_TYPES[key]


def list_document_types() -> List[dict]:
    return [
        {
            "id": s.id,
            "title": s.title,
            "description": s.description,
            "photo_inches": list(s.photo_inches),
            "notes": list(s.notes),
        }
        for s in DOCUMENT_TYPES.values()
    ]
