"""Document photo specifications.

Sources cross-checked:
  - Passport Seva upload (May 2026): physical 35×45 mm, digital 630×810, <250 KB
  - Passportindia ICAO photo guidelines: 630×810, face ~80–85%, white BG
  - VFS / US-style Indian consular (common abroad): 2×2 inch
  - IDPhoto4You public sizes (reference layout only): 2×2 @ 600×600 / 35×45 @ 413×531 @300dpi
    https://www.idphoto4you.com/?Target=SamplePage_IN
    https://www.idphoto4you.com/?Target=SamplePage_Common
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class UploadVariant:
    """Digital file intended for online portals."""

    name: str
    filename_suffix: str
    # Square portals: set size_px. Rectangular: set width_px + height_px.
    size_px: int = 0
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    min_kb: int = 10
    max_kb: int = 100
    description: str = ""
    exact_pixels: bool = False  # if True, never downscale below W×H (Seva 630×810)

    def pixel_size(self) -> Tuple[int, int]:
        if self.width_px and self.height_px:
            return int(self.width_px), int(self.height_px)
        s = int(self.size_px or 600)
        return s, s


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
    # Physical photo size (inches). Also set photo_mm when metric is primary.
    photo_inches: Tuple[float, float] = (2.0, 2.0)
    photo_mm: Optional[Tuple[float, float]] = None  # (w_mm, h_mm) for labels
    # Head height (top of hair → chin) as fraction of photo height
    head_height_min: float = 1.0 / 2.0
    head_height_max: float = 1.375 / 2.0
    head_height_target: float = 1.22 / 2.0
    # Eye line from bottom of photo as fraction of height
    eye_from_bottom_min: float = 1.125 / 2.0
    eye_from_bottom_max: float = (1.0 + 1.0 / 3.0) / 2.0
    eye_from_bottom_target: float = 1.22 / 2.0
    background_rgb: Tuple[int, int, int] = (255, 255, 255)
    print_dpi: int = 600
    # Square legacy: single edge length. Prefer print_size_px for W×H.
    print_px: int = 1200
    print_size_px: Optional[Tuple[int, int]] = None
    upload_variants: Tuple[UploadVariant, ...] = field(default_factory=tuple)
    print_sheets: Tuple[PrintSheet, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    require_square: bool = True

    @property
    def is_square(self) -> bool:
        return abs(self.photo_inches[0] - self.photo_inches[1]) < 1e-6

    def output_size(self) -> Tuple[int, int]:
        """Working / print pixel size (width, height)."""
        if self.print_size_px is not None:
            return int(self.print_size_px[0]), int(self.print_size_px[1])
        if self.is_square:
            return self.print_px, self.print_px
        w = self.print_px
        h = max(1, int(round(self.print_px * self.photo_inches[1] / self.photo_inches[0])))
        return w, h


# ---------------------------------------------------------------------------
# VFS / US-style 2×2 (common for Indian passport abroad + US visa)
# ---------------------------------------------------------------------------

_SHEETS_2X2 = (
    PrintSheet(
        name="4×6 photo sheet",
        filename_suffix="sheet_4x6",
        page_inches=(4.0, 6.0),
        cols=2,
        rows=3,
        description="Six 2×2 photos on standard 4×6 photo paper (IDPhoto4You 10×15 cm style).",
    ),
    PrintSheet(
        name="3.5×5 photo sheet",
        filename_suffix="sheet_3p5x5",
        page_inches=(3.5, 5.0),
        cols=1,
        rows=2,
        description="Two 2×2 photos on 3.5×5 / 9×13 cm paper (IDPhoto4You print size).",
    ),
    PrintSheet(
        name="Letter print sheet (8.5×11)",
        filename_suffix="sheet_letter",
        page_inches=(8.5, 11.0),
        cols=3,
        rows=4,
        description=(
            "Twelve true 2×2 photos on US Letter. "
            "Canon GP-701: print at 100% scale, Photo Glossy, High quality."
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
)

INDIAN_PASSPORT_2X2 = PhotoSpec(
    id="indian-passport",
    title="India 2×2″ (VFS / abroad / US-style)",
    description=(
        "2×2 inch colour photo, plain white background. "
        "Typical for Indian passport/OCI via VFS abroad and US-style ICAO geometry. "
        "Also matches US visa 2×2."
    ),
    photo_inches=(2.0, 2.0),
    photo_mm=(50.8, 50.8),
    head_height_min=1.0 / 2.0,
    head_height_max=1.375 / 2.0,
    head_height_target=1.20 / 2.0,
    eye_from_bottom_min=1.125 / 2.0,
    eye_from_bottom_max=(1.0 + 1.0 / 3.0) / 2.0,
    eye_from_bottom_target=1.22 / 2.0,
    background_rgb=(255, 255, 255),
    print_dpi=600,
    print_px=1200,
    print_size_px=(1200, 1200),
    upload_variants=(
        UploadVariant(
            name="Portal upload (600×600 @300dpi)",
            filename_suffix="upload_600",
            size_px=600,
            min_kb=10,
            max_kb=100,
            description="IDPhoto4You-style 2×2 @ 300 dpi (600×600). Many portals accept this.",
        ),
        UploadVariant(
            name="Portal upload (350×350)",
            filename_suffix="upload_350",
            size_px=350,
            min_kb=10,
            max_kb=100,
            description="Smaller square fallback if a portal rejects larger files.",
        ),
    ),
    print_sheets=_SHEETS_2X2,
    notes=(
        "Use this for VFS US appointments / US visa when checklist says 2×2 inch.",
        "Print on photo paper at 100% / Actual size — not Fit to Page.",
        "Wear coloured clothing (not pure white).",
        "Not the same as Passport Seva India 35×45 mm upload (use that mode instead).",
    ),
    require_square=True,
)

# Back-compat alias name used in older code/docs
INDIAN_PASSPORT = INDIAN_PASSPORT_2X2

US_PASSPORT = PhotoSpec(
    id="us-passport",
    title="US Passport / Visa photo (2×2″)",
    description=(
        "2×2 inch colour photo, plain white or off-white background, "
        "full face, front view (State Department / ICAO style)."
    ),
    photo_inches=(2.0, 2.0),
    photo_mm=(50.8, 50.8),
    head_height_min=1.0 / 2.0,
    head_height_max=1.375 / 2.0,
    head_height_target=1.20 / 2.0,
    eye_from_bottom_min=1.125 / 2.0,
    eye_from_bottom_max=(1.0 + 1.0 / 3.0) / 2.0,
    eye_from_bottom_target=1.22 / 2.0,
    background_rgb=(255, 255, 255),
    print_dpi=600,
    print_px=1200,
    print_size_px=(1200, 1200),
    upload_variants=INDIAN_PASSPORT_2X2.upload_variants,
    print_sheets=_SHEETS_2X2,
    notes=(
        "Recent photo (within 6 months), neutral expression preferred.",
        "No glasses with glare; head coverings only for religious reasons.",
    ),
    require_square=True,
)

# ---------------------------------------------------------------------------
# Passport Seva India — 35×45 mm + 630×810 upload (May 2026 PDF + ICAO note)
# ---------------------------------------------------------------------------

# 35mm × 45mm @ 600 dpi ≈ 827 × 1063; @ 300 dpi = 413 × 531 (IDPhoto4You table)
_SEVA_PRINT_W = 827
_SEVA_PRINT_H = 1063

PASSPORT_SEVA_35X45 = PhotoSpec(
    id="passport-seva-35x45",
    title="Passport Seva India (35×45 mm / 630×810)",
    description=(
        "India Passport Seva physical 35×45 mm and portal upload 630×810 JPEG. "
        "White background; face ~80–85% of frame (Passport Seva / ICAO guidance)."
    ),
    # 35/25.4 ≈ 1.378 in, 45/25.4 ≈ 1.772 in
    photo_inches=(35.0 / 25.4, 45.0 / 25.4),
    photo_mm=(35.0, 45.0),
    # Face ~80–85% of photo height (Seva / passportindia ICAO note)
    head_height_min=0.72,
    head_height_max=0.88,
    head_height_target=0.82,
    # Eye band: less strict than 2×2 ICAO; keep head centered vertically
    eye_from_bottom_min=0.48,
    eye_from_bottom_max=0.62,
    eye_from_bottom_target=0.55,
    background_rgb=(255, 255, 255),
    print_dpi=600,
    print_px=_SEVA_PRINT_W,
    print_size_px=(_SEVA_PRINT_W, _SEVA_PRINT_H),
    upload_variants=(
        UploadVariant(
            name="Passport Seva upload (630×810)",
            filename_suffix="seva_630x810",
            width_px=630,
            height_px=810,
            min_kb=20,
            max_kb=250,
            exact_pixels=True,
            description=(
                "Exact 630×810 JPEG under 250 KB for Passport Seva portal "
                "(May 2026 photo upload instructions)."
            ),
        ),
        UploadVariant(
            name="Print-ready 300 dpi (413×531)",
            filename_suffix="print_413x531",
            width_px=413,
            height_px=531,
            min_kb=30,
            max_kb=500,
            exact_pixels=True,
            description="35×45 mm @ 300 dpi — matches IDPhoto4You 3.5×4.5 cm pixel table.",
        ),
    ),
    print_sheets=(
        PrintSheet(
            name="4×6 sheet (35×45 tiles)",
            filename_suffix="sheet_4x6",
            page_inches=(4.0, 6.0),
            cols=2,
            rows=3,
            description="Six 35×45 mm photos on 4×6 paper.",
        ),
        PrintSheet(
            name="3.5×5 sheet (35×45 tiles)",
            filename_suffix="sheet_3p5x5",
            page_inches=(3.5, 5.0),
            cols=2,
            rows=2,
            description="Four 35×45 mm photos on 3.5×5 / 9×13 cm paper.",
        ),
        PrintSheet(
            name="A4 sheet (35×45 tiles)",
            filename_suffix="sheet_a4",
            page_inches=(8.27, 11.69),
            cols=4,
            rows=5,
            description="Grid of 35×45 mm photos on A4 (print at 100% scale).",
        ),
    ),
    notes=(
        "Physical size: 35 mm wide × 45 mm high (not 2×2 inch).",
        "Portal: exactly 630×810 pixels, JPEG, under 250 KB.",
        "Capture: dark clothes, light/white BG, eyes open, natural expression, ears visible.",
        "Prefer source >2500×2500 px; take from ~1.5 m; no beauty filters.",
        "For VFS/US 2×2 appointments use the India 2×2 mode instead.",
    ),
    require_square=False,
)

DOCUMENT_TYPES: Dict[str, PhotoSpec] = {
    INDIAN_PASSPORT_2X2.id: INDIAN_PASSPORT_2X2,
    PASSPORT_SEVA_35X45.id: PASSPORT_SEVA_35X45,
    US_PASSPORT.id: US_PASSPORT,
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
            "photo_mm": list(s.photo_mm) if s.photo_mm else None,
            "require_square": s.require_square,
            "output_px": list(s.output_size()),
            "notes": list(s.notes),
        }
        for s in DOCUMENT_TYPES.values()
    ]
