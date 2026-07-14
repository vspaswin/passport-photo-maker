# Passport Photo Maker

**Local web tool** that turns any portrait into **print-ready** and **digital upload** photos for official documents.

**v0.1** focuses on **Indian Passport / Visa / OCI** photos (VFS Global / MEA / ICAO-style 2×2").

- Fully **local** — photos stay on your machine  
- **Strict automated QC** — rejects photos that would likely fail passport review (no face, blur, dark glasses, multi-person, bad lighting, etc.). **Downloads are only created after all checks pass.**  
- **White background** removal (on-device)  
- Face-aware crop with official **head height** and **eye line** targets  
- Exports: single 2×2", 4×6 sheet, A4 sheet, portal JPEGs (≈10–100 KB)

## Quick start

```bash
# Python 3.9+
cd passport-photo-maker
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the local web app
python -m app.main
# open http://127.0.0.1:8765
```

First conversion may download a small ONNX model for background removal (`rembg` / u2net). That stays in your user cache.

## What you get

| File | Purpose |
|------|---------|
| `*_upload_600.jpg` | Website / portal upload (try this first) |
| `*_upload_350.jpg` | Smaller fallback if the portal rejects size |
| `*_PRINT_2x2_inch.jpg` | Single 2×2" print @ 600 dpi |
| `*_sheet_4x6.jpg` | 6 copies on 4×6 photo paper |
| `*_sheet_a4.jpg` | 12 copies on A4 (print at **100%** scale) |
| `*_master.jpg` | High-res square archive |
| `*_all.zip` | Everything above + README |

## Indian passport rules (summary)

Aligned with the VFS photo specification used for Passport / Visa / OCI:

- Colour photo **2 × 2 inches** (51 × 51 mm)
- Plain **white** background, no shadows
- Full frontal face, eyes open, natural expression
- Head height (hair → chin) about **1–1⅜ inches**
- Eye line about **1⅛–1⅓ inches** from the bottom of the photo
- Coloured clothing (not pure white)
- Prefer a true likeness — avoid heavy beautify filters

Official reference (USA VFS one-pager):  
[photo-specifiation.pdf](https://visa.vfsglobal.com/one-pager/india/united-states-of-america/passport-services/pdf/photo-specifiation.pdf)

## Privacy

- Server binds to **127.0.0.1** only  
- No accounts, no cloud upload in the default app  
- Processing uses local libraries (`Pillow`, `OpenCV`, `rembg`)

## Project layout

```
app/
  main.py           # FastAPI web app
  engine/
    specs.py        # Document type definitions (extensible)
    process.py      # Background, face crop, exports
  templates/        # UI
  static/
```

Adding another country later means a new `PhotoSpec` in `specs.py` and (if needed) small framing tweaks — the pipeline is shared.

## CLI-style one-shot (optional)

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from app.engine import process_photo

data = Path("my-photo.jpg").read_bytes()
result = process_photo(data, doc_type="indian-passport")
out = Path("output"); out.mkdir(exist_ok=True)
for name, blob in result.files.items():
    (out / name).write_bytes(blob)
print("Wrote", len(result.files), "files to", out)
print("Warnings:", result.warnings)
PY
```

## Development

```bash
pip install -r requirements.txt
python -m app.main
# API docs: http://127.0.0.1:8765/docs
```

## Strict validation

Conversion is blocked (HTTP 422, no download files) when automated checks fail, including:

- No face / multiple people  
- Face too small, too close, or clipped  
- Blurry or poorly lit face  
- Eyes not clearly open / dark glasses / head tilt  
- Not a colour photo / likely pure white clothing  
- Final background not clean white / final geometry off  

When validation passes, the ZIP includes `VALIDATION_PASSED.txt` with check details.

## Disclaimer

Automated QC greatly reduces reject risk but is **not a legal guarantee** of acceptance by VFS / MEA / consulate. Always follow the latest official photo rules for your channel.

## License

MIT — see [LICENSE](LICENSE).
