"""Local web app for passport / ID photo conversion."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app import __version__
from app.engine.process import process_photo
from app.engine.specs import list_document_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("passport-photo-maker")

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title="Passport Photo Maker",
    description="Local tool: convert any photo into Indian passport (print + digital).",
    version=__version__,
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# In-memory last result for simple download session (single-user local app)
_LAST: dict = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": __version__,
            "doc_types": list_document_types(),
        },
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "version": __version__}


@app.get("/api/document-types")
async def document_types():
    return list_document_types()


def _as_bool(value: Union[str, bool, None], default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    doc_type: str = Form("indian-passport"),
    remove_bg: str = Form("true"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        # Allow octet-stream from some browsers
        if file.content_type not in (None, "application/octet-stream"):
            raise HTTPException(400, "Please upload an image file (JPEG, PNG, etc.).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > 40 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 40 MB).")

    do_remove_bg = _as_bool(remove_bg, default=True)

    try:
        result = process_photo(data, doc_type=doc_type, remove_bg=do_remove_bg)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed")
        raise HTTPException(500, f"Processing failed: {exc}") from exc

    # Stash for downloads
    _LAST.clear()
    _LAST["files"] = result.files
    _LAST["doc_type"] = result.doc_type
    _LAST["metrics"] = result.metrics
    _LAST["warnings"] = result.warnings

    preview_b64 = base64.b64encode(result.preview_jpeg).decode("ascii")
    file_list = [
        {
            "name": name,
            "size_kb": round(len(blob) / 1024, 1),
            "download_url": f"/api/download/{name}",
        }
        for name, blob in sorted(result.files.items())
    ]

    return JSONResponse(
        {
            "ok": True,
            "doc_type": result.doc_type,
            "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
            "metrics": result.metrics,
            "warnings": result.warnings,
            "files": file_list,
        }
    )


@app.get("/api/download/{filename}")
async def download(filename: str):
    files = _LAST.get("files") or {}
    if filename not in files:
        raise HTTPException(
            404,
            "File not found. Convert a photo first, then download.",
        )
    media = "application/zip" if filename.endswith(".zip") else "image/jpeg"
    return Response(
        content=files[filename],
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def run():
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
