"""Local web app for passport / ID photo conversion."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app import __version__
from app.engine.process import process_photo
from app.engine.specs import list_document_types
from app.engine.validate import PhotoValidationError

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


def _u2net_model_ready() -> bool:
    """True if rembg's default u2net weights are already cached locally."""
    path = Path.home() / ".u2net" / "u2net.onnx"
    return path.is_file() and path.stat().st_size > 1_000_000


@app.get("/api/health")
async def health():
    return {"ok": True, "version": __version__, "model_ready": _u2net_model_ready()}


@app.get("/api/status")
async def status():
    ready = _u2net_model_ready()
    return {
        "ok": True,
        "version": __version__,
        "model_ready": ready,
        "model_name": "u2net",
        "model_path": str(Path.home() / ".u2net" / "u2net.onnx"),
        "strict_validation": True,
    }


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
    strict: str = Form("true"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        if file.content_type not in (None, "application/octet-stream"):
            raise HTTPException(400, "Please upload an image file (JPEG, PNG, etc.).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > 40 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 40 MB).")

    # Always require white-bg pipeline for submittable Indian passport photos
    do_remove_bg = True
    do_strict = _as_bool(strict, default=True)

    try:
        result = process_photo(
            data,
            doc_type=doc_type,
            remove_bg=do_remove_bg,
            strict=do_strict,
        )
    except PhotoValidationError as exc:
        logger.info("Validation rejected photo: %s", exc.message)
        # Clear any previous successful downloads so user cannot re-download stale files
        _LAST.clear()
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "validation_failed",
                "message": exc.message,
                "validation": exc.report.to_dict(),
                "files": [],
            },
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed")
        raise HTTPException(500, f"Processing failed: {exc}") from exc

    _LAST.clear()
    _LAST["files"] = result.files
    _LAST["doc_type"] = result.doc_type
    _LAST["metrics"] = result.metrics
    _LAST["warnings"] = result.warnings
    _LAST["validation"] = result.validation

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
            "validation": result.validation,
            "files": file_list,
            "submittable": True,
        }
    )


@app.get("/api/download/{filename}")
async def download(filename: str):
    files = _LAST.get("files") or {}
    if filename not in files:
        raise HTTPException(
            404,
            "File not found. Convert a photo that passes validation first.",
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
