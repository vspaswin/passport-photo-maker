"""Production FastAPI app: validate, convert, jobs, freemium + Stripe."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from pathlib import Path
from typing import List, Optional

from fastapi import (
    Cookie,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.billing import stripe_billing
from app.core.config import get_settings
from app.engine.process import load_image, process_photo
from app.engine.specs import get_spec, list_document_types
from app.engine.validate import PhotoValidationError, assess_photo
from app.jobs.store import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("passport-photo-maker")

settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title=settings.app_name,
    description=(
        "Production passport photo service: validate, convert, print sheets "
        "(Letter/A4/4×6), freemium credits + optional Stripe."
    ),
    version=__version__,
)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

CLIENT_COOKIE = "ppm_client"


def _client_key(
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
) -> str:
    if ppm_client and len(ppm_client) >= 16:
        return ppm_client
    # Stable-ish anonymous id
    raw = secrets.token_hex(16)
    response.set_cookie(
        CLIENT_COOKIE,
        raw,
        max_age=365 * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
    )
    return raw


def _ip_fallback(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = (forwarded.split(",")[0].strip() if forwarded else None) or (
        request.client.host if request.client else "unknown"
    )
    return hashlib.sha256(f"{ip}:{settings.secret_key}".encode()).hexdigest()[:32]


async def _read_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        if file.content_type not in (None, "application/octet-stream"):
            raise HTTPException(400, "Please upload an image file (JPEG, PNG, etc.).")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    max_b = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_b:
        raise HTTPException(400, f"File too large (max {settings.max_upload_mb} MB).")
    return data


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": __version__,
            "doc_types": list_document_types(),
            "pricing": stripe_billing.pricing_public(),
            "app_name": settings.app_name,
        },
    )


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "version": __version__,
        "env": settings.app_env,
        "stripe_enabled": settings.stripe_enabled,
        "rembg_model": settings.rembg_model,
    }


@app.get("/api/status")
async def status(
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    usage = get_store().get_usage(key)
    model_path = Path.home() / ".u2net" / f"{settings.rembg_model}.onnx"
    # human_seg stores as u2net_human_seg.onnx
    ready = model_path.is_file() or (Path.home() / ".u2net" / "u2net.onnx").is_file()
    return {
        "ok": True,
        "version": __version__,
        "model_ready": ready,
        "model_name": settings.rembg_model,
        "strict_validation": True,
        "modes": ["validate", "convert", "batch"],
        "usage": usage,
        "pricing": stripe_billing.pricing_public(),
        "client_key_suffix": key[-6:],
    }


@app.get("/api/document-types")
async def document_types():
    return list_document_types()


@app.get("/api/pricing")
async def pricing():
    return stripe_billing.pricing_public()


@app.post("/api/validate")
async def validate_only(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    doc_type: str = Form("indian-passport"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    store = get_store()
    ok, usage = store.can_check(key, settings.free_daily_checks)
    if not ok:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": "check_quota",
                "message": "Daily free checks used. Buy credits for unlimited checks.",
                "usage": usage,
                "pricing": stripe_billing.pricing_public(),
            },
        )

    data = await _read_upload(file)
    try:
        spec = get_spec(doc_type)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        im = load_image(data)
        assessment = assess_photo(im, spec)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Validate failed")
        raise HTTPException(500, f"Validation failed: {exc}") from exc

    usage = store.record_check(key)
    return JSONResponse(
        {
            "ok": True,
            "mode": "validate",
            "doc_type": doc_type,
            "usage": usage,
            "disclaimer": (
                "Automated QC only — not official government approval. "
                "Final acceptance is decided by VFS / passport authority."
            ),
            **assessment,
        }
    )


@app.post("/api/convert")
async def convert(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    doc_type: str = Form("indian-passport"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    store = get_store()
    allowed, reason, usage = store.can_convert(
        key, settings.free_daily_converts, settings.convert_credit_cost
    )
    if not allowed:
        return JSONResponse(
            status_code=402,
            content={
                "ok": False,
                "error": "payment_required",
                "message": reason,
                "usage": usage,
                "pricing": stripe_billing.pricing_public(),
            },
        )

    data = await _read_upload(file)

    try:
        result = process_photo(data, doc_type=doc_type, remove_bg=True, strict=True)
    except PhotoValidationError as exc:
        logger.info("Convert rejected: %s", exc.message)
        stage = exc.report.stage
        error = (
            "not_convertible"
            if stage in ("source_convertible", "source", "source_as_is")
            else "output_validation_failed"
        )
        message = exc.message
        if error == "output_validation_failed":
            message = (
                "Converted, but the final photo still failed passport QC. " + message
            )
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "mode": "convert",
                "error": error,
                "message": message,
                "validation": exc.report.to_dict(),
                "usage": store.get_usage(key),
                "files": [],
            },
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed")
        raise HTTPException(500, f"Processing failed: {exc}") from exc

    # Charge only after successful final QC
    try:
        usage = store.consume_convert(
            key, settings.free_daily_converts, settings.convert_credit_cost
        )
    except RuntimeError:
        usage = store.get_usage(key)

    job_id = store.create_job(
        doc_type=result.doc_type,
        metrics=result.metrics,
        validation=result.validation or {},
        warnings=result.warnings,
        files=result.files,
        preview_jpeg=result.preview_jpeg,
    )
    meta = store.get_meta(job_id) or {}
    file_list = [
        {
            "name": name,
            "size_kb": round(len(result.files[name]) / 1024, 1),
            "download_url": f"/api/jobs/{job_id}/files/{name}",
        }
        for name in sorted(result.files.keys())
    ]
    preview_b64 = base64.b64encode(result.preview_jpeg).decode("ascii")

    return JSONResponse(
        {
            "ok": True,
            "mode": "convert",
            "job_id": job_id,
            "expires_at": meta.get("expires_at"),
            "doc_type": result.doc_type,
            "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
            "metrics": result.metrics,
            "warnings": result.warnings,
            "validation": result.validation,
            "files": file_list,
            "submittable": True,
            "usage": usage,
            "disclaimer": (
                "Automated QC passed — not official government approval. "
                "Verify likeness and print quality before submitting."
            ),
            "print_tip": {
                "letter_file": f"{result.doc_type}_sheet_letter.jpg",
                "settings": [
                    "Paper size: Letter (8.5×11)",
                    "Paper type: Photo Glossy (Canon GP-701)",
                    "Quality: High / Best (not Draft)",
                    "Scale: 100% / Actual size (not Fit to Page)",
                    "Load glossy side correctly; dry 1 minute before stacking",
                ],
            },
        }
    )


@app.post("/api/batch")
async def batch_convert(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(...),
    doc_type: str = Form("indian-passport"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    """Convert multiple images; each successful convert costs a free slot or credit."""
    key = _client_key(request, response, ppm_client)
    if len(files) > 20:
        raise HTTPException(400, "Max 20 files per batch.")

    store = get_store()
    results = []
    for f in files:
        allowed, reason, usage = store.can_convert(
            key, settings.free_daily_converts, settings.convert_credit_cost
        )
        entry = {"filename": f.filename, "ok": False}
        if not allowed:
            entry.update({"error": "payment_required", "message": reason, "usage": usage})
            results.append(entry)
            continue
        try:
            data = await f.read()
            if not data:
                entry["message"] = "Empty file"
                results.append(entry)
                continue
            result = process_photo(data, doc_type=doc_type, remove_bg=True, strict=True)
            usage = store.consume_convert(
                key, settings.free_daily_converts, settings.convert_credit_cost
            )
            job_id = store.create_job(
                doc_type=result.doc_type,
                metrics=result.metrics,
                validation=result.validation or {},
                warnings=result.warnings,
                files=result.files,
                preview_jpeg=result.preview_jpeg,
            )
            entry.update(
                {
                    "ok": True,
                    "job_id": job_id,
                    "files": [
                        {
                            "name": n,
                            "download_url": f"/api/jobs/{job_id}/files/{n}",
                        }
                        for n in sorted(result.files.keys())
                    ],
                    "usage": usage,
                }
            )
        except PhotoValidationError as exc:
            entry.update(
                {
                    "error": "validation_failed",
                    "message": exc.message,
                    "validation": exc.report.to_dict(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            entry.update({"error": "failed", "message": str(exc)})
        results.append(entry)

    return {
        "ok": True,
        "mode": "batch",
        "count": len(results),
        "passed": sum(1 for r in results if r.get("ok")),
        "results": results,
        "usage": store.get_usage(key),
    }


@app.get("/api/jobs/{job_id}")
async def job_meta(job_id: str):
    meta = get_store().get_meta(job_id)
    if not meta:
        raise HTTPException(404, "Job not found or expired.")
    return {
        "ok": True,
        "job_id": job_id,
        "doc_type": meta["doc_type"],
        "expires_at": meta["expires_at"],
        "metrics": meta["metrics"],
        "validation": meta["validation"],
        "warnings": meta["warnings"],
        "files": [
            {"name": n, "download_url": f"/api/jobs/{job_id}/files/{n}"}
            for n in meta["files"]
        ],
    }


@app.get("/api/jobs/{job_id}/files/{filename}")
async def job_file(job_id: str, filename: str):
    blob = get_store().get_file(job_id, filename)
    if blob is None:
        raise HTTPException(404, "File not found or job expired.")
    media = "application/zip" if filename.endswith(".zip") else "image/jpeg"
    return Response(
        content=blob,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{Path(filename).name}"'},
    )


@app.post("/api/billing/checkout")
async def billing_checkout(
    request: Request,
    response: Response,
    pack_id: str = Form("starter"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    try:
        session = stripe_billing.create_checkout_session(client_key=key, pack_id=pack_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return session


@app.post("/api/billing/webhook")
async def billing_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(default=None, alias="Stripe-Signature"),
):
    payload = await request.body()
    if not stripe_signature:
        raise HTTPException(400, "Missing Stripe-Signature")
    try:
        result = stripe_billing.handle_webhook(payload, stripe_signature)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Webhook error")
        raise HTTPException(400, str(exc)) from exc
    return result


@app.get("/api/me")
async def me(
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    return {"ok": True, "usage": get_store().get_usage(key), "pricing": stripe_billing.pricing_public()}


def run():
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.host if not s.is_production else "0.0.0.0",
        port=s.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
