"""Production FastAPI app: validate, convert, jobs, freemium + Stripe."""

from __future__ import annotations

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
from app.engine.process import load_image, reframe_photo
from app.engine.specs import get_spec, list_document_types
from app.engine.validate import PhotoValidationError, assess_photo
from app.jobs.store import QuotaExceeded, get_store
from app.services.convert_service import (
    PRINT_TIP,
    ConvertFailure,
    ConvertSuccess,
    file_list_for_job,
    run_convert,
)

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


def _request_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _ip_key(request: Request) -> str:
    ip = _request_ip(request)
    return hashlib.sha256(f"ip:{ip}:{settings.secret_key}".encode()).hexdigest()[:32]


def _client_key(
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
) -> str:
    if ppm_client and len(ppm_client) >= 16 and ppm_client.isalnum():
        return ppm_client
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


def _success_payload(result: ConvertSuccess) -> dict:
    import base64

    files = file_list_for_job(result.job_id, result.files)
    preview_b64 = base64.b64encode(result.preview_jpeg).decode("ascii")
    return {
        "ok": True,
        "mode": "convert",
        "job_id": result.job_id,
        "expires_at": result.expires_at,
        "doc_type": result.doc_type,
        "preview_url": f"/api/jobs/{result.job_id}/files/preview.jpg",
        "guide_url": f"/api/jobs/{result.job_id}/files/guide_preview.jpg",
        "original_url": f"/api/jobs/{result.job_id}/files/original_thumb.jpg",
        "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
        "metrics": result.metrics,
        "warnings": result.warnings,
        "validation": result.validation,
        "files": files,
        "submittable": True,
        "finetune": {
            "scale_factor": result.metrics.get("scale_factor", 1.0),
            "offset_x_frac": result.metrics.get("offset_x_frac", 0.0),
            "offset_y_frac": result.metrics.get("offset_y_frac", 0.0),
        },
        "usage": result.usage,
        "disclaimer": (
            "Automated QC passed — not official government approval. "
            "Verify likeness and print quality before submitting."
        ),
        "print_tip": {
            "letter_file": f"{result.doc_type}{PRINT_TIP['letter_suffix']}",
            "settings": PRINT_TIP["settings"],
        },
    }


def _failure_response(fail: ConvertFailure) -> JSONResponse:
    body = {
        "ok": False,
        "mode": "convert",
        "error": fail.error,
        "message": fail.message,
        "usage": fail.usage,
        "files": [],
        "pricing": stripe_billing.pricing_public(),
    }
    if fail.validation is not None:
        body["validation"] = fail.validation
    return JSONResponse(status_code=fail.http_status, content=body)


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
    ipk = _ip_key(request)
    usage = get_store().get_usage(key, ipk)
    model_path = Path.home() / ".u2net" / f"{settings.rembg_model}.onnx"
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
    ipk = _ip_key(request)
    store = get_store()
    try:
        usage = store.try_record_check(
            key,
            ipk,
            free_daily=settings.free_daily_checks,
            ip_free_daily=settings.ip_free_daily_checks,
        )
    except QuotaExceeded as exc:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": "check_quota",
                "message": exc.message,
                "usage": exc.usage,
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


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/convert")
async def convert(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    doc_type: str = Form("indian-passport"),
    child_mode: str = Form("false"),
    scale_factor: str = Form("1.0"),
    offset_x_frac: str = Form("0"),
    offset_y_frac: str = Form("0"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    ipk = _ip_key(request)
    data = await _read_upload(file)

    try:
        sf = float(scale_factor)
        ox = float(offset_x_frac)
        oy = float(offset_y_frac)
    except ValueError:
        raise HTTPException(400, "Invalid fine-tune parameters") from None

    outcome = run_convert(
        get_store(),
        client_key=key,
        ip_key=ipk,
        image_bytes=data,
        doc_type=doc_type,
        child_mode=_as_bool(child_mode),
        scale_factor=sf,
        offset_x_frac=ox,
        offset_y_frac=oy,
    )
    if isinstance(outcome, ConvertFailure):
        return _failure_response(outcome)
    return JSONResponse(_success_payload(outcome))


@app.post("/api/jobs/{job_id}/reframe")
async def reframe_job(
    job_id: str,
    request: Request,
    response: Response,
    scale_factor: str = Form("1.0"),
    offset_x_frac: str = Form("0"),
    offset_y_frac: str = Form("0"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    """Fine-tune framing without re-running background removal or charging credits."""
    key = _client_key(request, response, ppm_client)
    store = get_store()
    inter = store.get_intermediate(job_id, key)
    if not inter:
        raise HTTPException(
            404, "Job intermediate not found (expired, not owned, or old job)."
        )
    try:
        sf = float(scale_factor)
        ox = float(offset_x_frac)
        oy = float(offset_y_frac)
    except ValueError:
        raise HTTPException(400, "Invalid fine-tune parameters") from None

    try:
        result = reframe_photo(
            inter["prepared_png"],
            inter["face_dict"],
            inter["doc_type"],
            scale_factor=sf,
            offset_x_frac=ox,
            offset_y_frac=oy,
            child_mode=bool(inter.get("child_mode")),
            strict=True,
        )
    except PhotoValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "output_validation_failed",
                "message": exc.message,
                "validation": exc.report.to_dict(),
            },
        )

    store.replace_job_files(
        job_id,
        key,
        files=result.files,
        preview_jpeg=result.preview_jpeg,
        guide_preview=result.guide_preview_jpeg,
        metrics=result.metrics,
        validation=result.validation or {},
    )
    # Build a ConvertSuccess-shaped payload
    import base64

    preview_b64 = base64.b64encode(result.preview_jpeg).decode("ascii")
    return {
        "ok": True,
        "mode": "reframe",
        "job_id": job_id,
        "doc_type": result.doc_type,
        "preview_url": f"/api/jobs/{job_id}/files/preview.jpg",
        "guide_url": f"/api/jobs/{job_id}/files/guide_preview.jpg",
        "original_url": f"/api/jobs/{job_id}/files/original_thumb.jpg",
        "preview_data_url": f"data:image/jpeg;base64,{preview_b64}",
        "metrics": result.metrics,
        "validation": result.validation,
        "files": file_list_for_job(job_id, result.files),
        "submittable": True,
        "finetune": {
            "scale_factor": result.metrics.get("scale_factor", 1.0),
            "offset_x_frac": result.metrics.get("offset_x_frac", 0.0),
            "offset_y_frac": result.metrics.get("offset_y_frac", 0.0),
        },
        "disclaimer": (
            "Reframed with automated QC. Not official government approval."
        ),
        "print_tip": {
            "letter_file": f"{result.doc_type}{PRINT_TIP['letter_suffix']}",
            "settings": PRINT_TIP["settings"],
        },
    }


@app.post("/api/batch")
async def batch_convert(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(...),
    doc_type: str = Form("indian-passport"),
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    ipk = _ip_key(request)
    if len(files) > 20:
        raise HTTPException(400, "Max 20 files per batch.")

    store = get_store()
    results = []
    for f in files:
        entry: dict = {"filename": f.filename, "ok": False}
        try:
            data = await _read_upload(f)
        except HTTPException as exc:
            entry["message"] = exc.detail if isinstance(exc.detail, str) else "Invalid file"
            results.append(entry)
            continue

        outcome = run_convert(
            store,
            client_key=key,
            ip_key=ipk,
            image_bytes=data,
            doc_type=doc_type,
        )
        if isinstance(outcome, ConvertFailure):
            entry.update(
                {
                    "error": outcome.error,
                    "message": outcome.message,
                    "validation": outcome.validation,
                    "usage": outcome.usage,
                }
            )
        else:
            entry.update(
                {
                    "ok": True,
                    "job_id": outcome.job_id,
                    "files": file_list_for_job(outcome.job_id, outcome.files),
                    "usage": outcome.usage,
                }
            )
        results.append(entry)

    return {
        "ok": True,
        "mode": "batch",
        "count": len(results),
        "passed": sum(1 for r in results if r.get("ok")),
        "results": results,
        "usage": store.get_usage(key, ipk),
    }


@app.get("/api/jobs/{job_id}")
async def job_meta(
    job_id: str,
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    meta = get_store().get_meta(job_id, owner_key=key)
    if not meta:
        raise HTTPException(404, "Job not found, expired, or not owned by this client.")
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
        "preview_url": f"/api/jobs/{job_id}/files/preview.jpg",
    }


@app.get("/api/jobs/{job_id}/files/{filename}")
async def job_file(
    job_id: str,
    filename: str,
    request: Request,
    response: Response,
    ppm_client: Optional[str] = Cookie(default=None, alias=CLIENT_COOKIE),
):
    key = _client_key(request, response, ppm_client)
    safe_name = Path(filename).name
    if safe_name in {"prepared.png", "face.json", "meta_extra.json"}:
        raise HTTPException(404, "File not found.")
    blob = get_store().get_file(job_id, filename, owner_key=key)
    if blob is None:
        raise HTTPException(404, "File not found, expired, or not owned by this client.")
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
    ipk = _ip_key(request)
    return {
        "ok": True,
        "usage": get_store().get_usage(key, ipk),
        "pricing": stripe_billing.pricing_public(),
    }


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
