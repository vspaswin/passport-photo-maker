"""Shared convert pipeline used by HTTP convert + batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from app.core.config import get_settings
from app.engine.process import process_photo
from app.engine.validate import PhotoValidationError
from app.jobs.store import JobStore, QuotaExceeded, Reservation


@dataclass
class ConvertSuccess:
    job_id: str
    doc_type: str
    metrics: dict
    validation: Optional[dict]
    warnings: list
    files: Dict[str, bytes]
    preview_jpeg: bytes
    usage: dict
    expires_at: Optional[float]


@dataclass
class ConvertFailure:
    error: str  # not_convertible | output_validation_failed | payment_required | failed
    message: str
    validation: Optional[dict]
    usage: dict
    http_status: int


def run_convert(
    store: JobStore,
    *,
    client_key: str,
    ip_key: str,
    image_bytes: bytes,
    doc_type: str = "indian-passport",
    child_mode: bool = False,
    scale_factor: float = 1.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = 0.0,
    charge: bool = True,
) -> Union[ConvertSuccess, ConvertFailure]:
    """
    Reserve quota → process → create owned job.
    Refunds reservation if processing fails after debit.
    Set charge=False for reframe (already paid convert).
    """
    settings = get_settings()
    reservation: Optional[Reservation] = None
    if charge:
        try:
            reservation, usage = store.reserve_convert(
                client_key,
                ip_key,
                free_daily=settings.free_daily_converts,
                cost=settings.convert_credit_cost,
                ip_free_daily=settings.ip_free_daily_converts,
            )
        except QuotaExceeded as exc:
            return ConvertFailure(
                error="payment_required",
                message=exc.message,
                validation=None,
                usage=exc.usage,
                http_status=402,
            )

    try:
        result = process_photo(
            image_bytes,
            doc_type=doc_type,
            remove_bg=True,
            strict=True,
            child_mode=child_mode,
            scale_factor=scale_factor,
            offset_x_frac=offset_x_frac,
            offset_y_frac=offset_y_frac,
        )
    except PhotoValidationError as exc:
        if reservation is not None:
            store.refund_reservation(reservation)
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
        return ConvertFailure(
            error=error,
            message=message,
            validation=exc.report.to_dict(),
            usage=store.get_usage(client_key, ip_key),
            http_status=422,
        )
    except Exception as exc:  # noqa: BLE001
        if reservation is not None:
            store.refund_reservation(reservation)
        return ConvertFailure(
            error="failed",
            message=f"Processing failed: {exc}",
            validation=None,
            usage=store.get_usage(client_key, ip_key),
            http_status=500,
        )

    job_id = store.create_job(
        owner_key=client_key,
        doc_type=result.doc_type,
        metrics=result.metrics,
        validation=result.validation or {},
        warnings=result.warnings,
        files=result.files,
        preview_jpeg=result.preview_jpeg,
        prepared_png=result.prepared_png,
        original_thumb=result.original_thumb,
        guide_preview=result.guide_preview_jpeg,
        face_dict=result.face_dict,
        child_mode=child_mode,
    )
    meta = store.get_meta(job_id, owner_key=client_key) or {}
    return ConvertSuccess(
        job_id=job_id,
        doc_type=result.doc_type,
        metrics=result.metrics,
        validation=result.validation,
        warnings=result.warnings,
        files=result.files,
        preview_jpeg=result.preview_jpeg,
        usage=store.get_usage(client_key, ip_key),
        expires_at=meta.get("expires_at"),
    )


def file_list_for_job(job_id: str, files: Dict[str, bytes]) -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "size_kb": round(len(blob) / 1024, 1),
            "download_url": f"/api/jobs/{job_id}/files/{name}",
        }
        for name, blob in sorted(files.items())
    ]


PRINT_TIP = {
    "letter_suffix": "_sheet_letter.jpg",
    "settings": [
        "Paper size: Letter (8.5×11)",
        "Paper type: Photo Glossy (Canon GP-701)",
        "Quality: High / Best (not Draft)",
        "Scale: 100% / Actual size (not Fit to Page)",
        "Load glossy side correctly; dry 1 minute before stacking",
    ],
}
