"""Stripe Checkout credit packs (optional — app runs free without keys)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.jobs.store import get_store

logger = logging.getLogger(__name__)


def pricing_public() -> Dict[str, Any]:
    s = get_settings()
    return {
        "stripe_enabled": s.stripe_enabled,
        "free_daily_checks": s.free_daily_checks,
        "free_daily_converts": s.free_daily_converts,
        "ip_free_daily_converts": s.ip_free_daily_converts,
        "convert_credit_cost": s.convert_credit_cost,
        "packs": [
            {
                "id": "starter",
                "credits": s.credit_pack_starter,
                "price_id": s.stripe_price_starter,
                "label": f"{s.credit_pack_starter} conversions",
            },
            {
                "id": "pro",
                "credits": s.credit_pack_pro,
                "price_id": s.stripe_price_pro,
                "label": f"{s.credit_pack_pro} conversions",
            },
        ],
    }


def create_checkout_session(
    *,
    client_key: str,
    pack_id: str,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
) -> Dict[str, Any]:
    s = get_settings()
    if not s.stripe_enabled:
        raise RuntimeError("Stripe is not configured. Set STRIPE_SECRET_KEY.")

    pack_map = {
        "starter": (s.stripe_price_starter, s.credit_pack_starter),
        "pro": (s.stripe_price_pro, s.credit_pack_pro),
    }
    if pack_id not in pack_map:
        raise ValueError(f"Unknown pack: {pack_id}")
    price_id, credits = pack_map[pack_id]
    if not price_id:
        raise RuntimeError(
            f"Stripe price id not set for pack '{pack_id}'. "
            f"Set STRIPE_PRICE_{pack_id.upper()}."
        )

    import stripe

    stripe.api_key = s.stripe_secret_key
    success = success_url or f"{s.app_url}/?billing=success"
    cancel = cancel_url or f"{s.app_url}/?billing=cancel"

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success + "&session_id={CHECKOUT_SESSION_ID}",
        cancel_url=cancel,
        metadata={"client_key": client_key, "credits": str(credits), "pack": pack_id},
        client_reference_id=client_key,
    )
    # Persist before redirect so webhook can fulfill without double-credit path
    get_store().save_stripe_session(session.id, client_key, credits)
    return {"checkout_url": session.url, "session_id": session.id, "credits": credits}


def handle_webhook(payload: bytes, sig_header: str) -> Dict[str, Any]:
    s = get_settings()
    if not s.stripe_enabled or not s.stripe_webhook_secret:
        raise RuntimeError("Stripe webhook not configured")

    import stripe

    stripe.api_key = s.stripe_secret_key
    event = stripe.Webhook.construct_event(
        payload, sig_header, s.stripe_webhook_secret
    )

    if event["type"] != "checkout.session.completed":
        return {"ok": True, "ignored": event["type"]}

    session = event["data"]["object"]
    session_id = session["id"]
    meta = session.get("metadata") or {}
    client_key = meta.get("client_key") or session.get("client_reference_id")
    credits = int(meta.get("credits") or 0)

    store = get_store()
    # Ensure session row exists (webhook may arrive before our save in rare races)
    if client_key and credits > 0:
        store.save_stripe_session(session_id, client_key, credits)

    credited, balance, status = store.fulfill_stripe_session(session_id)
    logger.info(
        "Stripe fulfill session=%s status=%s credited=%s balance=%s",
        session_id,
        status,
        credited,
        balance,
    )
    return {
        "ok": True,
        "fulfilled": credited,
        "status": status,
        "balance": balance,
    }
