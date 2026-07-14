"""Job store, atomic ledger, Stripe idempotency."""

import tempfile
from pathlib import Path

import pytest

from app.jobs.store import JobStore, QuotaExceeded


@pytest.fixture()
def store(tmp_path: Path):
    return JobStore(db_path=tmp_path / "t.db", jobs_dir=tmp_path / "jobs")


def test_job_create_owned_download(store: JobStore):
    jid = store.create_job(
        owner_key="owner-a",
        doc_type="indian-passport",
        metrics={"x": 1},
        validation={"passed": True},
        warnings=[],
        files={"a.jpg": b"\xff\xd8\xffabc"},
        preview_jpeg=b"\xff\xd8\xffpreview",
    )
    assert store.get_meta(jid, owner_key="owner-a") is not None
    assert store.get_meta(jid, owner_key="owner-b") is None
    assert store.get_file(jid, "a.jpg", owner_key="owner-a") is not None
    assert store.get_file(jid, "a.jpg", owner_key="other") is None


def test_reserve_free_and_refund(store: JobStore):
    r, usage = store.reserve_convert(
        "c1", "ip1", free_daily=2, cost=1, ip_free_daily=10
    )
    assert r.mode == "free"
    assert usage["converts"] == 1
    store.refund_reservation(r)
    usage2 = store.get_usage("c1", "ip1")
    assert usage2["converts"] == 0


def test_reserve_credits_atomic(store: JobStore):
    store.add_credits("c2", 2)
    r, usage = store.reserve_convert(
        "c2", "ip2", free_daily=0, cost=1, ip_free_daily=0
    )
    assert r.mode == "credits"
    assert usage["credit_balance"] == 1
    r2, usage2 = store.reserve_convert(
        "c2", "ip2", free_daily=0, cost=1, ip_free_daily=0
    )
    assert usage2["credit_balance"] == 0
    with pytest.raises(QuotaExceeded):
        store.reserve_convert("c2", "ip2", free_daily=0, cost=1, ip_free_daily=0)


def test_free_quota_client_and_ip(store: JobStore):
    store.reserve_convert("a", "ipx", free_daily=1, cost=1, ip_free_daily=2)
    with pytest.raises(QuotaExceeded):
        store.reserve_convert("a", "ipx", free_daily=1, cost=1, ip_free_daily=2)
    # new cookie same IP still limited by IP after 2 total
    store.reserve_convert("b", "ipx", free_daily=1, cost=1, ip_free_daily=2)
    with pytest.raises(QuotaExceeded):
        store.reserve_convert("c", "ipx", free_daily=5, cost=1, ip_free_daily=2)


def test_check_quota(store: JobStore):
    store.try_record_check("u", "ip", free_daily=2, ip_free_daily=10)
    store.try_record_check("u", "ip", free_daily=2, ip_free_daily=10)
    with pytest.raises(QuotaExceeded):
        store.try_record_check("u", "ip", free_daily=2, ip_free_daily=10)
    store.add_credits("u", 1)
    # credits unlock further checks
    store.try_record_check("u", "ip", free_daily=2, ip_free_daily=10)


def test_stripe_fulfill_idempotent(store: JobStore):
    store.save_stripe_session("sess_1", "buyer", 10)
    ok1, bal1, st1 = store.fulfill_stripe_session("sess_1")
    assert ok1 is True and st1 == "credited" and bal1 == 10
    ok2, bal2, st2 = store.fulfill_stripe_session("sess_1")
    assert ok2 is False and st2 == "already_fulfilled" and bal2 == 10
    # no double credit
    assert store.get_usage("buyer")["credit_balance"] == 10


def test_stripe_unknown_session(store: JobStore):
    ok, bal, st = store.fulfill_stripe_session("missing")
    assert ok is False and st == "unknown_session"
