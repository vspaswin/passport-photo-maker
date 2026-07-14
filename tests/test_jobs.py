"""Job store and freemium usage."""

import tempfile
from pathlib import Path

from app.jobs.store import JobStore


def test_job_create_and_download():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = JobStore(db_path=root / "t.db", jobs_dir=root / "jobs")
        jid = store.create_job(
            doc_type="indian-passport",
            metrics={"x": 1},
            validation={"passed": True},
            warnings=[],
            files={"a.jpg": b"\xff\xd8\xffabc"},
            preview_jpeg=b"\xff\xd8\xffpreview",
        )
        meta = store.get_meta(jid)
        assert meta is not None
        assert "a.jpg" in meta["files"]
        blob = store.get_file(jid, "a.jpg")
        assert blob.startswith(b"\xff\xd8\xff")


def test_free_quota_and_credits():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = JobStore(db_path=root / "t.db", jobs_dir=root / "jobs")
        key = "client-test"
        assert store.can_check(key, free_daily=2)[0] is True
        store.record_check(key)
        store.record_check(key)
        assert store.can_check(key, free_daily=2)[0] is False
        store.add_credits(key, 5)
        assert store.can_check(key, free_daily=2)[0] is True
        ok, mode, _ = store.can_convert(key, free_daily=0, cost=1)
        assert ok and mode == "credits"
        store.consume_convert(key, free_daily=0, cost=1)
        assert store.get_usage(key)["credit_balance"] == 4
