"""SQLite-backed job store for convert outputs (multi-user safe)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import get_settings


@dataclass
class Job:
    id: str
    created_at: float
    expires_at: float
    doc_type: str
    metrics: dict
    validation: dict
    warnings: list
    files: Dict[str, bytes]  # name -> bytes (loaded on demand from disk)


class JobStore:
    def __init__(self, db_path: Optional[Path] = None, jobs_dir: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self.jobs_dir = jobs_dir or settings.jobs_dir
        self.ttl = settings.job_ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    doc_type TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    file_index_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage (
                    client_key TEXT NOT NULL,
                    day TEXT NOT NULL,
                    checks INTEGER NOT NULL DEFAULT 0,
                    converts INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (client_key, day)
                );
                CREATE TABLE IF NOT EXISTS credits (
                    client_key TEXT PRIMARY KEY,
                    balance INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stripe_sessions (
                    session_id TEXT PRIMARY KEY,
                    client_key TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    fulfilled INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                """
            )

    def create_job(
        self,
        *,
        doc_type: str,
        metrics: dict,
        validation: dict,
        warnings: list,
        files: Dict[str, bytes],
        preview_jpeg: bytes,
    ) -> str:
        self.cleanup_expired()
        job_id = uuid.uuid4().hex
        now = time.time()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        names = []
        for name, blob in files.items():
            safe = Path(name).name
            (job_dir / safe).write_bytes(blob)
            names.append(safe)
        (job_dir / "preview.jpg").write_bytes(preview_jpeg)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, created_at, expires_at, doc_type, metrics_json,
                                  validation_json, warnings_json, file_index_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now + self.ttl,
                    doc_type,
                    json.dumps(metrics),
                    json.dumps(validation or {}),
                    json.dumps(warnings or []),
                    json.dumps(names),
                ),
            )
        return job_id

    def get_meta(self, job_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        if row["expires_at"] < time.time():
            self.delete_job(job_id)
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "doc_type": row["doc_type"],
            "metrics": json.loads(row["metrics_json"]),
            "validation": json.loads(row["validation_json"]),
            "warnings": json.loads(row["warnings_json"]),
            "files": json.loads(row["file_index_json"]),
        }

    def get_file(self, job_id: str, filename: str) -> Optional[bytes]:
        meta = self.get_meta(job_id)
        if not meta:
            return None
        safe = Path(filename).name
        if safe not in meta["files"] and safe != "preview.jpg":
            return None
        path = self.jobs_dir / job_id / safe
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete_job(self, job_id: str) -> None:
        import shutil

        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE expires_at < ?", (now,)
            ).fetchall()
            ids = [r["id"] for r in rows]
        for jid in ids:
            self.delete_job(jid)
        return len(ids)

    # --- usage / credits ---

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def get_usage(self, client_key: str) -> dict:
        day = self._today()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checks, converts FROM usage WHERE client_key = ? AND day = ?",
                (client_key, day),
            ).fetchone()
            cred = conn.execute(
                "SELECT balance FROM credits WHERE client_key = ?", (client_key,)
            ).fetchone()
        return {
            "day": day,
            "checks": int(row["checks"]) if row else 0,
            "converts": int(row["converts"]) if row else 0,
            "credit_balance": int(cred["balance"]) if cred else 0,
        }

    def record_check(self, client_key: str) -> dict:
        day = self._today()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO usage (client_key, day, checks, converts) VALUES (?, ?, 1, 0)
                ON CONFLICT(client_key, day) DO UPDATE SET checks = checks + 1
                """,
                (client_key, day),
            )
        return self.get_usage(client_key)

    def can_convert(self, client_key: str, free_daily: int, cost: int) -> tuple[bool, str, dict]:
        usage = self.get_usage(client_key)
        if usage["credit_balance"] >= cost:
            return True, "credits", usage
        if usage["converts"] < free_daily:
            return True, "free", usage
        return (
            False,
            "You have used today's free conversions. Buy credits to continue.",
            usage,
        )

    def consume_convert(self, client_key: str, free_daily: int, cost: int) -> dict:
        ok, mode, usage = self.can_convert(client_key, free_daily, cost)
        if not ok:
            raise RuntimeError(usage if isinstance(usage, str) else "quota_exceeded")
        day = self._today()
        with self._conn() as conn:
            if mode == "credits":
                conn.execute(
                    "UPDATE credits SET balance = balance - ?, updated_at = ? WHERE client_key = ?",
                    (cost, time.time(), client_key),
                )
            conn.execute(
                """
                INSERT INTO usage (client_key, day, checks, converts) VALUES (?, ?, 0, 1)
                ON CONFLICT(client_key, day) DO UPDATE SET converts = converts + 1
                """,
                (client_key, day),
            )
        return self.get_usage(client_key)

    def can_check(self, client_key: str, free_daily: int) -> tuple[bool, dict]:
        usage = self.get_usage(client_key)
        # Paid users with credits get unlimited checks; free limited
        if usage["credit_balance"] > 0 or usage["checks"] < free_daily:
            return True, usage
        return False, usage

    def add_credits(self, client_key: str, amount: int) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO credits (client_key, balance, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(client_key) DO UPDATE SET
                    balance = balance + excluded.balance,
                    updated_at = excluded.updated_at
                """,
                (client_key, amount, time.time()),
            )
            row = conn.execute(
                "SELECT balance FROM credits WHERE client_key = ?", (client_key,)
            ).fetchone()
        return int(row["balance"])

    def save_stripe_session(self, session_id: str, client_key: str, credits: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stripe_sessions
                (session_id, client_key, credits, fulfilled, created_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (session_id, client_key, credits, time.time()),
            )

    def fulfill_stripe_session(self, session_id: str) -> Optional[int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM stripe_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row or row["fulfilled"]:
                return None
            conn.execute(
                "UPDATE stripe_sessions SET fulfilled = 1 WHERE session_id = ?",
                (session_id,),
            )
        return self.add_credits(row["client_key"], int(row["credits"]))


_store: Optional[JobStore] = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
