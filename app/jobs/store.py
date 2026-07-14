"""SQLite job store + atomic credit/usage ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.core.config import get_settings


class QuotaExceeded(Exception):
    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.message = message
        self.usage = usage


@dataclass
class Reservation:
    """Proof that a convert slot was reserved (debit already applied)."""

    mode: str  # "credits" | "free"
    cost: int
    client_key: str
    ip_key: str


class JobStore:
    def __init__(self, db_path: Optional[Path] = None, jobs_dir: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self.jobs_dir = jobs_dir or settings.jobs_dir
        self.ttl = settings.job_ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        owner_key TEXT NOT NULL DEFAULT '',
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
                    CREATE TABLE IF NOT EXISTS ip_usage (
                        ip_key TEXT NOT NULL,
                        day TEXT NOT NULL,
                        checks INTEGER NOT NULL DEFAULT 0,
                        converts INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (ip_key, day)
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
                # Migrate older DBs missing owner_key
                cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
                if "owner_key" not in cols:
                    conn.execute(
                        "ALTER TABLE jobs ADD COLUMN owner_key TEXT NOT NULL DEFAULT ''"
                    )

    def create_job(
        self,
        *,
        owner_key: str,
        doc_type: str,
        metrics: dict,
        validation: dict,
        warnings: list,
        files: Dict[str, bytes],
        preview_jpeg: bytes,
    ) -> str:
        with self._lock:
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
                    INSERT INTO jobs (id, owner_key, created_at, expires_at, doc_type,
                                      metrics_json, validation_json, warnings_json, file_index_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        owner_key,
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

    def get_meta(self, job_id: str, owner_key: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            if not row:
                return None
            if row["expires_at"] < time.time():
                self.delete_job(job_id)
                return None
            if owner_key is not None and row["owner_key"] and row["owner_key"] != owner_key:
                return None  # not owner — treat as missing
            return {
                "id": row["id"],
                "owner_key": row["owner_key"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "doc_type": row["doc_type"],
                "metrics": json.loads(row["metrics_json"]),
                "validation": json.loads(row["validation_json"]),
                "warnings": json.loads(row["warnings_json"]),
                "files": json.loads(row["file_index_json"]),
            }

    def get_file(
        self, job_id: str, filename: str, owner_key: Optional[str] = None
    ) -> Optional[bytes]:
        meta = self.get_meta(job_id, owner_key=owner_key)
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

        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            job_dir = self.jobs_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE expires_at < ?", (now,)
                ).fetchall()
                ids = [r["id"] for r in rows]
            for jid in ids:
                self.delete_job(jid)
        return len(ids)

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def get_usage(self, client_key: str, ip_key: Optional[str] = None) -> dict:
        day = self._today()
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT checks, converts FROM usage WHERE client_key = ? AND day = ?",
                    (client_key, day),
                ).fetchone()
                cred = conn.execute(
                    "SELECT balance FROM credits WHERE client_key = ?", (client_key,)
                ).fetchone()
                ip_row = None
                if ip_key:
                    ip_row = conn.execute(
                        "SELECT checks, converts FROM ip_usage WHERE ip_key = ? AND day = ?",
                        (ip_key, day),
                    ).fetchone()
        return {
            "day": day,
            "checks": int(row["checks"]) if row else 0,
            "converts": int(row["converts"]) if row else 0,
            "credit_balance": int(cred["balance"]) if cred else 0,
            "ip_checks": int(ip_row["checks"]) if ip_row else 0,
            "ip_converts": int(ip_row["converts"]) if ip_row else 0,
        }

    def _usage_from_conn(self, conn: sqlite3.Connection, client_key: str, ip_key: str) -> dict:
        day = self._today()
        row = conn.execute(
            "SELECT checks, converts FROM usage WHERE client_key = ? AND day = ?",
            (client_key, day),
        ).fetchone()
        cred = conn.execute(
            "SELECT balance FROM credits WHERE client_key = ?", (client_key,)
        ).fetchone()
        ip_row = conn.execute(
            "SELECT checks, converts FROM ip_usage WHERE ip_key = ? AND day = ?",
            (ip_key, day),
        ).fetchone()
        return {
            "day": day,
            "checks": int(row["checks"]) if row else 0,
            "converts": int(row["converts"]) if row else 0,
            "credit_balance": int(cred["balance"]) if cred else 0,
            "ip_checks": int(ip_row["checks"]) if ip_row else 0,
            "ip_converts": int(ip_row["converts"]) if ip_row else 0,
        }

    def try_record_check(
        self,
        client_key: str,
        ip_key: str,
        free_daily: int,
        ip_free_daily: int,
    ) -> dict:
        """Atomically allow + record one check, or raise QuotaExceeded."""
        day = self._today()
        with self._lock:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                usage = self._usage_from_conn(conn, client_key, ip_key)
                has_credits = usage["credit_balance"] > 0
                if not has_credits:
                    if usage["checks"] >= free_daily:
                        conn.execute("ROLLBACK")
                        raise QuotaExceeded(
                            "Daily free checks used. Buy credits for unlimited checks.",
                            usage,
                        )
                    if usage["ip_checks"] >= ip_free_daily:
                        conn.execute("ROLLBACK")
                        raise QuotaExceeded(
                            "Network free-check limit reached for today. Buy credits or try tomorrow.",
                            usage,
                        )
                conn.execute(
                    """
                    INSERT INTO usage (client_key, day, checks, converts) VALUES (?, ?, 1, 0)
                    ON CONFLICT(client_key, day) DO UPDATE SET checks = checks + 1
                    """,
                    (client_key, day),
                )
                conn.execute(
                    """
                    INSERT INTO ip_usage (ip_key, day, checks, converts) VALUES (?, ?, 1, 0)
                    ON CONFLICT(ip_key, day) DO UPDATE SET checks = checks + 1
                    """,
                    (ip_key, day),
                )
                conn.execute("COMMIT")
                return self._usage_from_conn(conn, client_key, ip_key)

    def reserve_convert(
        self,
        client_key: str,
        ip_key: str,
        free_daily: int,
        cost: int,
        ip_free_daily: int,
    ) -> Tuple[Reservation, dict]:
        """
        Atomically reserve one convert (debit credit OR free slot).
        Call refund_reservation() if processing fails after reserve.
        """
        day = self._today()
        with self._lock:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                usage = self._usage_from_conn(conn, client_key, ip_key)

                # Prefer credits
                if usage["credit_balance"] >= cost:
                    cur = conn.execute(
                        """
                        UPDATE credits SET balance = balance - ?, updated_at = ?
                        WHERE client_key = ? AND balance >= ?
                        """,
                        (cost, time.time(), client_key, cost),
                    )
                    if cur.rowcount != 1:
                        conn.execute("ROLLBACK")
                        usage = self._usage_from_conn(conn, client_key, ip_key)
                        raise QuotaExceeded(
                            "Insufficient credits.",
                            usage,
                        )
                    conn.execute(
                        """
                        INSERT INTO usage (client_key, day, checks, converts) VALUES (?, ?, 0, 1)
                        ON CONFLICT(client_key, day) DO UPDATE SET converts = converts + 1
                        """,
                        (client_key, day),
                    )
                    conn.execute(
                        """
                        INSERT INTO ip_usage (ip_key, day, checks, converts) VALUES (?, ?, 0, 1)
                        ON CONFLICT(ip_key, day) DO UPDATE SET converts = converts + 1
                        """,
                        (ip_key, day),
                    )
                    conn.execute("COMMIT")
                    usage = self._usage_from_conn(conn, client_key, ip_key)
                    return (
                        Reservation(
                            mode="credits",
                            cost=cost,
                            client_key=client_key,
                            ip_key=ip_key,
                        ),
                        usage,
                    )

                # Free path: client + IP caps
                if usage["converts"] >= free_daily:
                    conn.execute("ROLLBACK")
                    raise QuotaExceeded(
                        "You have used today's free conversions. Buy credits to continue.",
                        usage,
                    )
                if usage["ip_converts"] >= ip_free_daily:
                    conn.execute("ROLLBACK")
                    raise QuotaExceeded(
                        "Network free-conversion limit reached for today. Buy credits or try tomorrow.",
                        usage,
                    )

                conn.execute(
                    """
                    INSERT INTO usage (client_key, day, checks, converts) VALUES (?, ?, 0, 1)
                    ON CONFLICT(client_key, day) DO UPDATE SET converts = converts + 1
                    """,
                    (client_key, day),
                )
                conn.execute(
                    """
                    INSERT INTO ip_usage (ip_key, day, checks, converts) VALUES (?, ?, 0, 1)
                    ON CONFLICT(ip_key, day) DO UPDATE SET converts = converts + 1
                    """,
                    (ip_key, day),
                )
                conn.execute("COMMIT")
                usage = self._usage_from_conn(conn, client_key, ip_key)
                return (
                    Reservation(
                        mode="free", cost=0, client_key=client_key, ip_key=ip_key
                    ),
                    usage,
                )

    def refund_reservation(self, reservation: Reservation) -> dict:
        """Undo a successful reserve after processing failure."""
        day = self._today()
        with self._lock:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if reservation.mode == "credits" and reservation.cost > 0:
                    conn.execute(
                        """
                        INSERT INTO credits (client_key, balance, updated_at) VALUES (?, ?, ?)
                        ON CONFLICT(client_key) DO UPDATE SET
                            balance = balance + excluded.balance,
                            updated_at = excluded.updated_at
                        """,
                        (reservation.client_key, reservation.cost, time.time()),
                    )
                # decrement converts (not below 0)
                conn.execute(
                    """
                    UPDATE usage SET converts = MAX(0, converts - 1)
                    WHERE client_key = ? AND day = ?
                    """,
                    (reservation.client_key, day),
                )
                conn.execute(
                    """
                    UPDATE ip_usage SET converts = MAX(0, converts - 1)
                    WHERE ip_key = ? AND day = ?
                    """,
                    (reservation.ip_key, day),
                )
                conn.execute("COMMIT")
                return self._usage_from_conn(
                    conn, reservation.client_key, reservation.ip_key
                )

    def add_credits(self, client_key: str, amount: int) -> int:
        if amount <= 0:
            raise ValueError("amount must be positive")
        with self._lock:
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
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO stripe_sessions
                    (session_id, client_key, credits, fulfilled, created_at)
                    VALUES (?, ?, ?, 0, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        client_key = excluded.client_key,
                        credits = excluded.credits
                    WHERE fulfilled = 0
                    """,
                    (session_id, client_key, credits, time.time()),
                )

    def fulfill_stripe_session(self, session_id: str) -> Tuple[bool, Optional[int], str]:
        """
        Idempotent fulfill. Returns (credited_now, balance_or_none, status).
        status: 'credited' | 'already_fulfilled' | 'unknown_session'
        Never double-credits.
        """
        with self._lock:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM stripe_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    return False, None, "unknown_session"
                if row["fulfilled"]:
                    bal = conn.execute(
                        "SELECT balance FROM credits WHERE client_key = ?",
                        (row["client_key"],),
                    ).fetchone()
                    conn.execute("ROLLBACK")
                    return (
                        False,
                        int(bal["balance"]) if bal else 0,
                        "already_fulfilled",
                    )

                cur = conn.execute(
                    """
                    UPDATE stripe_sessions SET fulfilled = 1
                    WHERE session_id = ? AND fulfilled = 0
                    """,
                    (session_id,),
                )
                if cur.rowcount != 1:
                    bal = conn.execute(
                        "SELECT balance FROM credits WHERE client_key = ?",
                        (row["client_key"],),
                    ).fetchone()
                    conn.execute("ROLLBACK")
                    return (
                        False,
                        int(bal["balance"]) if bal else 0,
                        "already_fulfilled",
                    )

                credits = int(row["credits"])
                client_key = row["client_key"]
                conn.execute(
                    """
                    INSERT INTO credits (client_key, balance, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(client_key) DO UPDATE SET
                        balance = balance + excluded.balance,
                        updated_at = excluded.updated_at
                    """,
                    (client_key, credits, time.time()),
                )
                bal = conn.execute(
                    "SELECT balance FROM credits WHERE client_key = ?",
                    (client_key,),
                ).fetchone()
                conn.execute("COMMIT")
                return True, int(bal["balance"]), "credited"


_store: Optional[JobStore] = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
