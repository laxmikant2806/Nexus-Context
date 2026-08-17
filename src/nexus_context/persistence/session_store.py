"""
nexus_context.persistence.session_store
=========================================
Feature B: Session Persistence & Crash Recovery.

SQLite-backed async session store using the stdlib ``sqlite3`` module
(via a thread-pool executor) so there are zero additional dependencies.

Tables
------
sessions
    session_id    TEXT PRIMARY KEY
    turn_counter  INTEGER NOT NULL
    zone_p_hash   TEXT
    aligned_zone_p TEXT
    created_at    REAL
    updated_at    REAL

memory_tuples
    memory_id     TEXT PRIMARY KEY
    session_id    TEXT NOT NULL
    serialized_json TEXT NOT NULL

Reference: Feature B specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Async SQLite session store for Nexus-Context session persistence.

    Uses ``asyncio.to_thread`` to run blocking SQLite calls on the thread-pool
    executor, keeping the event loop free.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (created on first use).
    """

    def __init__(self, db_path: str | Path = "nexus_sessions.db") -> None:
        self.db_path = str(db_path)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables if they do not exist."""
        await asyncio.to_thread(self._init_db)
        self._initialized = True
        logger.info('{"event":"session_store_initialized","db_path":"%s"}', self.db_path)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id    TEXT PRIMARY KEY,
                    turn_counter  INTEGER NOT NULL DEFAULT 0,
                    zone_p_hash   TEXT,
                    aligned_zone_p TEXT,
                    created_at    REAL NOT NULL,
                    updated_at    REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_tuples (
                    memory_id       TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    serialized_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_session ON memory_tuples(session_id)")
            conn.commit()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    async def save_session(
        self,
        session_id: str,
        turn_counter: int,
        zone_p_hash: str | None,
        aligned_zone_p: str | None,
        memory_tuples_json: list[str] | None = None,
    ) -> None:
        """Upsert session row and its memory tuples."""
        now = time.time()
        await asyncio.to_thread(
            self._save_session_sync,
            session_id, turn_counter, zone_p_hash, aligned_zone_p,
            memory_tuples_json or [], now,
        )

    def _save_session_sync(
        self,
        session_id: str,
        turn_counter: int,
        zone_p_hash: str | None,
        aligned_zone_p: str | None,
        memory_tuples_json: list[str],
        now: float,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, turn_counter, zone_p_hash, aligned_zone_p, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    turn_counter   = excluded.turn_counter,
                    zone_p_hash    = excluded.zone_p_hash,
                    aligned_zone_p = excluded.aligned_zone_p,
                    updated_at     = excluded.updated_at
            """, (session_id, turn_counter, zone_p_hash, aligned_zone_p, now, now))

            # Replace all memory tuples for this session
            conn.execute("DELETE FROM memory_tuples WHERE session_id = ?", (session_id,))
            for raw_json in memory_tuples_json:
                try:
                    data = json.loads(raw_json)
                    memory_id = data.get("memory_id", f"{session_id}:{time.time()}")
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_tuples (memory_id, session_id, serialized_json) VALUES (?, ?, ?)",
                        (memory_id, session_id, raw_json),
                    )
                except Exception:
                    pass
            conn.commit()
        logger.debug('{"event":"session_saved","session_id":"%s","turns":%d}', session_id, turn_counter)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Return session data dict or None if not found."""
        return await asyncio.to_thread(self._load_session_sync, session_id)

    def _load_session_sync(self, session_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None

            tuples_rows = conn.execute(
                "SELECT serialized_json FROM memory_tuples WHERE session_id = ?",
                (session_id,),
            ).fetchall()

            return {
                "session_id": row["session_id"],
                "turn_counter": row["turn_counter"],
                "zone_p_hash": row["zone_p_hash"],
                "aligned_zone_p": row["aligned_zone_p"],
                "memory_tuples_json": [r["serialized_json"] for r in tuples_rows],
            }

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    async def list_sessions(self) -> list[str]:
        """Return all persisted session IDs."""
        return await asyncio.to_thread(self._list_sessions_sync)

    def _list_sessions_sync(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [r[0] for r in rows]

    async def delete_session(self, session_id: str) -> None:
        """Delete session and all associated memory tuples."""
        await asyncio.to_thread(self._delete_session_sync, session_id)

    def _delete_session_sync(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
