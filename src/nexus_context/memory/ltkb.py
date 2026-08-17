"""
nexus_context.memory.ltkb
==========================
Feature I: Cross-Session Long-Term Knowledge Base (LTKB).

Extracts high-confidence, high-connectivity knowledge nodes from the
ContextGraph at session end and persists them to an SQLite store.
On new session init, retrieves the top-K most relevant facts and injects
them into Zone P so agents benefit from past sessions automatically.

Tables
------
ltkb_facts
    fact_id          TEXT PRIMARY KEY
    session_id       TEXT NOT NULL
    node_type        TEXT NOT NULL
    content          TEXT NOT NULL
    weight           REAL NOT NULL
    in_degree        INTEGER NOT NULL
    created_at       REAL NOT NULL
    last_accessed_at REAL NOT NULL

Reference: Feature I specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PERSIST_WEIGHT_THRESHOLD = 0.8
_PERSIST_MIN_IN_DEGREE = 2


# ---------------------------------------------------------------------------
# LTKBFact dataclass
# ---------------------------------------------------------------------------


@dataclass
class LTKBFact:
    """A single persisted long-term knowledge fact."""

    fact_id: str
    session_id: str
    node_type: str
    content: str
    weight: float
    in_degree: int
    created_at: float
    last_accessed_at: float

    def compact_repr(self, max_len: int = 120) -> str:
        content = self.content[:max_len] + "..." if len(self.content) > max_len else self.content
        return f"[{self.node_type}] {content}"


# ---------------------------------------------------------------------------
# LongTermKnowledgeBase
# ---------------------------------------------------------------------------


class LongTermKnowledgeBase:
    """Persistent cross-session knowledge base backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    weight_threshold:
        Minimum node retention weight to qualify for persistence.
    min_in_degree:
        Minimum number of incoming edges a node must have to qualify.
    """

    def __init__(
        self,
        db_path: str | Path = "nexus_ltkb.db",
        weight_threshold: float = _PERSIST_WEIGHT_THRESHOLD,
        min_in_degree: int = _PERSIST_MIN_IN_DEGREE,
    ) -> None:
        self.db_path = str(db_path)
        self.weight_threshold = weight_threshold
        self.min_in_degree = min_in_degree
        self._initialized = False

    async def initialize(self) -> None:
        """Create LTKB tables if they do not exist."""
        await asyncio.to_thread(self._init_db)
        self._initialized = True
        logger.info('{"event":"ltkb_initialized","db_path":"%s"}', self.db_path)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ltkb_facts (
                    fact_id          TEXT PRIMARY KEY,
                    session_id       TEXT NOT NULL,
                    node_type        TEXT NOT NULL,
                    content          TEXT NOT NULL,
                    weight           REAL NOT NULL,
                    in_degree        INTEGER NOT NULL,
                    created_at       REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltkb_weight ON ltkb_facts(weight DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltkb_session ON ltkb_facts(session_id)")
            conn.commit()

    # ------------------------------------------------------------------
    # Extract & Persist
    # ------------------------------------------------------------------

    async def extract_and_persist(
        self,
        graph: object,
        session_id: str,
        current_turn: int,
        telemetry_bus: object | None = None,
    ) -> int:
        """Extract high-value nodes from *graph* and persist to LTKB.

        Returns the number of facts persisted.
        """
        facts = self._extract_facts(graph, session_id, current_turn)
        if not facts:
            return 0

        count = await asyncio.to_thread(self._persist_facts_sync, facts)
        logger.info(
            '{"event":"ltkb_extracted","session_id":"%s","facts_count":%d}',
            session_id, count,
        )

        # Emit telemetry for each persisted fact
        if telemetry_bus is not None:
            from nexus_context.telemetry import emit_ltkb_fact_persisted
            for fact in facts[:count]:
                emit_ltkb_fact_persisted(
                    telemetry_bus,
                    session_id=session_id,
                    fact_id=fact.fact_id,
                    content_preview=fact.content[:80],
                    weight=fact.weight,
                )

        return count

    def _extract_facts(
        self,
        graph: object,
        session_id: str,
        current_turn: int,
    ) -> list[LTKBFact]:
        """Extract qualifying nodes from ContextGraph."""
        facts: list[LTKBFact] = []

        # Access ContextGraph nodes and edges via duck typing
        try:
            nodes: dict = getattr(graph, "nodes", {})
            edges: list = getattr(graph, "edges", [])
        except Exception:
            return facts

        # Compute in-degree per node
        in_degree: dict[str, int] = {}
        for edge in edges:
            tgt = getattr(edge, "target_id", None)
            if tgt:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        now = time.time()
        for node_id, node in nodes.items():
            try:
                # Compute current retention weight
                weight = getattr(node, "retention_weight", 1.0)
                if hasattr(node, "compute_weight"):
                    weight = node.compute_weight(current_turn)

                node_in_degree = in_degree.get(node_id, 0)
                content = getattr(node, "content", "")

                if (
                    weight >= self.weight_threshold
                    and node_in_degree >= self.min_in_degree
                    and content.strip()
                ):
                    facts.append(LTKBFact(
                        fact_id=str(uuid.uuid4()),
                        session_id=session_id,
                        node_type=getattr(node, "node_type", "unknown"),
                        content=content,
                        weight=float(weight) if not math.isinf(weight) else 999.0,
                        in_degree=node_in_degree,
                        created_at=now,
                        last_accessed_at=now,
                    ))
            except Exception:
                continue

        # Sort by weight descending, cap at 50 facts per session
        facts.sort(key=lambda f: f.weight, reverse=True)
        return facts[:50]

    def _persist_facts_sync(self, facts: list[LTKBFact]) -> int:
        """Write facts to SQLite. Returns count of facts written."""
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for fact in facts:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO ltkb_facts
                        (fact_id, session_id, node_type, content, weight, in_degree, created_at, last_accessed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fact.fact_id, fact.session_id, str(fact.node_type),
                        fact.content, fact.weight, fact.in_degree,
                        fact.created_at, fact.last_accessed_at,
                    ))
                    count += 1
                except Exception:
                    pass
            conn.commit()
        return count

    # ------------------------------------------------------------------
    # Retrieve & Inject
    # ------------------------------------------------------------------

    async def get_relevant_facts(self, query: str, top_k: int = 5) -> list[LTKBFact]:
        """Return the top-K most relevant LTKB facts for *query* (TF-IDF scoring)."""
        all_facts = await asyncio.to_thread(self._load_all_facts_sync)
        if not all_facts:
            return []

        scored = self._score_facts(query, all_facts)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in scored[:top_k]]

    def _load_all_facts_sync(self) -> list[LTKBFact]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ltkb_facts ORDER BY weight DESC LIMIT 1000"
            ).fetchall()
        return [
            LTKBFact(
                fact_id=r["fact_id"],
                session_id=r["session_id"],
                node_type=r["node_type"],
                content=r["content"],
                weight=r["weight"],
                in_degree=r["in_degree"],
                created_at=r["created_at"],
                last_accessed_at=r["last_accessed_at"],
            )
            for r in rows
        ]

    def _score_facts(
        self, query: str, facts: list[LTKBFact]
    ) -> list[tuple[LTKBFact, float]]:
        """Simple word-overlap TF-IDF-like scoring."""
        query_words = set(query.lower().split())
        scored = []
        for fact in facts:
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                score = overlap / math.sqrt(len(fact_words) + 1) * fact.weight
                scored.append((fact, score))
        return scored

    async def inject_into_zone_p(self, facts: list[LTKBFact]) -> str:
        """Format facts as a compact NEXUS_LTKB annotation for Zone P injection."""
        if not facts:
            return ""
        entries = [
            {"node_type": f.node_type, "content": f.content[:200], "weight": round(f.weight, 3)}
            for f in facts
        ]
        return f"<!-- NEXUS_LTKB {json.dumps(entries)} -->"

    async def get_fact_count(self) -> int:
        """Return total number of persisted facts."""
        return await asyncio.to_thread(self._get_fact_count_sync)

    def _get_fact_count_sync(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM ltkb_facts").fetchone()
            return row[0] if row else 0

    async def get_recent_facts(self, limit: int = 10) -> list[LTKBFact]:
        """Return the most recently persisted facts."""
        return await asyncio.to_thread(self._get_recent_facts_sync, limit)

    def _get_recent_facts_sync(self, limit: int) -> list[LTKBFact]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ltkb_facts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            LTKBFact(
                fact_id=r["fact_id"], session_id=r["session_id"],
                node_type=r["node_type"], content=r["content"],
                weight=r["weight"], in_degree=r["in_degree"],
                created_at=r["created_at"], last_accessed_at=r["last_accessed_at"],
            )
            for r in rows
        ]
