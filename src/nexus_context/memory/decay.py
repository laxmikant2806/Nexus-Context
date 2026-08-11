"""
nexus_context.memory.decay
===========================
In-session memory pool with exponential temporal decay and AST-depth weighting.

The pool maintains a dict of :class:`MemoryTuple` objects keyed by
``memory_id``, deduplicated by ``(target_name, scope_path)``.  At each turn,
retention weights are recomputed and the pool is serialised as a compact JSON
annotation block for injection into Zone T.

Reference: docs/overview_and_research.md §4.4–4.5,
           docs/implementation_stage.md §3.2.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from filelock import FileLock  # type: ignore[import-untyped]

from nexus_context.memory.schemas import MemoryTuple

logger = logging.getLogger(__name__)


class MemoryPool:
    """Manages WWW memory tuples for one session.

    Parameters
    ----------
    session_id:
        Session identifier (used for logging and persistence filenames).
    lambda_:
        Temporal decay constant λ (default 0.05, half-life ≈ 13.9 turns).
    eta:
        AST depth amplification factor η (default 0.5).
    persist_path:
        Optional ``Path`` to a ``.jsonl`` file for cross-session persistence.
        If provided, the pool is loaded from disk on initialisation and saved
        after every :meth:`update_weights` call.
    """

    def __init__(
        self,
        session_id: str,
        lambda_: float = 0.05,
        eta: float = 0.5,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._session_id = session_id
        self._lambda = lambda_
        self._eta = eta
        self._persist_path = persist_path

        # Primary storage: memory_id → MemoryTuple
        self._pool: dict[str, MemoryTuple] = {}
        # Deduplication index: (target_name, scope_path) → memory_id
        self._key_index: dict[tuple[str, str], str] = {}

        if persist_path is not None and persist_path.exists():
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public: mutation
    # ------------------------------------------------------------------

    def add(self, tuples: list[MemoryTuple]) -> None:
        """Add *tuples* to the pool, deduplicating by ``(target_name, where)``.

        Deduplication rule (see docs/implementation_stage.md §3.2):
        - If a more-recent tuple exists for the same ``(name, scope)`` key,
          replace the older one — *unless* the older one is pinned.
        - Pinned tuples are versioned: the key is suffixed with
          ``_v{when}`` so both versions survive.
        """
        for t in tuples:
            key = (t.what.target_name, t.where)
            existing_id = self._key_index.get(key)

            if existing_id is not None and existing_id in self._pool:
                old = self._pool[existing_id]
                if old.is_pinned:
                    # Versioned key for pinned: keep both
                    versioned_key = (f"{t.what.target_name}_v{old.when}", t.where)
                    self._key_index[versioned_key] = existing_id
                    # New entry gets the canonical key
                    self._pool[t.memory_id] = t
                    self._key_index[key] = t.memory_id
                elif t.when >= old.when:
                    # Replace with newer
                    del self._pool[existing_id]
                    self._pool[t.memory_id] = t
                    self._key_index[key] = t.memory_id
                # else: existing is newer; drop incoming
            else:
                self._pool[t.memory_id] = t
                self._key_index[key] = t.memory_id

        logger.debug(
            '{"event":"memory_add","n_added":%d,"pool_size":%d}',
            len(tuples),
            len(self._pool),
        )

    def pin(self, memory_id: str) -> None:
        """Mark a memory as permanently retained (bypasses temporal decay)."""
        if memory_id in self._pool:
            # Use model_copy with update (Pydantic v2 pattern)
            old = self._pool[memory_id]
            self._pool[memory_id] = old.model_copy(
                update={"is_pinned": True, "retention_weight": math.inf}
            )
            logger.debug('{"event":"memory_pinned","memory_id":"%s"}', memory_id)

    def pin_by_dependency(self, referenced_names: set[str]) -> None:
        """Pin all tuples whose ``what.target_name`` is in *referenced_names*."""
        for mem_id, mem in list(self._pool.items()):
            if mem.what.target_name in referenced_names:
                self.pin(mem_id)

    # ------------------------------------------------------------------
    # Public: weight update
    # ------------------------------------------------------------------

    def update_weights(self, current_turn: int) -> None:
        """Recompute ``retention_weight`` for all non-pinned tuples.

        Called once per turn by the middleware pipeline before
        :meth:`select`.
        """
        for mem_id, mem in list(self._pool.items()):
            if not mem.is_pinned:
                new_weight = mem.compute_weight(current_turn, self._lambda, self._eta)
                self._pool[mem_id] = mem.model_copy(
                    update={"retention_weight": new_weight}
                )

        if self._persist_path is not None:
            self._save(self._persist_path)

    # ------------------------------------------------------------------
    # Public: selection
    # ------------------------------------------------------------------

    def select(self, budget_tokens: int) -> list[MemoryTuple]:
        """Greedily select high-weight tuples within *budget_tokens*.

        Pinned tuples (``is_pinned=True``) are always included first.
        Remaining budget is filled by descending ``retention_weight``.

        Returns
        -------
        list[MemoryTuple]
            Sorted by ``when`` ascending (chronological order for context
            injection).
        """
        sorted_pool = sorted(
            self._pool.values(),
            key=lambda m: (0 if m.is_pinned else 1, -m.retention_weight),
        )

        selected: list[MemoryTuple] = []
        tokens_used = 0

        for mem in sorted_pool:
            if tokens_used + mem.token_count <= budget_tokens:
                selected.append(mem)
                tokens_used += mem.token_count

        # Return in chronological order
        selected.sort(key=lambda m: m.when)

        logger.debug(
            '{"event":"memory_select","n_selected":%d,"tokens_used":%d,"budget":%d}',
            len(selected),
            tokens_used,
            budget_tokens,
        )
        return selected

    # ------------------------------------------------------------------
    # Public: serialisation for context injection
    # ------------------------------------------------------------------

    def serialize_for_context(self, selected: list[MemoryTuple]) -> str:
        """Serialise *selected* tuples as a compact HTML-comment memory block.

        Format (see ADR-006)::

            <!-- NEXUS_MEMORY
            [{"w":"tool:execute_python","d":"conn=psycopg2.connect(...)","t":5,"@":"module"},
             ...]
            -->

        Returns an empty string if *selected* is empty.
        """
        if not selected:
            return ""

        compact_list = [
            {
                "w": mem.who_detail,
                "d": mem.what.compact_repr(),
                "t": mem.when,
                "@": mem.where,
            }
            for mem in selected
        ]
        json_str = json.dumps(compact_list, separators=(",", ":"))
        return f"<!-- NEXUS_MEMORY\n{json_str}\n-->"

    # ------------------------------------------------------------------
    # Private: persistence
    # ------------------------------------------------------------------

    def _save(self, path: Path) -> None:
        """Atomically write the pool to *path* as JSONL (write-then-rename)."""
        tmp_path = path.with_suffix(".jsonl.tmp")
        try:
            lock = FileLock(str(path) + ".lock", timeout=5)
            with lock:
                with tmp_path.open("w", encoding="utf-8") as f:
                    for mem in self._pool.values():
                        f.write(mem.model_dump_json() + "\n")
                tmp_path.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '{"event":"memory_save_failed","path":"%s","reason":"%s"}',
                str(path),
                str(exc),
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load(self, path: Path) -> None:
        """Load pool from *path* JSONL; skip corrupted lines with a WARNING."""
        loaded = 0
        skipped = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        mem = MemoryTuple.model_validate_json(line)
                        self._pool[mem.memory_id] = mem
                        key = (mem.what.target_name, mem.where)
                        self._key_index[key] = mem.memory_id
                        loaded += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            '{"event":"memory_load_line_error","line":%d,"reason":"%s"}',
                            line_no,
                            str(exc),
                        )
                        skipped += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '{"event":"memory_load_failed","path":"%s","reason":"%s"}',
                str(path),
                str(exc),
            )
        logger.info(
            '{"event":"memory_loaded","loaded":%d,"skipped":%d,"path":"%s"}',
            loaded,
            skipped,
            str(path),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        return (
            f"MemoryPool(session_id={self._session_id!r}, "
            f"n_tuples={len(self._pool)}, "
            f"lambda={self._lambda}, eta={self._eta})"
        )
