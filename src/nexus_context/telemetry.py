"""
nexus_context.telemetry
========================
Central in-process telemetry event bus for Nexus-Context.

All pipeline stages emit structured telemetry events to a shared asyncio.Queue.
The dashboard SSE endpoint fans out to all active browser subscribers.

Event types
-----------
session_created        New session initialized
turn_processed         Per-turn metrics after pipeline completion
tool_call_intercepted  Tool-role message compressed by ToolCallCompressor
chunk_boundary         Adaptive chunker boundary detection result
ltkb_fact_persisted    Long-term knowledge base fact written to SQLite
session_restored       Session state restored from SQLite on restart
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Event Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryEvent:
    """A single structured telemetry event."""

    event_type: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Serialize event as a Server-Sent Events data line."""
        import json
        data = json.dumps({
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            **self.payload,
        })
        return f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# TelemetryBus
# ---------------------------------------------------------------------------


class TelemetryBus:
    """In-process async event bus.

    Producers call :meth:`emit` from any coroutine.
    Consumers call :meth:`subscribe` to get a dedicated asyncio.Queue
    that receives copies of every event (fan-out pattern).
    """

    def __init__(self, max_history: int = 500) -> None:
        self._subscribers: list[asyncio.Queue[TelemetryEvent]] = []
        self._history: list[TelemetryEvent] = []
        self._max_history = max_history

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to all active subscribers and history buffer (non-blocking)."""
        # Update history ring buffer
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Fan-out to all subscribers
        dead: list[asyncio.Queue[TelemetryEvent]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def subscribe(self, maxsize: int = 200) -> asyncio.Queue[TelemetryEvent]:
        """Return a new Queue that will receive all future events."""
        q: asyncio.Queue[TelemetryEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TelemetryEvent]) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def get_recent_history(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[TelemetryEvent]:
        """Return recent events, optionally filtered by type."""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# ---------------------------------------------------------------------------
# Convenience emit helpers
# ---------------------------------------------------------------------------


def emit_session_created(bus: TelemetryBus, session_id: str) -> None:
    bus.emit(TelemetryEvent(
        event_type="session_created",
        session_id=session_id,
    ))


def emit_turn_processed(
    bus: TelemetryBus,
    session_id: str,
    pipeline_ms: float,
    tokens_in: int,
    tokens_out: int,
    zone_p_hit: bool,
    compaction_applied: bool,
    memory_tuples_count: int,
    graph_node_count: int,
    graph_edge_count: int,
    turn_counter: int,
) -> None:
    bus.emit(TelemetryEvent(
        event_type="turn_processed",
        session_id=session_id,
        payload={
            "pipeline_ms": round(pipeline_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "zone_p_hit": zone_p_hit,
            "compaction_applied": compaction_applied,
            "memory_tuples_count": memory_tuples_count,
            "graph_node_count": graph_node_count,
            "graph_edge_count": graph_edge_count,
            "turn_counter": turn_counter,
        },
    ))


def emit_tool_call_intercepted(
    bus: TelemetryBus,
    session_id: str,
    original_tokens: int,
    compressed_tokens: int,
) -> None:
    bus.emit(TelemetryEvent(
        event_type="tool_call_intercepted",
        session_id=session_id,
        payload={
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": original_tokens - compressed_tokens,
        },
    ))


def emit_chunk_boundary(
    bus: TelemetryBus,
    session_id: str,
    token_index: int,
    cosine_shift: float,
    token_entropy: float,
    boundary_score: float,
    is_boundary: bool,
    suppressed_by_syntax: bool,
) -> None:
    bus.emit(TelemetryEvent(
        event_type="chunk_boundary",
        session_id=session_id,
        payload={
            "token_index": token_index,
            "cosine_shift": round(cosine_shift, 4),
            "token_entropy": round(token_entropy, 4),
            "boundary_score": round(boundary_score, 4),
            "is_boundary": is_boundary,
            "suppressed_by_syntax": suppressed_by_syntax,
        },
    ))


def emit_ltkb_fact_persisted(
    bus: TelemetryBus,
    session_id: str,
    fact_id: str,
    content_preview: str,
    weight: float,
) -> None:
    bus.emit(TelemetryEvent(
        event_type="ltkb_fact_persisted",
        session_id=session_id,
        payload={
            "fact_id": fact_id,
            "content_preview": content_preview[:80],
            "weight": round(weight, 4),
        },
    ))


def emit_session_restored(bus: TelemetryBus, session_id: str, turn_counter: int) -> None:
    bus.emit(TelemetryEvent(
        event_type="session_restored",
        session_id=session_id,
        payload={"turn_counter": turn_counter},
    ))
