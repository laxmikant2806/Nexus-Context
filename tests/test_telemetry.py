"""tests/test_telemetry.py — Feature C: TelemetryBus unit tests."""
from __future__ import annotations

import asyncio
import pytest
from nexus_context.telemetry import (
    TelemetryBus,
    TelemetryEvent,
    emit_session_created,
    emit_turn_processed,
    emit_tool_call_intercepted,
)


class TestTelemetryBus:
    def test_emit_reaches_subscriber(self) -> None:
        bus = TelemetryBus()
        q = bus.subscribe()
        event = TelemetryEvent(event_type="session_created", session_id="s1")
        bus.emit(event)
        assert not q.empty()
        received = q.get_nowait()
        assert received.event_type == "session_created"
        assert received.session_id == "s1"

    def test_fan_out_to_multiple_subscribers(self) -> None:
        bus = TelemetryBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.emit(TelemetryEvent(event_type="test", session_id="s1"))
        assert q1.qsize() == 1
        assert q2.qsize() == 1

    def test_unsubscribe_removes_queue(self) -> None:
        bus = TelemetryBus()
        q = bus.subscribe()
        assert bus.subscriber_count == 1
        bus.unsubscribe(q)
        assert bus.subscriber_count == 0

    def test_history_capped_at_max(self) -> None:
        bus = TelemetryBus(max_history=5)
        for i in range(10):
            bus.emit(TelemetryEvent(event_type="test", session_id=f"s{i}"))
        assert len(bus.get_recent_history()) == 5

    def test_history_filter_by_type(self) -> None:
        bus = TelemetryBus()
        bus.emit(TelemetryEvent(event_type="type_a", session_id="s1"))
        bus.emit(TelemetryEvent(event_type="type_b", session_id="s2"))
        bus.emit(TelemetryEvent(event_type="type_a", session_id="s3"))
        results = bus.get_recent_history(event_type="type_a")
        assert len(results) == 2
        assert all(e.event_type == "type_a" for e in results)

    def test_sse_serialization_format(self) -> None:
        event = TelemetryEvent(
            event_type="turn_processed",
            session_id="sess-abc",
            payload={"pipeline_ms": 42.5},
        )
        sse = event.to_sse()
        assert sse.startswith("data: {")
        assert sse.endswith("\n\n")
        assert "turn_processed" in sse
        assert "sess-abc" in sse

    def test_emit_session_created_helper(self) -> None:
        bus = TelemetryBus()
        q = bus.subscribe()
        emit_session_created(bus, "test-session")
        ev = q.get_nowait()
        assert ev.event_type == "session_created"
        assert ev.session_id == "test-session"

    def test_emit_turn_processed_helper(self) -> None:
        bus = TelemetryBus()
        q = bus.subscribe()
        emit_turn_processed(
            bus, "s1", pipeline_ms=10.5, tokens_in=200, tokens_out=0,
            zone_p_hit=True, compaction_applied=False,
            memory_tuples_count=5, graph_node_count=12,
            graph_edge_count=8, turn_counter=3,
        )
        ev = q.get_nowait()
        assert ev.event_type == "turn_processed"
        assert ev.payload["pipeline_ms"] == 10.5
        assert ev.payload["tokens_in"] == 200
        assert ev.payload["zone_p_hit"] is True

    def test_emit_tool_call_intercepted_helper(self) -> None:
        bus = TelemetryBus()
        q = bus.subscribe()
        emit_tool_call_intercepted(bus, "s1", original_tokens=1000, compressed_tokens=200)
        ev = q.get_nowait()
        assert ev.event_type == "tool_call_intercepted"
        assert ev.payload["tokens_saved"] == 800
