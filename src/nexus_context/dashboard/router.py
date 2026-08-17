"""
nexus_context.dashboard.router
================================
FastAPI router serving the real-time observability dashboard.

Routes
------
GET /dashboard           Serves the inline HTML dashboard page
GET /dashboard/stream    SSE endpoint — fans telemetry events to browser
GET /dashboard/api/state Snapshot of current state (JSON, for initial load)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from nexus_context.dashboard.ui import get_dashboard_html
from nexus_context.telemetry import TelemetryBus, TelemetryEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page() -> HTMLResponse:
    """Serve the single-page dashboard HTML."""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)


@router.get("/stream")
async def dashboard_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream for the dashboard.

    The browser connects to this endpoint and receives a continuous stream of
    JSON-encoded telemetry events separated by ``data: ...\\n\\n`` SSE lines.
    Each event is relayed from the :class:`TelemetryBus` fan-out queue.
    """
    bus: TelemetryBus | None = getattr(request.app.state, "telemetry", None)

    async def _generate() -> AsyncIterator[str]:
        if bus is None:
            # No telemetry bus available — send a single error event and stop
            yield 'data: {"event_type":"error","session_id":"","message":"Telemetry bus not initialized"}\n\n'
            return

        q = bus.subscribe(maxsize=300)

        # Replay last 30 events from history so dashboard pre-populates on connect
        for past_event in bus.get_recent_history(limit=30):
            yield past_event.to_sse()

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event: TelemetryEvent = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    # Send keepalive comment every 15s to prevent proxy timeout
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/state")
async def dashboard_state(request: Request) -> JSONResponse:
    """Return a JSON snapshot of current server state for REST clients."""
    sessions_state = getattr(request.app.state, "sessions", {})
    telemetry: TelemetryBus | None = getattr(request.app.state, "telemetry", None)

    sessions_summary = {
        sid: {
            "turn_counter": getattr(s, "turn_counter", 0),
            "zone_p_hash": getattr(s, "zone_p_hash", None),
            "memory_pool_size": len(getattr(s, "memory_pool", None) or []),
        }
        for sid, s in sessions_state.items()
    }

    return JSONResponse({
        "active_sessions": len(sessions_state),
        "sessions": sessions_summary,
        "telemetry_subscribers": telemetry.subscriber_count if telemetry else 0,
        "recent_events": len(telemetry.get_recent_history() if telemetry else []),
    })
