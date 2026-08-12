"""
nexus_context.cache.middleware
================================
FastAPI transparent proxy middleware for the Nexus-Context pipeline.

This module is the entry point when running ``nexus-serve``.  It intercepts
``POST /v1/chat/completions`` requests, runs the five-stage Nexus-Context
pipeline (block alignment → zone segmentation → AST graph → submodular
pruning → WWW memory injection), and forwards the transformed payload to
the configured backend (vLLM, SGLang, or Ollama).

Reference: docs/implementation_stage.md §5  (Phase 4)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from nexus_context import BackendProxyError, ConfigurationError
from nexus_context.cache.block_align import BlockAligner
from nexus_context.cache.differential import ZoneSegmenter
from nexus_context.cache.schemas import ChatMessage
from nexus_context.guard.ast_graph import ContextGraphBuilder
from nexus_context.guard.submodular import SubmodularSolver
from nexus_context.memory.decay import MemoryPool
from nexus_context.memory.www_parser import WWWParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (overridden by nexus_config.yaml at startup)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "backend": {
        "url": "http://localhost:8000",
        "type": "vllm",
    },
    "cache": {
        "block_size": 16,
        "tail_budget_tokens": 1024,
        "tail_retention_turns": 3,
    },
    "guard": {
        "alpha": 0.5,
        "beta": 0.5,
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "memory": {
        "lambda": 0.05,
        "eta": 0.5,
        "budget_fraction": 0.15,
        "persist": False,
    },
    "server": {
        "model_name": "default",  # set from /v1/models probe at startup
        "total_budget": 4096,
    },
}


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


class _SessionState:
    """Mutable per-session state stored in-process (see ADR-005)."""

    __slots__ = (
        "session_id",
        "aligner",
        "segmenter",
        "memory_pool",
        "aligned_zone_p",
        "zone_p_hash",
        "turn_counter",
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.aligner: BlockAligner | None = None
        self.segmenter: ZoneSegmenter | None = None
        self.memory_pool: MemoryPool | None = None
        self.aligned_zone_p: str | None = None
        self.zone_p_hash: str | None = None
        self.turn_counter: int = 0


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared subsystems on startup; clean up on shutdown."""
    cfg = app.state.config  # type: ignore[attr-defined]

    app.state.graph_builder = ContextGraphBuilder(session_id="shared")
    app.state.solver = SubmodularSolver(
        embedding_model=cfg["guard"]["embedding_model"],
        alpha=cfg["guard"]["alpha"],
        beta=cfg["guard"]["beta"],
    )
    app.state.www_parser_factory = lambda sid: WWWParser(session_id=sid)
    app.state.sessions: dict[str, _SessionState] = {}
    app.state.http_client = httpx.AsyncClient(
        base_url=cfg["backend"]["url"],
        timeout=httpx.Timeout(300.0),
    )

    logger.info(
        '{"event":"middleware_startup","backend":"%s","type":"%s"}',
        cfg["backend"]["url"],
        cfg["backend"]["type"],
    )
    yield

    await app.state.http_client.aclose()
    logger.info('{"event":"middleware_shutdown"}')


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Create and return the configured FastAPI application instance."""
    app = FastAPI(
        title="Nexus-Context Middleware",
        description=(
            "Transparent proxy that applies referential integrity, KV cache "
            "alignment, and WWW memory governance to SLM context payloads."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.config = config or _DEFAULT_CONFIG
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        session_id = request.headers.get("X-Session-ID") or _derive_session_id(request)
        cfg: dict[str, Any] = app.state.config
        t_start = time.perf_counter()

        # --------------------------------------------------------
        # Session initialisation (first call only)
        # --------------------------------------------------------
        session = app.state.sessions.get(session_id)
        if session is None:
            session = _SessionState(session_id)
            app.state.sessions[session_id] = session

        # --------------------------------------------------------
        # Parse messages
        # --------------------------------------------------------
        raw_messages: list[dict[str, str]] = body.get("messages", [])
        messages = _parse_messages(raw_messages, session.turn_counter)

        # --------------------------------------------------------
        # Zone P alignment (once per session)
        # --------------------------------------------------------
        system_msgs = [m for m in messages if m.role == "system"]
        if session.aligned_zone_p is None and system_msgs:
            model_name = body.get("model", cfg["server"]["model_name"])
            aligner = BlockAligner(
                model_name=model_name,
                backend=cfg["backend"]["type"],
                block_size=cfg["cache"]["block_size"],
                backend_url=cfg["backend"]["url"],
            )
            align_result = aligner.align(system_msgs[0].content)
            session.aligner = aligner
            session.aligned_zone_p = align_result.padded_content
            session.zone_p_hash = align_result.zone_p_hash
            session.segmenter = ZoneSegmenter(
                total_budget=cfg["server"]["total_budget"],
                block_size=cfg["cache"]["block_size"],
                zone_p_aligned_tokens=align_result.aligned_token_count,
                tail_budget=cfg["cache"]["tail_budget_tokens"],
                tail_retention_turns=cfg["cache"]["tail_retention_turns"],
            )
            session.memory_pool = MemoryPool(
                session_id=session_id,
                lambda_=cfg["memory"]["lambda"],
                eta=cfg["memory"]["eta"],
            )

        # Replace system message content with frozen Zone P
        if session.aligned_zone_p is not None:
            messages = [
                m.model_copy(update={"content": session.aligned_zone_p})
                if m.role == "system" else m
                for m in messages
            ]

        # --------------------------------------------------------
        # Zone segmentation
        # --------------------------------------------------------
        transformed_messages: list[dict[str, str]] = []

        if session.segmenter is not None:
            bundle = session.segmenter.segment(messages)
            bundle = session.segmenter.graduate_tail_turns(
                bundle, session.turn_counter
            )

            # --------------------------------------------------------
            # Submodular compaction (if Zone T over budget)
            # --------------------------------------------------------
            compaction = None
            pruned_turns: list[ChatMessage] = []
            if bundle.compaction_required and bundle.zone_t_candidates:
                graph = app.state.graph_builder.build(bundle.zone_t_candidates)
                graph.compute_transitive_closure()
                query = (
                    bundle.zone_r_messages[-1].content
                    if bundle.zone_r_messages
                    else "continue task"
                )
                compaction = app.state.solver.solve(
                    graph,
                    budget=bundle.zone_t_budget,
                    query=query,
                    turn_index=session.turn_counter,
                )
                selected_ids = set(compaction.selected_node_ids)
                pruned_turns = [
                    m for m in bundle.zone_t_candidates
                    if not any(nid.startswith(str(m.turn_index) + ":") for nid in selected_ids)
                ]

            # --------------------------------------------------------
            # WWW memory update
            # --------------------------------------------------------
            if session.memory_pool is not None and pruned_turns:
                pool = session.memory_pool
                for turn in pruned_turns:
                    tuples = app.state.www_parser_factory(session_id).extract(turn)
                    pool.add(tuples)
                if compaction and compaction.forced_inclusions:
                    forced_names = {nid.split(":")[-1] for nid in compaction.forced_inclusions}
                    pool.pin_by_dependency(forced_names)
                pool.update_weights(session.turn_counter)
                memory_budget = int(
                    cfg["server"]["total_budget"] * cfg["memory"]["budget_fraction"]
                )
                selected_memories = pool.select(memory_budget)
                memory_block = pool.serialize_for_context(selected_memories)
            else:
                memory_block = ""

            # --------------------------------------------------------
            # Reassemble messages list
            # --------------------------------------------------------
            transformed_messages.append(
                {"role": "system", "content": bundle.zone_p_message.content}
            )
            if memory_block:
                transformed_messages.append(
                    {"role": "system", "content": memory_block}
                )
            if compaction:
                transformed_messages.append(
                    {"role": "user", "content": compaction.reconstructed_text}
                )
            else:
                for m in bundle.zone_t_candidates:
                    transformed_messages.append({"role": m.role, "content": m.content})
            for m in bundle.zone_r_messages:
                transformed_messages.append({"role": m.role, "content": m.content})
        else:
            transformed_messages = raw_messages

        session.turn_counter += 1

        # --------------------------------------------------------
        # Forward to backend
        # --------------------------------------------------------
        forward_body = {**body, "messages": transformed_messages}
        pipeline_ms = (time.perf_counter() - t_start) * 1000

        try:
            if body.get("stream", False):
                return await _stream_response(
                    app.state.http_client, forward_body, pipeline_ms
                )
            else:
                return await _json_response(
                    app.state.http_client, forward_body, pipeline_ms
                )
        except Exception as exc:
            backend_url = cfg["backend"]["url"]
            logger.error(
                '{"event":"backend_connection_error","url":"%s","error":"%s"}',
                backend_url,
                str(exc),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": (
                            f"Nexus-Context Proxy Error: Could not connect to backend SLM server "
                            f"at '{backend_url}'. Please ensure your vLLM, SGLang, or Ollama server "
                            f"is running and accessible."
                        ),
                        "type": "backend_connection_error",
                        "param": None,
                        "code": 503,
                    }
                },
            )

    @app.get("/nexus/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "sessions": len(app.state.sessions),
                "backend": app.state.config["backend"]["url"],
            }
        )

    @app.get("/nexus/session/{session_id}/stats")
    async def session_stats(session_id: str) -> JSONResponse:
        session = app.state.sessions.get(session_id)
        if session is None:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse(
            {
                "session_id": session_id,
                "turn_counter": session.turn_counter,
                "zone_p_hash": session.zone_p_hash,
                "memory_pool_size": len(session.memory_pool) if session.memory_pool else 0,
            }
        )

    @app.delete("/nexus/session/{session_id}")
    async def clear_session(session_id: str) -> JSONResponse:
        if session_id in app.state.sessions:
            del app.state.sessions[session_id]
        return JSONResponse({"status": "cleared", "session_id": session_id})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_session_id(request: Request) -> str:
    """Generate a deterministic session ID from request metadata."""
    import hashlib

    fingerprint = (
        request.client.host if request.client else "unknown"
    ) + request.headers.get("user-agent", "")
    return "s-" + hashlib.md5(fingerprint.encode(), usedforsecurity=False).hexdigest()[:12]  # noqa: S324


def _parse_messages(raw: list[dict[str, str]], base_turn: int) -> list[ChatMessage]:
    """Convert raw dicts to :class:`ChatMessage` with turn indices assigned."""
    messages: list[ChatMessage] = []
    for i, m in enumerate(raw):
        content = m.get("content", "")
        messages.append(
            ChatMessage(
                role=m.get("role", "user"),
                content=content,
                token_count=max(1, len(content) // 4),  # rough estimate
                turn_index=base_turn + i,
                name=m.get("name"),
            )
        )
    return messages


async def _json_response(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    pipeline_ms: float,
) -> Response:
    """Forward body to backend and return JSON response."""
    resp = await client.post("/v1/chat/completions", json=body)
    headers = {
        "X-Nexus-Context-Stats": f"pipeline_ms={pipeline_ms:.1f}",
        "Content-Type": "application/json",
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=headers,
    )


async def _stream_response(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    pipeline_ms: float,
) -> StreamingResponse:
    """Proxy SSE chunks from backend to client transparently."""

    async def _gen() -> AsyncIterator[bytes]:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            async for chunk in r.aiter_bytes():
                yield chunk

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"X-Nexus-Context-Stats": f"pipeline_ms={pipeline_ms:.1f}"},
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``nexus-serve`` console script."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Nexus-Context Middleware Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9000, help="Bind port")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend SLM server URL",
    )
    parser.add_argument(
        "--backend-type",
        choices=["vllm", "sglang", "ollama"],
        default="vllm",
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--total-budget", type=int, default=4096)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    cfg = _DEFAULT_CONFIG.copy()
    cfg["backend"]["url"] = args.backend_url
    cfg["backend"]["type"] = args.backend_type
    cfg["cache"]["block_size"] = args.block_size
    cfg["server"]["total_budget"] = args.total_budget

    logging.basicConfig(level=args.log_level.upper())

    app = create_app(config=cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
