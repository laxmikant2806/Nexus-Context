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
from nexus_context.cache.tool_compressor import ToolCallCompressor  # Feature D
from nexus_context.guard.ast_graph import ContextGraphBuilder
from nexus_context.guard.submodular import SubmodularSolver
from nexus_context.memory.decay import MemoryPool
from nexus_context.memory.ltkb import LongTermKnowledgeBase  # Feature I
from nexus_context.memory.www_parser import WWWParser
from nexus_context.persistence.session_store import SessionStore  # Feature B
from nexus_context.telemetry import (  # Feature C
    TelemetryBus,
    emit_session_created,
    emit_session_restored,
    emit_tool_call_intercepted,
    emit_turn_processed,
)

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
    "persistence": {  # Feature B
        "enabled": False,
        "db_path": "nexus_sessions.db",
        "persist_every": 5,
    },
    "ltkb": {  # Feature I
        "enabled": True,
        "db_path": "nexus_ltkb.db",
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
        "graph",          # Feature A: most recent ContextGraph for LTKB extraction
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.aligner: BlockAligner | None = None
        self.segmenter: ZoneSegmenter | None = None
        self.memory_pool: MemoryPool | None = None
        self.aligned_zone_p: str | None = None
        self.zone_p_hash: str | None = None
        self.turn_counter: int = 0
        self.graph: Any = None  # Feature A


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared subsystems on startup; clean up on shutdown."""
    cfg = app.state.config  # type: ignore[attr-defined]

    # Feature C: Telemetry bus (must be first — others depend on it)
    app.state.telemetry = TelemetryBus(max_history=500)

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

    # Feature D: Shared tool call compressor
    app.state.tool_compressor = ToolCallCompressor(budget_tokens=512)

    # Feature B: Session persistence store
    persist_cfg = cfg.get("persistence", {})
    if persist_cfg.get("enabled", False):
        store = SessionStore(db_path=persist_cfg.get("db_path", "nexus_sessions.db"))
        await store.initialize()
        app.state.session_store: SessionStore | None = store
        # Restore all persisted sessions on startup
        for sid in await store.list_sessions():
            data = await store.load_session(sid)
            if data:
                restored = _SessionState(sid)
                restored.turn_counter = data["turn_counter"]
                restored.zone_p_hash = data["zone_p_hash"]
                restored.aligned_zone_p = data["aligned_zone_p"]
                app.state.sessions[sid] = restored
                emit_session_restored(app.state.telemetry, sid, data["turn_counter"])
                logger.info('{"event":"session_restored","session_id":"%s"}', sid)
    else:
        app.state.session_store = None

    # Feature I: Long-Term Knowledge Base
    ltkb_cfg = cfg.get("ltkb", {})
    if ltkb_cfg.get("enabled", True):
        ltkb = LongTermKnowledgeBase(db_path=ltkb_cfg.get("db_path", "nexus_ltkb.db"))
        await ltkb.initialize()
        app.state.ltkb: LongTermKnowledgeBase | None = ltkb
    else:
        app.state.ltkb = None

    logger.info(
        '{"event":"middleware_startup","backend":"%s","type":"%s"}',
        cfg["backend"]["url"],
        cfg["backend"]["type"],
    )
    yield

    # Shutdown: persist LTKB facts for all active sessions
    ltkb_inst: LongTermKnowledgeBase | None = getattr(app.state, "ltkb", None)
    if ltkb_inst is not None:
        for sid, sess in app.state.sessions.items():
            if sess.graph is not None:
                await ltkb_inst.extract_and_persist(
                    sess.graph, sid, sess.turn_counter,
                    telemetry_bus=app.state.telemetry,
                )

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
        version="0.2.0",
        lifespan=_lifespan,
    )
    app.state.config = config or _DEFAULT_CONFIG
    _register_routes(app)
    return app


# Public alias used by the top-level nexus_context.create_app factory
_build_app = create_app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:

    # Feature C: Mount the real-time dashboard router
    from nexus_context.dashboard.router import router as dashboard_router
    app.include_router(dashboard_router)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        session_id = request.headers.get("X-Session-ID") or _derive_session_id(request)
        cfg: dict[str, Any] = app.state.config
        t_start = time.perf_counter()
        telemetry: TelemetryBus = app.state.telemetry

        # --------------------------------------------------------
        # Session initialisation (first call only)
        # --------------------------------------------------------
        session = app.state.sessions.get(session_id)
        if session is None:
            session = _SessionState(session_id)
            app.state.sessions[session_id] = session
            emit_session_created(telemetry, session_id)

            # Feature I: Inject LTKB facts into Zone P for new sessions
            ltkb_inst: LongTermKnowledgeBase | None = getattr(app.state, "ltkb", None)
            if ltkb_inst is not None:
                raw_msgs = body.get("messages", [])
                query_hint = next(
                    (m.get("content", "") for m in raw_msgs if m.get("role") == "user"), ""
                )
                ltkb_facts = await ltkb_inst.get_relevant_facts(query_hint, top_k=5)
                if ltkb_facts:
                    annotation = await ltkb_inst.inject_into_zone_p(ltkb_facts)
                    # Append LTKB annotation to system message if present
                    for i, m in enumerate(body.get("messages", [])):
                        if m.get("role") == "system":
                            body["messages"][i]["content"] += f"\n{annotation}"
                            break

        # --------------------------------------------------------
        # Parse messages
        # --------------------------------------------------------
        raw_messages: list[dict[str, str]] = body.get("messages", [])

        # Feature D: Tool call interception — compress oversized tool returns
        compressor: ToolCallCompressor = app.state.tool_compressor
        for i, msg_dict in enumerate(raw_messages):
            if msg_dict.get("role") == "tool":
                content = msg_dict.get("content", "")
                original_tokens = compressor.estimate_tokens(content)
                if original_tokens > 512:
                    compressed, orig_tok = compressor.compress(content)
                    raw_messages[i] = {**msg_dict, "content": compressed}
                    emit_tool_call_intercepted(
                        telemetry, session_id,
                        original_tokens=orig_tok,
                        compressed_tokens=compressor.estimate_tokens(compressed),
                    )

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
                session.graph = graph  # Feature A: store for LTKB extraction
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

        # Feature C: Emit turn_processed telemetry for dashboard
        pipeline_ms_pre = (time.perf_counter() - t_start) * 1000
        _graph = getattr(session, "graph", None)
        emit_turn_processed(
            telemetry,
            session_id=session_id,
            pipeline_ms=pipeline_ms_pre,
            tokens_in=sum(len(m.get("content", "")) // 4 for m in transformed_messages),
            tokens_out=0,
            zone_p_hit=session.zone_p_hash is not None,
            compaction_applied=compaction is not None,
            memory_tuples_count=len(session.memory_pool) if session.memory_pool else 0,
            graph_node_count=len(_graph.nodes) if _graph and hasattr(_graph, "nodes") else 0,
            graph_edge_count=len(_graph.edges) if _graph and hasattr(_graph, "edges") else 0,
            turn_counter=session.turn_counter,
        )

        # Feature B: Persist session every N turns
        persist_cfg = cfg.get("persistence", {})
        store: SessionStore | None = getattr(app.state, "session_store", None)
        if store is not None and persist_cfg.get("enabled", False):
            every = persist_cfg.get("persist_every", 5)
            if session.turn_counter % every == 0:
                memory_tuples_json: list[str] = []
                if session.memory_pool is not None:
                    try:
                        memory_tuples_json = [
                            t.model_dump_json()
                            for t in list(session.memory_pool._pool.values())[:200]
                        ]
                    except Exception:
                        pass
                await store.save_session(
                    session_id=session_id,
                    turn_counter=session.turn_counter,
                    zone_p_hash=session.zone_p_hash,
                    aligned_zone_p=session.aligned_zone_p,
                    memory_tuples_json=memory_tuples_json,
                )

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
        session = app.state.sessions.get(session_id)
        if session is not None:
            # Feature I: Persist LTKB facts before clearing
            ltkb_inst: LongTermKnowledgeBase | None = getattr(app.state, "ltkb", None)
            if ltkb_inst is not None and session.graph is not None:
                await ltkb_inst.extract_and_persist(
                    session.graph, session_id, session.turn_counter,
                    telemetry_bus=getattr(app.state, "telemetry", None),
                )
            # Feature B: Remove from persistent store
            store: SessionStore | None = getattr(app.state, "session_store", None)
            if store is not None:
                await store.delete_session(session_id)
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
    # Feature B: Session persistence flags
    parser.add_argument("--persist", action="store_true", help="Enable SQLite session persistence")
    parser.add_argument("--db-path", default="nexus_sessions.db", help="Path to sessions SQLite DB")
    parser.add_argument("--persist-every", type=int, default=5, help="Save session every N turns")
    # Feature I: LTKB flags
    parser.add_argument("--ltkb-db", default="nexus_ltkb.db", help="Path to LTKB SQLite DB")
    parser.add_argument("--no-ltkb", action="store_true", help="Disable long-term knowledge base")
    args = parser.parse_args()

    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    cfg["backend"]["url"] = args.backend_url
    cfg["backend"]["type"] = args.backend_type
    cfg["cache"]["block_size"] = args.block_size
    cfg["server"]["total_budget"] = args.total_budget
    cfg["persistence"]["enabled"] = args.persist
    cfg["persistence"]["db_path"] = args.db_path
    cfg["persistence"]["persist_every"] = args.persist_every
    cfg["ltkb"]["enabled"] = not args.no_ltkb
    cfg["ltkb"]["db_path"] = args.ltkb_db

    logging.basicConfig(level=args.log_level.upper())

    app = create_app(config=cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
