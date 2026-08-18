"""
nexus-context
=============
Transparent middleware for referential integrity, KV-cache alignment,
and WWW memory governance in local SLM deployments (vLLM, SGLang, Ollama).

Quick-start
-----------
Start the proxy server::

    nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000

Then point any OpenAI-compatible client at port 9000 instead of your LLM backend.
Open the real-time dashboard at http://localhost:9000/dashboard

Subpackages
-----------
nexus_context.cache       – PagedAttention block alignment + Zone P/T/R segmentation
nexus_context.guard       – AST dependency graph + submodular compaction solver
nexus_context.memory      – WWW episodic-to-semantic memory decay + LTKB
nexus_context.persistence – SQLite-backed session crash recovery
nexus_context.dashboard   – Real-time observability SSE dashboard
nexus_context.telemetry   – Async event bus for live metrics
"""

from __future__ import annotations

__version__ = "0.2.1"
__author__ = "Laxmikant Bhagat"
__email__ = "laxmikant2806@gmail.com"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Top-level error hierarchy (see docs/implementation_stage.md §7.3)
# ---------------------------------------------------------------------------


class NexusContextError(Exception):
    """Base exception for all nexus-context errors."""


class TokenizerError(NexusContextError):
    """Tokenizer load or encode failure."""


class AlignmentError(NexusContextError):
    """Block alignment computation failure."""


class SegmentationError(NexusContextError):
    """Zone segmentation invariant violation."""


class GraphBuildError(NexusContextError):
    """Context dependency graph construction failure."""


class SolverError(NexusContextError):
    """Submodular solver failure (budget too small, empty graph, etc.)."""


class MemoryExtractionError(NexusContextError):
    """WWW tuple extraction failure."""


class BackendProxyError(NexusContextError):
    """Backend server communication failure."""


class ConfigurationError(NexusContextError):
    """Invalid or missing configuration value."""


def create_app(
    backend_url: str = "http://localhost:8000",
    backend_type: str = "vllm",
    block_size: int = 16,
    total_budget: int = 4096,
    persist: bool = False,
    db_path: str = "nexus_sessions.db",
    ltkb_db: str = "nexus_ltkb.db",
) -> "Any":
    """Convenience factory: create and return a configured FastAPI application.

    Parameters
    ----------
    backend_url:
        URL of the backend SLM server (vLLM, SGLang, or Ollama).
    backend_type:
        One of ``"vllm"``, ``"sglang"``, or ``"ollama"``.
    block_size:
        KV-cache block size in tokens (16 or 32). Must match your backend config.
    total_budget:
        Maximum total token budget for the context window.
    persist:
        Enable SQLite-backed session persistence for crash recovery.
    db_path:
        File path for the sessions SQLite database (if ``persist=True``).
    ltkb_db:
        File path for the long-term knowledge base SQLite database.

    Returns
    -------
    fastapi.FastAPI
        A fully configured Nexus-Context FastAPI application ready to serve.

    Example
    -------
    ::

        import uvicorn
        from nexus_context import create_app

        app = create_app(
            backend_url="http://localhost:11434",
            backend_type="ollama",
            total_budget=8192,
            persist=True,
        )
        uvicorn.run(app, host="0.0.0.0", port=9000)
    """
    import copy
    from nexus_context.cache.middleware import _DEFAULT_CONFIG, _build_app  # type: ignore[attr-defined]

    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    cfg["backend"]["url"] = backend_url
    cfg["backend"]["type"] = backend_type
    cfg["cache"]["block_size"] = block_size
    cfg["server"]["total_budget"] = total_budget
    cfg["persistence"]["enabled"] = persist
    cfg["persistence"]["db_path"] = db_path
    cfg["ltkb"]["db_path"] = ltkb_db
    return _build_app(cfg)


__all__ = [
    "__version__",
    # Convenience factory
    "create_app",
    # Error hierarchy
    "NexusContextError",
    "TokenizerError",
    "AlignmentError",
    "SegmentationError",
    "GraphBuildError",
    "SolverError",
    "MemoryExtractionError",
    "BackendProxyError",
    "ConfigurationError",
]
