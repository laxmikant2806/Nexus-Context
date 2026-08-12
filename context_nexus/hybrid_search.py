"""
context_nexus.hybrid_search
============================
Hybrid Search & RRF Dispatcher for ContextNexus.
Combines vector distance scoring and graph reachability traversal via Reciprocal Rank Fusion (RRF).
Invokes native Rust core functions when available; automatically falls back to pure-Python
routines if NEXUS_DISABLE_RUST=1 or native binary is missing.
"""

from __future__ import annotations

import os
import time
from typing import Any

from context_nexus.fallback import (
    compute_cosine_distances_fallback,
    fast_chunk_text_fallback,
    rrf_fusion_fallback,
    traverse_graph_fallback,
)

_DISABLE_RUST = os.getenv("NEXUS_DISABLE_RUST", "0").lower() in ("1", "true", "yes")

_RUST_CORE: Any = None
if not _DISABLE_RUST:
    try:
        import nexus_core  # type: ignore[import-not-found]
        _RUST_CORE = nexus_core
    except ImportError:
        _RUST_CORE = None


def is_rust_available() -> bool:
    """Return True if native Rust extension module is loaded and enabled."""
    return _RUST_CORE is not None and not _DISABLE_RUST


def compute_cosine_distances(query_vec: list[float], doc_vecs: list[list[float]]) -> list[float]:
    """Compute batch cosine distances using Rust SIMD if available, else Python fallback."""
    if is_rust_available() and hasattr(_RUST_CORE, "py_compute_cosine_distances"):
        try:
            return list(_RUST_CORE.py_compute_cosine_distances(query_vec, doc_vecs))
        except Exception:
            pass
    return compute_cosine_distances_fallback(query_vec, doc_vecs)


def traverse_graph(nodes: list[str], edges: list[tuple[str, str]], start_node: str, depth: int) -> list[str]:
    """Traverse graph using BFS in Rust if available, else Python fallback."""
    if is_rust_available() and hasattr(_RUST_CORE, "py_traverse_graph"):
        try:
            return list(_RUST_CORE.py_traverse_graph(nodes, edges, start_node, depth))
        except Exception:
            pass
    return traverse_graph_fallback(nodes, edges, start_node, depth)


def rrf_fusion(vector_ranks: list[str], graph_ranks: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Perform Reciprocal Rank Fusion using Rust engine if available, else Python fallback."""
    if is_rust_available() and hasattr(_RUST_CORE, "py_rrf_fusion"):
        try:
            return list(_RUST_CORE.py_rrf_fusion(vector_ranks, graph_ranks, k))
        except Exception:
            pass
    return rrf_fusion_fallback(vector_ranks, graph_ranks, k)


def fast_chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """Chunk text using Rust engine if available, else Python fallback."""
    if is_rust_available() and hasattr(_RUST_CORE, "py_fast_chunk_text"):
        try:
            return list(_RUST_CORE.py_fast_chunk_text(text, chunk_size, overlap))
        except Exception:
            pass
    return fast_chunk_text_fallback(text, chunk_size, overlap)
