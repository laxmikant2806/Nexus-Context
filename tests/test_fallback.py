"""
tests/test_fallback.py
=======================
Unit tests for context_nexus.fallback pure Python routines.
"""

from __future__ import annotations

from context_nexus.fallback import (
    compute_cosine_distances_fallback,
    fast_chunk_text_fallback,
    rrf_fusion_fallback,
    traverse_graph_fallback,
)


def test_fallback_cosine_distances() -> None:
    q = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.0, 1.0]]
    dists = compute_cosine_distances_fallback(q, docs)
    assert len(dists) == 2
    assert abs(dists[0] - 0.0) < 1e-5
    assert abs(dists[1] - 1.0) < 1e-5


def test_fallback_traverse_graph() -> None:
    nodes = ["1", "2", "3"]
    edges = [("1", "2"), ("2", "3")]
    res = traverse_graph_fallback(nodes, edges, "1", depth=2)
    assert res == ["1", "2", "3"]


def test_fallback_rrf_fusion() -> None:
    v_ranks = ["a", "b"]
    g_ranks = ["b", "c"]
    fused = rrf_fusion_fallback(v_ranks, g_ranks, k=60)
    assert fused[0][0] == "b"


def test_fallback_fast_chunk_text() -> None:
    text = "word1 word2 word3 word4 word5 word6"
    chunks = fast_chunk_text_fallback(text, chunk_size=3, overlap=1)
    assert len(chunks) >= 2
