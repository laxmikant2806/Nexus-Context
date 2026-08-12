"""
tests/test_hybrid_search.py
============================
Unit tests for context_nexus.hybrid_search.
"""

from __future__ import annotations

from context_nexus.hybrid_search import (
    compute_cosine_distances,
    fast_chunk_text,
    rrf_fusion,
    traverse_graph,
)


def test_compute_cosine_distances() -> None:
    q = [1.0, 0.0, 0.0]
    docs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    dists = compute_cosine_distances(q, docs)
    assert len(dists) == 2
    assert abs(dists[0] - 0.0) < 1e-4
    assert abs(dists[1] - 1.0) < 1e-4


def test_traverse_graph() -> None:
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]
    traversed = traverse_graph(nodes, edges, "A", depth=2)
    assert traversed == ["A", "B", "C"]


def test_rrf_fusion() -> None:
    v_ranks = ["doc1", "doc2"]
    g_ranks = ["doc2", "doc3"]
    fused = rrf_fusion(v_ranks, g_ranks, k=60)
    assert fused[0][0] == "doc2"


def test_fast_chunk_text() -> None:
    text = "a b c d e f g h i j"
    chunks = fast_chunk_text(text, chunk_size=4, overlap=1)
    assert len(chunks) >= 2
