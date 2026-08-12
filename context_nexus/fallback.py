"""
context_nexus.fallback
======================
Pure-Python fallback implementation of core vector, graph, RRF, and text chunking
routines. Used when native Rust extension binaries (crates/nexus-core) are not
compiled or when NEXUS_DISABLE_RUST=1 environment variable is set.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np


def compute_cosine_distances_fallback(
    query_vec: list[float], doc_vecs: list[list[float]]
) -> list[float]:
    """Pure-Python fallback for batch cosine distance computation."""
    if not query_vec or not doc_vecs:
        return [1.0] * len(doc_vecs)

    q = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [1.0] * len(doc_vecs)
    q_unit = q / q_norm

    distances: list[float] = []
    for doc in doc_vecs:
        if len(doc) != len(query_vec):
            distances.append(1.0)
            continue
        d = np.array(doc, dtype=np.float32)
        d_norm = np.linalg.norm(d)
        if d_norm == 0:
            distances.append(1.0)
        else:
            sim = float(np.dot(q_unit, d / d_norm))
            sim = max(-1.0, min(1.0, sim))
            distances.append(max(0.0, 1.0 - sim))
    return distances


def traverse_graph_fallback(
    nodes: list[str],
    edges: list[tuple[str, str]],
    start_node: str,
    depth: int,
) -> list[str]:
    """Pure-Python BFS graph traversal fallback."""
    adj: dict[str, list[str]] = defaultdict(list)
    for src, tgt in edges:
        adj[src].append(tgt)

    visited: set[str] = {start_node}
    result: list[str] = [start_node]
    queue: deque[tuple[str, int]] = deque([(start_node, 0)])

    while queue:
        curr, cur_depth = queue.popleft()
        if cur_depth >= depth:
            continue
        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                result.append(neighbor)
                queue.append((neighbor, cur_depth + 1))

    return result


def rrf_fusion_fallback(
    vector_ranks: list[str],
    graph_ranks: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Pure-Python Reciprocal Rank Fusion (RRF) fallback."""
    scores: dict[str, float] = defaultdict(float)

    for rank_idx, item_id in enumerate(vector_ranks):
        rank = rank_idx + 1
        scores[item_id] += 1.0 / (k + rank)

    for rank_idx, item_id in enumerate(graph_ranks):
        rank = rank_idx + 1
        scores[item_id] += 1.0 / (k + rank)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def fast_chunk_text_fallback(
    text: str, chunk_size: int = 256, overlap: int = 32
) -> list[str]:
    """Pure-Python sliding window text chunker fallback."""
    words = text.split()
    if not words or chunk_size <= 0:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []

    i = 0
    while i < len(words):
        end = min(len(words), i + chunk_size)
        chunks.append(" ".join(words[i:end]))
        if end >= len(words):
            break
        i += step

    return chunks
