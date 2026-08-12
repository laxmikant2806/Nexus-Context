"""
tests/test_adaptive_chunking.py
================================
Unit tests for Feature 3: Self-Healing Adaptive Chunking via Semantic Entropy
Boundary Detection.

Covers:
    - Cosine shift gradient computation ΔS = 1 - cos(A, B)
    - Conditional token entropy H(T_i | T_{i-w...i-1})
    - Dynamic boundary thresholding ΔS · H(T_i) > τ_boundary
    - Self-healing syntax protection (code fences ``` and JSON structures)
    - Graph edge generation (Chunk sequence dependency edges k-1 -> k)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from nexus_context.guard.adaptive_chunking import (
    AdaptiveSemanticChunker,
    _compute_conditional_entropy,
    _compute_cosine_shift,
)
from nexus_context.guard.schemas import (
    ContextGraph,
    EdgeType,
    NodeType,
    SemanticChunk,
)


# ---------------------------------------------------------------------------
# Cosine Shift Gradient Tests
# ---------------------------------------------------------------------------


class TestCosineShift:

    def test_identical_vectors_zero_shift(self) -> None:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        shift = _compute_cosine_shift(vec, vec)
        assert abs(shift - 0.0) < 1e-5

    def test_orthogonal_vectors_unit_shift(self) -> None:
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0], dtype=np.float32)
        shift = _compute_cosine_shift(vec_a, vec_b)
        assert abs(shift - 1.0) < 1e-5

    def test_opposite_vectors_double_shift(self) -> None:
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_b = np.array([-1.0, 0.0], dtype=np.float32)
        shift = _compute_cosine_shift(vec_a, vec_b)
        assert abs(shift - 2.0) < 1e-5

    def test_zero_vector_returns_zero(self) -> None:
        vec_a = np.array([0.0, 0.0], dtype=np.float32)
        vec_b = np.array([1.0, 2.0], dtype=np.float32)
        shift = _compute_cosine_shift(vec_a, vec_b)
        assert shift == 0.0


# ---------------------------------------------------------------------------
# Conditional Token Entropy Tests
# ---------------------------------------------------------------------------


class TestConditionalEntropy:

    def test_single_repeated_token_zero_entropy(self) -> None:
        tokens = ["database", "database", "database", "database"]
        entropy = _compute_conditional_entropy(tokens)
        assert abs(entropy - 0.0) < 1e-5

    def test_uniform_distribution_entropy(self) -> None:
        tokens = ["a", "b", "c", "d"]  # 4 unique tokens -> H = log2(4) = 2.0
        entropy = _compute_conditional_entropy(tokens)
        assert abs(entropy - 2.0) < 1e-5

    def test_empty_tokens_zero_entropy(self) -> None:
        assert _compute_conditional_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# AdaptiveSemanticChunker Boundary & Self-Healing Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSemanticChunker:

    @pytest.fixture
    def chunker(self) -> AdaptiveSemanticChunker:
        return AdaptiveSemanticChunker(
            window_size=8,
            tau_boundary=0.30,
            min_chunk_tokens=8,
            max_chunk_tokens=256,
            syntax_protection=True,
        )

    def test_short_text_single_chunk(self, chunker: AdaptiveSemanticChunker) -> None:
        text = "Short prompt that fits inside minimum chunk tokens."
        chunks = chunker.chunk_text(text, turn_index=0)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_semantic_transition_creates_multiple_chunks(
        self, chunker: AdaptiveSemanticChunker
    ) -> None:
        # Create a text with a distinct topic shift between Database setup and Machine Learning
        db_text = " ".join(["postgresql", "database", "connection", "port", "sql", "query"] * 10)
        ml_text = " ".join(["neural", "network", "tensor", "backpropagation", "loss", "optimizer"] * 10)
        full_text = f"{db_text}\n{ml_text}"

        chunks = chunker.chunk_text(full_text, turn_index=1)
        assert len(chunks) >= 2, f"Expected multiple chunks on topic shift, got {len(chunks)}"

    def test_syntax_protection_prevents_bisecting_code_block(
        self, chunker: AdaptiveSemanticChunker
    ) -> None:
        """Code block inside ``` must not be split across chunks."""
        code_block = (
            "```python\n"
            "def connect_database():\n"
            "    host = 'prod.db.internal'\n"
            "    port = 5432\n"
            "    return psycopg2.connect(host=host, port=port)\n"
            "```"
        )
        prose_before = "Here is the database setup code block:"
        prose_after = "Call connect_database to initialize the connection."
        full_text = f"{prose_before}\n{code_block}\n{prose_after}"

        chunks = chunker.chunk_text(full_text, turn_index=2)

        # Verify that code fence ``` is completely enclosed in a chunk
        for chunk in chunks:
            if "```python" in chunk.content:
                assert "```" in chunk.content[chunk.content.find("```python") + 9 :], (
                    "Code block was bisected across chunk boundary!"
                )

    def test_graph_population_with_chunk_sequence_edges(
        self, chunker: AdaptiveSemanticChunker
    ) -> None:
        text = " ".join(["word" + str(i % 10) for i in range(200)])
        graph = ContextGraph(session_id="s_test")

        chunks = chunker.chunk_text(text, turn_index=3, graph=graph)
        assert len(chunks) >= 2

        # Verify graph contains chunk nodes and CHUNK_SEQUENCE edges
        chunk_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.ADAPTIVE_CHUNK]
        assert len(chunk_nodes) == len(chunks)

        seq_edges = [e for e in graph.edges if e.edge_type == EdgeType.CHUNK_SEQUENCE]
        assert len(seq_edges) == len(chunks) - 1
        assert seq_edges[0].source_id == chunks[0].chunk_id
        assert seq_edges[0].target_id == chunks[1].chunk_id
