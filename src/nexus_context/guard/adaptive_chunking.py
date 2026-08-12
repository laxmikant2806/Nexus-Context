"""
nexus_context.guard.adaptive_chunking
======================================
Feature 3: Self-Healing Adaptive Chunking via Semantic Entropy Boundary Detection.

Standard text chunking relies on fixed character counts or rigid token windows,
frequently bisecting logical units, code block definitions, or JSON structures.

This module evaluates streaming token sequences in real time, calculating:
1. Directional semantic shift gradient (ΔS = 1 - cos(E(A), E(B)))
2. Local conditional token entropy (H(T_i | T_{i-w...i-1}))

Dynamic boundary condition:
    ΔS · H(T_i) > τ_boundary

Includes syntax-aware self-healing protection to delay chunking when inside
unclosed code fences (```), multi-line string literals, or JSON structures.

Reference: Feature 3 Spec
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np

from nexus_context.guard.schemas import (
    BoundaryEvaluationResult,
    ContextGraph,
    ContextNode,
    DependencyEdge,
    EdgeType,
    NodeType,
    SemanticChunk,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorization & Entropy Helpers
# ---------------------------------------------------------------------------


def _token_vector(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    """Return a normalized term-frequency embedding vector for *tokens*."""
    dim = max(len(vocab), 1)
    vec = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        idx = vocab.get(t.lower())
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _compute_cosine_shift(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute directional semantic shift gradient ΔS = 1 - cos(A, B)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    shift = 1.0 - cos_sim
    return max(0.0, shift)


def _compute_conditional_entropy(tokens: list[str]) -> float:
    """Compute Shannon conditional entropy H(T_i | T_{i-w...i-1}) over *tokens*.

    H = - ∑ p(x) log2(p(x))
    """
    if not tokens:
        return 0.0
    n = len(tokens)
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)
    return max(0.0, entropy)


# ---------------------------------------------------------------------------
# Syntax depth tracker for self-healing chunk boundaries
# ---------------------------------------------------------------------------


class _SyntaxTracker:
    """Tracks code fences, JSON brackets, and string literals to prevent bisecting syntax."""

    def __init__(self) -> None:
        self.code_fence_open: bool = False
        self.curly_depth: int = 0
        self.square_depth: int = 0
        self.paren_depth: int = 0
        self.in_string: bool = False
        self._string_char: str | None = None

    def update(self, token: str) -> None:
        """Scan *token* and update open/close syntax depth."""
        # Code fence ```
        if "```" in token:
            self.code_fence_open = not self.code_fence_open

        # String literals (basic tracking)
        for char in token:
            if char in ('"', "'") and not self.code_fence_open:
                if not self.in_string:
                    self.in_string = True
                    self._string_char = char
                elif self._string_char == char:
                    self.in_string = False
                    self._string_char = None

            if not self.in_string and not self.code_fence_open:
                if char == "{":
                    self.curly_depth += 1
                elif char == "}" and self.curly_depth > 0:
                    self.curly_depth -= 1
                elif char == "[":
                    self.square_depth += 1
                elif char == "]" and self.square_depth > 0:
                    self.square_depth -= 1
                elif char == "(":
                    self.paren_depth += 1
                elif char == ")" and self.paren_depth > 0:
                    self.paren_depth -= 1

    @property
    def is_syntax_locked(self) -> bool:
        """True if inside an unclosed code block, string, or nested JSON structure."""
        return (
            self.code_fence_open
            or self.in_string
            or self.curly_depth > 0
            or self.square_depth > 0
        )


# ---------------------------------------------------------------------------
# AdaptiveSemanticChunker
# ---------------------------------------------------------------------------


class AdaptiveSemanticChunker:
    """Self-Healing Adaptive Chunking via Semantic Entropy Boundary Detection.

    Parameters
    ----------
    window_size:
        Sliding window size *w* in tokens for sequence partitioning (default 16).
    tau_boundary:
        Baseline semantic boundary threshold τ_boundary (default 0.35).
    adaptive_threshold:
        If True, dynamically updates threshold based on sliding mean + k*std.
    min_chunk_tokens:
        Minimum number of tokens required in a chunk before a split can occur.
    max_chunk_tokens:
        Hard safety limit; forces a split even if syntax is locked.
    syntax_protection:
        If True, delays boundary splits that fall inside unclosed code blocks/JSON.
    """

    def __init__(
        self,
        window_size: int = 16,
        tau_boundary: float = 0.35,
        adaptive_threshold: bool = True,
        min_chunk_tokens: int = 16,
        max_chunk_tokens: int = 512,
        syntax_protection: bool = True,
    ) -> None:
        self.window_size = max(4, window_size)
        self.tau_boundary = tau_boundary
        self.adaptive_threshold = adaptive_threshold
        self.min_chunk_tokens = max(1, min_chunk_tokens)
        self.max_chunk_tokens = max_chunk_tokens
        self.syntax_protection = syntax_protection

        # Native Rust engine hook (PyO3)
        self._rust_engine: object | None = self._load_rust_engine()

        logger.info(
            '{"event":"adaptive_chunker_init","window":%d,"tau":%.2f,"adaptive":%s}',
            window_size,
            tau_boundary,
            str(adaptive_threshold).lower(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_boundary(
        self,
        window_a: list[str],
        window_b: list[str],
        vocab: dict[str, int],
        syntax_locked: bool = False,
        token_index: int = 0,
        historical_scores: list[float] | None = None,
    ) -> BoundaryEvaluationResult:
        """Evaluate continuous cosine distance gradient ΔS and entropy H(T_i).

        Parameters
        ----------
        window_a:
            Left token sequence A = T[i - 2w : i - w].
        window_b:
            Right token sequence B = T[i - w : i].
        vocab:
            Vocabulary map for term-frequency vectorization.
        syntax_locked:
            True if current position is inside an unclosed code block or JSON.
        token_index:
            Stream position index for metadata.
        historical_scores:
            Optional score history for dynamic quantile thresholding.

        Returns
        -------
        BoundaryEvaluationResult
        """
        vec_a = _token_vector(window_a, vocab)
        vec_b = _token_vector(window_b, vocab)

        shift = _compute_cosine_shift(vec_a, vec_b)
        entropy = _compute_conditional_entropy(window_b)
        score = shift * entropy

        # Determine dynamic threshold
        threshold = self.tau_boundary
        if self.adaptive_threshold and historical_scores and len(historical_scores) >= 5:
            arr = np.array(historical_scores[-20:], dtype=np.float32)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            threshold = max(0.1, mean + 1.2 * std)

        crosses_threshold = score > threshold
        suppressed = False
        is_boundary = False

        if crosses_threshold:
            if self.syntax_protection and syntax_locked:
                suppressed = True
                is_boundary = False
            else:
                is_boundary = True

        return BoundaryEvaluationResult(
            token_index=token_index,
            cosine_shift=shift,
            token_entropy=entropy,
            boundary_score=score,
            threshold=threshold,
            is_boundary=is_boundary,
            suppressed_by_syntax=suppressed,
        )

    def chunk_text(
        self,
        text: str,
        turn_index: int = 0,
        graph: ContextGraph | None = None,
    ) -> list[SemanticChunk]:
        """Process *text* into self-healing adaptive semantic chunks.

        Runs real-time sliding window evaluation over token stream, calculates
        boundary scores ΔS · H(T_i), splits chunks at natural semantic transitions,
        and constructs sequence dependency edges in *graph* if provided.

        Parameters
        ----------
        text:
            Raw input text (code transcript, tool return, or prose).
        turn_index:
            Originating turn index.
        graph:
            Optional :class:`ContextGraph` to populate with chunk nodes and edges.

        Returns
        -------
        list[SemanticChunk]
        """
        if not text.strip():
            return []

        # Tokenize by whitespace and punctuation for streaming stream
        tokens = re.findall(r"\S+|\n", text)
        n_tokens = len(tokens)

        if n_tokens <= self.min_chunk_tokens:
            chunk = SemanticChunk(
                chunk_id=f"chunk:{turn_index}:0-{n_tokens}",
                turn_index=turn_index,
                start_token=0,
                end_token=n_tokens,
                content=text,
                token_count=n_tokens,
                contains_code_block="```" in text,
            )
            if graph is not None:
                self._add_chunk_to_graph(chunk, graph)
            return [chunk]

        # Build vocabulary for vector representations
        vocab = {tok.lower(): i for i, tok in enumerate(set(tokens))}

        w = self.window_size
        syntax_tracker = _SyntaxTracker()
        historical_scores: list[float] = []

        chunks: list[SemanticChunk] = []
        chunk_start_idx = 0
        current_chunk_tokens: list[str] = []

        for i, token in enumerate(tokens):
            syntax_tracker.update(token)
            current_chunk_tokens.append(token)
            current_len = len(current_chunk_tokens)

            # Evaluate boundary when sliding window has enough history
            if i >= 2 * w and current_len >= self.min_chunk_tokens:
                window_a = tokens[i - 2 * w : i - w]
                window_b = tokens[i - w : i]

                result = self.evaluate_boundary(
                    window_a=window_a,
                    window_b=window_b,
                    vocab=vocab,
                    syntax_locked=syntax_tracker.is_syntax_locked,
                    token_index=i,
                    historical_scores=historical_scores,
                )
                historical_scores.append(result.boundary_score)

                # Split condition: boundary detected OR max chunk safety limit hit
                if result.is_boundary or current_len >= self.max_chunk_tokens:
                    chunk_content = " ".join(current_chunk_tokens)
                    chunk = SemanticChunk(
                        chunk_id=f"chunk:{turn_index}:{chunk_start_idx}-{i+1}",
                        turn_index=turn_index,
                        start_token=chunk_start_idx,
                        end_token=i + 1,
                        content=chunk_content,
                        token_count=current_len,
                        boundary_score=result.boundary_score,
                        contains_code_block="```" in chunk_content,
                        has_unclosed_scope=syntax_tracker.is_syntax_locked,
                    )
                    chunks.append(chunk)

                    chunk_start_idx = i + 1
                    current_chunk_tokens = []

        # Flush remaining active tokens into final chunk
        if current_chunk_tokens:
            chunk_content = " ".join(current_chunk_tokens)
            chunk = SemanticChunk(
                chunk_id=f"chunk:{turn_index}:{chunk_start_idx}-{n_tokens}",
                turn_index=turn_index,
                start_token=chunk_start_idx,
                end_token=n_tokens,
                content=chunk_content,
                token_count=len(current_chunk_tokens),
                contains_code_block="```" in chunk_content,
            )
            chunks.append(chunk)

        # Populate ContextGraph with chunk nodes and sequence edges
        if graph is not None:
            prev_chunk: SemanticChunk | None = None
            for chk in chunks:
                self._add_chunk_to_graph(chk, graph)
                if prev_chunk is not None:
                    # Relational edge connecting adjacent chunks k-1 -> k
                    edge = DependencyEdge(
                        edge_id=f"{prev_chunk.chunk_id}->{chk.chunk_id}",
                        source_id=prev_chunk.chunk_id,
                        target_id=chk.chunk_id,
                        edge_type=EdgeType.CHUNK_SEQUENCE,
                        weight=0.8,
                    )
                    graph.add_edge(edge)
                prev_chunk = chk

        logger.debug(
            '{"event":"text_chunked","turn_index":%d,"n_chunks":%d,"total_tokens":%d}',
            turn_index,
            len(chunks),
            n_tokens,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _add_chunk_to_graph(self, chunk: SemanticChunk, graph: ContextGraph) -> None:
        """Register *chunk* as a ContextNode in *graph*."""
        node = ContextNode(
            node_id=chunk.chunk_id,
            node_type=NodeType.ADAPTIVE_CHUNK,
            turn_index=chunk.turn_index,
            content=chunk.content,
            token_count=chunk.token_count,
            metadata={
                "start_token": chunk.start_token,
                "end_token": chunk.end_token,
                "boundary_score": chunk.boundary_score,
                "contains_code_block": chunk.contains_code_block,
            },
        )
        graph.add_node(node)

    def _load_rust_engine(self) -> object | None:
        """Try loading PyO3 native Rust parsing engine if compiled."""
        try:
            import nexus_core  # type: ignore[import-not-found]

            logger.info('{"event":"rust_engine_loaded","crate":"nexus-core"}')
            return nexus_core
        except ImportError:
            logger.debug(
                '{"event":"rust_engine_unavailable","fallback":"python_numpy"}'
            )
            return None
