"""
nexus_context.guard.submodular
================================
Budget-constrained greedy submodular maximisation with guaranteed zero
referential dangling.

Objective
---------
    max_{S ⊆ V, tokens(S) ≤ B}  f(S) = α·Relevance(S) + β·Coverage(S) − γ·Dangling(S)

Algorithm
---------
Lazy greedy (Minoux, 1978) with a max-heap of upper-bound gains.  Ancestor
forcing integrates the hard dangling constraint into the selection loop.

Reference: docs/architecture.md §5.2–5.3, docs/decisions.md ADR-003.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from nexus_context import SolverError
from nexus_context.guard.schemas import CompactionResult, ContextGraph, ContextNode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _cosine_similarity_matrix(a: np.ndarray) -> np.ndarray:
    """Return the (n × n) pairwise cosine similarity matrix for row vectors *a*."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    a_norm = a / norms
    return a_norm @ a_norm.T


def _cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (n,) cosine similarity of each row of *a* against vector *b*."""
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    norm_a = np.where(norm_a == 0, 1e-10, norm_a)
    norm_b = max(norm_b, 1e-10)
    return (a @ b) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SubmodularSolver
# ---------------------------------------------------------------------------


class SubmodularSolver:
    """Budget-constrained greedy submodular solver for context compaction.

    Parameters
    ----------
    embedding_model:
        SentenceTransformers model name for computing Relevance scores.
        Falls back to TF-IDF cosine similarity if the model is unavailable.
    alpha:
        Weight on the Relevance term (default 0.5).
    beta:
        Weight on the Coverage term (default 0.5).
    gamma_factor:
        Multiplier to compute the hard-constraint penalty coefficient γ.
        γ = gamma_factor × max_positive_objective (default 1 × 10^6).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma_factor: float = 1e6,
    ) -> None:
        if abs(alpha + beta - 1.0) > 1e-6:
            raise SolverError(
                f"alpha + beta must equal 1.0, got alpha={alpha}, beta={beta}."
            )
        self._alpha = alpha
        self._beta = beta
        self._gamma_factor = gamma_factor
        self._embedding_model_name = embedding_model
        self._embed_model: object | None = self._load_embedding_model(embedding_model)
        self._embedding_cache: dict[str, np.ndarray] = {}

        logger.info(
            '{"event":"submodular_solver_init","model":"%s","alpha":%.2f,"beta":%.2f}',
            embedding_model,
            alpha,
            beta,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        graph: ContextGraph,
        budget: int,
        query: str,
        turn_index: int = 0,
    ) -> CompactionResult:
        """Run the density-based lazy-greedy solver on *graph*.

        Parameters
        ----------
        graph:
            Context dependency graph with transitive closure already computed.
        budget:
            Token budget B_T for Zone T.
        query:
            The current user query / task description, used for Relevance scoring.
        turn_index:
            Current turn index for the CompactionResult metadata.

        Returns
        -------
        CompactionResult
        """
        t_start = time.perf_counter()

        if not graph.transitive_closure_computed:
            logger.warning(
                '{"event":"closure_not_computed","note":"computing now; '
                'call compute_transitive_closure() before solve() for best performance"}'
            )
            graph.compute_transitive_closure()

        nodes_list: list[ContextNode] = sorted(
            graph.nodes.values(), key=lambda n: n.turn_index
        )
        n = len(nodes_list)

        if n == 0:
            return CompactionResult(
                session_id=graph.session_id,
                turn_index=turn_index,
                selected_node_ids=[],
                pruned_node_ids=[],
                token_budget=budget,
                tokens_used=0,
                objective_value=0.0,
                reconstructed_text="",
                latency_ms=0.0,
            )

        idx_map: dict[str, int] = {nd.node_id: i for i, nd in enumerate(nodes_list)}

        # ----------------------------------------------------------------
        # Compute embeddings (encode nodes + query together to guarantee matching dimensions)
        # ----------------------------------------------------------------
        all_texts = [nd.content for nd in nodes_list] + [query[:512]]
        all_vecs = self._raw_encode(all_texts)
        embeddings = all_vecs[:-1]
        query_emb = all_vecs[-1]
        relevance = _cosine_similarity_vec(embeddings, query_emb)  # (n,)
        sim_matrix = _cosine_similarity_matrix(embeddings)        # (n, n)

        # ----------------------------------------------------------------
        # Penalty coefficient γ
        # ----------------------------------------------------------------
        max_positive = float(
            self._alpha * float(np.max(relevance)) + self._beta * float(n)
        )
        gamma = self._gamma_factor * max(max_positive, 1.0)

        # ----------------------------------------------------------------
        # Ancestor map for fast lookup
        # ----------------------------------------------------------------
        # ancestors[i] = set of node indices that must be in S if i is in S
        ancestors: dict[int, set[int]] = {i: set() for i in range(n)}
        for edge in graph.edges:
            src_i = idx_map.get(edge.source_id)
            tgt_i = idx_map.get(edge.target_id)
            if src_i is not None and tgt_i is not None:
                ancestors[tgt_i].add(src_i)

        # ----------------------------------------------------------------
        # Lazy greedy loop
        # ----------------------------------------------------------------
        S: set[int] = set()                         # selected indices
        remaining_budget: int = budget
        current_max_sim: np.ndarray = np.zeros(n)  # coverage tracking
        objective_value: float = 0.0
        forced_inclusions: list[str] = []

        # Initialise heap: (-upper_bound, node_index)
        # Upper bound initialised to Relevance (Coverage gain starts at 0).
        heap: list[tuple[float, int]] = [(-float(relevance[i]), i) for i in range(n)]
        heapq.heapify(heap)

        while heap and remaining_budget > 0:
            neg_ub, v_idx = heapq.heappop(heap)

            if v_idx in S:
                continue

            v_node = nodes_list[v_idx]
            if v_node.token_count > remaining_budget:
                continue

            # Recompute exact marginal gain
            exact_gain = self._marginal_gain(
                v_idx, S, ancestors, relevance, sim_matrix, current_max_sim, gamma, n
            )

            # Lazy greedy: if not still best, re-insert and continue
            if heap and exact_gain < -heap[0][0] - 1e-9:
                heapq.heappush(heap, (-exact_gain, v_idx))
                continue

            if exact_gain < -gamma * 0.9:
                # Infeasible (dangling violation) – skip
                continue

            # --------------------------------------------------------
            # Ancestor forcing: collect all ancestors of v not in S
            # --------------------------------------------------------
            to_add: list[int] = self._force_ancestors(v_idx, S, ancestors, nodes_list)
            to_add.append(v_idx)

            # Check combined token cost
            total_cost = sum(nodes_list[i].token_count for i in to_add)
            if total_cost > remaining_budget:
                # Can't afford v + ancestors together; skip
                continue

            # --------------------------------------------------------
            # Commit selection
            # --------------------------------------------------------
            for idx in sorted(to_add, key=lambda i: nodes_list[i].turn_index):
                if idx in S:
                    continue
                S.add(idx)
                remaining_budget -= nodes_list[idx].token_count
                # Update coverage
                current_max_sim = np.maximum(current_max_sim, sim_matrix[idx])
                if idx != v_idx:
                    forced_inclusions.append(nodes_list[idx].node_id)

            # Update objective
            objective_value += exact_gain

        # ----------------------------------------------------------------
        # Post-selection: verify no dangling remains (invariant check)
        # ----------------------------------------------------------------
        dangling_count = 0
        for edge in graph.edges:
            src_i = idx_map.get(edge.source_id)
            tgt_i = idx_map.get(edge.target_id)
            if tgt_i in S and src_i is not None and src_i not in S:
                dangling_count += 1
                logger.error(
                    '{"event":"dangling_violation_detected","source":"%s","target":"%s"}',
                    edge.source_id,
                    edge.target_id,
                )

        # ----------------------------------------------------------------
        # Reconstruct Zone T text (nodes in turn_index order)
        # ----------------------------------------------------------------
        selected_sorted = sorted(S, key=lambda i: nodes_list[i].turn_index)
        selected_ids = [nodes_list[i].node_id for i in selected_sorted]
        pruned_ids = [
            nodes_list[i].node_id for i in range(n) if i not in S
        ]
        reconstructed_text = "\n\n".join(nodes_list[i].content for i in selected_sorted)
        tokens_used = sum(nodes_list[i].token_count for i in selected_sorted)

        latency_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            '{"event":"compaction_complete","n_selected":%d,"n_pruned":%d,'
            '"tokens_used":%d,"budget":%d,"dangling":%d,"latency_ms":%.2f}',
            len(selected_ids),
            len(pruned_ids),
            tokens_used,
            budget,
            dangling_count,
            latency_ms,
        )

        return CompactionResult(
            session_id=graph.session_id,
            turn_index=turn_index,
            selected_node_ids=selected_ids,
            pruned_node_ids=pruned_ids,
            token_budget=budget,
            tokens_used=tokens_used,
            objective_value=objective_value,
            dangling_violations=dangling_count,
            forced_inclusions=forced_inclusions,
            reconstructed_text=reconstructed_text,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Marginal gain
    # ------------------------------------------------------------------

    def _marginal_gain(
        self,
        v_idx: int,
        S: set[int],
        ancestors: dict[int, set[int]],
        relevance: np.ndarray,
        sim_matrix: np.ndarray,
        current_max_sim: np.ndarray,
        gamma: float,
        n: int,
    ) -> float:
        """Compute Δf(v | S) including hard dangling penalty."""
        # Dangling check: count unresolved ancestors
        unresolved = sum(1 for a in ancestors[v_idx] if a not in S)
        if unresolved > 0:
            return -gamma * unresolved

        rel_gain = self._alpha * float(relevance[v_idx])
        cov_gain = self._beta * float(
            np.sum(np.maximum(0.0, sim_matrix[v_idx] - current_max_sim))
        )
        return rel_gain + cov_gain

    # ------------------------------------------------------------------
    # Ancestor forcing
    # ------------------------------------------------------------------

    def _force_ancestors(
        self,
        v_idx: int,
        S: set[int],
        ancestors: dict[int, set[int]],
        nodes_list: list[ContextNode],
    ) -> list[int]:
        """Return all ancestors of *v_idx* not yet in *S*, in BFS order."""
        needed: list[int] = []
        queue = list(ancestors[v_idx])
        visited: set[int] = set(S)
        while queue:
            a_idx = queue.pop(0)
            if a_idx in visited:
                continue
            visited.add(a_idx)
            needed.append(a_idx)
            # Recursively pull in ancestors of the ancestor
            queue.extend(a for a in ancestors[a_idx] if a not in visited)
        return needed

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_nodes(self, nodes: list[ContextNode]) -> np.ndarray:
        """Batch-encode node contents; use cache to avoid recomputation."""
        result = np.zeros((len(nodes), 384), dtype=np.float32)  # default dim
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, node in enumerate(nodes):
            if node.node_id in self._embedding_cache:
                result[i] = self._embedding_cache[node.node_id]
            else:
                uncached_indices.append(i)
                uncached_texts.append(node.content[:512])  # truncate for speed

        if uncached_texts:
            embeddings = self._raw_encode(uncached_texts)
            if embeddings.shape[1] != result.shape[1]:
                result = np.zeros((len(nodes), embeddings.shape[1]), dtype=np.float32)
                # Re-fill cached values with correct dim
                for i, node in enumerate(nodes):
                    if node.node_id in self._embedding_cache:
                        cached = self._embedding_cache[node.node_id]
                        if cached.shape[0] == embeddings.shape[1]:
                            result[i] = cached

            for local_i, global_i in enumerate(uncached_indices):
                emb = embeddings[local_i]
                result[global_i] = emb
                self._embedding_cache[nodes[global_i].node_id] = emb

        return result

    def _embed_query(self, query: str) -> np.ndarray:
        """Encode query string to a 1-D embedding vector."""
        return self._raw_encode([query[:512]])[0]

    def _raw_encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* using the loaded embedding model or TF-IDF fallback."""
        if self._embed_model is not None:
            try:
                vecs = self._embed_model.encode(  # type: ignore[union-attr]
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.array(vecs, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    '{"event":"embedding_model_error","reason":"%s",'
                    '"fallback":"tfidf"}',
                    str(exc),
                )

        # TF-IDF fallback ---------------------------------------------------
        return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: list[str]) -> np.ndarray:
        """Minimal TF-IDF cosine encoding as a deterministic fallback."""
        from collections import Counter

        vocab: dict[str, int] = {}
        tokenised = [t.lower().split() for t in texts]
        for tokens in tokenised:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        dim = max(len(vocab), 1)
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, tokens in enumerate(tokenised):
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                if tok in vocab:
                    result[i, vocab[tok]] = float(cnt)
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm

        return result

    # ------------------------------------------------------------------
    # Embedding model loading
    # ------------------------------------------------------------------

    def _load_embedding_model(self, model_name: str) -> object | None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            model = SentenceTransformer(model_name)
            logger.info(
                '{"event":"embedding_model_loaded","model":"%s"}', model_name
            )
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '{"event":"embedding_model_load_failed","model":"%s","reason":"%s",'
                '"fallback":"tfidf"}',
                model_name,
                str(exc),
            )
            return None
