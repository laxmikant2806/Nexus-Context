"""
tests/test_guard.py
====================
Unit and property-based tests for nexus_context.guard (Phase 2).

Critical invariant (must never fail):
    After submodular compaction, every selected node's ancestors are also
    selected — zero referential dangling.
"""

from __future__ import annotations

import pytest
from nexus_context.cache.schemas import ChatMessage
from nexus_context.guard.ast_graph import ContextGraphBuilder
from nexus_context.guard.schemas import (
    CompactionResult,
    ContextGraph,
    ContextNode,
    DependencyEdge,
    EdgeType,
    NodeType,
)
from nexus_context.guard.submodular import SubmodularSolver


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_code_turn(code: str, language: str, turn_index: int) -> ChatMessage:
    fence = f"```{language}\n{code}\n```"
    return ChatMessage(
        role="user",
        content=fence,
        token_count=max(1, len(fence) // 4),
        turn_index=turn_index,
    )


def _make_nl_turn(text: str, turn_index: int) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=text,
        token_count=max(1, len(text) // 4),
        turn_index=turn_index,
    )


def _db_setup_turns() -> list[ChatMessage]:
    """Three-turn DB setup scenario from overview_and_research.md §2.4."""
    return [
        _make_nl_turn("Set up a PostgreSQL connection to the production database.", 0),
        _make_code_turn(
            'DB_HOST = "prod.db.internal"\n'
            "DB_PORT = 5432\n"
            'conn = connect(host=DB_HOST, port=DB_PORT)',
            "python",
            1,
        ),
        _make_code_turn(
            "cursor = conn.cursor()\n"
            "cursor.execute('SELECT COUNT(*) FROM orders')",
            "python",
            2,
        ),
    ]


# ---------------------------------------------------------------------------
# ContextGraph unit tests
# ---------------------------------------------------------------------------


class TestContextGraph:

    def test_add_node_and_retrieve(self) -> None:
        g = ContextGraph(session_id="s1")
        node = ContextNode(
            node_id="0:ast_assign:x",
            node_type=NodeType.AST_ASSIGNMENT,
            turn_index=0,
            content="x = 42",
            token_count=3,
        )
        g.add_node(node)
        assert "0:ast_assign:x" in g.nodes

    def test_get_ancestors_direct(self) -> None:
        g = ContextGraph(session_id="s1")
        n1 = ContextNode(
            node_id="n1", node_type=NodeType.AST_ASSIGNMENT,
            turn_index=0, content="x=1", token_count=2,
        )
        n2 = ContextNode(
            node_id="n2", node_type=NodeType.AST_CALL,
            turn_index=1, content="y=x+1", token_count=4,
        )
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DependencyEdge(
            edge_id="n1->n2", source_id="n1", target_id="n2",
            edge_type=EdgeType.VAR_DEF_TO_REF,
        ))
        ancestors = g.get_ancestors("n2")
        assert "n1" in ancestors

    def test_transitive_closure_adds_transitive_edges(self) -> None:
        g = ContextGraph(session_id="s1")
        for nid in ["n1", "n2", "n3"]:
            g.add_node(ContextNode(
                node_id=nid, node_type=NodeType.AST_ASSIGNMENT,
                turn_index=0, content="x", token_count=1,
            ))
        g.add_edge(DependencyEdge(
            edge_id="n1->n2", source_id="n1", target_id="n2",
            edge_type=EdgeType.VAR_DEF_TO_REF,
        ))
        g.add_edge(DependencyEdge(
            edge_id="n2->n3", source_id="n2", target_id="n3",
            edge_type=EdgeType.VAR_DEF_TO_REF,
        ))
        g.compute_transitive_closure()
        assert g.transitive_closure_computed
        # n1 → n3 transitive edge should have been added
        transitive_pairs = {(e.source_id, e.target_id) for e in g.edges if e.is_transitive}
        assert ("n1", "n3") in transitive_pairs

    def test_compute_transitive_closure_idempotent(self) -> None:
        g = ContextGraph(session_id="s1")
        g.add_node(ContextNode(
            node_id="n1", node_type=NodeType.AST_ASSIGNMENT,
            turn_index=0, content="x", token_count=1,
        ))
        g.compute_transitive_closure()
        edge_count_first = len(g.edges)
        g.compute_transitive_closure()  # second call must be a no-op
        assert len(g.edges) == edge_count_first


# ---------------------------------------------------------------------------
# ContextGraphBuilder tests
# ---------------------------------------------------------------------------


class TestContextGraphBuilder:

    @pytest.fixture
    def builder(self) -> ContextGraphBuilder:
        return ContextGraphBuilder(session_id="test")

    def test_python_assignment_creates_node(self, builder: ContextGraphBuilder) -> None:
        turns = [_make_code_turn('DB_HOST = "prod.internal"', "python", 0)]
        graph = builder.build(turns)
        assign_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == NodeType.AST_ASSIGNMENT
        ]
        assert len(assign_nodes) >= 1

    def test_empty_turns_returns_empty_graph(self, builder: ContextGraphBuilder) -> None:
        graph = builder.build([])
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_nl_turn_creates_nl_sentence_node(self, builder: ContextGraphBuilder) -> None:
        turns = [_make_nl_turn("Deploy the app to production.", 0)]
        graph = builder.build(turns)
        nl_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.NL_SENTENCE]
        assert len(nl_nodes) >= 1

    def test_malformed_code_does_not_raise(self, builder: ContextGraphBuilder) -> None:
        """Malformed Python must not crash the builder."""
        turns = [_make_code_turn("def broken(: pass", "python", 0)]
        graph = builder.build(turns)  # must not raise
        assert graph is not None

    def test_multiple_turns_build_single_graph(self, builder: ContextGraphBuilder) -> None:
        turns = [
            _make_code_turn("x = 1", "python", 0),
            _make_code_turn("y = x + 1", "python", 1),
        ]
        graph = builder.build(turns)
        turn_indices = {n.turn_index for n in graph.nodes.values()}
        assert 0 in turn_indices
        assert 1 in turn_indices


# ---------------------------------------------------------------------------
# SubmodularSolver tests (core invariant)
# ---------------------------------------------------------------------------


class TestSubmodularSolver:

    @pytest.fixture
    def solver(self) -> SubmodularSolver:
        # Use TF-IDF fallback (no model download in CI)
        s = SubmodularSolver.__new__(SubmodularSolver)
        s._alpha = 0.5
        s._beta = 0.5
        s._gamma_factor = 1e6
        s._embedding_model_name = "tfidf-fallback"
        s._embed_model = None
        s._embedding_cache = {}
        return s

    @pytest.fixture
    def builder(self) -> ContextGraphBuilder:
        return ContextGraphBuilder(session_id="test")

    def _check_no_dangling(
        self, graph: ContextGraph, result: CompactionResult
    ) -> None:
        """Assert the core invariant: no selected node has an unresolved ancestor."""
        selected = set(result.selected_node_ids)
        violations = []
        for edge in graph.edges:
            if edge.target_id in selected and edge.source_id not in selected:
                violations.append((edge.source_id, edge.target_id))
        assert violations == [], (
            f"Dangling violations found after compaction:\n"
            + "\n".join(f"  {src} -> {tgt}" for src, tgt in violations)
        )

    def test_selected_tokens_never_exceed_budget(
        self, solver: SubmodularSolver, builder: ContextGraphBuilder
    ) -> None:
        turns = _db_setup_turns()
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        for budget in [50, 100, 200, 500, 2048]:
            result = solver.solve(graph, budget=budget, query="query database")
            total = sum(
                graph.nodes[nid].token_count
                for nid in result.selected_node_ids
                if nid in graph.nodes
            )
            assert total <= budget, (
                f"Budget {budget} exceeded: {total} tokens selected"
            )

    def test_no_dangling_db_setup_scenario(
        self, solver: SubmodularSolver, builder: ContextGraphBuilder
    ) -> None:
        """The canonical 3-turn DB scenario must produce zero dangling nodes."""
        turns = _db_setup_turns()
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=2048, query="query database")
        self._check_no_dangling(graph, result)

    def test_empty_graph_returns_empty_compaction(
        self, solver: SubmodularSolver
    ) -> None:
        graph = ContextGraph(session_id="s1")
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=1000, query="task")
        assert result.selected_node_ids == []
        assert result.tokens_used == 0

    def test_dangling_violations_field_is_zero(
        self, solver: SubmodularSolver, builder: ContextGraphBuilder
    ) -> None:
        """CompactionResult.dangling_violations must always be 0."""
        turns = _db_setup_turns()
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=500, query="database query")
        assert result.dangling_violations == 0, (
            f"dangling_violations={result.dangling_violations} (expected 0)"
        )

    def test_latency_ms_is_positive(
        self, solver: SubmodularSolver, builder: ContextGraphBuilder
    ) -> None:
        turns = _db_setup_turns()
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=1000, query="task")
        assert result.latency_ms >= 0.0
