"""tests/test_ltkb.py — Feature I unit tests."""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from nexus_context.memory.ltkb import LongTermKnowledgeBase, LTKBFact


@pytest.fixture
async def ltkb(tmp_path) -> LongTermKnowledgeBase:  # type: ignore[type-arg]
    db = LongTermKnowledgeBase(
        db_path=str(tmp_path / "test_ltkb.db"),
        weight_threshold=0.8,
        min_in_degree=2,
    )
    await db.initialize()
    return db


def _make_mock_graph(nodes_with_weights: list[tuple[str, float, int]]) -> MagicMock:
    """Build a mock ContextGraph with nodes at specified weights and in-degrees."""
    graph = MagicMock()

    # Create nodes
    nodes: dict[str, MagicMock] = {}
    for node_id, weight, in_degree in nodes_with_weights:
        node = MagicMock()
        node.node_id = node_id
        node.content = f"content for {node_id}"
        node.node_type = "ast_assign"
        node.retention_weight = weight
        node.is_pinned = False
        node.compute_weight = MagicMock(return_value=weight)
        node.metadata = {"name": node_id}
        nodes[node_id] = node

    graph.nodes = nodes

    # Create edges to satisfy min_in_degree
    edges = []
    for node_id, _, in_degree in nodes_with_weights:
        for j in range(in_degree):
            edge = MagicMock()
            edge.target_id = node_id
            edges.append(edge)
    graph.edges = edges
    return graph


class TestExtractAndPersist:
    async def test_high_weight_nodes_persisted(self, ltkb: LongTermKnowledgeBase) -> None:
        """Nodes with weight >= 0.8 and in_degree >= 2 must be persisted."""
        graph = _make_mock_graph([
            ("heavy_node", 0.95, 3),   # should persist
            ("light_node", 0.3, 5),    # should NOT persist (low weight)
            ("low_degree", 0.9, 1),    # should NOT persist (in_degree < 2)
        ])
        count = await ltkb.extract_and_persist(graph, "test-session", current_turn=5)
        assert count == 1  # only heavy_node qualifies

    async def test_zero_facts_when_no_qualifying_nodes(self, ltkb: LongTermKnowledgeBase) -> None:
        """If no nodes qualify, persist must return 0."""
        graph = _make_mock_graph([
            ("light1", 0.1, 10),
            ("light2", 0.2, 10),
        ])
        count = await ltkb.extract_and_persist(graph, "test-session", current_turn=5)
        assert count == 0

    async def test_fact_count_increments(self, ltkb: LongTermKnowledgeBase) -> None:
        """get_fact_count must reflect the number of persisted facts."""
        initial = await ltkb.get_fact_count()
        graph = _make_mock_graph([("n1", 0.9, 3), ("n2", 0.85, 2)])
        await ltkb.extract_and_persist(graph, "s-count", current_turn=2)
        after = await ltkb.get_fact_count()
        assert after >= initial + 1

    async def test_pinned_node_always_qualifies(self, ltkb: LongTermKnowledgeBase) -> None:
        """A pinned node (infinite weight) must always qualify."""
        graph = MagicMock()
        pinned = MagicMock()
        pinned.node_id = "pinned_node"
        pinned.content = "very important fact"
        pinned.node_type = "ast_funcdef"
        pinned.is_pinned = True
        pinned.retention_weight = math.inf
        pinned.compute_weight = MagicMock(return_value=math.inf)
        pinned.metadata = {}
        graph.nodes = {"pinned_node": pinned}
        edge = MagicMock(); edge.target_id = "pinned_node"
        graph.edges = [edge, edge, edge]

        count = await ltkb.extract_and_persist(graph, "s-pinned", current_turn=1)
        assert count >= 1


class TestGetRelevantFacts:
    async def test_returns_matching_facts(self, ltkb: LongTermKnowledgeBase) -> None:
        """get_relevant_facts must return facts relevant to the query."""
        # Directly insert facts via sync method
        facts = [
            LTKBFact("f1", "s1", "ast_assign", "database connection host port credentials", 0.9, 3, 1.0, 1.0),
            LTKBFact("f2", "s1", "ast_funcdef", "unrelated cooking recipe ingredients", 0.85, 2, 1.0, 1.0),
        ]
        ltkb._persist_facts_sync(facts)

        results = await ltkb.get_relevant_facts("database connection", top_k=5)
        assert len(results) >= 1
        assert any("database" in r.content for r in results)

    async def test_empty_db_returns_empty_list(self, ltkb: LongTermKnowledgeBase) -> None:
        results = await ltkb.get_relevant_facts("any query", top_k=5)
        assert results == []


class TestInjectIntoZoneP:
    async def test_empty_facts_returns_empty_string(self, ltkb: LongTermKnowledgeBase) -> None:
        result = await ltkb.inject_into_zone_p([])
        assert result == ""

    async def test_annotation_format(self, ltkb: LongTermKnowledgeBase) -> None:
        facts = [LTKBFact("f1", "s1", "ast_assign", "important variable x = 42", 0.9, 3, 1.0, 1.0)]
        result = await ltkb.inject_into_zone_p(facts)
        assert result.startswith("<!-- NEXUS_LTKB")
        assert "important variable" in result
        assert result.endswith("-->")
