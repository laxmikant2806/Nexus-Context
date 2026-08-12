"""
context_nexus.graph_store
=========================
Knowledge Graph store for managing nodes, directed dependency edges,
and fast BFS traversal queries.
"""

from __future__ import annotations

from typing import Any

from context_nexus.hybrid_search import traverse_graph


class GraphStore:
    """In-memory Knowledge Graph node and edge manager."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str]] = []

    def add_node(self, node_id: str, label: str = "", metadata: dict[str, Any] | None = None) -> None:
        self._nodes[node_id] = {
            "node_id": node_id,
            "label": label or node_id,
            "metadata": metadata or {},
        }

    def add_edge(self, source_id: str, target_id: str, edge_type: str = "relates_to") -> None:
        if source_id not in self._nodes:
            self.add_node(source_id)
        if target_id not in self._nodes:
            self.add_node(target_id)
        self._edges.append((source_id, target_id))

    def traverse(self, start_node: str, depth: int = 2) -> list[str]:
        """Traverse graph up to depth hops starting from start_node."""
        nodes = list(self._nodes.keys())
        return traverse_graph(nodes, self._edges, start_node, depth)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
