"""
nexus_context.guard
===================
Referential integrity via AST dependency graphing and submodular pruning.

Public API
----------
ContextGraphBuilder  – builds the context dependency graph G = (V, E)
SubmodularSolver     – budget-constrained greedy submodular maximisation
"""

from nexus_context.guard.schemas import (
    CompactionResult,
    ContextGraph,
    ContextNode,
    DependencyEdge,
    EdgeType,
    NodeType,
)

__all__ = [
    "CompactionResult",
    "ContextGraph",
    "ContextNode",
    "DependencyEdge",
    "EdgeType",
    "NodeType",
]
