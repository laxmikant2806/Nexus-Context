"""
nexus_context.guard
===================
Referential integrity via AST dependency graphing and submodular pruning.

Public API
----------
ContextGraphBuilder  – builds the context dependency graph G = (V, E)
SubmodularSolver     – budget-constrained greedy submodular maximisation
"""

from nexus_context.guard.adaptive_chunking import AdaptiveSemanticChunker
from nexus_context.guard.schemas import (
    BoundaryEvaluationResult,
    CompactionResult,
    ContextGraph,
    ContextNode,
    DependencyEdge,
    EdgeType,
    NodeType,
    SemanticChunk,
)

__all__ = [
    "AdaptiveSemanticChunker",
    "BoundaryEvaluationResult",
    "CompactionResult",
    "ContextGraph",
    "ContextNode",
    "DependencyEdge",
    "EdgeType",
    "NodeType",
    "SemanticChunk",
]
