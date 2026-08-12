"""
context_nexus
=============
Hybrid RAG Framework (Vector Embeddings + Knowledge Graphs + Token Budgeting).
Accelerated by native Rust extension module (crates/nexus-core).
"""

from context_nexus.agent import ContextAgent
from context_nexus.budget import TokenBudgetAllocator
from context_nexus.client import ContextNexus
from context_nexus.graph_store import GraphStore
from context_nexus.hybrid_search import (
    compute_cosine_distances,
    fast_chunk_text,
    is_rust_available,
    rrf_fusion,
    traverse_graph,
)
from context_nexus.ingestion import DocumentIngestor, IngestedDocument
from context_nexus.observability import ObservabilityTracer, QueryTraceSpan
from context_nexus.vector_store import VectorStore

__version__ = "0.1.0"
__author__ = "Nexus-Context Team"
__license__ = "MIT"

__all__ = [
    "ContextNexus",
    "ContextAgent",
    "DocumentIngestor",
    "IngestedDocument",
    "VectorStore",
    "GraphStore",
    "TokenBudgetAllocator",
    "ObservabilityTracer",
    "QueryTraceSpan",
    "compute_cosine_distances",
    "traverse_graph",
    "rrf_fusion",
    "fast_chunk_text",
    "is_rust_available",
]
