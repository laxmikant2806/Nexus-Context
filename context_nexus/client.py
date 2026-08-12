"""
context_nexus.client
====================
Main ContextNexus unified entrypoint providing ingest(), query(), and get_context() API methods.
Integrates vector storage, knowledge graph topology, RRF hybrid search, token budget allocation,
and observability metrics.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from context_nexus.agent import ContextAgent
from context_nexus.budget import TokenBudgetAllocator
from context_nexus.graph_store import GraphStore
from context_nexus.hybrid_search import is_rust_available, rrf_fusion
from context_nexus.ingestion import DocumentIngestor, IngestedDocument
from context_nexus.observability import ObservabilityTracer, QueryTraceSpan
from context_nexus.vector_store import VectorStore


class ContextNexus:
    """Unified entrypoint for Hybrid RAG (Vector Embeddings + Knowledge Graph + Token Budgeting).

    Parameters
    ----------
    embedding_model:
        Model name for vector embeddings (default 'all-MiniLM-L6-v2').
    token_budget:
        Default max token budget for context retrieval.
    chunk_size:
        Text chunk size in tokens.
    overlap:
        Text chunk overlap in tokens.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        token_budget: int = 4096,
        chunk_size: int = 256,
        overlap: int = 32,
    ) -> None:
        self.default_budget = token_budget
        self.vector_store = VectorStore(model_name=embedding_model)
        self.graph_store = GraphStore()
        self.budget_allocator = TokenBudgetAllocator()
        self.ingestor = DocumentIngestor(chunk_size=chunk_size, overlap=overlap)
        self.tracer = ObservabilityTracer()
        self.agent = ContextAgent(self)

    def ingest(self, source: str | Path | list[str], doc_id: str | None = None) -> IngestedDocument:
        """Ingest a file path, raw string, or list of text strings into vector & graph stores."""
        if isinstance(source, (str, Path)) and Path(source).exists():
            doc = self.ingestor.ingest_file(source)
        elif isinstance(source, list):
            text = "\n\n".join(source)
            doc = self.ingestor.ingest_text(text, doc_id=doc_id or "doc_batch")
        else:
            doc = self.ingestor.ingest_text(str(source), doc_id=doc_id or "doc_inline")

        # 1. Add chunks to vector store
        chunk_ids = [f"{doc.doc_id}:chunk:{i}" for i in range(len(doc.chunks))]
        self.vector_store.add_documents(chunk_ids, doc.chunks)

        # 2. Add node & import edges to graph store
        self.graph_store.add_node(doc.doc_id, metadata={"file_type": doc.file_type})
        for src, tgt in doc.import_edges:
            self.graph_store.add_edge(src, tgt)

        return doc

    def get_context(
        self,
        query: str,
        session_id: str = "default",
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve token-budgeted hybrid context (Vector + Graph) for *query*."""
        t_start = time.perf_counter()
        query_id = str(uuid.uuid4())[:8]
        budget = token_budget or self.default_budget

        span = self.tracer.start_span(query_id, session_id, query, budget)
        span.rust_accelerated = is_rust_available()

        # 1. Vector Search
        t_vec_start = time.perf_counter()
        vector_results = self.vector_store.search(query, top_k=20)
        span.vector_search_ms = (time.perf_counter() - t_vec_start) * 1000
        vector_ranks = [doc_id for doc_id, _ in vector_results]

        # 2. Graph Traversal
        t_graph_start = time.perf_counter()
        graph_ranks: list[str] = []
        if vector_ranks:
            start_node = vector_ranks[0].split(":")[0]  # root doc_id
            graph_ranks = self.graph_store.traverse(start_node, depth=2)
        span.graph_traversal_ms = (time.perf_counter() - t_graph_start) * 1000

        # 3. Hybrid RRF Fusion
        t_rrf_start = time.perf_counter()
        fused = rrf_fusion(vector_ranks, graph_ranks, k=60)
        span.rrf_fusion_ms = (time.perf_counter() - t_rrf_start) * 1000

        # 4. Map top item IDs back to chunk text
        fused_ids = [item_id for item_id, _ in fused]
        chunk_texts: list[str] = []
        chunk_map = dict(zip(self.vector_store._doc_ids, self.vector_store._doc_texts))

        for item_id in fused_ids:
            if item_id in chunk_map:
                chunk_texts.append(chunk_map[item_id])

        # 5. Deterministic Token Budget Allocation
        t_budget_start = time.perf_counter()
        allocated_chunks = self.budget_allocator.allocate_snippets(chunk_texts, budget)
        span.budget_allocation_ms = (time.perf_counter() - t_budget_start) * 1000

        span.total_latency_ms = (time.perf_counter() - t_start) * 1000
        span.total_tokens_used = sum(self.budget_allocator.count_tokens(c) for c in allocated_chunks)
        self.tracer.record_span(span)

        context_text = "\n\n".join(allocated_chunks)
        citations = fused_ids[: len(allocated_chunks)]

        return {
            "query_id": query_id,
            "session_id": session_id,
            "context_text": context_text,
            "citations": citations,
            "trace_stats": {
                "total_latency_ms": round(span.total_latency_ms, 2),
                "vector_search_ms": round(span.vector_search_ms, 2),
                "graph_traversal_ms": round(span.graph_traversal_ms, 2),
                "rrf_fusion_ms": round(span.rrf_fusion_ms, 2),
                "tokens_used": span.total_tokens_used,
                "token_budget": budget,
                "rust_accelerated": span.rust_accelerated,
            },
        }

    def query(self, query_text: str, session_id: str = "default") -> dict[str, Any]:
        """Execute agent query via ContextAgent."""
        return self.agent.run(query_text, session_id=session_id)
