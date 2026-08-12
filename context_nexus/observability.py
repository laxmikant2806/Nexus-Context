"""
context_nexus.observability
===========================
Observability, span tracing, and execution analytics for ContextNexus queries.
Tracks total latency, vector search time, graph traversal overhead, token budget usage,
and memory decay statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryTraceSpan:
    """A single span trace capturing execution metadata."""

    query_id: str
    session_id: str
    query_text: str
    total_latency_ms: float = 0.0
    vector_search_ms: float = 0.0
    graph_traversal_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    budget_allocation_ms: float = 0.0
    total_tokens_used: int = 0
    token_budget: int = 4096
    cache_hit: bool = False
    rust_accelerated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservabilityTracer:
    """Collects and reports observability traces for ContextNexus sessions."""

    def __init__(self) -> None:
        self._spans: list[QueryTraceSpan] = []

    def start_span(self, query_id: str, session_id: str, query_text: str, token_budget: int = 4096) -> QueryTraceSpan:
        span = QueryTraceSpan(
            query_id=query_id,
            session_id=session_id,
            query_text=query_text,
            token_budget=token_budget,
        )
        return span

    def record_span(self, span: QueryTraceSpan) -> None:
        self._spans.append(span)

    def get_session_spans(self, session_id: str) -> list[QueryTraceSpan]:
        return [s for s in self._spans if s.session_id == session_id]

    def get_summary_stats(self) -> dict[str, Any]:
        if not self._spans:
            return {"total_queries": 0, "avg_latency_ms": 0.0}

        total_lat = sum(s.total_latency_ms for s in self._spans)
        vec_lat = sum(s.vector_search_ms for s in self._spans)
        graph_lat = sum(s.graph_traversal_ms for s in self._spans)
        n = len(self._spans)

        return {
            "total_queries": n,
            "avg_latency_ms": round(total_lat / n, 2),
            "avg_vector_search_ms": round(vec_lat / n, 2),
            "avg_graph_traversal_ms": round(graph_lat / n, 2),
            "rust_accelerated_count": sum(1 for s in self._spans if s.rust_accelerated),
        }
