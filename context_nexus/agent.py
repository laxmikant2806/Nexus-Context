"""
context_nexus.agent
===================
Agent query interface for executing context-aware RAG queries, fetching citations,
and interfacing with local or remote LLM completions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_nexus.client import ContextNexus


class ContextAgent:
    """Agent interface for executing RAG queries with citations."""

    def __init__(self, nexus: ContextNexus) -> None:
        self.nexus = nexus

    def run(self, query_text: str, session_id: str = "default", token_budget: int = 4096) -> dict[str, Any]:
        """Execute a context-aware agent query."""
        context_bundle = self.nexus.get_context(
            query=query_text,
            session_id=session_id,
            token_budget=token_budget,
        )

        return {
            "query": query_text,
            "context_text": context_bundle["context_text"],
            "citations": context_bundle["citations"],
            "trace_stats": context_bundle["trace_stats"],
        }
