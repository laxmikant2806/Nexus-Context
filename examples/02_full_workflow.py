"""
examples/02_full_workflow.py
============================
Full workflow: Document ingestion, graph edge creation, token budget enforcement,
and query execution with citations and trace metrics.
"""

from context_nexus import ContextNexus

def main():
    nexus = ContextNexus(token_budget=4096)

    # Ingest multiple document sources
    doc1 = (
        "Nexus-Context uses PagedAttention prefix KV cache alignment to freeze Zone P system prompts. "
        "Block size is set to 16 or 32 tokens, ensuring 100% cache hit rates across multi-turn sessions."
    )
    doc2 = (
        "The referential integrity guard constructs an AST dependency graph G=(V,E). "
        "If a variable definition in Turn 1 is referenced in Turn 10, the submodular solver "
        "force-includes Turn 1 into context."
    )

    nexus.ingest(doc1, doc_id="doc_cache")
    nexus.ingest(doc2, doc_id="doc_guard")

    # Connect documents in graph store
    nexus.graph_store.add_edge("doc_cache", "doc_guard", edge_type="relates_to")

    # Run query
    response = nexus.query("How does Nexus-Context preserve prefix cache and prevent dangling variables?")
    print("=== Agent Result ===")
    print("Context Text:")
    print(response["context_text"])
    print("\nCitations:")
    print(response["citations"])
    print("\nTrace Stats:")
    print(response["trace_stats"])

if __name__ == "__main__":
    main()
