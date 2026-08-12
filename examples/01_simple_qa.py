"""
examples/01_simple_qa.py
========================
Basic inline text ingestion and agent querying using ContextNexus.
"""

from context_nexus import ContextNexus

def main():
    # 1. Initialize ContextNexus unified client
    nexus = ContextNexus(token_budget=2048)

    # 2. Ingest inline text document
    sample_text = (
        "PostgreSQL production server is deployed at host prod.db.internal on port 5432. "
        "The master database user is nexus_admin. Active order queries must filter by status='active'."
    )
    nexus.ingest(sample_text, doc_id="db_config")

    # 3. Retrieve context for query
    query = "What is the host and port for PostgreSQL database?"
    context = nexus.get_context(query)

    print("=== Query ===")
    print(query)
    print("\n=== Retrieved Context ===")
    print(context["context_text"])
    print("\n=== Trace Stats ===")
    print(context["trace_stats"])

if __name__ == "__main__":
    main()
