"""
examples/04_research_agent.py
=============================
Multi-turn research loop fetching from web sources and building a knowledge graph.
"""

from context_nexus import ContextNexus

def main():
    nexus = ContextNexus(token_budget=4096)

    # Turn 1: Web fetch ingestion
    web_doc_1 = "vLLM implements PagedAttention to partition KV cache memory into non-contiguous physical blocks."
    nexus.ingest(web_doc_1, doc_id="web_vllm_paper")

    # Turn 2: Web fetch ingestion
    web_doc_2 = "SGLang introduces RadixAttention for automatic prefix KV cache sharing across complex workflows."
    nexus.ingest(web_doc_2, doc_id="web_sglang_paper")

    # Connect related research findings
    nexus.graph_store.add_edge("web_vllm_paper", "web_sglang_paper", edge_type="cites")

    # Multi-turn query
    response = nexus.query("Compare PagedAttention in vLLM with RadixAttention in SGLang.")
    print("=== Research Agent Query Output ===")
    print("Context Text:")
    print(response["context_text"])
    print("\nCitations:")
    print(response["citations"])

if __name__ == "__main__":
    main()
