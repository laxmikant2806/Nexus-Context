# Quickstart Guide: ContextNexus

`context-nexus` (repo: `Nexus-Context`) is a high-performance Python package combining a native Rust-accelerated core via PyO3 with a clean Python SDK for **Hybrid RAG** (Vector Embeddings + Knowledge Graphs + Token Budgeting).

---

## 1. Quick Installation

```bash
pip install nexus-context
```

Or install from source with Rust acceleration:

```bash
git clone https://github.com/laxmikant2806/Nexus-Context.git
cd Nexus-Context
pip install -e .
```

---

## 2. Basic Ingestion & Context Retrieval

```python
from context_nexus import ContextNexus

# Initialize client
nexus = ContextNexus(token_budget=4096)

# Ingest documents or code files
nexus.ingest("PostgreSQL host is prod.db.internal on port 5432.", doc_id="db_config")

# Query budgeted context
context = nexus.get_context("What is the database host?")
print(context["context_text"])
```

---

## 3. Running Proxy Server

Start the transparent proxy server between your agent and local backend (vLLM / Ollama):

```bash
nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000
```
