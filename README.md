# nexus-context

> **Transparent middleware for referential integrity, KV-cache alignment,
> and WWW memory governance in local SLM deployments.**

[![PyPI version](https://img.shields.io/pypi/v/nexus-context.svg)](https://pypi.org/project/nexus-context/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/laxmikant2806/Nexus-Context/actions/workflows/ci.yml/badge.svg)](https://github.com/laxmikant2806/Nexus-Context/actions)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## Install

```bash
# Core install (~50 MB — no ML models)
pip install nexus-context

# With semantic embedding support (adds ~2 GB torch + sentence-transformers)
pip install "nexus-context[embeddings]"

# Full install — all optional extras
pip install "nexus-context[full]"
```

---

## The Problem

When running autonomous agents on local SLMs (vLLM, SGLang, Ollama), three compounding
failure modes emerge as context grows:

1. **Referential Dangling** — Standard prompt compression tools prune variable definitions
   while retaining code that depends on them. Result: `NameError: name 'DB_HOST' is not defined`.
   Over 88% of compression-induced errors in agentic sessions are of this class.

2. **KV Cache Invalidation Tax** — Any modification to the middle of a PagedAttention context
   invalidates the prefix hash, forcing full KV recomputation. On a 7B model this adds ~1.1s
   TTFT penalty per turn — accumulating to minutes of wasted GPU time per long session.

3. **Episodic Memory Bloat** — Verbose tool outputs accumulate in context.
   A 127-token `CREATE TABLE...` record can be losslessly compressed to a 31-token semantic tuple.

---

## Solution

Nexus-Context sits transparently between your agent and its local SLM:

```
Agent ──► nexus-serve (localhost:9000) ──► vLLM / SGLang / Ollama
                │
                ├── nexus.guard:   AST dependency graph + submodular pruning
                ├── nexus.cache:   Block-aligned prefix zone locking
                └── nexus.memory:  WWW episodic-to-semantic decay
```

---

## Quick-Start (60 seconds)

**Terminal 1** — Start your local model:
```bash
ollama run qwen2.5-coder:7b
```

**Terminal 2** — Start Nexus-Context proxy:
```bash
nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000
```

**Terminal 3** — Point your existing OpenAI client at port 9000 instead of 11434:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",   # <- only this line changes
    api_key="local",
)

# Turn 1: define a variable
r1 = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[
        {"role": "system", "content": "You are a Python coding agent."},
        {"role": "user", "content": "Define DB_HOST = 'prod.db.internal' and write connect_db()."},
    ],
    extra_headers={"X-Session-ID": "my-session"},
)

# Turn 2: reference it — DB_HOST is GUARANTEED to survive compaction
r2 = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[
        {"role": "system", "content": "You are a Python coding agent."},
        {"role": "user", "content": "Now write query_orders() using connect_db()."},
    ],
    extra_headers={"X-Session-ID": "my-session"},
)
```

**Dashboard** — Open http://localhost:9000/dashboard for live metrics.

---

## Embedding in Your FastAPI App

```python
import uvicorn
from nexus_context import create_app

app = create_app(
    backend_url="http://localhost:11434",
    backend_type="ollama",
    total_budget=8192,
    persist=True,             # SQLite session crash-recovery
)

uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

## Key Guarantees

| Guarantee | Target | Mechanism |
|---|---|---|
| Zero referential dangling | 0% NameError rate | γ→∞ submodular penalty on orphaned nodes |
| Prefix KV cache preservation | >85% hit rate | Block-aligned Zone P locked for session lifetime |
| Memory compression | >4:1 ratio | WWW semantic mutation tuple extraction |
| Low overhead | <80ms P95 per request | Lazy greedy + async pipeline |
| Backend agnostic | vLLM, SGLang, Ollama | OpenAI `/v1/chat/completions` proxy |

---

## Optional Extras

| Extra | Installs | Enables |
|---|---|---|
| `[nlp]` | spaCy, coreferee | NL coreference edges in context graph |
| `[embeddings]` | sentence-transformers, torch | Semantic similarity scoring in submodular solver |
| `[parsers]` | tree-sitter-* | Multi-language AST parsing (Python/SQL/Bash/JS) |
| `[hnswlib]` | hnswlib | ANN vector index for faster similarity lookup |
| `[full]` | All of the above | Complete feature set |
| `[dev]` | pytest, ruff, mypy, build, twine | Development tooling |

---

## CLI Reference

```
nexus-serve [OPTIONS]

Options:
  --backend-url URL       Backend SLM server URL (default: http://localhost:8000)
  --backend-type TYPE     vllm | sglang | ollama (default: vllm)
  --port INT              Port to listen on (default: 9000)
  --host STR              Bind host (default: 0.0.0.0)
  --block-size INT        KV block size: 16 or 32 (default: 16)
  --total-budget INT      Max context token budget (default: 4096)
  --persist               Enable SQLite session persistence
  --db-path PATH          Session DB path (default: nexus_sessions.db)
  --persist-every INT     Save session every N turns (default: 5)
  --ltkb-db PATH          LTKB DB path (default: nexus_ltkb.db)
  --no-ltkb               Disable long-term knowledge base
  --log-level LEVEL       Logging level (default: info)
```

---

## Architecture

```
POST /v1/chat/completions
         │
         ▼
┌─────────────────────────────────┐
│ 1. Tool Call Interception       │  JSON/text compression for role=tool messages
├─────────────────────────────────┤
│ 2. Block-Aligned Zone P Lock    │  SHA-256 prefix hash frozen for session lifetime
├─────────────────────────────────┤
│ 3. AST Dependency Graph         │  Multi-modal graph: code + NL + tool output nodes
├─────────────────────────────────┤
│ 4. Submodular Compaction Solver │  Budget-safe pruning with ∞ dangling penalty
├─────────────────────────────────┤
│ 5. WWW Memory + LTKB Injection  │  Persistent facts injected across sessions
└─────────────────────────────────┘
         │
         ▼
Backend SLM (vLLM / SGLang / Ollama)
```

---

## License

MIT © 2026 Laxmikant Bhagat
