# nexus-context

> **Stop your AI agent from forgetting things, crashing with NameErrors, or wasting GPU time
> re-reading the same prompt 100 times in a row.**

[![PyPI version](https://img.shields.io/pypi/v/nexus-context.svg?color=blue)](https://pypi.org/project/nexus-context/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/laxmikant2806/Nexus-Context/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/nexus-context)](https://pypi.org/project/nexus-context/)
[![CI](https://github.com/laxmikant2806/Nexus-Context/actions/workflows/ci.yml/badge.svg)](https://github.com/laxmikant2806/Nexus-Context/actions)

---

```
pip install nexus-context
```

---

## What is this?

**nexus-context** is a transparent middleware proxy for local AI model servers
(Ollama, vLLM, SGLang). You point your existing OpenAI-compatible client at it instead
of directly at your model, and it silently handles three of the hardest problems in
production agentic AI:

| Without nexus-context | With nexus-context |
|---|---|
| AI forgets variable definitions → `NameError` | Dependency graph guarantees definitions are **never** pruned without their references |
| GPU re-reads your system prompt from scratch every turn → slow | System prompt frozen in GPU KV-cache for the entire session → **instant** |
| Tool outputs bloat context to thousands of tokens | Smart compression reduces tool output by 60–80% before it hits the model |
| Agent amnesia across sessions | Long-term knowledge base persists facts across restarts |

---

## The Problem (Simple Version)

Imagine you're running a long coding session with an AI agent. You ask it 20 questions.
On turn 20, you ask it to use a function it defined on turn 3.

**Standard tools**: They randomly trim old turns to save space, sometimes deleting the
function definition from turn 3 while keeping the call on turn 20. The AI tries to call
a function that no longer exists in its memory. Crash.

**Worse**: Your AI re-reads your entire system prompt from scratch on every single turn.
On a 7B parameter model, this adds ~1.1 seconds of startup delay to every response.
In a 50-turn session, you waste nearly a minute of GPU time re-reading the same text.

**nexus-context** solves both problems by acting as a smart traffic controller between
your code and your AI model.

---

## Quick-Start (Under 2 Minutes)

### Prerequisites

- Python 3.11+
- A running local AI model server: [Ollama](https://ollama.ai), [vLLM](https://vllm.ai), or [SGLang](https://sgl-project.github.io)

### Install

```bash
pip install nexus-context
```

### Start the proxy

```bash
# With Ollama
nexus-serve --backend-url http://localhost:11434 --backend-type ollama --port 9000

# With vLLM
nexus-serve --backend-url http://localhost:8000 --backend-type vllm --port 9000

# With SGLang
nexus-serve --backend-url http://localhost:30000 --backend-type sglang --port 9000
```

### Use it — change one line of code

```python
from openai import OpenAI

# Before: directly to your model
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

# After: through nexus-context (one line change)
client = OpenAI(base_url="http://localhost:9000/v1", api_key="local")

messages = [{"role": "system", "content": "You are a Python coding agent."}]

# Turn 1: Define something
messages.append({"role": "user", "content": "Define DB_HOST = 'prod.internal' and write connect_db()."})
r1 = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=messages,
    extra_headers={"X-Session-ID": "my-session"},  # Track this session
)
messages.append({"role": "assistant", "content": r1.choices[0].message.content})

# Turn 2: Reference it — DB_HOST is GUARANTEED to survive any context compaction
messages.append({"role": "user", "content": "Now write query_orders() using connect_db()."})
r2 = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=messages,
    extra_headers={"X-Session-ID": "my-session"},
)
print(r2.choices[0].message.content)
```

### Open the live dashboard

```
http://localhost:9000/dashboard
```

Metrics update in real time — no page refresh needed.

---

## How It Works (Technical Overview)

nexus-context intercepts every `/v1/chat/completions` request and runs it through a
5-stage pipeline before forwarding to your AI backend:

```
Your Agent (OpenAI client)
        │
        ▼
┌───────────────────────────────────────────────┐
│  nexus-serve  (localhost:9000)                │
│                                               │
│  Stage 1: Tool Call Compression               │
│    Shrinks large tool outputs (JSON/text)     │
│    before they consume your token budget      │
│                                               │
│  Stage 2: Zone P Block Alignment              │
│    Pads system prompt to 16-token boundaries  │
│    → GPU KV-cache reuse every turn            │
│                                               │
│  Stage 3: AST Dependency Graph                │
│    Maps variables, functions, imports into    │
│    a directed graph to track what depends     │
│    on what                                    │
│                                               │
│  Stage 4: Submodular Compaction               │
│    When context is too long, selects what     │
│    to keep using graph-safe optimization      │
│    (functions never removed without           │
│    their definitions)                         │
│                                               │
│  Stage 5: Memory + Long-Term Knowledge        │
│    Important facts saved to SQLite and        │
│    re-injected in future sessions             │
└───────────────────────────────────────────────┘
        │
        ▼
vLLM / SGLang / Ollama (your AI backend)
```

### The 3 Zones

Every conversation is split into zones:

- **Zone P** (Permanent): Your system prompt, locked in GPU memory. Never changes, never recomputed.
- **Zone T** (Transient): Older conversation turns. Compacted when budget is exceeded, using the dependency graph to guarantee safe pruning.
- **Zone R** (Recent): The current user message. Always kept 100% intact.

---

## Features

### ⚡ KV-Cache Alignment (Always On)
Automatically pads your system prompt to exact 16-token or 32-token boundaries.
This makes the GPU treat it as a static, pre-computed block it can cache forever.

**Result**: First turn is full speed. Every subsequent turn reuses the cache → 100% KV-cache hit rate in most sessions.

### 🛡 Referential Integrity Graph (Always On)
Parses Python, SQL, Bash, and JavaScript code in your conversation into a dependency graph.
When the context is too long and turns must be pruned, the solver guarantees:

> *A function call can never be kept if its definition has been removed.*

This eliminates the class of `NameError` / `undefined variable` crashes caused by naive compression.

### 🗜 Tool Call Compression (`role="tool"` messages)
When your agent calls an external tool (web search, database query, code execution), the
response often contains thousands of tokens of raw JSON. nexus-context intercepts these
before they enter the context budget and compresses them:
- JSON arrays truncated to the 3 most important items
- Text truncated at sentence boundaries
- Structural metadata preserved

**Result**: 60–80% token reduction on typical tool outputs.

### 💾 Session Persistence & Crash Recovery
Enable with `--persist`. Session state is saved to SQLite every N turns. If nexus-serve
crashes or you restart it, every active session is restored from disk automatically.

```bash
nexus-serve --backend-url http://localhost:11434 --backend-type ollama \
            --persist --db-path sessions.db --persist-every 5
```

### 🧠 Long-Term Knowledge Base (Cross-Session Memory)
Important facts (variable assignments, function definitions, configuration values) that
appear frequently across your session are automatically extracted and saved to a SQLite
knowledge base at session end. On the next session, the most relevant facts are injected
back into Zone P so your AI starts with context from previous work.

```bash
nexus-serve --backend-url http://localhost:11434 --backend-type ollama \
            --ltkb-db knowledge.db
```

### 📊 Real-Time Observability Dashboard
A professional dark-mode dashboard served at `/dashboard`. No configuration needed —
it's always on. Metrics update live via Server-Sent Events (SSE).

**Panels:**
- Pipeline latency (ms) with rolling sparkline
- Token budget usage gauge (used / total)
- KV-cache hit rate (%)
- Memory pool size (active WWW tuples)
- Context graph topology (nodes and edges)
- Tool call interception log
- Chunk boundary event feed
- Long-term knowledge base fact feed

### 🔤 LLM-Agnostic Tokenizer
Automatically detects which tokenizer to use based on your model name:
- `qwen*` → Qwen tokenizer
- `llama*`, `mistral*` → LLaMA tokenizer
- `gemma*` → Gemma tokenizer
- `phi*` → Phi tokenizer
- Everything else → tiktoken `cl100k_base` fallback

---

## Embedding in Your Own FastAPI App

Instead of using the CLI, you can embed nexus-context as a library:

```python
import uvicorn
from nexus_context import create_app

app = create_app(
    backend_url="http://localhost:11434",
    backend_type="ollama",
    block_size=16,         # KV block size (match your backend config)
    total_budget=8192,     # max token budget
    persist=True,          # enable SQLite session recovery
    db_path="sessions.db",
    ltkb_db="knowledge.db",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

## Install Options

The core package is intentionally lightweight (~50 MB). ML models are optional extras
so you don't have to download gigabytes just to try it.

```bash
# Core only (no ML models required — uses tiktoken for token counting)
pip install nexus-context

# Add spaCy for natural-language coreference resolution in the graph
pip install "nexus-context[nlp]"

# Add semantic embeddings for more accurate context relevance scoring
pip install "nexus-context[embeddings]"

# Add multi-language AST parsing (Python/SQL/Bash/JavaScript)
pip install "nexus-context[parsers]"

# Everything
pip install "nexus-context[full]"

# Development (testing, linting, type-checking, build tools)
pip install "nexus-context[dev]"
```

---

## CLI Reference

```
nexus-serve [OPTIONS]

Server options:
  --host STR              Bind host (default: 0.0.0.0)
  --port INT              Bind port (default: 9000)
  --log-level LEVEL       Logging verbosity: debug|info|warning (default: info)

Backend options:
  --backend-url URL       Local SLM server URL (default: http://localhost:8000)
  --backend-type TYPE     vllm | sglang | ollama (default: vllm)

Context budget options:
  --block-size INT        KV block size in tokens: 16 or 32 (default: 16)
  --total-budget INT      Max context window tokens (default: 4096)

Session persistence options (Feature B):
  --persist               Enable SQLite-backed session crash recovery
  --db-path PATH          Session database file (default: nexus_sessions.db)
  --persist-every INT     Save every N turns (default: 5)

Long-term knowledge base options (Feature I):
  --ltkb-db PATH          Knowledge base file (default: nexus_ltkb.db)
  --no-ltkb               Disable cross-session knowledge base
```

---

## API Reference

### `nexus_context.create_app()`

```python
from nexus_context import create_app

app = create_app(
    backend_url="http://localhost:8000",  # str
    backend_type="vllm",                  # "vllm" | "sglang" | "ollama"
    block_size=16,                        # int: 16 or 32
    total_budget=4096,                    # int: max tokens
    persist=False,                        # bool: enable session persistence
    db_path="nexus_sessions.db",          # str: SQLite path for sessions
    ltkb_db="nexus_ltkb.db",             # str: SQLite path for knowledge base
) -> FastAPI
```

Returns a fully configured `FastAPI` application. Mount it with any ASGI server.

### Session Tracking

Add `X-Session-ID` header to track conversation sessions:

```python
response = client.chat.completions.create(
    model="my-model",
    messages=[...],
    extra_headers={"X-Session-ID": "unique-session-id"},
)
```

Sessions are tracked in-memory by default. Use `--persist` for crash-recovery.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint (proxied + managed) |
| `GET` | `/v1/models` | Lists available models from backend |
| `DELETE` | `/nexus/session/{id}` | Clear a session's in-memory state |
| `GET` | `/nexus/health` | Health check |
| `GET` | `/dashboard` | Real-time observability dashboard (HTML) |
| `GET` | `/dashboard/stream` | Server-Sent Events stream for dashboard |
| `GET` | `/dashboard/api/state` | Current state snapshot (JSON) |

---

## Benchmarks

Tested on a local machine with Ollama + qwen2.5-coder:7b, 50-turn coding session:

| Metric | Without nexus-context | With nexus-context |
|---|---|---|
| NameError rate from context compaction | 88% of long sessions | 0% |
| KV-cache hit rate (system prompt) | ~30% (random) | >95% |
| TTFT overhead per turn (from nexus) | — | <1ms P95 |
| Token budget usage at turn 50 | Exceeds limit → crash | Managed within budget |
| Tool output tokens (avg) | 2,400 raw | 480 compressed |

---

## Architecture Deep-Dive

### Zone P/T/R Segmentation

```
Full Conversation Context
├── Zone P  (System Prompt — block-aligned, SHA-256 locked, never modified)
├── Zone T  (Turn History — submodular compaction when over budget)
│   ├── turn_0: "Define DB_HOST..."       ← safe to prune IF no references downstream
│   ├── turn_1: "Now write connect_db..."  ← has AST reference to DB_HOST → must keep
│   └── turn_N: ...
└── Zone R  (Current Request — always kept 100% intact)
```

### AST Dependency Graph

For each code block in the conversation, nexus-context builds a directed graph:

```
AST_ASSIGNMENT(DB_HOST)  ──→  AST_ASSIGNMENT(conn_str)
                                      │
                                      ▼
AST_IMPORT(psycopg2)     ──→  AST_FUNCDEF(connect_db)
                                      │
                                      ▼
                               AST_CALL(connect_db)   ←── this is in the current prompt
```

When budget is exceeded, the solver selects which turns to keep. The constraint:
`connect_db` cannot be kept unless `DB_HOST`, `conn_str`, `psycopg2`, and `connect_db`'s
definition are all also kept.

### Submodular Optimization

Compaction is formulated as:

```
maximize  f(S) = Relevance(S) + β·Coverage(S) - γ·DanglingPenalty(S)
subject to  token_cost(S) ≤ budget
```

With γ set to infinity, the dangling penalty makes it mathematically impossible to
select a node without all its antecedent definitions.

---

## Supported Backends

| Backend | Status | Notes |
|---|---|---|
| [Ollama](https://ollama.ai) | ✅ Fully supported | Use `--backend-type ollama` |
| [vLLM](https://vllm.ai) | ✅ Fully supported | Use `--backend-type vllm` |
| [SGLang](https://sgl-project.github.io) | ✅ Fully supported | Use `--backend-type sglang` |
| Any OpenAI-compatible server | ✅ Works | Use `--backend-type vllm` |
| OpenAI API (cloud) | ⚠️ Works but not recommended | Designed for local deployment |

---

## Contributing

```bash
# Clone the repository
git clone https://github.com/laxmikant2806/Nexus-Context.git
cd Nexus-Context

# Install in development mode with all extras
pip install -e ".[dev,full]"

# Run the test suite
pytest --no-cov -q

# Run the linter
ruff check src/ tests/

# Run type checking
mypy src/
```

Pull requests are welcome. Please open an issue first to discuss major changes.

---

## Changelog

See [CHANGELOG.md](https://github.com/laxmikant2806/Nexus-Context/blob/main/CHANGELOG.md)
for a full list of changes per version.

---

## License

MIT License © 2026 Laxmikant Bhagat

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software.

---

## Links

- **PyPI**: https://pypi.org/project/nexus-context/
- **GitHub**: https://github.com/laxmikant2806/Nexus-Context
- **Issues**: https://github.com/laxmikant2806/Nexus-Context/issues
- **Dashboard docs**: https://github.com/laxmikant2806/Nexus-Context/blob/main/docs/how_it_works_simple_to_tech.md
