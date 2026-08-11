# Nexus-Context

> **Lightweight Python framework for referential integrity, KV cache alignment, and WWW memory
> governance in local Small Language Model (SLM) deployments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## The Problem

When running autonomous coding agents on local SLMs (vLLM, SGLang, Ollama), three compounding
failure modes emerge as context grows:

1. **Referential Dangling**: Standard prompt compression tools (LLMLingua, Selective Context)
   prune variable definitions and tool schemas while retaining code that depends on them. The
   result: `NameError: name 'DB_HOST' is not defined` — over 88% of all compression-induced
   errors in agentic sessions are of this class.

2. **KV Cache Invalidation Tax**: Any modification to the middle of a PagedAttention context
   invalidates the prefix hash for all subsequent tokens, forcing full recomputation. On a 7B
   model, this adds ~1.1 seconds of TTFT penalty per turn — accumulating to minutes of
   wasted GPU time per session.

3. **Episodic Memory Bloat**: Agents accumulate verbose tool outputs and conversational history
   that contain only a fraction of actionable state. A 127-token "CREATE TABLE..." execution
   record can be losslessly compressed to a 31-token semantic mutation tuple.

## The Solution

Nexus-Context is a **transparent middleware layer** that sits between your agent and its local
SLM backend. It transforms context payloads through three cooperating subsystems:

```
Agent ──► Nexus-Context Middleware ──► vLLM / SGLang / Ollama
               │
               ├── nexus.guard:  AST dependency graph + submodular pruning
               ├── nexus.cache:  Block-aligned prefix zone locking
               └── nexus.memory: WWW episodic-to-semantic decay
```

## Key Guarantees

| Guarantee | Target | Mechanism |
|---|---|---|
| **Zero referential dangling** | 0% NameError rate | γ→∞ submodular penalty on orphaned nodes |
| **Prefix KV cache preservation** | >85% hit rate | Block-aligned Zone P locked for session lifetime |
| **Memory compression** | >4:1 ratio | WWW semantic mutation tuple extraction |
| **Low overhead** | <80ms P95 per request | Lazy greedy + embedding cache + async pipeline |
| **Backend agnostic** | vLLM, SGLang, Ollama | OpenAI `/v1/chat/completions` proxy |

## Architecture Overview

```
Raw Payload
    │
    ▼
FastAPI Middleware (nexus.cache.middleware)
    │
    ├─[Stage 1]─► Block Alignment (nexus.cache.block_align)
    │               Pad system prompt to B_block boundary. Freeze Zone P forever.
    │
    ├─[Stage 2]─► Zone Segmentation (nexus.cache.differential)
    │               Partition: [Zone P | Zone T | Zone R]
    │
    ├─[Stage 3]─► AST Dependency Graph (nexus.guard.ast_graph)
    │               Tree-Sitter (Python/SQL/Bash) + spaCy coreference
    │
    ├─[Stage 4]─► Submodular Pruning (nexus.guard.submodular)
    │               max f(S) = α·Relevance + β·Coverage − γ·DanglingPenalty
    │
    ├─[Stage 5]─► WWW Memory Injection (nexus.memory)
    │               Pruned turns → ⟨Who, What, When, Where⟩ tuples
    │               W(t,s) = exp(−λ(T−t))·(1 + η·AST_Depth(s))
    │
    └────────────► Transformed Payload ──► vLLM Server
                    (prefix cache hit guaranteed on Zone P)
```

## Quick Start

```bash
# Install
pip install nexus-context

# Download spaCy model
python -m spacy download en_core_web_trf
python -m coreferee install en

# Start middleware (proxying to local vLLM on port 8000)
nexus-serve --backend-url http://localhost:8000 --backend-type vllm --port 9000

# Point your agent at port 9000 instead of 8000
# Zero code changes required in your agent
```

## Configuration

Create `nexus_config.yaml`:

```yaml
backend:
  url: "http://localhost:8000"   # vLLM / SGLang / Ollama URL
  type: "vllm"                   # vllm | sglang | ollama

cache:
  block_size: 16                 # PagedAttention block size (must match backend)
  tail_budget_tokens: 1024       # Zone R verbatim tail size
  tail_retention_turns: 3        # Turns to keep in Zone R before graduation

guard:
  alpha: 0.5                     # Relevance weight in submodular objective
  beta: 0.5                      # Coverage weight in submodular objective
  embedding_model: "all-MiniLM-L6-v2"

memory:
  lambda: 0.05                   # Temporal decay constant (half-life ~13.9 turns)
  eta: 0.5                       # AST depth amplification factor
  budget_fraction: 0.15          # Memory block uses 15% of total token budget
  persist: false                 # Cross-session memory persistence
```

## Mathematical Foundations

### Referential Integrity (nexus.guard)

Context compression is formulated as submodular maximization:

```
max_{S ⊆ V, tokens(S) ≤ B}  f(S) = α·Relevance(S) + β·Coverage(S) − γ·DanglingPenalty(S)

DanglingPenalty(S) = Σ_{(u,v) ∈ E*} I(v ∈ S ∧ u ∉ S)

γ → ∞  (hard constraint: no orphaned nodes permitted)
```

### Block Alignment (nexus.cache)

```
L_P = ⌈L_raw / B_block⌉ × B_block    (Zone P padded to block boundary)
h_i = SHA256(T[i·B_block : (i+1)·B_block] ∥ h_{i-1})    (prefix hash chain)
```

### Memory Decay (nexus.memory)

```
W(t_i, s_i) = exp(−λ(T − t_i)) · (1 + η · AST_Depth(s_i))

AST_Depth(s_i) = (D_max − d(s_i)) / D_max    (root scope = 1.0, deep nesting → 0)
```

## Development Phases

| Phase | Subsystem | Status |
|---|---|---|
| 1 | Block Alignment & Zone Segmentation | 🔲 Not started |
| 2 | AST Dependency Graph & Submodular Solver | 🔲 Not started |
| 3 | WWW Memory Extraction & Decay | 🔲 Not started |
| 4 | FastAPI Middleware & Backend Proxy | 🔲 Not started |
| 5 | Verification & Benchmark Suite | 🔲 Not started |

See [`docs/planner.md`](docs/planner.md) for detailed milestone breakdown.

## Documentation

| Document | Description |
|---|---|
| [`docs/overview_and_research.md`](docs/overview_and_research.md) | Theoretical foundations, empirical evidence, and prior art |
| [`docs/architecture.md`](docs/architecture.md) | System design, Pydantic schemas, dataflow diagram |
| [`docs/planner.md`](docs/planner.md) | Phase-by-phase milestones, risk register, complexity metrics |
| [`docs/decisions.md`](docs/decisions.md) | Architecture Decision Records (ADR-001 through ADR-006) |
| [`docs/implementation_stage.md`](docs/implementation_stage.md) | Step-by-step implementation guide with checklists |

## Repository Structure

```
nexus-context/
├── docs/
│   ├── overview_and_research.md   # Theory, empirical data, prior art
│   ├── architecture.md            # System design & Pydantic schemas
│   ├── planner.md                 # Development milestones & risk register
│   ├── decisions.md               # Architecture Decision Records
│   └── implementation_stage.md   # Step-by-step implementation guide
├── src/
│   └── nexus_context/
│       ├── __init__.py
│       ├── guard/
│       │   ├── __init__.py
│       │   ├── ast_graph.py       # Tree-Sitter & spaCy dependency graph builder
│       │   ├── submodular.py      # Budget-constrained greedy submodular solver
│       │   └── schemas.py         # ContextNode, DependencyEdge, ContextGraph, CompactionResult
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── block_align.py     # PagedAttention block padding logic
│       │   ├── differential.py    # Zone P/T/R segmentation engine
│       │   ├── middleware.py      # FastAPI transparent proxy
│       │   └── schemas.py         # ChatMessage, ZoneBundle, CacheBoundary
│       └── memory/
│           ├── __init__.py
│           ├── www_parser.py      # Who-What-When/Where tuple extractor
│           ├── decay.py           # Exponential decay and memory pool management
│           └── schemas.py         # MemoryTuple, WhatDelta, WhoActor
├── tests/
│   ├── test_guard.py
│   ├── test_cache.py
│   └── test_memory.py
├── pyproject.toml
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Research References

- Kwon et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention. SOSP 2023.
- Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference. EMNLP 2023.
- Nemhauser, Wolsey, Fisher (1978). An analysis of approximations for maximizing submodular set
  functions. Mathematical Programming.
- Zheng et al. (2023). SGLang: Efficient Execution of Structured LM Programs. arXiv:2312.07104.

See [`docs/overview_and_research.md`](docs/overview_and_research.md) for the complete bibliography.
