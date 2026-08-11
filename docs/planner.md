# Nexus-Context: Development Planner

> **Version**: 0.1.0-draft
> **Status**: Planning Phase
> **Last Updated**: 2026-08-11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Performance Thresholds and Success Criteria](#2-performance-thresholds-and-success-criteria)
3. [External Dependency Catalog](#3-external-dependency-catalog)
4. [Risk Register](#4-risk-register)
5. [Phase 1: Block Alignment and Differential Segmentation](#5-phase-1-block-alignment-and-differential-segmentation)
6. [Phase 2: Dependency Graphing and Referential Guard](#6-phase-2-dependency-graphing-and-referential-guard)
7. [Phase 3: WWW Memory Governance](#7-phase-3-www-memory-governance)
8. [Phase 4: Middleware Integration and Proxy Server](#8-phase-4-middleware-integration-and-proxy-server)
9. [Phase 5: Verification and Benchmark Suite](#9-phase-5-verification-and-benchmark-suite)
10. [Milestone Gantt Timeline](#10-milestone-gantt-timeline)
11. [Complexity Metrics Summary](#11-complexity-metrics-summary)

---

## 1. Executive Summary

Nexus-Context is a five-phase project. Each phase builds directly on the previous phase's
deliverables, with no phase requiring rollback or rework of earlier stages assuming the
verification targets defined in this document are met.

| Phase | Name                              | Duration | Complexity | Primary Risk                  |
|-------|-----------------------------------|----------|------------|-------------------------------|
| 1     | Block Alignment & Segmentation    | 1 week   | Low        | Tokenizer compatibility       |
| 2     | Dependency Graphing & Guard       | 3 weeks  | High       | Tree-Sitter grammar coverage  |
| 3     | WWW Memory Governance             | 2 weeks  | Medium     | AST extraction completeness   |
| 4     | Middleware Integration            | 1 week   | Low–Medium | Streaming SSE pass-through    |
| 5     | Verification & Benchmarks         | 2 weeks  | Medium     | Benchmark reproducibility     |

**Total estimated duration**: 9 weeks (sequential), 7 weeks (Phase 2 and 3 parallelized).

---

## 2. Performance Thresholds and Success Criteria

These are **hard requirements**. The project is not complete if any threshold is unmet.

### 2.1 Latency Thresholds

| Operation                                | Target         | Measurement Method                      |
|------------------------------------------|----------------|-----------------------------------------|
| Block alignment (Zone P padding)         | < 2ms          | `time.perf_counter()` around tokenizer  |
| Zone segmentation (P/T/R partitioning)   | < 1ms          | `time.perf_counter()` slice operation   |
| AST graph construction (4,000 tokens)    | < 15ms         | `time.perf_counter()` around full build |
| Submodular solver (100 nodes, B=2048)    | < 50ms         | `time.perf_counter()` around solve loop |
| WWW tuple extraction (per turn)          | < 5ms          | `time.perf_counter()` around www_parser |
| Memory weight scoring (100 tuples)       | < 2ms          | Vectorized numpy operation              |
| Full pipeline overhead (per request)     | < 80ms P95     | FastAPI middleware timing header        |

**Rationale**: At 80ms pipeline overhead, the user perceives no additional latency in interactive
sessions where model TTFT is typically 200ms–2,000ms. The 15ms graph construction target is
derived from the requirement that processing must complete in less than 10% of a typical 150ms
TTFT on a 7B model.

### 2.2 Correctness Thresholds

| Metric                                         | Target | Acceptable Floor |
|------------------------------------------------|--------|------------------|
| Referential dangling rate (compressed output)  | 0.0%   | 0.0% (hard zero) |
| Prefix KV cache hit rate improvement           | > 85%  | > 75%            |
| WWW extraction completeness (Python AST)       | > 95%  | > 90%            |
| WWW extraction completeness (SQL DDL/DML)      | > 95%  | > 90%            |
| Runtime execution failure rate vs. baseline    | ≤ 105% | ≤ 110%           |

**Note on dangling rate**: The 0% target is achievable by design — the infinite penalty
coefficient γ mathematically prevents any dangling node from being included in the selection.
This is a correctness guarantee, not a probabilistic target. Any test failure here indicates a
bug in the transitive closure computation.

### 2.3 Memory Efficiency Thresholds

| Metric                              | Target        |
|-------------------------------------|---------------|
| Episodic-to-semantic compression    | > 4:1 for code turns (Python/SQL) |
| Memory pool token overhead          | < 10% of Zone T budget |
| Cross-session memory file size      | < 1 MB per 1,000 turns |

---

## 3. External Dependency Catalog

### 3.1 Core Dependencies

| Package                      | Version   | Purpose                                      | License    |
|------------------------------|-----------|----------------------------------------------|------------|
| `tree-sitter`                | >=0.21    | Core AST parsing engine                      | MIT        |
| `tree-sitter-python`         | >=0.21    | Python language grammar                      | MIT        |
| `tree-sitter-sql`            | >=0.2     | SQL language grammar (DDL/DML)               | MIT        |
| `tree-sitter-bash`           | >=0.21    | Bash/shell language grammar                  | MIT        |
| `tree-sitter-javascript`     | >=0.21    | JS/TS language grammar                       | MIT        |
| `spacy`                      | >=3.7     | NLP pipeline for NL turn analysis            | MIT        |
| `en_core_web_trf`            | >=3.7     | Transformer-based spaCy model                | MIT        |
| `coreferee`                  | >=1.4     | Coreference resolution for spaCy             | MIT        |
| `sentence-transformers`      | >=2.7     | Embedding model for Relevance scoring        | Apache-2.0 |
| `fastapi`                    | >=0.111   | ASGI middleware framework                    | MIT        |
| `httpx`                      | >=0.27    | Async HTTP client for backend proxy          | BSD-3      |
| `pydantic`                   | >=2.7     | Data validation and serialization            | MIT        |
| `tiktoken`                   | >=0.7     | OpenAI-compatible BPE tokenizer              | MIT        |
| `transformers`               | >=4.40    | HuggingFace tokenizer (AutoTokenizer)        | Apache-2.0 |
| `numpy`                      | >=1.26    | Vectorized weight computation                | BSD-3      |

### 3.2 Optional / Backend-Specific Dependencies

| Package                       | Purpose                                      | Required For    |
|-------------------------------|----------------------------------------------|-----------------|
| `vllm`                        | vLLM serving backend (dev/test only)         | Phase 5 benchmarks |
| `sglang`                      | SGLang serving backend (dev/test only)       | Phase 5 benchmarks |
| `ollama` (Python client)      | Ollama backend client                        | Phase 4 integration |
| `llama-cpp-python`            | Ollama block size detection                  | Phase 1 (optional) |
| `nomic-embed-text`            | Higher-accuracy embedding model              | Phase 2 (optional) |

### 3.3 Development Dependencies

| Package        | Version  | Purpose                                      |
|----------------|----------|----------------------------------------------|
| `pytest`       | >=8.0    | Test runner                                  |
| `pytest-asyncio` | >=0.23 | Async test support                           |
| `pytest-cov`   | >=5.0    | Coverage measurement                         |
| `hypothesis`   | >=6.100  | Property-based testing for graph algorithms  |
| `ruff`         | >=0.4    | Linting and formatting                       |
| `mypy`         | >=1.10   | Static type checking                         |
| `locust`       | >=2.29   | Load testing for middleware                  |
| `memory-profiler` | >=0.61 | Memory overhead profiling                   |

---

## 4. Risk Register

| Risk ID | Description                                          | Probability | Impact   | Mitigation Strategy                                          |
|---------|------------------------------------------------------|-------------|----------|--------------------------------------------------------------|
| R-001   | Tree-Sitter grammar breaks on malformed agent code   | High        | Medium   | Wrap all parse calls in try/except; degrade to NL_SENTENCE   |
| R-002   | spaCy coreference misidentifies entity antecedents   | Medium      | Low      | Coreference edges are non-critical (E_nl); false positives are safe |
| R-003   | Block size mismatch between nexus.cache and backend  | Medium      | High     | Auto-detect via backend API probe at session start           |
| R-004   | Embedding model too slow for 15ms latency target     | Medium      | High     | Cache embeddings per node_id; lazy computation on first access |
| R-005   | Submodular greedy exceeds 50ms for large graphs      | Low         | Medium   | Lazy greedy with priority queue; node count cap at 500       |
| R-006   | WWW extraction misses side-effectful AST nodes       | Low         | High     | Extensive test coverage; conservative fallback to raw text   |
| R-007   | Cross-session memory file corrupts (concurrent write)| Low         | Medium   | File locking via `filelock`; atomic write-then-rename        |
| R-008   | Tokenizer produces different token IDs than backend  | Medium      | High     | Use backend's actual tokenizer via `/tokenize` API endpoint  |
| R-009   | FastAPI middleware adds > 80ms P95 overhead          | Low         | High     | Profile hot paths; async embedding with cache hit shortcircuit |
| R-010   | License incompatibility in dependency chain          | Very Low    | Medium   | All listed dependencies are permissively licensed            |

---

## 5. Phase 1: Block Alignment and Differential Segmentation

**Goal**: Implement the KV cache zone management layer that guarantees Zone P prefix stability
and defines the token budget allocation for Zones T and R.

**Duration**: 1 week
**Complexity**: Low (algorithmic complexity O(n); primary challenge is tokenizer compatibility)

### 5.1 Deliverables

#### `src/nexus_context/cache/block_align.py`

```
BlockAligner class:
  __init__(model_name: str, backend: Literal["vllm", "sglang", "ollama"])
    → loads tokenizer, detects block_size via backend API probe

  align(system_prompt: str) -> tuple[str, int]:
    → returns (padded_system_prompt, n_padding_tokens)
    → computes L_P = ceil(L_raw / B_block) * B_block
    → appends neutral padding tokens to reach L_P
    → stores SHA256(token_ids[0:L_P]) as zone_p_hash

  tokenize(text: str) -> list[int]:
    → returns token ID list using loaded tokenizer
    → raises TokenizerError if text exceeds model's max_length

  detect_block_size() -> int:
    → for vLLM: parse --block-size from server /v1/models metadata
    → for SGLang: probe server config endpoint
    → for Ollama: return default 32
```

#### `src/nexus_context/cache/differential.py`

```
ZoneSegmenter class:
  __init__(cache_boundary: CacheBoundary)

  segment(messages: list[ChatMessage]) -> ZoneBundle:
    → assigns each message to zone P, T, or R
    → Zone R = last tail_budget tokens of non-system messages
    → Zone T = remaining messages above Zone R up to zone_t_budget
    → Zone P = system message (already aligned)

  graduate_tail_turns(zone_bundle: ZoneBundle, current_turn: int) -> ZoneBundle:
    → moves turns older than tail_retention_turns from R to T candidates
    → returns updated ZoneBundle with revised zone_t_candidates
```

#### `src/nexus_context/cache/schemas.py`

```
ChatMessage(BaseModel): role, content, token_count
ZoneBundle(BaseModel): zone_p, zone_t_candidates, zone_r, boundary
```

### 5.2 Implementation Notes

- **Padding token selection**: For models without `pad_token_id`, use `eos_token_id` with a
  preceding `# ` (hash-space) prefix comment, which prevents the model from treating the EOS
  token as a generation termination signal when it appears in the middle of the context.

- **Block size auto-detection for vLLM**: Make a GET request to `{backend_url}/v1/models` and
  parse the `meta` field if present. If unavailable, fall back to a heuristic: send a 16-token
  prompt and check `/metrics` for `vllm:gpu_cache_usage_perc`.

- **Tokenizer mismatch**: The tokenizer used by nexus.cache must produce the exact same token IDs
  as the backend model. For HuggingFace models, use `AutoTokenizer.from_pretrained(model_name)`.
  Never use tiktoken as a proxy tokenizer for HuggingFace models (different vocabulary tables).

### 5.3 Verification Targets for Phase 1

| Test Case                                      | Expected Result                         |
|------------------------------------------------|-----------------------------------------|
| align("hello world") with block_size=16       | Padded to 16 tokens exactly             |
| align(513_token_prompt) with block_size=16    | Padded to 528 tokens (33 × 16)          |
| segment() with 0 trunk candidates              | ZoneBundle.zone_t_candidates = []       |
| segment() total tokens <= budget              | Always holds (invariant)                |
| zone_p_hash stable across two identical calls  | Equal SHA256 strings                    |

---

## 6. Phase 2: Dependency Graphing and Referential Guard

**Goal**: Build the AST-based context dependency graph and implement the submodular pruning
solver with guaranteed zero referential dangling.

**Duration**: 3 weeks
**Complexity**: High (multi-language AST parsing, graph algorithm implementation, embedding
integration)

### 6.1 Deliverables

#### `src/nexus_context/guard/ast_graph.py`

```
ContextGraphBuilder class:
  __init__(languages: list[str] = ["python", "sql", "bash", "json"])
    → initializes Tree-Sitter parsers for each language
    → loads spaCy model for NL turns
    → initializes coreferee pipeline

  build(turns: list[Turn]) -> ContextGraph:
    → entry point: processes each turn, dispatches to _parse_code or _parse_nl
    → merges all node and edge sets into a single ContextGraph
    → computes transitive closure of edge set

  _parse_code(turn: Turn, language: str) -> tuple[list[ContextNode], list[DependencyEdge]]:
    → Tree-Sitter parse → AST visitor
    → extract: assignments, funcdef, classdef, imports, calls
    → build E_code edges: match names in store to names in load contexts

  _parse_nl(turn: Turn) -> tuple[list[ContextNode], list[DependencyEdge]]:
    → spaCy NLP pipeline
    → coreferee.coref_chains → extract antecedent-anaphor pairs
    → build E_nl edges

  _compute_transitive_closure(edges: list[DependencyEdge]) -> list[DependencyEdge]:
    → Floyd-Warshall over node adjacency matrix (feasible for n <= 500 nodes)
    → alternatively: BFS from each node for sparse graphs
    → marks transitive edges with is_transitive = True
```

**AST Visitor Pattern** (Python grammar, key node types):

```
class PythonASTVisitor:
  _scope_stack: list[str]   # tracks current scope for scope_path computation
  _binding_store: dict[str, str]  # name → node_id of most recent binding

  def visit_assignment(node):
    # node.children[0] = identifier (LHS), node.children[2] = expression (RHS)
    name = node.children[0].text
    node_id = create_node(AST_ASSIGNMENT, name, scope_path=current_scope)
    if name in _binding_store:
      # superseded binding: mark old node as overridden
      old_node_id = _binding_store[name]
      mark_overridden(old_node_id)
    _binding_store[name] = node_id

  def visit_identifier(node, context: Literal["load", "store"]):
    if context == "load" and node.text in _binding_store:
      # create DEF→USE edge
      source_id = _binding_store[node.text]
      target_id = create_node(AST_CALL, node.text)
      add_edge(source_id, target_id, VAR_DEF_TO_REF)
```

#### `src/nexus_context/guard/submodular.py`

```
SubmodularSolver class:
  __init__(embedding_model: str = "all-MiniLM-L6-v2", alpha: float = 0.5, beta: float = 0.5)
    → loads sentence-transformer embedding model
    → initializes embedding cache dict[node_id → np.ndarray]

  solve(graph: ContextGraph, budget: int, query: str) -> CompactionResult:
    → embed all nodes and query
    → compute pairwise similarity matrix (cached)
    → run density-greedy loop with priority queue
    → enforce ancestor forcing on each selection
    → return CompactionResult

  _embed_nodes(nodes: list[ContextNode]) -> np.ndarray:
    → batch encode with sentence-transformer
    → cache results by node_id

  _marginal_gain(v: ContextNode, S: set[str], nodes: dict, sim_matrix: np.ndarray,
                  query_sims: np.ndarray, gamma: float) -> float:
    → compute: alpha * query_sim(v) + beta * coverage_gain(v, S) - gamma * dangling_count(v, S)
    → coverage_gain = sum of max(0, sim(v, w) - max_{u in S} sim(u, w)) for all w

  _force_ancestors(v_id: str, graph: ContextGraph, S: set[str]) -> list[str]:
    → return sorted list of v's transitive ancestors not in S
    → sorted by turn_index ascending (include definitions before uses)
```

#### `src/nexus_context/guard/schemas.py`

Pydantic models: `ContextNode`, `DependencyEdge`, `ContextGraph`, `CompactionResult`
(full definitions in `docs/architecture.md`, Section 4).

### 6.2 Implementation Notes

- **Scope tracking**: Python AST scopes are tracked via a stack: `module → ClassDef → FunctionDef
  → inner FunctionDef`. Each function definition pushes its name onto the stack; each close pops.

- **Name shadowing**: When a name is re-assigned within an inner scope, the binding_store entry
  is scope-qualified: `{scope}.{name}`. This prevents incorrect cross-scope dangling detection.

- **SQL parsing**: Tree-Sitter SQL grammar tokenizes identifiers. For SELECT queries, track
  referenced table names. For CREATE/ALTER/DROP, track schema-level bindings.

- **Embedding cache**: Node embeddings are computed once per session and cached in memory. For
  sessions exceeding 1,000 nodes, use approximate nearest neighbor (HNSW via `hnswlib`) for
  coverage computation to maintain latency targets.

- **Lazy greedy optimization**: Maintain a priority queue sorted by last-computed marginal gain
  upper bounds. Before selecting a node, recompute its gain; if still the maximum, select it.
  This reduces the expected number of gain computations from O(n^2 k) to O(n log n).

### 6.3 Verification Targets for Phase 2

| Test Case                                          | Expected Result                           |
|----------------------------------------------------|-------------------------------------------|
| 3-turn DB setup example (Section 2.4 of overview)  | Turn 2 always selected when Turn 3 is     |
| Single-turn code with no dependencies              | No edges created                          |
| Import used 3 turns later                          | import → use edge in E*                   |
| Python function defined and called across turns    | func_def → call edge; both selected       |
| Budget forces exclusion of unreferenced turns      | Unreferenced turns pruned first           |
| Graph with 200 nodes, solve in < 50ms              | Wall-clock < 50ms (P95 across 100 runs)   |
| Graph construction: 4,000 tokens in < 15ms         | Wall-clock < 15ms (P95 across 100 runs)   |

---

## 7. Phase 3: WWW Memory Governance

**Goal**: Implement the memory tuple extractor and decay engine that converts pruned episodic
turns into compact semantic state-mutation records.

**Duration**: 2 weeks
**Complexity**: Medium (AST extraction is well-defined; primary challenge is coverage completeness)

### 7.1 Deliverables

#### `src/nexus_context/memory/www_parser.py`

```
WWWParser class:
  __init__(languages: list[str] = ["python", "sql", "bash"])
    → shares Tree-Sitter parser instances with ast_graph (via shared parser pool)
    → spaCy pipeline for NL extraction

  extract(turn: Turn, session_id: str, current_turn_index: int,
          d_max: int) -> list[MemoryTuple]:
    → dispatches to _extract_code or _extract_nl based on turn content type
    → may return multiple MemoryTuples per turn (e.g., multiple assignments)

  _extract_code(turn: Turn, ...) -> list[MemoryTuple]:
    → Tree-Sitter AST walk
    → for each mutation node, construct WhatDelta:
        * VAR_ASSIGN: {target_name: lhs, new_value_repr: rhs_str}
        * FUNC_DEF: {target_name: func_name, new_value_repr: signature}
        * SCHEMA_CHANGE: {target_name: table_name, mutation_type: op}
    → compute scope_path from AST stack
    → compute ast_depth_score = (d_max - depth) / d_max

  _extract_nl(turn: Turn, ...) -> list[MemoryTuple]:
    → spaCy NER: extract entities (Person, ORG, LOC, PRODUCT, etc.)
    → dependency parse: extract SVO (subject-verb-object) triples
    → construct NL_STATEMENT WhatDelta for each triple
```

#### `src/nexus_context/memory/decay.py`

```
MemoryPool class:
  __init__(session_id: str, lambda_: float = 0.05, eta: float = 0.5,
           persist_path: Optional[Path] = None)
    → loads existing pool from JSONL if persist_path provided

  add(tuples: list[MemoryTuple]) -> None:
    → appends new tuples to pool
    → deduplicates by (target_name, scope_path) keeping most recent

  update_weights(current_turn: int) -> None:
    → recomputes W(t_i, s_i) for all non-pinned tuples
    → sorts pool by retention_weight descending

  pin(memory_id: str) -> None:
    → sets is_pinned = True on specified memory
    → pinned memories are always included regardless of weight

  select(budget_tokens: int) -> list[MemoryTuple]:
    → knapsack selection: greedily pick highest-weight tuples within budget
    → pinned tuples always included first
    → returns sorted by when ascending (chronological order for context)

  serialize_for_context(selected: list[MemoryTuple]) -> str:
    → formats tuples as compact JSON block for context injection:
      "<!-- NEXUS_MEMORY\n{json_array}\n-->"
    → compact format example:
      {"w":"tool:execute_python","δ":"DB_HOST=prod.db.internal","t":3,"@":"module"}

  save(path: Path) -> None:
    → atomic write to JSONL (write to .tmp then rename)
    → one tuple per line

  load(path: Path) -> None:
    → parse JSONL, validate each line as MemoryTuple
    → skip corrupted lines with WARNING log
```

### 7.2 Deduplication Strategy

When a variable is reassigned multiple times across turns (a common pattern in agentic loops),
the memory pool should retain only the **most recent** assignment for that (name, scope_path)
pair, unless the prior value is still referenced by a retained dependency edge.

```
Deduplication rule:
  key = (what.target_name, where)   # (name, scope)
  if key in pool:
    old = pool[key]
    if old.is_pinned:
      keep both (versioned: append "_v{when}" to key)
    else:
      if old.when < new.when:
        replace old with new
      else:
        keep old (out-of-order update, warn)
  else:
    add new to pool
```

### 7.3 Context Injection Format

The serialized memory block is injected between Zone P and Zone T in the assembled payload:

```
[SYSTEM_PROMPT + PADDING]
<!-- NEXUS_MEMORY
[{"w":"agent","δ":{"type":"schema_change","target":"users","op":"CREATE"},"t":3,"@":"module"},
 {"w":"tool:execute_python","δ":{"type":"var_assign","target":"conn","new":"psycopg2.conn"},"t":5,"@":"module"},
 {"w":"user","δ":{"type":"nl_statement","target":"task","new":"Deploy to staging"},"t":12,"@":"session"}]
-->
[ZONE T RECONSTRUCTED TEXT]
[ZONE R VERBATIM TURNS]
```

The model is instructed in the system prompt (Zone P) to treat `NEXUS_MEMORY` blocks as
authoritative state history and to prefer their content over any conflicting in-context
statements.

### 7.4 Verification Targets for Phase 3

| Test Case                                              | Expected Result                        |
|--------------------------------------------------------|----------------------------------------|
| Python turn: 3 assignments                             | 3 MemoryTuples with VAR_ASSIGN type    |
| Python turn: function def + 2 assignments              | 3 MemoryTuples                         |
| SQL turn: CREATE TABLE users (...)                     | 1 MemoryTuple with SCHEMA_CHANGE/CREATE|
| SQL turn: UPDATE + WHERE                               | 1 MemoryTuple with SCHEMA_CHANGE/UPDATE|
| Duplicate variable reassignment (same scope)           | Only most recent retained in pool      |
| Weight at turn 0 (d=0, eta=0.5)                       | W = 1.0 * (1 + 0.5 * 1.0) = 1.5      |
| Weight at turn 14 (lambda=0.05, d=0)                  | W ≈ 0.5 * 1.5 = 0.75 (≈ half-life)   |
| Memory 4:1 compression ratio (Python code turn)        | Token count of tuple < 25% of turn    |

---

## 8. Phase 4: Middleware Integration and Proxy Server

**Goal**: Assemble the three subsystems into a deployable FastAPI middleware that transparently
wraps any OpenAI-compatible backend.

**Duration**: 1 week
**Complexity**: Low–Medium (integration work; key challenge is SSE streaming pass-through)

### 8.1 Deliverables

#### `src/nexus_context/cache/middleware.py`

```
NexusContextMiddleware (FastAPI app):
  startup():
    → load default configuration from nexus_config.yaml or environment variables
    → initialize shared BlockAligner, ZoneSegmenter, ContextGraphBuilder,
      SubmodularSolver, WWWParser, and MemoryPool instances
    → connect to backend server (health check)

  POST /v1/chat/completions:
    → parse request body as ChatCompletionRequest
    → extract or generate session_id
    → load per-session state (CacheBoundary, ContextGraph, MemoryPool)
    → run NexusContextProcessor pipeline (Stages 1–5)
    → forward transformed request to backend
    → if stream=True: proxy SSE chunks transparently
    → if stream=False: forward JSON response directly
    → update per-session state with new turn
    → return response with X-Nexus-Context-Stats header

  GET /nexus/health:
    → return {"status": "ok", "sessions": n, "backend": backend_url}

  GET /nexus/session/{session_id}/stats:
    → return CacheBoundary, CompactionResult, MemoryPool summary for session

  DELETE /nexus/session/{session_id}:
    → clear per-session state and memory pool
```

**Configuration** (via `nexus_config.yaml`):

```yaml
backend:
  url: "http://localhost:8000"        # vLLM / SGLang / Ollama URL
  type: "vllm"                        # vllm | sglang | ollama

cache:
  block_size: 16                      # PagedAttention block size
  tail_budget_tokens: 1024            # Zone R budget
  tail_retention_turns: 3             # turns to keep in Zone R before graduation

guard:
  alpha: 0.5                          # Relevance weight
  beta: 0.5                           # Coverage weight
  embedding_model: "all-MiniLM-L6-v2"

memory:
  lambda: 0.05                        # Temporal decay constant
  eta: 0.5                            # AST depth amplification
  budget_fraction: 0.15               # B_mem = budget_fraction * B_total
  persist: false                      # Cross-session memory persistence
```

### 8.2 Streaming SSE Pass-Through

When `stream=True`, the backend returns Server-Sent Events. The middleware must:
1. Forward the transformed request to the backend as a streaming request.
2. Yield SSE chunks from the backend directly to the client without buffering.
3. Reconstruct the full response content from chunks for session state update.
4. Update session state after the stream completes (in a background task).

This is implemented using FastAPI's `StreamingResponse` with `httpx.AsyncClient.stream()`.

### 8.3 Session State Storage

Session state (CacheBoundary, ContextGraph node set, MemoryPool) is stored in-process via a
`SessionStore` dict keyed by session_id. For multi-worker deployments (Gunicorn + uvicorn),
session state must be externalized to Redis or a shared volume.

For Phase 4 (MVP), in-process storage is acceptable. Multi-worker support is a Phase 5+
enhancement.

### 8.4 Verification Targets for Phase 4

| Test Case                                        | Expected Result                           |
|--------------------------------------------------|-------------------------------------------|
| Request forwarded to backend                     | Identical JSON (minus nexus processing)   |
| stream=True: SSE chunks proxied correctly        | Client receives identical event stream    |
| X-Nexus-Context-Stats header present             | Contains pipeline timing breakdown        |
| Total overhead P95 < 80ms (mocked backend)       | Measured via pytest-benchmark             |
| Session state persists across 10 requests        | Zone P hash unchanged across all turns    |
| /nexus/health returns 200                        | {"status": "ok", ...}                     |

---

## 9. Phase 5: Verification and Benchmark Suite

**Goal**: Validate all performance thresholds, referential integrity guarantees, and cache hit
rate improvements against the synthetic and real agentic task benchmarks.

**Duration**: 2 weeks
**Complexity**: Medium (primarily test authoring and benchmark harness setup)

### 9.1 Unit Test Coverage Targets

| Module                       | Target Line Coverage | Target Branch Coverage |
|------------------------------|----------------------|------------------------|
| nexus.cache.block_align      | 100%                 | 100%                   |
| nexus.cache.differential     | 95%                  | 90%                    |
| nexus.guard.ast_graph        | 90%                  | 85%                    |
| nexus.guard.submodular       | 95%                  | 90%                    |
| nexus.memory.www_parser      | 90%                  | 85%                    |
| nexus.memory.decay           | 95%                  | 90%                    |
| nexus.cache.middleware       | 85%                  | 80%                    |

### 9.2 Benchmark Suite: NCATS-v1

**Nexus-Context Synthetic Agentic Task Suite** (NCATS-v1) consists of 12 task categories:

| Category | Turns | Languages | Key Failure Mode Tested |
|----------|-------|-----------|-------------------------|
| DB Schema Setup | 15–25 | Python + SQL | VAR_DEF_TO_REF dangling |
| REST API Chaining | 20–30 | Python + JSON | TOOL_RETURN_DEP dangling |
| File System Ops | 10–20 | Python + Bash | FS mutation extraction |
| Code Refactoring | 25–40 | Python | FUNC_DEF_TO_CALL + CLASS_DEF |
| Multi-step SQL Migrations | 15–25 | SQL | SCHEMA_CHANGE chain |
| Config Management | 10–15 | Python + JSON | Import + VAR_ASSIGN chain |
| NL-to-Code Task Spec | 5–10 | NL → Python | COREF_ANTECEDENT |
| Debug Loop | 20–35 | Python | Retry storm prevention |
| Data Pipeline | 30–50 | Python + SQL + Bash | Cross-language dependencies |
| Agent Self-Modification | 15–25 | Python | CLASS_DEF recursive |
| Multi-Agent Handoff | 25–40 | Python + JSON | Cross-turn TOOL_RETURN_DEP |
| Long-Horizon Planning | 50–100 | Python + NL | Memory decay correctness |

### 9.3 Property-Based Tests (Hypothesis)

```python
# Dangling invariant: MUST hold for all graphs and budgets
@given(st.lists(Turn.strategy(), min_size=2, max_size=50),
       st.integers(min_value=100, max_value=10000))
def test_no_dangling_after_compaction(turns, budget):
    graph = builder.build(turns)
    result = solver.solve(graph, budget, query="task")
    selected = set(result.selected_node_ids)
    for edge in graph.edges:
        if edge.target_id in selected:
            assert edge.source_id in selected, (
                f"Dangling: {edge.target_id} selected but {edge.source_id} not"
            )
```

### 9.4 KV Cache Hit Rate Benchmark

The cache hit rate benchmark runs a 50-turn agentic session against a live vLLM instance and
measures prefix cache hit rate via `vllm:cache_hit_rate` Prometheus metric:

```
Baseline (no nexus): run 50-turn session, record H_baseline
Nexus-enabled:       run identical 50-turn session with middleware, record H_nexus

Assert: H_nexus > 0.85
Assert: H_nexus > H_baseline * 1.1   (at least 10% relative improvement)
```

---

## 10. Milestone Gantt Timeline

```
Week:  1    2    3    4    5    6    7    8    9
       │    │    │    │    │    │    │    │    │
P1:    ████░                                     Block Alignment & Segmentation
P2:         █████████████░                       Dependency Graphing & Guard
P3:              ██████████░                     WWW Memory Governance (parallel with P2 weeks 2-3)
P4:                         ████░                Middleware Integration
P5:                              █████████░      Verification & Benchmarks
       │    │    │    │    │    │    │    │    │
       ◆              ◆              ◆         ◆
     Start         P1+P2          P4         Done
                  complete       done
```

---

## 11. Complexity Metrics Summary

| Phase | Files | LOC (est.) | Cyclomatic Complexity (avg) | Key Algorithm                        |
|-------|-------|------------|-----------------------------|--------------------------------------|
| 1     | 3     | ~400       | 3.2                         | Ceiling division, tokenizer BPE       |
| 2     | 3     | ~900       | 7.8                         | Tree-Sitter visitor, lazy greedy, BFS |
| 3     | 2     | ~600       | 5.1                         | AST walk, fractional knapsack         |
| 4     | 1     | ~350       | 4.5                         | FastAPI ASGI, httpx streaming         |
| 5     | 3     | ~800       | 2.9                         | Hypothesis strategies, pytest fixtures|
| **Total** | **12** | **~3,050** | **4.7** | — |

> LOC estimates are for production code only, excluding test files, documentation, and
> configuration. All estimates assume Pythonic, well-documented code with type annotations.
