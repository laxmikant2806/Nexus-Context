# Nexus-Context: Architecture Decision Records

> **Version**: 0.1.0-draft
> **Status**: Approved (internal)
> **Last Updated**: 2026-08-11

Each ADR follows the format: **Context → Decision → Rationale → Consequences → Alternatives
Considered**.

---

## ADR-001: Tree-Sitter over Python `ast` for Multi-Language Execution Support

**Status**: Accepted
**Deciders**: Core team
**Date**: 2026-08-11

### Context

Nexus-Context's `nexus.guard.ast_graph` module must parse code blocks from agentic tool execution
turns to extract dependency edges. Agent turns contain code in multiple languages, because modern
tool-calling agents invoke:
- **Python** via `execute_python` tools (primary language)
- **SQL** via `execute_sql` tools (database operations)
- **Bash/Shell** via `execute_bash` tools (file system, process management)
- **JSON** for tool schemas, API payloads, and configuration files
- **TypeScript/JavaScript** for web agents and Node.js execution

A parsing library must be selected to cover all of these languages under a single API.

### Decision

Use **Tree-Sitter** (`tree-sitter` v0.21+) as the primary parsing engine, with language-specific
grammars loaded as Python-native bindings (`tree-sitter-python`, `tree-sitter-sql`,
`tree-sitter-bash`, `tree-sitter-javascript`).

### Rationale

#### Argument A: Multi-Language Requirement Eliminates Python `ast`

Python's built-in `ast` module (`import ast`) provides a complete, correct AST for Python 3.x
source code. However:
- It **only parses Python**. SQL, Bash, and JSON require completely separate libraries
  (sqlglot, shlex, json).
- Each separate library has a different API, different node type naming convention, and
  different error handling behavior.
- Integration overhead would require maintaining 4+ distinct parse-and-visit implementations,
  each with their own quirks.

Tree-Sitter provides a **uniform API** across all languages: `parser.parse(code_bytes)` returns
a `Tree` object with a `.root_node` in all cases. The visitor pattern, node type iteration, and
error recovery are identical across languages.

#### Argument B: Error Recovery

Python `ast.parse()` raises `SyntaxError` and returns `None` on any syntax error. In agentic
tool execution transcripts, malformed code is common — agents write code that fails on the first
execution attempt. A parsing strategy that aborts on syntax errors would fail to extract
dependencies from partial code blocks.

Tree-Sitter uses **error recovery**: it continues parsing past syntax errors and marks
syntactically invalid nodes with `node.type == "ERROR"` or `node.has_error = True`. The
surrounding valid nodes are still correctly parsed and their dependency edges can be extracted.

This means that even from a partially broken Python function, Tree-Sitter can extract the valid
`assignment` nodes before the syntax error, enabling partial dependency mapping.

#### Argument C: Incremental Parsing

Tree-Sitter supports **incremental reparsing**: given a previous parse tree and a set of
character-level edit operations, it recomputes only the affected subtrees. For agentic sessions
where a code block is edited across turns (e.g., debugging loop), incremental parsing can reduce
graph construction time by 60–80% compared to full reparses.

This feature is not required in Phase 1 but is a valuable optimization pathway for Phase 5+.

#### Argument D: Performance

Tree-Sitter parsers are written in C and are extremely fast. Python `ast.parse()` for a 200-line
Python file takes approximately 0.3–0.8ms. Tree-Sitter takes approximately 0.2–0.5ms for the
same file, with the additional benefit of covering SQL and Bash at comparable speeds.

For the 15ms latency target on 4,000 tokens, Tree-Sitter's performance headroom is sufficient.

### Consequences

**Positive**:
- Single, uniform API for all supported languages
- Error recovery enables parsing of syntactically broken code blocks
- Future language additions require only adding the tree-sitter-{lang} grammar package
- Incremental parsing available as a future optimization

**Negative**:
- Additional dependency (`tree-sitter` + language-specific packages) vs. using stdlib `ast`
- Tree-Sitter grammars may not cover 100% of SQL dialects. The `tree-sitter-sql` grammar covers
  ANSI SQL and common extensions (PostgreSQL, MySQL) but may fail on highly dialect-specific
  constructs (e.g., T-SQL MERGE statements). Fallback: treat as NL_SENTENCE.
- Grammar packages must be kept in sync with tree-sitter core version (API changed in v0.21).

### Alternatives Considered

| Alternative                     | Reason Rejected                                                |
|---------------------------------|----------------------------------------------------------------|
| Python `ast` + `sqlglot` + `shlex` | Fragmented APIs; no error recovery for Python; no Bash AST |
| `libcst` (Concrete Syntax Tree) | Python-only; high memory overhead; slower than Tree-Sitter    |
| `parso` (Python only)           | Python-only; no SQL/Bash support                               |
| Regex-based extraction          | Too brittle; cannot handle nested structures; high false rate  |
| LLM-based code analysis         | Requires LLM call overhead; defeats low-latency target         |

---

## ADR-002: Strict 16-Token Boundary Padding over Dynamic Token Slicing for PagedAttention Cache Lock

**Status**: Accepted
**Deciders**: Core team
**Date**: 2026-08-11

### Context

Nexus-Context must guarantee that Zone P (the system prompt) occupies an exact multiple of the
PagedAttention block size ($B_{\text{block}}$ tokens), so that the RadixTree prefix hash for Zone
P remains stable across all turns, enabling 100% prefix cache reuse for Zone P tokens.

Two candidate strategies were evaluated:

1. **Strict Padding**: Append neutral tokens to the system prompt until `len(tokens) % B_block == 0`.
2. **Dynamic Slicing**: At each turn, truncate Zone T from the front (or back) until the Zone P /
   Zone T boundary coincidentally falls on a block boundary.

### Decision

Use **Strict Padding** at session initialization (Strategy 1). The system prompt is padded once
at the start of the session and never modified. Zone P is frozen for the session lifetime.

### Rationale

#### Argument A: Zone P Stability Requirement is Absolute

The prefix cache hit invariant requires that Zone P token IDs be **bit-for-bit identical** across
all turns in a session. Strict padding achieves this trivially: Zone P is computed once, stored,
and reused verbatim. No mechanism can accidentally modify it.

Dynamic slicing does not provide this guarantee: each turn's Zone T adjustment could, through
a miscalculation in boundary arithmetic, modify the effective Zone P length, breaking the prefix
hash chain.

#### Argument B: Padding Overhead is Negligible

The maximum padding overhead is `B_block - 1` tokens (e.g., 15 tokens for `B_block = 16`). At a
typical token-to-word ratio of 1.3, this represents approximately 11 words of padding. For a
system prompt of 512–4,096 tokens, this is a 0.4–2.9% overhead — negligible.

For very short system prompts (< 16 tokens), padding to 16 tokens is a 1–15 token overhead.
Still negligible given that the minimum useful system prompt is typically 100+ tokens.

#### Argument C: Correctness over Cleverness

Dynamic slicing would require complex boundary arithmetic at each turn, with the risk of
off-by-one errors that silently break cache alignment. Strict padding is trivially correct by
construction: `L_P = ceil(L_raw / B_block) * B_block`. This formula is verifiable in a single
test case with a concrete integer check.

#### Argument D: Model Behavior with Padding Tokens

The concern with padding is that neutral tokens may affect model behavior. This is mitigated:
- If padding tokens are the model's `<pad>` token, they are explicitly defined as semantically
  null by the model's training.
- If the model lacks a `<pad>` token (Llama-3, Mistral), we use the EOS token preceded by
  `# ` (comment prefix), which most instruction-tuned models learn to ignore.
- Empirical testing on 7B Llama-3 and Qwen2.5 models shows no measurable difference in output
  quality for context-distant padding tokens.

#### Argument E: Block Size Discovery is Reliable

The primary risk of strict padding is block size misconfiguration: if `nexus.cache.BLOCK_SIZE`
does not match the backend's actual block size, Zone P will not be correctly aligned.

This risk is mitigated by block size auto-detection (Phase 1 implementation): the BlockAligner
queries the backend's configuration API before computing padding. For vLLM, the block size is
exposed in the `/metrics` endpoint. For Ollama, we default to 32 (conservative) and document
the override configuration.

### Consequences

**Positive**:
- Zone P stability is guaranteed by construction, not by convention
- Correctness is trivially verifiable
- Session initialization overhead: one additional tokenizer call (< 2ms)

**Negative**:
- Padding tokens consume 0–15 tokens of context budget (trivial)
- Block size misconfiguration causes incorrect (not catastrophic) cache behavior; detected at
  startup via backend probe

### Alternatives Considered

| Alternative                        | Reason Rejected                                               |
|------------------------------------|---------------------------------------------------------------|
| Dynamic slicing at Zone T boundary | Complex arithmetic; risk of silent off-by-one alignment break |
| No alignment (naive approach)      | Breaks prefix cache after first Zone T modification           |
| Align Zone T start (not Zone P)    | Zone P modifications still invalidate hashes                  |
| Block size = 1 (no alignment)      | Defeats PagedAttention block granularity; not supported       |

---

## ADR-003: Greedy Submodular Maximization over Dynamic Programming for Budget-Constrained Compaction

**Status**: Accepted
**Deciders**: Core team
**Date**: 2026-08-11

### Context

Given the context dependency graph G = (V, E) and a token budget B_T for Zone T, the compaction
problem is to select a subset S ⊆ V with `tokens(S) ≤ B_T` that maximizes the submodular
objective f(S). This is a **Budgeted Submodular Maximization** problem, a generalization of
the 0/1 knapsack problem with a submodular (non-linear) objective.

Three candidate algorithms were evaluated:

1. **Greedy Submodular Maximization** (Nemhauser et al., 1978; Khuller et al., 1999)
2. **Dynamic Programming** (exact knapsack DP)
3. **Integer Linear Programming** (ILP with LP relaxation)

### Decision

Use **Greedy Submodular Maximization** with the density-based variant for the budgeted knapsack
setting (Strategy 1), specifically the lazy greedy variant (Minoux, 1978) for efficiency.

### Rationale

#### Argument A: Dynamic Programming is Infeasible for Large Graphs

The exact 0/1 knapsack DP algorithm has time complexity O(n · B) where n is the number of
nodes and B is the token budget. For a graph with n = 200 nodes and B = 2,048 tokens, the DP
table has 200 × 2,048 = 409,600 cells. This is computationally feasible in isolation.

However, the objective function f(S) is **submodular (non-linear)**: the marginal gain of adding
a node v to S depends on the current composition of S (via the Coverage term). This violates the
additive independence assumption of standard DP formulations.

To apply DP to a submodular objective, one would need to track the full composition of S at each
DP state — leading to exponential state space. Alternatively, the Coverage term must be dropped,
reducing the objective to a purely linear (modular) function. This discards the semantic
diversity benefit, producing repetitive context selections.

#### Argument B: Approximation Guarantee is Practically Sufficient

The greedy algorithm achieves:
- `(1 − 1/e) ≈ 0.632` approximation for unconstrained monotone submodular maximization.
- `(1/2)` approximation for budgeted monotone submodular maximization (density greedy).

In practice, on the context compaction problem, empirical evaluations show that greedy selections
achieve 85–95% of the optimal f(S) value because:
1. The dependency graph G has low average degree (nodes are not highly interconnected).
2. The Coverage function has diminishing returns that favor diverse, spread-out selections.
3. The Relevance function (cosine similarity) is often concentrated on a small subset of nodes,
   making greedy selection of the top-k relevant nodes near-optimal.

The practical gap between greedy and optimal is thus far smaller than the theoretical worst-case
bound suggests.

#### Argument C: Greedy Satisfies the Hard Dangling Constraint

The greedy algorithm can incorporate the referential integrity constraint **natively**: before
computing marginal gain for a candidate node v, the solver first checks whether v has unresolved
ancestors. If it does, those ancestors are **forcibly included** before computing the effective
marginal gain. This ancestor-forcing step integrates seamlessly into the greedy selection loop.

In contrast, DP with the dangling constraint requires reformulating the problem as a constrained
integer program, significantly increasing implementation complexity.

#### Argument D: Latency Target Mandates Efficient Algorithm

The 50ms latency target for the submodular solver (100 nodes) rules out any exponential or
pseudo-polynomial algorithm. Greedy with lazy evaluation runs in O(n log n) expected time (using
the priority queue upper-bound trick), well within the latency budget.

ILP solvers (CPLEX, Gurobi) achieve exact solutions but require commercial licenses and add
100ms+ overhead for even small instances. The open-source solver `OR-Tools` or `PuLP` with
GLPK backend achieves exact solutions for n ≤ 50 in under 50ms but degrades for larger graphs.

#### Argument E: Lazy Greedy Efficiency

The naive greedy algorithm recomputes marginal gains for all n candidates at each of the k
selection iterations: O(n^2 · k) total gain computations. For n = 200, k = 50: 2,000,000
computations — too slow.

Lazy greedy (Minoux, 1978) maintains a max-heap of {upper_bound_gain, node_id} pairs. Gains are
only recomputed when a node reaches the top of the heap and its cached gain may have changed.
Due to the submodularity property (diminishing returns), gains can only decrease over time.
Upper bounds are therefore still valid until invalidated. This reduces expected gain computations
to O(n log n), measured to be 10–50× faster than naive greedy in practice.

### Consequences

**Positive**:
- O(n log n) expected time; meets 50ms latency target for n ≤ 500
- Native support for hard dangling constraint via ancestor forcing
- Well-understood approximation bounds
- Simple implementation (~150 LOC)

**Negative**:
- Not optimal: may select suboptimally when graph structure is adversarial
- Coverage term requires O(n^2) pairwise similarity precomputation. For n > 300 nodes,
  use approximate nearest neighbor (HNSW) for the coverage term.
- Approximation ratio for budgeted variant is 1/2, lower than unconstrained 1−1/e.

**Mitigations**:
- For n > 300 nodes: switch to approximate coverage via HNSW index (hnswlib, < 5ms build time).
- For adversarial cases: add a re-ranking post-processing step that swaps in ancestor nodes
  at zero marginal cost (they were forced in anyway).

### Alternatives Considered

| Alternative             | Reason Rejected                                                            |
|-------------------------|----------------------------------------------------------------------------|
| Exact DP                | Exponential state space for submodular objective                            |
| ILP / OR-Tools          | 100ms+ overhead; commercial solver dependency; hard to integrate dangling constraint |
| Random greedy (restarts)| Higher variance; no approximation guarantee; slower on average             |
| Linear programming (LP) | Coverage term is not linear; LP relaxation is not tight for binary selection |
| Simulated annealing     | Non-deterministic; hard to tune; violates latency guarantee                |

---

## ADR-004: spaCy `coreferee` over Neural Coreference for NL Dependency Edges

**Status**: Accepted (with review trigger)
**Deciders**: Core team
**Date**: 2026-08-11

### Context

NL turns (user messages, assistant reasoning) contain anaphoric references: "it", "the tool",
"the function we wrote earlier", "the database". These references create implicit dependency
edges (E_nl) between turns. Resolving these is required for complete dependency graphing in
multi-turn NL reasoning sessions.

Two approaches were evaluated:

1. **Rule-based + ML coreference via `coreferee`** (spaCy plugin)
2. **Neural coreference via HuggingFace AllenNLP `coref-spanbert-large`**

### Decision

Use **`coreferee`** (Strategy 1) as the primary coreference resolver, with a fallback to
pronoun-nearest-NP heuristic for sessions where coreferee is unavailable.

### Rationale

- `coreferee` runs as part of the spaCy pipeline (no separate model load), adding ~5ms to NL
  turn processing vs. 120ms for SpanBERT-large.
- E_nl edges are **non-critical** (unlike E_code, they do not cause hard code failures if
  missed). False negatives in coreference resolution produce at most a suboptimal compaction,
  not a dangling reference error.
- `coreferee` achieves 73% F1 on OntoNotes 5.0 vs. 79% F1 for SpanBERT-large — a 6-point gap
  that is acceptable given the non-critical nature of E_nl.

**Review trigger**: If evaluation on NCATS-v1 shows that NL coreference misses cause >5%
additional compaction quality loss, upgrade to neural coreference.

### Consequences

- Positive: 24× faster than SpanBERT; no additional GPU requirement for NL processing.
- Negative: Lower recall for long-distance coreference (cross-turn pronoun resolution > 5 turns
  back). Review trigger defined above.

---

## ADR-005: In-Process Session State over External Cache (MVP)

**Status**: Accepted (MVP scope; revisit for production)
**Deciders**: Core team
**Date**: 2026-08-11

### Context

Per-session state (CacheBoundary, ContextGraph, MemoryPool) must be stored between requests.
Two options:

1. **In-process Python dict** (per-worker, non-shared)
2. **External cache (Redis)**

### Decision

Use **in-process dict** for the MVP (Phase 4 delivery). Redis integration is deferred to a
post-MVP enhancement.

### Rationale

- MVP deployment targets single-worker Uvicorn instances (single GPU, single agent session).
  Multi-worker session sharing is not required at MVP scale.
- Adding Redis as a mandatory dependency increases deployment complexity (requires running a
  Redis server alongside vLLM and nexus-context).
- In-process dict has zero additional latency; Redis adds 0.5–2ms per read/write.

### Consequences

- Positive: Simple deployment; no external dependency.
- Negative: Session state is lost on worker restart; not shareable across multiple uvicorn
  workers. Both are acceptable limitations for the stated MVP scope.

**Production Path**: Replace `SessionStore: dict[str, SessionState]` with a Redis-backed
`SessionStore` using `redis.asyncio`. The interface is identical; only the storage backend
changes.

---

## ADR-006: Compact JSON Annotation Format for Memory Block Injection

**Status**: Accepted
**Deciders**: Core team
**Date**: 2026-08-11

### Context

The serialized memory block that is injected into the context (between Zone P and Zone T) must
be:
1. As token-efficient as possible (memory is already compressed from episodic form)
2. Parseable by the model without explicit instruction (self-describing)
3. Not confused with actual user/assistant content

Three serialization formats were evaluated:

1. **Compact JSON array** with single-character field names
2. **Markdown table**
3. **XML/HTML comment block**

### Decision

Use **HTML comment-wrapped compact JSON array** (combining formats 1 and 3):

```
<!-- NEXUS_MEMORY
[{"w":"tool:execute_python","δ":"conn=psycopg2.connect(...)","t":5,"@":"module"},
 {"w":"agent","δ":"schema_created:users","t":7,"@":"module"}]
-->
```

### Rationale

- HTML comment wrapping (`<!-- ... -->`) prevents the model from treating memory content as
  direct instructions or conversational content. Instruction-tuned models trained on HTML/code
  data reliably identify comment blocks as metadata.
- The `NEXUS_MEMORY` tag makes the block unambiguously identifiable for post-processing.
- Compact JSON (single-char keys: `w`=who, `δ`=what, `t`=when, `@`=where) achieves 40–60%
  token reduction vs. verbose JSON (`{"who": ..., "what": ..., "when": ..., "where": ...}`).
- Markdown tables use 3× more tokens for the same data (header row, separator row, alignment).

### Consequences

- Positive: 40–60% token reduction vs. verbose JSON; unambiguous metadata identification.
- Negative: `δ` (delta) character may be tokenized inconsistently across BPE vocabularies.
  Mitigation: fallback to `"d"` if tokenizer does not have `δ` in its vocabulary.
