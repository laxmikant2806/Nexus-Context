# Nexus-Context: System Architecture

> **Version**: 0.1.0-draft
> **Status**: Design Phase
> **Last Updated**: 2026-08-11

---

## Table of Contents

1. [System Overview and Design Goals](#1-system-overview-and-design-goals)
2. [ASCII Dataflow Diagram](#2-ascii-dataflow-diagram)
3. [Component Descriptions](#3-component-descriptions)
4. [Pydantic Schema Definitions](#4-pydantic-schema-definitions)
5. [Mathematical Formulations](#5-mathematical-formulations)
6. [Inter-Module Communication Contracts](#6-inter-module-communication-contracts)
7. [Failure Modes and Fallback Paths](#7-failure-modes-and-fallback-paths)

---

## 1. System Overview and Design Goals

Nexus-Context operates as a **transparent middleware layer** that intercepts OpenAI-compatible
`/v1/chat/completions` requests destined for a local SLM server (vLLM, SGLang, Ollama). The
payload is processed through three sequential stages — cache alignment, dependency graphing, and
memory governance — and the transformed payload is forwarded to the backend server.

### Design Goals

| Goal                              | Metric                                                      |
|-----------------------------------|-------------------------------------------------------------|
| Zero referential dangling         | 0% NameError/KeyError rate in compressed output             |
| Prefix KV cache preservation      | >85% cache hit rate across multi-turn tool sessions         |
| Low latency overhead              | Graph construction <15ms per 4,000 input tokens             |
| Memory compression ratio          | >4:1 episodic-to-semantic compression for code turns        |
| Transparency                      | No model fine-tuning; pure prompt engineering at the layer  |
| Backend agnosticism               | Compatible with vLLM, SGLang, Ollama without code changes   |

---

## 2. ASCII Dataflow Diagram

```
                        NEXUS-CONTEXT MIDDLEWARE PIPELINE
                        ==================================

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           CLIENT APPLICATION                                │
 │         (Agent loop / IDE plugin / CLI tool)                                │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │  POST /v1/chat/completions
                                    │  { "model": "...", "messages": [...] }
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     FASTAPI MIDDLEWARE (nexus.cache.middleware)              │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  Request Interception                                                 │   │
 │  │  • Parse JSON payload                                                 │   │
 │  │  • Extract messages[], model, session_id                              │   │
 │  │  • Route to NexusContextProcessor                                     │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                   STAGE 1: BLOCK ALIGNMENT (nexus.cache.block_align)        │
 │                                                                             │
 │  Input:  Raw messages[] from client                                         │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  1. Tokenize system prompt with HuggingFace tokenizer                 │   │
 │  │  2. Compute L_raw = len(system_prompt_tokens)                         │   │
 │  │  3. L_P = ceil(L_raw / B_block) × B_block                            │   │
 │  │  4. Append (L_P − L_raw) neutral padding tokens                      │   │
 │  │  5. Freeze Zone P: hash and store for session lifetime                │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │                                                                             │
 │  Output: Padded Zone P (block-aligned) + unmodified Zones T and R           │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                STAGE 2: DIFFERENTIAL SEGMENTATION (nexus.cache.differential)│
 │                                                                             │
 │  Input:  Full message sequence, current token count, budget B               │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  Zone P: System prompt (block-aligned, immutable)                     │   │
 │  │  Zone R: Last R_max tokens of messages (verbatim, append-only)       │   │
 │  │  Zone T: Remaining middle turns, token budget = B − |P| − |R|        │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │                                                                             │
 │  Output: ZoneBoundary struct with token offsets for P, T, R                │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │  (if Zone T exceeds budget → trigger pruning)
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │              STAGE 3: TREE-SITTER AST GRAPH BUILD (nexus.guard.ast_graph)  │
 │                                                                             │
 │  Input:  Turn messages in Zone T (code + natural language)                  │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  For code turns (Python, SQL, Bash, JSON):                           │   │
 │  │    • Parse with Tree-Sitter grammar                                   │   │
 │  │    • Extract: assignments, function_defs, imports, calls              │   │
 │  │    • Build E_code edges: (def_node → use_node)                       │   │
 │  │                                                                       │   │
 │  │  For NL turns (system, user, assistant):                              │   │
 │  │    • Parse with spaCy en_core_web_trf                                 │   │
 │  │    • Run coreference resolution (coreferee / neuralcoref)             │   │
 │  │    • Build E_nl edges: (antecedent_span → anaphor_span)              │   │
 │  │                                                                       │   │
 │  │  Cross-turn tool dependency:                                          │   │
 │  │    • Match tool return values to downstream call arguments            │   │
 │  │    • Build E_tool edges: (return_turn → argument_turn)               │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │                                                                             │
 │  Output: ContextGraph G = (V, E = E_code ∪ E_nl ∪ E_tool)                 │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │              STAGE 4: SUBMODULAR PRUNING (nexus.guard.submodular)           │
 │                                                                             │
 │  Input:  ContextGraph G, token budget B_T for Zone T, query embedding q     │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  1. Compute transitive closure E* of G                               │   │
 │  │  2. Initialize S = {} (selected set), remaining = B_T                │   │
 │  │  3. Greedy loop:                                                      │   │
 │  │     a. For each candidate v ∉ S:                                     │   │
 │  │        - Check dangling: if v has ∃ ancestor u ∉ S in E*, force u   │   │
 │  │        - Compute marginal gain Δf(v | S)                             │   │
 │  │     b. Select v* = argmax Δf(v | S) / tokens(v)                     │   │
 │  │        (density-based greedy for knapsack budget)                    │   │
 │  │     c. Add v* and all forced ancestors to S                          │   │
 │  │     d. Deduct tokens(S_added) from remaining                         │   │
 │  │     e. Repeat until remaining < min_node_size                        │   │
 │  │  4. Reconstruct Zone T text from S in original turn order            │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │                                                                             │
 │  Output: CompactionResult with selected nodes, Zone T text, stats           │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │              STAGE 5: WWW MEMORY INJECTION (nexus.memory)                   │
 │                                                                             │
 │  Input:  Pruned Zone T, memory pool P = {M_1, ..., M_k}                    │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │  1. For each turn NOT selected by submodular solver:                  │   │
 │  │     - Run www_parser to extract ⟨Who, What, When, Where⟩ tuple      │   │
 │  │     - Add M_i to memory pool P                                        │   │
 │  │  2. Score all M_i: W(t_i, s_i) = exp(−λ(T−t_i))·(1+η·AST_Depth)   │   │
 │  │  3. Pin memories whose What field is referenced by retained nodes     │   │
 │  │  4. Greedily select top-W memories within B_mem token budget         │   │
 │  │  5. Serialize selected memories as compact JSON annotations           │   │
 │  │  6. Prepend memory block to Zone T reconstructed text                 │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │                                                                             │
 │  Output: Final reconstructed payload: [Zone P][Memory Block][Zone T][Zone R]│
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    PAYLOAD RECONSTRUCTION (nexus.cache.middleware)           │
 │  • Reassemble messages[] list from zone texts                               │
 │  • Verify total token count ≤ B                                             │
 │  • Attach nexus-context-stats header for observability                      │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    │  POST /v1/chat/completions (forwarded)
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                   BACKEND SLM SERVER                                         │
 │   vLLM  /  SGLang  /  Ollama                                                │
 │   • Serves /v1/chat/completions                                             │
 │   • PagedAttention prefix caching active                                    │
 │   • Prefix cache hit on Zone P guaranteed (block-aligned, unchanged)         │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 nexus.cache.middleware (FastAPI Proxy)

The middleware is a FastAPI application that proxies all `/v1/chat/completions` and
`/v1/completions` requests. It operates as a **transparent intermediary**: the client sends
standard OpenAI-format requests; the middleware processes and forwards them to the backend
without the client needing modification.

**Key responsibilities**:
- Maintain per-session state (Zone P hash, memory pool, turn counter)
- Route to NexusContextProcessor for payload transformation
- Forward transformed payloads to configured backend
- Stream responses back transparently (SSE pass-through for streaming=True)
- Attach `X-Nexus-Context-Stats` headers for observability

**Session identification**: Via `X-Session-ID` header or auto-generated UUID from client IP +
model + user-agent fingerprint.

### 3.2 nexus.cache.block_align (Block Alignment Engine)

Responsible for ensuring Zone P satisfies the block alignment invariant.

**Tokenizer support**:
- HuggingFace `AutoTokenizer`: for vLLM and SGLang deployments
- `tiktoken` (`cl100k_base`, `o200k_base`): for OpenAI-compatible APIs
- `llama-cpp-python` binding: for Ollama deployments

**Padding strategy selection**:
1. Probe model tokenizer for `pad_token_id`
2. If available: use pad token (safest, no semantic content)
3. If unavailable: use `eos_token` (works for most Llama/Mistral variants)
4. Fallback: use single space character token

### 3.3 nexus.cache.differential (Zone Segmentation Engine)

Partitions the message sequence into Zones P, T, R with configurable token budgets.

**Graduation policy** (Zone R → Zone T):
- When a turn in Zone R becomes older than `tail_retention_turns` (default: 3), it graduates
  to Zone T candidate pool.
- Graduation triggers a submodular re-evaluation of Zone T with the new candidate.

### 3.4 nexus.guard.ast_graph (Dependency Graph Builder)

Builds the Context Dependency Graph G from turn message content.

**Language support** (Tree-Sitter grammars):
- Python: `tree-sitter-python`
- SQL: `tree-sitter-sql`
- Bash/Shell: `tree-sitter-bash`
- JSON: `tree-sitter-json`
- TypeScript/JavaScript: `tree-sitter-javascript`, `tree-sitter-typescript`

**NL coreference** (spaCy pipeline):
- Model: `en_core_web_trf` (transformer-based, highest accuracy)
- Coreference plugin: `coreferee` (v1.4+) or `spacy-experimental` coref
- Fallback: rule-based pronoun-to-nearest-NP resolution for speed-critical deployments

### 3.5 nexus.guard.submodular (Greedy Optimization Solver)

Implements the density-based greedy algorithm for the budgeted submodular maximization problem
with hard referential integrity constraints.

**Algorithm variant**: Lazy greedy evaluation (Minoux, 1978) with priority queue optimization,
reducing worst-case O(n^2 k) greedy iterations to O(n log n) via upper-bound caching.

**Embedding model** (for Relevance computation):
- Default: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 80ms/4000 tokens on CPU)
- Optional: `nomic-ai/nomic-embed-text-v1.5` (higher accuracy, requires GPU)

### 3.6 nexus.memory.www_parser (WWW Tuple Extractor)

Parses each agent turn and extracts a structured WWW memory tuple.

**Extraction pipeline**:
1. Detect turn type: code (Python/SQL/Bash/JSON) or natural language
2. For code turns: Tree-Sitter parse → AST visitor → mutation extraction
3. For NL turns: spaCy NER + dependency parse → event extraction → tuple construction
4. Assign scope identifier from AST namespace stack

### 3.7 nexus.memory.decay (Weight Computation Engine)

Computes retention weights and manages the memory pool lifecycle.

**Memory pool persistence**: Serialized to disk as a JSONL file per session, enabling
cross-session memory continuity (optional feature, disabled by default).

---

## 4. Pydantic Schema Definitions

All inter-module data structures use Pydantic v2 models for validation, serialization, and
OpenAPI schema generation.

```python
from __future__ import annotations

import enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import math


# ─────────────────────────────────────────────────────────────────────────────
# nexus.guard.schemas
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, enum.Enum):
    """Semantic classification of a context graph node."""
    TURN             = "turn"           # A complete conversational turn
    CODE_BLOCK       = "code_block"     # A code execution block
    AST_ASSIGNMENT   = "ast_assign"     # Variable assignment AST node
    AST_FUNCDEF      = "ast_funcdef"    # Function definition AST node
    AST_CLASSDEF     = "ast_classdef"   # Class definition AST node
    AST_IMPORT       = "ast_import"     # Import statement AST node
    AST_CALL         = "ast_call"       # Function call AST node
    NL_SENTENCE      = "nl_sentence"    # A natural language sentence
    TOOL_RETURN      = "tool_return"    # Tool invocation return value
    SCHEMA_FIELD     = "schema_field"   # JSON/Pydantic schema field


class EdgeType(str, enum.Enum):
    """Semantic classification of a context dependency edge."""
    VAR_DEF_TO_REF   = "var_def_to_ref"    # Variable definition → reference
    FUNC_DEF_TO_CALL = "func_def_to_call"  # Function definition → call site
    CLASS_DEF_TO_USE = "class_def_to_use"  # Class definition → instantiation
    IMPORT_TO_USE    = "import_to_use"     # Import → usage in code
    COREF_ANTECEDENT = "coref_antecedent"  # NL coreference chain
    TOOL_RETURN_DEP  = "tool_return_dep"   # Tool return → downstream argument
    SCHEMA_FIELD_DEP = "schema_field_dep"  # Schema field → field usage


class ContextNode(BaseModel):
    """A node in the context dependency graph."""
    node_id: str = Field(
        description="Unique node identifier, format: {turn_index}:{node_type}:{name}"
    )
    node_type: NodeType
    turn_index: int = Field(ge=0, description="Originating turn index (0-based)")
    content: str = Field(description="Raw text content of this node")
    token_count: int = Field(ge=0, description="Number of tokens in this node's content")
    language: Optional[str] = Field(
        default=None,
        description="Programming language if code node (python, sql, bash, json)"
    )
    ast_depth: int = Field(
        default=0,
        ge=0,
        description="Nesting depth within AST scope hierarchy"
    )
    scope_path: str = Field(
        default="module",
        description="Dot-separated AST scope path, e.g. 'module.MyClass.my_method'"
    )
    is_pinned: bool = Field(
        default=False,
        description="If True, node cannot be pruned regardless of budget"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Language-specific metadata (e.g., variable name, function signature)"
    )


class DependencyEdge(BaseModel):
    """A directed dependency edge in the context graph."""
    edge_id: str = Field(description="Unique edge identifier: {source_id}->{target_id}")
    source_id: str = Field(description="node_id of the defining/antecedent node")
    target_id: str = Field(description="node_id of the referencing/dependent node")
    edge_type: EdgeType
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Edge weight for weighted dependency scoring"
    )
    is_transitive: bool = Field(
        default=False,
        description="True if this edge was derived via transitive closure computation"
    )


class ContextGraph(BaseModel):
    """The full context dependency graph for a session."""
    session_id: str
    nodes: dict[str, ContextNode] = Field(default_factory=dict)
    edges: list[DependencyEdge] = Field(default_factory=list)
    transitive_closure_computed: bool = Field(default=False)

    def get_ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor node_ids for a given node (direct + transitive)."""
        ancestors: set[str] = set()
        queue = [node_id]
        while queue:
            nid = queue.pop()
            for edge in self.edges:
                if edge.target_id == nid and edge.source_id not in ancestors:
                    ancestors.add(edge.source_id)
                    queue.append(edge.source_id)
        return ancestors


class CompactionResult(BaseModel):
    """Output of the submodular pruning stage for Zone T."""
    session_id: str
    turn_index: int = Field(description="Current turn at which compaction was performed")
    selected_node_ids: list[str] = Field(description="Ordered list of selected node IDs")
    pruned_node_ids: list[str] = Field(description="Node IDs excluded from selection")
    token_budget: int = Field(description="Token budget B_T for Zone T")
    tokens_used: int = Field(description="Actual tokens consumed by selected nodes")
    objective_value: float = Field(description="Final f(S) value from submodular objective")
    dangling_violations: int = Field(
        default=0,
        description="Number of dangling violations detected (should always be 0)"
    )
    forced_inclusions: list[str] = Field(
        default_factory=list,
        description="Node IDs forced into selection to satisfy referential integrity"
    )
    reconstructed_text: str = Field(description="Concatenated text of selected Zone T nodes")
    latency_ms: float = Field(description="Wall-clock time for compaction in milliseconds")


# ─────────────────────────────────────────────────────────────────────────────
# nexus.cache.schemas
# ─────────────────────────────────────────────────────────────────────────────

class ZoneType(str, enum.Enum):
    LOCKED_HEAD    = "P"   # Zone P: system prompt, tool schemas
    COMPACTED_TRUNK = "T"  # Zone T: pruned historical turns
    RAW_TAIL       = "R"   # Zone R: verbatim recent turns


class CacheBoundary(BaseModel):
    """Defines the three-zone token boundaries for a session context."""
    session_id: str
    block_size: int = Field(
        default=16,
        description="PagedAttention block size in tokens (16 or 32)"
    )

    # Zone P
    zone_p_token_start: int = Field(default=0)
    zone_p_token_end: int = Field(description="Must satisfy zone_p_token_end % block_size == 0")
    zone_p_hash: str = Field(description="SHA256 of Zone P token IDs, for integrity verification")
    zone_p_padding_tokens: int = Field(
        default=0,
        description="Number of neutral padding tokens appended to achieve block alignment"
    )

    # Zone T
    zone_t_token_start: int
    zone_t_token_end: int
    zone_t_budget: int = Field(description="Maximum tokens allocated to Zone T")

    # Zone R
    zone_r_token_start: int
    zone_r_token_end: int
    zone_r_budget: int = Field(
        default=1024,
        description="Maximum tokens allocated to Zone R (raw tail)"
    )

    # Total
    total_budget: int = Field(description="Total context window budget B")

    @field_validator("zone_p_token_end")
    @classmethod
    def validate_block_alignment(cls, v: int, info) -> int:
        block_size = info.data.get("block_size", 16)
        if v % block_size != 0:
            raise ValueError(
                f"zone_p_token_end={v} is not divisible by block_size={block_size}. "
                f"Zone P must be block-aligned for prefix cache preservation."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# nexus.memory.schemas
# ─────────────────────────────────────────────────────────────────────────────

class WhoActor(str, enum.Enum):
    USER    = "user"
    AGENT   = "agent"
    TOOL    = "tool"
    SYSTEM  = "system"


class MutationType(str, enum.Enum):
    VAR_ASSIGN       = "var_assign"
    FUNC_DEF         = "func_def"
    CLASS_DEF        = "class_def"
    IMPORT           = "import"
    FS_MUTATION      = "fs_mutation"
    SCHEMA_CHANGE    = "schema_change"
    NETWORK_STATE    = "network_state"
    NL_STATEMENT     = "nl_statement"


class WhatDelta(BaseModel):
    """The state mutation payload (What component of WWW tuple)."""
    mutation_type: MutationType
    target_name: str = Field(description="Primary entity affected (variable name, table name, etc.)")
    old_value_repr: Optional[str] = Field(
        default=None,
        description="String representation of prior value (if known)"
    )
    new_value_repr: str = Field(description="String representation of new value or state")
    side_effects: list[str] = Field(
        default_factory=list,
        description="List of secondary entities affected by this mutation"
    )
    is_destructive: bool = Field(
        default=False,
        description="True if the mutation cannot be reversed (DROP TABLE, file DELETE, etc.)"
    )


class MemoryTuple(BaseModel):
    """A complete WWW Memory Tuple representing one agent state mutation."""
    memory_id: str = Field(description="Unique memory ID: {session_id}:{turn_index}:{seq}")
    session_id: str

    # WWW Components
    who: WhoActor
    who_detail: str = Field(
        description="Detailed actor: 'tool:execute_python@scope_3', 'user@turn_5', etc."
    )
    what: WhatDelta
    when: int = Field(ge=0, description="Turn index at which the mutation occurred")
    where: str = Field(
        description="AST scope path: 'module.MyClass.init' or 'session.subtask_2.step_1'"
    )

    # Computed fields
    token_count: int = Field(description="Token count of the serialized memory tuple")
    retention_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Current W(t, s) value; recomputed at each turn"
    )
    is_pinned: bool = Field(
        default=False,
        description="If True, memory bypasses temporal decay and is always retained"
    )
    ast_depth_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="AST_Depth(s) = (D_max - d(s)) / D_max, pre-computed at extraction"
    )

    def compute_weight(self, current_turn: int, lambda_: float = 0.05, eta: float = 0.5) -> float:
        """
        Compute W(t_i, s_i) = exp(-lambda * (T - t_i)) * (1 + eta * AST_Depth(s_i))

        Args:
            current_turn: Current turn index T.
            lambda_: Temporal decay constant (default 0.05).
            eta: AST depth amplification factor (default 0.5).

        Returns:
            Retention weight in range [0, 1 + eta].
        """
        if self.is_pinned:
            return math.inf
        age = current_turn - self.when
        temporal_factor = math.exp(-lambda_ * age)
        depth_factor = 1.0 + eta * self.ast_depth_score
        return temporal_factor * depth_factor
```

---

## 5. Mathematical Formulations

### 5.1 Block Alignment (Zone P)

Given raw system prompt token length L_raw and PagedAttention block size B_block:

```
L_P = ceil(L_raw / B_block) × B_block         (target padded length)
n_pad = L_P − L_raw                             (padding tokens needed)
```

The prefix hash chain for Zone P:

```
h_0 = SHA256(T[0 : B_block])
h_i = SHA256(T[i·B_block : (i+1)·B_block] || h_{i-1})   for i in [1, L_P / B_block)
```

Zone P prefix is stable if and only if all token IDs in T[0 : L_P] are unchanged across turns.
Since Zone P includes neutral padding tokens that never change, this invariant holds for the
session lifetime.

### 5.2 Submodular Objective Function

Full expansion of f(S):

```
f(S) = α · Σ_{v ∈ S} cos(emb(v), q)
     + β · Σ_{v ∈ V} max_{u ∈ S} sim(u, v)
     − γ · Σ_{(u,v) ∈ E*} I(v ∈ S ∧ u ∉ S)

where:
  α, β ∈ [0, 1]  (mixing weights, α + β = 1)
  γ = 10^6 × (α · max_{v} cos(emb(v), q) + β · |V|)   (effectively infinite)
  sim(u, v) = cos(emb(u), emb(v))                       (pairwise embedding similarity)
  E*          = transitive closure of E                  (all ancestor-descendant pairs)
```

**Marginal gain of node v given current selection S**:

```
Δf(v | S) = α · cos(emb(v), q)
           + β · Σ_{w ∈ V} max(0, sim(v, w) − max_{u ∈ S} sim(u, w))
           − γ · DanglingCheck(v, S)

DanglingCheck(v, S) = |{ u ∈ ancestors_E*(v) : u ∉ S }|  (count of unresolved ancestors)
```

When selecting node v, the greedy solver simultaneously forces inclusion of all u ∈
ancestors_E*(v) \ S, adjusting the token deduction accordingly.

### 5.3 Density-Based Greedy for Knapsack Budgets

For the budgeted knapsack variant, the greedy algorithm uses **density** rather than raw marginal
gain to maintain the (1/2)-approximation guarantee for monotone submodular functions with
heterogeneous token costs (Khuller et al., 1999):

```
v* = argmax_{v ∉ S, tokens(v) ≤ remaining} Δf(v | S) / tokens(v)
```

### 5.4 Memory Weight Decay

```
W(t_i, s_i) = exp(−λ(T − t_i)) · (1 + η · AST_Depth(s_i))

AST_Depth(s_i) = (D_max − d(s_i)) / D_max

Ranges:
  d(s_i) = 0  (root scope)  → AST_Depth = 1.0  → W_max = (1 + η) · exp(−λ(T − t_i))
  d(s_i) = D_max            → AST_Depth = 0.0  → W_min = 1.0    · exp(−λ(T − t_i))
```

**Half-life**: The turn age at which a root-scope memory's weight falls to 50% of its initial
value (for newly created memories with temporal factor ≈ 1):

```
T_{1/2} = ln(2) / λ = ln(2) / 0.05 ≈ 13.9 turns
```

### 5.5 Graph Edge Weight Formula

Edge weights are used to break ties in the greedy selection when multiple nodes have equal
marginal gain:

```
w(u, v) = EdgeType_weight(edge_type) × (1 / (1 + distance(u, v)))

distance(u, v) = |turn_index(v) − turn_index(u)|   (turn distance between definition and use)

EdgeType_weight:
  VAR_DEF_TO_REF   → 1.0   (critical: variable use without definition = NameError)
  FUNC_DEF_TO_CALL → 0.9   (critical: function call without definition = NameError)
  CLASS_DEF_TO_USE → 0.9   (critical)
  IMPORT_TO_USE    → 0.8   (high: missing import = ModuleNotFoundError)
  TOOL_RETURN_DEP  → 0.8   (high: tool result used as argument)
  COREF_ANTECEDENT → 0.5   (medium: NL confusion, not code crash)
  SCHEMA_FIELD_DEP → 0.7   (high: ValidationError on schema field access)
```

---

## 6. Inter-Module Communication Contracts

```
nexus.cache.middleware
    │
    ├──► nexus.cache.block_align.BlockAligner.align(system_prompt) → (str, CacheBoundary)
    │
    ├──► nexus.cache.differential.ZoneSegmenter.segment(messages, boundary) → ZoneBundle
    │        (ZoneBundle: {zone_p: str, zone_t_candidates: list[Turn], zone_r: list[Turn]})
    │
    ├──► nexus.guard.ast_graph.ContextGraphBuilder.build(zone_t_candidates) → ContextGraph
    │
    ├──► nexus.guard.submodular.SubmodularSolver.solve(graph, budget, query) → CompactionResult
    │
    ├──► nexus.memory.www_parser.WWWParser.extract(pruned_turns) → list[MemoryTuple]
    │
    ├──► nexus.memory.decay.MemoryPool.update(new_tuples, current_turn) → list[MemoryTuple]
    │        (returns selected memories within B_mem budget)
    │
    └──► PayloadAssembler.assemble(zone_p, memories, compaction_result, zone_r) → messages[]
```

All inter-module calls are **synchronous** within a single request processing pipeline. Async
execution (via `asyncio.gather`) is used only for independent embedding computations.

---

## 7. Failure Modes and Fallback Paths

| Component            | Failure Mode                         | Fallback Behavior                                |
|----------------------|--------------------------------------|--------------------------------------------------|
| block_align          | Tokenizer load failure               | Default to character-based estimation (4 chars/token) |
| block_align          | Tokenizer returns 0 tokens           | Raise ConfigurationError, abort session          |
| differential         | Zone T candidates = 0                | Skip submodular stage, use raw Zone R only       |
| ast_graph            | Tree-Sitter parse failure (syntax error) | Mark turn as NL_SENTENCE, skip code edge extraction |
| ast_graph            | spaCy model not found                | Skip NL coreference; E_nl = {} (conservative)   |
| submodular           | Embedding model unavailable          | Fall back to TF-IDF cosine similarity            |
| submodular           | Budget B_T = 0                       | Return empty CompactionResult                     |
| www_parser           | AST extraction returns empty What    | Create generic NL_STATEMENT tuple with raw text summary |
| memory.decay         | memory pool file corrupted           | Reset pool to empty; log warning                 |
| middleware           | Backend server unreachable           | Return 503 with nexus-context-error header       |
| middleware           | Total tokens exceed B after assembly | Truncate Zone T from the front (oldest first)    |

All fallback paths are logged at WARNING level with structured JSON fields for observability.
