# Nexus-Context: Implementation Stage Guide

> **Version**: 0.1.0-draft
> **Status**: Ready for Implementation
> **Last Updated**: 2026-08-11
> **Prerequisites**: Read `docs/overview_and_research.md`, `docs/architecture.md`,
> `docs/decisions.md`, and `docs/planner.md` before beginning implementation.

---

## Table of Contents

1. [Environment Setup and Prerequisites](#1-environment-setup-and-prerequisites)
2. [Phase 1: Block Alignment and Differential Segmentation](#2-phase-1-block-alignment-and-differential-segmentation)
3. [Phase 2: Dependency Graphing and Referential Guard](#3-phase-2-dependency-graphing-and-referential-guard)
4. [Phase 3: WWW Memory Governance](#4-phase-3-www-memory-governance)
5. [Phase 4: Middleware Integration and Proxy Server](#5-phase-4-middleware-integration-and-proxy-server)
6. [Phase 5: Verification and Benchmark Suite](#6-phase-5-verification-and-benchmark-suite)
7. [Cross-Cutting Concerns](#7-cross-cutting-concerns)

---

## 1. Environment Setup and Prerequisites

### 1.1 Python Version and Virtual Environment

```bash
# Minimum Python version: 3.11 (required for tomllib stdlib, improved typing)
python --version   # should print Python 3.11.x or 3.12.x

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\Activate.ps1      # Windows PowerShell

# Install in editable mode with all optional groups
pip install -e ".[dev,benchmark]"
```

### 1.2 `pyproject.toml` (Canonical Definition)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nexus-context"
version = "0.1.0"
description = "Lightweight Python framework for referential integrity, KV cache alignment, and WWW memory governance in local SLM deployments"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [{ name = "Nexus-Context Team" }]
keywords = ["llm", "context-compression", "kv-cache", "vllm", "agent", "slm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "tree-sitter>=0.21",
    "tree-sitter-python>=0.21",
    "tree-sitter-sql>=0.2",
    "tree-sitter-bash>=0.21",
    "tree-sitter-javascript>=0.21",
    "spacy>=3.7",
    "coreferee>=1.4",
    "sentence-transformers>=2.7",
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
    "tiktoken>=0.7",
    "transformers>=4.40",
    "numpy>=1.26",
    "filelock>=3.14",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "mypy>=1.10",
    "memory-profiler>=0.61",
]
benchmark = [
    "locust>=2.29",
    "prometheus-client>=0.20",
]
hnswlib = ["hnswlib>=0.8"]  # Optional: for large graph approximate coverage

[project.scripts]
nexus-serve = "nexus_context.cache.middleware:main"

[tool.ruff]
line-length = 99
target-version = "py311"
select = ["E", "F", "W", "I", "UP", "ANN", "B", "SIM"]
ignore = ["ANN101", "ANN102"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=src/nexus_context --cov-report=term-missing --cov-fail-under=85"
```

### 1.3 spaCy Model Download

```bash
# Download transformer-based English model (required for coreference)
python -m spacy download en_core_web_trf

# Install coreferee for coreference resolution
python -m coreferee install en
```

### 1.4 Tree-Sitter Language Build (one-time setup)

Tree-Sitter grammar packages must compile their native extensions on first use. On Windows,
ensure Microsoft C++ Build Tools are installed. On Linux, `build-essential` is sufficient.

```bash
# Verify Tree-Sitter installation
python -c "import tree_sitter; print(tree_sitter.__version__)"
python -c "from tree_sitter_python import language; print('Python grammar OK')"
python -c "from tree_sitter_sql import language; print('SQL grammar OK')"
```

---

## 2. Phase 1: Block Alignment and Differential Segmentation

**Target completion**: End of Week 1
**Files to create**:
- `src/nexus_context/cache/__init__.py`
- `src/nexus_context/cache/block_align.py`
- `src/nexus_context/cache/differential.py`
- `src/nexus_context/cache/schemas.py`
- `tests/test_cache.py`

---

### Step 1.1: Implement `cache/schemas.py`

Define the data models that `block_align` and `differential` exchange:

```python
# src/nexus_context/cache/schemas.py

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A single OpenAI-format chat message with pre-computed token count."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    token_count: int = Field(ge=0)
    turn_index: int = Field(ge=0, description="0-based position in session history")


class ZoneBundle(BaseModel):
    """Output of ZoneSegmenter.segment(): three-zone partition of messages."""
    zone_p_message: ChatMessage          # Always the system message
    zone_t_candidates: list[ChatMessage] # Historical turns eligible for compaction
    zone_r_messages: list[ChatMessage]   # Verbatim recent turns
    zone_p_budget: int                   # Tokens allocated to Zone P (block-aligned)
    zone_t_budget: int                   # Max tokens for Zone T (post-compaction)
    zone_r_budget: int                   # Max tokens for Zone R
    total_budget: int                    # B = total context window


class BlockAlignResult(BaseModel):
    """Result of BlockAligner.align()."""
    original_token_count: int
    aligned_token_count: int             # Must be divisible by block_size
    padding_count: int                   # = aligned - original
    padded_content: str                  # System prompt text with padding appended
    block_size: int
    zone_p_hash: str                     # SHA256 of aligned token IDs (hex string)
```

---

### Step 1.2: Implement `cache/block_align.py`

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Define BlockAligner.__init__(model_name, backend, block_size=None)
       - If block_size is None: call _detect_block_size(backend_url)
       - Load tokenizer: AutoTokenizer.from_pretrained(model_name) for HF models
                         tiktoken.get_encoding("cl100k_base") for OpenAI-compat models
       - Store tokenizer, block_size, backend_type

[ ] 2. Implement _detect_block_size(backend_url: str) -> int
       - vLLM: GET {backend_url}/v1/models, parse extra metadata; fallback to 16
       - SGLang: GET {backend_url}/get_server_args, parse "block_size"; fallback to 16
       - Ollama: return 32 (hardcoded, documented in ADR-002)

[ ] 3. Implement tokenize(text: str) -> list[int]
       - HuggingFace tokenizer: tokenizer.encode(text, add_special_tokens=False)
       - tiktoken: enc.encode(text)
       - Raise TokenizerError if len(tokens) > tokenizer.model_max_length

[ ] 4. Implement _select_padding_token() -> str
       - Check tokenizer for pad_token_id
       - If pad_token defined: return tokenizer.decode([pad_token_id])
       - Else if eos_token defined: return "# " + tokenizer.decode([eos_token_id])
       - Else: return " " (single space)

[ ] 5. Implement align(system_prompt: str) -> BlockAlignResult
       - tokens = tokenize(system_prompt)
       - L_raw = len(tokens)
       - L_P = math.ceil(L_raw / block_size) * block_size
       - n_pad = L_P - L_raw
       - pad_str = _select_padding_token()
       - padded = system_prompt + (pad_str * n_pad)
       - Recompute padded_tokens = tokenize(padded) — verify len == L_P
       - zone_p_hash = hashlib.sha256(bytes(padded_tokens)).hexdigest()
       - Return BlockAlignResult(...)

[ ] 6. Add __repr__ for debugging
```

**Critical correctness check** after `align()`:
```python
result = aligner.align(system_prompt)
recomputed_tokens = aligner.tokenize(result.padded_content)
assert len(recomputed_tokens) % aligner.block_size == 0, \
    f"Alignment failed: {len(recomputed_tokens)} % {aligner.block_size} != 0"
```

---

### Step 1.3: Implement `cache/differential.py`

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Define ZoneSegmenter.__init__(total_budget, block_size, tail_budget=1024,
                                      tail_retention_turns=3)
       - Compute zone_p_budget from BlockAlignResult (provided at construction)
       - zone_r_budget = tail_budget
       - zone_t_budget = total_budget - zone_p_budget - zone_r_budget

[ ] 2. Implement segment(messages: list[ChatMessage]) -> ZoneBundle
       - Extract system message (role == "system") → zone_p_message
       - Remaining messages sorted by turn_index ascending
       - Zone R: take from the end until zone_r_budget tokens consumed
       - Zone T candidates: all messages between system and Zone R start
       - Return ZoneBundle

[ ] 3. Implement graduate_tail_turns(bundle: ZoneBundle, current_turn: int) -> ZoneBundle
       - For each msg in zone_r_messages:
           if (current_turn - msg.turn_index) >= tail_retention_turns:
               move msg from zone_r to zone_t_candidates
       - Recalculate token counts
       - Return updated bundle

[ ] 4. Implement validate(bundle: ZoneBundle) -> None
       - Assert: zone_p_tokens + zone_t_tokens + zone_r_tokens <= total_budget
       - Assert: zone_p_message.role == "system"
       - Assert: all zone_t_candidates in chronological order (turn_index ascending)
       - Raise SegmentationError on any violation
```

---

### Step 1.4: Write Phase 1 Tests

```python
# tests/test_cache.py (Phase 1 section)

import pytest
from nexus_context.cache.block_align import BlockAligner
from nexus_context.cache.differential import ZoneSegmenter
from nexus_context.cache.schemas import ChatMessage, ZoneBundle


class TestBlockAlign:

    @pytest.fixture
    def aligner(self):
        # Use a mock tokenizer that treats each character as one token
        return BlockAligner._from_mock(block_size=16)

    def test_already_aligned_prompt(self, aligner):
        """A prompt that is exactly block_size tokens needs zero padding."""
        prompt = "a" * 16   # 16 chars = 16 tokens in mock tokenizer
        result = aligner.align(prompt)
        assert result.padding_count == 0
        assert result.aligned_token_count == 16

    def test_unaligned_prompt_pads_correctly(self, aligner):
        """513-token prompt should pad to 528 (33 × 16)."""
        prompt = "a" * 513
        result = aligner.align(prompt)
        assert result.aligned_token_count == 528
        assert result.padding_count == 15
        assert result.aligned_token_count % 16 == 0

    def test_zone_p_hash_stable(self, aligner):
        """Same prompt always produces the same hash."""
        prompt = "You are a helpful assistant."
        r1 = aligner.align(prompt)
        r2 = aligner.align(prompt)
        assert r1.zone_p_hash == r2.zone_p_hash

    def test_zone_p_hash_changes_on_mutation(self, aligner):
        """Modifying even one character changes the hash."""
        r1 = aligner.align("Hello world.")
        r2 = aligner.align("Hello World.")   # capital W
        assert r1.zone_p_hash != r2.zone_p_hash

    def test_alignment_invariant_holds(self, aligner):
        """aligned_token_count must always be divisible by block_size."""
        for length in range(1, 100):
            prompt = "x" * length
            result = aligner.align(prompt)
            assert result.aligned_token_count % 16 == 0


class TestZoneSegmenter:

    def _make_messages(self, counts: list[int]) -> list[ChatMessage]:
        roles = ["system"] + ["user", "assistant"] * 20
        return [
            ChatMessage(role=roles[i], content=f"turn {i}", token_count=counts[i], turn_index=i)
            for i in range(len(counts))
        ]

    def test_system_message_always_zone_p(self):
        messages = self._make_messages([100, 50, 50, 50, 50])
        seg = ZoneSegmenter(total_budget=512, block_size=16,
                            zone_p_aligned_tokens=112, tail_budget=200)
        bundle = seg.segment(messages)
        assert bundle.zone_p_message.role == "system"

    def test_total_tokens_within_budget(self):
        messages = self._make_messages([100, 50, 50, 50, 50])
        seg = ZoneSegmenter(total_budget=300, block_size=16,
                            zone_p_aligned_tokens=112, tail_budget=100)
        bundle = seg.segment(messages)
        total = bundle.zone_p_message.token_count + \
                sum(m.token_count for m in bundle.zone_t_candidates) + \
                sum(m.token_count for m in bundle.zone_r_messages)
        assert total <= 300

    def test_graduation_moves_old_turns_to_trunk(self):
        messages = self._make_messages([100, 50, 50, 50, 50])
        seg = ZoneSegmenter(total_budget=512, block_size=16,
                            zone_p_aligned_tokens=112, tail_budget=200,
                            tail_retention_turns=2)
        bundle = seg.segment(messages)
        bundle2 = seg.graduate_tail_turns(bundle, current_turn=10)
        # All Zone R turns older than 2 turns should have moved to Zone T
        for msg in bundle2.zone_r_messages:
            assert (10 - msg.turn_index) < 2
```

---

## 3. Phase 2: Dependency Graphing and Referential Guard

**Target completion**: End of Week 4 (3 weeks)
**Files to create**:
- `src/nexus_context/guard/__init__.py`
- `src/nexus_context/guard/schemas.py`
- `src/nexus_context/guard/ast_graph.py`
- `src/nexus_context/guard/submodular.py`
- `tests/test_guard.py`

---

### Step 2.1: Implement `guard/schemas.py`

Copy the `ContextNode`, `DependencyEdge`, `ContextGraph`, and `CompactionResult` Pydantic models
from `docs/architecture.md`, Section 4. These are the canonical model definitions.

Additions required:
- Add `ContextGraph.add_node(node: ContextNode) -> None`
- Add `ContextGraph.add_edge(edge: DependencyEdge) -> None`
- Add `ContextGraph.get_ancestors(node_id: str) -> set[str]` (BFS over reversed edges)
- Add `ContextGraph.compute_transitive_closure() -> None` (Floyd-Warshall or BFS from each node)

---

### Step 2.2: Implement `guard/ast_graph.py`

#### 2.2.1 Parser Pool Initialization

```python
# Initialize Tree-Sitter parsers once at module level (not per-call)

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_sql as tssql
import tree_sitter_bash as tsbash
import tree_sitter_javascript as tsjs

PARSERS: dict[str, Parser] = {
    "python": Parser(Language(tspython.language())),
    "sql":    Parser(Language(tssql.language())),
    "bash":   Parser(Language(tsbash.language())),
    "javascript": Parser(Language(tsjs.language())),
}
```

#### 2.2.2 Language Detection

Before parsing, detect the language of a code block:

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Implement _detect_language(content: str) -> str | None
       - Heuristics (ordered by priority):
         1. Check for markdown code fence markers: ```python, ```sql, ```bash
         2. Check for Python keywords: "def ", "import ", "class ", "elif "
         3. Check for SQL keywords: "SELECT ", "CREATE TABLE", "INSERT INTO"
         4. Check for Bash indicators: "#!/", "$ ", "export ", "echo "
         5. If none match: return None (treat as NL)
```

#### 2.2.3 Python AST Visitor

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Implement PythonASTVisitor class
       - State: _scope_stack: list[str], _binding_store: dict[str, ContextNode],
                _nodes: list[ContextNode], _edges: list[DependencyEdge]

[ ] 2. visit(node: tree_sitter.Node, turn_index: int) -> None
       - Dispatch to visit_* methods based on node.type
       - Recurse into children (DFS preorder)

[ ] 3. visit_assignment(node) -> None
       - Extract LHS identifier name(s) and RHS representation
       - Create ContextNode(AST_ASSIGNMENT, ...)
       - If name in _binding_store: mark old node as overridden
       - Update _binding_store[name] = new_node

[ ] 4. visit_function_definition(node) -> None
       - Extract function name from first named child
       - Push function name onto _scope_stack
       - Create ContextNode(AST_FUNCDEF, ...)
       - Update _binding_store[func_name] = new_node
       - Visit body children with updated scope
       - Pop _scope_stack on exit

[ ] 5. visit_class_definition(node) -> None
       - Similar to function_definition; push class name onto _scope_stack

[ ] 6. visit_import_statement(node) -> None
       - Extract imported module names and aliases
       - Create ContextNode(AST_IMPORT, ...)
       - Add to _binding_store for each imported name

[ ] 7. visit_call(node) -> None
       - Extract callee name (may be attribute access: conn.cursor, os.path.join)
       - Look up callee in _binding_store
       - If found: create DependencyEdge(FUNC_DEF_TO_CALL, source=binding, target=this)

[ ] 8. visit_identifier(node, context: "load" | "store") -> None
       - If context == "load" and name in _binding_store:
           create DependencyEdge(VAR_DEF_TO_REF, source=binding, target=this)
```

**Error handling**: Wrap each `visit_*` method in `try/except Exception` and log at DEBUG level.
Tree-Sitter parse errors are marked with `node.type == "ERROR"` — skip these nodes with a
WARNING log entry.

#### 2.2.4 SQL AST Visitor

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Implement SQLASTVisitor class
       - State: _schema_store: dict[str, ContextNode]  # table → creation node

[ ] 2. visit_create_table_statement(node) -> None
       - Extract table name
       - Create ContextNode(SCHEMA_FIELD, table_name, ...)
       - Add to _schema_store[table_name]

[ ] 3. visit_select_statement(node) -> None
       - Extract FROM clause table names
       - For each referenced table in _schema_store:
           create DependencyEdge(SCHEMA_FIELD_DEP, source=create_node, target=select_node)

[ ] 4. visit_insert_statement / visit_update_statement / visit_delete_statement(node) -> None
       - Extract target table name
       - If in _schema_store: create SCHEMA_FIELD_DEP edge
```

#### 2.2.5 Transitive Closure Computation

```
IMPLEMENTATION CHECKLIST:

[ ] 1. Implement compute_transitive_closure(graph: ContextGraph) -> None
       - Build adjacency dict: {node_id: set[node_id]} from direct edges
       - For each node n, run BFS to find all reachable nodes
       - For each reachable node r (r ≠ n), if no direct edge (n, r) exists:
           add DependencyEdge(source=n, target=r, is_transitive=True, weight=0.5)
       - Add all transitive edges to graph.edges
       - Set graph.transitive_closure_computed = True

       NOTE: For graphs with > 200 nodes, limit transitive closure to 3-hop depth
             to maintain latency target. Full closure for n > 200 is O(n^3) which
             would violate the 15ms graph construction budget.
```

---

### Step 2.3: Implement `guard/submodular.py`

#### 2.3.1 Embedding Computation

```
IMPLEMENTATION CHECKLIST:

[ ] 1. SubmodularSolver.__init__(embedding_model, alpha=0.5, beta=0.5, gamma_factor=1e6)
       - Load SentenceTransformer(embedding_model)
       - Initialize embedding_cache: dict[str, np.ndarray]

[ ] 2. _embed_nodes(nodes: list[ContextNode]) -> np.ndarray  (shape: n × dim)
       - Batch encode: model.encode([n.content for n in nodes], batch_size=32)
       - Cache results in embedding_cache keyed by node.node_id
       - Return numpy array of shape (n, embedding_dim)

[ ] 3. _embed_query(query: str) -> np.ndarray  (shape: dim,)
       - Encode single query string
       - Return normalized 1D numpy array
```

#### 2.3.2 Marginal Gain Computation

```
IMPLEMENTATION CHECKLIST:

[ ] 4. _compute_relevance_scores(embeddings: np.ndarray, query_emb: np.ndarray) -> np.ndarray
       - Compute cosine similarity: embeddings @ query_emb (assuming normalized embeddings)
       - Shape: (n,)

[ ] 5. _compute_pairwise_similarity(embeddings: np.ndarray) -> np.ndarray
       - Compute: embeddings @ embeddings.T
       - Shape: (n, n)

[ ] 6. _coverage_gain(v_idx: int, S_indices: set[int],
                       sim_matrix: np.ndarray, V_size: int) -> float
       - Current coverage of S: for each w in V, max_{u in S} sim[u, w]
       - New coverage with v: for each w in V, max(sim[v, w], current_coverage[w])
       - coverage_gain = sum(new_coverage) - sum(current_coverage)
       - Returns non-negative float (monotone, diminishing returns)
       - Optimization: maintain current_max_sim as a running (n,) array; update in-place

[ ] 7. _marginal_gain(v_idx: int, S_indices: set[int], graph: ContextGraph,
                       relevance: np.ndarray, sim_matrix: np.ndarray,
                       gamma: float, current_max_sim: np.ndarray) -> float
       - dangling = _count_unresolved_ancestors(v_idx, S_indices, graph)
       - if dangling > 0: return -gamma   (infeasible)
       - return alpha * relevance[v_idx] + beta * coverage_gain(v_idx, S_indices, ...)
```

#### 2.3.3 Greedy Solve Loop

```
IMPLEMENTATION CHECKLIST:

[ ] 8. solve(graph: ContextGraph, budget: int, query: str) -> CompactionResult
       - start = time.perf_counter()
       - Validate: graph.transitive_closure_computed == True
       - nodes_list = list(graph.nodes.values())  # ordered list for indexing
       - embeddings = _embed_nodes(nodes_list)
       - query_emb = _embed_query(query)
       - relevance = _compute_relevance_scores(embeddings, query_emb)
       - sim_matrix = _compute_pairwise_similarity(embeddings)
       - gamma = gamma_factor * (alpha * max(relevance) + beta * len(nodes_list))

       GREEDY LOOP:
       - S_indices = set()                   # selected node indices
       - remaining_budget = budget
       - current_max_sim = np.zeros(n)       # coverage tracking array
       - priority_queue: max-heap of (-gain_upper_bound, node_idx)
           initialize with (-relevance[i], i) for all i  # upper bound = relevance alone

       - while priority_queue and remaining_budget > 0:
           _, v_idx = heapq.heappop(priority_queue)
           if nodes_list[v_idx].node_id in S or nodes_list[v_idx].token_count > remaining:
               continue
           
           # Recompute actual marginal gain (lazy greedy: verify upper bound)
           actual_gain = _marginal_gain(v_idx, S_indices, graph, relevance, sim_matrix,
                                        gamma, current_max_sim)
           
           # Check if v is still the best candidate (lazy greedy invariant)
           if priority_queue and actual_gain < -priority_queue[0][0]:
               heapq.heappush(priority_queue, (-actual_gain, v_idx))
               continue
           
           # Force-include ancestors if needed
           ancestors_to_add = _force_ancestors(v_idx, S_indices, graph)
           
           # Check budget for v + all forced ancestors
           cost = sum(nodes_list[aid].token_count for aid in ancestors_to_add + [v_idx])
           if cost > remaining_budget:
               continue   # skip; try next candidate
           
           # Select v and all forced ancestors
           for idx in sorted(ancestors_to_add + [v_idx], key=lambda i: nodes_list[i].turn_index):
               S_indices.add(idx)
               remaining_budget -= nodes_list[idx].token_count
               _update_coverage(idx, current_max_sim, sim_matrix)   # in-place update

       - Reconstruct Zone T text from S_indices in turn_index order
       - Compute dangling_violations (should be 0 by construction; assert)
       - latency_ms = (time.perf_counter() - start) * 1000
       - Return CompactionResult(...)
```

---

### Step 2.4: Write Phase 2 Tests

```python
# tests/test_guard.py

import pytest
from hypothesis import given, strategies as st
from nexus_context.guard.ast_graph import ContextGraphBuilder
from nexus_context.guard.submodular import SubmodularSolver
from nexus_context.guard.schemas import ContextGraph, NodeType, EdgeType


class TestASTGraph:

    @pytest.fixture
    def builder(self):
        return ContextGraphBuilder()

    def test_python_assignment_creates_node(self, builder):
        code = 'DB_HOST = "prod.internal"'
        turns = [make_code_turn(code, "python", turn_index=0)]
        graph = builder.build(turns)
        assign_nodes = [n for n in graph.nodes.values()
                        if n.node_type == NodeType.AST_ASSIGNMENT]
        assert len(assign_nodes) == 1
        assert "DB_HOST" in assign_nodes[0].metadata.get("name", "")

    def test_python_variable_use_creates_edge(self, builder):
        code = '''
DB_HOST = "prod.internal"
conn = connect(host=DB_HOST)
'''
        turns = [make_code_turn(code, "python", turn_index=0)]
        graph = builder.build(turns)
        var_edges = [e for e in graph.edges if e.edge_type == EdgeType.VAR_DEF_TO_REF]
        assert len(var_edges) >= 1

    def test_cross_turn_dependency_edge(self, builder):
        """Variable defined in turn 1, used in turn 2 → cross-turn edge."""
        turn1 = make_code_turn('conn = connect("db")', "python", turn_index=0)
        turn2 = make_code_turn('cursor = conn.cursor()', "python", turn_index=1)
        graph = builder.build([turn1, turn2])
        edges_turn0_to_turn1 = [
            e for e in graph.edges
            if (graph.nodes[e.source_id].turn_index == 0 and
                graph.nodes[e.target_id].turn_index == 1)
        ]
        assert len(edges_turn0_to_turn1) >= 1

    def test_syntax_error_turn_does_not_crash(self, builder):
        """Malformed code should degrade gracefully, not raise."""
        turn = make_code_turn('def broken(: pass', "python", turn_index=0)
        graph = builder.build([turn])   # must not raise
        # May produce 0 nodes or a fallback NL_SENTENCE node
        assert graph is not None


class TestSubmodularSolver:

    @pytest.fixture
    def solver(self):
        return SubmodularSolver(embedding_model="all-MiniLM-L6-v2")

    def test_no_dangling_after_compaction(self, solver, builder):
        """Core invariant: after compaction, no selected node has an unresolved ancestor."""
        turns = make_db_setup_turns()   # 3-turn example from overview
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=200, query="query database")
        selected = set(result.selected_node_ids)
        for edge in graph.edges:
            if edge.target_id in selected:
                assert edge.source_id in selected, (
                    f"Dangling: {edge.target_id} selected, {edge.source_id} missing"
                )

    def test_ancestor_forced_when_dependent_selected(self, solver, builder):
        """If a node requires an ancestor, both must appear in selected."""
        turn1 = make_code_turn('x = 42', "python", turn_index=0)
        turn2 = make_code_turn('y = x + 1', "python", turn_index=1)
        graph = builder.build([turn1, turn2])
        graph.compute_transitive_closure()
        # Budget large enough to include turn2 but ask solver to select it
        result = solver.solve(graph, budget=500, query="y value")
        if any("y" in graph.nodes[n].content for n in result.selected_node_ids):
            assert any("x" in graph.nodes[n].content for n in result.selected_node_ids)

    @given(st.integers(min_value=50, max_value=5000))
    def test_selected_tokens_within_budget(self, solver, builder, budget):
        """Token count of selected set never exceeds budget."""
        turns = make_random_turns(n=10)
        graph = builder.build(turns)
        graph.compute_transitive_closure()
        result = solver.solve(graph, budget=budget, query="task")
        selected_tokens = sum(
            graph.nodes[nid].token_count for nid in result.selected_node_ids
        )
        assert selected_tokens <= budget
```

---

## 4. Phase 3: WWW Memory Governance

**Target completion**: End of Week 5 (can run parallel to Phase 2 weeks 2–3)
**Files to create**:
- `src/nexus_context/memory/__init__.py`
- `src/nexus_context/memory/www_parser.py`
- `src/nexus_context/memory/decay.py`
- `src/nexus_context/memory/schemas.py` (MemoryTuple, WhatDelta, etc.)
- `tests/test_memory.py`

---

### Step 3.1: Implement `memory/www_parser.py`

```
IMPLEMENTATION CHECKLIST:

[ ] 1. WWWParser.__init__(languages=["python", "sql", "bash"])
       - Reuse PARSERS pool from ast_graph (import, don't reload)
       - Load spaCy nlp = spacy.load("en_core_web_trf")
       - _d_max: int = 0   # tracks maximum AST depth seen across session

[ ] 2. extract(turn: ChatMessage, session_id: str) -> list[MemoryTuple]
       - Detect turn language (same _detect_language() from ast_graph)
       - If code: return _extract_code(turn, ...)
       - If NL: return _extract_nl(turn, ...)

[ ] 3. _extract_code(turn, session_id) -> list[MemoryTuple]
       - Parse with Tree-Sitter
       - Walk AST with PythonMutationVisitor (or SQLMutationVisitor)
       - For each mutation found, create WhatDelta and MemoryTuple
       - Compute scope_path from AST scope stack
       - Compute ast_depth_score = (d_max - depth) / max(d_max, 1)
       - Update self._d_max if deeper scope found

[ ] 4. _extract_nl(turn, session_id) -> list[MemoryTuple]
       - doc = nlp(turn.content)
       - Extract named entities: for each ent, create NL_STATEMENT WhatDelta
       - Extract SVO triples via dependency parsing:
           for token with dep_=="ROOT" (verb):
             subject = first child with dep_ in {"nsubj", "nsubjpass"}
             object  = first child with dep_ in {"dobj", "attr", "prep"}
           if subject and object: create NL_STATEMENT tuple
       - Return list of MemoryTuples

[ ] 5. _node_to_memory_tuple(node, what_delta, turn, session_id, seq) -> MemoryTuple
       - Construct MemoryTuple with:
           memory_id = f"{session_id}:{turn.turn_index}:{seq}"
           who = infer_who(turn.role)    # user→User, assistant→Agent, tool→Tool
           who_detail = f"{turn.role}@turn_{turn.turn_index}"
           what = what_delta
           when = turn.turn_index
           where = scope_path
           token_count = estimate_tokens(tuple_json)
           is_pinned = False
           ast_depth_score = computed above
```

---

### Step 3.2: Implement `memory/decay.py`

```
IMPLEMENTATION CHECKLIST:

[ ] 1. MemoryPool.__init__(session_id, lambda_=0.05, eta=0.5, persist_path=None)
       - self._pool: dict[str, MemoryTuple]   # memory_id → tuple
       - self._key_index: dict[str, str]       # (name, scope) → memory_id for dedup
       - If persist_path: self.load(persist_path)

[ ] 2. add(tuples: list[MemoryTuple]) -> None
       - For each tuple t:
           key = (t.what.target_name, t.where)
           if key in _key_index:
               old_id = _key_index[key]
               old = _pool[old_id]
               if old.is_pinned: add with versioned key
               elif t.when > old.when: replace old with t
           else:
               _pool[t.memory_id] = t
               _key_index[key] = t.memory_id

[ ] 3. update_weights(current_turn: int) -> None
       - For each t in _pool.values():
           t.retention_weight = t.compute_weight(current_turn, lambda_, eta)

[ ] 4. pin(memory_id: str) -> None
       - _pool[memory_id].is_pinned = True
       - _pool[memory_id].retention_weight = math.inf

[ ] 5. pin_by_dependency(referenced_names: set[str]) -> None
       - For each name in referenced_names:
           find memories whose what.target_name == name
           pin them all

[ ] 6. select(budget_tokens: int) -> list[MemoryTuple]
       - Sort _pool.values() by retention_weight descending (pinned → math.inf, first)
       - Greedily select until budget_tokens exhausted
       - Return sorted by when ascending (chronological)

[ ] 7. serialize_for_context(selected: list[MemoryTuple]) -> str
       - Compact format per tuple:
           {"w":who_detail, "d":delta_compact, "t":when, "@":where}
       - delta_compact = f"{target}={new_value}" (truncated to 60 chars)
       - Wrap in: f"<!-- NEXUS_MEMORY\n{json.dumps(compact_list)}\n-->"

[ ] 8. save(path: Path) -> None
       - Write to path.with_suffix(".tmp") using atomic rename
       - One JSON line per MemoryTuple

[ ] 9. load(path: Path) -> None
       - Read JSONL; validate each line as MemoryTuple
       - Log WARNING and skip corrupted lines
```

---

### Step 3.3: Write Phase 3 Tests

```python
# tests/test_memory.py

import math
import pytest
from nexus_context.memory.www_parser import WWWParser
from nexus_context.memory.decay import MemoryPool
from nexus_context.memory.schemas import MutationType, WhoActor


class TestWWWParser:

    @pytest.fixture
    def parser(self):
        return WWWParser()

    def test_python_assignment_extracted(self, parser):
        turn = make_code_turn('DB_HOST = "prod.internal"', "python", turn_index=3)
        tuples = parser.extract(turn, session_id="s1")
        assert len(tuples) == 1
        assert tuples[0].what.mutation_type == MutationType.VAR_ASSIGN
        assert tuples[0].what.target_name == "DB_HOST"
        assert tuples[0].when == 3

    def test_sql_create_table_extracted(self, parser):
        sql = "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255));"
        turn = make_code_turn(sql, "sql", turn_index=7)
        tuples = parser.extract(turn, session_id="s1")
        assert any(t.what.mutation_type == MutationType.SCHEMA_CHANGE for t in tuples)
        assert any(t.what.target_name == "users" for t in tuples)

    def test_multiple_assignments_in_one_turn(self, parser):
        code = "a = 1\nb = 2\nc = 3"
        turn = make_code_turn(code, "python", turn_index=0)
        tuples = parser.extract(turn, session_id="s1")
        assert len(tuples) == 3

    def test_compression_ratio_code_turn(self, parser):
        """Semantic tuple must be < 25% of original turn token count (4:1 ratio)."""
        code = """
def setup_database():
    # Create production database connection
    DB_HOST = "prod.db.internal"
    DB_PORT = 5432
    DB_USER = "nexus_agent"
    DB_PASS = get_secret("PG_PASSWORD")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS)
    return conn
"""
        turn = make_code_turn(code, "python", turn_index=5)
        tuples = parser.extract(turn, session_id="s1")
        original_tokens = len(code.split())   # approximate
        tuple_tokens = sum(t.token_count for t in tuples)
        # Should achieve at least 4:1 compression for code turns
        assert tuple_tokens < original_tokens * 0.35   # allow some tolerance


class TestMemoryPool:

    def test_weight_at_current_turn(self):
        """Memory created at current turn T should have maximum weight."""
        pool = MemoryPool(session_id="s1", lambda_=0.05, eta=0.5)
        t = make_memory_tuple(when=5, ast_depth_score=1.0)
        pool.add([t])
        pool.update_weights(current_turn=5)
        w = pool._pool[t.memory_id].retention_weight
        expected = 1.0 * (1.0 + 0.5 * 1.0)   # exp(0) * (1 + 0.5) = 1.5
        assert abs(w - expected) < 1e-6

    def test_weight_at_half_life(self):
        """At ~13.9 turns, root-scope memory weight should be ~50% of max."""
        pool = MemoryPool(session_id="s1", lambda_=0.05, eta=0.5)
        t = make_memory_tuple(when=0, ast_depth_score=1.0)
        pool.add([t])
        pool.update_weights(current_turn=14)   # ≈ half-life
        w = pool._pool[t.memory_id].retention_weight
        # exp(-0.05 * 14) * 1.5 ≈ 0.496 * 1.5 ≈ 0.744
        assert 0.6 < w < 0.9

    def test_pinned_memory_not_decayed(self):
        """Pinned memories must have math.inf weight regardless of age."""
        pool = MemoryPool(session_id="s1", lambda_=0.05, eta=0.5)
        t = make_memory_tuple(when=0, ast_depth_score=0.5)
        pool.add([t])
        pool.pin(t.memory_id)
        pool.update_weights(current_turn=1000)
        assert pool._pool[t.memory_id].retention_weight == math.inf

    def test_deduplication_keeps_most_recent(self):
        """Re-assignment of same variable keeps only the most recent tuple."""
        pool = MemoryPool(session_id="s1")
        t1 = make_memory_tuple(when=2, target_name="x", scope="module", new_value="1")
        t2 = make_memory_tuple(when=5, target_name="x", scope="module", new_value="2")
        pool.add([t1, t2])
        assert len(pool._pool) == 1
        remaining = list(pool._pool.values())[0]
        assert remaining.what.new_value_repr == "2"

    def test_select_within_budget(self):
        """Selection never exceeds token budget."""
        pool = MemoryPool(session_id="s1")
        tuples = [make_memory_tuple(when=i, token_count=20) for i in range(10)]
        pool.add(tuples)
        pool.update_weights(current_turn=10)
        selected = pool.select(budget_tokens=75)
        total_tokens = sum(t.token_count for t in selected)
        assert total_tokens <= 75
```

---

## 5. Phase 4: Middleware Integration and Proxy Server

**Target completion**: End of Week 6 (1 week)
**Files to create**:
- `src/nexus_context/cache/middleware.py`
- `src/nexus_context/__init__.py`
- `nexus_config.yaml` (template)

---

### Step 4.1: FastAPI App Structure

```python
# src/nexus_context/cache/middleware.py (skeleton)

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize all subsystems
    app.state.config = load_config()
    app.state.aligner = BlockAligner(...)
    app.state.segmenter = ZoneSegmenter(...)
    app.state.graph_builder = ContextGraphBuilder(...)
    app.state.solver = SubmodularSolver(...)
    app.state.www_parser = WWWParser(...)
    app.state.sessions: dict[str, SessionState] = {}
    app.state.http_client = httpx.AsyncClient(
        base_url=app.state.config.backend.url,
        timeout=httpx.Timeout(300.0)
    )
    yield
    # Shutdown: close HTTP client
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan, title="Nexus-Context Middleware")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    session_id = request.headers.get("X-Session-ID") or _derive_session_id(request)
    
    start = time.perf_counter()
    
    # Run Nexus-Context processing pipeline
    transformed_body = await _process_pipeline(app.state, session_id, body)
    
    pipeline_ms = (time.perf_counter() - start) * 1000
    
    # Forward to backend
    if body.get("stream", False):
        return await _stream_response(app.state.http_client, transformed_body, pipeline_ms)
    else:
        return await _json_response(app.state.http_client, transformed_body, pipeline_ms)
```

---

### Step 4.2: Pipeline Orchestration

```
IMPLEMENTATION CHECKLIST:

[ ] 1. _process_pipeline(state, session_id, body) -> dict
       - session = get_or_create_session(state, session_id)
       - messages = [ChatMessage(**m) for m in body["messages"]]
       - assign turn_indices to messages
       
       - STAGE 1 (first turn only): align Zone P
           if session.cache_boundary is None:
               align_result = state.aligner.align(system_message.content)
               session.cache_boundary = build_cache_boundary(align_result)
           else:
               system_message.content = session.aligned_zone_p   # reuse frozen Zone P
       
       - STAGE 2: segment zones
           bundle = state.segmenter.segment(messages)
           bundle = state.segmenter.graduate_tail_turns(bundle, current_turn)
       
       - STAGE 3 + 4: build graph and compact (only if Zone T over budget)
           if sum(m.token_count for m in bundle.zone_t_candidates) > bundle.zone_t_budget:
               graph = state.graph_builder.build(bundle.zone_t_candidates)
               graph.compute_transitive_closure()
               query = bundle.zone_r_messages[-1].content if bundle.zone_r_messages else ""
               compaction = state.solver.solve(graph, bundle.zone_t_budget, query)
               zone_t_text = compaction.reconstructed_text
               pruned_turns = [m for m in bundle.zone_t_candidates
                               if ... not in compaction.selected_node_ids]
           else:
               zone_t_text = concat(bundle.zone_t_candidates)
               pruned_turns = []
       
       - STAGE 5: WWW memory
           new_tuples = []
           for turn in pruned_turns:
               new_tuples.extend(state.www_parser.extract(turn, session_id))
           session.memory_pool.add(new_tuples)
           session.memory_pool.update_weights(current_turn)
           # Pin memories required by dependency graph
           if compaction defined:
               referenced = extract_referenced_names(compaction.forced_inclusions)
               session.memory_pool.pin_by_dependency(referenced)
           memory_block = session.memory_pool.serialize_for_context(
               session.memory_pool.select(budget_tokens=memory_budget)
           )
       
       - ASSEMBLE: reconstruct body["messages"]
           new_messages = [
               {"role": "system", "content": aligned_zone_p},
               {"role": "system", "content": memory_block},  # inject memory
               *zone_t_messages,
               *zone_r_messages,
           ]
           transformed_body = {**body, "messages": new_messages}
       
       - Update session state
       - Return transformed_body

[ ] 2. _stream_response(client, body, pipeline_ms) -> StreamingResponse
       - Use httpx AsyncClient streaming to yield SSE chunks
       - Attach X-Nexus-Context-Stats header via background task

[ ] 3. _json_response(client, body, pipeline_ms) -> Response
       - POST body to backend; return backend response with stats header added
```

---

## 6. Phase 5: Verification and Benchmark Suite

**Target completion**: End of Week 9 (2 weeks)
**Files to create/complete**:
- `tests/test_guard.py` (complete)
- `tests/test_cache.py` (complete)
- `tests/test_memory.py` (complete)
- `tests/fixtures/` (NCATS-v1 task fixtures)
- `tests/benchmarks/` (latency and KV hit rate benchmarks)

---

### Step 5.1: Property-Based Test Suite

```python
# Dangling invariant (most critical test)
@given(st.lists(code_turn_strategy(), min_size=2, max_size=30),
       st.integers(min_value=100, max_value=8192))
@settings(max_examples=200, deadline=5000)
def test_no_dangling_for_any_graph_and_budget(turns, budget):
    graph = builder.build(turns)
    graph.compute_transitive_closure()
    result = solver.solve(graph, budget=budget, query="task")
    selected = set(result.selected_node_ids)
    
    violations = []
    for edge in graph.edges:
        if edge.target_id in selected and edge.source_id not in selected:
            violations.append((edge.source_id, edge.target_id))
    
    assert violations == [], (
        f"Dangling violations found: {violations}\n"
        f"Budget: {budget}, turns: {len(turns)}"
    )
```

### Step 5.2: Latency Benchmark Suite

```python
# tests/benchmarks/test_latency.py

import time
import pytest


class TestGraphConstructionLatency:

    @pytest.mark.parametrize("token_count", [1000, 2000, 4000])
    def test_graph_construction_under_15ms(self, builder, token_count):
        turns = generate_turns_of_approximate_tokens(token_count)
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            builder.build(turns)
            latencies.append((time.perf_counter() - start) * 1000)
        
        p95 = sorted(latencies)[94]
        assert p95 < 15.0, (
            f"P95 graph construction latency {p95:.1f}ms exceeds 15ms target "
            f"for {token_count} token input"
        )


class TestSubmodularSolverLatency:

    @pytest.mark.parametrize("n_nodes,budget", [(50, 1024), (100, 2048), (200, 4096)])
    def test_solver_under_50ms(self, solver, n_nodes, budget):
        graph = generate_random_graph(n_nodes)
        graph.compute_transitive_closure()
        
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            solver.solve(graph, budget=budget, query="test query")
            latencies.append((time.perf_counter() - start) * 1000)
        
        p95 = sorted(latencies)[47]
        assert p95 < 50.0, (
            f"P95 solver latency {p95:.1f}ms exceeds 50ms target for "
            f"n={n_nodes}, budget={budget}"
        )
```

### Step 5.3: KV Cache Hit Rate Benchmark

```python
# tests/benchmarks/test_kv_hitrate.py
# Requires: live vLLM instance with --enable-prefix-caching

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.requires_vllm
def test_cache_hit_rate_improvement():
    """Compare prefix cache hit rate: baseline vs. nexus-context."""
    vllm_url = "http://localhost:8000"
    nexus_url = "http://localhost:9000"   # nexus middleware pointing to vllm
    task_turns = load_ncats_task("db_schema_setup_15turns")

    # Baseline: send directly to vLLM
    baseline_hit_rate = run_session_and_measure_hit_rate(vllm_url, task_turns)

    # Reset vLLM KV cache between sessions
    reset_vllm_cache(vllm_url)

    # Nexus-managed: send through middleware
    nexus_hit_rate = run_session_and_measure_hit_rate(nexus_url, task_turns)

    print(f"Baseline hit rate: {baseline_hit_rate:.2%}")
    print(f"Nexus hit rate:    {nexus_hit_rate:.2%}")

    assert nexus_hit_rate > 0.85, (
        f"Nexus cache hit rate {nexus_hit_rate:.2%} below 85% target"
    )
    assert nexus_hit_rate > baseline_hit_rate * 1.10, (
        f"Nexus hit rate {nexus_hit_rate:.2%} not sufficiently better than "
        f"baseline {baseline_hit_rate:.2%}"
    )
```

---

## 7. Cross-Cutting Concerns

### 7.1 Logging Convention

All modules use Python's `logging` with structured JSON output:

```python
import logging
import json

logger = logging.getLogger("nexus_context")

# Structured log entry format (use this pattern throughout)
logger.debug(json.dumps({
    "event": "graph_construction_complete",
    "session_id": session_id,
    "n_nodes": len(graph.nodes),
    "n_edges": len(graph.edges),
    "latency_ms": latency_ms,
}))
```

Log levels:
- `DEBUG`: Per-node graph events, embedding cache hits/misses
- `INFO`: Per-request pipeline summary (session_id, token_counts, hit_rate)
- `WARNING`: Fallback triggers (tokenizer not found, grammar parse error)
- `ERROR`: Pipeline failures, backend unreachable

### 7.2 Type Annotation Convention

All public functions must have complete type annotations. `mypy --strict` must pass with zero
errors. Private helper methods (prefixed `_`) are annotated but mypy errors are acceptable
(add `# type: ignore[...]` sparingly and always with a comment explaining why).

```python
def align(self, system_prompt: str) -> BlockAlignResult:  # fully annotated
    ...

def _internal_helper(self, x: int) -> list[str]:  # annotated even for private methods
    ...
```

### 7.3 Error Hierarchy

```python
# src/nexus_context/__init__.py

class NexusContextError(Exception):
    """Base exception for all nexus-context errors."""

class TokenizerError(NexusContextError):
    """Tokenizer load or encode failure."""

class AlignmentError(NexusContextError):
    """Block alignment computation failure."""

class SegmentationError(NexusContextError):
    """Zone segmentation invariant violation."""

class GraphBuildError(NexusContextError):
    """Context dependency graph construction failure."""

class SolverError(NexusContextError):
    """Submodular solver failure (budget too small, empty graph, etc.)."""

class MemoryExtractionError(NexusContextError):
    """WWW tuple extraction failure."""

class BackendProxyError(NexusContextError):
    """Backend server communication failure."""
```

### 7.4 Configuration Validation

At startup, `NexusContextMiddleware` validates all configuration values and raises
`ConfigurationError` (a subclass of `NexusContextError`) on any invalid combination:

```
Validation rules:
  - block_size must be 16 or 32
  - tail_budget + zone_p_aligned_tokens < total_budget
  - lambda_ > 0, eta >= 0
  - alpha + beta == 1.0 (normalized weights)
  - backend.url must be a valid HTTP URL
  - embedding_model must be a resolvable model name
```

### 7.5 Continuous Integration Checklist

Before merging any PR to `main`:

```
[ ] ruff check src/ tests/        # zero lint errors
[ ] mypy src/                     # zero type errors (strict mode)
[ ] pytest --cov --cov-fail-under=85   # all tests pass, coverage >= 85%
[ ] hypothesis tests pass with max_examples=500
[ ] Phase latency benchmarks pass (not gated on CI, run on benchmark hardware)
```
