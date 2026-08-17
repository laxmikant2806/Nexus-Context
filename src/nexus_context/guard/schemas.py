"""
nexus_context.guard.schemas
============================
Pydantic v2 data models for the guard sub-package.

Models
------
NodeType         – enum of semantic node categories
EdgeType         – enum of dependency edge categories
ContextNode      – a vertex in the context dependency graph
DependencyEdge   – a directed edge (u → v) in the graph
ContextGraph     – full graph for one session with traversal helpers
CompactionResult – output of SubmodularSolver.solve()

Reference: docs/architecture.md §4.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(str, enum.Enum):
    """Semantic classification of a context graph node."""

    TURN = "turn"                  # A complete conversational turn
    CODE_BLOCK = "code_block"      # A code execution block (un-parsed)
    AST_ASSIGNMENT = "ast_assign"  # Variable assignment AST node
    AST_FUNCDEF = "ast_funcdef"    # Function definition AST node
    AST_CLASSDEF = "ast_classdef"  # Class definition AST node
    AST_IMPORT = "ast_import"      # Import statement AST node
    AST_CALL = "ast_call"          # Function/method call AST node
    NL_SENTENCE = "nl_sentence"    # Natural-language sentence
    TOOL_RETURN = "tool_return"    # Tool invocation return value
    SCHEMA_FIELD = "schema_field"  # JSON / Pydantic / SQL schema field
    ADAPTIVE_CHUNK = "adaptive_chunk"  # Self-healing adaptive semantic chunk
    TOOL_JSON_FIELD = "tool_json_field"  # Feature A: key-value field from tool-call JSON response
    API_RESPONSE = "api_response"        # Feature A: full captured API/tool response node


class EdgeType(str, enum.Enum):
    """Semantic classification of a context dependency edge."""

    VAR_DEF_TO_REF = "var_def_to_ref"       # Variable definition → reference
    FUNC_DEF_TO_CALL = "func_def_to_call"   # Function definition → call site
    CLASS_DEF_TO_USE = "class_def_to_use"   # Class definition → instantiation
    IMPORT_TO_USE = "import_to_use"         # Import → usage in code
    COREF_ANTECEDENT = "coref_antecedent"   # NL coreference chain
    TOOL_RETURN_DEP = "tool_return_dep"     # Tool return → downstream argument
    SCHEMA_FIELD_DEP = "schema_field_dep"   # Schema field definition → usage
    CHUNK_SEQUENCE = "chunk_sequence"       # Relational link between adjacent adaptive chunks
    TOOL_FIELD_TO_CODE_REF = "tool_field_to_code_ref"  # Feature A: JSON response field → code symbol
    SCHEMA_TO_TOOL_RETURN = "schema_to_tool_return"    # Feature A: DB/JSON schema → matching tool return


# ---------------------------------------------------------------------------
# ContextNode
# ---------------------------------------------------------------------------


class ContextNode(BaseModel):
    """A node in the context dependency graph.

    Attributes
    ----------
    node_id:
        Unique identifier; format ``{turn_index}:{node_type}:{name}`` where
        *name* is the binding name for code nodes or a short content hash for
        natural-language nodes.
    node_type:
        Semantic classification of this node.
    turn_index:
        0-based index of the originating conversational turn.
    content:
        Raw text content of the node (code fragment or NL sentence).
    token_count:
        Pre-computed number of tokens in *content*.
    language:
        Programming language if this is a code node (``"python"``,
        ``"sql"``, ``"bash"``, ``"json"``).  ``None`` for NL nodes.
    ast_depth:
        Nesting depth within the AST scope hierarchy (0 = module/root level).
    scope_path:
        Dot-separated AST scope path, e.g. ``"module.MyClass.my_method"``.
    is_pinned:
        If ``True``, this node cannot be pruned regardless of budget.
    metadata:
        Language-specific metadata dict (e.g. ``{"name": "DB_HOST",
        "rhs": '"prod.internal"'}`` for an assignment node).
    """

    node_id: str = Field(
        description="Unique node ID: '{turn_index}:{node_type}:{name}'."
    )
    node_type: NodeType
    turn_index: int = Field(ge=0)
    content: str
    token_count: int = Field(ge=0)
    language: str | None = Field(default=None)
    ast_depth: int = Field(default=0, ge=0)
    scope_path: str = Field(default="module")
    is_pinned: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# DependencyEdge
# ---------------------------------------------------------------------------


class DependencyEdge(BaseModel):
    """A directed dependency edge (source → target) in the context graph.

    A retained node *target* requires *source* to also be retained.
    """

    edge_id: str = Field(
        description="'{source_id}->{target_id}'."
    )
    source_id: str = Field(description="node_id of the defining / antecedent node.")
    target_id: str = Field(description="node_id of the referencing / dependent node.")
    edge_type: EdgeType
    weight: float = Field(
        default=1.0, ge=0.0,
        description="Edge weight for weighted dependency scoring (see architecture.md §5.5).",
    )
    is_transitive: bool = Field(
        default=False,
        description="True if derived via transitive-closure computation.",
    )


# ---------------------------------------------------------------------------
# ContextGraph
# ---------------------------------------------------------------------------


class ContextGraph(BaseModel):
    """The full context dependency graph G = (V, E) for one session.

    Nodes are stored in a dict keyed by ``node_id`` for O(1) lookup.
    Edges are stored as a list; an adjacency index is built lazily.
    """

    session_id: str
    nodes: dict[str, ContextNode] = Field(default_factory=dict)
    edges: list[DependencyEdge] = Field(default_factory=list)
    transitive_closure_computed: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_node(self, node: ContextNode) -> None:
        """Insert or overwrite a node in the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        """Append an edge to the graph (duplicates are allowed during build)."""
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def get_ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor ``node_id``\\ s for *node_id* via BFS.

        Includes both direct and transitive ancestors from the current edge set.
        Call :meth:`compute_transitive_closure` first for the full ancestor set.
        """
        ancestors: set[str] = set()
        queue: list[str] = [node_id]
        while queue:
            nid = queue.pop()
            for edge in self.edges:
                if edge.target_id == nid and edge.source_id not in ancestors:
                    ancestors.add(edge.source_id)
                    queue.append(edge.source_id)
        return ancestors

    def get_descendants(self, node_id: str) -> set[str]:
        """Return all descendant ``node_id``\\ s for *node_id* via BFS."""
        descendants: set[str] = set()
        queue: list[str] = [node_id]
        while queue:
            nid = queue.pop()
            for edge in self.edges:
                if edge.source_id == nid and edge.target_id not in descendants:
                    descendants.add(edge.target_id)
                    queue.append(edge.target_id)
        return descendants

    def compute_transitive_closure(self) -> None:
        """Augment *self.edges* with all transitive dependency edges.

        Uses BFS from each node to find reachable nodes and adds a
        ``is_transitive=True`` edge for each pair (source, descendant) that
        does not yet have a direct edge.

        For graphs with more than 200 nodes, closure is limited to a 3-hop
        depth to preserve latency targets (see docs/planner.md §6.2).
        """
        if self.transitive_closure_computed:
            return

        node_ids = list(self.nodes.keys())
        max_hops = 3 if len(node_ids) > 200 else None  # None = unlimited
        existing_pairs = {(e.source_id, e.target_id) for e in self.edges}
        new_edges: list[DependencyEdge] = []

        for src_id in node_ids:
            visited: set[str] = set()
            queue: list[tuple[str, int]] = [(src_id, 0)]
            while queue:
                cur_id, hops = queue.pop(0)
                if max_hops is not None and hops > max_hops:
                    continue
                for edge in self.edges:
                    if edge.source_id == cur_id and edge.target_id not in visited:
                        tgt_id = edge.target_id
                        visited.add(tgt_id)
                        if tgt_id != src_id and (src_id, tgt_id) not in existing_pairs:
                            new_edges.append(
                                DependencyEdge(
                                    edge_id=f"{src_id}->{tgt_id}",
                                    source_id=src_id,
                                    target_id=tgt_id,
                                    edge_type=edge.edge_type,
                                    weight=edge.weight * 0.5,  # discount transitive
                                    is_transitive=True,
                                )
                            )
                            existing_pairs.add((src_id, tgt_id))
                        queue.append((tgt_id, hops + 1))

        self.edges.extend(new_edges)
        self.transitive_closure_computed = True


# ---------------------------------------------------------------------------
# CompactionResult
# ---------------------------------------------------------------------------


class CompactionResult(BaseModel):
    """Output of :meth:`SubmodularSolver.solve` for Zone T.

    Contains the selected node IDs, the reconstructed Zone T text, and
    diagnostic statistics for observability.
    """

    session_id: str
    turn_index: int = Field(
        ge=0, description="Turn index at which compaction was performed."
    )
    selected_node_ids: list[str] = Field(
        description="Ordered list of selected node IDs (turn_index ascending)."
    )
    pruned_node_ids: list[str] = Field(
        description="Node IDs excluded from selection."
    )
    token_budget: int = Field(description="Token budget B_T for Zone T.")
    tokens_used: int = Field(description="Actual tokens consumed by selected nodes.")
    objective_value: float = Field(
        description="Final f(S) value from submodular objective."
    )
    dangling_violations: int = Field(
        default=0,
        description=(
            "Dangling violations detected post-selection (must always be 0 by design)."
        ),
    )
    forced_inclusions: list[str] = Field(
        default_factory=list,
        description="Node IDs force-included to satisfy referential integrity.",
    )
    reconstructed_text: str = Field(
        description="Concatenated text of selected Zone T nodes in turn order."
    )
    latency_ms: float = Field(
        description="Wall-clock time for the compaction pass in milliseconds."
    )


# ---------------------------------------------------------------------------
# SemanticChunk & BoundaryEvaluationResult
# ---------------------------------------------------------------------------


class BoundaryEvaluationResult(BaseModel):
    """Evaluation metrics for a token position during streaming chunking."""

    token_index: int = Field(ge=0)
    cosine_shift: float = Field(
        ge=0.0,
        description="Directional semantic shift gradient ΔS = 1 - cos(A, B).",
    )
    token_entropy: float = Field(
        ge=0.0,
        description="Conditional token entropy H(T_i | T_{i-w...i-1}).",
    )
    boundary_score: float = Field(
        ge=0.0,
        description="Score = ΔS · H(T_i).",
    )
    threshold: float = Field(
        ge=0.0,
        description="Adaptive threshold τ_boundary.",
    )
    is_boundary: bool = Field(
        description="True if score > threshold and not suppressed by syntax protection.",
    )
    suppressed_by_syntax: bool = Field(
        default=False,
        description="True if score crossed threshold but split was delayed to protect unclosed code/JSON syntax.",
    )


class SemanticChunk(BaseModel):
    """A self-healing adaptive semantic chunk produced by entropy boundary detection."""

    chunk_id: str = Field(
        description="Unique identifier for this chunk (e.g. 'chunk:0:0-64')."
    )
    turn_index: int = Field(ge=0)
    start_token: int = Field(ge=0)
    end_token: int = Field(ge=0)
    content: str
    token_count: int = Field(ge=0)
    boundary_score: float = Field(default=0.0, ge=0.0)
    contains_code_block: bool = Field(default=False)
    has_unclosed_scope: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

