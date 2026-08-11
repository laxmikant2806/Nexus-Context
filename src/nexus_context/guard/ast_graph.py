"""
nexus_context.guard.ast_graph
==============================
Builds the Context Dependency Graph G = (V, E) from a session's Zone-T
candidate turns.

Supported languages (Tree-Sitter grammars):
    python, sql, bash, javascript

Natural-language coreference resolution:
    spaCy en_core_web_trf + coreferee (E_nl edges)

Design decisions: docs/decisions.md ADR-001
Reference:        docs/implementation_stage.md §3.2
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

from nexus_context import GraphBuildError
from nexus_context.cache.schemas import ChatMessage
from nexus_context.guard.schemas import (
    ContextGraph,
    ContextNode,
    DependencyEdge,
    EdgeType,
    NodeType,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection heuristics
# ---------------------------------------------------------------------------

_PYTHON_KEYWORDS = re.compile(
    r"\b(def |import |class |elif |lambda |yield |async def )", re.MULTILINE
)
_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|ALTER TABLE|DROP TABLE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_BASH_KEYWORDS = re.compile(
    r"(#!/|^\$\s|^export |^echo |^if \[|^for \w+ in )", re.MULTILINE
)
_CODE_FENCE = re.compile(r"```(\w+)", re.MULTILINE)


def _detect_language(content: str) -> str | None:
    """Heuristically detect the programming language of *content*.

    Returns ``None`` if the content is likely natural language.
    """
    # 1. Explicit markdown code-fence markers
    fence_match = _CODE_FENCE.search(content)
    if fence_match:
        lang = fence_match.group(1).lower()
        if lang in ("python", "py"):
            return "python"
        if lang in ("sql",):
            return "sql"
        if lang in ("bash", "sh", "shell"):
            return "bash"
        if lang in ("js", "javascript", "ts", "typescript"):
            return "javascript"

    # 2. Keyword-based heuristics (priority order matters)
    if _PYTHON_KEYWORDS.search(content):
        return "python"
    if _SQL_KEYWORDS.search(content):
        return "sql"
    if _BASH_KEYWORDS.search(content):
        return "bash"

    return None  # treat as natural language


# ---------------------------------------------------------------------------
# Tree-Sitter parser pool (lazy initialisation)
# ---------------------------------------------------------------------------

_PARSERS: dict[str, object] | None = None


def _get_parsers() -> dict[str, object]:
    """Lazily load Tree-Sitter parsers; return cached instances on repeat calls."""
    global _PARSERS  # noqa: PLW0603
    if _PARSERS is not None:
        return _PARSERS

    _PARSERS = {}
    try:
        from tree_sitter import Language, Parser  # type: ignore[import-untyped]

        grammars = {
            "python": "tree_sitter_python",
            "sql": "tree_sitter_sql",
            "bash": "tree_sitter_bash",
            "javascript": "tree_sitter_javascript",
        }
        for lang, module_name in grammars.items():
            try:
                mod = __import__(module_name)
                _PARSERS[lang] = Parser(Language(mod.language()))
                logger.debug(
                    '{"event":"parser_loaded","language":"%s"}', lang
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    '{"event":"parser_load_failed","language":"%s","reason":"%s"}',
                    lang,
                    str(exc),
                )
    except ImportError:
        logger.warning(
            '{"event":"tree_sitter_unavailable",'
            '"note":"code turns will be treated as NL_SENTENCE nodes"}'
        )

    return _PARSERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_id(text: str, length: int = 8) -> str:
    """Return a short hex hash of *text* for use in node IDs."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:length]  # noqa: S324


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Python AST visitor
# ---------------------------------------------------------------------------


class _PythonVisitor:
    """Walk a Tree-Sitter Python parse tree and extract dependency edges."""

    def __init__(self, turn_index: int, session_id: str) -> None:
        self._turn_index = turn_index
        self._session_id = session_id
        self._scope_stack: list[str] = ["module"]
        self._binding_store: dict[str, ContextNode] = {}  # name → ContextNode
        self.nodes: list[ContextNode] = []
        self.edges: list[DependencyEdge] = []

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def visit(self, tree_node: object) -> None:  # type: ignore[override]
        """Recursively visit a Tree-Sitter ``Node``."""
        node_type: str = getattr(tree_node, "type", "")
        try:
            handler = getattr(self, f"visit_{node_type}", None)
            if handler is not None:
                handler(tree_node)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                '{"event":"ast_visit_error","node_type":"%s","reason":"%s"}',
                node_type,
                str(exc),
            )

        for child in getattr(tree_node, "children", []):
            self.visit(child)

    # ------------------------------------------------------------------ #
    # Node visitors
    # ------------------------------------------------------------------ #

    def visit_assignment(self, node: object) -> None:  # type: ignore[override]
        children = getattr(node, "children", [])
        if len(children) < 3:
            return
        lhs = children[0]
        rhs = children[2]
        name = getattr(lhs, "text", b"").decode("utf-8", errors="replace")
        rhs_text = getattr(rhs, "text", b"").decode("utf-8", errors="replace")
        if not name:
            return

        ctx_node = self._make_node(
            NodeType.AST_ASSIGNMENT,
            name,
            f"{name} = {rhs_text}",
            metadata={"name": name, "rhs": rhs_text},
        )
        # Supersede previous binding
        self._binding_store[name] = ctx_node

    def visit_function_definition(self, node: object) -> None:  # type: ignore[override]
        name_child = next(
            (c for c in getattr(node, "children", []) if getattr(c, "type", "") == "identifier"),
            None,
        )
        if name_child is None:
            return
        name = getattr(name_child, "text", b"").decode("utf-8", errors="replace")
        full_text = getattr(node, "text", b"").decode("utf-8", errors="replace")[:200]
        ctx_node = self._make_node(NodeType.AST_FUNCDEF, name, full_text, metadata={"name": name})
        self._binding_store[name] = ctx_node
        self._scope_stack.append(name)

    def visit_class_definition(self, node: object) -> None:  # type: ignore[override]
        name_child = next(
            (c for c in getattr(node, "children", []) if getattr(c, "type", "") == "identifier"),
            None,
        )
        if name_child is None:
            return
        name = getattr(name_child, "text", b"").decode("utf-8", errors="replace")
        full_text = getattr(node, "text", b"").decode("utf-8", errors="replace")[:200]
        ctx_node = self._make_node(NodeType.AST_CLASSDEF, name, full_text, metadata={"name": name})
        self._binding_store[name] = ctx_node
        self._scope_stack.append(name)

    def visit_import_statement(self, node: object) -> None:  # type: ignore[override]
        full_text = getattr(node, "text", b"").decode("utf-8", errors="replace")
        # Extract module names from "import foo, bar" or "from foo import bar"
        names = re.findall(r"\b([A-Za-z_]\w*)\b", full_text)
        for name in names:
            if name in ("import", "from", "as"):
                continue
            ctx_node = self._make_node(
                NodeType.AST_IMPORT, name, full_text, metadata={"module": name}
            )
            self._binding_store.setdefault(name, ctx_node)

    def visit_identifier(self, node: object) -> None:  # type: ignore[override]
        # Only process identifiers in a load context (not definitions).
        parent = getattr(node, "parent", None)
        if parent is None:
            return
        parent_type: str = getattr(parent, "type", "")
        if parent_type in ("assignment", "function_definition", "class_definition",
                            "import_statement", "import_from_statement",
                            "parameters", "typed_parameter"):
            return  # this is a definition site, not a use site

        name = getattr(node, "text", b"").decode("utf-8", errors="replace")
        if name not in self._binding_store:
            return

        source_node = self._binding_store[name]
        target_node = self._make_node(
            NodeType.AST_CALL,
            f"use_{name}_{_short_id(name)}",
            name,
            metadata={"ref": name},
        )
        edge_type = (
            EdgeType.FUNC_DEF_TO_CALL
            if source_node.node_type == NodeType.AST_FUNCDEF
            else EdgeType.VAR_DEF_TO_REF
        )
        self._add_edge(source_node.node_id, target_node.node_id, edge_type)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _current_scope(self) -> str:
        return ".".join(self._scope_stack)

    def _make_node(
        self,
        node_type: NodeType,
        name: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> ContextNode:
        node_id = f"{self._turn_index}:{node_type.value}:{name}"
        depth = len(self._scope_stack) - 1
        ctx_node = ContextNode(
            node_id=node_id,
            node_type=node_type,
            turn_index=self._turn_index,
            content=content,
            token_count=_estimate_tokens(content),
            language="python",
            ast_depth=depth,
            scope_path=self._current_scope(),
            metadata=metadata or {},
        )
        self.nodes.append(ctx_node)
        return ctx_node

    def _add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        self.edges.append(
            DependencyEdge(
                edge_id=f"{source_id}->{target_id}",
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=1.0,
            )
        )


# ---------------------------------------------------------------------------
# SQL visitor (simplified)
# ---------------------------------------------------------------------------


class _SQLVisitor:
    """Walk a Tree-Sitter SQL parse tree and extract schema edges."""

    def __init__(self, turn_index: int) -> None:
        self._turn_index = turn_index
        self._schema_store: dict[str, ContextNode] = {}
        self.nodes: list[ContextNode] = []
        self.edges: list[DependencyEdge] = []

    def visit(self, tree_node: object) -> None:  # type: ignore[override]
        node_type: str = getattr(tree_node, "type", "")
        try:
            handler = getattr(self, f"visit_{node_type}", None)
            if handler is not None:
                handler(tree_node)
        except Exception:  # noqa: BLE001
            pass
        for child in getattr(tree_node, "children", []):
            self.visit(child)

    def visit_create_table_statement(self, node: object) -> None:  # type: ignore[override]
        text = getattr(node, "text", b"").decode("utf-8", errors="replace")
        # Extract table name via simple regex (Tree-Sitter SQL grammar varies)
        m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", text, re.IGNORECASE)
        if not m:
            return
        table_name = m.group(1)
        ctx_node = ContextNode(
            node_id=f"{self._turn_index}:{NodeType.SCHEMA_FIELD.value}:{table_name}",
            node_type=NodeType.SCHEMA_FIELD,
            turn_index=self._turn_index,
            content=text[:300],
            token_count=_estimate_tokens(text),
            language="sql",
            metadata={"table": table_name, "op": "CREATE"},
        )
        self.nodes.append(ctx_node)
        self._schema_store[table_name] = ctx_node

    def visit_select_statement(self, node: object) -> None:  # type: ignore[override]
        text = getattr(node, "text", b"").decode("utf-8", errors="replace")
        # Look for table references in FROM clauses
        tables = re.findall(r"\bFROM\s+(\w+)", text, re.IGNORECASE)
        for tname in tables:
            if tname in self._schema_store:
                src_node = self._schema_store[tname]
                tgt_id = f"{self._turn_index}:{NodeType.AST_CALL.value}:select_{tname}"
                self.edges.append(
                    DependencyEdge(
                        edge_id=f"{src_node.node_id}->{tgt_id}",
                        source_id=src_node.node_id,
                        target_id=tgt_id,
                        edge_type=EdgeType.SCHEMA_FIELD_DEP,
                        weight=0.9,
                    )
                )


# ---------------------------------------------------------------------------
# ContextGraphBuilder
# ---------------------------------------------------------------------------


class ContextGraphBuilder:
    """Build the Context Dependency Graph from Zone-T candidate turns.

    Usage
    -----
    ::

        builder = ContextGraphBuilder()
        graph   = builder.build(zone_t_candidates)
        graph.compute_transitive_closure()

    The resulting graph is passed to :class:`SubmodularSolver`.
    """

    def __init__(self, session_id: str = "default") -> None:
        self._session_id = session_id
        self._parsers = _get_parsers()
        self._spacy_nlp: object | None = self._load_spacy()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, turns: list[ChatMessage]) -> ContextGraph:
        """Construct G = (V, E) from *turns*.

        Each turn is dispatched to the appropriate language-specific visitor
        (Python, SQL) or the NL pipeline (spaCy + coreference).  All nodes
        and edges are merged into a single :class:`ContextGraph`.

        Parameters
        ----------
        turns:
            Zone-T candidate messages, ordered by ``turn_index``.

        Returns
        -------
        ContextGraph
            Ready for :meth:`ContextGraph.compute_transitive_closure`.
        """
        import time

        start = time.perf_counter()
        graph = ContextGraph(session_id=self._session_id)

        for msg in turns:
            try:
                lang = _detect_language(msg.content)
                if lang in ("python", "sql", "bash", "javascript"):
                    nodes, edges = self._parse_code(msg, lang)
                else:
                    nodes, edges = self._parse_nl(msg)

                for n in nodes:
                    graph.add_node(n)
                for e in edges:
                    graph.add_edge(e)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    '{"event":"turn_parse_failed","turn_index":%d,"reason":"%s",'
                    '"fallback":"NL_SENTENCE"}',
                    msg.turn_index,
                    str(exc),
                )
                # Fallback: add a single NL_SENTENCE node, no edges
                fallback_node = ContextNode(
                    node_id=f"{msg.turn_index}:{NodeType.NL_SENTENCE.value}:{_short_id(msg.content)}",
                    node_type=NodeType.NL_SENTENCE,
                    turn_index=msg.turn_index,
                    content=msg.content[:500],
                    token_count=msg.token_count,
                )
                graph.add_node(fallback_node)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            '{"event":"graph_built","n_nodes":%d,"n_edges":%d,"latency_ms":%.2f}',
            len(graph.nodes),
            len(graph.edges),
            elapsed_ms,
        )
        return graph

    # ------------------------------------------------------------------
    # Private: code parsing
    # ------------------------------------------------------------------

    def _parse_code(
        self, msg: ChatMessage, language: str
    ) -> tuple[list[ContextNode], list[DependencyEdge]]:
        """Parse *msg.content* with the appropriate Tree-Sitter or stdlib fallback visitor."""
        parser = self._parsers.get(language)
        if parser is None:
            if language == "python":
                return self._parse_python_stdlib(msg)
            raise GraphBuildError(
                f"No Tree-Sitter parser available for language '{language}'. "
                "Install the corresponding tree-sitter-{lang} package."
            )

        try:
            code_bytes = msg.content.encode("utf-8")
            tree = getattr(parser, "parse")(code_bytes)  # type: ignore[union-attr]
            root = tree.root_node

            if language == "python":
                visitor = _PythonVisitor(msg.turn_index, self._session_id)
            elif language == "sql":
                visitor = _SQLVisitor(msg.turn_index)  # type: ignore[assignment]
            else:
                return self._parse_nl(msg)

            visitor.visit(root)
            return visitor.nodes, visitor.edges
        except Exception:
            if language == "python":
                return self._parse_python_stdlib(msg)
            raise

    def _parse_python_stdlib(
        self, msg: ChatMessage
    ) -> tuple[list[ContextNode], list[DependencyEdge]]:
        """Fallback Python code parser using standard library `ast` module."""
        import ast

        nodes: list[ContextNode] = []
        edges: list[DependencyEdge] = []
        binding_store: dict[str, ContextNode] = {}

        code = msg.content
        if "```" in code:
            lines = code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)

        try:
            parsed = ast.parse(code)
        except SyntaxError:
            return nodes, edges

        turn_idx = msg.turn_index

        class StdlibVisitor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    name = None
                    if isinstance(target, ast.Name):
                        name = target.id
                    elif isinstance(target, ast.Attribute):
                        name = target.attr
                    if name:
                        rhs_str = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                        ctx_node = ContextNode(
                            node_id=f"{turn_idx}:{NodeType.AST_ASSIGNMENT.value}:{name}",
                            node_type=NodeType.AST_ASSIGNMENT,
                            turn_index=turn_idx,
                            content=f"{name} = {rhs_str}",
                            token_count=_estimate_tokens(f"{name} = {rhs_str}"),
                            language="python",
                            ast_depth=0,
                            scope_path="module",
                            metadata={"name": name, "rhs": rhs_str},
                        )
                        nodes.append(ctx_node)
                        binding_store[name] = ctx_node
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                name = node.name
                ctx_node = ContextNode(
                    node_id=f"{turn_idx}:{NodeType.AST_FUNCDEF.value}:{name}",
                    node_type=NodeType.AST_FUNCDEF,
                    turn_index=turn_idx,
                    content=f"def {name}(...)",
                    token_count=_estimate_tokens(f"def {name}(...)"),
                    language="python",
                    ast_depth=0,
                    scope_path="module",
                    metadata={"name": name},
                )
                nodes.append(ctx_node)
                binding_store[name] = ctx_node
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load) and node.id in binding_store:
                    src_node = binding_store[node.id]
                    tgt_id = f"{turn_idx}:{NodeType.AST_CALL.value}:use_{node.id}_{_short_id(node.id)}"
                    tgt_node = ContextNode(
                        node_id=tgt_id,
                        node_type=NodeType.AST_CALL,
                        turn_index=turn_idx,
                        content=node.id,
                        token_count=_estimate_tokens(node.id),
                        language="python",
                        ast_depth=0,
                        scope_path="module",
                        metadata={"ref": node.id},
                    )
                    nodes.append(tgt_node)
                    edge_type = (
                        EdgeType.FUNC_DEF_TO_CALL
                        if src_node.node_type == NodeType.AST_FUNCDEF
                        else EdgeType.VAR_DEF_TO_REF
                    )
                    edges.append(
                        DependencyEdge(
                            edge_id=f"{src_node.node_id}->{tgt_id}",
                            source_id=src_node.node_id,
                            target_id=tgt_id,
                            edge_type=edge_type,
                            weight=1.0,
                        )
                    )

        StdlibVisitor().visit(parsed)
        return nodes, edges

    # ------------------------------------------------------------------
    # Private: NL parsing
    # ------------------------------------------------------------------

    def _parse_nl(
        self, msg: ChatMessage
    ) -> tuple[list[ContextNode], list[DependencyEdge]]:
        """Create NL_SENTENCE nodes and E_nl coreference edges via spaCy."""
        nodes: list[ContextNode] = []
        edges: list[DependencyEdge] = []

        # One node per sentence
        if self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(msg.content)
                sents = list(getattr(doc, "sents", []))
                for i, sent in enumerate(sents):
                    text = sent.text.strip()
                    if not text:
                        continue
                    node_id = (
                        f"{msg.turn_index}:{NodeType.NL_SENTENCE.value}:s{i}"
                    )
                    nodes.append(
                        ContextNode(
                            node_id=node_id,
                            node_type=NodeType.NL_SENTENCE,
                            turn_index=msg.turn_index,
                            content=text,
                            token_count=max(1, len(text.split())),
                        )
                    )

                # Coreference edges (coreferee)
                coref_chains = getattr(getattr(doc, "_", None), "coref_chains", None)
                if coref_chains:
                    for chain in coref_chains:
                        mentions = list(chain)
                        for j in range(1, len(mentions)):
                            antecedent_sent = mentions[j - 1].sent.start
                            anaphor_sent = mentions[j].sent.start
                            if antecedent_sent != anaphor_sent:
                                src_id = f"{msg.turn_index}:{NodeType.NL_SENTENCE.value}:s{antecedent_sent}"
                                tgt_id = f"{msg.turn_index}:{NodeType.NL_SENTENCE.value}:s{anaphor_sent}"
                                edges.append(
                                    DependencyEdge(
                                        edge_id=f"{src_id}->{tgt_id}",
                                        source_id=src_id,
                                        target_id=tgt_id,
                                        edge_type=EdgeType.COREF_ANTECEDENT,
                                        weight=0.5,
                                    )
                                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    '{"event":"spacy_parse_failed","turn_index":%d,"reason":"%s"}',
                    msg.turn_index,
                    str(exc),
                )

        # Fallback: single NL_SENTENCE node if spaCy produced nothing
        if not nodes:
            node_id = (
                f"{msg.turn_index}:{NodeType.NL_SENTENCE.value}:{_short_id(msg.content)}"
            )
            nodes.append(
                ContextNode(
                    node_id=node_id,
                    node_type=NodeType.NL_SENTENCE,
                    turn_index=msg.turn_index,
                    content=msg.content[:500],
                    token_count=msg.token_count,
                )
            )

        return nodes, edges

    # ------------------------------------------------------------------
    # Private: spaCy initialisation
    # ------------------------------------------------------------------

    def _load_spacy(self) -> object | None:
        try:
            import spacy  # type: ignore[import-untyped]

            try:
                nlp = spacy.load("en_core_web_trf")
            except OSError:
                # Fall back to smaller model
                try:
                    nlp = spacy.load("en_core_web_sm")
                    logger.warning(
                        '{"event":"spacy_fallback","model":"en_core_web_sm",'
                        '"note":"coreference accuracy reduced"}'
                    )
                except OSError:
                    logger.warning(
                        '{"event":"spacy_unavailable",'
                        '"note":"NL coreference disabled; E_nl edges will be empty"}'
                    )
                    return None

            # Try loading coreferee
            try:
                nlp.add_pipe("coreferee")
            except Exception:  # noqa: BLE001
                logger.debug('{"event":"coreferee_unavailable","note":"E_nl limited to sentences"}')

            return nlp
        except ImportError:
            logger.warning(
                '{"event":"spacy_import_failed",'
                '"note":"install spacy and en_core_web_trf for NL analysis"}'
            )
            return None
