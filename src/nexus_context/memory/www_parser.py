"""
nexus_context.memory.www_parser
================================
Extracts ⟨Who, What, When, Where⟩ memory tuples from agent turns.

Extraction paths
----------------
* **Code turns** (Python, SQL): Tree-Sitter AST walk → mutation nodes
* **NL turns** (user/assistant): spaCy NER + SVO dependency extraction

Reference: docs/implementation_stage.md §3.1,
           docs/overview_and_research.md §4.3
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from nexus_context import MemoryExtractionError
from nexus_context.cache.schemas import ChatMessage
from nexus_context.guard.ast_graph import _detect_language, _estimate_tokens, _get_parsers
from nexus_context.memory.schemas import MemoryTuple, MutationType, WhatDelta, WhoActor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role → WhoActor mapping
# ---------------------------------------------------------------------------

_ROLE_TO_ACTOR: dict[str, WhoActor] = {
    "system": WhoActor.SYSTEM,
    "user": WhoActor.USER,
    "assistant": WhoActor.AGENT,
    "tool": WhoActor.TOOL,
}


def _infer_who(role: str) -> WhoActor:
    return _ROLE_TO_ACTOR.get(role, WhoActor.AGENT)


# ---------------------------------------------------------------------------
# Python mutation extractor
# ---------------------------------------------------------------------------


class _PythonMutationExtractor:
    """Walk a Tree-Sitter Python AST and collect mutation descriptors."""

    def __init__(self, turn_index: int, d_max: int) -> None:
        self._turn_index = turn_index
        self._d_max = max(d_max, 1)
        self._scope_stack: list[str] = ["module"]
        self._depth: int = 0
        self.mutations: list[tuple[WhatDelta, str, float]] = []
        # Each entry: (WhatDelta, scope_path, ast_depth_score)

    def visit(self, node: object) -> None:  # type: ignore[override]
        """Recursively visit a Tree-Sitter node."""
        node_type: str = getattr(node, "type", "")
        try:
            handler = getattr(self, f"visit_{node_type}", None)
            if handler is not None:
                handler(node)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                '{"event":"mutation_visit_error","node_type":"%s","reason":"%s"}',
                node_type,
                str(exc),
            )

        push = node_type in ("function_definition", "class_definition")
        if push:
            name_child = next(
                (c for c in getattr(node, "children", [])
                 if getattr(c, "type", "") == "identifier"),
                None,
            )
            scope_name = (
                getattr(name_child, "text", b"").decode("utf-8", errors="replace")
                if name_child else "anonymous"
            )
            self._scope_stack.append(scope_name)
            self._depth += 1

        for child in getattr(node, "children", []):
            self.visit(child)

        if push:
            self._scope_stack.pop()
            self._depth -= 1

    def visit_assignment(self, node: object) -> None:  # type: ignore[override]
        children = getattr(node, "children", [])
        if len(children) < 3:
            return
        lhs_text = getattr(children[0], "text", b"").decode("utf-8", errors="replace")
        rhs_text = getattr(children[2], "text", b"").decode("utf-8", errors="replace")
        if not lhs_text:
            return
        depth_score = (self._d_max - self._depth) / self._d_max
        self.mutations.append((
            WhatDelta(
                mutation_type=MutationType.VAR_ASSIGN,
                target_name=lhs_text,
                new_value_repr=rhs_text[:120],
            ),
            ".".join(self._scope_stack),
            max(0.0, min(1.0, depth_score)),
        ))

    def visit_function_definition(self, node: object) -> None:  # type: ignore[override]
        name_child = next(
            (c for c in getattr(node, "children", []) if getattr(c, "type", "") == "identifier"),
            None,
        )
        if name_child is None:
            return
        name = getattr(name_child, "text", b"").decode("utf-8", errors="replace")
        params_child = next(
            (c for c in getattr(node, "children", []) if getattr(c, "type", "") == "parameters"),
            None,
        )
        params_text = (
            getattr(params_child, "text", b"").decode("utf-8", errors="replace")
            if params_child else "()"
        )
        depth_score = (self._d_max - self._depth) / self._d_max
        self.mutations.append((
            WhatDelta(
                mutation_type=MutationType.FUNC_DEF,
                target_name=name,
                new_value_repr=f"def {name}{params_text}",
            ),
            ".".join(self._scope_stack),
            max(0.0, min(1.0, depth_score)),
        ))

    def visit_import_statement(self, node: object) -> None:  # type: ignore[override]
        text = getattr(node, "text", b"").decode("utf-8", errors="replace")
        names = re.findall(r"\b([A-Za-z_]\w*)\b", text)
        filtered = [n for n in names if n not in ("import", "from", "as")]
        if not filtered:
            return
        depth_score = (self._d_max - self._depth) / self._d_max
        for mod_name in filtered[:3]:  # cap at 3 imports per statement
            self.mutations.append((
                WhatDelta(
                    mutation_type=MutationType.IMPORT,
                    target_name=mod_name,
                    new_value_repr=text.strip()[:80],
                ),
                ".".join(self._scope_stack),
                max(0.0, min(1.0, depth_score)),
            ))


# ---------------------------------------------------------------------------
# SQL mutation extractor
# ---------------------------------------------------------------------------


class _SQLMutationExtractor:
    """Extract DDL/DML mutations from a Tree-Sitter SQL AST."""

    _CREATE_RE = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.+?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    _ALTER_RE = re.compile(r"ALTER\s+TABLE\s+(\w+)", re.IGNORECASE)
    _DROP_RE = re.compile(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)", re.IGNORECASE)
    _INSERT_RE = re.compile(r"INSERT\s+INTO\s+(\w+)", re.IGNORECASE)
    _UPDATE_RE = re.compile(r"UPDATE\s+(\w+)\s+SET", re.IGNORECASE)
    _DELETE_RE = re.compile(r"DELETE\s+FROM\s+(\w+)", re.IGNORECASE)

    def extract(self, sql_text: str) -> list[tuple[WhatDelta, str, float]]:
        mutations: list[tuple[WhatDelta, str, float]] = []

        for m in self._CREATE_RE.finditer(sql_text):
            table = m.group(1)
            cols = m.group(2)[:200]
            mutations.append((
                WhatDelta(
                    mutation_type=MutationType.SCHEMA_CHANGE,
                    target_name=table,
                    new_value_repr=f"CREATE({cols})",
                    is_destructive=False,
                ),
                "module",
                1.0,
            ))

        for m in self._ALTER_RE.finditer(sql_text):
            mutations.append((
                WhatDelta(
                    mutation_type=MutationType.SCHEMA_CHANGE,
                    target_name=m.group(1),
                    new_value_repr="ALTER",
                ),
                "module",
                1.0,
            ))

        for m in self._DROP_RE.finditer(sql_text):
            mutations.append((
                WhatDelta(
                    mutation_type=MutationType.SCHEMA_CHANGE,
                    target_name=m.group(1),
                    new_value_repr="DROP",
                    is_destructive=True,
                ),
                "module",
                1.0,
            ))

        for m in self._INSERT_RE.finditer(sql_text):
            mutations.append((
                WhatDelta(
                    mutation_type=MutationType.SCHEMA_CHANGE,
                    target_name=m.group(1),
                    new_value_repr="INSERT",
                ),
                "module",
                0.8,
            ))

        for m in self._UPDATE_RE.finditer(sql_text):
            mutations.append((
                WhatDelta(
                    mutation_type=MutationType.SCHEMA_CHANGE,
                    target_name=m.group(1),
                    new_value_repr="UPDATE",
                ),
                "module",
                0.8,
            ))

        return mutations


# ---------------------------------------------------------------------------
# WWWParser
# ---------------------------------------------------------------------------

_SQL_EXTRACTOR = _SQLMutationExtractor()


class WWWParser:
    """Extract WWW memory tuples from agent conversation turns.

    Parameters
    ----------
    languages:
        Languages to enable code-based extraction.
    session_id:
        Session identifier embedded in generated ``memory_id`` fields.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        session_id: str = "default",
    ) -> None:
        self._session_id = session_id
        self._languages = set(languages or ["python", "sql", "bash"])
        self._parsers = _get_parsers()
        self._d_max: int = 1  # updated as deeper scopes are observed
        self._spacy_nlp: object | None = self._load_spacy()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, turn: ChatMessage) -> list[MemoryTuple]:
        """Extract all WWW tuples from *turn*.

        Parameters
        ----------
        turn:
            A single chat message from Zone T (already graduated from Zone R).

        Returns
        -------
        list[MemoryTuple]
            One tuple per detected state mutation.  May be empty for
            turns that produce no detectable mutations.
        """
        lang = _detect_language(turn.content)
        if lang in self._languages and lang in ("python", "sql"):
            raw_mutations = self._extract_code(turn, lang)
        else:
            raw_mutations = self._extract_nl(turn)

        tuples: list[MemoryTuple] = []
        for seq, (what_delta, scope_path, depth_score) in enumerate(raw_mutations):
            tok_estimate = _estimate_tokens(
                json.dumps(
                    {
                        "w": turn.role[:4],
                        "d": what_delta.compact_repr(),
                        "t": turn.turn_index,
                        "@": scope_path[:4],
                    },
                    separators=(",", ":"),
                )
            )
            mem = MemoryTuple(
                memory_id=f"{self._session_id}:{turn.turn_index}:{seq}",
                session_id=self._session_id,
                who=_infer_who(turn.role),
                who_detail=f"{turn.role}@turn_{turn.turn_index}",
                what=what_delta,
                when=turn.turn_index,
                where=scope_path,
                token_count=tok_estimate,
                ast_depth_score=depth_score,
            )
            tuples.append(mem)

        if not tuples:
            # Always emit at least a generic NL_STATEMENT for any pruned turn
            tuples.append(self._fallback_tuple(turn))

        return tuples

    # ------------------------------------------------------------------
    # Private: code extraction
    # ------------------------------------------------------------------

    def _extract_code(
        self, turn: ChatMessage, language: str
    ) -> list[tuple[WhatDelta, str, float]]:
        if language == "sql":
            return _SQL_EXTRACTOR.extract(turn.content)

        # Python
        parser = self._parsers.get("python")
        if parser is None:
            return self._extract_python_stdlib(turn.content)

        try:
            tree = parser.parse(turn.content.encode("utf-8"))  # type: ignore[union-attr]
            extractor = _PythonMutationExtractor(turn.turn_index, self._d_max)
            extractor.visit(tree.root_node)
            if extractor._depth > self._d_max:
                self._d_max = extractor._depth
            return extractor.mutations
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '{"event":"python_extraction_failed","turn_index":%d,"reason":"%s"}',
                turn.turn_index,
                str(exc),
            )
            return self._extract_python_stdlib(turn.content)

    def _extract_python_stdlib(self, content: str) -> list[tuple[WhatDelta, str, float]]:
        """Fallback Python AST mutation extractor using standard library ast module."""
        import ast

        mutations: list[tuple[WhatDelta, str, float]] = []
        code = content
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
            return mutations

        class StdlibExtractor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    name = None
                    if isinstance(target, ast.Name):
                        name = target.id
                    elif isinstance(target, ast.Attribute):
                        name = target.attr
                    if name:
                        rhs_str = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                        mutations.append((
                            WhatDelta(
                                mutation_type=MutationType.VAR_ASSIGN,
                                target_name=name,
                                new_value_repr=rhs_str[:120],
                            ),
                            "module",
                            1.0,
                        ))
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                mutations.append((
                    WhatDelta(
                        mutation_type=MutationType.FUNC_DEF,
                        target_name=node.name,
                        new_value_repr=f"def {node.name}(...)",
                    ),
                    "module",
                    1.0,
                ))
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    mutations.append((
                        WhatDelta(
                            mutation_type=MutationType.IMPORT,
                            target_name=alias.name,
                            new_value_repr=f"import {alias.name}",
                        ),
                        "module",
                        1.0,
                    ))

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                mod = node.module or ""
                for alias in node.names:
                    mutations.append((
                        WhatDelta(
                            mutation_type=MutationType.IMPORT,
                            target_name=alias.name,
                            new_value_repr=f"from {mod} import {alias.name}",
                        ),
                        "module",
                        1.0,
                    ))

        StdlibExtractor().visit(parsed)
        return mutations

    # ------------------------------------------------------------------
    # Private: NL extraction
    # ------------------------------------------------------------------

    def _extract_nl(
        self, turn: ChatMessage
    ) -> list[tuple[WhatDelta, str, float]]:
        mutations: list[tuple[WhatDelta, str, float]] = []

        if self._spacy_nlp is None:
            return mutations

        try:
            doc = self._spacy_nlp(turn.content[:1000])

            # Named entity assertions
            for ent in getattr(doc, "ents", []):
                mutations.append((
                    WhatDelta(
                        mutation_type=MutationType.NL_STATEMENT,
                        target_name=ent.text,
                        new_value_repr=ent.label_,
                    ),
                    "session",
                    0.5,
                ))

            # SVO triples via dependency parse
            for token in doc:
                if getattr(token, "dep_", "") == "ROOT":
                    subject = next(
                        (c for c in token.children
                         if c.dep_ in ("nsubj", "nsubjpass")),
                        None,
                    )
                    obj = next(
                        (c for c in token.children
                         if c.dep_ in ("dobj", "attr", "prep")),
                        None,
                    )
                    if subject and obj:
                        mutations.append((
                            WhatDelta(
                                mutation_type=MutationType.NL_STATEMENT,
                                target_name=subject.text,
                                new_value_repr=f"{token.text} {obj.text}",
                            ),
                            "session",
                            0.3,
                        ))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                '{"event":"nl_extraction_failed","turn_index":%d,"reason":"%s"}',
                turn.turn_index,
                str(exc),
            )

        return mutations

    def _fallback_tuple(self, turn: ChatMessage) -> MemoryTuple:
        summary = turn.content[:80].replace("\n", " ")
        return MemoryTuple(
            memory_id=f"{self._session_id}:{turn.turn_index}:fallback",
            session_id=self._session_id,
            who=_infer_who(turn.role),
            who_detail=f"{turn.role}@turn_{turn.turn_index}",
            what=WhatDelta(
                mutation_type=MutationType.NL_STATEMENT,
                target_name=f"turn_{turn.turn_index}",
                new_value_repr=summary,
            ),
            when=turn.turn_index,
            where="session",
            token_count=_estimate_tokens(summary),
            ast_depth_score=0.0,
        )

    # ------------------------------------------------------------------
    # Private: spaCy loading
    # ------------------------------------------------------------------

    def _load_spacy(self) -> object | None:
        try:
            import spacy  # type: ignore[import-untyped]

            for model in ("en_core_web_trf", "en_core_web_sm"):
                try:
                    return spacy.load(model)
                except OSError:
                    continue
            logger.warning('{"event":"spacy_model_not_found","www_parser":"NL extraction limited"}')
            return None
        except ImportError:
            return None
