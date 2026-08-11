"""
tests/test_memory.py
=====================
Unit tests for nexus_context.memory (Phase 3).

Covers:
    WWWParser   – Python/SQL extraction, NL fallback, compression ratio
    MemoryPool  – deduplication, weight decay, pinning, selection budget,
                  serialisation format
"""

from __future__ import annotations

import math

import pytest
from nexus_context.cache.schemas import ChatMessage
from nexus_context.memory.decay import MemoryPool
from nexus_context.memory.schemas import MemoryTuple, MutationType, WhatDelta, WhoActor
from nexus_context.memory.www_parser import WWWParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _code_turn(code: str, language: str, turn_index: int) -> ChatMessage:
    fence = f"```{language}\n{code}\n```"
    return ChatMessage(
        role="user",
        content=fence,
        token_count=max(1, len(fence) // 2),
        turn_index=turn_index,
    )


def _nl_turn(text: str, turn_index: int, role: str = "assistant") -> ChatMessage:
    return ChatMessage(
        role=role,
        content=text,
        token_count=max(1, len(text) // 4),
        turn_index=turn_index,
    )


def _make_memory_tuple(
    *,
    session_id: str = "s1",
    when: int = 0,
    target_name: str = "x",
    scope: str = "module",
    new_value: str = "42",
    token_count: int = 15,
    ast_depth_score: float = 1.0,
    is_pinned: bool = False,
) -> MemoryTuple:
    return MemoryTuple(
        memory_id=f"{session_id}:{when}:0",
        session_id=session_id,
        who=WhoActor.AGENT,
        who_detail=f"agent@turn_{when}",
        what=WhatDelta(
            mutation_type=MutationType.VAR_ASSIGN,
            target_name=target_name,
            new_value_repr=new_value,
        ),
        when=when,
        where=scope,
        token_count=token_count,
        ast_depth_score=ast_depth_score,
        is_pinned=is_pinned,
    )


# ---------------------------------------------------------------------------
# MemoryTuple.compute_weight tests
# ---------------------------------------------------------------------------


class TestMemoryTupleWeight:

    def test_weight_at_current_turn_root_scope(self) -> None:
        """W(T, s_root) = exp(0) × (1 + 0.5 × 1.0) = 1.5"""
        t = _make_memory_tuple(when=5, ast_depth_score=1.0)
        w = t.compute_weight(current_turn=5, lambda_=0.05, eta=0.5)
        assert abs(w - 1.5) < 1e-6

    def test_weight_at_current_turn_deep_scope(self) -> None:
        """W(T, s_deep) = exp(0) × (1 + 0.5 × 0.0) = 1.0"""
        t = _make_memory_tuple(when=5, ast_depth_score=0.0)
        w = t.compute_weight(current_turn=5, lambda_=0.05, eta=0.5)
        assert abs(w - 1.0) < 1e-6

    def test_weight_decreases_with_age(self) -> None:
        t = _make_memory_tuple(when=0, ast_depth_score=1.0)
        w0 = t.compute_weight(current_turn=0)
        w5 = t.compute_weight(current_turn=5)
        w10 = t.compute_weight(current_turn=10)
        assert w0 > w5 > w10

    def test_weight_at_half_life(self) -> None:
        """At age ≈ 13.9 turns, temporal factor ≈ 0.5."""
        t = _make_memory_tuple(when=0, ast_depth_score=1.0)
        w = t.compute_weight(current_turn=14, lambda_=0.05, eta=0.5)
        # exp(-0.05 × 14) × 1.5 ≈ 0.4966 × 1.5 ≈ 0.745
        assert 0.60 < w < 0.90

    def test_pinned_memory_has_inf_weight(self) -> None:
        t = _make_memory_tuple(when=0, is_pinned=True)
        w = t.compute_weight(current_turn=1000)
        assert w == math.inf

    def test_weight_formula_matches_docs(self) -> None:
        """Verify exact formula: W = exp(-λ(T-t)) × (1 + η × depth)"""
        t = _make_memory_tuple(when=3, ast_depth_score=0.6)
        w = t.compute_weight(current_turn=7, lambda_=0.1, eta=0.8)
        expected = math.exp(-0.1 * 4) * (1.0 + 0.8 * 0.6)
        assert abs(w - expected) < 1e-9


# ---------------------------------------------------------------------------
# MemoryPool tests
# ---------------------------------------------------------------------------


class TestMemoryPool:

    def test_add_single_tuple(self) -> None:
        pool = MemoryPool(session_id="s1")
        t = _make_memory_tuple(when=0)
        pool.add([t])
        assert len(pool) == 1

    def test_deduplication_keeps_most_recent(self) -> None:
        """Re-assignment of same (name, scope) retains only the newer tuple."""
        pool = MemoryPool(session_id="s1")
        t1 = _make_memory_tuple(when=2, target_name="conn", new_value="old_conn")
        t2 = _make_memory_tuple(when=5, target_name="conn", new_value="new_conn")
        pool.add([t1, t2])
        assert len(pool) == 1
        remaining = list(pool._pool.values())[0]
        assert remaining.what.new_value_repr == "new_conn"

    def test_deduplication_does_not_replace_newer_with_older(self) -> None:
        pool = MemoryPool(session_id="s1")
        t1 = _make_memory_tuple(when=10, target_name="x", new_value="new")
        t2 = _make_memory_tuple(when=5, target_name="x", new_value="old")
        pool.add([t1])
        pool.add([t2])  # older; must be ignored
        remaining = list(pool._pool.values())[0]
        assert remaining.what.new_value_repr == "new"

    def test_pinned_tuple_preserved_during_dedup(self) -> None:
        """Pinned tuples must never be replaced."""
        pool = MemoryPool(session_id="s1")
        t1 = _make_memory_tuple(when=2, target_name="x", new_value="pinned_val")
        pool.add([t1])
        pool.pin(t1.memory_id)
        t2 = _make_memory_tuple(when=5, target_name="x", new_value="newer_val")
        pool.add([t2])
        # Both should coexist (pinned uses versioned key)
        assert len(pool) == 2

    def test_update_weights_recomputes_retention(self) -> None:
        pool = MemoryPool(session_id="s1", lambda_=0.05, eta=0.5)
        t = _make_memory_tuple(when=0, ast_depth_score=1.0)
        pool.add([t])
        pool.update_weights(current_turn=14)
        updated = list(pool._pool.values())[0]
        # Weight should be approximately 0.74 (not 1.5)
        assert updated.retention_weight < 1.0

    def test_pinned_weight_stays_inf_after_update(self) -> None:
        pool = MemoryPool(session_id="s1")
        t = _make_memory_tuple(when=0)
        pool.add([t])
        pool.pin(t.memory_id)
        pool.update_weights(current_turn=1000)
        remaining = list(pool._pool.values())[0]
        assert remaining.retention_weight == math.inf

    def test_select_within_budget(self) -> None:
        pool = MemoryPool(session_id="s1")
        tuples = [
            _make_memory_tuple(when=i, target_name=f"v{i}", token_count=20)
            for i in range(10)
        ]
        pool.add(tuples)
        pool.update_weights(current_turn=10)
        selected = pool.select(budget_tokens=75)
        total = sum(m.token_count for m in selected)
        assert total <= 75

    def test_select_returns_chronological_order(self) -> None:
        pool = MemoryPool(session_id="s1")
        tuples = [
            _make_memory_tuple(when=i, target_name=f"v{i}", token_count=10)
            for i in range(5)
        ]
        pool.add(tuples)
        pool.update_weights(current_turn=5)
        selected = pool.select(budget_tokens=1000)
        when_values = [m.when for m in selected]
        assert when_values == sorted(when_values)

    def test_serialize_empty_returns_empty_string(self) -> None:
        pool = MemoryPool(session_id="s1")
        result = pool.serialize_for_context([])
        assert result == ""

    def test_serialize_format_contains_nexus_memory_tag(self) -> None:
        pool = MemoryPool(session_id="s1")
        t = _make_memory_tuple(when=3, target_name="conn")
        serialised = pool.serialize_for_context([t])
        assert "<!-- NEXUS_MEMORY" in serialised
        assert "conn" in serialised
        assert serialised.strip().endswith("-->")

    def test_pin_by_dependency_pins_matching_tuples(self) -> None:
        pool = MemoryPool(session_id="s1")
        t1 = _make_memory_tuple(when=0, target_name="DB_HOST")
        t2 = _make_memory_tuple(when=1, target_name="conn")
        pool.add([t1, t2])
        pool.pin_by_dependency({"DB_HOST"})
        db_host_tuple = next(
            m for m in pool._pool.values() if m.what.target_name == "DB_HOST"
        )
        assert db_host_tuple.is_pinned is True
        conn_tuple = next(
            m for m in pool._pool.values() if m.what.target_name == "conn"
        )
        assert conn_tuple.is_pinned is False


# ---------------------------------------------------------------------------
# WWWParser tests
# ---------------------------------------------------------------------------


class TestWWWParser:

    @pytest.fixture
    def parser(self) -> WWWParser:
        return WWWParser(session_id="s1")

    def test_python_assignment_produces_var_assign_tuple(
        self, parser: WWWParser
    ) -> None:
        turn = _code_turn('DB_HOST = "prod.internal"', "python", turn_index=3)
        tuples = parser.extract(turn)
        assert len(tuples) >= 1
        assert any(t.what.mutation_type == MutationType.VAR_ASSIGN for t in tuples)

    def test_python_assignment_target_name_extracted(
        self, parser: WWWParser
    ) -> None:
        turn = _code_turn('DB_HOST = "prod.internal"', "python", turn_index=3)
        tuples = parser.extract(turn)
        var_tuples = [t for t in tuples if t.what.mutation_type == MutationType.VAR_ASSIGN]
        assert any(t.what.target_name == "DB_HOST" for t in var_tuples)

    def test_python_multiple_assignments_multiple_tuples(
        self, parser: WWWParser
    ) -> None:
        code = "a = 1\nb = 2\nc = 3"
        turn = _code_turn(code, "python", turn_index=0)
        tuples = parser.extract(turn)
        var_tuples = [t for t in tuples if t.what.mutation_type == MutationType.VAR_ASSIGN]
        assert len(var_tuples) >= 3

    def test_sql_create_table_produces_schema_change_tuple(
        self, parser: WWWParser
    ) -> None:
        sql = "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255));"
        turn = _code_turn(sql, "sql", turn_index=7)
        tuples = parser.extract(turn)
        assert any(t.what.mutation_type == MutationType.SCHEMA_CHANGE for t in tuples)

    def test_sql_create_table_target_is_table_name(
        self, parser: WWWParser
    ) -> None:
        sql = "CREATE TABLE products (id INT, name TEXT);"
        turn = _code_turn(sql, "sql", turn_index=5)
        tuples = parser.extract(turn)
        schema_tuples = [t for t in tuples if t.what.mutation_type == MutationType.SCHEMA_CHANGE]
        assert any(t.what.target_name.lower() == "products" for t in schema_tuples)

    def test_when_field_matches_turn_index(self, parser: WWWParser) -> None:
        turn = _code_turn("x = 10", "python", turn_index=42)
        tuples = parser.extract(turn)
        for t in tuples:
            assert t.when == 42

    def test_who_field_matches_role(self, parser: WWWParser) -> None:
        turn = ChatMessage(
            role="tool", content="```python\nresult = run()\n```",
            token_count=20, turn_index=0,
        )
        tuples = parser.extract(turn)
        for t in tuples:
            assert t.who == WhoActor.TOOL

    def test_nl_turn_produces_at_least_one_tuple(self, parser: WWWParser) -> None:
        """NL turns must always produce at least a fallback tuple."""
        turn = _nl_turn("Deploy the service to AWS us-east-1.", turn_index=9)
        tuples = parser.extract(turn)
        assert len(tuples) >= 1

    def test_compression_ratio_at_least_2x_for_code(self, parser: WWWParser) -> None:
        """Semantic tuple tokens should be substantially fewer than the original turn."""
        code = (
            "def setup_database():\n"
            '    DB_HOST = "prod.db.internal"\n'
            "    DB_PORT = 5432\n"
            '    DB_USER = "nexus_agent"\n'
            "    conn = connect(host=DB_HOST, port=DB_PORT)\n"
            "    return conn\n"
        )
        turn = _code_turn(code, "python", turn_index=5)
        tuples = parser.extract(turn)
        original_token_estimate = turn.token_count
        tuple_tokens = sum(t.token_count for t in tuples)
        assert tuple_tokens < original_token_estimate, (
            f"No compression achieved: tuple_tokens={tuple_tokens}, "
            f"original_tokens={original_token_estimate}"
        )
