"""tests/test_session_persistence.py — Feature B unit tests."""
from __future__ import annotations

import asyncio
import os
import tempfile
import pytest
from nexus_context.persistence.session_store import SessionStore


@pytest.fixture
async def store(tmp_path) -> SessionStore:  # type: ignore[type-arg]
    s = SessionStore(db_path=str(tmp_path / "test_sessions.db"))
    await s.initialize()
    return s


class TestSessionStore:
    async def test_save_and_load_basic_fields(self, store: SessionStore) -> None:
        """Saving a session must restore turn_counter and zone_p_hash."""
        await store.save_session(
            session_id="sess-001",
            turn_counter=7,
            zone_p_hash="abc123",
            aligned_zone_p="System prompt content",
            memory_tuples_json=[],
        )
        data = await store.load_session("sess-001")
        assert data is not None
        assert data["turn_counter"] == 7
        assert data["zone_p_hash"] == "abc123"
        assert data["aligned_zone_p"] == "System prompt content"

    async def test_session_not_found_returns_none(self, store: SessionStore) -> None:
        result = await store.load_session("nonexistent-session")
        assert result is None

    async def test_list_sessions_returns_all_ids(self, store: SessionStore) -> None:
        for i in range(3):
            await store.save_session(
                session_id=f"sess-{i:03d}",
                turn_counter=i,
                zone_p_hash=None,
                aligned_zone_p=None,
            )
        sessions = await store.list_sessions()
        assert "sess-000" in sessions
        assert "sess-001" in sessions
        assert "sess-002" in sessions

    async def test_delete_session_removes_from_db(self, store: SessionStore) -> None:
        await store.save_session("sess-del", 3, "hash", "content")
        await store.delete_session("sess-del")
        result = await store.load_session("sess-del")
        assert result is None

    async def test_upsert_updates_existing(self, store: SessionStore) -> None:
        """Saving twice must overwrite, not create a duplicate."""
        await store.save_session("sess-upsert", 1, "h1", "content1")
        await store.save_session("sess-upsert", 5, "h2", "content2")
        data = await store.load_session("sess-upsert")
        assert data is not None
        assert data["turn_counter"] == 5
        assert data["zone_p_hash"] == "h2"

    async def test_memory_tuples_saved_and_loaded(self, store: SessionStore) -> None:
        """Memory tuple JSON must survive a save-load round trip."""
        tuples_json = [
            '{"memory_id":"sess-mt:0:0","session_id":"sess-mt","who":"user","who_detail":"user@turn_0","what":{"mutation_type":"nl_statement","target_name":"task","old_value_repr":null,"new_value_repr":"database setup","side_effects":[],"is_destructive":false},"when":0,"where":"session","token_count":10,"retention_weight":1.0,"is_pinned":false,"ast_depth_score":1.0}'
        ]
        await store.save_session("sess-mt", 1, None, None, tuples_json)
        data = await store.load_session("sess-mt")
        assert data is not None
        assert len(data["memory_tuples_json"]) == 1
