"""
tests/test_cache.py
====================
Unit tests for nexus_context.cache (Phase 1).

Covers:
    BlockAligner   – padding, block alignment, hash stability
    ZoneSegmenter  – zone partitioning, token budget invariant, tail graduation
"""

from __future__ import annotations

import pytest
from nexus_context.cache.block_align import BlockAligner
from nexus_context.cache.differential import ZoneSegmenter
from nexus_context.cache.schemas import ChatMessage, ZoneBundle


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_aligner(block_size: int = 16) -> BlockAligner:
    """Return a mock BlockAligner (1 char = 1 token) for fast unit tests."""
    return BlockAligner._from_mock(block_size=block_size)


def _make_messages(token_counts: list[int]) -> list[ChatMessage]:
    """Build a message list with the given per-message token counts.

    First message is always system; remainder alternate user/assistant.
    """
    roles = ["system"] + ["user", "assistant"] * 50
    return [
        ChatMessage(
            role=roles[i],
            content="x" * token_counts[i],
            token_count=token_counts[i],
            turn_index=i,
        )
        for i in range(len(token_counts))
    ]


# ---------------------------------------------------------------------------
# BlockAligner tests
# ---------------------------------------------------------------------------


class TestBlockAlign:

    def test_already_aligned_prompt_no_padding(self) -> None:
        aligner = _make_aligner(block_size=16)
        result = aligner.align("a" * 16)
        assert result.padding_count == 0
        assert result.aligned_token_count == 16

    def test_unaligned_513_pads_to_528(self) -> None:
        aligner = _make_aligner(block_size=16)
        result = aligner.align("a" * 513)
        assert result.aligned_token_count == 528  # 33 × 16
        assert result.padding_count == 15
        assert result.aligned_token_count % 16 == 0

    def test_block_alignment_invariant_holds_for_all_lengths(self) -> None:
        aligner = _make_aligner(block_size=16)
        for length in range(1, 130):
            result = aligner.align("a" * length)
            assert result.aligned_token_count % 16 == 0, (
                f"alignment failed for length={length}: "
                f"aligned={result.aligned_token_count}"
            )

    def test_zone_p_hash_stable_for_identical_input(self) -> None:
        aligner = _make_aligner()
        r1 = aligner.align("You are a helpful assistant.")
        r2 = aligner.align("You are a helpful assistant.")
        assert r1.zone_p_hash == r2.zone_p_hash

    def test_zone_p_hash_differs_for_different_input(self) -> None:
        aligner = _make_aligner()
        r1 = aligner.align("Hello world")
        r2 = aligner.align("Hello World")  # capital W
        assert r1.zone_p_hash != r2.zone_p_hash

    def test_hash_is_64_char_hex_string(self) -> None:
        aligner = _make_aligner()
        result = aligner.align("test prompt")
        assert len(result.zone_p_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.zone_p_hash)

    def test_block_size_32_alignment(self) -> None:
        aligner = _make_aligner(block_size=32)
        result = aligner.align("a" * 50)
        assert result.aligned_token_count % 32 == 0
        assert result.aligned_token_count == 64  # next multiple of 32 above 50


# ---------------------------------------------------------------------------
# ZoneSegmenter tests
# ---------------------------------------------------------------------------


class TestZoneSegmenter:

    def _make_segmenter(
        self,
        total_budget: int = 512,
        zone_p_aligned: int = 112,  # 7 × 16
        tail_budget: int = 200,
    ) -> ZoneSegmenter:
        return ZoneSegmenter(
            total_budget=total_budget,
            block_size=16,
            zone_p_aligned_tokens=zone_p_aligned,
            tail_budget=tail_budget,
            tail_retention_turns=2,
        )

    def test_system_message_always_zone_p(self) -> None:
        messages = _make_messages([112, 50, 50, 50])
        seg = self._make_segmenter()
        bundle = seg.segment(messages)
        assert bundle.zone_p_message.role == "system"

    def test_total_tokens_within_budget(self) -> None:
        messages = _make_messages([112, 50, 50, 50])
        seg = self._make_segmenter()
        bundle = seg.segment(messages)
        total = (
            bundle.zone_p_message.token_count
            + bundle.zone_t_token_count
            + bundle.zone_r_token_count
        )
        assert total <= 512

    def test_zone_t_candidates_chronological(self) -> None:
        messages = _make_messages([112, 30, 30, 30, 30])
        seg = self._make_segmenter()
        bundle = seg.segment(messages)
        turn_indices = [m.turn_index for m in bundle.zone_t_candidates]
        assert turn_indices == sorted(turn_indices)

    def test_compaction_required_flag(self) -> None:
        # Zone T budget = 512 - 112 - 200 = 200 tokens
        # With 4 × 50-token messages, Zone T gets at least 2 messages (100 tok)
        # within budget = no compaction.
        messages = _make_messages([112, 50, 50, 50, 50])
        seg = self._make_segmenter()
        bundle = seg.segment(messages)
        # compaction_required == True only when Zone T candidates exceed budget
        if bundle.zone_t_token_count > bundle.zone_t_budget:
            assert bundle.compaction_required is True
        else:
            assert bundle.compaction_required is False

    def test_graduation_moves_old_turns_to_zone_t(self) -> None:
        messages = _make_messages([112, 30, 30, 30, 30])
        seg = self._make_segmenter(tail_budget=300)
        bundle = seg.segment(messages)

        # At current_turn=10, all Zone R turns (turn_index 1–4) have age >= 6
        # which is >= tail_retention_turns=2, so all should graduate.
        updated = seg.graduate_tail_turns(bundle, current_turn=10)
        for msg in updated.zone_r_messages:
            age = 10 - msg.turn_index
            assert age < 2, (
                f"Message with turn_index={msg.turn_index} stayed in Zone R "
                f"despite age={age} >= tail_retention_turns=2"
            )

    def test_no_system_message_raises(self) -> None:
        msgs = [
            ChatMessage(role="user", content="hello", token_count=10, turn_index=0)
        ]
        seg = self._make_segmenter()
        with pytest.raises(Exception):  # SegmentationError
            seg.segment(msgs)
