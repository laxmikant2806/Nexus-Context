"""
nexus_context.cache.schemas
===========================
Pydantic v2 data models for the cache sub-package.

Models
------
ChatMessage       – an OpenAI-format chat message with a pre-computed token count
BlockAlignResult  – output of BlockAligner.align()
ZoneType          – enum for the three context zones (P / T / R)
ZoneBundle        – output of ZoneSegmenter.segment()
CacheBoundary     – token-offset boundaries for Zones P, T, R in one session
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single OpenAI-format chat message with a pre-computed token count.

    The ``token_count`` field is populated by the :class:`BlockAligner` or
    :class:`ZoneSegmenter` during session initialisation and must not be
    mutated afterwards.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    token_count: int = Field(ge=0, description="Pre-computed token count for this message.")
    turn_index: int = Field(
        ge=0, description="0-based position in session history (monotonically increasing)."
    )
    name: str | None = Field(
        default=None,
        description="Optional name field (used for tool-role messages in some APIs).",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# BlockAlignResult
# ---------------------------------------------------------------------------


class BlockAlignResult(BaseModel):
    """Output of :meth:`BlockAligner.align`.

    Captures the before/after token counts, the number of neutral padding
    tokens appended, the padded system-prompt text, and the SHA-256 hex
    digest of the aligned token-ID sequence.
    """

    original_token_count: int = Field(ge=0)
    aligned_token_count: int = Field(
        ge=0,
        description="Token count after padding; must satisfy aligned_token_count % block_size == 0.",
    )
    padding_count: int = Field(
        ge=0, description="Number of neutral tokens appended (= aligned − original)."
    )
    padded_content: str = Field(
        description="System-prompt text with padding appended; this is Zone P verbatim."
    )
    block_size: int = Field(ge=1)
    zone_p_hash: str = Field(
        description="SHA-256 hex digest of the aligned token-ID byte sequence."
    )

    @field_validator("aligned_token_count")
    @classmethod
    def must_be_block_aligned(cls, v: int, info: object) -> int:  # type: ignore[override]
        # Access block_size from the partially-constructed model data.
        # Pydantic v2 passes a ValidationInfo object; use .data dict.
        data = getattr(info, "data", {})
        bs = data.get("block_size", 1)
        if bs > 0 and v % bs != 0:
            raise ValueError(
                f"aligned_token_count={v} is not divisible by block_size={bs}. "
                "Zone P must be exactly block-aligned."
            )
        return v


# ---------------------------------------------------------------------------
# ZoneType
# ---------------------------------------------------------------------------


class ZoneType(str, enum.Enum):
    """Identifies one of the three context zones managed by Nexus-Context."""

    LOCKED_HEAD = "P"      # System prompt (block-aligned, immutable)
    COMPACTED_TRUNK = "T"  # Submodular-pruned historical turns
    RAW_TAIL = "R"         # Verbatim recent turns (append-only)


# ---------------------------------------------------------------------------
# ZoneBundle
# ---------------------------------------------------------------------------


class ZoneBundle(BaseModel):
    """The three-zone partition of a session's message list.

    Produced by :meth:`ZoneSegmenter.segment` and updated by
    :meth:`ZoneSegmenter.graduate_tail_turns` on every subsequent turn.
    """

    zone_p_message: ChatMessage = Field(
        description="The system-prompt message (role='system'), already block-aligned."
    )
    zone_t_candidates: list[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Historical turns eligible for submodular compaction, ordered by "
            "turn_index ascending."
        ),
    )
    zone_r_messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Verbatim recent turns (Zone R), ordered by turn_index ascending.",
    )
    zone_p_budget: int = Field(
        ge=0,
        description="Tokens allocated to Zone P (= BlockAlignResult.aligned_token_count).",
    )
    zone_t_budget: int = Field(
        ge=0,
        description="Maximum tokens Zone T may consume after submodular pruning.",
    )
    zone_r_budget: int = Field(
        ge=0, default=1024, description="Maximum tokens Zone R may consume."
    )
    total_budget: int = Field(ge=1, description="Total context-window token budget B.")

    @field_validator("zone_p_message")
    @classmethod
    def system_role_required(cls, v: ChatMessage) -> ChatMessage:
        if v.role != "system":
            raise ValueError(
                f"zone_p_message must have role='system', got role='{v.role}'."
            )
        return v

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def zone_t_token_count(self) -> int:
        """Sum of token counts for all Zone T candidates."""
        return sum(m.token_count for m in self.zone_t_candidates)

    @property
    def zone_r_token_count(self) -> int:
        """Sum of token counts for all Zone R messages."""
        return sum(m.token_count for m in self.zone_r_messages)

    @property
    def compaction_required(self) -> bool:
        """``True`` when Zone T candidates exceed their token budget."""
        return self.zone_t_token_count > self.zone_t_budget


# ---------------------------------------------------------------------------
# CacheBoundary
# ---------------------------------------------------------------------------


class CacheBoundary(BaseModel):
    """Token-offset boundaries for Zones P, T, R within one session.

    Created once during session initialisation and stored in
    :class:`SessionState`. The ``zone_p_hash`` field is used at every
    subsequent turn to verify that Zone P has not been mutated.
    """

    session_id: str
    block_size: int = Field(
        default=16,
        description="PagedAttention block size in tokens (16 or 32).",
    )

    # Zone P ----------------------------------------------------------------
    zone_p_token_start: int = Field(default=0, ge=0)
    zone_p_token_end: int = Field(
        ge=0,
        description="Exclusive end offset; must satisfy zone_p_token_end % block_size == 0.",
    )
    zone_p_hash: str = Field(
        description="SHA-256 of Zone P token IDs; verified on every turn for integrity."
    )
    zone_p_padding_tokens: int = Field(
        default=0,
        ge=0,
        description="Neutral padding tokens appended to reach block alignment.",
    )

    # Zone T ----------------------------------------------------------------
    zone_t_token_start: int = Field(ge=0)
    zone_t_token_end: int = Field(ge=0)
    zone_t_budget: int = Field(ge=0, description="Maximum Zone T token budget.")

    # Zone R ----------------------------------------------------------------
    zone_r_token_start: int = Field(ge=0)
    zone_r_token_end: int = Field(ge=0)
    zone_r_budget: int = Field(default=1024, ge=0)

    # Total -----------------------------------------------------------------
    total_budget: int = Field(ge=1, description="Full context-window budget B.")

    @field_validator("zone_p_token_end")
    @classmethod
    def zone_p_must_be_block_aligned(cls, v: int, info: object) -> int:  # type: ignore[override]
        data = getattr(info, "data", {})
        bs = data.get("block_size", 16)
        if bs > 0 and v % bs != 0:
            raise ValueError(
                f"zone_p_token_end={v} is not divisible by block_size={bs}. "
                "Zone P must end on a block boundary to preserve prefix-cache hashes."
            )
        return v
