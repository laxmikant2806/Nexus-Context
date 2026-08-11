"""
nexus_context.cache.differential
==================================
Partitions a session's message list into the three immutable context zones:

    Zone P  –  Locked Head    (block-aligned system prompt, frozen forever)
    Zone T  –  Compacted Trunk (historical turns eligible for submodular pruning)
    Zone R  –  Raw Tail        (verbatim recent turns, append-only)

Reference: docs/architecture.md §3.4, docs/implementation_stage.md §2.

The graduation policy moves turns from Zone R into Zone T candidates once
they are older than ``tail_retention_turns`` turns, triggering a
re-evaluation by the submodular solver.
"""

from __future__ import annotations

import logging

from nexus_context import SegmentationError
from nexus_context.cache.schemas import ChatMessage, ZoneBundle

logger = logging.getLogger(__name__)


class ZoneSegmenter:
    """Partitions messages into Zones P, T, and R.

    Parameters
    ----------
    total_budget:
        Total context-window token budget *B* (must match backend ``max_model_len``).
    block_size:
        PagedAttention block size (16 or 32), used only for logging/validation.
    zone_p_aligned_tokens:
        Token count of the already-padded Zone P (from :class:`BlockAlignResult`).
    tail_budget:
        Maximum tokens Zone R may consume.  Default: 1 024.
    tail_retention_turns:
        A Zone-R turn graduates to Zone T after this many turns have elapsed.
        Default: 3.
    """

    def __init__(
        self,
        total_budget: int,
        block_size: int,
        zone_p_aligned_tokens: int,
        tail_budget: int = 1024,
        tail_retention_turns: int = 3,
    ) -> None:
        if total_budget < 1:
            raise SegmentationError(f"total_budget must be >= 1, got {total_budget}.")
        if zone_p_aligned_tokens % block_size != 0:
            raise SegmentationError(
                f"zone_p_aligned_tokens={zone_p_aligned_tokens} is not divisible by "
                f"block_size={block_size}.  Ensure BlockAligner.align() was called first."
            )
        remaining = total_budget - zone_p_aligned_tokens
        if tail_budget > remaining:
            raise SegmentationError(
                f"tail_budget={tail_budget} exceeds remaining budget "
                f"({remaining} = {total_budget} − {zone_p_aligned_tokens})."
            )

        self._total_budget = total_budget
        self._block_size = block_size
        self._zone_p_tokens = zone_p_aligned_tokens
        self._zone_r_budget = tail_budget
        self._zone_t_budget = remaining - tail_budget
        self._tail_retention_turns = tail_retention_turns

        logger.info(
            '{"event":"zone_segmenter_init","B":%d,"P":%d,"T":%d,"R":%d}',
            total_budget,
            zone_p_aligned_tokens,
            self._zone_t_budget,
            tail_budget,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, messages: list[ChatMessage]) -> ZoneBundle:
        """Partition *messages* into the three zones.

        The algorithm is:

        1. Extract the ``system`` message → **Zone P**.
        2. Collect remaining messages in chronological order.
        3. Greedily assign the most-recent messages to **Zone R** until the
           Zone R budget is exhausted.
        4. Everything older goes to **Zone T candidates**.

        Parameters
        ----------
        messages:
            Full message list for the current turn, ordered by ``turn_index``.

        Returns
        -------
        ZoneBundle
        """
        if not messages:
            raise SegmentationError("messages list must not be empty.")

        # ----------------------------------------------------------------
        # 1. Extract system message (Zone P)
        # ----------------------------------------------------------------
        system_messages = [m for m in messages if m.role == "system"]
        if not system_messages:
            raise SegmentationError(
                "No system-role message found.  Zone P requires exactly one system message."
            )
        if len(system_messages) > 1:
            logger.warning(
                '{"event":"multiple_system_messages","count":%d,'
                '"note":"using first occurrence for Zone P"}',
                len(system_messages),
            )
        zone_p_msg = system_messages[0]

        # ----------------------------------------------------------------
        # 2. Remaining messages in chronological order
        # ----------------------------------------------------------------
        non_system = sorted(
            [m for m in messages if m.role != "system"],
            key=lambda m: m.turn_index,
        )

        # ----------------------------------------------------------------
        # 3. Fill Zone R from the tail
        # ----------------------------------------------------------------
        zone_r: list[ChatMessage] = []
        r_tokens_used = 0
        for msg in reversed(non_system):
            if r_tokens_used + msg.token_count <= self._zone_r_budget:
                zone_r.insert(0, msg)
                r_tokens_used += msg.token_count
            else:
                break

        # ----------------------------------------------------------------
        # 4. Everything older goes to Zone T candidates
        # ----------------------------------------------------------------
        zone_r_indices = {m.turn_index for m in zone_r}
        zone_t_candidates = [m for m in non_system if m.turn_index not in zone_r_indices]

        bundle = ZoneBundle(
            zone_p_message=zone_p_msg,
            zone_t_candidates=zone_t_candidates,
            zone_r_messages=zone_r,
            zone_p_budget=self._zone_p_tokens,
            zone_t_budget=self._zone_t_budget,
            zone_r_budget=self._zone_r_budget,
            total_budget=self._total_budget,
        )
        self._validate(bundle)

        logger.debug(
            '{"event":"segment_complete","n_zone_t":%d,"n_zone_r":%d,'
            '"zone_t_tokens":%d,"zone_r_tokens":%d,"compaction_required":%s}',
            len(zone_t_candidates),
            len(zone_r),
            bundle.zone_t_token_count,
            bundle.zone_r_token_count,
            str(bundle.compaction_required).lower(),
        )
        return bundle

    def graduate_tail_turns(self, bundle: ZoneBundle, current_turn: int) -> ZoneBundle:
        """Move Zone-R turns older than ``tail_retention_turns`` to Zone T.

        This is called at the *start* of each new turn before segmentation,
        so that stale Zone-R turns become compaction candidates.

        Parameters
        ----------
        bundle:
            The :class:`ZoneBundle` produced by the previous call to
            :meth:`segment`.
        current_turn:
            The current turn index (0-based, monotonically increasing).

        Returns
        -------
        ZoneBundle
            Updated bundle with graduated turns moved to ``zone_t_candidates``.
        """
        graduating: list[ChatMessage] = []
        staying: list[ChatMessage] = []

        for msg in bundle.zone_r_messages:
            age = current_turn - msg.turn_index
            if age >= self._tail_retention_turns:
                graduating.append(msg)
            else:
                staying.append(msg)

        if not graduating:
            return bundle  # nothing changed

        new_zone_t = sorted(
            bundle.zone_t_candidates + graduating,
            key=lambda m: m.turn_index,
        )

        logger.debug(
            '{"event":"graduation","n_graduated":%d,"current_turn":%d}',
            len(graduating),
            current_turn,
        )

        updated = ZoneBundle(
            zone_p_message=bundle.zone_p_message,
            zone_t_candidates=new_zone_t,
            zone_r_messages=staying,
            zone_p_budget=bundle.zone_p_budget,
            zone_t_budget=bundle.zone_t_budget,
            zone_r_budget=bundle.zone_r_budget,
            total_budget=bundle.total_budget,
        )
        self._validate(updated)
        return updated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, bundle: ZoneBundle) -> None:
        """Assert zone-bundle invariants; raise :class:`SegmentationError` on violation."""
        p_tokens = bundle.zone_p_message.token_count
        t_tokens = bundle.zone_t_token_count
        r_tokens = bundle.zone_r_token_count
        total = p_tokens + t_tokens + r_tokens

        if total > bundle.total_budget:
            raise SegmentationError(
                f"Zone token sum ({total}) exceeds total_budget ({bundle.total_budget}). "
                f"P={p_tokens}, T={t_tokens}, R={r_tokens}."
            )

        if bundle.zone_p_message.role != "system":
            raise SegmentationError(
                f"zone_p_message must have role='system', got '{bundle.zone_p_message.role}'."
            )

        for i, msg in enumerate(bundle.zone_t_candidates):
            if i > 0:
                prev = bundle.zone_t_candidates[i - 1]
                if msg.turn_index <= prev.turn_index:
                    raise SegmentationError(
                        f"zone_t_candidates are not in strictly ascending turn_index order "
                        f"at index {i}: turn_index={msg.turn_index} <= {prev.turn_index}."
                    )
