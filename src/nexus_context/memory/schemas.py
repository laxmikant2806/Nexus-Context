"""
nexus_context.memory.schemas
==============================
Pydantic v2 data models for the memory sub-package.

Models
------
WhoActor     – enum of actor types
MutationType – enum of state-mutation categories
WhatDelta    – the delta payload of a WWW memory tuple
MemoryTuple  – full ⟨Who, What, When, Where⟩ state-mutation record

Reference: docs/architecture.md §4, docs/overview_and_research.md §4.
"""

from __future__ import annotations

import enum
import math

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WhoActor(str, enum.Enum):
    """The actor that generated the state mutation."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class MutationType(str, enum.Enum):
    """Semantic category of the state mutation."""

    VAR_ASSIGN = "var_assign"          # Variable binding / update
    FUNC_DEF = "func_def"              # Function definition
    CLASS_DEF = "class_def"            # Class definition
    IMPORT = "import"                  # Module import
    FS_MUTATION = "fs_mutation"        # File-system create / modify / delete
    SCHEMA_CHANGE = "schema_change"    # DB / JSON schema change (DDL)
    NETWORK_STATE = "network_state"    # HTTP call result
    NL_STATEMENT = "nl_statement"      # Natural-language assertion


# ---------------------------------------------------------------------------
# WhatDelta
# ---------------------------------------------------------------------------


class WhatDelta(BaseModel):
    """The state mutation payload — the *What* component of a WWW tuple.

    Attributes
    ----------
    mutation_type:
        Semantic category of the mutation.
    target_name:
        Primary entity affected (variable name, table name, file path, …).
    old_value_repr:
        String representation of the prior value (``None`` if unknown or
        not applicable).
    new_value_repr:
        String representation of the new value or state after the mutation.
    side_effects:
        Secondary entities affected as a list of ``"name=value"`` strings.
    is_destructive:
        ``True`` for irreversible mutations (DROP TABLE, file DELETE, …).
    """

    mutation_type: MutationType
    target_name: str
    old_value_repr: str | None = Field(default=None)
    new_value_repr: str
    side_effects: list[str] = Field(default_factory=list)
    is_destructive: bool = Field(default=False)

    def compact_repr(self, max_len: int = 60) -> str:
        """Return a compact ``target=new_value`` representation for context injection.

        Truncated to *max_len* characters to minimise memory-block token count.
        """
        raw = f"{self.target_name}={self.new_value_repr}"
        if len(raw) > max_len:
            raw = raw[: max_len - 1] + "…"
        return raw


# ---------------------------------------------------------------------------
# MemoryTuple
# ---------------------------------------------------------------------------


class MemoryTuple(BaseModel):
    """A complete WWW Memory Tuple representing one agent state mutation.

    The four-tuple ⟨Who, What, When, Where⟩ captures the minimal
    information required to reconstruct the agent's operational state
    without retaining the verbose episodic turn.

    Attributes
    ----------
    memory_id:
        Unique ID; format ``{session_id}:{turn_index}:{seq}``.
    session_id:
        Owning session identifier.
    who:
        Actor category.
    who_detail:
        Detailed actor string (e.g. ``"tool:execute_python@scope_3"``).
    what:
        The state mutation payload.
    when:
        Turn index at which the mutation occurred (0-based).
    where:
        AST scope path (e.g. ``"module.MyClass.method"``).
    token_count:
        Estimated token count of the serialised memory tuple.
    retention_weight:
        Current ``W(t, s)`` value; recomputed each turn by
        :meth:`compute_weight`.
    is_pinned:
        If ``True``, bypasses temporal decay and is always retained.
    ast_depth_score:
        ``AST_Depth(s) = (D_max − d(s)) / D_max``; pre-computed at
        extraction.  Root scope = 1.0, deepest nesting → 0.
    """

    memory_id: str
    session_id: str

    # WWW components --------------------------------------------------------
    who: WhoActor
    who_detail: str = Field(
        description="'tool:execute_python@scope_3', 'user@turn_5', etc."
    )
    what: WhatDelta
    when: int = Field(ge=0, description="Turn index at which mutation occurred.")
    where: str = Field(
        description="AST scope path: 'module.MyClass.init' or 'session.subtask_2'."
    )

    # Computed fields -------------------------------------------------------
    token_count: int = Field(ge=0, description="Token count of the serialised tuple.")
    retention_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="W(t, s); recomputed by MemoryPool.update_weights().",
    )
    is_pinned: bool = Field(
        default=False,
        description="Pinned memories are never pruned regardless of age.",
    )
    ast_depth_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="(D_max − d(s)) / D_max; 1.0 for root scope.",
    )

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_weight(
        self,
        current_turn: int,
        lambda_: float = 0.05,
        eta: float = 0.5,
    ) -> float:
        """Compute the retention weight W(t_i, s_i).

        .. math::

            W(t_i, s_i) = \\exp(-\\lambda (T - t_i)) \\cdot (1 + \\eta \\cdot \\text{AST\\_Depth}(s_i))

        Parameters
        ----------
        current_turn:
            Current turn index *T*.
        lambda_:
            Temporal decay constant (default 0.05, half-life ≈ 13.9 turns).
        eta:
            AST depth amplification factor (default 0.5).

        Returns
        -------
        float
            Retention weight in range ``(0, 1 + eta]``, or ``math.inf``
            if the tuple is pinned.
        """
        if self.is_pinned:
            return math.inf
        age = max(0, current_turn - self.when)
        temporal = math.exp(-lambda_ * age)
        depth_boost = 1.0 + eta * self.ast_depth_score
        return temporal * depth_boost
