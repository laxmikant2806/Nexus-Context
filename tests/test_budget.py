"""
tests/test_budget.py
=====================
Unit tests for context_nexus.budget.
"""

from __future__ import annotations

from context_nexus.budget import TokenBudgetAllocator


def test_count_tokens() -> None:
    allocator = TokenBudgetAllocator()
    count = allocator.count_tokens("Hello world from Nexus-Context")
    assert count >= 1


def test_allocate_snippets_within_budget() -> None:
    allocator = TokenBudgetAllocator()
    snippets = [
        "First short snippet about PostgreSQL database.",
        "Second short snippet about vLLM prefix cache alignment.",
        "Third short snippet about submodular referential integrity solver.",
    ]
    scores = [0.9, 0.8, 0.7]
    selected = allocator.allocate_snippets(snippets, budget_tokens=50, scores=scores)
    assert len(selected) >= 1
    total_tokens = sum(allocator.count_tokens(s) for s in selected)
    assert total_tokens <= 50
