"""
context_nexus.budget
====================
Deterministic token budget allocator using tiktoken (or fallback char estimation).
Truncates and ranks context snippets to strictly fit inside a target token budget
(e.g., 4000 or 8000 tokens) without cutting off mid-sentence.
"""

from __future__ import annotations

import re
from typing import Any


class TokenBudgetAllocator:
    """Allocates and truncates context snippets to strictly fit within token budget."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._tokenizer: Any = self._load_tokenizer(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def allocate_snippets(
        self,
        snippets: list[str],
        budget_tokens: int,
        scores: list[float] | None = None,
    ) -> list[str]:
        """Select highest-scored snippets that strictly fit within budget_tokens.

        Does not cut off mid-sentence unless a single snippet exceeds budget.
        """
        if not snippets:
            return []

        if scores is None or len(scores) != len(snippets):
            scores = [1.0] * len(snippets)

        # Pair snippets with scores and token counts
        paired = []
        for snip, sc in zip(snippets, scores):
            cnt = self.count_tokens(snip)
            paired.append((snip, sc, cnt))

        # Sort descending by score
        paired.sort(key=lambda x: x[1], reverse=True)

        selected: list[str] = []
        tokens_used = 0

        for snip, sc, cnt in paired:
            if tokens_used + cnt <= budget_tokens:
                selected.append(snip)
                tokens_used += cnt
            elif tokens_used < budget_tokens:
                # Partial truncate at sentence boundary if space remains
                remaining_budget = budget_tokens - tokens_used
                truncated = self.truncate_to_sentences(snip, remaining_budget)
                if truncated.strip():
                    selected.append(truncated)
                    tokens_used += self.count_tokens(truncated)
                break

        return selected

    def truncate_to_sentences(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens ending at a complete sentence boundary."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        current_tokens = 0

        for s in sentences:
            s_tokens = self.count_tokens(s)
            if current_tokens + s_tokens <= max_tokens:
                result.append(s)
                current_tokens += s_tokens
            else:
                break

        return " ".join(result)

    def _load_tokenizer(self, encoding_name: str) -> Any:
        try:
            import tiktoken
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            return None
