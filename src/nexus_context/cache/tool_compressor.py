"""
nexus_context.cache.tool_compressor
=====================================
Feature D: Agent Tool Call Interception & Budget-Aware Compression.

Intercepts ``role == "tool"`` messages whose content exceeds a token budget,
applies JSON-aware or sentence-boundary truncation, and returns the compressed
content alongside the original token count saved.

This runs *before* zone segmentation so the compacted tool outputs count
against the real token budget correctly.

Reference: Feature D specification
"""

from __future__ import annotations

import json
import re


class ToolCallCompressor:
    """Compress oversized tool-call response content.

    Parameters
    ----------
    budget_tokens:
        Maximum tokens the compressed output should occupy (default 512).
    max_array_items:
        How many items to retain in JSON arrays before truncating (default 3).
    """

    def __init__(self, budget_tokens: int = 512, max_array_items: int = 3) -> None:
        self.budget_tokens = budget_tokens
        self.max_array_items = max_array_items

    def estimate_tokens(self, text: str) -> int:
        """Fast character-based token estimation (4 chars per token on average)."""
        return max(1, len(text) // 4)

    def compress(self, content: str, budget_tokens: int | None = None) -> tuple[str, int]:
        """Compress *content* to fit within *budget_tokens*.

        Returns
        -------
        tuple[str, int]
            ``(compressed_content, original_token_count)``
            If content is already within budget, returns it unchanged.
        """
        limit = budget_tokens or self.budget_tokens
        original_tokens = self.estimate_tokens(content)

        if original_tokens <= limit:
            return content, original_tokens

        # Attempt JSON-aware compression first
        compressed = self._compress_json(content, limit)
        if compressed is not None:
            return compressed, original_tokens

        # Fall back to sentence-boundary truncation
        return self._compress_text(content, limit), original_tokens

    # ------------------------------------------------------------------
    # JSON-aware compression
    # ------------------------------------------------------------------

    def _compress_json(self, content: str, budget_tokens: int) -> str | None:
        """Try to parse content as JSON and compress arrays/deep objects."""
        try:
            data = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            return None

        compressed_data = self._truncate_json_value(data)
        result = json.dumps(compressed_data, separators=(",", ":"), ensure_ascii=False)

        # Add compression annotation
        original_keys = self._count_keys(data)
        compressed_keys = self._count_keys(compressed_data)
        if original_keys > compressed_keys:
            result = result.rstrip("}")
            result += f', "_nexus_compressed": true, "_original_size": "{budget_tokens}+ tokens"}}'

        if self.estimate_tokens(result) <= budget_tokens:
            return result

        # If still too large, return a summarized skeleton
        return self._json_skeleton(data, budget_tokens)

    def _truncate_json_value(self, value: object) -> object:
        """Recursively truncate arrays and nested objects."""
        if isinstance(value, list):
            truncated = value[: self.max_array_items]
            remainder = len(value) - self.max_array_items
            result = [self._truncate_json_value(v) for v in truncated]
            if remainder > 0:
                result.append(f"... {remainder} more items")
            return result
        if isinstance(value, dict):
            return {k: self._truncate_json_value(v) for k, v in list(value.items())[:20]}
        if isinstance(value, str) and len(value) > 200:
            return value[:197] + "..."
        return value

    def _json_skeleton(self, data: object, budget_tokens: int) -> str:
        """Return a compact key-only skeleton for deeply nested JSON."""
        if isinstance(data, dict):
            keys = list(data.keys())[:15]
            skeleton = {k: type(data[k]).__name__ for k in keys}
            extra = len(data) - 15
            note = f"... {extra} more keys" if extra > 0 else ""
            result = json.dumps({"_nexus_skeleton": True, "keys": skeleton, "note": note})
        else:
            result = f"[Array of {len(data)} items — compressed by Nexus-Context]"  # type: ignore[arg-type]
        return result

    def _count_keys(self, data: object, depth: int = 0) -> int:
        if isinstance(data, dict):
            return len(data) + sum(self._count_keys(v, depth + 1) for v in data.values() if depth < 3)
        if isinstance(data, list):
            return sum(self._count_keys(v, depth + 1) for v in data[:self.max_array_items] if depth < 3)
        return 0

    # ------------------------------------------------------------------
    # Text sentence-boundary compression
    # ------------------------------------------------------------------

    def _compress_text(self, content: str, budget_tokens: int) -> str:
        """Truncate content at a sentence boundary to fit budget_tokens."""
        # Split at sentence boundaries: period, exclamation, question followed by space
        sentences = re.split(r"(?<=[.!?\n])\s+", content)
        result_parts: list[str] = []
        tokens_used = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            # Check BEFORE adding to enforce strict budget
            if tokens_used + sentence_tokens > budget_tokens:
                break
            result_parts.append(sentence)
            tokens_used += sentence_tokens

        result = " ".join(result_parts)
        if not result.strip():
            # At minimum return the first budget_tokens*4 characters
            result = content[: budget_tokens * 4]

        return result + "\n<!-- nexus:text_compressed -->"
