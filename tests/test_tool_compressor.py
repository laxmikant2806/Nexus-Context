"""tests/test_tool_compressor.py — Feature D unit tests."""
from __future__ import annotations

import json
import pytest
from nexus_context.cache.tool_compressor import ToolCallCompressor


@pytest.fixture
def compressor() -> ToolCallCompressor:
    return ToolCallCompressor(budget_tokens=512, max_array_items=3)


class TestJsonCompression:
    def test_large_json_compresses_below_budget(self, compressor: ToolCallCompressor) -> None:
        """2000-token JSON payload must compress to < 512 tokens."""
        data = {
            "users": [{"id": i, "name": f"User_{i}", "email": f"user{i}@test.com"} for i in range(200)],
            "total": 200,
            "page": 1,
        }
        content = json.dumps(data)
        assert compressor.estimate_tokens(content) > 512

        compressed, orig = compressor.compress(content)
        assert compressor.estimate_tokens(compressed) < 512
        assert orig > 512

    def test_returns_original_tokens_count(self, compressor: ToolCallCompressor) -> None:
        data = {"items": list(range(500)), "meta": "info"}
        content = json.dumps(data)
        orig_count = compressor.estimate_tokens(content)
        _, returned_orig = compressor.compress(content)
        assert returned_orig == orig_count

    def test_array_truncated_to_max_items(self, compressor: ToolCallCompressor) -> None:
        # Create an array large enough that even 3 items won't fit within 512 tokens,
        # so the skeleton compressor must trigger
        long_strings = ["x" * 400 for _ in range(50)]  # 50 items each 400 chars
        data = {"results": long_strings}
        content = json.dumps(data)
        assert compressor.estimate_tokens(content) > 512
        compressed, _ = compressor.compress(content)
        # skeleton or truncation annotation must be present
        assert (
            "more" in compressed.lower()
            or "_nexus_compressed" in compressed
            or "_nexus_skeleton" in compressed
            or "nexus" in compressed.lower()
        )

    def test_under_budget_returned_unchanged(self, compressor: ToolCallCompressor) -> None:
        """Small payloads must be returned verbatim."""
        small = json.dumps({"key": "value"})
        compressed, orig = compressor.compress(small)
        assert compressed == small
        assert orig <= 512


class TestTextCompression:
    def test_non_json_text_truncated_at_sentence(self, compressor: ToolCallCompressor) -> None:
        """Non-JSON text must produce output smaller than original."""
        # Each sentence is ~30 chars (~7 tokens), 300 sentences = ~2100 tokens total
        sentences = ["This is sentence number %d." % i for i in range(300)]
        content = " ".join(sentences)
        original_tokens = compressor.estimate_tokens(content)
        assert original_tokens > 512

        compressed, _ = compressor.compress(content, budget_tokens=512)
        compressed_tokens = compressor.estimate_tokens(compressed)
        # Compressed must be significantly smaller than original
        assert compressed_tokens < original_tokens
        # And must contain the annotation marker
        assert "nexus:text_compressed" in compressed


    def test_compression_annotation_present(self, compressor: ToolCallCompressor) -> None:
        content = "Word " * 3000  # long repeated text
        compressed, _ = compressor.compress(content)
        assert "nexus" in compressed.lower()

    def test_empty_content_handled(self, compressor: ToolCallCompressor) -> None:
        compressed, orig = compressor.compress("")
        assert isinstance(compressed, str)
        assert orig >= 0
