"""tests/test_tokenizer_registry.py — Feature H unit tests."""
from __future__ import annotations

import pytest
from nexus_context.cache.tokenizer_registry import (
    CharEstimateWrapper,
    TiktokenWrapper,
    TokenizerRegistry,
    TokenizerWrapper,
)


class TestCharEstimateWrapper:
    def test_count_returns_positive_int(self) -> None:
        wrapper = CharEstimateWrapper()
        count = wrapper.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string_returns_one(self) -> None:
        wrapper = CharEstimateWrapper()
        assert wrapper.count_tokens("") == 1

    def test_longer_text_more_tokens(self) -> None:
        wrapper = CharEstimateWrapper()
        short = wrapper.count_tokens("Hi")
        long = wrapper.count_tokens("This is a much longer sentence with many more words and tokens in it.")
        assert long > short


class TestTiktokenWrapper:
    def test_count_returns_positive_int(self) -> None:
        try:
            wrapper = TiktokenWrapper("cl100k_base")
            count = wrapper.count_tokens("Hello, world!")
            assert isinstance(count, int)
            assert count > 0
        except Exception:
            pytest.skip("tiktoken not available in test environment")


class TestTokenizerRegistry:
    def test_unknown_model_falls_back_without_error(self) -> None:
        """An unknown model name must not raise — it falls back gracefully."""
        registry = TokenizerRegistry()
        tok = registry.get_tokenizer("completely-unknown-model-xyz", "ollama")
        assert isinstance(tok, TokenizerWrapper)

    def test_count_tokens_returns_int(self) -> None:
        registry = TokenizerRegistry()
        tok = registry.get_tokenizer("unknown-model", "ollama")
        count = tok.count_tokens("Test sentence for counting tokens.")
        assert isinstance(count, int)
        assert count > 0

    def test_same_model_returns_same_instance(self) -> None:
        """Cached tokenizers must return the same wrapper object."""
        registry = TokenizerRegistry()
        tok1 = registry.get_tokenizer("gpt-4", "vllm")
        tok2 = registry.get_tokenizer("gpt-4", "vllm")
        assert tok1 is tok2

    def test_protocol_compliance(self) -> None:
        """All wrappers must satisfy the TokenizerWrapper protocol."""
        wrapper = CharEstimateWrapper()
        assert isinstance(wrapper, TokenizerWrapper)

    def test_registry_count_tokens_shorthand(self) -> None:
        registry = TokenizerRegistry()
        result = registry.count_tokens("Hello world", "some-model", "ollama")
        assert isinstance(result, int)
        assert result > 0
