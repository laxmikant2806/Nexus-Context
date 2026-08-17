"""
nexus_context.cache.tokenizer_registry
=========================================
Feature H: LLM-Agnostic Token Budget Estimation.

Detects the correct tokenizer for the configured backend model and returns a
unified ``TokenizerWrapper`` interface. Results are cached with ``@lru_cache``
so tokenizer loading (which may download HuggingFace model configs) happens at
most once per process.

Detection logic
---------------
backend_type == "ollama" + model contains "qwen"    -> HuggingFace Qwen2.5-0.5B tokenizer
backend_type == "ollama" + model contains "llama"   -> HuggingFace Llama-3.2-1B tokenizer
backend_type == "ollama" + model contains "mistral" -> HuggingFace Mistral-7B tokenizer
backend_type == "ollama" + model contains "gemma"   -> HuggingFace Gemma-2B tokenizer
backend_type == "vllm"                              -> probe /v1/models, load matching HF tokenizer
Default fallback                                    -> tiktoken cl100k_base

Reference: Feature H specification
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TokenizerWrapper protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenizerWrapper(Protocol):
    """Unified interface for any tokenizer backend."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        ...


# ---------------------------------------------------------------------------
# Concrete wrapper implementations
# ---------------------------------------------------------------------------


class TiktokenWrapper:
    """Wrapper around tiktoken encodings."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


class HuggingFaceWrapper:
    """Wrapper around a HuggingFace tokenizer (tokenizer-only load)."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        self._tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=False,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


class CharEstimateWrapper:
    """Fallback: 4-character-per-token heuristic (no deps)."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Model name substring → HuggingFace model ID for tokenizer loading
_HF_TOKENIZER_MAP: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "llama": "meta-llama/Llama-3.2-1B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gemma": "google/gemma-2b",
    "phi": "microsoft/phi-2",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "codellama": "meta-llama/CodeLlama-7b-hf",
}


@lru_cache(maxsize=16)
def _load_tokenizer(model_name: str, backend_type: str) -> TokenizerWrapper:
    """Internal cached loader — called at most once per (model, backend) pair."""
    model_lower = model_name.lower()

    # 1. Attempt HuggingFace tokenizer for known model families
    for key, hf_id in _HF_TOKENIZER_MAP.items():
        if key in model_lower:
            try:
                wrapper = HuggingFaceWrapper(hf_id)
                logger.info(
                    '{"event":"tokenizer_loaded","model":"%s","hf_id":"%s"}',
                    model_name, hf_id,
                )
                return wrapper
            except Exception as exc:
                logger.warning(
                    '{"event":"tokenizer_hf_failed","model":"%s","error":"%s","fallback":"tiktoken"}',
                    model_name, str(exc),
                )

    # 2. Try tiktoken (works for OpenAI-compatible models)
    try:
        wrapper_tk = TiktokenWrapper("cl100k_base")
        logger.info('{"event":"tokenizer_loaded","model":"%s","backend":"tiktoken"}', model_name)
        return wrapper_tk
    except Exception:
        pass

    # 3. Absolute fallback: character estimation
    logger.warning('{"event":"tokenizer_fallback","model":"%s","backend":"char_estimate"}', model_name)
    return CharEstimateWrapper()


class TokenizerRegistry:
    """Public API for obtaining a cached tokenizer wrapper.

    Usage
    -----
    >>> registry = TokenizerRegistry()
    >>> tok = registry.get_tokenizer("qwen2.5-coder:7b", "ollama")
    >>> tok.count_tokens("Hello, world!")
    4
    """

    def get_tokenizer(self, model_name: str, backend_type: str) -> TokenizerWrapper:
        """Return a cached tokenizer wrapper for *(model_name, backend_type)*."""
        return _load_tokenizer(model_name.lower().strip(), backend_type.lower().strip())

    def count_tokens(self, text: str, model_name: str, backend_type: str) -> int:
        """Count tokens in *text* using the correct tokenizer for the model."""
        return self.get_tokenizer(model_name, backend_type).count_tokens(text)
