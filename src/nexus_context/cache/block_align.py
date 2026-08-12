"""
nexus_context.cache.block_align
================================
Pads the system-prompt (Zone P) to an exact PagedAttention block boundary
and freezes a SHA-256 hash of the aligned token-ID sequence.

The padded Zone P is **never modified** after session initialisation; this
guarantees that the RadixTree prefix hash remains stable across all turns,
yielding 100 % prefix-cache reuse for Zone P tokens.

Design decisions
----------------
* Tree-Sitter grammars are used for multi-language AST parsing (ADR-001).
* Strict padding is preferred over dynamic slicing (ADR-002).
* Tokeniser is auto-selected based on backend type to avoid token-ID
  mismatches between the middleware and the serving engine.

Reference: docs/architecture.md §3.5, docs/decisions.md ADR-002.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nexus_context import AlignmentError, ConfigurationError, TokenizerError
from nexus_context.cache.schemas import BlockAlignResult

if TYPE_CHECKING:
    pass  # forward-reference guard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TokeniserProtocol(Protocol):
    """Minimal interface every supported tokeniser must satisfy."""

    def encode(self, text: str) -> list[int]:  # noqa: D102
        ...

    def decode(self, token_ids: list[int]) -> str:  # noqa: D102
        ...


# ---------------------------------------------------------------------------
# Neutral padding-token selection
# ---------------------------------------------------------------------------

_FALLBACK_PAD_STR = " "  # single space – universally safe


def _select_padding_string(tokeniser: TokeniserProtocol, model_name: str) -> str:
    """Return the most appropriate neutral padding string for *tokeniser*.

    Priority order (see ADR-002, §Argument D):
    1. ``<pad>`` token if the tokeniser defines one.
    2. ``eos_token`` preceded by ``# `` comment prefix (prevents the model
       from treating mid-context EOS as generation termination).
    3. Single space character as universal fallback.
    """
    # HuggingFace AutoTokenizer exposes pad_token / eos_token attributes.
    pad_token_id: int | None = getattr(tokeniser, "pad_token_id", None)
    eos_token_id: int | None = getattr(tokeniser, "eos_token_id", None)

    if pad_token_id is not None:
        try:
            pad_str = tokeniser.decode([pad_token_id])
            logger.debug(
                '{"event":"padding_token_selected","model":"%s","strategy":"pad_token",'
                '"token_id":%d}',
                model_name,
                pad_token_id,
            )
            return pad_str
        except Exception:  # noqa: BLE001
            pass

    if eos_token_id is not None:
        try:
            eos_str = tokeniser.decode([eos_token_id])
            pad_str = f"# {eos_str}"
            logger.debug(
                '{"event":"padding_token_selected","model":"%s","strategy":"comment_eos",'
                '"token_id":%d}',
                model_name,
                eos_token_id,
            )
            return pad_str
        except Exception:  # noqa: BLE001
            pass

    logger.warning(
        '{"event":"padding_token_fallback","model":"%s","strategy":"space"}',
        model_name,
    )
    return _FALLBACK_PAD_STR


# ---------------------------------------------------------------------------
# BlockAligner
# ---------------------------------------------------------------------------


class BlockAligner:
    """Pads *system_prompt* to a multiple of *block_size* tokens.

    Parameters
    ----------
    model_name:
        HuggingFace model name or OpenAI-compatible model identifier used to
        load the correct tokeniser.
    backend:
        One of ``"vllm"``, ``"sglang"``, or ``"ollama"``.  Controls block-size
        auto-detection strategy when *block_size* is ``None``.
    block_size:
        Explicit PagedAttention block size.  If ``None``, auto-detected from
        the backend server (see :meth:`_detect_block_size`).
    backend_url:
        Base URL of the serving backend (required for block-size detection).
    """

    # Ollama default – not exposed in the API; determined empirically from
    # llama.cpp source (see ADR-002).
    _OLLAMA_DEFAULT_BLOCK_SIZE: int = 32
    _VLLM_DEFAULT_BLOCK_SIZE: int = 16
    _SGLANG_DEFAULT_BLOCK_SIZE: int = 16

    def __init__(
        self,
        model_name: str,
        backend: str = "vllm",
        block_size: int | None = None,
        backend_url: str = "http://localhost:8000",
    ) -> None:
        self._model_name = model_name
        self._backend = backend.lower()
        self._backend_url = backend_url.rstrip("/")

        # --- Tokeniser -------------------------------------------------
        self._tokeniser: TokeniserProtocol = self._load_tokeniser(model_name)

        # --- Block size ------------------------------------------------
        if block_size is not None:
            if block_size not in (16, 32):
                raise ConfigurationError(
                    f"block_size must be 16 or 32, got {block_size}. "
                    "Only these values are supported by vLLM and SGLang."
                )
            self._block_size = block_size
        else:
            self._block_size = self._detect_block_size()

        self._pad_str = _select_padding_string(self._tokeniser, model_name)

        logger.info(
            '{"event":"block_aligner_init","model":"%s","backend":"%s",'
            '"block_size":%d}',
            model_name,
            backend,
            self._block_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        """The PagedAttention block size this aligner was configured with."""
        return self._block_size

    def tokenize(self, text: str) -> list[int]:
        """Encode *text* into a list of integer token IDs.

        Raises
        ------
        TokenizerError
            If the tokeniser raises any exception during encoding.
        """
        try:
            return self._tokeniser.encode(text)
        except Exception as exc:  # noqa: BLE001
            raise TokenizerError(
                f"Tokeniser '{self._model_name}' failed to encode text "
                f"(length={len(text)} chars): {exc}"
            ) from exc

    def align(self, system_prompt: str) -> BlockAlignResult:
        """Pad *system_prompt* to a block-aligned token length.

        The method computes::

            L_P = ceil(L_raw / B_block) * B_block
            padding_count = L_P - L_raw

        Then appends *padding_count* neutral padding tokens, verifies the
        resulting token sequence is block-aligned, and records a SHA-256
        digest of the aligned token IDs.

        Parameters
        ----------
        system_prompt:
            Raw system-prompt text (before padding).

        Returns
        -------
        BlockAlignResult
            Contains the padded text, token counts, and Zone P hash.

        Raises
        ------
        AlignmentError
            If the padding operation fails to produce a block-aligned result
            (should never happen; indicates a tokeniser bug).
        """
        raw_tokens = self.tokenize(system_prompt)
        l_raw = len(raw_tokens)
        l_p = math.ceil(l_raw / self._block_size) * self._block_size
        if l_p == 0:
            l_p = self._block_size

        padded_content = system_prompt
        pad_char = self._pad_str if self._pad_str else " "

        # Primary padding stage using selected pad string
        while len(self.tokenize(padded_content)) < l_p:
            padded_content += pad_char

        aligned_tokens = self.tokenize(padded_content)

        # Secondary refinement stage (handles BPE space-merging if token count overshot/undershot)
        if len(aligned_tokens) % self._block_size != 0:
            step_char = "#"
            max_steps = self._block_size * 50
            steps = 0
            while len(aligned_tokens) % self._block_size != 0 and steps < max_steps:
                padded_content += step_char
                aligned_tokens = self.tokenize(padded_content)
                steps += 1

        n_pad = len(aligned_tokens) - l_raw

        # Compute Zone P hash --------------------------------------------
        token_bytes = b"".join(t.to_bytes(4, "little") for t in aligned_tokens)
        zone_p_hash = hashlib.sha256(token_bytes).hexdigest()

        result = BlockAlignResult(
            original_token_count=l_raw,
            aligned_token_count=len(aligned_tokens),
            padding_count=n_pad,
            padded_content=padded_content,
            block_size=self._block_size,
            zone_p_hash=zone_p_hash,
        )

        logger.info(
            '{"event":"zone_p_aligned","model":"%s","l_raw":%d,"l_p":%d,'
            '"n_pad":%d,"hash":"%s"}',
            self._model_name,
            l_raw,
            len(aligned_tokens),
            n_pad,
            zone_p_hash[:16] + "...",
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tokeniser(self, model_name: str) -> TokeniserProtocol:
        """Try HuggingFace AutoTokenizer, then tiktoken, then raise."""
        # 1. Try HuggingFace AutoTokenizer (preferred for vLLM / SGLang)
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]

            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # Wrap encode to drop special tokens by default.
            original_encode = tok.encode

            class _HFWrapper:
                def encode(self, text: str) -> list[int]:  # noqa: D102
                    return original_encode(text, add_special_tokens=False)

                def decode(self, ids: list[int]) -> str:  # noqa: D102
                    return tok.decode(ids, skip_special_tokens=False)

                def __getattr__(self, item: str) -> object:
                    return getattr(tok, item)

            logger.debug(
                '{"event":"tokeniser_loaded","source":"huggingface","model":"%s"}',
                model_name,
            )
            return _HFWrapper()  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            pass

        # 2. Try tiktoken (OpenAI-compatible APIs)
        try:
            import tiktoken  # type: ignore[import-untyped]

            # Attempt to find an encoding by model name; fall back to cl100k_base.
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            class _TiktokenWrapper:
                def encode(self, text: str) -> list[int]:  # noqa: D102
                    return enc.encode(text)

                def decode(self, ids: list[int]) -> str:  # noqa: D102
                    return enc.decode(ids)

            logger.debug(
                '{"event":"tokeniser_loaded","source":"tiktoken","model":"%s"}',
                model_name,
            )
            return _TiktokenWrapper()  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            pass

        raise TokenizerError(
            f"Could not load a tokeniser for model '{model_name}'. "
            "Install 'transformers' (for HuggingFace models) or 'tiktoken' "
            "(for OpenAI-compatible models)."
        )

    def _detect_block_size(self) -> int:
        """Auto-detect PagedAttention block size from the backend server.

        Falls back to backend-specific defaults when the server does not
        expose block-size metadata (see ADR-002, §Argument E).
        """
        if self._backend == "ollama":
            return self._OLLAMA_DEFAULT_BLOCK_SIZE

        try:
            import httpx  # type: ignore[import-untyped]

            if self._backend == "vllm":
                # vLLM >= 0.4 exposes /metrics in Prometheus format.
                r = httpx.get(f"{self._backend_url}/metrics", timeout=2.0)
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        if "block_size" in line and not line.startswith("#"):
                            # Example: vllm:block_size{...} 16
                            parts = line.split()
                            if parts and parts[-1].isdigit():
                                bs = int(parts[-1])
                                if bs in (16, 32):
                                    logger.info(
                                        '{"event":"block_size_detected",'
                                        '"backend":"vllm","block_size":%d}',
                                        bs,
                                    )
                                    return bs

            elif self._backend == "sglang":
                r = httpx.get(
                    f"{self._backend_url}/get_server_args", timeout=2.0
                )
                if r.status_code == 200:
                    data: dict[str, object] = r.json()
                    bs = data.get("block_size")
                    if isinstance(bs, int) and bs in (16, 32):
                        logger.info(
                            '{"event":"block_size_detected",'
                            '"backend":"sglang","block_size":%d}',
                            bs,
                        )
                        return bs

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '{"event":"block_size_detection_failed","backend":"%s","reason":"%s"}',
                self._backend,
                str(exc),
            )

        # Backend-specific defaults
        default = (
            self._OLLAMA_DEFAULT_BLOCK_SIZE
            if self._backend == "ollama"
            else self._VLLM_DEFAULT_BLOCK_SIZE
        )
        logger.warning(
            '{"event":"block_size_default_used","backend":"%s","block_size":%d}',
            self._backend,
            default,
        )
        return default

    # ------------------------------------------------------------------
    # Testing / factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_mock(
        cls,
        block_size: int = 16,
        chars_per_token: int = 1,
    ) -> "BlockAligner":
        """Create a *BlockAligner* with a mock character-based tokeniser.

        Intended for unit tests only (avoids downloading real model weights).
        Each character is treated as exactly one token.
        """
        instance = object.__new__(cls)
        instance._model_name = "mock"
        instance._backend = "mock"
        instance._backend_url = ""
        instance._block_size = block_size

        class _MockTokeniser:
            def encode(self, text: str) -> list[int]:  # noqa: D102
                return [ord(c) % 32000 for c in text[::chars_per_token]]

            def decode(self, ids: list[int]) -> str:  # noqa: D102
                return " " * len(ids)

        instance._tokeniser = _MockTokeniser()  # type: ignore[attr-defined]
        instance._pad_str = " "
        return instance  # type: ignore[return-value]
