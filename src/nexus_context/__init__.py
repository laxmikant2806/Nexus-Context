"""
nexus-context
=============
Lightweight Python framework for referential integrity, KV cache alignment,
and WWW memory governance in local SLM deployments (vLLM, SGLang, Ollama).

Subpackages
-----------
nexus_context.guard   – AST dependency graph + submodular pruning
nexus_context.cache   – PagedAttention block alignment + zone segmentation
nexus_context.memory  – WWW episodic-to-semantic memory decay
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Nexus-Context Team"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Top-level error hierarchy (see docs/implementation_stage.md §7.3)
# ---------------------------------------------------------------------------


class NexusContextError(Exception):
    """Base exception for all nexus-context errors."""


class TokenizerError(NexusContextError):
    """Tokenizer load or encode failure."""


class AlignmentError(NexusContextError):
    """Block alignment computation failure."""


class SegmentationError(NexusContextError):
    """Zone segmentation invariant violation."""


class GraphBuildError(NexusContextError):
    """Context dependency graph construction failure."""


class SolverError(NexusContextError):
    """Submodular solver failure (budget too small, empty graph, etc.)."""


class MemoryExtractionError(NexusContextError):
    """WWW tuple extraction failure."""


class BackendProxyError(NexusContextError):
    """Backend server communication failure."""


class ConfigurationError(NexusContextError):
    """Invalid or missing configuration value."""


__all__ = [
    "__version__",
    "NexusContextError",
    "TokenizerError",
    "AlignmentError",
    "SegmentationError",
    "GraphBuildError",
    "SolverError",
    "MemoryExtractionError",
    "BackendProxyError",
    "ConfigurationError",
]
