"""
nexus_context.cache
===================
PagedAttention block alignment, three-zone context segmentation, and the
FastAPI transparent proxy middleware.

Public API
----------
BlockAligner      – pads Zone P to a B_block boundary and freezes its hash
ZoneSegmenter     – partitions messages into Zones P / T / R
"""

from nexus_context.cache.schemas import (
    BlockAlignResult,
    CacheBoundary,
    ChatMessage,
    ZoneBundle,
    ZoneType,
)

__all__ = [
    "BlockAlignResult",
    "CacheBoundary",
    "ChatMessage",
    "ZoneBundle",
    "ZoneType",
]
