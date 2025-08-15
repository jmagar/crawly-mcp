"""
Data models for crawler_mcp using Pydantic.
"""

from .crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .rag import (
    DocumentChunk,
    EmbeddingResult,
    RagQuery,
    RagResult,
    SearchMatch,
)
from .sources import (
    SourceFilter,
    SourceInfo,
    SourceMetadata,
    SourceType,
)

__all__ = [
    "CrawlRequest",
    "CrawlResult",
    "CrawlStatistics",
    "CrawlStatus",
    "DocumentChunk",
    "EmbeddingResult",
    "PageContent",
    "RagQuery",
    "RagResult",
    "SearchMatch",
    "SourceFilter",
    "SourceInfo",
    "SourceMetadata",
    "SourceType",
]
