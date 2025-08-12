"""
Data models for Crawlerr using Pydantic.
"""

from .crawl_models import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .rag_models import (
    DocumentChunk,
    EmbeddingResult,
    RagQuery,
    RagResult,
    SearchMatch,
)
from .source_models import (
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
