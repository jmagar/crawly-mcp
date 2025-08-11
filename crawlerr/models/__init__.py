"""
Data models for Crawlerr using Pydantic.
"""
from .crawl_models import (
    CrawlResult,
    CrawlRequest,
    CrawlStatus,
    PageContent,
    CrawlStatistics,
)
from .rag_models import (
    RagQuery,
    RagResult,
    SearchMatch,
    DocumentChunk,
    EmbeddingResult,
)
from .source_models import (
    SourceInfo,
    SourceType,
    SourceFilter,
    SourceMetadata,
)

__all__ = [
    # Crawl models
    "CrawlResult",
    "CrawlRequest", 
    "CrawlStatus",
    "PageContent",
    "CrawlStatistics",
    # RAG models
    "RagQuery",
    "RagResult",
    "SearchMatch",
    "DocumentChunk",
    "EmbeddingResult",
    # Source models
    "SourceInfo",
    "SourceType",
    "SourceFilter",
    "SourceMetadata",
]