"""
Data models for source management and filtering.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """Type of crawled source."""

    WEBPAGE = "webpage"
    SITEMAP = "sitemap"
    REPOSITORY = "repository"
    DIRECTORY = "directory"
    API = "api"
    DOCUMENT = "document"


class SourceMetadata(BaseModel):
    model_config = ConfigDict()

    """Metadata for a crawled source."""
    domain: str | None = None
    language: str | None = None
    content_type: str | None = None
    last_modified: datetime | None = None
    page_rank: float | None = None
    word_count: int = 0
    character_count: int = 0
    link_count: int = 0
    image_count: int = 0
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    author: str | None = None
    publish_date: datetime | None = None
    crawl_depth: int = 0
    response_time: float | None = None
    http_status: int | None = None
    file_size: int | None = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    model_config = ConfigDict()

    """Information about a crawled source."""
    id: str
    url: str
    title: str | None = None
    source_type: SourceType
    status: str = "active"  # active, inactive, error
    chunk_count: int = 0
    total_content_length: int = 0
    metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_crawled: datetime | None = None
    crawl_frequency: str | None = None  # daily, weekly, monthly, never

    # Deduplication fields
    content_hash: str | None = None  # Overall source content hash
    chunk_hashes: dict[str, str] = Field(
        default_factory=dict
    )  # Map of chunk_id → content_hash
    last_crawl_status: str | None = None  # "unchanged", "modified", "new", "error"

    # Incremental update tracking
    etag: str | None = None  # HTTP ETag header from last crawl
    last_modified_header: datetime | None = None  # HTTP Last-Modified header
    cache_control: str | None = None  # HTTP Cache-Control header
    should_revalidate: bool = (
        True  # Whether content should be revalidated on next crawl
    )

    @property
    def is_stale(self) -> bool:
        """Check if source needs re-crawling based on frequency."""
        if not self.last_crawled or not self.crawl_frequency:
            return False

        now = datetime.utcnow()
        delta = now - self.last_crawled

        if self.crawl_frequency == "daily":
            return delta.days >= 1
        elif self.crawl_frequency == "weekly":
            return delta.days >= 7
        elif self.crawl_frequency == "monthly":
            return delta.days >= 30

        return False

    @property
    def avg_chunk_size(self) -> float:
        """Calculate average chunk size."""
        if self.chunk_count == 0:
            return 0.0
        return self.total_content_length / self.chunk_count

    def should_skip_crawl(
        self,
        response_etag: str | None = None,
        response_last_modified: datetime | None = None,
    ) -> bool:
        """
        Check if crawling should be skipped based on HTTP caching headers.

        Args:
            response_etag: ETag from current HTTP response
            response_last_modified: Last-Modified from current HTTP response

        Returns:
            True if content likely hasn't changed and crawl can be skipped
        """
        # If we don't have caching info, always crawl
        if not self.etag and not self.last_modified_header:
            return False

        # If response doesn't provide caching headers, always crawl
        if not response_etag and not response_last_modified:
            return False

        # Check ETag match (most reliable)
        if self.etag and response_etag:
            return self.etag == response_etag

        # Check Last-Modified (less reliable but useful)
        if self.last_modified_header and response_last_modified:
            return self.last_modified_header >= response_last_modified

        return False

    def update_cache_headers(
        self,
        etag: str | None = None,
        last_modified: datetime | None = None,
        cache_control: str | None = None,
    ) -> None:
        """Update HTTP caching headers from crawl response."""
        self.etag = etag
        self.last_modified_header = last_modified
        self.cache_control = cache_control
        self.should_revalidate = True  # Reset revalidation flag


class SourceFilter(BaseModel):
    model_config = ConfigDict()

    """Filter criteria for sources."""
    source_types: list[SourceType] | None = None
    domains: list[str] | None = None
    statuses: list[str] | None = None
    date_range: tuple[datetime, datetime] | None = None
    min_word_count: int | None = None
    max_word_count: int | None = None
    tags: list[str] | None = None
    categories: list[str] | None = None
    languages: list[str] | None = None
    search_term: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

    def matches_source(self, source: SourceInfo) -> bool:
        """Check if a source matches the filter criteria."""
        # Type filter
        if self.source_types and source.source_type not in self.source_types:
            return False

        # Domain filter
        if (
            self.domains
            and source.metadata.domain
            and not any(domain in source.metadata.domain for domain in self.domains)
        ):
            return False

        # Status filter
        if self.statuses and source.status not in self.statuses:
            return False

        # Date range filter
        if self.date_range:
            start, end = self.date_range
            if not (start <= source.created_at <= end):
                return False

        # Word count filter
        if self.min_word_count and source.metadata.word_count < self.min_word_count:
            return False
        if self.max_word_count and source.metadata.word_count > self.max_word_count:
            return False

        # Tags filter
        if self.tags and not any(tag in source.metadata.tags for tag in self.tags):
            return False

        # Categories filter
        if self.categories and not any(
            cat in source.metadata.categories for cat in self.categories
        ):
            return False

        # Language filter
        if (
            self.languages
            and source.metadata.language
            and source.metadata.language not in self.languages
        ):
            return False

        # Search term filter (basic text search)
        if self.search_term:
            search_lower = self.search_term.lower()
            title = (source.title or "").lower()
            url = source.url.lower()
            if search_lower not in title and search_lower not in url:
                return False

        return True
