"""
Data models for web crawling operations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CrawlStatus(str, Enum):
    """Status of a crawl operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PageContent(BaseModel):
    """Content extracted from a single page."""

    model_config = ConfigDict()

    url: str
    title: str | None = None
    content: str
    markdown: str | None = None
    html: str | None = None
    links: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    word_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("word_count", mode="before")
    @classmethod
    def calculate_word_count(cls, v: int, info: Any) -> int:
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content:
                return len(content.split())
        return v


class CrawlRequest(BaseModel):
    """Request for crawling operation."""

    model_config = ConfigDict()

    url: str | list[str]
    max_pages: int | None = Field(default=100, ge=1, le=1000)
    max_depth: int | None = Field(default=3, ge=1, le=10)
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    extraction_strategy: str = Field(default="css")
    wait_for: str | None = None
    remove_overlay_elements: bool = True
    extract_media: bool = False
    include_raw_html: bool = False
    session_id: str | None = None

    @field_validator("url")
    @classmethod
    def validate_urls(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class CrawlStatistics(BaseModel):
    model_config = ConfigDict()

    """Statistics for a crawl operation."""
    total_pages_requested: int = 0
    total_pages_crawled: int = 0
    total_pages_failed: int = 0
    total_bytes_downloaded: int = 0
    average_page_size: float = 0.0
    crawl_duration_seconds: float = 0.0
    pages_per_second: float = 0.0
    unique_domains: int = 0
    total_links_discovered: int = 0
    total_images_found: int = 0
    error_counts: dict[str, int] = Field(default_factory=dict)


class CrawlResult(BaseModel):
    model_config = ConfigDict()

    """Result of a crawl operation."""
    request_id: str
    status: CrawlStatus
    urls: list[str]
    pages: list[PageContent] = Field(default_factory=list)
    statistics: CrawlStatistics = Field(default_factory=CrawlStatistics)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.statistics.total_pages_requested
        if total == 0:
            return 0.0
        return (self.statistics.total_pages_crawled / total) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if crawl is complete."""
        return self.status in [
            CrawlStatus.COMPLETED,
            CrawlStatus.FAILED,
            CrawlStatus.CANCELLED,
        ]
