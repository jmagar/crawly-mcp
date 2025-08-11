"""
Data models for web crawling operations.
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict


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
    title: Optional[str] = None
    content: str
    markdown: Optional[str] = None
    html: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    word_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("word_count", mode="before")
    @classmethod
    def calculate_word_count(cls, v, info):
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content:
                return len(content.split())
        return v


class CrawlRequest(BaseModel):
    """Request for crawling operation."""
    model_config = ConfigDict()
    
    url: Union[str, List[str]]
    max_pages: Optional[int] = Field(default=100, ge=1, le=1000)
    max_depth: Optional[int] = Field(default=3, ge=1, le=10)
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    extraction_strategy: str = Field(default="css")
    wait_for: Optional[str] = None
    remove_overlay_elements: bool = True
    extract_media: bool = False
    include_raw_html: bool = False
    session_id: Optional[str] = None
    
    @field_validator("url")
    @classmethod
    def validate_urls(cls, v):
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
    error_counts: Dict[str, int] = Field(default_factory=dict)


class CrawlResult(BaseModel):
    model_config = ConfigDict()

    """Result of a crawl operation."""
    request_id: str
    status: CrawlStatus
    urls: List[str]
    pages: List[PageContent] = Field(default_factory=list)
    statistics: CrawlStatistics = Field(default_factory=CrawlStatistics)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
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
        return self.status in [CrawlStatus.COMPLETED, CrawlStatus.FAILED, CrawlStatus.CANCELLED]