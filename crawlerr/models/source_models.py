"""
Data models for source management and filtering.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


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
    domain: Optional[str] = None
    language: Optional[str] = None
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    page_rank: Optional[float] = None
    word_count: int = 0
    character_count: int = 0
    link_count: int = 0
    image_count: int = 0
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    crawl_depth: int = 0
    response_time: Optional[float] = None
    http_status: Optional[int] = None
    file_size: Optional[int] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    model_config = ConfigDict()

    """Information about a crawled source."""
    id: str
    url: str
    title: Optional[str] = None
    source_type: SourceType
    status: str = "active"  # active, inactive, error
    chunk_count: int = 0
    total_content_length: int = 0
    metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_crawled: Optional[datetime] = None
    crawl_frequency: Optional[str] = None  # daily, weekly, monthly, never
    
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


class SourceFilter(BaseModel):
    model_config = ConfigDict()

    """Filter criteria for sources."""
    source_types: Optional[List[SourceType]] = None
    domains: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    search_term: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    def matches_source(self, source: SourceInfo) -> bool:
        """Check if a source matches the filter criteria."""
        # Type filter
        if self.source_types and source.source_type not in self.source_types:
            return False
        
        # Domain filter
        if self.domains and source.metadata.domain:
            if not any(domain in source.metadata.domain for domain in self.domains):
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
        if self.tags:
            if not any(tag in source.metadata.tags for tag in self.tags):
                return False
        
        # Categories filter  
        if self.categories:
            if not any(cat in source.metadata.categories for cat in self.categories):
                return False
        
        # Language filter
        if self.languages and source.metadata.language:
            if source.metadata.language not in self.languages:
                return False
        
        # Search term filter (basic text search)
        if self.search_term:
            search_lower = self.search_term.lower()
            title = (source.title or "").lower()
            url = source.url.lower()
            if search_lower not in title and search_lower not in url:
                return False
        
        return True