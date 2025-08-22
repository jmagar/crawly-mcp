"""
Test Pydantic model classes for validation and serialization.
"""

from datetime import datetime

import pytest

from crawler_mcp.models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from crawler_mcp.models.sources import SourceInfo, SourceMetadata, SourceType


class TestCrawlModels:
    """Test crawling-related Pydantic models."""

    @pytest.mark.unit
    def test_crawl_status_enum(self):
        """Test CrawlStatus enum values."""
        assert CrawlStatus.PENDING == "pending"
        assert CrawlStatus.RUNNING == "running"
        assert CrawlStatus.COMPLETED == "completed"
        assert CrawlStatus.FAILED == "failed"
        assert CrawlStatus.CANCELLED == "cancelled"

    @pytest.mark.unit
    def test_page_content_creation(self):
        """Test PageContent model creation."""
        content = PageContent(
            url="https://example.com",
            title="Test Page",
            content="This is test content for the page.",
            word_count=7,
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert content.content == "This is test content for the page."
        assert content.word_count == 7
        assert isinstance(content.timestamp, datetime)

    @pytest.mark.unit
    def test_page_content_word_count_validator(self):
        """Test automatic word count calculation."""
        content = PageContent(
            url="https://example.com",
            content="This is a test sentence with eight words.",
        )

        # Should calculate word count automatically if not provided
        assert content.word_count == 8

    @pytest.mark.unit
    def test_crawl_request_validation(self):
        """Test CrawlRequest model validation."""
        # Single URL
        request = CrawlRequest(url="https://example.com", max_pages=10, max_depth=2)

        assert request.url == ["https://example.com"]  # Converted to list
        assert request.max_pages == 10
        assert request.max_depth == 2

        # Multiple URLs
        request_multi = CrawlRequest(url=["https://example.com", "https://test.com"])
        assert len(request_multi.url) == 2

    @pytest.mark.unit
    def test_crawl_request_defaults(self):
        """Test CrawlRequest default values."""
        request = CrawlRequest(url="https://example.com")

        assert request.max_pages == 100
        assert request.max_depth == 3
        assert request.remove_overlay_elements is True
        assert request.extract_media is False

    @pytest.mark.unit
    def test_crawl_statistics_creation(self):
        """Test CrawlStatistics model creation."""
        stats = CrawlStatistics(
            total_pages_requested=100,
            total_pages_crawled=85,
            total_pages_failed=15,
            crawl_duration_seconds=120.5,
        )

        assert stats.total_pages_requested == 100
        assert stats.total_pages_crawled == 85
        assert stats.total_pages_failed == 15
        assert stats.crawl_duration_seconds == 120.5

    @pytest.mark.unit
    def test_crawl_result_creation(self):
        """Test CrawlResult model creation."""
        pages = [
            PageContent(url="https://example.com/page1", content="Content 1"),
            PageContent(url="https://example.com/page2", content="Content 2"),
        ]

        result = CrawlResult(
            request_id="test_123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com"],
            pages=pages,
        )

        assert result.request_id == "test_123"
        assert result.status == CrawlStatus.COMPLETED
        assert len(result.pages) == 2
        assert isinstance(result.start_time, datetime)

    @pytest.mark.unit
    def test_crawl_statistics_attempted_pages(self):
        """Test CrawlStatistics attempted_pages property."""
        stats = CrawlStatistics(total_pages_crawled=80, total_pages_failed=20)
        assert stats.attempted_pages == 100

        # Test with zero values
        empty_stats = CrawlStatistics()
        assert empty_stats.attempted_pages == 0

    @pytest.mark.unit
    def test_crawl_result_success_rate(self):
        """Test CrawlResult success_rate property."""
        stats = CrawlStatistics(total_pages_crawled=80, total_pages_failed=20)

        result = CrawlResult(
            request_id="test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com"],
            statistics=stats,
        )

        # 80 successful out of 100 total = 80%
        assert result.success_rate == 80.0

    @pytest.mark.unit
    def test_crawl_result_is_complete(self):
        """Test CrawlResult is_complete property."""
        result_complete = CrawlResult(
            request_id="test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com"],
        )
        assert result_complete.is_complete is True

        result_running = CrawlResult(
            request_id="test", status=CrawlStatus.RUNNING, urls=["https://example.com"]
        )
        assert result_running.is_complete is False


class TestSourceModels:
    """Test source-related Pydantic models."""

    @pytest.mark.unit
    def test_source_type_enum(self):
        """Test SourceType enum values."""
        assert SourceType.WEBPAGE == "webpage"
        assert SourceType.REPOSITORY == "repository"
        assert SourceType.DIRECTORY == "directory"
        assert SourceType.SITEMAP == "sitemap"
        assert SourceType.API == "api"
        assert SourceType.DOCUMENT == "document"

    @pytest.mark.unit
    def test_source_metadata_creation(self):
        """Test SourceMetadata model creation."""
        metadata = SourceMetadata(
            domain="example.com",
            language="en",
            word_count=1500,
            character_count=8000,
            link_count=25,
            image_count=5,
            tags=["technology", "programming"],
            author="John Doe",
        )

        assert metadata.domain == "example.com"
        assert metadata.language == "en"
        assert metadata.word_count == 1500
        assert metadata.character_count == 8000
        assert len(metadata.tags) == 2
        assert metadata.author == "John Doe"

    @pytest.mark.unit
    def test_source_metadata_defaults(self):
        """Test SourceMetadata default values."""
        metadata = SourceMetadata()

        assert metadata.domain is None
        assert metadata.word_count == 0
        assert metadata.character_count == 0
        assert metadata.tags == []
        assert metadata.categories == []
        assert metadata.custom_fields == {}

    @pytest.mark.unit
    def test_source_info_creation(self):
        """Test SourceInfo model creation."""
        metadata = SourceMetadata(domain="test.com", word_count=500)

        source = SourceInfo(
            source_id="test_source_123",
            source_type=SourceType.WEBPAGE,
            url="https://test.com/page",
            title="Test Page",
            metadata=metadata,
        )

        assert source.source_id == "test_source_123"
        assert source.source_type == SourceType.WEBPAGE
        assert source.url == "https://test.com/page"
        assert source.title == "Test Page"
        assert source.metadata.word_count == 500

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test that models can be serialized to dict."""
        request = CrawlRequest(url="https://example.com", max_pages=50)

        data = request.model_dump()
        assert isinstance(data, dict)
        assert data["url"] == ["https://example.com"]
        assert data["max_pages"] == 50

    @pytest.mark.unit
    def test_model_json_serialization(self):
        """Test JSON serialization of models."""
        stats = CrawlStatistics(total_pages_crawled=10, crawl_duration_seconds=30.5)

        json_str = stats.model_dump_json()
        assert isinstance(json_str, str)
        assert "10" in json_str
        assert "30.5" in json_str

    @pytest.mark.unit
    def test_field_validation(self):
        """Test field validation works."""
        # Valid max_pages
        request = CrawlRequest(url="https://example.com", max_pages=1)
        assert request.max_pages == 1

        # Invalid max_pages (too high)
        with pytest.raises(ValueError):
            CrawlRequest(url="https://example.com", max_pages=2000)
