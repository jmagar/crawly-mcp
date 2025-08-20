"""
Simple tests for source models to maximize coverage.

This module focuses on testing the actual classes that exist in sources.py
to achieve high coverage on the models/sources.py module.
"""

from datetime import datetime, timedelta

import pytest

from crawler_mcp.models.sources import (
    SourceInfo,
    SourceMetadata,
    SourceType,
)


class TestSourceType:
    """Test SourceType enum."""

    def test_source_type_values(self):
        """Test all source type enum values."""
        assert SourceType.WEBPAGE == "webpage"
        assert SourceType.REPOSITORY == "repository"
        assert SourceType.DIRECTORY == "directory"
        assert SourceType.SITEMAP == "sitemap"
        assert SourceType.API == "api"
        assert SourceType.DOCUMENT == "document"

    def test_source_type_iteration(self):
        """Test iterating over source types."""
        types = list(SourceType)
        assert len(types) == 6
        assert SourceType.WEBPAGE in types
        assert SourceType.REPOSITORY in types
        assert SourceType.DIRECTORY in types
        assert SourceType.SITEMAP in types
        assert SourceType.API in types
        assert SourceType.DOCUMENT in types


class TestSourceMetadata:
    """Test SourceMetadata model."""

    def test_source_metadata_creation(self):
        """Test creating SourceMetadata with various fields."""
        now = datetime.utcnow()
        metadata = SourceMetadata(
            domain="example.com",
            language="en",
            content_type="text/html",
            last_modified=now,
            page_rank=0.85,
            word_count=1000,
            character_count=5000,
            link_count=50,
            image_count=10,
            tags=["python", "programming"],
            categories=["technical", "tutorial"],
            author="John Doe",
            publish_date=now,
            crawl_depth=2,
            response_time=0.5,
            http_status=200,
            file_size=10240,
            custom_fields={"priority": "high", "source": "manual"},
        )

        assert metadata.domain == "example.com"
        assert metadata.language == "en"
        assert metadata.content_type == "text/html"
        assert metadata.last_modified == now
        assert metadata.page_rank == 0.85
        assert metadata.word_count == 1000
        assert metadata.character_count == 5000
        assert metadata.link_count == 50
        assert metadata.image_count == 10
        assert metadata.tags == ["python", "programming"]
        assert metadata.categories == ["technical", "tutorial"]
        assert metadata.author == "John Doe"
        assert metadata.publish_date == now
        assert metadata.crawl_depth == 2
        assert metadata.response_time == 0.5
        assert metadata.http_status == 200
        assert metadata.file_size == 10240
        assert metadata.custom_fields == {"priority": "high", "source": "manual"}

    def test_source_metadata_defaults(self):
        """Test SourceMetadata with default values."""
        metadata = SourceMetadata()

        assert metadata.domain is None
        assert metadata.language is None
        assert metadata.content_type is None
        assert metadata.last_modified is None
        assert metadata.page_rank is None
        assert metadata.word_count == 0
        assert metadata.character_count == 0
        assert metadata.link_count == 0
        assert metadata.image_count == 0
        assert metadata.tags == []
        assert metadata.categories == []
        assert metadata.author is None
        assert metadata.publish_date is None
        assert metadata.crawl_depth == 0
        assert metadata.response_time is None
        assert metadata.http_status is None
        assert metadata.file_size is None
        assert metadata.custom_fields == {}

    def test_source_metadata_serialization(self):
        """Test SourceMetadata serialization."""
        metadata = SourceMetadata(
            domain="test.com",
            word_count=500,
            tags=["tag1", "tag2"],
            custom_fields={"key": "value"},
        )

        # Test model_dump
        data = metadata.model_dump()
        assert data["domain"] == "test.com"
        assert data["word_count"] == 500
        assert data["tags"] == ["tag1", "tag2"]
        assert data["custom_fields"] == {"key": "value"}

        # Test reconstruction
        new_metadata = SourceMetadata(**data)
        assert new_metadata.domain == metadata.domain
        assert new_metadata.word_count == metadata.word_count
        assert new_metadata.tags == metadata.tags
        assert new_metadata.custom_fields == metadata.custom_fields


class TestSourceInfo:
    """Test SourceInfo model."""

    def test_source_info_creation_full(self):
        """Test creating SourceInfo with all fields."""
        created_time = datetime.utcnow()
        updated_time = created_time + timedelta(hours=1)
        last_crawled = created_time + timedelta(minutes=30)

        metadata = SourceMetadata(domain="example.com", word_count=1000, tags=["test"])

        source = SourceInfo(
            id="source-123",
            url="https://example.com/page",
            title="Test Page",
            source_type=SourceType.WEBPAGE,
            chunk_count=5,
            total_content_length=5000,
            average_chunk_size=1000.0,
            created_at=created_time,
            updated_at=updated_time,
            last_crawled=last_crawled,
            metadata=metadata,
        )

        assert source.id == "source-123"
        assert source.url == "https://example.com/page"
        assert source.title == "Test Page"
        assert source.source_type == SourceType.WEBPAGE
        assert source.chunk_count == 5
        assert source.total_content_length == 5000
        assert source.avg_chunk_size == 1000.0
        assert source.created_at == created_time
        assert source.updated_at == updated_time
        assert source.last_crawled == last_crawled
        assert source.metadata == metadata

    def test_source_info_defaults(self):
        """Test SourceInfo with minimal required fields."""
        source = SourceInfo(
            id="min-source", url="https://minimal.com", source_type=SourceType.WEBPAGE
        )

        assert source.id == "min-source"
        assert source.url == "https://minimal.com"
        assert source.source_type == SourceType.WEBPAGE
        assert source.title is None
        assert source.chunk_count == 0
        assert source.total_content_length == 0
        assert source.avg_chunk_size == 0.0
        assert isinstance(source.metadata, SourceMetadata)

    def test_source_info_time_fields(self):
        """Test automatic time field population."""
        before_creation = datetime.utcnow()
        source = SourceInfo(
            id="time-test", url="https://time.com", source_type=SourceType.WEBPAGE
        )
        after_creation = datetime.utcnow()

        # Should have created_at and updated_at set automatically
        assert before_creation <= source.created_at <= after_creation
        assert before_creation <= source.updated_at <= after_creation
        # Times should be very close but may not be exactly equal due to microsecond differences
        time_diff = abs((source.created_at - source.updated_at).total_seconds())
        assert time_diff < 0.01  # Should be within 10ms

    def test_source_info_serialization(self):
        """Test SourceInfo serialization and deserialization."""
        original = SourceInfo(
            id="serialize-test",
            url="https://serialize.com",
            title="Serialize Test",
            source_type=SourceType.REPOSITORY,
            chunk_count=3,
            metadata=SourceMetadata(domain="serialize.com", word_count=100),
        )

        # Serialize
        data = original.model_dump()
        assert data["id"] == "serialize-test"
        assert data["source_type"] == "repository"

        # Deserialize
        reconstructed = SourceInfo(**data)
        assert reconstructed.id == original.id
        assert reconstructed.url == original.url
        assert reconstructed.source_type == original.source_type
        assert reconstructed.metadata.domain == original.metadata.domain

    def test_source_info_with_different_types(self):
        """Test SourceInfo with different source types."""
        # Test WEBPAGE
        webpage = SourceInfo(
            id="web-1", url="https://example.com", source_type=SourceType.WEBPAGE
        )
        assert webpage.source_type == SourceType.WEBPAGE

        # Test REPOSITORY
        repo = SourceInfo(
            id="repo-1",
            url="https://github.com/user/repo",
            source_type=SourceType.REPOSITORY,
        )
        assert repo.source_type == SourceType.REPOSITORY

        # Test DIRECTORY
        directory = SourceInfo(
            id="dir-1", url="file:///home/user/docs", source_type=SourceType.DIRECTORY
        )
        assert directory.source_type == SourceType.DIRECTORY

        # Test SITEMAP
        sitemap = SourceInfo(
            id="sitemap-1",
            url="https://example.com/sitemap.xml",
            source_type=SourceType.SITEMAP,
        )
        assert sitemap.source_type == SourceType.SITEMAP

        # Test API
        api = SourceInfo(
            id="api-1", url="https://api.example.com/data", source_type=SourceType.API
        )
        assert api.source_type == SourceType.API

        # Test DOCUMENT
        document = SourceInfo(
            id="doc-1",
            url="https://example.com/document.pdf",
            source_type=SourceType.DOCUMENT,
        )
        assert document.source_type == SourceType.DOCUMENT

    def test_source_info_complex_metadata(self):
        """Test SourceInfo with complex metadata."""
        complex_metadata = SourceMetadata(
            domain="complex.com",
            language="en-US",
            content_type="application/json",
            word_count=2500,
            character_count=15000,
            link_count=75,
            image_count=25,
            tags=["api", "json", "rest", "v2"],
            categories=["technical", "documentation", "api"],
            author="API Team",
            crawl_depth=3,
            response_time=1.2,
            http_status=200,
            file_size=51200,
            custom_fields={
                "api_version": "v2",
                "rate_limit": 1000,
                "authentication": "bearer",
                "endpoints": ["users", "posts", "comments"],
            },
        )

        source = SourceInfo(
            id="complex-source",
            url="https://api.complex.com/v2",
            title="Complex API Documentation",
            source_type=SourceType.API,
            chunk_count=15,
            total_content_length=15000,
            average_chunk_size=1000.0,
            metadata=complex_metadata,
        )

        assert source.metadata.domain == "complex.com"
        assert source.metadata.language == "en-US"
        assert source.metadata.content_type == "application/json"
        assert len(source.metadata.tags) == 4
        assert len(source.metadata.categories) == 3
        assert source.metadata.custom_fields["api_version"] == "v2"
        assert len(source.metadata.custom_fields["endpoints"]) == 3

    def test_source_info_edge_cases(self):
        """Test SourceInfo edge cases and boundary conditions."""
        # Test with zero values
        zero_source = SourceInfo(
            id="zero-source",
            url="https://zero.com",
            source_type=SourceType.WEBPAGE,
            chunk_count=0,
            total_content_length=0,
            average_chunk_size=0.0,
        )
        assert zero_source.chunk_count == 0
        assert zero_source.total_content_length == 0
        assert zero_source.avg_chunk_size == 0.0

        # Test with large values
        large_source = SourceInfo(
            id="large-source",
            url="https://large.com",
            source_type=SourceType.REPOSITORY,
            chunk_count=10000,
            total_content_length=50000000,
            average_chunk_size=5000.0,
        )
        assert large_source.chunk_count == 10000
        assert large_source.total_content_length == 50000000
        assert large_source.avg_chunk_size == 5000.0

        # Test with very long strings
        long_title = "A" * 1000
        long_url = "https://example.com/" + "path/" * 100 + "file.html"

        long_source = SourceInfo(
            id="long-source",
            url=long_url,
            title=long_title,
            source_type=SourceType.DOCUMENT,
        )
        assert len(long_source.title) == 1000
        assert long_source.url.startswith("https://example.com/")
        assert long_source.url.endswith("file.html")

    def test_source_info_model_validation(self):
        """Test Pydantic model validation."""
        # Test that required fields are enforced
        with pytest.raises((ValueError, TypeError)):
            SourceInfo()  # Missing required fields

        with pytest.raises((ValueError, TypeError)):
            SourceInfo(id="test")  # Missing url and source_type

        with pytest.raises((ValueError, TypeError)):
            SourceInfo(id="test", url="https://test.com")  # Missing source_type

        # Test that valid minimal instance works
        valid_source = SourceInfo(
            id="valid", url="https://valid.com", source_type=SourceType.WEBPAGE
        )
        assert valid_source.id == "valid"
