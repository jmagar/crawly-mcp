"""
Comprehensive tests for source models to maximize coverage.

This module focuses on testing all code paths in the source models
to achieve high coverage on the models/sources.py module.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

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
        metadata = SourceMetadata(
            domain="example.com",
            word_count=1000,
            character_count=5000,
            link_count=50,
            image_count=10,
            language="en",
            content_type="text/html",
            tags=["python", "programming"],
            categories=["technical", "tutorial"],
        )

        assert metadata.domain == "example.com"
        assert metadata.word_count == 1000
        assert metadata.character_count == 5000
        assert metadata.link_count == 50
        assert metadata.image_count == 10
        assert metadata.language == "en"
        assert metadata.content_type == "text/html"
        assert metadata.tags == ["python", "programming"]
        assert metadata.categories == ["technical", "tutorial"]

    def test_source_metadata_defaults(self):
        """Test SourceMetadata with default values."""
        metadata = SourceMetadata()

        assert metadata.domain is None
        assert metadata.word_count == 0
        assert metadata.character_count == 0
        assert metadata.link_count == 0
        assert metadata.image_count == 0
        assert metadata.language is None
        assert metadata.content_type is None
        assert metadata.tags == []
        assert metadata.categories == []

    def test_source_metadata_serialization(self):
        """Test SourceMetadata serialization."""
        metadata = SourceMetadata(
            domain="test.com", word_count=500, tags=["tag1", "tag2"]
        )

        # Test model_dump
        data = metadata.model_dump()
        assert data["domain"] == "test.com"
        assert data["word_count"] == 500
        assert data["tags"] == ["tag1", "tag2"]

        # Test reconstruction
        new_metadata = SourceMetadata(**data)
        assert new_metadata.domain == metadata.domain
        assert new_metadata.word_count == metadata.word_count
        assert new_metadata.tags == metadata.tags


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
            status="active",
            chunk_count=5,
            total_content_length=5000,
            created_at=created_time,
            updated_at=updated_time,
            last_crawled=last_crawled,
            metadata=metadata,
        )

        assert source.id == "source-123"
        assert source.url == "https://example.com/page"
        assert source.title == "Test Page"
        assert source.source_type == SourceType.WEBPAGE
        assert source.status == "active"
        assert source.chunk_count == 5
        assert source.total_content_length == 5000
        # Test computed property: average chunk size should be total_content_length / chunk_count
        assert source.avg_chunk_size == 1000.0  # 5000 / 5 = 1000.0
        assert source.created_at == created_time
        assert source.updated_at == updated_time
        assert source.last_crawled == last_crawled
        # Test computed property: is_stale should be False since last_crawled is recent
        assert source.is_stale is False
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
        assert source.status == "active"  # Default
        assert source.chunk_count == 0
        assert source.total_content_length == 0
        assert source.avg_chunk_size == 0.0
        assert source.is_stale is False
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
        assert source.created_at == source.updated_at  # Should be same initially

    def test_source_info_serialization(self):
        """Test SourceInfo serialization and deserialization."""
        original = SourceInfo(
            id="serialize-test",
            url="https://serialize.com",
            title="Serialize Test",
            source_type=SourceType.REPOSITORY,
            status="pending",
            chunk_count=3,
            metadata=SourceMetadata(domain="serialize.com", word_count=100),
        )

        # Serialize
        data = original.model_dump()
        assert data["id"] == "serialize-test"
        assert data["source_type"] == "repository"
        assert data["status"] == "pending"

        # Deserialize
        reconstructed = SourceInfo(**data)
        assert reconstructed.id == original.id
        assert reconstructed.url == original.url
        assert reconstructed.source_type == original.source_type
        assert reconstructed.status == original.status
        assert reconstructed.metadata.domain == original.metadata.domain

    def test_source_info_validation(self):
        """Test SourceInfo field validation."""
        # Test valid chunk_count (non-negative)
        source = SourceInfo(
            id="valid",
            url="https://valid.com",
            source_type=SourceType.WEBPAGE,
            chunk_count=10,
        )
        assert source.chunk_count == 10

        # Test chunk_count validation (should reject negative values)
        with pytest.raises(ValidationError):
            SourceInfo(
                id="invalid",
                url="https://invalid.com",
                source_type=SourceType.WEBPAGE,
                chunk_count=-1,
            )
