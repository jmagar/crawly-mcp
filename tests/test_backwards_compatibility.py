"""
Tests for backwards compatibility with mixed ID types.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from crawler_mcp.core.rag import RagService
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent


class TestBackwardsCompatibility:
    """Test cases for backwards compatibility with mixed ID types."""

    def test_is_random_uuid(self):
        """Test random UUID detection."""
        rag_service = RagService()

        # Test random UUIDs
        random_uuids = [
            "550e8400e29b41d4a716446655440000",  # 32 hex chars
            "550e8400-e29b-41d4-a716-446655440000",  # Standard UUID format
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # Another valid UUID
            "abcdef1234567890abcdef1234567890abcd",  # 34 hex chars
        ]

        for uuid_str in random_uuids:
            assert rag_service._is_random_uuid(uuid_str), (
                f"Should detect {uuid_str} as random UUID"
            )

        # Test deterministic IDs (should NOT be detected as random)
        deterministic_ids = [
            "1234567890123456",  # 16 hex chars (deterministic format)
            "abcdef1234567890",  # 16 hex chars
            "short123",  # Too short
            "",  # Empty string
        ]

        for det_id in deterministic_ids:
            assert not rag_service._is_random_uuid(det_id), (
                f"Should NOT detect {det_id} as random UUID"
            )

    def test_should_use_backwards_compatibility(self):
        """Test backwards compatibility detection logic."""
        rag_service = RagService()

        # Test with mostly random UUIDs (should enable backwards compatibility)
        random_chunks_map = {
            "550e8400e29b41d4a716446655440000": "hash1",
            "550e8400-e29b-41d4-a716-446655440001": "hash2",
            "550e8400-e29b-41d4-a716-446655440002": "hash3",
            "1234567890123456": "hash4",  # One deterministic ID
        }

        assert rag_service._should_use_backwards_compatibility(random_chunks_map), (
            "Should enable backwards compatibility for mostly random UUIDs"
        )

        # Test with mostly deterministic IDs (should NOT enable backwards compatibility)
        deterministic_chunks_map = {
            "1234567890123456": "hash1",
            "abcdef1234567890": "hash2",
            "fedcba0987654321": "hash3",
            "550e8400e29b41d4a716446655440000": "hash4",  # One random UUID
        }

        assert not rag_service._should_use_backwards_compatibility(
            deterministic_chunks_map
        ), "Should NOT enable backwards compatibility for mostly deterministic IDs"

        # Test with empty map
        assert not rag_service._should_use_backwards_compatibility({}), (
            "Should NOT enable backwards compatibility for empty map"
        )

    def test_find_legacy_chunk_by_content(self):
        """Test legacy chunk finding by content."""
        rag_service = RagService()

        test_content = "This is test content"
        content_hash = rag_service._calculate_content_hash(test_content)

        existing_chunks = [
            {
                "id": "550e8400e29b41d4a716446655440000",
                "content": test_content,
                "content_hash": content_hash,
            },
            {
                "id": "550e8400e29b41d4a716446655440001",
                "content": "Different content",
                "content_hash": rag_service._calculate_content_hash(
                    "Different content"
                ),
            },
        ]

        # Should find chunk by content hash
        found_chunk = rag_service._find_legacy_chunk_by_content(
            existing_chunks, test_content
        )
        assert found_chunk is not None, "Should find chunk with matching content"
        assert found_chunk["id"] == "550e8400e29b41d4a716446655440000", (
            "Should find correct chunk"
        )

        # Should not find chunk with non-matching content
        not_found = rag_service._find_legacy_chunk_by_content(
            existing_chunks, "Non-existent content"
        )
        assert not_found is None, "Should not find chunk with non-matching content"

        # Test fallback to direct content comparison (for chunks without content_hash)
        chunks_without_hash = [
            {
                "id": "legacy-chunk-1",
                "content": test_content,
                # No content_hash field
            }
        ]

        fallback_found = rag_service._find_legacy_chunk_by_content(
            chunks_without_hash, test_content
        )
        assert fallback_found is not None, (
            "Should find chunk by direct content comparison"
        )
        assert fallback_found["id"] == "legacy-chunk-1", (
            "Should find correct chunk via fallback"
        )

    @pytest.mark.asyncio
    async def test_backwards_compatibility_integration(self):
        """Test full backwards compatibility flow."""
        rag_service = RagService()

        # Mock services
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()

        # Setup: existing chunks with random UUIDs
        test_content = "Test content for page 0" + " " * 100
        content_hash = rag_service._calculate_content_hash(test_content)

        existing_chunks = [
            {
                "id": "550e8400e29b41d4a716446655440000",  # Random UUID
                "content": test_content,
                "content_hash": content_hash,
            },
            {
                "id": "550e8400e29b41d4a716446655440001",  # Another random UUID
                "content": "Different content",
                "content_hash": rag_service._calculate_content_hash(
                    "Different content"
                ),
            },
        ]

        rag_service.vector_service.get_chunks_by_source.return_value = existing_chunks
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        rag_service.vector_service.upsert_documents.return_value = 1
        rag_service.vector_service.delete_chunks_by_ids.return_value = 1

        # Create crawl result with same content (should skip but upgrade ID)
        crawl_result = CrawlResult(
            request_id="test-backwards-compat",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content=test_content,  # Same content as legacy chunk
                    word_count=len(test_content.split()),
                    metadata={"chunk_metadata": {"chunk_index": 0}},
                )
            ],
        )

        # Process with deduplication enabled
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )

        # Should skip the chunk (same content) but delete the legacy chunk
        assert result["chunks_created"] == 0, (
            "Should not create new chunk for same content"
        )
        assert result["chunks_skipped"] == 1, "Should skip chunk with same content"
        assert result["chunks_deleted"] == 1, (
            "Should delete legacy chunk with random UUID"
        )

        # Verify delete was called with the legacy chunk ID
        rag_service.vector_service.delete_chunks_by_ids.assert_called_once()
        deleted_ids = rag_service.vector_service.delete_chunks_by_ids.call_args[0][0]
        assert "550e8400e29b41d4a716446655440000" in deleted_ids, (
            "Should delete legacy chunk ID"
        )

    @pytest.mark.asyncio
    async def test_mixed_environment_gradual_migration(self):
        """Test gradual migration in mixed environment."""
        rag_service = RagService()

        # Mock services
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()

        # Setup: mixed environment with both types of IDs
        content1 = "Content 1" + " " * 100
        content2 = "Content 2" + " " * 100
        content3 = "Content 3" + " " * 100  # New content

        existing_chunks = [
            {
                "id": "550e8400e29b41d4a716446655440000",  # Random UUID
                "content": content1,
                "content_hash": rag_service._calculate_content_hash(content1),
            },
            {
                "id": rag_service._generate_deterministic_id(
                    "https://example.com/page", 1
                ),  # Already deterministic
                "content": content2,
                "content_hash": rag_service._calculate_content_hash(content2),
            },
        ]

        rag_service.vector_service.get_chunks_by_source.return_value = existing_chunks
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        rag_service.vector_service.upsert_documents.return_value = 1
        rag_service.vector_service.delete_chunks_by_ids.return_value = 1

        # Create crawl result with updated content1 and new content3
        crawl_result = CrawlResult(
            request_id="test-mixed-env",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content=content1 + " UPDATED",  # Changed content1
                    word_count=20,
                    metadata={"chunk_metadata": {"chunk_index": 0}},
                ),
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content=content3,  # New content
                    word_count=20,
                    metadata={"chunk_metadata": {"chunk_index": 2}},
                ),
            ],
        )

        # Process with deduplication enabled
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )

        # Should create 2 chunks: updated content1 (replacing legacy) and new content3
        assert result["chunks_created"] == 2, (
            "Should create 2 chunks (1 updated + 1 new)"
        )
        assert result["chunks_updated"] == 1, "Should update 1 legacy chunk"
        assert result["chunks_deleted"] >= 1, "Should delete at least the legacy chunk"

        # Verify that legacy chunk and orphaned deterministic chunk are deleted
        rag_service.vector_service.delete_chunks_by_ids.assert_called_once()

    @pytest.mark.asyncio
    async def test_backwards_compatibility_disabled_for_deterministic_env(self):
        """Test that backwards compatibility is not used in purely deterministic environment."""
        rag_service = RagService()

        # Mock services
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()

        # Setup: existing chunks with deterministic IDs only
        content1 = "Content 1" + " " * 100
        det_id1 = rag_service._generate_deterministic_id("https://example.com/page", 0)

        existing_chunks = [
            {
                "id": det_id1,
                "content": content1,
                "content_hash": rag_service._calculate_content_hash(content1),
            }
        ]

        rag_service.vector_service.get_chunks_by_source.return_value = existing_chunks
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        rag_service.vector_service.upsert_documents.return_value = 0
        rag_service.vector_service.delete_chunks_by_ids.return_value = 0

        # Create crawl result with same content
        crawl_result = CrawlResult(
            request_id="test-deterministic-env",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content=content1,  # Same content
                    word_count=20,
                    metadata={"chunk_metadata": {"chunk_index": 0}},
                )
            ],
        )

        # Process with deduplication enabled
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )

        # Should use normal deduplication (not backwards compatibility)
        assert result["chunks_created"] == 0, (
            "Should not create chunk for unchanged content"
        )
        assert result["chunks_skipped"] == 1, "Should skip unchanged chunk"
        assert result["chunks_deleted"] == 0, "Should not delete any chunks"

        # Verify no deletion was called (normal deduplication behavior)
        rag_service.vector_service.delete_chunks_by_ids.assert_not_called()
