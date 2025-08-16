"""
Tests for deduplication functionality.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from crawler_mcp.core.rag import RagService
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent


class TestDeduplication:
    """Test cases for deduplication functionality."""

    def test_generate_deterministic_id(self):
        """Test deterministic ID generation."""
        rag_service = RagService()

        # Same URL and chunk index should produce same ID
        id1 = rag_service._generate_deterministic_id("https://example.com/page", 0)
        id2 = rag_service._generate_deterministic_id("https://example.com/page", 0)
        assert id1 == id2

        # Different chunk index should produce different ID
        id3 = rag_service._generate_deterministic_id("https://example.com/page", 1)
        assert id1 != id3

        # Different URL should produce different ID
        id4 = rag_service._generate_deterministic_id("https://example.com/other", 0)
        assert id1 != id4

        # ID should be a valid UUID format (36 characters with dashes)
        assert len(id1) == 36
        assert id1.count("-") == 4  # UUID format has 4 dashes

    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        rag_service = RagService()

        # Same content should produce same hash
        hash1 = rag_service._calculate_content_hash("Hello, world!")
        hash2 = rag_service._calculate_content_hash("Hello, world!")
        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = rag_service._calculate_content_hash("Hello, universe!")
        assert hash1 != hash3

        # Hash should be 64 characters (SHA256 hex)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_normalize_url(self):
        """Test URL normalization."""
        rag_service = RagService()

        # Test basic normalization
        assert (
            rag_service._normalize_url("http://example.com") == "https://example.com/"
        )
        assert (
            rag_service._normalize_url("https://example.com/") == "https://example.com/"
        )
        assert (
            rag_service._normalize_url("https://example.com/path/")
            == "https://example.com/path"
        )

        # Test domain case normalization
        assert (
            rag_service._normalize_url("https://EXAMPLE.COM/Path")
            == "https://example.com/Path"
        )

        # Test query parameter sorting
        url_with_params = "https://example.com/page?b=2&a=1"
        expected = "https://example.com/page?a=1&b=2"
        assert rag_service._normalize_url(url_with_params) == expected

        # Test fragment removal
        assert (
            rag_service._normalize_url("https://example.com/page#section")
            == "https://example.com/page"
        )

    @pytest.mark.asyncio
    async def test_deduplication_logic(self):
        """Test the core deduplication logic."""
        rag_service = RagService()

        # Mock the vector service
        rag_service.vector_service = AsyncMock()
        # Generate the correct deterministic ID for the test content
        expected_chunk_id = rag_service._generate_deterministic_id(
            "https://example.com/page", 0
        )
        rag_service.vector_service.get_chunks_by_source.return_value = [
            {
                "id": expected_chunk_id,
                "content": "Existing content",
                "content_hash": rag_service._calculate_content_hash("Existing content"),
            }
        ]

        # Mock the embedding service
        rag_service.embedding_service = AsyncMock()
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]

        # Mock the vector service upsert
        rag_service.vector_service.upsert_documents.return_value = 1

        # Create a crawl result with the same content (should be skipped)
        crawl_result = CrawlResult(
            request_id="test-request-dedup",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content="Existing content",
                    word_count=2,
                    metadata={},
                )
            ],
        )

        # Process with deduplication enabled
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True, force_update=False
        )

        # Should skip the unchanged chunk
        assert result["chunks_skipped"] == 1
        assert result["chunks_created"] == 0
        assert result["embeddings_generated"] == 0

    @pytest.mark.asyncio
    async def test_force_update_override(self):
        """Test force_update parameter overrides deduplication."""
        rag_service = RagService()

        # Mock the vector service with existing chunk
        rag_service.vector_service = AsyncMock()
        rag_service.vector_service.get_chunks_by_source.return_value = [
            {
                "id": "abc123",
                "content": "Existing content",
                "content_hash": rag_service._calculate_content_hash("Existing content"),
            }
        ]

        # Mock the embedding service
        rag_service.embedding_service = AsyncMock()
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]

        # Mock the vector service upsert
        rag_service.vector_service.upsert_documents.return_value = 1

        # Create a crawl result with the same content
        crawl_result = CrawlResult(
            request_id="test-request-force",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content="Existing content",
                    word_count=2,
                    metadata={},
                )
            ],
        )

        # Process with force_update=True
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True, force_update=True
        )

        # Should process the chunk even though content is unchanged
        assert result["chunks_skipped"] == 0
        assert result["chunks_created"] == 1
        assert result["embeddings_generated"] == 1

    @pytest.mark.asyncio
    async def test_integration_full_deduplication_flow(self):
        """Test complete deduplication flow: first crawl, unchanged re-crawl, changed re-crawl."""
        rag_service = RagService()

        # Mock services
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        rag_service.vector_service.upsert_documents.return_value = 1
        rag_service.vector_service.delete_chunks_by_ids.return_value = 0

        # STEP 1: First crawl - no existing chunks
        rag_service.vector_service.get_chunks_by_source.return_value = []

        crawl_result_1 = CrawlResult(
            request_id="test-request-1",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content="Initial content",
                    word_count=2,
                    metadata={},
                )
            ],
        )

        result_1 = await rag_service.process_crawl_result(
            crawl_result_1, deduplication=True
        )

        # First crawl should process everything (fast path)
        assert result_1["chunks_created"] == 1
        assert result_1["chunks_skipped"] == 0
        assert result_1["chunks_deleted"] == 0
        assert result_1["embeddings_generated"] == 1

        # STEP 2: Re-crawl with same content - should skip
        # Mock existing chunk with same content hash
        chunk_id = rag_service._generate_deterministic_id("https://example.com/page", 0)
        content_hash = rag_service._calculate_content_hash("Initial content")

        rag_service.vector_service.get_chunks_by_source.return_value = [
            {
                "id": chunk_id,
                "content": "Initial content",
                "content_hash": content_hash,
            }
        ]

        result_2 = await rag_service.process_crawl_result(
            crawl_result_1, deduplication=True
        )

        # Should skip unchanged content
        assert result_2["chunks_created"] == 0
        assert result_2["chunks_skipped"] == 1
        assert result_2["chunks_deleted"] == 0
        assert result_2["embeddings_generated"] == 0

        # STEP 3: Re-crawl with changed content - should update
        crawl_result_3 = CrawlResult(
            request_id="test-request-3",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page Updated",
                    content="Updated content",  # Changed content
                    word_count=2,
                    metadata={},
                )
            ],
        )

        result_3 = await rag_service.process_crawl_result(
            crawl_result_3, deduplication=True
        )

        # Should update changed content
        assert result_3["chunks_created"] == 1
        assert result_3["chunks_skipped"] == 0
        assert result_3["chunks_updated"] == 1  # This should be 1 since content changed
        assert result_3["chunks_deleted"] == 0
        assert result_3["embeddings_generated"] == 1

    @pytest.mark.asyncio
    async def test_orphaned_chunk_cleanup(self):
        """Test that orphaned chunks are properly deleted."""
        rag_service = RagService()

        # Mock services
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()
        rag_service.embedding_service.generate_embeddings_true_batch.return_value = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        rag_service.vector_service.upsert_documents.return_value = 1
        rag_service.vector_service.delete_chunks_by_ids.return_value = (
            2  # Deleted 2 orphans
        )

        # Mock existing chunks - more chunks than in new crawl (orphans)
        existing_chunks = [
            {
                "id": rag_service._generate_deterministic_id(
                    "https://example.com/page", 0
                ),
                "content": "Content 1",
                "content_hash": rag_service._calculate_content_hash("Content 1"),
            },
            {
                "id": rag_service._generate_deterministic_id(
                    "https://example.com/page", 1
                ),
                "content": "Content 2",
                "content_hash": rag_service._calculate_content_hash("Content 2"),
            },
            {
                "id": rag_service._generate_deterministic_id(
                    "https://example.com/page", 2
                ),
                "content": "Content 3 - will be orphaned",
                "content_hash": rag_service._calculate_content_hash(
                    "Content 3 - will be orphaned"
                ),
            },
        ]
        rag_service.vector_service.get_chunks_by_source.return_value = existing_chunks

        # New crawl result has only 1 page (chunks 1 and 2 will be orphaned)
        crawl_result = CrawlResult(
            request_id="test-request-orphan",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/page"],
            pages=[
                PageContent(
                    url="https://example.com/page",
                    title="Test Page",
                    content="Content 1",  # Only first chunk remains
                    word_count=2,
                    metadata={},
                )
            ],
        )

        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )

        # Should delete orphaned chunks
        assert result["chunks_deleted"] == 2  # 2 orphaned chunks deleted
        assert result["chunks_skipped"] == 1  # 1 chunk unchanged
        assert result["chunks_created"] == 0  # No new chunks

        # Verify delete was called with correct orphaned IDs
        expected_orphan_ids = [
            rag_service._generate_deterministic_id("https://example.com/page", 1),
            rag_service._generate_deterministic_id("https://example.com/page", 2),
        ]
        rag_service.vector_service.delete_chunks_by_ids.assert_called_once()
        called_ids = rag_service.vector_service.delete_chunks_by_ids.call_args[0][0]
        assert set(called_ids) == set(expected_orphan_ids)

    def test_performance_metrics_tracking(self):
        """Test that deduplication metrics are properly tracked."""
        rag_service = RagService()

        # Test URL normalization performance
        urls_to_test = [
            "https://example.com/page?b=2&a=1",
            "http://EXAMPLE.COM/Page#section",
            "https://example.com/page/",
        ]

        normalized_urls = []
        for url in urls_to_test:
            normalized = rag_service._normalize_url(url)
            normalized_urls.append(normalized)

        # All should normalize consistently
        assert normalized_urls[0] == "https://example.com/page?a=1&b=2"
        assert (
            normalized_urls[1] == "https://example.com/Page"
        )  # Path case is preserved
        assert normalized_urls[2] == "https://example.com/page"

        # Test ID generation performance and consistency
        chunk_ids = []
        for i in range(100):
            chunk_id = rag_service._generate_deterministic_id(
                "https://example.com/page", i
            )
            chunk_ids.append(chunk_id)

        # All IDs should be unique
        assert len(set(chunk_ids)) == 100

        # Same inputs should produce same outputs
        duplicate_id = rag_service._generate_deterministic_id(
            "https://example.com/page", 50
        )
        assert duplicate_id == chunk_ids[50]
