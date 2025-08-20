"""
Comprehensive tests for vector service to maximize coverage.

This module focuses on testing all code paths in the vector service
to achieve high coverage on the vectors.py module.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawler_mcp.config import settings
from crawler_mcp.core.vectors import VectorService
from crawler_mcp.models.rag import DocumentChunk


class TestVectorServiceComprehensive:
    """Comprehensive tests for VectorService to maximize coverage."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = True
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=384, distance="Cosine"),
                    shard_number=1,
                    replication_factor=1,
                    write_consistency_factor=1,
                ),
                hnsw_config=MagicMock(
                    m=16, ef_construct=100, full_scan_threshold=10000
                ),
                optimizer_config=MagicMock(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0,
                ),
                wal_config=MagicMock(wal_capacity_mb=32, wal_segments_ahead=0),
            ),
            status="green",
            points_count=100,
            vectors_count=100,
        )
        return mock_client

    @pytest.mark.asyncio
    async def test_init_and_context_management(self, mock_qdrant_client):
        """Test initialization and context management."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            # Test context manager
            async with VectorService() as service:
                assert service.client is mock_qdrant_client
                assert service.collection_name == settings.qdrant_collection

            # Test manual initialization
            service = VectorService()
            await service.__aenter__()
            assert service.client is mock_qdrant_client
            await service.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_health_check(self, mock_qdrant_client):
        """Test health check functionality."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Test successful health check
                mock_qdrant_client.get_collections.return_value = MagicMock()
                result = await service.health_check()
                assert result is True

                # Test failed health check
                mock_qdrant_client.get_collections.side_effect = Exception(
                    "Connection failed"
                )
                result = await service.health_check()
                assert result is False

    @pytest.mark.asyncio
    async def test_ensure_collection_exists(self, mock_qdrant_client):
        """Test collection creation when it doesn't exist."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Test collection creation
                mock_qdrant_client.collection_exists.return_value = False
                result = await service.ensure_collection()
                assert result is True
                mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_already_exists(self, mock_qdrant_client):
        """Test when collection already exists."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock get_collections to return existing collection
                mock_collection = MagicMock()
                mock_collection.name = (
                    service.collection_name
                )  # Set the collection name
                mock_qdrant_client.get_collections.return_value = MagicMock(
                    collections=[mock_collection]
                )

                result = await service.ensure_collection()
                assert result is True
                mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_single(self, mock_qdrant_client):
        """Test adding a single document."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock successful upsert
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

                # Test data
                chunks = [
                    DocumentChunk(
                        id="test-1",
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content",
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
                        chunk_index=0,
                        word_count=2,
                        timestamp=datetime.utcnow(),
                        metadata={"test": "meta"},
                    )
                ]

                result = await service.upsert_documents(chunks)
                assert result == 1
                mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, mock_qdrant_client):
        """Test adding documents in batches."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock successful batch upsert
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

                # Create large batch to trigger batching
                chunks = [
                    DocumentChunk(
                        id=f"test-{i}",
                        source_url="https://example.com",
                        source_title="Test Page",
                        content=f"Test content {i}",
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
                        chunk_index=i,
                        word_count=2,
                        timestamp=datetime.utcnow(),
                        metadata={"test": "meta"},
                    )
                    for i in range(150)
                ]

                result = await service.upsert_documents(chunks, batch_size=100)
                assert result == 150
                # Should be called twice (two batches)
                assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_search_vectors(self, mock_qdrant_client):
        """Test vector search functionality."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock search results
                mock_point = MagicMock(
                    id="test-1",
                    score=0.95,
                    payload={
                        "source_url": "https://example.com",
                        "source_title": "Test Page",
                        "content": "Test content",
                        "chunk_index": 0,
                        "word_count": 2,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "metadata": {"test": "meta"},
                    },
                )
                mock_response = MagicMock()
                mock_response.points = [mock_point]
                mock_qdrant_client.query_points.return_value = mock_response

                query_vector = [0.1, 0.2, 0.3] * 128
                results = await service.search_similar(
                    query_vector, limit=10, score_threshold=0.7
                )

                assert len(results) == 1
                assert results[0].score == 0.95
                assert results[0].document.content == "Test content"
                mock_qdrant_client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_vectors_with_filter(self, mock_qdrant_client):
        """Test vector search with source filters."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock empty search results with query_points
                mock_response = MagicMock()
                mock_response.points = []
                mock_qdrant_client.query_points.return_value = mock_response

                query_vector = [0.1, 0.2, 0.3] * 128
                source_filters = ["https://example.com"]

                results = await service.search_similar(
                    query_vector, limit=10, source_filter=source_filters
                )

                # Verify query_points was called with filter
                call_args = mock_qdrant_client.query_points.call_args
                assert "query_filter" in call_args.kwargs
                assert call_args.kwargs["query_filter"] is not None
                assert results == []

    @pytest.mark.asyncio
    async def test_delete_by_source_url(self, mock_qdrant_client):
        """Test deleting documents by source URL."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock successful deletion using UpdateStatus
                from qdrant_client.models import UpdateStatus

                mock_qdrant_client.delete.return_value = MagicMock(
                    operation_id=0, status=UpdateStatus.COMPLETED
                )

                deleted_count = await service.delete_documents_by_source(
                    "https://example.com"
                )
                assert deleted_count == 1  # Returns 1 for successful deletion
                mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_unique_sources(self, mock_qdrant_client):
        """Test getting unique sources."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock scroll results
                mock_scroll_result = (
                    [
                        MagicMock(
                            payload={
                                "source_url": "https://example.com",
                                "source_title": "Test Page",
                                "timestamp": "2024-01-01T00:00:00Z",
                                "content": "Test content",
                            }
                        )
                    ],
                    None,  # Next page offset
                )
                mock_qdrant_client.scroll.return_value = mock_scroll_result
                mock_qdrant_client.count.return_value = MagicMock(count=10)

                result = await service.get_unique_sources()

                assert "sources" in result
                assert "pagination" in result
                assert len(result["sources"]) == 1
                assert result["sources"][0]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_get_unique_sources_with_filters(self, mock_qdrant_client):
        """Test getting unique sources with filters."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                mock_qdrant_client.scroll.return_value = ([], None)
                mock_qdrant_client.count.return_value = MagicMock(count=0)

                result = await service.get_unique_sources(
                    domains=["example.com"], search_term="test"
                )

                assert "sources" in result
                assert len(result["sources"]) == 0

    @pytest.mark.asyncio
    async def test_get_sources_stats(self, mock_qdrant_client):
        """Test getting source statistics."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock scroll for source statistics
                mock_points = []
                for i in range(5):
                    mock_point = MagicMock()
                    mock_point.payload = {
                        "source_url": f"https://example{i}.com",
                        "char_count": 100,
                        "word_count": 20,
                    }
                    mock_points.append(mock_point)

                mock_qdrant_client.scroll.return_value = (mock_points, None)

                stats = await service.get_sources_stats()

                assert "total_documents" in stats
                assert "unique_sources" in stats
                assert "source_counts" in stats
                assert "total_content_length" in stats
                assert "average_chunk_size" in stats
                assert stats["total_documents"] == 5
                assert stats["unique_sources"] == 5

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_qdrant_client):
        """Test getting collection information."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                info = await service.get_collection_info()

                assert "points_count" in info
                assert "vectors_count" in info
                assert "status" in info
                assert "config" in info
                assert info["points_count"] == 100

    @pytest.mark.asyncio
    async def test_error_handling_in_search(self, mock_qdrant_client):
        """Test error handling in search operations."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock search failure
                mock_qdrant_client.query_points.side_effect = Exception("Search failed")

                query_vector = [0.1, 0.2, 0.3] * 128

                # Should raise ToolError on search failure
                from fastmcp.exceptions import ToolError

                with pytest.raises(ToolError, match="Vector search failed"):
                    await service.search_similar(query_vector)

    @pytest.mark.asyncio
    async def test_error_handling_in_add_documents(self, mock_qdrant_client):
        """Test error handling in add documents."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock upsert failure
                mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")

                chunks = [
                    DocumentChunk(
                        id="test-1",
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content",
                        embedding=[0.1, 0.2, 0.3] * 128,  # Include embedding in chunk
                        chunk_index=0,
                        word_count=2,
                        timestamp=datetime.utcnow(),
                        metadata={},
                    )
                ]

                # Should handle error gracefully and return 0
                result = await service.upsert_documents(chunks)
                assert result == 0

    @pytest.mark.asyncio
    async def test_validation_errors(self, mock_qdrant_client):
        """Test validation of input parameters."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Test chunks without embeddings - should be skipped
                chunks = [
                    DocumentChunk(
                        id="test-1",
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content 1",
                        embedding=None,  # No embedding
                        chunk_index=0,
                        word_count=2,
                        timestamp=datetime.utcnow(),
                        metadata={},
                    ),
                    DocumentChunk(
                        id="test-2",
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content 2",
                        embedding=None,  # No embedding
                        chunk_index=1,
                        word_count=2,
                        timestamp=datetime.utcnow(),
                        metadata={},
                    ),
                ]

                # Should skip all chunks without embeddings and return 0
                result = await service.upsert_documents(chunks)
                assert result == 0

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, mock_qdrant_client):
        """Test retrieving a document by ID."""
        with patch("crawler_mcp.core.vectors.AsyncQdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            async with VectorService() as service:
                # Mock collection exists for ensure_collection
                mock_qdrant_client.collection_exists.return_value = True

                # Mock successful retrieval
                mock_point = MagicMock(
                    id="test-1",
                    payload={
                        "content": "Test content",
                        "source_url": "https://example.com",
                        "source_title": "Test Page",
                        "chunk_index": 0,
                        "word_count": 2,
                        "char_count": 12,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "metadata": {"test": "meta"},
                    },
                )
                mock_qdrant_client.retrieve.return_value = [mock_point]

                document = await service.get_document_by_id("test-1")

                assert document is not None
                assert document.id == "test-1"
                assert document.content == "Test content"
                assert document.source_url == "https://example.com"
                mock_qdrant_client.retrieve.assert_called_once()

                # Test document not found
                mock_qdrant_client.retrieve.return_value = []
                document = await service.get_document_by_id("nonexistent")
                assert document is None
