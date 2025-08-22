"""
Tests for the modular vector service implementation.

These tests ensure the new modular vector service maintains backward compatibility
and provides the same functionality as the original implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Any

from qdrant_client.models import UpdateStatus

from crawler_mcp.core.vectors import (
    VectorService,
    CollectionManager,
    DocumentOperations,
    SearchEngine,
    StatisticsCollector,
    BaseVectorService,
)
from crawler_mcp.models.rag import DocumentChunk, SearchMatch


class TestBaseVectorService:
    """Test the base vector service functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def base_service(self, mock_client):
        """Create a base vector service with mock client."""
        return BaseVectorService(client=mock_client)

    @pytest.mark.asyncio
    async def test_client_error_handling(self, base_service):
        """Test client error handling and recreation."""
        # Test client closed error
        error = Exception("client has been closed")
        should_recreate = await base_service._handle_client_error(error)
        assert should_recreate is True

        # Test other error
        error = Exception("some other error")
        should_recreate = await base_service._handle_client_error(error)
        assert should_recreate is False

    @pytest.mark.asyncio
    async def test_close(self, base_service, mock_client):
        """Test closing the client."""
        await base_service.close()
        mock_client.close.assert_called_once()


class TestCollectionManager:
    """Test the collection manager module."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def collection_manager(self, mock_client):
        """Create a collection manager with mock client."""
        return CollectionManager(client=mock_client)

    @pytest.mark.asyncio
    async def test_health_check_success(self, collection_manager, mock_client):
        """Test successful health check."""
        mock_client.get_collections.return_value = MagicMock()
        result = await collection_manager.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, collection_manager, mock_client):
        """Test failed health check."""
        mock_client.get_collections.side_effect = Exception("Connection failed")
        result = await collection_manager.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_collection_exists(self, collection_manager, mock_client):
        """Test ensuring collection exists when it already exists."""
        # Mock existing collection with the actual collection name
        mock_collections = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = collection_manager.collection_name  # Use actual collection name
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        result = await collection_manager.ensure_collection_exists()
        assert result is True
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_collection(self, collection_manager, mock_client):
        """Test creating a new collection."""
        # Mock no existing collections
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        result = await collection_manager.ensure_collection_exists()
        assert result is True
        mock_client.create_collection.assert_called_once()


class TestDocumentOperations:
    """Test the document operations module."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def operations(self, mock_client):
        """Create document operations with mock client."""
        return DocumentOperations(client=mock_client)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document chunk."""
        return DocumentChunk(
            id="test_id",
            content="Test content",
            source_url="https://example.com",
            source_title="Test Title",
            chunk_index=0,
            word_count=2,
            char_count=12,
            timestamp=datetime.utcnow(),
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "metadata"}
        )

    @pytest.mark.asyncio
    async def test_upsert_empty_documents(self, operations):
        """Test upserting empty document list."""
        result = await operations.upsert_documents([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_upsert_documents_success(self, operations, mock_client, sample_document):
        """Test successful document upsert."""
        # Mock collection manager
        with patch('crawler_mcp.core.vectors.collections.CollectionManager') as MockCollectionManager:
            mock_manager = AsyncMock()
            mock_manager.ensure_collection_exists = AsyncMock(return_value=True)
            MockCollectionManager.return_value = mock_manager
            
            # Mock successful upsert
            mock_result = MagicMock()
            mock_result.status = UpdateStatus.COMPLETED
            mock_client.upsert.return_value = mock_result

            result = await operations.upsert_documents([sample_document])
            assert result == 1
            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, operations, mock_client):
        """Test retrieving document by ID."""
        # Mock collection manager
        with patch('crawler_mcp.core.vectors.collections.CollectionManager') as MockCollectionManager:
            mock_manager = AsyncMock()
            mock_manager.ensure_collection_exists = AsyncMock(return_value=True)
            MockCollectionManager.return_value = mock_manager
            
            # Mock document retrieval
            mock_point = MagicMock()
            mock_point.id = "test_id"
            mock_point.payload = {
                "content": "Test content",
                "source_url": "https://example.com",
                "source_title": "Test Title",
                "chunk_index": 0,
                "word_count": 2,
                "char_count": 12,
                "timestamp": "2023-01-01T00:00:00Z"
            }
            mock_client.retrieve.return_value = [mock_point]

            result = await operations.get_document_by_id("test_id")
            assert result is not None
            assert result.id == "test_id"
            assert result.content == "Test content"


class TestSearchEngine:
    """Test the search engine module."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def search_engine(self, mock_client):
        """Create search engine with mock client."""
        return SearchEngine(client=mock_client)

    @pytest.mark.asyncio
    async def test_search_similar(self, search_engine, mock_client):
        """Test vector similarity search."""
        # Mock collection manager
        with patch('crawler_mcp.core.vectors.collections.CollectionManager') as MockCollectionManager:
            mock_manager = AsyncMock()
            mock_manager.ensure_collection_exists = AsyncMock(return_value=True)
            MockCollectionManager.return_value = mock_manager
            
            # Mock search results
            mock_result = MagicMock()
            mock_result.id = "test_id"
            mock_result.score = 0.95
            mock_result.payload = {
                "content": "Test content",
                "source_url": "https://example.com",
                "source_title": "Test Title",
                "chunk_index": 0,
                "word_count": 2,
                "char_count": 12,
                "timestamp": "2023-01-01T00:00:00Z"
            }
            
            mock_query_response = MagicMock()
            mock_query_response.points = [mock_result]
            mock_client.query_points.return_value = mock_query_response

            query_vector = [0.1, 0.2, 0.3]
            results = await search_engine.search_similar(query_vector, limit=5)
            
            assert len(results) == 1
            assert isinstance(results[0], SearchMatch)
            assert results[0].score == 0.95
            assert results[0].document.content == "Test content"


class TestStatisticsCollector:
    """Test the statistics collector module."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def statistics(self, mock_client):
        """Create statistics collector with mock client."""
        return StatisticsCollector(client=mock_client)

    @pytest.mark.asyncio
    async def test_get_sources_stats(self, statistics, mock_client):
        """Test getting source statistics."""
        # Mock collection manager
        with patch('crawler_mcp.core.vectors.collections.CollectionManager') as MockCollectionManager:
            mock_manager = AsyncMock()
            mock_manager.ensure_collection_exists = AsyncMock(return_value=True)
            MockCollectionManager.return_value = mock_manager
            
            # Mock scroll results
            mock_point = MagicMock()
            mock_point.payload = {
                "source_url": "https://example.com",
                "char_count": 100
            }
            mock_client.scroll.return_value = ([mock_point], None)

            stats = await statistics.get_sources_stats()
            
            assert "total_documents" in stats
            assert "unique_sources" in stats
            assert stats["total_documents"] == 1


class TestUnifiedVectorService:
    """Test the unified vector service with backward compatibility."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_vector_service_initialization(self):
        """Test that VectorService initializes all modules correctly."""
        with patch('crawler_mcp.core.vectors.base.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            service = VectorService()
            
            assert isinstance(service.collections, CollectionManager)
            assert isinstance(service.operations, DocumentOperations)
            assert isinstance(service.search, SearchEngine)
            assert isinstance(service.statistics, StatisticsCollector)
            
            # Check that all modules share the same client
            assert service.collections.client == service.client
            assert service.operations.client == service.client
            assert service.search.client == service.client
            assert service.statistics.client == service.client

    @pytest.mark.asyncio
    async def test_backward_compatibility_methods(self):
        """Test that all backward compatibility methods exist and delegate correctly."""
        with patch('crawler_mcp.core.vectors.base.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            service = VectorService()
            
            # Mock the module methods
            service.collections.health_check = AsyncMock(return_value=True)
            service.collections.ensure_collection_exists = AsyncMock(return_value=True)
            service.collections.get_collection_info = AsyncMock(return_value={})
            service.operations.upsert_documents = AsyncMock(return_value=1)
            service.operations.get_document_by_id = AsyncMock(return_value=None)
            service.operations.delete_documents_by_source = AsyncMock(return_value=1)
            service.operations.get_chunks_by_source = AsyncMock(return_value=[])
            service.operations.delete_chunks_by_ids = AsyncMock(return_value=1)
            service.search.search_similar = AsyncMock(return_value=[])
            service.statistics.get_sources_stats = AsyncMock(return_value={})
            service.statistics.get_unique_sources = AsyncMock(return_value={})

            # Test all backward compatibility methods
            assert await service.health_check() is True
            assert await service.ensure_collection() is True
            assert await service.get_collection_info() == {}
            assert await service.upsert_documents([]) == 1
            assert await service.get_document_by_id("test") is None
            assert await service.delete_documents_by_source("url") == 1
            assert await service.get_chunks_by_source("url") == []
            assert await service.delete_chunks_by_ids(["id"]) == 1
            assert await service.search_similar([0.1, 0.2, 0.3]) == []
            assert await service.get_sources_stats() == {}
            assert await service.get_unique_sources() == {}

            # Verify delegation worked
            service.collections.health_check.assert_called_once()
            service.collections.ensure_collection_exists.assert_called_once()
            service.collections.get_collection_info.assert_called_once()
            service.operations.upsert_documents.assert_called_once()
            service.operations.get_document_by_id.assert_called_once()
            service.operations.delete_documents_by_source.assert_called_once()
            service.operations.get_chunks_by_source.assert_called_once()
            service.operations.delete_chunks_by_ids.assert_called_once()
            service.search.search_similar.assert_called_once()
            service.statistics.get_sources_stats.assert_called_once()
            service.statistics.get_unique_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that VectorService works as an async context manager."""
        with patch('crawler_mcp.core.vectors.base.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            async with VectorService() as service:
                assert service is not None
                assert isinstance(service, VectorService)
                # The client should be accessible
                assert service.client == mock_client
            
            # Verify client was closed
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_recreate_client_updates_all_modules(self):
        """Test that recreating the client updates all modules."""
        with patch('crawler_mcp.core.vectors.base.AsyncQdrantClient') as mock_client_class:
            mock_client_1 = AsyncMock()
            mock_client_2 = AsyncMock()
            mock_client_class.side_effect = [mock_client_1, mock_client_2]
            
            service = VectorService()
            original_client = service.client
            
            # Recreate client
            await service._recreate_client()
            
            # Verify all modules got the new client
            assert service.client != original_client
            assert service.collections.client == service.client
            assert service.operations.client == service.client
            assert service.search.client == service.client
            assert service.statistics.client == service.client