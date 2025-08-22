"""
Comprehensive tests for RAG service using FastMCP in-memory testing.

These tests follow FastMCP best practices:
- In-memory testing with direct server instance
- Live Qdrant and TEI services (no mocking)
- Pytest fixtures for reusable server setup
- Direct result.data access (no JSON parsing)
- Behavior-focused testing approach
"""

import time
from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from crawler_mcp.core.rag import (
    FixedSizeChunker,
    QueryCache,
    RagService,
    TokenBasedChunker,
    find_line_boundary,
    find_paragraph_boundary,
    find_sentence_boundary,
    find_word_boundary,
)
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent
from crawler_mcp.models.rag import RagQuery
from crawler_mcp.server import mcp


@pytest.fixture
async def test_server():
    """Create a test server instance for in-memory testing."""
    async with Client(mcp) as client:
        yield client


@pytest.fixture
async def rag_service():
    """Create a RagService instance for direct testing."""
    service = RagService()
    async with service:
        yield service


class TestQueryCache:
    """Test the QueryCache class functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with default and custom parameters."""
        # Default initialization
        cache = QueryCache()
        assert cache.max_size == 1000
        assert cache.ttl == timedelta(minutes=15)
        assert len(cache.cache) == 0

        # Custom initialization
        cache = QueryCache(max_size=500, ttl_minutes=30)
        assert cache.max_size == 500
        assert cache.ttl == timedelta(minutes=30)

    def test_generate_cache_key(self):
        """Test cache key generation."""
        cache = QueryCache()

        # Test deterministic key generation
        key1 = cache._generate_cache_key("test query", 10, 0.7, ["source1"], True)
        key2 = cache._generate_cache_key("test query", 10, 0.7, ["source1"], True)
        assert key1 == key2

        # Test different parameters generate different keys
        key3 = cache._generate_cache_key("test query", 10, 0.7, ["source2"], True)
        assert key1 != key3

    def test_cache_put_get(self):
        """Test cache put and get operations."""
        cache = QueryCache()

        # Test cache miss
        result = cache.get("test query", 10, 0.7, ["source1"], True)
        assert result is None

        # Test cache put and hit - need to create a proper RagResult mock
        from crawler_mcp.models.rag import RagResult

        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("test query", 10, 0.7, ["source1"], True, mock_result)
        result = cache.get("test query", 10, 0.7, ["source1"], True)
        assert result == mock_result

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = QueryCache(ttl_minutes=0.01)  # 0.6 seconds

        from crawler_mcp.models.rag import RagResult

        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("test query", 10, 0.7, ["source1"], True, mock_result)

        # Should hit immediately
        result = cache.get("test query", 10, 0.7, ["source1"], True)
        assert result == mock_result

        # Wait for expiration and test miss
        time.sleep(1)
        result = cache.get("test query", 10, 0.7, ["source1"], True)
        assert result is None

    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = QueryCache(max_size=2)

        from crawler_mcp.models.rag import RagResult

        # Create mock results
        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result3 = RagResult(
            query="query3",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )

        # Fill cache to capacity
        cache.put("query1", 10, 0.7, ["source1"], True, result1)
        cache.put("query2", 10, 0.7, ["source1"], True, result2)

        # Both should be in cache
        assert cache.get("query1", 10, 0.7, ["source1"], True) == result1
        assert cache.get("query2", 10, 0.7, ["source1"], True) == result2

        # Add third item - should evict oldest (LRU)
        cache.put("query3", 10, 0.7, ["source1"], True, result3)

        # First item should be evicted
        assert cache.get("query1", 10, 0.7, ["source1"], True) is None
        assert cache.get("query2", 10, 0.7, ["source1"], True) == result2
        assert cache.get("query3", 10, 0.7, ["source1"], True) == result3

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = QueryCache()

        from crawler_mcp.models.rag import RagResult

        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )

        cache.put("query1", 10, 0.7, ["source1"], True, result1)
        cache.put("query2", 10, 0.7, ["source1"], True, result2)

        assert len(cache.cache) == 2

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get("query1", 10, 0.7, ["source1"], True) is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = QueryCache()

        # Initial stats - check actual keys returned by get_stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["max_size"] == 1000
        assert stats["ttl_minutes"] == 15

        # Add item
        from crawler_mcp.models.rag import RagResult

        result = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("query1", 10, 0.7, ["source1"], True, result)

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 0


class TestBoundaryFunctions:
    """Test text boundary detection functions."""

    def test_find_paragraph_boundary(self):
        """Test paragraph boundary detection."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        # Test finding boundary near paragraph break (within 100 chars of ideal_end=20)
        # Returns max suitable boundary, which is the second "\n\n" at position 35
        boundary = find_paragraph_boundary(text, 20)
        assert boundary == 35  # Position of second "\n\n"

        # Test when no boundary found within range
        boundary = find_paragraph_boundary("No breaks here", 5)
        assert boundary is None  # Returns None when no suitable break found

    def test_find_sentence_boundary(self):
        """Test sentence boundary detection."""
        text = "First sentence. Second sentence! Third sentence?"

        # Test finding boundary near sentence end (within 50 chars of ideal_end=20)
        # Returns max suitable boundary, which is after "! " at position 33
        boundary = find_sentence_boundary(text, 20)
        assert boundary == 33  # Position after "! " (31 + 2 = 33)

        # Test when no boundary found within range
        boundary = find_sentence_boundary("No sentences", 5)
        assert boundary is None

    def test_find_line_boundary(self):
        """Test line boundary detection."""
        text = "First line\nSecond line\nThird line"

        # Test finding boundary near line break (within 30 chars of ideal_end=15)
        # Returns max suitable boundary, which is after second "\n" at position 23
        boundary = find_line_boundary(text, 15)
        assert boundary == 23  # Position after second "\n" (22 + 1 = 23)

        # Test when no boundary found within range
        boundary = find_line_boundary("No newlines", 5)
        assert boundary is None

    def test_find_word_boundary(self):
        """Test word boundary detection."""
        text = "First word second word third word"

        # Test finding boundary near word break (within 20 chars of ideal_end=15)
        # Returns max suitable boundary, which is after third space at position 29
        boundary = find_word_boundary(text, 15)
        assert boundary == 29  # Position after third space (28 + 1 = 29)

        # Test when no boundary found within range
        boundary = find_word_boundary("Nospaceshere", 5)
        assert boundary is None

    def test_boundary_functions_edge_cases(self):
        """Test edge cases for boundary functions."""
        # Empty text - should return None when no boundaries found
        assert find_paragraph_boundary("", 0) is None
        assert find_sentence_boundary("", 0) is None
        assert find_line_boundary("", 0) is None
        assert find_word_boundary("", 0) is None

        # Position beyond text - should return None when no suitable boundaries
        text = "Short text"
        assert find_paragraph_boundary(text, 200) is None  # No paragraph breaks
        assert find_sentence_boundary(text, 200) is None  # No sentence breaks
        assert find_line_boundary(text, 200) is None  # No line breaks
        assert find_word_boundary(text, 200) is None  # No spaces within range


class TestRagServiceBasics:
    """Test basic RagService functionality."""

    @pytest.mark.asyncio
    async def test_rag_service_singleton(self):
        """Test RagService singleton pattern."""
        service1 = RagService()
        service2 = RagService()
        assert service1 is service2

    @pytest.mark.asyncio
    async def test_rag_service_context_manager(self, rag_service):
        """Test RagService context manager behavior."""
        # Service should be initialized
        assert rag_service._context_count > 0  # Correct attribute name
        assert rag_service.vector_service is not None
        assert rag_service.embedding_service is not None

    @pytest.mark.asyncio
    async def test_health_check(self, rag_service):
        """Test RagService health check."""
        result = await rag_service.health_check()
        assert isinstance(result, dict)
        assert "vector_service" in result
        assert "embedding_service" in result

    @pytest.mark.asyncio
    async def test_normalize_url(self, rag_service):
        """Test URL normalization."""
        # Test basic URL normalization
        url1 = "https://example.com/path?param=value#fragment"
        normalized1 = rag_service._normalize_url(url1)
        assert normalized1 == "https://example.com/path?param=value"

        # Test URL without query or fragment
        url2 = "https://example.com/path"
        normalized2 = rag_service._normalize_url(url2)
        assert normalized2 == "https://example.com/path"

        # Test consistency
        assert rag_service._normalize_url(url1) == rag_service._normalize_url(url1)

    @pytest.mark.asyncio
    async def test_generate_deterministic_id(self, rag_service):
        """Test deterministic ID generation."""
        # Test consistent ID generation
        id1 = rag_service._generate_deterministic_id("https://example.com", 0)
        id2 = rag_service._generate_deterministic_id("https://example.com", 0)
        assert id1 == id2

        # Test different parameters generate different IDs
        id3 = rag_service._generate_deterministic_id("https://example.com", 1)
        assert id1 != id3

        # Test ID format (should be UUID format)
        assert isinstance(id1, str)
        assert len(id1) == 36  # UUID format: 8-4-4-4-12 = 36 chars
        assert id1.count("-") == 4  # UUID has 4 hyphens

    @pytest.mark.asyncio
    async def test_content_hash(self, rag_service):
        """Test content hashing for deduplication."""
        hash1 = rag_service._calculate_content_hash("test content")
        hash2 = rag_service._calculate_content_hash("test content")
        assert hash1 == hash2

        hash3 = rag_service._calculate_content_hash("different content")
        assert hash1 != hash3


class TestRagServiceCoreWorkflows:
    """Test core RAG service workflows with live services."""

    @pytest.mark.asyncio
    async def test_process_crawl_result_basic(self, rag_service):
        """Test basic crawl result processing."""
        # Create test crawl result
        pages = [
            PageContent(
                url="https://example.com/test",
                title="Test Page",
                content="This is test content for the RAG service. It should be long enough to create meaningful chunks and test the embedding process properly.",
                word_count=25,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="test-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/test"],
            pages=pages,
        )

        # Process the crawl result
        result = await rag_service.process_crawl_result(crawl_result)

        # Verify result structure (actual keys from process_crawl_result)
        assert isinstance(result, dict)
        assert "chunks_created" in result
        assert "chunks_stored" in result
        assert "chunks_deleted" in result
        assert "chunks_skipped" in result

        # Should have processed and stored chunks
        assert result["chunks_created"] >= 1
        assert result["chunks_stored"] >= 1

    @pytest.mark.asyncio
    async def test_text_chunking_character_based(self, rag_service):
        """Test character-based text chunking."""
        long_text = "This is a long text. " * 100  # 2100 characters

        # Use the new chunking module directly with smaller chunk size to ensure multiple chunks
        chunker = FixedSizeChunker(chunk_size=500, overlap=rag_service.chunk_overlap)
        chunks = chunker.chunk_text(long_text)

        assert len(chunks) > 1  # Should create multiple chunks
        assert isinstance(chunks, list)

        # Verify chunk structure (returns dict objects)
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "word_count" in chunk
            assert "char_count" in chunk
            assert len(chunk["text"]) > 0

        # Verify chunks are properly indexed
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_text_chunking_token_based(self, rag_service):
        """Test token-based text chunking."""
        # Create text larger than default chunk_size (1024 tokens) to ensure multiple chunks
        # Each sentence is ~5 tokens, so 250 sentences = ~1250 tokens > 1024 chunk_size
        long_text = "This is a test sentence. " * 250  # ~1250 tokens

        # Use the new chunking module directly with smaller chunk size to ensure multiple chunks
        chunker = TokenBasedChunker(chunk_size=500, overlap=rag_service.chunk_overlap)
        chunks = chunker.chunk_text(long_text)

        assert len(chunks) > 1  # Should create multiple chunks
        assert isinstance(chunks, list)

        # Verify chunk structure (returns dict objects)
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "word_count" in chunk
            assert "char_count" in chunk
            assert len(chunk["text"]) > 0
            # Token-based chunking should include token information
            assert "token_count" in chunk or "token_count_estimate" in chunk

        # Verify chunks are properly indexed
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_query_processing(self, rag_service):
        """Test query processing workflow."""
        # First add some test content
        pages = [
            PageContent(
                url="https://example.com/test-query",
                title="Test Query Page",
                content="This is content about FastMCP testing and RAG services. FastMCP provides excellent testing capabilities for MCP servers.",
                word_count=20,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="test-query-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/test-query"],
            pages=pages,
        )

        # Process the content first
        await rag_service.process_crawl_result(crawl_result)

        # Now query for it
        query = RagQuery(
            query="FastMCP testing capabilities",
            sources=["https://example.com/test-query"],
            limit=5,
            min_score=0.1,
        )

        result = await rag_service.query(query)

        # Verify result structure (RagResult object)
        assert hasattr(result, "matches")  # Correct attribute name
        assert hasattr(result, "total_matches")
        assert hasattr(result, "processing_time")
        assert isinstance(result.matches, list)

    @pytest.mark.asyncio
    async def test_get_sources_via_vector_service(self, rag_service):
        """Test getting unique sources via vector service."""
        # Add test content first
        pages = [
            PageContent(
                url="https://example.com/source1",
                title="Source 1",
                content="Content for source 1",
                word_count=4,
                timestamp=datetime.utcnow(),
            ),
            PageContent(
                url="https://example.com/source2",
                title="Source 2",
                content="Content for source 2",
                word_count=4,
                timestamp=datetime.utcnow(),
            ),
        ]

        crawl_result = CrawlResult(
            request_id="test-sources-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/source1", "https://example.com/source2"],
            pages=pages,
        )

        await rag_service.process_crawl_result(crawl_result)

        # Get sources via vector service
        async with rag_service.vector_service as vector_service:
            result = await vector_service.get_unique_sources()

        assert isinstance(result, dict)
        assert "sources" in result
        assert isinstance(result["sources"], list)

    @pytest.mark.asyncio
    async def test_delete_source(self, rag_service):
        """Test deleting a source."""
        # Add test content first
        pages = [
            PageContent(
                url="https://example.com/delete-test",
                title="Delete Test",
                content="Content to be deleted",
                word_count=4,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="test-delete-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/delete-test"],
            pages=pages,
        )

        await rag_service.process_crawl_result(crawl_result)

        # Delete the source
        result = await rag_service.delete_source("https://example.com/delete-test")

        # delete_source returns an integer count, not a dict
        assert isinstance(result, int)
        assert result >= 0  # Should return number of deleted documents

    @pytest.mark.asyncio
    async def test_get_stats(self, rag_service):
        """Test getting RAG service statistics."""
        result = await rag_service.get_stats()

        assert isinstance(result, dict)
        # Check actual keys returned by get_stats
        assert "health" in result or "collection" in result or "cache" in result


class TestRagServiceMCPIntegration:
    """Test RAG service integration through MCP tools."""

    @pytest.mark.asyncio
    async def test_rag_query_tool(self, test_server):
        """Test rag_query MCP tool."""
        # First add some content using scrape tool
        await test_server.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Now test rag_query
        query_result = await test_server.call_tool(
            "rag_query", {"query": "test content", "limit": 5}
        )

        assert isinstance(query_result.data, dict)
        # Use correct attribute name from RagResult
        assert "query" in query_result.data or "matches" in query_result.data

    @pytest.mark.asyncio
    async def test_rag_get_sources_tool(self, test_server):
        """Test list_sources MCP tool."""
        result = await test_server.call_tool("list_sources", {})

        assert isinstance(result.data, dict)
        assert "sources" in result.data

    @pytest.mark.asyncio
    async def test_rag_delete_source_tool(self, test_server):
        """Test delete_source MCP tool."""
        # Add some content first
        await test_server.call_tool(
            "scrape", {"url": "https://example.com/to-delete", "process_with_rag": True}
        )

        # Delete it
        result = await test_server.call_tool(
            "delete_source",
            {"source_url": "https://example.com/to-delete", "confirm": True},
        )

        assert isinstance(result.data, dict)
        assert "documents_deleted" in result.data

    @pytest.mark.asyncio
    async def test_rag_statistics_tool(self, test_server):
        """Test get_rag_stats MCP tool."""
        result = await test_server.call_tool("get_rag_stats", {})

        assert isinstance(result.data, dict)
        # Check for actual keys returned by get_rag_stats
        assert "rag_system" in result.data or "health_summary" in result.data
