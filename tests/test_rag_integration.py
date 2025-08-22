"""
Integration tests for RAG service with live services.

Following FastMCP best practices:
- Direct testing with live Qdrant and TEI services
- Minimal mocking, focus on behavior testing
- In-memory FastMCP client testing
- Comprehensive coverage of real code paths
"""

import asyncio
from datetime import datetime

import pytest
from fastmcp import Client

from crawler_mcp.core.rag import RagService
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent
from crawler_mcp.models.rag import RagQuery
from crawler_mcp.server import mcp


@pytest.fixture
async def rag_service():
    """Create a real RagService instance with live services."""
    service = RagService()
    async with service:
        yield service


@pytest.fixture
async def test_client():
    """Create FastMCP test client with live services."""
    async with Client(mcp) as client:
        yield client


class TestRagServiceIntegration:
    """Test RAG service with real dependencies."""

    @pytest.mark.asyncio
    async def test_rag_service_initialization(self, rag_service):
        """Test RAG service initializes properly with live services."""
        assert rag_service.vector_service is not None
        assert rag_service.embedding_service is not None
        assert rag_service._context_count > 0

    @pytest.mark.asyncio
    async def test_health_check_with_live_services(self, rag_service):
        """Test health check returns proper status with live services."""
        health = await rag_service.health_check()

        assert isinstance(health, dict)
        assert "vector_service" in health
        assert "embedding_service" in health

    @pytest.mark.asyncio
    async def test_process_simple_content(self, rag_service):
        """Test processing simple content through the full pipeline."""
        # Create test content
        pages = [
            PageContent(
                url="https://example.com/test1",
                title="Test Document 1",
                content="This is a test document about FastMCP and RAG systems. FastMCP provides excellent testing capabilities.",
                word_count=17,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="integration-test-1",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/test1"],
            pages=pages,
        )

        # Process the content
        result = await rag_service.process_crawl_result(crawl_result)

        # Verify processing completed successfully
        assert isinstance(result, dict)
        assert "chunks_created" in result
        assert "chunks_stored" in result
        assert result["chunks_created"] >= 1
        assert result["chunks_stored"] >= 1

    @pytest.mark.asyncio
    async def test_query_processed_content(self, rag_service):
        """Test querying content that was previously processed."""
        # First, add some content
        pages = [
            PageContent(
                url="https://example.com/query-test",
                title="Query Test Document",
                content="FastMCP is a framework for building MCP servers. It provides excellent testing capabilities and supports in-memory testing patterns.",
                word_count=21,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="query-test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/query-test"],
            pages=pages,
        )

        # Process the content
        await rag_service.process_crawl_result(crawl_result)

        # Query for the content
        query = RagQuery(
            query="FastMCP testing capabilities",
            source_filters=["https://example.com/query-test"],
            limit=5,
            min_score=0.1,
        )

        result = await rag_service.query(query)

        # Verify query results
        assert hasattr(result, "matches")
        assert hasattr(result, "total_matches")
        assert hasattr(result, "processing_time")
        assert isinstance(result.matches, list)
        assert result.total_matches >= 0

    @pytest.mark.asyncio
    async def test_delete_source_functionality(self, rag_service):
        """Test deleting a source and its associated chunks."""
        # Add content to delete
        pages = [
            PageContent(
                url="https://example.com/delete-test",
                title="Delete Test",
                content="This content will be deleted during testing.",
                word_count=8,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="delete-test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/delete-test"],
            pages=pages,
        )

        # Process the content
        await rag_service.process_crawl_result(crawl_result)

        # Delete the source
        deleted_count = await rag_service.delete_source(
            "https://example.com/delete-test"
        )

        # Verify deletion
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

    @pytest.mark.asyncio
    async def test_content_chunking_strategies(self, rag_service):
        """Test different chunking strategies with real content."""
        # Create content that will require chunking
        long_content = "This is a long document. " * 100  # 500 words

        pages = [
            PageContent(
                url="https://example.com/chunking-test",
                title="Chunking Test Document",
                content=long_content,
                word_count=500,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="chunking-test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/chunking-test"],
            pages=pages,
        )

        # Process with chunking
        result = await rag_service.process_crawl_result(crawl_result)

        # Verify chunking occurred
        assert result["chunks_created"] >= 1
        assert result["chunks_stored"] >= 1

    @pytest.mark.asyncio
    async def test_url_normalization_in_processing(self, rag_service):
        """Test URL normalization during content processing."""
        # Use URLs with fragments and query params
        pages = [
            PageContent(
                url="https://example.com/norm-test?param=value#fragment",
                title="Normalization Test",
                content="Testing URL normalization functionality.",
                word_count=5,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="norm-test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/norm-test?param=value#fragment"],
            pages=pages,
        )

        # Process the content
        result = await rag_service.process_crawl_result(crawl_result)

        # Verify processing worked (URLs should be normalized internally)
        assert result["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_duplicate_content_handling(self, rag_service):
        """Test how duplicate content is handled."""
        # Create identical content
        pages = [
            PageContent(
                url="https://example.com/dup-test",
                title="Duplicate Test",
                content="This is duplicate content for testing.",
                word_count=7,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="dup-test-1",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/dup-test"],
            pages=pages,
        )

        # Process first time
        result1 = await rag_service.process_crawl_result(crawl_result)

        # Process same content again
        crawl_result.request_id = "dup-test-2"
        result2 = await rag_service.process_crawl_result(crawl_result)

        # Verify deduplication behavior
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        # Second processing should detect duplicates
        assert "chunks_skipped" in result2


class TestRagMCPIntegration:
    """Test RAG functionality through MCP tools."""

    @pytest.mark.asyncio
    async def test_rag_query_tool_integration(self, test_client):
        """Test rag_query tool with live services."""
        # First add some content using scrape tool
        scrape_result = await test_client.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Verify scrape worked
        assert isinstance(scrape_result.data, dict)

        # Query the scraped content
        query_result = await test_client.call_tool(
            "rag_query", {"query": "example content", "limit": 5, "min_score": 0.0}
        )

        # Verify query results
        assert isinstance(query_result.data, dict)
        assert "query" in query_result.data
        assert "matches" in query_result.data
        assert "total_matches" in query_result.data

    @pytest.mark.asyncio
    async def test_list_sources_integration(self, test_client):
        """Test list_sources tool with live services."""
        result = await test_client.call_tool("list_sources", {})

        assert isinstance(result.data, dict)
        assert "sources" in result.data
        assert "pagination" in result.data
        assert isinstance(result.data["sources"], list)

    @pytest.mark.asyncio
    async def test_get_rag_stats_integration(self, test_client):
        """Test get_rag_stats tool with live services."""
        result = await test_client.call_tool("get_rag_stats", {})

        assert isinstance(result.data, dict)
        assert "rag_system" in result.data or "health_summary" in result.data

    @pytest.mark.asyncio
    async def test_delete_source_tool_integration(self, test_client):
        """Test delete_source tool with live services."""
        # First add content to delete
        await test_client.call_tool(
            "scrape",
            {"url": "https://example.com/delete-integration", "process_with_rag": True},
        )

        # Delete the source
        result = await test_client.call_tool(
            "delete_source",
            {"source_url": "https://example.com/delete-integration", "confirm": True},
        )

        assert isinstance(result.data, dict)
        assert "success" in result.data or "documents_deleted" in result.data


class TestRagErrorHandling:
    """Test error handling in real scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_content_processing(self, rag_service):
        """Test processing invalid/empty content."""
        # Create content with empty/invalid data
        pages = [
            PageContent(
                url="https://example.com/empty",
                title="",
                content="",
                word_count=0,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="empty-test",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/empty"],
            pages=pages,
        )

        # Should handle gracefully
        result = await rag_service.process_crawl_result(crawl_result)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_query_with_no_results(self, rag_service):
        """Test querying for content that doesn't exist."""
        query = RagQuery(
            query="completely unrelated nonsensical query that should match nothing",
            limit=5,
            min_score=0.9,  # High threshold
        )

        result = await rag_service.query(query)

        # Should return empty results gracefully
        assert hasattr(result, "matches")
        assert isinstance(result.matches, list)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_source(self, rag_service):
        """Test deleting a source that doesn't exist."""
        deleted_count = await rag_service.delete_source(
            "https://nonexistent.example.com"
        )

        # Should handle gracefully
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0


class TestRagAdvancedFeatures:
    """Test advanced RAG features with live services."""

    @pytest.mark.asyncio
    async def test_reranking_functionality(self, rag_service):
        """Test reranking with multiple similar documents."""
        # Add multiple documents with varying relevance
        pages = [
            PageContent(
                url=f"https://example.com/rerank-test-{i}",
                title=f"Document {i} about FastMCP",
                content=f"This document {i} discusses FastMCP testing and development practices. "
                + ("FastMCP is excellent for testing. " * (i + 1)),  # Varying relevance
                word_count=10 + i * 5,
                timestamp=datetime.utcnow(),
            )
            for i in range(3)
        ]

        crawl_result = CrawlResult(
            request_id="rerank-test",
            status=CrawlStatus.COMPLETED,
            urls=[f"https://example.com/rerank-test-{i}" for i in range(3)],
            pages=pages,
        )

        # Process all documents
        await rag_service.process_crawl_result(crawl_result)

        # Query to test reranking
        query = RagQuery(query="FastMCP testing practices", limit=3, min_score=0.0)

        result = await rag_service.query(query)

        # Verify reranking occurred (results should be ordered by relevance)
        assert len(result.matches) <= 3
        if len(result.matches) > 1:
            # Scores should be in descending order
            scores = [match.score for match in result.matches]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_context_manager_behavior(self, rag_service):
        """Test context manager behavior under normal conditions."""
        # Verify service is in context
        assert rag_service._context_count > 0

        # Test multiple operations
        health1 = await rag_service.health_check()
        health2 = await rag_service.health_check()

        assert isinstance(health1, dict)
        assert isinstance(health2, dict)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, rag_service):
        """Test concurrent RAG operations."""
        # Create multiple tasks
        tasks = []

        for i in range(3):
            pages = [
                PageContent(
                    url=f"https://example.com/concurrent-{i}",
                    title=f"Concurrent Test {i}",
                    content=f"Concurrent processing test document {i}.",
                    word_count=5,
                    timestamp=datetime.utcnow(),
                )
            ]

            crawl_result = CrawlResult(
                request_id=f"concurrent-{i}",
                status=CrawlStatus.COMPLETED,
                urls=[f"https://example.com/concurrent-{i}"],
                pages=pages,
            )

            task = rag_service.process_crawl_result(crawl_result)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for result in results:
            assert isinstance(result, dict)
            assert "chunks_created" in result
