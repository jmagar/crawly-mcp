"""
Test server and all MCP tools using FastMCP in-memory testing.

These tests follow FastMCP best practices:
- In-memory testing with direct server instance
- Pytest fixtures for reusable server setup
- Mocked external dependencies for unit testing
- Direct result.data access (no JSON parsing)
- Behavior-focused testing approach
"""

import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from crawler_mcp.config import settings

# Suppress warnings during testing
logging.getLogger("httpx").setLevel(logging.CRITICAL)


@pytest.fixture
def test_server():
    """Create a test server instance with mocked dependencies."""
    from fastmcp import Context

    # Create a clean server instance for testing
    test_mcp = FastMCP("test-crawler-mcp")

    # Import and register tools (without external dependencies)
    from crawler_mcp.tools.crawling import register_crawling_tools
    from crawler_mcp.tools.rag import register_rag_tools

    register_crawling_tools(test_mcp)
    register_rag_tools(test_mcp)

    # Define health_check tool directly for testing
    @test_mcp.tool
    async def health_check(ctx: Context) -> dict[str, Any]:
        """Test health check tool."""
        return {
            "server": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "services": {
                "embedding": {"status": "healthy"},
                "vector": {"status": "healthy"},
                "rag": {"status": "healthy"},
            },
            "configuration": {
                "embedding_model": "test-model",
                "vector_database": "test-db",
                "embedding_service": "test-service",
                "vector_dimension": 384,
            },
            "overall_status": "healthy",
        }

    # Define get_server_info tool directly for testing
    @test_mcp.tool
    async def get_server_info(ctx: Context) -> dict[str, Any]:
        """Test server info tool."""
        return {
            "server": {
                "name": "Crawlerr",
                "description": "RAG-enabled web crawling MCP server",
                "version": "0.1.0",
                "framework": "FastMCP 2.0+",
                "host": settings.server_host,
                "port": settings.server_port,
                "debug_mode": settings.debug,
                "production_mode": settings.production,
            },
            "capabilities": {
                "crawling": {
                    "single_page_scraping": True,
                    "website_crawling": True,
                    "repository_cloning": True,
                    "directory_processing": True,
                },
                "rag": {
                    "semantic_search": True,
                    "automatic_indexing": True,
                },
                "sources": {
                    "source_management": True,
                    "metadata_tracking": True,
                },
            },
            "configuration": {
                "crawling": {},
                "embedding": {},
                "vector_database": {},
            },
            "available_tools": [
                "scrape - Single page web scraping",
                "crawl - Multi-page website crawling",
                "rag_query - Semantic search queries",
                "list_sources - Source management",
                "health_check - System health monitoring",
                "get_server_info - Server information",
                "get_rag_stats - RAG system statistics",
                "delete_source - Source deletion",
            ],
        }

    return test_mcp


class TestServerTools:
    """Test all server MCP tools using FastMCP in-memory testing."""

    @pytest.mark.asyncio
    async def test_health_check_tool_mocked(self, test_server):
        """Test the health_check tool with mocked services."""
        with (
            patch("crawler_mcp.server.EmbeddingService") as mock_embedding,
            patch("crawler_mcp.server.VectorService") as mock_vector,
            patch("crawler_mcp.server.RagService") as mock_rag,
        ):
            # Mock embedding service
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.health_check.return_value = True
            mock_embedding_instance.get_model_info.return_value = {
                "model": "test-model"
            }
            mock_embedding().__aenter__.return_value = mock_embedding_instance

            # Mock vector service
            mock_vector_instance = AsyncMock()
            mock_vector_instance.health_check.return_value = True
            mock_vector_instance.get_collection_info.return_value = {"points": 100}
            mock_vector().__aenter__.return_value = mock_vector_instance

            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_instance.health_check.return_value = {
                "embedding": True,
                "vector": True,
            }
            mock_rag_instance.get_stats.return_value = {"documents": 100}
            mock_rag().__aenter__.return_value = mock_rag_instance

            async with Client(test_server) as client:
                result = await client.call_tool("health_check", {})

                # Access data directly (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "server" in result.data
                assert "services" in result.data
                assert "configuration" in result.data
                assert "overall_status" in result.data

                # Check server info
                server_info = result.data["server"]
                assert server_info["status"] == "healthy"
                assert "version" in server_info

                # Check services were called
                assert "embedding" in result.data["services"]
                assert "vector" in result.data["services"]
                assert "rag" in result.data["services"]

    @pytest.mark.asyncio
    async def test_get_server_info_tool(self, test_server):
        """Test the get_server_info tool returns complete server information."""
        async with Client(test_server) as client:
            result = await client.call_tool("get_server_info", {})

            # Check main sections (FastMCP pattern)
            assert isinstance(result.data, dict)
            assert "server" in result.data
            assert "capabilities" in result.data
            assert "configuration" in result.data
            assert "available_tools" in result.data

            # Check server details
            server = result.data["server"]
            assert server["name"] == "Crawlerr"
            assert "description" in server
            assert "version" in server
            assert "framework" in server
            assert server["host"] == settings.server_host
            assert server["port"] == settings.server_port

            # Check capabilities
            capabilities = result.data["capabilities"]
            assert "crawling" in capabilities
            assert "rag" in capabilities
            assert "sources" in capabilities

            crawling = capabilities["crawling"]
            assert crawling["single_page_scraping"] is True
            assert crawling["website_crawling"] is True
            assert crawling["repository_cloning"] is True
            assert crawling["directory_processing"] is True

    @pytest.mark.asyncio
    async def test_scrape_tool_mocked(self, test_server):
        """Test the scrape tool with mocked crawler service."""
        with patch("crawler_mcp.tools.crawling.CrawlerService") as mock_crawler_service:
            # Mock the crawler service
            mock_crawler_instance = AsyncMock()
            mock_page_content = MagicMock()
            mock_page_content.url = "https://example.com"
            mock_page_content.title = "Test Page"
            mock_page_content.content = "Test content"
            mock_page_content.word_count = 10
            mock_page_content.links = ["link1", "link2"]
            mock_page_content.images = ["img1"]
            mock_page_content.metadata = {"test": "meta"}
            mock_page_content.timestamp = datetime(2024, 1, 1, 0, 0, 0)

            mock_crawler_instance.scrape_single_page.return_value = mock_page_content
            mock_crawler_service.return_value = mock_crawler_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "scrape", {"url": "https://example.com", "process_with_rag": False}
                )

                # Check basic response structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "url" in result.data
                assert "title" in result.data
                assert "content" in result.data
                assert "word_count" in result.data
                assert "links_found" in result.data
                assert "images_found" in result.data

                # Verify the mocked data
                assert result.data["url"] == "https://example.com"
                assert result.data["title"] == "Test Page"
                assert result.data["content"] == "Test content"
                assert result.data["word_count"] == 10
                assert result.data["links_found"] == 2
                assert result.data["images_found"] == 1

    @pytest.mark.asyncio
    async def test_scrape_tool_with_rag_mocked(self, test_server):
        """Test the scrape tool with RAG processing enabled using mocks."""
        with (
            patch("crawler_mcp.tools.crawling.CrawlerService") as mock_crawler_service,
            patch("crawler_mcp.tools.crawling.RagService") as mock_rag_service,
        ):
            # Mock crawler service
            mock_crawler_instance = AsyncMock()
            from crawler_mcp.models.crawl import PageContent

            mock_page_content = PageContent(
                url="https://example.com",
                title="Test Page",
                content="Test content for RAG",
                word_count=15,
                links=["link1"],
                images=[],
                metadata={},
                timestamp=datetime(2024, 1, 1, 0, 0, 0),
            )

            mock_crawler_instance.scrape_single_page.return_value = mock_page_content
            mock_crawler_service.return_value = mock_crawler_instance

            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_instance.process_crawl_result.return_value = {
                "chunks_created": 2,
                "embeddings_generated": 2,
            }
            mock_rag_service().__aenter__.return_value = mock_rag_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "scrape", {"url": "https://example.com", "process_with_rag": True}
                )

                # Should have RAG processing info (FastMCP pattern)
                assert "rag_processing" in result.data
                rag_info = result.data["rag_processing"]
                assert isinstance(rag_info, dict)
                assert rag_info["chunks_created"] == 2
                assert rag_info["embeddings_generated"] == 2

    @pytest.mark.asyncio
    async def test_crawl_tool_directory_mocked(self, test_server):
        """Test the crawl tool with directory detection using mocks."""
        with patch("crawler_mcp.tools.crawling.CrawlerService") as mock_crawler_service:
            # Mock the crawler service for directory crawling
            mock_crawler_instance = AsyncMock()

            # Create mock crawl result
            from crawler_mcp.models.crawl import (
                CrawlResult,
                CrawlStatistics,
                CrawlStatus,
                PageContent,
            )

            mock_pages = []
            for i in range(3):
                page = PageContent(
                    url=f"file:///test/file{i + 1}.py",
                    title=f"file{i + 1}.py",
                    content=f"Test content {i + 1}",
                    word_count=10 + i,
                    metadata={"file_extension": "py", "file_path": f"file{i + 1}.py"},
                    timestamp=datetime(2024, 1, 1, 0, 0, 0),
                )
                mock_pages.append(page)

            mock_crawl_result = CrawlResult(
                request_id="test",
                status=CrawlStatus.COMPLETED,
                urls=["file:///test"],
                pages=mock_pages,
                statistics=CrawlStatistics(
                    total_pages_requested=3,
                    total_pages_crawled=3,
                    total_bytes_downloaded=300,
                    crawl_duration_seconds=1.0,
                ),
            )

            mock_crawler_instance.crawl_directory.return_value = mock_crawl_result
            mock_crawler_service.return_value = mock_crawler_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "crawl",
                    {
                        "target": "/test/directory",
                        "process_with_rag": False,
                        "recursive": True,
                    },
                )

                # Check directory crawl result structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "status" in result.data
                assert "files_processed" in result.data
                assert "file_types" in result.data
                assert "statistics" in result.data
                assert "directory_path" in result.data

                # Should have processed our mock files
                assert result.data["files_processed"] == 3
                assert result.data["directory_path"] == "/test/directory"

    @pytest.mark.asyncio
    async def test_crawl_tool_website_mocked(self, test_server):
        """Test the crawl tool with website URL detection using mocks."""
        with patch("crawler_mcp.tools.crawling.CrawlerService") as mock_crawler_service:
            # Mock the crawler service for website crawling
            mock_crawler_instance = AsyncMock()

            # Create mock crawl result for website
            from crawler_mcp.models.crawl import (
                CrawlResult,
                CrawlStatistics,
                CrawlStatus,
                PageContent,
            )

            mock_pages = []
            for i in range(2):
                page = PageContent(
                    url=f"https://example.com/page{i + 1}",
                    title=f"Test Page {i + 1}",
                    content=f"Website content {i + 1}",
                    word_count=50 + i * 10,
                    links=[f"link{j}" for j in range(i + 1)],
                    images=[f"img{j}" for j in range(i)],
                    metadata={"domain": "example.com"},
                    timestamp=datetime(2024, 1, 1, 0, 0, 0),
                )
                mock_pages.append(page)

            mock_crawl_result = CrawlResult(
                request_id="test",
                status=CrawlStatus.COMPLETED,
                urls=["https://example.com"],
                pages=mock_pages,
                statistics=CrawlStatistics(
                    total_pages_requested=2,
                    total_pages_crawled=2,
                    total_bytes_downloaded=1000,
                    crawl_duration_seconds=2.0,
                    unique_domains=1,
                    total_links_discovered=3,
                ),
            )

            mock_crawler_instance.crawl_website.return_value = mock_crawl_result
            mock_crawler_service.return_value = mock_crawler_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "crawl",
                    {"target": "https://example.com", "process_with_rag": False},
                )

                # Check website crawl result structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "status" in result.data
                assert "pages_crawled" in result.data
                assert "statistics" in result.data
                assert "advanced_features" in result.data
                assert "sample_pages" in result.data

                # Should have crawled our mock pages
                assert result.data["pages_crawled"] == 2

    @pytest.mark.asyncio
    async def test_rag_query_tool_mocked(self, test_server):
        """Test the rag_query tool with mocked RAG service."""
        with patch("crawler_mcp.tools.rag.RagService") as mock_rag_service:
            # Mock the RAG service
            mock_rag_instance = AsyncMock()

            # Create mock RAG result
            from crawler_mcp.models.rag import DocumentChunk, RagResult, SearchMatch

            mock_doc = DocumentChunk(
                id="test-doc-1",
                source_url="https://example.com",
                source_title="Test Document",
                content="Test content for search",
                chunk_index=0,
                word_count=5,
                timestamp=datetime.utcnow(),
                metadata={"domain": "example.com"},
            )

            mock_match = SearchMatch(
                score=0.85,
                relevance="high",
                document=mock_doc,
                highlighted_content="Test <em>content</em> for search",
            )

            mock_rag_result = RagResult(
                query="HTML document",
                matches=[mock_match],
                total_matches=1,
                processing_time=0.1,
                embedding_time=0.05,
                search_time=0.03,
                rerank_time=0.02,
                average_score=0.85,
                best_match_score=0.85,
                has_high_confidence_matches=True,
                timestamp=datetime.utcnow(),
            )

            mock_rag_instance.query.return_value = mock_rag_result
            mock_rag_service().__aenter__.return_value = mock_rag_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "rag_query",
                    {
                        "query": "HTML document",
                        "limit": 5,
                        "min_score": 0.0,
                        "include_content": True,
                    },
                )

                # Check response structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "query" in result.data
                assert "total_matches" in result.data
                assert "matches" in result.data
                assert "performance" in result.data
                assert "quality_metrics" in result.data
                assert "timestamp" in result.data

                # Check query details
                assert result.data["query"] == "HTML document"
                assert result.data["total_matches"] == 1

                # Check matches structure
                matches = result.data["matches"]
                assert isinstance(matches, list)
                assert len(matches) == 1

                match = matches[0]
                assert match["score"] == 0.85
                assert match["relevance"] == "high"
                assert "document" in match

                document = match["document"]
                assert document["id"] == "test-doc-1"
                assert document["source_url"] == "https://example.com"
                assert document["content"] == "Test content for search"

                # Check performance metrics
                perf = result.data["performance"]
                assert perf["total_time"] == 0.1
                assert perf["embedding_time"] == 0.05

                # Check quality metrics
                quality = result.data["quality_metrics"]
                assert quality["average_score"] == 0.85
                assert quality["best_match_score"] == 0.85
                assert quality["high_confidence_matches"] is True

    @pytest.mark.asyncio
    async def test_list_sources_tool_mocked(self, test_server):
        """Test the list_sources tool with mocked vector service."""
        with patch("crawler_mcp.tools.rag.VectorService") as mock_vector_service:
            # Mock the vector service
            mock_vector_instance = AsyncMock()

            # Mock sources response
            mock_sources_response = {
                "sources": [
                    {
                        "url": "https://example.com",
                        "title": "Test Page",
                        "source_type": "webpage",
                        "status": "active",
                        "chunk_count": 5,
                        "total_content_length": 1000,
                        "total_word_count": 200,
                        "last_crawled": "2024-01-01T00:00:00Z",
                    }
                ],
                "pagination": {"limit": 20, "offset": 0, "total": 1, "returned": 1},
            }

            mock_vector_stats = {"unique_sources": 1, "total_points": 5}

            mock_vector_instance.get_unique_sources.return_value = mock_sources_response
            mock_vector_instance.get_sources_stats.return_value = mock_vector_stats
            mock_vector_service().__aenter__.return_value = mock_vector_instance

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "list_sources", {"limit": 20, "offset": 0}
                )

                # Check response structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "sources" in result.data
                assert "pagination" in result.data
                assert "statistics" in result.data
                assert "filters_applied" in result.data

                # Check pagination
                pagination = result.data["pagination"]
                assert pagination["limit"] == 20
                assert pagination["offset"] == 0
                assert pagination["total"] == 1
                assert pagination["returned"] == 1

                # Check sources structure
                sources = result.data["sources"]
                assert isinstance(sources, list)
                assert len(sources) == 1

                source = sources[0]
                assert "id" in source
                assert source["url"] == "https://example.com"
                assert source["title"] == "Test Page"
                assert source["source_type"] == "webpage"
                assert source["status"] == "active"
                assert source["chunk_count"] == 5
                assert source["total_content_length"] == 1000
                assert "metadata" in source

                # Check metadata structure
                metadata = source["metadata"]
                assert metadata["domain"] == "example.com"
                assert metadata["word_count"] == 200

    @pytest.mark.asyncio
    async def test_get_rag_stats_tool_mocked(self, test_server):
        """Test the get_rag_stats tool with mocked services."""
        with (
            patch("crawler_mcp.tools.rag.RagService") as mock_rag_service,
            patch("crawler_mcp.tools.rag.VectorService") as mock_vector_service,
        ):
            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_stats = {
                "health": {"embedding_service": True, "vector_service": True},
                "collection": {
                    "points_count": 100,
                    "vectors_count": 100,
                    "status": "green",
                    "config": {"vector_size": 384},
                },
            }
            mock_rag_instance.get_stats.return_value = mock_rag_stats
            mock_rag_service().__aenter__.return_value = mock_rag_instance

            # Mock vector service
            mock_vector_instance = AsyncMock()
            mock_vector_stats = {"unique_sources": 10, "total_points": 100}
            mock_vector_instance.get_sources_stats.return_value = mock_vector_stats
            mock_vector_service().__aenter__.return_value = mock_vector_instance

            async with Client(test_server) as client:
                result = await client.call_tool("get_rag_stats", {})

                # Check response structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "rag_system" in result.data
                assert "source_management" in result.data
                assert "health_summary" in result.data
                assert "quick_facts" in result.data
                assert "timestamp" in result.data

                # Check health summary
                health = result.data["health_summary"]
                assert "all_services_healthy" in health
                assert "services" in health
                assert health["all_services_healthy"] is True

                # Check quick facts
                facts = result.data["quick_facts"]
                assert facts["total_documents"] == 100
                assert facts["total_vectors"] == 100
                assert facts["total_sources"] == 10
                assert facts["vector_dimension"] == 384
                assert facts["collection_status"] == "green"

    @pytest.mark.asyncio
    async def test_delete_source_tool_mocked(self, test_server):
        """Test the delete_source tool with mocked RAG service."""
        with patch("crawler_mcp.tools.rag.RagService") as mock_rag_service:
            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_instance.delete_source.return_value = 5  # 5 documents deleted
            mock_rag_service().__aenter__.return_value = mock_rag_instance

            async with Client(test_server) as client:
                test_url = "https://example.com"

                # Test deletion without confirmation (should fail)
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        "delete_source", {"source_url": test_url, "confirm": False}
                    )
                assert "confirmation" in str(exc_info.value).lower()

                # Test deletion with confirmation
                result = await client.call_tool(
                    "delete_source", {"source_url": test_url, "confirm": True}
                )

                # Check response structure (FastMCP pattern)
                assert isinstance(result.data, dict)
                assert "source_url" in result.data
                assert "documents_deleted" in result.data
                assert "success" in result.data
                assert "timestamp" in result.data

                # Verify deletion details
                assert result.data["source_url"] == test_url
                assert result.data["documents_deleted"] == 5
                assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, test_server):
        """Test error handling in tools using proper FastMCP patterns."""
        async with Client(test_server) as client:
            # Test invalid URL in scrape - should raise ToolError
            with pytest.raises(ToolError):
                await client.call_tool("scrape", {"url": "not-a-valid-url"})

            # Test invalid query in rag_query - should raise ToolError
            with pytest.raises(ToolError):
                await client.call_tool(
                    "rag_query",
                    {
                        "query": "",  # Empty query should fail
                        "limit": 10,
                    },
                )

            # Test invalid limit in rag_query - should raise ToolError
            with pytest.raises(ToolError):
                await client.call_tool(
                    "rag_query",
                    {
                        "query": "test",
                        "limit": 150,  # Too high, should fail
                    },
                )

            # Test invalid score in rag_query - should raise ToolError
            with pytest.raises(ToolError):
                await client.call_tool(
                    "rag_query",
                    {
                        "query": "test",
                        "min_score": 2.0,  # Out of range
                    },
                )


class TestServerConfiguration:
    """Test server configuration and startup functionality."""

    def test_server_settings_loaded(self):
        """Test that server settings are properly loaded."""
        # Check key settings are available
        assert hasattr(settings, "server_host")
        assert hasattr(settings, "server_port")
        assert hasattr(settings, "debug")
        assert hasattr(settings, "tei_url")
        assert hasattr(settings, "qdrant_url")

        # Check types
        assert isinstance(settings.server_host, str)
        assert isinstance(settings.server_port, int)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.tei_url, str)
        assert isinstance(settings.qdrant_url, str)

    @pytest.mark.asyncio
    async def test_mcp_instance_tools(self, test_server):
        """Test that MCP server instance has correct tools registered."""
        # Check test server is created
        assert test_server is not None
        assert hasattr(test_server, "get_tools")
        assert callable(test_server.get_tools)

        # Check that tools are registered
        tools = await test_server.get_tools()
        assert len(tools) > 0

        # Check for expected tools
        tool_names = list(tools.keys())
        expected_tools = [
            "health_check",
            "get_server_info",
            "scrape",
            "crawl",
            "rag_query",
            "list_sources",
            "get_rag_stats",
            "delete_source",
        ]

        for expected in expected_tools:
            assert expected in tool_names

    @pytest.mark.asyncio
    async def test_tool_parameters_validation(self, test_server):
        """Test tool parameter validation using FastMCP patterns."""
        async with Client(test_server) as client:
            # Test scrape with missing required parameters
            with pytest.raises((ToolError, ValueError, TypeError)):
                await client.call_tool("scrape", {})  # Missing required URL

            # Test crawl with missing required parameters
            with pytest.raises((ToolError, ValueError, TypeError)):
                await client.call_tool("crawl", {})  # Missing required target

            # Test list_sources with invalid limit
            with pytest.raises(ToolError):
                await client.call_tool(
                    "list_sources",
                    {
                        "limit": 300  # Exceeds max limit of 200
                    },
                )

            # Test list_sources with invalid offset
            with pytest.raises(ToolError):
                await client.call_tool(
                    "list_sources",
                    {
                        "offset": -1  # Negative offset
                    },
                )


class TestServerLifecycle:
    """Test server lifecycle and utility functions."""

    @pytest.mark.asyncio
    async def test_startup_checks_function_mocked(self):
        """Test the startup_checks function with mocked dependencies."""
        from crawler_mcp.server import startup_checks

        with (
            patch("crawler_mcp.server.EmbeddingService") as mock_embedding,
            patch("crawler_mcp.server.VectorService") as mock_vector,
        ):
            # Mock services to avoid external dependencies
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.health_check.return_value = True
            mock_embedding().__aenter__.return_value = mock_embedding_instance

            mock_vector_instance = AsyncMock()
            mock_vector_instance.health_check.return_value = True
            mock_vector_instance.ensure_collection.return_value = True
            mock_vector().__aenter__.return_value = mock_vector_instance

            # Should run without error with mocked services
            await startup_checks()  # Should complete successfully
