"""
Test crawling tools with in-memory FastMCP client.
"""

from pathlib import Path

import pytest
from fastmcp import Client, ToolError


class TestCrawlingTools:
    """Test web crawling and scraping functionality."""

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_scrape_single_page(self, mcp_client: Client):
        """Test scraping a single webpage."""
        # Use a simple, reliable test page
        test_url = "https://httpbin.org/html"

        result = await mcp_client.call_tool(
            "scrape",
            {
                "url": test_url,
                "extraction_strategy": "css",
                "css_selector": "body",
            },
        )

        assert result.data is not None
        scrape_data = result.data

        # Check basic structure
        assert "success" in scrape_data
        assert "url" in scrape_data
        assert "content" in scrape_data
        assert "metadata" in scrape_data

        assert scrape_data["success"] is True
        assert scrape_data["url"] == test_url
        assert len(scrape_data["content"]) > 0

        # Check metadata
        metadata = scrape_data["metadata"]
        assert "title" in metadata
        assert "extracted_at" in metadata
        assert "word_count" in metadata
        assert metadata["word_count"] > 0

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_scrape_with_auto_rag(self, mcp_client: Client):
        """Test scraping with automatic RAG indexing."""
        test_url = "https://httpbin.org/html"

        result = await mcp_client.call_tool(
            "scrape",
            {
                "url": test_url,
                "auto_index": True,  # Enable automatic RAG indexing
            },
        )

        assert result.data is not None
        scrape_data = result.data

        assert scrape_data["success"] is True

        # Should have indexing information
        assert "indexed" in scrape_data
        assert "source_id" in scrape_data

        # If indexing succeeded
        if scrape_data["indexed"]:
            assert scrape_data["source_id"] is not None
            assert "chunks_created" in scrape_data
            assert isinstance(scrape_data["chunks_created"], int)

    @pytest.mark.unit
    async def test_crawl_directory(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test directory crawling functionality."""
        directory_path = sample_text_files[0].parent

        result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "max_files": 10,
                "auto_index": False,  # Don't index for unit test
            },
        )

        assert result.data is not None
        crawl_data = result.data

        # Check basic structure
        assert "success" in crawl_data
        assert "crawl_type" in crawl_data
        assert "results" in crawl_data
        assert "summary" in crawl_data

        assert crawl_data["success"] is True
        assert crawl_data["crawl_type"] == "directory"

        # Check results
        results = crawl_data["results"]
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should have proper structure
        for result_item in results:
            assert "source" in result_item
            assert "content" in result_item
            assert "metadata" in result_item
            assert len(result_item["content"]) > 0

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_crawl_with_indexing(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test directory crawling with RAG indexing."""
        directory_path = sample_text_files[0].parent

        result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "max_files": 5,
                "auto_index": True,  # Enable indexing
            },
        )

        assert result.data is not None
        crawl_data = result.data

        assert crawl_data["success"] is True

        # Check indexing results
        if "indexed_sources" in crawl_data:
            indexed = crawl_data["indexed_sources"]
            assert isinstance(indexed, int)
            assert indexed > 0

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=2)
    async def test_crawl_small_website(self, mcp_client: Client):
        """Test crawling a small website (marked as slow)."""
        # Use a simple, reliable test site
        test_url = "https://httpbin.org/"

        result = await mcp_client.call_tool(
            "crawl",
            {
                "target": test_url,
                "max_pages": 3,  # Limit to avoid long test times
                "max_depth": 1,
                "auto_index": False,  # Don't index for this test
            },
        )

        assert result.data is not None
        crawl_data = result.data

        # Should detect as website crawl
        assert crawl_data["crawl_type"] == "website"
        assert crawl_data["success"] is True

        # Should have crawled some pages
        results = crawl_data["results"]
        assert isinstance(results, list)
        assert len(results) >= 1  # At least the main page

    @pytest.mark.unit
    async def test_invalid_url_handling(self, mcp_client: Client):
        """Test handling of invalid URLs."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "scrape",
                {
                    "url": "not-a-valid-url",
                },
            )

    @pytest.mark.unit
    async def test_nonexistent_directory(self, mcp_client: Client):
        """Test handling of nonexistent directory."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "crawl",
                {
                    "target": "/nonexistent/directory/path",
                },
            )

    @pytest.mark.unit
    async def test_crawl_parameter_validation(self, mcp_client: Client):
        """Test parameter validation for crawl tool."""
        # Test missing target
        with pytest.raises(ToolError):
            await mcp_client.call_tool("crawl", {})

        # Test invalid max_pages
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "crawl",
                {
                    "target": "https://httpbin.org",
                    "max_pages": -1,
                },
            )

    @pytest.mark.unit
    async def test_scrape_parameter_validation(self, mcp_client: Client):
        """Test parameter validation for scrape tool."""
        # Test missing URL
        with pytest.raises(ToolError):
            await mcp_client.call_tool("scrape", {})

        # Test invalid extraction strategy
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "scrape",
                {
                    "url": "https://httpbin.org",
                    "extraction_strategy": "invalid_strategy",
                },
            )
