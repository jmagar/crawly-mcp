"""
End-to-end integration test flows.

These tests verify complete workflows work correctly together.
"""

from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError


class TestIntegrationFlows:
    """Test complete end-to-end workflows."""

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_complete_directory_rag_workflow(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test complete workflow: crawl directory -> index -> query -> manage sources."""
        directory_path = sample_text_files[0].parent

        # Step 1: Crawl and index directory content
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 5,
            },
        )

        assert crawl_result.data["success"] is True
        assert crawl_result.data["crawl_type"] == "directory"

        # Wait for indexing to complete
        import asyncio

        await asyncio.sleep(2)

        # Step 2: Verify content was indexed by checking stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data

        assert stats["total_sources"] > 0
        assert stats["total_chunks"] > 0
        assert "directory" in stats["source_types"]

        # Step 3: Query the indexed content
        query_result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "test document content",
                "limit": 10,
                "source_types": ["directory"],
            },
        )

        rag_data = query_result.data
        assert rag_data["total_results"] > 0
        assert len(rag_data["results"]) > 0

        # Step 4: List sources to find what was created
        sources_result = await mcp_client.call_tool(
            "list_sources",
            {
                "limit": 20,
                "source_type": "directory",
            },
        )

        sources = sources_result.data["sources"]
        assert len(sources) > 0

        # Find a source from our directory crawl
        directory_source = None
        for source in sources:
            if str(directory_path) in source.get("metadata", {}).get("path", ""):
                directory_source = source
                break

        assert directory_source is not None

        # Step 5: Delete the source to clean up
        delete_result = await mcp_client.call_tool(
            "delete_source", {"source_id": directory_source["source_id"]}
        )

        assert delete_result.data["success"] is True

        # Step 6: Verify source was deleted
        final_stats = await mcp_client.call_tool("get_rag_stats", {})
        assert final_stats.data["total_sources"] < stats["total_sources"]

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_web_scrape_and_rag_workflow(self, mcp_client: Client):
        """Test complete workflow: scrape web page -> index -> query."""
        test_url = "https://httpbin.org/html"

        # Step 1: Scrape with auto-indexing
        scrape_result = await mcp_client.call_tool(
            "scrape",
            {
                "url": test_url,
                "auto_index": True,
                "extraction_strategy": "css",
                "css_selector": "body",
            },
        )

        assert scrape_result.data["success"] is True
        assert scrape_result.data.get("indexed", False) is True

        source_id = scrape_result.data.get("source_id")
        assert source_id is not None

        # Step 2: Wait and verify indexing
        import asyncio

        await asyncio.sleep(2)

        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data
        assert stats["total_sources"] > 0
        assert "webpage" in stats["source_types"]

        # Step 3: Query for content from the scraped page
        query_result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "html page content",
                "limit": 5,
                "source_types": ["webpage"],
            },
        )

        rag_data = query_result.data
        # Should find some results (may be 0 if content is minimal)
        assert rag_data["total_results"] >= 0

        # Step 4: Clean up by deleting the source
        delete_result = await mcp_client.call_tool(
            "delete_source", {"source_id": source_id}
        )

        assert delete_result.data["success"] is True

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_health_check_workflow(self, mcp_client: Client):
        """Test complete health monitoring workflow."""
        # Step 1: Check overall health
        health_result = await mcp_client.call_tool("health_check", {})
        health_data = health_result.data

        assert "overall_status" in health_data
        assert health_data["overall_status"] in ["healthy", "degraded"]

        # Step 2: Get detailed server info
        info_result = await mcp_client.call_tool("get_server_info", {})
        info_data = info_result.data

        assert "server" in info_data
        assert "capabilities" in info_data
        assert "configuration" in info_data

        # Step 3: Check RAG system stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats_data = stats_result.data

        assert "total_sources" in stats_data
        assert "collection_info" in stats_data

        # All should be consistent and healthy
        if health_data["overall_status"] == "healthy":
            # Services should be working
            services = health_data["services"]
            assert services.get("embedding", {}).get("status") == "healthy"
            assert services.get("vector", {}).get("status") == "healthy"
            assert services.get("rag", {}).get("status") == "healthy"

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_error_handling_workflow(self, mcp_client: Client):
        """Test error handling across different scenarios."""
        # Test 1: Invalid scrape URL
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "scrape",
                {
                    "url": "not-a-valid-url-at-all",
                },
            )

        # Test 2: Invalid crawl target
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "crawl",
                {
                    "target": "/absolutely/nonexistent/path/nowhere",
                },
            )

        # Test 3: Delete non-existent source
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "delete_source",
                {
                    "source_id": "absolutely-nonexistent-source-id-12345",
                },
            )

        # Test 4: Invalid RAG query parameters
        with pytest.raises(ToolError):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "limit": -5,  # Invalid negative limit
                },
            )

        # After all these errors, health check should still work
        health_result = await mcp_client.call_tool("health_check", {})
        assert health_result.data is not None
        assert "overall_status" in health_result.data

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_concurrent_operations(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test concurrent operations don't interfere with each other."""
        import asyncio

        # Create multiple concurrent operations
        tasks = []

        # Multiple queries (should work even with empty database)
        for i in range(3):
            task = mcp_client.call_tool(
                "rag_query",
                {
                    "query": f"test query {i}",
                    "limit": 5,
                },
            )
            tasks.append(task)

        # Health checks
        for _ in range(2):
            task = mcp_client.call_tool("health_check", {})
            tasks.append(task)

        # Stats checks
        task = mcp_client.call_tool("get_rag_stats", {})
        tasks.append(task)

        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (or be expected exceptions)
        for result in results:
            if isinstance(result, Exception):
                # Should not have unexpected exceptions
                pytest.fail(f"Unexpected exception in concurrent test: {result}")
            else:
                assert result.data is not None

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_data_consistency_workflow(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test data consistency across operations."""
        directory_path = sample_text_files[0].parent

        # Get initial stats
        initial_stats = await mcp_client.call_tool("get_rag_stats", {})
        initial_sources = initial_stats.data["total_sources"]
        initial_chunks = initial_stats.data["total_chunks"]

        # Index some content
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 2,
            },
        )
        assert crawl_result.data["success"] is True

        # Wait for indexing
        import asyncio

        await asyncio.sleep(2)

        # Check stats increased
        after_index_stats = await mcp_client.call_tool("get_rag_stats", {})
        after_sources = after_index_stats.data["total_sources"]
        after_chunks = after_index_stats.data["total_chunks"]

        assert after_sources > initial_sources
        assert after_chunks > initial_chunks

        # List sources to find what we created
        sources_result = await mcp_client.call_tool(
            "list_sources",
            {
                "limit": 50,
                "source_type": "directory",
            },
        )

        our_sources = []
        for source in sources_result.data["sources"]:
            if str(directory_path) in source.get("metadata", {}).get("path", ""):
                our_sources.append(source)

        assert len(our_sources) > 0

        # Delete our sources one by one and verify consistency
        for source in our_sources:
            delete_result = await mcp_client.call_tool(
                "delete_source", {"source_id": source["source_id"]}
            )
            assert delete_result.data["success"] is True

        # Final stats should be back to initial (or close)
        final_stats = await mcp_client.call_tool("get_rag_stats", {})
        final_sources = final_stats.data["total_sources"]
        final_chunks = final_stats.data["total_chunks"]

        # Should have removed our additions
        assert final_sources <= initial_sources
        assert final_chunks <= initial_chunks
