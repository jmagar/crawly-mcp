"""
Test RAG (Retrieval Augmented Generation) tools.
"""

from pathlib import Path

import pytest
from fastmcp import Client


class TestRagTools:
    """Test RAG functionality with real vector database."""

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_rag_query_empty_database(self, mcp_client: Client):
        """Test RAG query when database is empty."""
        result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "test query with no results",
                "limit": 5,
            },
        )

        assert result.data is not None
        rag_data = result.data

        # Check basic structure
        assert "query" in rag_data
        assert "results" in rag_data
        assert "total_results" in rag_data
        assert "search_metadata" in rag_data

        assert rag_data["query"] == "test query with no results"
        assert rag_data["results"] == []
        assert rag_data["total_results"] == 0

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_rag_workflow(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test complete RAG workflow: index content then query."""
        directory_path = sample_text_files[0].parent

        # Step 1: Index some content
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        assert crawl_result.data is not None
        assert crawl_result.data["success"] is True

        # Wait a moment for indexing to complete
        import asyncio

        await asyncio.sleep(1)

        # Step 2: Query the indexed content
        query_result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "test document content",
                "limit": 5,
            },
        )

        assert query_result.data is not None
        rag_data = query_result.data

        # Should find some results now
        assert rag_data["total_results"] > 0
        assert len(rag_data["results"]) > 0

        # Check result structure
        for result in rag_data["results"]:
            assert "content" in result
            assert "metadata" in result
            assert "score" in result
            assert "source_id" in result

            # Score should be reasonable
            assert 0 <= result["score"] <= 1

            # Content should not be empty
            assert len(result["content"]) > 0

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_rag_query_with_filters(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test RAG query with source type filters."""
        directory_path = sample_text_files[0].parent

        # Index content first
        await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # Query with directory filter
        result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "document",
                "source_types": ["directory"],
                "limit": 10,
            },
        )

        assert result.data is not None
        rag_data = result.data

        # Should have results from directory crawl
        if rag_data["total_results"] > 0:
            for result in rag_data["results"]:
                metadata = result["metadata"]
                # Should be from directory source
                assert metadata.get("source_type") == "directory"

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_get_rag_stats(self, mcp_client: Client):
        """Test getting RAG system statistics."""
        result = await mcp_client.call_tool("get_rag_stats", {})

        assert result.data is not None
        stats_data = result.data

        # Check basic structure
        assert "total_sources" in stats_data
        assert "total_chunks" in stats_data
        assert "source_types" in stats_data
        assert "collection_info" in stats_data

        # Values should be non-negative integers
        assert isinstance(stats_data["total_sources"], int)
        assert isinstance(stats_data["total_chunks"], int)
        assert stats_data["total_sources"] >= 0
        assert stats_data["total_chunks"] >= 0

        # Source types should be a dict
        assert isinstance(stats_data["source_types"], dict)

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_list_sources(self, mcp_client: Client):
        """Test listing sources in the database."""
        result = await mcp_client.call_tool(
            "list_sources",
            {
                "limit": 10,
                "offset": 0,
            },
        )

        assert result.data is not None
        sources_data = result.data

        # Check structure
        assert "sources" in sources_data
        assert "total_count" in sources_data
        assert "limit" in sources_data
        assert "offset" in sources_data

        # Should be valid values
        assert isinstance(sources_data["sources"], list)
        assert isinstance(sources_data["total_count"], int)
        assert sources_data["total_count"] >= 0
        assert sources_data["limit"] == 10
        assert sources_data["offset"] == 0

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_delete_source(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test deleting a source from the database."""
        directory_path = sample_text_files[0].parent

        # Index content first to have something to delete
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 1,
            },
        )

        assert crawl_result.data["success"] is True

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # List sources to get a source_id
        list_result = await mcp_client.call_tool("list_sources", {"limit": 1})
        sources = list_result.data["sources"]

        if len(sources) > 0:
            source_id = sources[0]["source_id"]

            # Delete the source
            delete_result = await mcp_client.call_tool(
                "delete_source", {"source_id": source_id}
            )

            assert delete_result.data is not None
            delete_data = delete_result.data

            assert "success" in delete_data
            assert delete_data["success"] is True
            assert "source_id" in delete_data
            assert delete_data["source_id"] == source_id

    @pytest.mark.unit
    async def test_rag_query_parameter_validation(self, mcp_client: Client):
        """Test RAG query parameter validation."""
        # Test missing query
        with pytest.raises(Exception):
            await mcp_client.call_tool("rag_query", {})

        # Test invalid limit
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "limit": 0,
                },
            )

        # Test negative offset
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "offset": -1,
                },
            )

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_rag_query_performance(self, mcp_client: Client):
        """Test RAG query performance and response time."""
        import time

        start_time = time.time()

        result = await mcp_client.call_tool(
            "rag_query",
            {
                "query": "performance test query",
                "limit": 10,
            },
        )

        end_time = time.time()
        query_time = end_time - start_time

        # Query should complete in reasonable time (adjust as needed)
        assert query_time < 5.0  # 5 seconds max

        assert result.data is not None

        # Should have timing information in search metadata
        search_metadata = result.data.get("search_metadata", {})
        if "search_time_ms" in search_metadata:
            search_time_ms = search_metadata["search_time_ms"]
            assert isinstance(search_time_ms, (int, float))
            assert search_time_ms > 0
