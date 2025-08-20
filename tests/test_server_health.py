"""
Test server health and basic functionality.
"""

import pytest
from fastmcp import Client


class TestServerHealth:
    """Test basic server health and information endpoints."""

    @pytest.mark.unit
    async def test_health_check(self, mcp_client: Client):
        """Test the health check tool returns proper status."""
        result = await mcp_client.call_tool("health_check", {})

        assert result.data is not None
        health_data = result.data

        # Check basic structure
        assert "server" in health_data
        assert "services" in health_data
        assert "configuration" in health_data
        assert "overall_status" in health_data

        # Check server info
        server_info = health_data["server"]
        assert server_info["status"] == "healthy"
        assert server_info["version"] == "0.1.0"

        # Services should be checked
        services = health_data["services"]
        assert "embedding" in services
        assert "vector" in services
        assert "rag" in services

        # Overall status should be determined
        assert health_data["overall_status"] in ["healthy", "degraded"]

    @pytest.mark.unit
    async def test_get_server_info(self, mcp_client: Client):
        """Test the server info tool returns comprehensive information."""
        result = await mcp_client.call_tool("get_server_info", {})

        assert result.data is not None
        info_data = result.data

        # Check main sections
        assert "server" in info_data
        assert "capabilities" in info_data
        assert "configuration" in info_data
        assert "available_tools" in info_data

        # Check server details
        server = info_data["server"]
        assert server["name"] == "Crawlerr"
        assert server["framework"] == "FastMCP 2.0+"
        assert server["version"] == "0.1.0"

        # Check capabilities
        capabilities = info_data["capabilities"]
        assert "crawling" in capabilities
        assert "rag" in capabilities
        assert "sources" in capabilities

        # Check crawling capabilities
        crawling = capabilities["crawling"]
        assert crawling["single_page_scraping"] is True
        assert crawling["website_crawling"] is True
        assert crawling["repository_cloning"] is True

        # Check RAG capabilities
        rag = capabilities["rag"]
        assert rag["semantic_search"] is True
        assert rag["automatic_indexing"] is True

        # Check available tools list
        tools = info_data["available_tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert any("scrape" in tool for tool in tools)
        assert any("crawl" in tool for tool in tools)
        assert any("rag_query" in tool for tool in tools)

    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_health_check_with_services(self, mcp_client: Client):
        """Test health check with actual services running."""
        result = await mcp_client.call_tool("health_check", {})

        health_data = result.data
        services = health_data["services"]

        # With real services, these should be healthy
        if "embedding" in services:
            embedding = services["embedding"]
            assert embedding["status"] in ["healthy", "error"]
            if embedding["status"] == "healthy":
                assert "model_info" in embedding

        if "vector" in services:
            vector = services["vector"]
            assert vector["status"] in ["healthy", "error"]
            if vector["status"] == "healthy":
                assert "collection_info" in vector

        if "rag" in services:
            rag = services["rag"]
            assert rag["status"] in ["healthy", "error"]
            if rag["status"] == "healthy":
                assert "component_health" in rag
                assert "stats" in rag
