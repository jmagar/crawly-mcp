import pytest
from fastmcp import Client, FastMCP

# Import the server factory or instance from your main server file
# from project_name.server import create_server


# This is a pytest fixture. It creates a new server instance for each test function.
# This ensures tests are isolated from each other.
@pytest.fixture
def mcp_server() -> FastMCP:
    """Provides a new FastMCP server instance for testing."""
    # In a real project, you would import and call your create_server() factory.
    # e.g., server = create_server()
    # For this template, we'll create a simple server directly.
    server = FastMCP(name="TestServer")

    @server.tool
    def add(x: int, y: int) -> int:
        """Adds two numbers."""
        return x + y

    return server


@pytest.mark.asyncio
async def test_add_tool_in_memory(mcp_server: FastMCP):
    """
    Tests the 'add' tool using the recommended in-memory pattern.
    """
    # The Client is initialized directly with the server instance.
    # No network communication or separate process is needed.
    async with Client(mcp_server) as client:
        # Call the tool
        result = await client.call_tool("add", {"x": 5, "y": 10})

        # Assert the result
        assert result.data == 15
