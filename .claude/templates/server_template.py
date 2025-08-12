from fastmcp import FastMCP
from fastmcp_server.config import ServerConfig


def create_server(config: ServerConfig = None) -> FastMCP:
    if config is None:
        config = ServerConfig()

    mcp = FastMCP(name=config.server_name, instructions=config.server_instructions)

    # Register tools and resources here

    return mcp


def main():
    config = ServerConfig()
    server = create_server(config)

    server.run(
        transport="http",  # Streamable-HTTP
        host=config.host,
        port=config.port,
    )
