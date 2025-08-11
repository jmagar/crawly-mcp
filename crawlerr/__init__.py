"""
Crawlerr - FastMCP RAG-enabled web crawling MCP server

A production-ready MCP server that combines:
- Advanced web crawling with Crawl4AI 0.7.0
- Vector search with Qdrant
- Text embeddings via HF TEI
- Comprehensive source management
- Real-time progress reporting
"""

from typing import Any

__version__ = "0.1.0"
__author__ = "Crawlerr Team"
__description__ = "RAG-enabled web crawling MCP server"

# Main exports - use lazy imports to avoid module conflicts
from .config import settings

__all__ = ["settings", "get_mcp", "get_main"]

# Lazy imports to prevent conflicts when running server module directly
def get_mcp() -> Any:
    from .server import mcp
    return mcp

def get_main() -> Any:
    from .server import main  
    return main