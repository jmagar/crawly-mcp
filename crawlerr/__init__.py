"""
Crawlerr - FastMCP RAG-enabled web crawling MCP server

A production-ready MCP server that combines:
- Advanced web crawling with Crawl4AI 0.7.0
- Vector search with Qdrant
- Text embeddings via HF TEI
- Comprehensive source management
- Real-time progress reporting
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from fastmcp import FastMCP

__version__ = "0.1.0"
__author__ = "Crawlerr Team"
__description__ = "RAG-enabled web crawling MCP server"

# Main exports - use lazy imports to avoid module conflicts
from .config import settings

__all__ = ["get_main", "get_mcp", "settings"]

# Lazy imports to prevent conflicts when running server module directly
def get_mcp() -> FastMCP:
    """Get the FastMCP server instance."""
    from .server import mcp
    return mcp

def get_main() -> Callable[[], None]:
    """Get the main CLI entry point function."""
    from .server import main  
    return main