"""
Test package for Crawler MCP FastMCP server.

This __init__.py file is intentionally present to make tests/ a Python package.
This is required for inter-test imports, specifically:
- tests/helpers.py imports EMBEDDING_DIM from tests.conftest

Removing this file would break these imports. Pytest can still discover and run
tests correctly with this file present.
"""
