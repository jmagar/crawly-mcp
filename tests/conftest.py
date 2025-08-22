"""
Pytest configuration and fixtures for Crawler MCP testing.

This module provides comprehensive test fixtures using real services
(Qdrant and HF TEI) for integration testing with FastMCP Client.
"""

import asyncio
import contextlib
import os
import tempfile
import warnings
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from crawler_mcp.config import settings
from crawler_mcp.core import EmbeddingService, RagService, VectorService

# Fix NumPy 2.x + SciPy 1.16.1 compatibility issue for coverage measurement
# Force NumPy and SciPy imports before coverage instrumentation to prevent conflicts
try:
    import numpy as np
    import scipy.stats

    # Force initialization to happen before pytest-cov instruments the code
    _ = np.__version__
    _ = scipy.stats.__version__
except Exception:
    # If there are import issues, proceed without the imports
    pass

# Suppress warnings that can cause coverage issues
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*int.*argument must be a string.*")
warnings.filterwarnings("ignore", message=".*_NoValueType.*")

# Test constants
EMBEDDING_DIM = 384  # Qwen3-Embedding-0.6B dimension


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """
    Configure global settings with test-specific values.

    Mutates the module-level settings so services pick up test values.
    """
    # Store original values for restoration
    original_values = {
        "qdrant_collection": settings.qdrant_collection,
        "log_level": settings.log_level,
        "debug": settings.debug,
        "production": settings.production,
    }

    # Override with test-specific values
    settings.qdrant_collection = "test_crawler_mcp"  # Separate test collection
    settings.log_level = "WARNING"  # Reduce noise in tests
    settings.debug = False
    settings.production = False

    yield settings

    # Restore original values after tests
    for key, value in original_values.items():
        setattr(settings, key, value)


@pytest.fixture(scope="session")
async def setup_test_services(
    test_settings,
) -> AsyncGenerator[None, None]:
    """
    Ensure test services are ready and create test collection.

    This fixture runs once per test session to set up the test environment.
    """
    # Wait for services to be available
    embedding_ready = False
    vector_ready = False

    try:
        # Check embedding service
        async with EmbeddingService() as embedding_service:
            embedding_ready = await embedding_service.health_check()

        # Check vector service and create test collection
        async with VectorService() as vector_service:
            vector_ready = await vector_service.health_check()
            if vector_ready:
                # Ensure test collection exists
                await vector_service.ensure_collection()

    except Exception as e:
        pytest.skip(f"Required services not available: {e}")

    if not (embedding_ready and vector_ready):
        pytest.skip("Required services (Qdrant and/or TEI) are not healthy")

    yield

    # Cleanup: Delete test collection after tests
    try:
        async with VectorService() as vector_service:
            if hasattr(vector_service, "client") and callable(
                getattr(vector_service.client, "delete_collection", None)
            ):
                await vector_service.client.delete_collection(
                    settings.qdrant_collection
                )
    except Exception:
        pass  # Cleanup is best-effort


@pytest.fixture
async def clean_test_collection(
    test_settings,
) -> AsyncGenerator[None, None]:
    """
    Clean the test collection before each test.

    This ensures each test starts with a fresh vector database state.
    """
    try:
        async with VectorService() as vector_service:
            # Delete and recreate collection for clean state
            with contextlib.suppress(Exception):
                await vector_service.client.delete_collection(
                    settings.qdrant_collection
                )

            await vector_service.ensure_collection()

    except Exception as e:
        pytest.skip(f"Could not clean test collection: {e}")

    yield

    # No cleanup needed - next test will clean


@pytest.fixture
async def embedding_service() -> AsyncGenerator[EmbeddingService, None]:
    """Provide a configured embedding service."""
    async with EmbeddingService() as service:
        yield service


@pytest.fixture
async def vector_service() -> AsyncGenerator[VectorService, None]:
    """Provide a configured vector service."""
    async with VectorService() as service:
        yield service


@pytest.fixture
async def rag_service() -> AsyncGenerator[RagService, None]:
    """Provide a configured RAG service."""
    async with RagService() as service:
        yield service


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_html_file(temp_directory: Path) -> Path:
    """Create a sample HTML file for testing."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Test Content</h1>
        <p>This is a test paragraph with some content for crawling.</p>
        <div class="content">
            <p>More test content here.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    </body>
    </html>
    """

    html_file = temp_directory / "test.html"
    html_file.write_text(html_content)
    return html_file


@pytest.fixture
def sample_text_files(temp_directory: Path) -> list[Path]:
    """Create sample text files for directory crawling tests."""
    files = []

    # Create some nested structure
    (temp_directory / "subdir").mkdir()

    content_map = {
        "readme.md": "# Test Project\nThis is a test readme file.",
        "doc1.txt": "This is the first document with some content.",
        "doc2.txt": "This is the second document with different content.",
        "subdir/nested.txt": "This is a nested document in a subdirectory.",
    }

    for filename, content in content_map.items():
        file_path = temp_directory / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        files.append(file_path)

    return files


@pytest.fixture
async def crawler_mcp_server() -> FastMCP:
    """
    Create a test instance of the Crawler MCP server.

    This fixture provides a clean server instance for in-memory testing.
    """
    # Import here to avoid circular imports and ensure proper initialization
    from crawler_mcp.server import mcp

    # Return the configured server instance
    return mcp


@pytest.fixture
async def mcp_client(
    crawler_mcp_server: FastMCP,
    setup_test_services: None,
    clean_test_collection: None,
) -> AsyncGenerator[Client, None]:
    """
    Create an in-memory FastMCP client for testing.

    This is the main fixture for testing MCP tools. It provides a direct
    in-memory connection to the server without network overhead.
    """
    async with Client(crawler_mcp_server) as client:
        yield client


@pytest.fixture
def skip_if_no_services():
    """
    Skip test if required services are not available.

    Use this fixture to mark tests that require Qdrant and TEI services.
    """

    def _skip_if_services_unavailable():
        if not all(
            [
                os.getenv("QDRANT_URL"),
                os.getenv("TEI_URL"),
            ]
        ):
            pytest.skip("Required services (QDRANT_URL, TEI_URL) not configured")

    return _skip_if_services_unavailable


# Pytest marks for service dependencies
pytestmark = pytest.mark.asyncio
