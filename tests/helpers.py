"""
Test helper functions for Crawler MCP testing.

This module provides common utilities for testing crawling, embedding,
and RAG functionality with consistent patterns.
"""

import math
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawler_mcp.models.crawl import PageContent
from tests.conftest import EMBEDDING_DIM


def create_mock_crawl_result(
    url: str,
    success: bool = True,
    title: str = "Test Page",
    content: str = "Test content for crawling.",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Create mock crawl result for testing.

    Args:
        url: The URL being crawled
        success: Whether the crawl was successful
        title: Page title
        content: Page content
        metadata: Additional metadata

    Returns:
        Mock crawl result object
    """
    mock_result = MagicMock()
    mock_result.url = url
    mock_result.success = success
    mock_result.title = title
    mock_result.content = content
    mock_result.metadata = metadata or {}
    mock_result.error_message = None if success else "Mock error"

    return mock_result


async def create_mock_crawl_result_async(
    url: str,
    success: bool = True,
    title: str = "Test Page",
    content: str = "Test content for crawling.",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Async wrapper for backward compatibility."""
    return create_mock_crawl_result(url, success, title, content, metadata)


async def mock_embedding_client() -> AsyncMock:
    """
    Create mock embedding client for tests.

    Returns:
        AsyncMock configured to simulate embedding service
    """
    client = AsyncMock()

    # Mock embedding generation
    async def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings with correct dimensions."""
        base_pattern = [0.1, 0.2, 0.3]
        repeat_count = math.ceil(EMBEDDING_DIM / len(base_pattern))
        full_pattern = base_pattern * repeat_count
        return [full_pattern[:EMBEDDING_DIM] for _ in texts]

    client.embed_texts = mock_embed_texts
    client.health_check.return_value = True

    return client


def assert_chunking_preserves_metadata(
    original: PageContent, chunks: list[PageContent]
) -> None:
    """
    Verify metadata preservation during chunking.

    Args:
        original: Original page content
        chunks: List of chunks created from original

    Raises:
        AssertionError: If metadata is not properly preserved
    """
    assert len(chunks) > 0, "No chunks were created"

    # Check that essential metadata is preserved in all chunks
    for i, chunk in enumerate(chunks):
        assert chunk.url == original.url, (
            f"URL not preserved in chunk: {chunk.content[:50]}..."
        )
        assert chunk.title == original.title, (
            f"Title not preserved in chunk: {chunk.content[:50]}..."
        )

        # Check metadata for language if it exists
        if "language" in original.metadata:
            assert chunk.metadata.get("language") == original.metadata["language"], (
                f"Language not preserved in chunk: {chunk.content[:50]}..."
            )

        # Check that chunk-specific fields are set
        assert chunk.content is not None and len(chunk.content) > 0, (
            "Chunk content is empty"
        )

        # Check chunk metadata for chunk-specific fields
        assert chunk.metadata.get("chunk_index", i) is not None, "Chunk index not set"
        assert chunk.metadata.get("total_chunks", len(chunks)) is not None, (
            "Total chunks not set"
        )


def create_mock_page_content(
    url: str = "https://example.com/test",
    title: str = "Test Page",
    content: str = "Test content for processing.",
    language: str = "en",
    file_path: str | None = None,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> PageContent:
    """
    Create PageContent object for testing.

    Args:
        url: Page URL
        title: Page title
        content: Page content
        language: Content language
        file_path: Optional file path for local files
        chunk_index: Chunk index if this is part of chunked content
        total_chunks: Total number of chunks

    Returns:
        PageContent object for testing
    """
    metadata = {
        "language": language,
        "content_hash": "test_hash_123",
        "char_count": len(content),
        "extracted_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
    }

    if file_path is not None:
        metadata["file_path"] = file_path
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    if total_chunks is not None:
        metadata["total_chunks"] = total_chunks

    return PageContent(
        url=url,
        title=title,
        content=content,
        word_count=len(content.split()),
        metadata=metadata,
        timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
    )


async def create_mock_vector_search_result(
    content: str = "Mock search result",
    score: float = 0.95,
    url: str = "https://example.com/result",
) -> Any:
    """
    Create mock vector search result.

    Args:
        content: Result content
        score: Similarity score
        url: Result URL

    Returns:
        Mock search result object
    """
    mock_result = MagicMock()
    mock_result.score = score
    mock_result.document = create_mock_page_content(url=url, content=content)
    return mock_result


def mock_qdrant_client() -> AsyncMock:
    """
    Create mock Qdrant client for testing.

    Returns:
        AsyncMock configured to simulate Qdrant operations
    """
    client = AsyncMock()

    # Mock collection operations
    client.collection_exists.return_value = True
    client.create_collection.return_value = True
    client.delete_collection.return_value = True
    client.health_check.return_value = True

    # Mock upsert operations
    client.upsert.return_value = MagicMock(status="completed")

    # Mock search operations
    async def mock_query_points(*args, **kwargs):
        """Mock query_points with realistic response."""
        mock_response = MagicMock()
        mock_response.points = [await create_mock_vector_search_result()]
        return mock_response

    client.query_points = mock_query_points

    return client


def assert_valid_embedding_dimensions(embeddings: list[list[float]]) -> None:
    """
    Assert that embeddings have correct dimensions.

    Args:
        embeddings: List of embedding vectors

    Raises:
        AssertionError: If dimensions are incorrect
    """
    assert len(embeddings) > 0, "No embeddings provided"

    for i, embedding in enumerate(embeddings):
        assert len(embedding) == EMBEDDING_DIM, (
            f"Embedding {i} has dimension {len(embedding)}, expected {EMBEDDING_DIM}"
        )
        assert all(
            isinstance(x, int | float) and not isinstance(x, bool) for x in embedding
        ), f"Embedding {i} contains non-numeric values or booleans"


class MockProgressCallback:
    """Mock progress callback for testing."""

    def __init__(self):
        self.calls = []

    def __call__(self, current: int, total: int, message: str = ""):
        """Record progress callback calls."""
        self.calls.append({"current": current, "total": total, "message": message})

    def assert_called(self, min_calls: int = 1):
        """Assert the callback was called at least min_calls times."""
        assert len(self.calls) >= min_calls, (
            f"Expected at least {min_calls} progress calls, got {len(self.calls)}"
        )

    def assert_progress_sequence(self):
        """Assert progress calls show increasing progress."""
        if len(self.calls) <= 1:
            return

        for i in range(1, len(self.calls)):
            current_total = self.calls[i]["total"]
            prev_total = self.calls[i - 1]["total"]

            assert current_total > 0, (
                f"Progress total must be > 0 at call {i}, got {current_total}"
            )
            assert prev_total > 0, (
                f"Progress total must be > 0 at call {i - 1}, got {prev_total}"
            )

            current_progress = self.calls[i]["current"] / current_total
            prev_progress = self.calls[i - 1]["current"] / prev_total

            assert current_progress >= prev_progress, (
                f"Progress decreased from {prev_progress:.2f} to {current_progress:.2f}"
            )


# Async test decorators
def async_test(func):
    """Decorator for async test functions."""
    return pytest.mark.asyncio(func)


def requires_services(func):
    """Decorator for tests requiring external services."""
    return pytest.mark.requires_services(func)


def slow_test(func):
    """Decorator for slow-running tests."""
    return pytest.mark.slow(func)


def integration_test(func):
    """Decorator for integration tests."""
    return pytest.mark.integration(func)
