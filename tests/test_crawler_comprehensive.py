"""
Comprehensive crawler tests to maximize coverage on web, directory, and repository crawlers.

This module focuses on testing all code paths in the crawler implementations
to achieve 80%+ total project coverage.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from crawler_mcp.crawlers.base import BaseCrawlStrategy
from crawler_mcp.crawlers.directory import DirectoryCrawlStrategy, DirectoryRequest
from crawler_mcp.crawlers.repository import RepositoryCrawlStrategy, RepositoryRequest
from crawler_mcp.crawlers.web import WebCrawlStrategy, suppress_stdout
from crawler_mcp.models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)


class TestWebCrawlStrategyComprehensive:
    """Comprehensive tests for WebCrawlStrategy to maximize coverage."""

    @pytest.mark.asyncio
    async def test_validate_request_edge_cases(self):
        """Test validation with edge cases and invalid inputs."""
        crawler = WebCrawlStrategy()

        # Test Pydantic validation errors (these should raise ValidationError)
        with pytest.raises(ValidationError):
            CrawlRequest(url="https://example.com", max_pages=0)

        with pytest.raises(ValidationError):
            CrawlRequest(url="https://example.com", max_pages=3000)

        with pytest.raises(ValidationError):
            CrawlRequest(url="https://example.com", max_depth=0)

        with pytest.raises(ValidationError):
            CrawlRequest(url="https://example.com", max_depth=11)

        # Test validate_request method behavior for edge cases that pass Pydantic
        # Empty URL list (after Pydantic validation converts strings to lists)
        request = CrawlRequest(url="https://example.com")
        # Manually set url to empty list to test validate_request method
        request.url = []
        assert await crawler.validate_request(request) is False

        # Valid edge case values that should pass both Pydantic and validate_request
        request = CrawlRequest(url="https://example.com", max_pages=1, max_depth=1)
        assert await crawler.validate_request(request) is True

        request = CrawlRequest(url="https://example.com", max_pages=1000, max_depth=5)
        assert await crawler.validate_request(request) is True

        # Test WebCrawlStrategy specific limits (beyond Pydantic constraints)
        # According to web.py, max_pages limit is 2000 and max_depth is 5
        request = CrawlRequest(url="https://example.com", max_pages=1000, max_depth=5)
        assert await crawler.validate_request(request) is True

    @pytest.mark.asyncio
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()
        assert crawler.memory_manager is None

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()
            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self):
        """Test execute method when memory manager fails to initialize."""
        crawler = WebCrawlStrategy()
        request = CrawlRequest(url="https://example.com")

        # Mock pre_execute_setup to pass, but ensure memory manager stays None
        with (
            patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
            patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
        ):
            # Keep memory manager as None - should return failed CrawlResult
            crawler.memory_manager = None
            result = await crawler.execute(request)

            # Should return a failed CrawlResult, not raise RuntimeError
            assert result.status == CrawlStatus.FAILED
            assert len(result.errors) > 0
            assert "Memory manager not initialized" in result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_with_mocked_crawl4ai(self):
        """Test execute method with comprehensive mocking."""
        crawler = WebCrawlStrategy()
        request = CrawlRequest(
            url="https://example.com",
            max_pages=5,
            max_depth=2,
            include_patterns=["*.html"],
            exclude_patterns=["*.css"],
            extraction_strategy="llm",
        )

        # Mock memory manager
        mock_memory_manager = AsyncMock()
        mock_memory_manager.get_memory_usage.return_value = {
            "used": 1000,
            "total": 8000,
        }
        mock_memory_manager.can_handle_crawl.return_value = True
        mock_memory_manager.check_memory_pressure.return_value = False

        # Create a mock that will match the fallback condition in web.py:
        # hasattr(crawl_result, "success") and hasattr(crawl_result, "url")
        mock_crawl4ai_result = MagicMock()

        # Ensure it doesn't match other conditions first
        # Remove __aiter__ completely so hasattr returns False
        if hasattr(mock_crawl4ai_result, "__aiter__"):
            delattr(mock_crawl4ai_result, "__aiter__")
        # Also ensure it's not a list
        mock_crawl4ai_result.__class__ = type("MockCrawlResult", (), {})

        # Make it match the fallback condition
        mock_crawl4ai_result.success = True
        mock_crawl4ai_result.url = "https://example.com"

        # Add all required attributes for _to_page_content
        mock_crawl4ai_result.html = "<html>Test content</html>"

        # Create proper markdown mock that _safe_get_markdown can use
        # The method requires len(content) > 16 to avoid hash placeholders
        test_content = (
            "# Test content with enough text to pass the 16 character minimum"
        )

        mock_markdown = MagicMock()
        mock_markdown.fit_markdown = test_content
        mock_markdown.raw_markdown = test_content

        # Ensure the attributes exist and return the right values
        mock_markdown.configure_mock(
            **{"fit_markdown": test_content, "raw_markdown": test_content}
        )

        mock_crawl4ai_result.markdown = mock_markdown

        mock_crawl4ai_result.media = {"images": [], "videos": []}
        mock_crawl4ai_result.links = {
            "internal": ["https://example.com/page1"],
            "external": [],
        }
        mock_crawl4ai_result.metadata = {"title": "Test Page"}
        mock_crawl4ai_result.status_code = 200
        mock_crawl4ai_result.response_headers = {"content-type": "text/html"}

        # Add required attributes for sanitization
        mock_crawl4ai_result._markdown = mock_markdown
        mock_crawl4ai_result.error_message = ""

        with (
            patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager,
            patch("crawler_mcp.crawlers.web.AsyncWebCrawler") as mock_crawler_class,
            patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
            patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
        ):
            # Setup mocks
            mock_get_manager.return_value = mock_memory_manager
            mock_crawler_instance = AsyncMock()
            mock_crawler_class.return_value = mock_crawler_instance

            # Mock async context manager behavior
            mock_crawler_instance.__aenter__.return_value = mock_crawler_instance
            mock_crawler_instance.__aexit__.return_value = None

            # Mock browser lifecycle methods
            mock_crawler_instance.start = AsyncMock()
            mock_crawler_instance.close = AsyncMock()

            # Mock arun to return single CrawlResult (not streaming)
            mock_crawler_instance.arun.return_value = mock_crawl4ai_result

            # Execute the crawl
            result = await crawler.execute(request)

            # Verify results
            assert isinstance(result, CrawlResult)
            assert result.status == CrawlStatus.COMPLETED
            assert len(result.pages) == 1
            assert result.pages[0].url == "https://example.com"
            assert "Test content with enough text" in result.pages[0].content
            assert result.statistics.total_pages_crawled == 1

    def test_suppress_stdout_context_manager(self):
        """Test the suppress_stdout context manager."""
        import sys

        # Capture original stdout
        original_stdout = sys.stdout

        # Test that stdout is suppressed
        with suppress_stdout():
            print("This should not appear")
            # Verify stdout is redirected
            assert sys.stdout != original_stdout

        # Verify stdout is restored
        assert sys.stdout == original_stdout


class TestDirectoryCrawlStrategyComprehensive:
    """Comprehensive tests for DirectoryCrawlStrategy to maximize coverage."""

    @pytest.mark.asyncio
    async def test_validate_request_edge_cases(self):
        """Test validation with various edge cases."""
        crawler = DirectoryCrawlStrategy()

        # Non-existent directory
        request = DirectoryRequest(directory_path="/nonexistent/path")
        assert await crawler.validate_request(request) is False

        # File instead of directory
        with tempfile.NamedTemporaryFile() as tmp_file:
            request = DirectoryRequest(directory_path=tmp_file.name)
            assert await crawler.validate_request(request) is False

        # Valid directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            request = DirectoryRequest(directory_path=tmp_dir)
            assert await crawler.validate_request(request) is True

    @pytest.mark.asyncio
    async def test_execute_comprehensive(self):
        """Test execute method with comprehensive file processing."""
        crawler = DirectoryCrawlStrategy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            test_files = {
                "test.py": "# Python file\nprint('hello')",
                "test.md": "# Markdown\nContent here",
                "test.txt": "Plain text content",
                "subdir/nested.py": "# Nested Python file",
                "ignored.log": "Log content",  # Should be filtered
                ".hidden": "Hidden file",
            }

            for file_path, content in test_files.items():
                full_path = Path(tmp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Test with patterns
            request = DirectoryRequest(
                directory_path=tmp_dir, file_patterns=["*.py", "*.md"], recursive=True
            )

            with (
                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
            ):
                result = await crawler.execute(request)

                # Verify results
                assert isinstance(result, CrawlResult)
                assert result.status == CrawlStatus.COMPLETED
                assert len(result.pages) >= 2  # Should find .py and .md files

                # Check that files were processed correctly
                found_files = [
                    page.metadata.get("file_path", "") for page in result.pages
                ]
                assert any("test.py" in f for f in found_files)
                assert any("test.md" in f for f in found_files)

    @pytest.mark.asyncio
    async def test_scan_directory_error_handling(self):
        """Test error handling in directory scanning."""
        crawler = DirectoryCrawlStrategy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            request = DirectoryRequest(directory_path=tmp_dir)

            # Mock pathlib.Path.iterdir to raise an exception
            with patch(
                "pathlib.Path.iterdir", side_effect=PermissionError("Access denied")
            ):
                result = await crawler.execute(request)

                # Should handle error gracefully - directory crawler may just log warnings
                # instead of adding to errors list for permission issues
                assert result.status == CrawlStatus.COMPLETED
                # The directory crawler handles permission errors gracefully
                # by continuing and just warning about no files found
                assert (
                    len(result.pages) == 0
                )  # No pages should be found due to permission error


class TestRepositoryCrawlStrategyComprehensive:
    """Comprehensive tests for RepositoryCrawlStrategy to maximize coverage."""

    @pytest.mark.asyncio
    async def test_validate_request_edge_cases(self):
        """Test validation with various repository URLs."""
        crawler = RepositoryCrawlStrategy()

        # Empty repo URL
        request = RepositoryRequest(repo_url="")
        assert await crawler.validate_request(request) is False

        # Invalid repo URL
        request = RepositoryRequest(repo_url="not-a-url")
        assert await crawler.validate_request(request) is False

        # Valid GitHub URL
        request = RepositoryRequest(repo_url="https://github.com/user/repo")
        assert await crawler.validate_request(request) is True

        # Valid GitLab URL
        request = RepositoryRequest(repo_url="https://gitlab.com/user/repo")
        assert await crawler.validate_request(request) is True

        # SSH format - should fail with current implementation (no scheme/netloc in urlparse)
        request = RepositoryRequest(repo_url="git@github.com:user/repo.git")
        assert await crawler.validate_request(request) is False

    @pytest.mark.asyncio
    async def test_execute_with_mocked_git(self):
        """Test execute method with mocked git operations."""
        crawler = RepositoryCrawlStrategy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            request = RepositoryRequest(
                repo_url="https://github.com/test/repo",
                clone_path=tmp_dir,
                file_patterns=["*.py"],
            )

            # Create mock repository structure
            repo_dir = Path(tmp_dir) / "repo"
            repo_dir.mkdir()
            (repo_dir / "main.py").write_text("# Main Python file")
            (repo_dir / "utils.py").write_text("# Utilities")
            (repo_dir / "README.md").write_text("# Repository")

            # Mock git subprocess call and directory strategy
            async def mock_create_subprocess_exec(*args, **kwargs):
                mock_process = MagicMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(return_value=(b"", b""))
                return mock_process

            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=mock_create_subprocess_exec,
                ),
                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
                patch.object(crawler, "_clone_repository", return_value=repo_dir),
            ):
                result = await crawler.execute(request)

                    # Verify results
                    assert isinstance(result, CrawlResult)
                    assert result.status == CrawlStatus.COMPLETED
                    assert len(result.pages) >= 1  # Should find Python files

    @pytest.mark.asyncio
    async def test_execute_git_clone_error(self):
        """Test error handling when git clone fails."""
        crawler = RepositoryCrawlStrategy()

        request = RepositoryRequest(repo_url="https://github.com/nonexistent/repo")

        # Mock git subprocess to fail
        async def mock_create_subprocess_exec(*args, **kwargs):
            mock_process = MagicMock()
            mock_process.returncode = 1  # Git clone failure
            mock_process.communicate = AsyncMock(
                return_value=(b"", b"Repository not found")
            )
            return mock_process

        with (
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=mock_create_subprocess_exec,
            ),
            patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
            patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
            patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
        ):
            result = await crawler.execute(request)

            # Should handle error gracefully
            assert result.status == CrawlStatus.FAILED
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_git_available_check(self):
        """Test git availability check."""
        crawler = RepositoryCrawlStrategy()

        # Mock git check success
        async def mock_create_subprocess_exec(*args, **kwargs):
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(
                return_value=(b"git version 2.39.0", b"")
            )
            return mock_process

        with patch(
            "asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec
        ):
            result = await crawler._check_git_available()
            assert result is True

        # Mock git check failure (git not found)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            result = await crawler._check_git_available()
            assert result is False


class TestBaseCrawlStrategy:
    """Test the base crawler strategy class."""

    @pytest.mark.asyncio
    async def test_base_strategy_abstract_methods(self):
        """Test that base strategy cannot be instantiated directly."""
        # Should raise TypeError when trying to instantiate abstract class
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseCrawlStrategy()

    @pytest.mark.asyncio
    async def test_pre_post_execute_methods(self):
        """Test pre and post execute methods on concrete implementation."""
        # Use concrete implementation to test base class methods
        crawler = WebCrawlStrategy()

        # These should run without error (default implementations)
        await crawler.pre_execute_setup()
        await crawler.post_execute_cleanup()

        # Verify logger is set
        assert crawler.logger is not None
        assert isinstance(crawler.logger, logging.Logger)
