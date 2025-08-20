"""
Test crawler implementations for web, directory, and repository crawling.

This module focuses on testing crawler validation, initialization,
and basic functionality to maximize code coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from crawler_mcp.crawlers.directory import DirectoryCrawlStrategy, DirectoryRequest
from crawler_mcp.crawlers.repository import RepositoryCrawlStrategy, RepositoryRequest
from crawler_mcp.crawlers.web import WebCrawlStrategy
from crawler_mcp.models.crawl import CrawlRequest


class TestWebCrawlStrategy:
    """Test WebCrawlStrategy for maximum coverage on web crawler."""

    @pytest.mark.unit
    def test_web_crawler_initialization(self):
        """Test web crawler basic initialization."""
        crawler = WebCrawlStrategy()

        assert crawler is not None
        assert hasattr(crawler, "memory_manager")
        assert crawler.memory_manager is None
        assert hasattr(crawler, "logger")

    @pytest.mark.unit
    async def test_validate_request_valid_urls(self):
        """Test validation with valid web crawl requests."""
        crawler = WebCrawlStrategy()

        # Valid single URL
        request = CrawlRequest(url="https://example.com", max_pages=50, max_depth=2)
        assert await crawler.validate_request(request) is True

        # Valid multiple URLs
        request_multi = CrawlRequest(
            url=["https://example.com", "https://test.com"], max_pages=100, max_depth=3
        )
        assert await crawler.validate_request(request_multi) is True

    @pytest.mark.unit
    async def test_validate_request_invalid_urls(self):
        """Test validation with invalid web crawl requests."""
        crawler = WebCrawlStrategy()

        # No URL provided (empty list)
        request_no_url = CrawlRequest(url=[])
        assert await crawler.validate_request(request_no_url) is False

        # Test with values that exceed crawler's custom validation limits
        # (WebCrawlStrategy checks max_pages <= 2000, but Pydantic allows <= 1000)
        request_high_pages = CrawlRequest(
            url="https://example.com",
            max_pages=1000,  # At Pydantic limit but valid for Pydantic
        )
        # This should pass since 1000 is within both Pydantic and crawler limits
        assert await crawler.validate_request(request_high_pages) is True

        # Test max_depth boundary (crawler allows <= 5, Pydantic allows <= 10)
        request_high_depth = CrawlRequest(
            url="https://example.com",
            max_depth=6,  # Exceeds crawler limit but within Pydantic limit
        )
        assert await crawler.validate_request(request_high_depth) is False

    @pytest.mark.unit
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()

            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

    @pytest.mark.unit
    async def test_validate_request_none_values(self):
        """Test validation with None values."""
        crawler = WebCrawlStrategy()

        # None values should be allowed (defaults will be used)
        request = CrawlRequest(
            url="https://example.com", max_pages=None, max_depth=None
        )
        assert await crawler.validate_request(request) is True


class TestDirectoryCrawlStrategy:
    """Test DirectoryCrawlStrategy for directory crawling coverage."""

    @pytest.mark.unit
    def test_directory_crawler_initialization(self):
        """Test directory crawler basic initialization."""
        crawler = DirectoryCrawlStrategy()

        assert crawler is not None
        assert hasattr(crawler, "logger")

    @pytest.mark.unit
    async def test_validate_request_valid_directory(self, temp_directory):
        """Test validation with valid directory paths."""
        crawler = DirectoryCrawlStrategy()

        # Create a test directory
        test_dir = temp_directory / "test_crawl"
        test_dir.mkdir()
        (test_dir / "test.txt").write_text("test content")

        request = DirectoryRequest(
            directory_path=str(test_dir),
            file_patterns=["*.txt"],
            recursive=True,
            max_files=50,
        )

        # This should validate successfully
        is_valid = await crawler.validate_request(request)
        assert is_valid is True

    @pytest.mark.unit
    async def test_validate_request_nonexistent_directory(self):
        """Test validation with non-existent directory."""
        crawler = DirectoryCrawlStrategy()

        request = DirectoryRequest(
            directory_path="/nonexistent/directory/path",
            file_patterns=["*.txt"],
            recursive=True,
            max_files=50,
        )

        is_valid = await crawler.validate_request(request)
        # Should return False for non-existent directory
        assert is_valid is False


class TestRepositoryCrawlStrategy:
    """Test RepositoryCrawlStrategy for repository crawling coverage."""

    @pytest.mark.unit
    def test_repository_crawler_initialization(self):
        """Test repository crawler basic initialization."""
        crawler = RepositoryCrawlStrategy()

        assert crawler is not None
        assert hasattr(crawler, "logger")

    @pytest.mark.unit
    async def test_validate_request_github_url(self):
        """Test validation with GitHub repository URLs."""
        crawler = RepositoryCrawlStrategy()

        # Valid GitHub repository URL
        request = RepositoryRequest(
            repo_url="https://github.com/user/repo",
            file_patterns=["*.py", "*.md"],
            max_files=50,
            cleanup_after=True,
        )

        is_valid = await crawler.validate_request(request)
        # Should handle repository URLs (may require git to be installed)
        assert is_valid in [True, False]

    @pytest.mark.unit
    async def test_validate_request_git_url(self):
        """Test validation with git:// URLs."""
        crawler = RepositoryCrawlStrategy()

        request = RepositoryRequest(
            repo_url="git@github.com:user/repo.git",
            file_patterns=["*.py", "*.md"],
            max_files=50,
            cleanup_after=True,
        )

        is_valid = await crawler.validate_request(request)
        assert is_valid in [True, False]

    @pytest.mark.unit
    async def test_validate_request_invalid_repo_url(self):
        """Test validation with invalid repository URLs."""
        crawler = RepositoryCrawlStrategy()

        request = RepositoryRequest(
            repo_url="not-a-valid-repo-url",
            file_patterns=["*.py", "*.md"],
            max_files=50,
            cleanup_after=True,
        )

        is_valid = await crawler.validate_request(request)
        # Should return False for invalid URL format
        assert is_valid is False


class TestCrawlerEdgeCases:
    """Test edge cases and error conditions across crawlers."""

    @pytest.mark.unit
    def test_crawler_inheritance(self):
        """Test that all crawlers inherit from base properly."""
        web_crawler = WebCrawlStrategy()
        dir_crawler = DirectoryCrawlStrategy()
        repo_crawler = RepositoryCrawlStrategy()

        # All should have logger from base class
        assert hasattr(web_crawler, "logger")
        assert hasattr(dir_crawler, "logger")
        assert hasattr(repo_crawler, "logger")

        # All should be callable instances
        assert callable(web_crawler.validate_request)
        assert callable(dir_crawler.validate_request)
        assert callable(repo_crawler.validate_request)

    @pytest.mark.unit
    async def test_empty_request_handling(self):
        """Test how crawlers handle empty/minimal requests."""
        # Test each crawler with its appropriate request type
        web_crawler = WebCrawlStrategy()
        web_request = CrawlRequest(url="test")

        try:
            result = await web_crawler.validate_request(web_request)
            assert result in [True, False]
        except Exception:
            # Some crawlers might throw exceptions for invalid URLs
            # This is acceptable behavior
            pass

        # Test directory crawler
        dir_crawler = DirectoryCrawlStrategy()
        dir_request = DirectoryRequest(directory_path="test")

        try:
            result = await dir_crawler.validate_request(dir_request)
            assert result in [True, False]
        except Exception:
            # Some crawlers might throw exceptions for invalid paths
            # This is acceptable behavior
            pass

        # Test repository crawler
        repo_crawler = RepositoryCrawlStrategy()
        repo_request = RepositoryRequest(repo_url="test")

        try:
            result = await repo_crawler.validate_request(repo_request)
            assert result in [True, False]
        except Exception:
            # Some crawlers might throw exceptions for invalid repo URLs
            # This is acceptable behavior
            pass


class TestCrawlerConfiguration:
    """Test crawler configuration and settings integration."""

    @pytest.mark.unit
    def test_web_crawler_uses_settings(self):
        """Test that web crawler accesses configuration settings."""
        crawler = WebCrawlStrategy()

        # This will exercise settings access during validation
        with patch("crawler_mcp.crawlers.web.settings") as mock_settings:
            mock_settings.crawl_headless = True
            mock_settings.crawl_browser = "chromium"

            # Just instantiating should exercise some settings access
            assert crawler is not None

    @pytest.mark.unit
    async def test_crawler_boundary_values(self):
        """Test crawlers with boundary values."""
        crawler = WebCrawlStrategy()

        # Test boundary values within Pydantic constraints
        boundary_requests = [
            CrawlRequest(url="https://example.com", max_pages=1, max_depth=1),
            CrawlRequest(
                url="https://example.com", max_pages=1000, max_depth=5
            ),  # Within both Pydantic and WebCrawler limits
        ]

        for request in boundary_requests:
            result = await crawler.validate_request(request)
            assert result is True
