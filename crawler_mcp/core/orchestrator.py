"""
Slim crawl orchestrator that delegates to specialized managers and strategies.
Replaces the massive crawler_service.py with a clean, maintainable architecture.
"""

import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..config import settings
from ..crawlers import (
    DirectoryCrawlStrategy,
    DirectoryRequest,
    RepositoryCrawlStrategy,
    RepositoryRequest,
    WebCrawlStrategy,
)
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .memory import MemoryManager, cleanup_memory_manager, get_memory_manager

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output (redirect to devnull)."""
    old_stdout = sys.stdout
    try:
        # Redirect stdout to devnull to prevent interference with MCP protocol
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout


class CrawlerService:
    """
    Optimized crawler orchestrator with modular architecture.

    This replaces the 2189-line monolithic crawler_service.py with a clean,
    maintainable service that delegates to specialized components:
    - MemoryManager: 70% threshold with predictive cleanup
    - Strategies: Modular crawling for web, directory, repository
    - Direct AsyncWebCrawler usage without custom browser pooling
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Managers (initialized lazily)
        self._memory_manager: MemoryManager | None = None

        # Strategies
        self._web_strategy = WebCrawlStrategy()
        self._directory_strategy = DirectoryCrawlStrategy()
        self._repository_strategy = RepositoryCrawlStrategy()

        # State
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure all managers are initialized."""
        if self._initialized:
            return

        self.logger.info("Initializing crawler service with optimized components")

        # Initialize managers
        self._memory_manager = get_memory_manager()

        self.logger.info(
            f"Crawler service initialized - "
            f"Memory threshold: {settings.crawl_memory_threshold}%, "
            f"Direct AsyncWebCrawler usage enabled"
        )

        self._initialized = True

    async def crawl_website(
        self,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a website using the optimized web strategy.

        Args:
            request: Web crawl request
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with crawled pages and statistics
        """
        await self._ensure_initialized()

        self.logger.info(f"Starting website crawl: {request.url}")

        # Validate request
        if not await self._web_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_web_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[request.url] if isinstance(request.url, str) else request.url,
                pages=[],
                errors=["Invalid web crawl request"],
                statistics=getattr(request, "statistics", CrawlStatistics()),
            )

        # Execute web crawling strategy
        return await self._web_strategy.execute(request, progress_callback)

    async def crawl_directory(
        self,
        directory_path: str,
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a local directory using the optimized directory strategy.

        Args:
            directory_path: Path to directory to crawl
            file_patterns: File patterns to include (e.g., ["*.py", "*.md"])
            recursive: Whether to crawl subdirectories
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with processed files and statistics
        """
        await self._ensure_initialized()

        self.logger.info(f"Starting directory crawl: {directory_path}")

        # Create directory request
        request = DirectoryRequest(
            directory_path=directory_path,
            file_patterns=file_patterns,
            recursive=recursive,
        )

        # Validate request
        if not await self._directory_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_dir_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[directory_path],
                pages=[],
                errors=["Invalid directory crawl request"],
                statistics=CrawlStatistics(),
            )

        # Execute directory crawling strategy
        return await self._directory_strategy.execute(request, progress_callback)

    async def crawl_repository(
        self,
        repo_url: str,
        clone_path: str | None = None,
        file_patterns: list[str] | None = None,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a git repository using the optimized repository strategy.

        Args:
            repo_url: Git repository URL
            clone_path: Optional custom clone path
            file_patterns: File patterns to include
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with analyzed repository files and statistics
        """
        await self._ensure_initialized()

        self.logger.info(f"Starting repository crawl: {repo_url}")

        # Create repository request
        request = RepositoryRequest(
            repo_url=repo_url,
            clone_path=clone_path,
            file_patterns=file_patterns,
        )

        # Validate request
        if not await self._repository_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_repo_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[repo_url],
                pages=[],
                errors=["Invalid repository crawl request"],
                statistics=CrawlStatistics(),
            )

        # Execute repository crawling strategy
        return await self._repository_strategy.execute(request, progress_callback)

    async def scrape_single_page(
        self,
        url: str,
        extraction_strategy: str = "css",
        wait_for: str | None = None,
        custom_config: dict[str, Any] | None = None,
        use_virtual_scroll: bool = False,
        virtual_scroll_config: dict[str, Any] | None = None,
    ) -> PageContent:
        """
        Scrape a single page using direct AsyncWebCrawler.

        Args:
            url: URL to scrape
            extraction_strategy: Extraction strategy to use
            wait_for: CSS selector or JS condition to wait for
            custom_config: Custom configuration options
            use_virtual_scroll: Whether to use virtual scrolling
            virtual_scroll_config: Virtual scroll configuration

        Returns:
            PageContent for the scraped page
        """
        await self._ensure_initialized()

        self.logger.debug(f"Scraping single page: {url}")

        from crawl4ai import AsyncWebCrawler, BrowserConfig  # type: ignore
        from crawl4ai.content_filter_strategy import (
            PruningContentFilter,  # type: ignore
        )
        from crawl4ai.markdown_generation_strategy import (
            DefaultMarkdownGenerator,  # type: ignore
        )

        # Create minimal browser config
        browser_config = BrowserConfig(
            headless=settings.crawl_headless,
            browser_type=settings.crawl_browser,
            light_mode=True,
            verbose=False,  # Suppress Crawl4AI output for MCP compatibility
            text_mode=getattr(settings, "crawl_block_images", False),
        )

        with suppress_stdout():
            browser = AsyncWebCrawler(config=browser_config)
            await browser.start()

        try:
            # Create content filter for fit markdown generation
            content_filter = PruningContentFilter(
                threshold=0.45,  # Prune nodes below 45% relevance score
                threshold_type="dynamic",  # Dynamic scoring
                min_word_threshold=5,  # Ignore very short text blocks
            )

            # Create markdown generator with content filter
            markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

            # Use Crawl4AI to scrape the page with fit markdown optimization
            with suppress_stdout():
                result = await browser.arun(
                    url=url,
                    bypass_cache=not settings.crawl_enable_caching,
                    process_iframes=False,  # Disable for performance
                    remove_overlay_elements=settings.crawl_remove_overlays,
                    word_count_threshold=settings.crawl_min_words,
                    # Optimize for clean markdown extraction
                    excluded_tags=[
                        "nav",
                        "footer",
                        "header",
                        "aside",
                        "script",
                        "style",
                    ],
                    exclude_external_links=True,
                    markdown_generator=markdown_generator,  # Enable fit markdown generation
                )

            if not result.success:
                raise Exception(f"Scraping failed: {result.error_message}")

            # Create PageContent - prioritize fit markdown for clean content
            page_content = PageContent(
                url=url,
                title=result.metadata.get("title", ""),
                content=getattr(result.markdown, "fit_markdown", None)
                or result.markdown
                or result.cleaned_html
                or "",
                html=result.html,
                markdown=getattr(result.markdown, "fit_markdown", None)
                or result.markdown,
                links=[
                    link.get("href", link) if isinstance(link, dict) else link
                    for link in result.links.get("internal", [])
                ]
                if result.links
                else [],
                images=[
                    img.get("src", img) if isinstance(img, dict) else img
                    for img in result.media.get("images", [])
                ]
                if result.media
                else [],
                metadata={
                    "extraction_strategy": extraction_strategy,
                    "word_count": len(
                        (
                            getattr(result.markdown, "fit_markdown", None)
                            or result.markdown
                            or ""
                        ).split()
                    ),
                    "status_code": result.status_code,
                    "response_headers": dict(result.response_headers or {}),
                },
                timestamp=datetime.fromtimestamp(time.time()),
                word_count=len(
                    (
                        getattr(result.markdown, "fit_markdown", None)
                        or result.markdown
                        or ""
                    ).split()
                ),
            )

            return page_content

        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            raise
        finally:
            with suppress_stdout():
                await browser.close()

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all crawler components."""
        await self._ensure_initialized()

        health_status: dict[str, Any] = {
            "crawler_service": "healthy",
            "components": {},
            "performance_optimizations": {
                "memory_threshold": f"{settings.crawl_memory_threshold}%",
                "direct_browser_usage": True,
                "light_mode_enabled": True,
                "streaming_enabled": settings.crawl_enable_streaming,
                "caching_enabled": settings.crawl_enable_caching,
            },
        }

        # Memory manager health
        if self._memory_manager:
            memory_stats = self._memory_manager.get_stats()
            health_status["components"]["memory_manager"] = {
                "status": "healthy",
                "stats": memory_stats,
            }

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all crawler resources."""
        self.logger.info("Cleaning up crawler service")

        # Cleanup memory manager
        if self._memory_manager:
            cleanup_memory_manager()

        self._initialized = False
        self.logger.info("Crawler service cleanup completed")

    async def __aenter__(self) -> "CrawlerService":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
