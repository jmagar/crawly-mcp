"""
Slim crawl orchestrator that delegates to specialized managers and strategies.
Replaces the massive crawler_service.py with a clean, maintainable architecture.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from ..config import settings
from ..models.crawl_models import CrawlRequest, CrawlResult, CrawlStatus, PageContent
from .browser_manager import cleanup_browser_pool, get_browser_pool
from .gpu_manager import cleanup_gpu_manager, get_gpu_manager
from .memory_manager import cleanup_memory_manager, get_memory_manager
from .strategies.directory_strategy import DirectoryCrawlStrategy, DirectoryRequest
from .strategies.repository_strategy import RepositoryCrawlStrategy, RepositoryRequest
from .strategies.web_strategy import WebCrawlStrategy

logger = logging.getLogger(__name__)


class CrawlerService:
    """
    Optimized crawler orchestrator with modular architecture.

    This replaces the 2189-line monolithic crawler_service.py with a clean,
    maintainable service that delegates to specialized components:
    - BrowserManager: Session pooling and browser lifecycle
    - GPUManager: GPU monitoring with reduced overhead
    - MemoryManager: 70% threshold with predictive cleanup
    - Strategies: Modular crawling for web, directory, repository
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Managers (initialized lazily)
        self._browser_pool = None
        self._gpu_manager = None
        self._memory_manager = None

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
        self._browser_pool = await get_browser_pool()
        self._gpu_manager = await get_gpu_manager()
        self._memory_manager = get_memory_manager()

        self.logger.info(
            f"Crawler service initialized - Browser pool: {settings.browser_pool_size}, "
            f"Memory threshold: {settings.crawl_memory_threshold}%, "
            f"GPU monitoring: {settings.gpu_monitor_interval}s intervals"
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
                urls=[request.url],
                pages=[],
                errors=["Invalid web crawl request"],
                statistics=request.statistics
                if hasattr(request, "statistics")
                else None,
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
                statistics=None,
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
                statistics=None,
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
        Scrape a single page with optimized browser session reuse.

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

        try:
            # Get optimized browser session (with caching)
            browser = await self._browser_pool.get_session(url)

            # Use Crawl4AI to scrape the page
            result = await browser.arun(
                url=url,
                bypass_cache=not settings.crawl_enable_caching,
                process_iframes=False,  # Disable for performance
                remove_overlay_elements=settings.crawl_remove_overlays,
                word_count_threshold=settings.crawl_min_words,
            )

            if not result.success:
                raise Exception(f"Scraping failed: {result.error_message}")

            # Create PageContent
            page_content = PageContent(
                url=url,
                title=result.metadata.get("title", ""),
                content=result.cleaned_html or result.markdown or "",
                html=result.html,
                markdown=result.markdown,
                links=list(result.links.get("internal", [])) if result.links else [],
                images=list(result.media.get("images", [])) if result.media else [],
                metadata={
                    "extraction_strategy": extraction_strategy,
                    "word_count": len((result.cleaned_html or "").split()),
                    "status_code": result.status_code,
                    "response_headers": dict(result.response_headers or {}),
                },
                timestamp=time.time(),
                word_count=len((result.cleaned_html or "").split()),
            )

            return page_content

        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            raise

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all crawler components."""
        await self._ensure_initialized()

        health_status = {
            "crawler_service": "healthy",
            "components": {},
            "performance_optimizations": {
                "memory_threshold": f"{settings.crawl_memory_threshold}%",
                "browser_pool_size": settings.browser_pool_size,
                "gpu_monitor_interval": f"{settings.gpu_monitor_interval}s",
                "streaming_enabled": settings.crawl_enable_streaming,
                "caching_enabled": settings.crawl_enable_caching,
            },
        }

        # Browser pool health
        if self._browser_pool:
            pool_stats = await self._browser_pool.get_stats()
            health_status["components"]["browser_pool"] = {
                "status": "healthy",
                "stats": pool_stats,
            }

        # GPU manager health
        if self._gpu_manager:
            gpu_stats = self._gpu_manager.get_stats()
            health_status["components"]["gpu_manager"] = {
                "status": "healthy" if gpu_stats.get("gpu_healthy") else "degraded",
                "stats": gpu_stats,
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

        # Cleanup in reverse initialization order
        if self._browser_pool:
            await cleanup_browser_pool()

        if self._gpu_manager:
            await cleanup_gpu_manager()

        if self._memory_manager:
            cleanup_memory_manager()

        self._initialized = False
        self.logger.info("Crawler service cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
