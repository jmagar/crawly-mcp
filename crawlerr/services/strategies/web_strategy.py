"""
Optimized web crawling strategy with streaming and caching support.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler
from crawl4ai import CrawlResult as Crawl4aiResult
from crawl4ai.extraction_strategy import CosineStrategy, LLMExtractionStrategy

from ...config import settings
from ...models.crawl_models import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from ..browser_manager import get_browser_pool
from ..gpu_manager import get_gpu_manager
from ..memory_manager import get_memory_manager
from .base_strategy import BaseCrawlStrategy

logger = logging.getLogger(__name__)


class WebCrawlStrategy(BaseCrawlStrategy):
    """
    High-performance web crawling strategy with streaming, caching, and GPU acceleration.
    Optimized for RTX 4070 + i7-13700K performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.browser_pool = None
        self.gpu_manager = None
        self.memory_manager = None

    async def _initialize_managers(self) -> None:
        """Initialize required managers."""
        if not self.browser_pool:
            self.browser_pool = await get_browser_pool()
        if not self.gpu_manager:
            self.gpu_manager = await get_gpu_manager()
        if not self.memory_manager:
            self.memory_manager = get_memory_manager()

    async def validate_request(self, request: CrawlRequest) -> bool:
        """Validate web crawl request."""
        if not request.url:
            self.logger.error("URL is required for web crawling")
            return False

        if request.max_pages < 1 or request.max_pages > 2000:
            self.logger.error(
                f"max_pages must be between 1 and 2000, got {request.max_pages}"
            )
            return False

        if request.max_depth < 1 or request.max_depth > 5:
            self.logger.error(
                f"max_depth must be between 1 and 5, got {request.max_depth}"
            )
            return False

        return True

    async def execute(
        self,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Execute optimized web crawling with streaming and performance enhancements.
        """
        await self._initialize_managers()

        start_time = time.time()
        self.logger.info(
            f"Starting web crawl: {request.url} (max_pages: {request.max_pages}, max_depth: {request.max_depth})"
        )

        # Pre-crawl optimizations
        await self.pre_execute_setup()

        try:
            # Memory check
            if not await self.memory_manager.can_handle_crawl(request.max_pages):
                self.logger.warning(
                    "System may have insufficient memory for crawl, proceeding with caution"
                )

            # Get optimized browser session
            browser = await self.browser_pool.get_session(request.url)

            # Configure for streaming and performance
            crawl_config = await self._build_crawl_config()

            # Execute crawl with streaming
            pages = []
            errors = []
            total_bytes = 0
            unique_domains = set()
            total_links_discovered = 0

            if progress_callback:
                progress_callback(0, request.max_pages, "Starting crawl...")

            # Use Crawl4AI's streaming capabilities
            async for result in self._stream_crawl(browser, request, crawl_config):
                if isinstance(result, Exception):
                    errors.append(str(result))
                    continue

                if isinstance(result, PageContent):
                    pages.append(result)
                    total_bytes += len(result.content)
                    unique_domains.add(urlparse(result.url).netloc)
                    total_links_discovered += len(result.links)

                    # Memory pressure check during crawl
                    if await self.memory_manager.check_memory_pressure():
                        self.logger.warning(
                            "Memory pressure detected during crawl, may slow down"
                        )

                    if progress_callback:
                        progress_callback(
                            len(pages),
                            request.max_pages,
                            f"Crawled: {result.url[:60]}...",
                        )

                    # Break if we've hit our limits
                    if len(pages) >= request.max_pages:
                        break

            # Calculate statistics
            end_time = time.time()
            crawl_duration = end_time - start_time
            pages_per_second = len(pages) / crawl_duration if crawl_duration > 0 else 0
            avg_page_size = total_bytes / len(pages) if pages else 0
            success_rate = (
                len(pages) / (len(pages) + len(errors))
                if (len(pages) + len(errors)) > 0
                else 0
            )

            statistics = CrawlStatistics(
                total_pages_requested=request.max_pages,
                total_pages_crawled=len(pages),
                total_pages_failed=len(errors),
                unique_domains=len(unique_domains),
                total_links_discovered=total_links_discovered,
                total_bytes_downloaded=total_bytes,
                crawl_duration_seconds=crawl_duration,
                pages_per_second=pages_per_second,
                average_page_size=avg_page_size,
            )

            result = CrawlResult(
                request_id=f"web_crawl_{int(time.time())}",
                status=CrawlStatus.COMPLETED,
                urls=[request.url],
                pages=pages,
                errors=errors,
                statistics=statistics,
                success_rate=success_rate,
            )

            self.logger.info(
                f"Web crawl completed: {len(pages)} pages, {crawl_duration:.1f}s, "
                f"{pages_per_second:.1f} pages/sec, {len(errors)} errors"
            )

            return result

        except Exception as e:
            self.logger.error(f"Web crawl failed: {e}")

            # Return partial result on failure
            return CrawlResult(
                request_id=f"web_crawl_failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[request.url],
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

        finally:
            await self.post_execute_cleanup()

    async def _build_crawl_config(self) -> dict[str, Any]:
        """Build optimized crawl configuration."""
        gpu_config = self.gpu_manager.get_gpu_config() if self.gpu_manager else {}

        config = {
            # Streaming and caching
            "streaming": settings.crawl_enable_streaming,
            "cache_enabled": settings.crawl_enable_caching,
            # Performance settings
            "headless": settings.crawl_headless,
            "browser_type": settings.crawl_browser,
            "delay": settings.crawler_delay,
            "timeout": settings.crawler_timeout,
            # Content filtering
            "word_count_threshold": settings.crawl_min_words,
            "remove_overlay_elements": settings.crawl_remove_overlays,
            "extract_media": settings.crawl_extract_media,
            # Resource blocking for performance
            "block_resources": [],
        }

        # Add resource blocking
        if settings.crawl_block_images:
            config["block_resources"].append("image")
        if settings.crawl_block_stylesheets:
            config["block_resources"].append("stylesheet")
        if settings.crawl_block_fonts:
            config["block_resources"].append("font")
        if settings.crawl_block_media:
            config["block_resources"].extend(["media", "video", "audio"])

        # Add GPU configuration if available
        if gpu_config.get("gpu_enabled"):
            config["gpu_enabled"] = True
            config["chrome_flags"] = gpu_config.get("chrome_flags", [])

        return config

    async def _stream_crawl(
        self, browser: AsyncWebCrawler, request: CrawlRequest, config: dict[str, Any]
    ) -> Any:
        """
        Stream crawling results using Crawl4AI's streaming capabilities.
        """
        try:
            # Configure extraction strategy
            extraction_strategy = None
            if hasattr(request, "extraction_strategy") and request.extraction_strategy:
                if request.extraction_strategy == "llm":
                    extraction_strategy = LLMExtractionStrategy(
                        provider="openai",  # This would need to be configured
                        api_token="",  # This would need to be provided
                        instruction="Extract main content and key information from the page",
                    )
                elif request.extraction_strategy == "cosine":
                    extraction_strategy = CosineStrategy(
                        semantic_filter="main content, articles, blog posts",
                        word_count_threshold=settings.crawl_min_words,
                    )

            # Start crawling with streaming
            urls_to_crawl = [request.url]
            crawled_urls = set()
            depth = 0

            while (
                urls_to_crawl
                and depth < request.max_depth
                and len(crawled_urls) < request.max_pages
            ):
                current_level_urls = urls_to_crawl.copy()
                urls_to_crawl.clear()
                next_level_urls = []

                # Process URLs in current level with limited concurrency
                semaphore = asyncio.Semaphore(min(settings.max_concurrent_crawls, 5))

                async def crawl_single_url(url: str) -> PageContent | Exception:
                    async with semaphore:
                        if url in crawled_urls:
                            return Exception(f"URL already crawled: {url}")

                        crawled_urls.add(url)

                        try:
                            # Use browser session for crawling
                            result: Crawl4aiResult = await browser.arun(
                                url=url,
                                extraction_strategy=extraction_strategy,
                                bypass_cache=not config.get("cache_enabled", True),
                                process_iframes=False,  # Disable for performance
                                remove_overlay_elements=config.get(
                                    "remove_overlay_elements", True
                                ),
                                word_count_threshold=config.get(
                                    "word_count_threshold", 0
                                ),
                            )

                            if not result.success:
                                return Exception(
                                    f"Crawl failed for {url}: {result.error_message}"
                                )

                            # Extract links for next level
                            links = []
                            if result.links and depth + 1 < request.max_depth:
                                for link_url in result.links.get("internal", []):
                                    if self._should_include_url(link_url, request):
                                        absolute_url = urljoin(url, link_url)
                                        if absolute_url not in crawled_urls:
                                            next_level_urls.append(absolute_url)
                                            links.append(absolute_url)

                            # Create PageContent
                            page_content = PageContent(
                                url=url,
                                title=result.metadata.get("title", ""),
                                content=result.cleaned_html or result.markdown or "",
                                html=result.html,
                                markdown=result.markdown,
                                links=links,
                                images=list(result.media.get("images", []))
                                if result.media
                                else [],
                                metadata={
                                    "depth": depth,
                                    "word_count": len(
                                        (result.cleaned_html or "").split()
                                    ),
                                    "status_code": result.status_code,
                                    "response_headers": dict(
                                        result.response_headers or {}
                                    ),
                                },
                                timestamp=time.time(),
                            )

                            return page_content

                        except Exception as e:
                            return Exception(f"Error crawling {url}: {e}")

                # Execute crawls concurrently
                tasks = [
                    crawl_single_url(url)
                    for url in current_level_urls[
                        : request.max_pages - len(crawled_urls)
                    ]
                ]

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        yield result

                # Add next level URLs
                if next_level_urls and depth + 1 < request.max_depth:
                    urls_to_crawl.extend(
                        next_level_urls[: request.max_pages - len(crawled_urls)]
                    )

                depth += 1

        except Exception as e:
            yield e

    def _should_include_url(self, url: str, request: CrawlRequest) -> bool:
        """Check if URL should be included in crawl."""
        # Apply include patterns
        if request.include_patterns:
            if not any(pattern in url for pattern in request.include_patterns):
                return False

        # Apply exclude patterns
        if request.exclude_patterns:
            if any(pattern in url for pattern in request.exclude_patterns):
                return False

        # Apply global exclude patterns
        for pattern in settings.crawl_exclude_url_patterns:
            if pattern.replace("*", "") in url:
                return False

        return True

    async def pre_execute_setup(self) -> None:
        """Setup before crawling begins."""
        await self._initialize_managers()

        # Optimize memory for crawling
        if self.memory_manager:
            await self.memory_manager.optimize_for_crawl()

        # Cleanup expired browser sessions
        if self.browser_pool:
            await self.browser_pool.cleanup_expired_sessions()

    async def post_execute_cleanup(self) -> None:
        """Cleanup after crawling completes."""
        # Force memory cleanup
        if self.memory_manager:
            await self.memory_manager.check_memory_pressure(force_check=True)
