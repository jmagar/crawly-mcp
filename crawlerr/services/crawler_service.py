"""
Service for web crawling operations using Crawl4AI 0.7.0.
"""

import warnings

# Suppress Crawl4AI Pydantic v1 deprecation warnings before import
warnings.filterwarnings(
    "ignore",
    message="Support for class-based.*config.*is deprecated.*",
    category=DeprecationWarning,
)

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from collections.abc import Callable  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from urllib.parse import urljoin, urlparse  # noqa: E402

import git  # noqa: E402
from crawl4ai import (  # type: ignore[import-untyped] # noqa: E402
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    VirtualScrollConfig,
)
from crawl4ai.async_dispatcher import (  # type: ignore[import-untyped] # noqa: E402
    MemoryAdaptiveDispatcher,
)
from crawl4ai.deep_crawling import (  # type: ignore[import-untyped] # noqa: E402
    BFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.filters import (  # type: ignore[import-untyped] # noqa: E402
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.extraction_strategy import (  # type: ignore[import-untyped] # noqa: E402
    CosineStrategy,
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from fastmcp.exceptions import ToolError  # noqa: E402

from ..config import settings  # noqa: E402
from ..models.crawl_models import (  # noqa: E402
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)

logger = logging.getLogger(__name__)


class CrawlerService:
    """
    Service for web crawling operations using advanced Crawl4AI 0.7.0 features.
    """

    def __init__(self) -> None:
        # Build Chrome arguments with GPU acceleration
        chrome_args = self._build_chrome_args()

        self.browser_config = BrowserConfig(
            browser_type=settings.crawl_browser,
            headless=settings.crawl_headless,
            viewport_width=1920,
            viewport_height=1080,
            user_agent=settings.crawl_user_agent,
            accept_downloads=False,
            chrome_channel="chromium" if settings.crawl_browser == "chromium" else None,
            extra_args=chrome_args,
        )

        # Memory-adaptive dispatcher for intelligent resource management
        # Limit concurrent sessions based on GPU capabilities when GPU acceleration is enabled
        max_sessions = settings.max_concurrent_crawls
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            max_sessions = min(
                settings.gpu_concurrent_browsers, settings.max_concurrent_crawls
            )
            logger.info(
                "🎮 GPU-aware concurrent limit: %d browsers (RTX 4070 optimized)",
                max_sessions,
            )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=settings.crawl_memory_threshold,
            max_session_permit=max_sessions,
        )

        # Default crawl configuration - no extraction strategy by default
        self.default_run_config = CrawlerRunConfig(
            page_timeout=int(
                settings.crawler_timeout * 1000
            ),  # Convert to milliseconds
            delay_before_return_html=0.1,
            remove_overlay_elements=settings.crawl_remove_overlays,
            process_iframes=False,
            word_count_threshold=settings.crawl_min_words,
        )

        # Note: Advanced features like AsyncUrlSeeder, LinkPreview, AdaptiveCrawler
        # are not available in Crawl4AI 0.7.0+. Using standard crawling approaches.

        # Initialize GPU health checks
        if settings.gpu_health_check_enabled:
            self._check_gpu_health()

    def _build_chrome_args(self) -> list[str]:
        """
        Build Chrome arguments optimized for RTX 4070 GPU acceleration.

        Returns:
            List of Chrome command-line arguments
        """
        # Base arguments for stability
        base_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
        ]

        # Add GPU acceleration arguments if enabled
        gpu_args = []
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            # Check GPU availability first
            if settings.gpu_force_enable or self._is_gpu_available():
                # Core GPU acceleration flags
                gpu_args.extend(settings.chrome_gpu_flags.split())

                # Advanced GPU flags for better performance
                gpu_args.extend(settings.chrome_advanced_gpu_flags.split())

                logger.info(
                    "🎮 RTX 4070 GPU acceleration enabled with %d flags", len(gpu_args)
                )
                logger.info("💾 GPU memory limit: %dMB", settings.gpu_memory_limit_mb)
                logger.info(
                    "🔄 Max concurrent browsers: %d", settings.gpu_concurrent_browsers
                )
            else:
                if settings.gpu_fallback_enabled:
                    logger.warning("⚠️ GPU not available, falling back to CPU rendering")
                else:
                    logger.error("❌ GPU required but not available")
                    raise RuntimeError(
                        "GPU acceleration required but GPU not available"
                    )
        else:
            logger.info("⚪ GPU acceleration disabled")

        # Memory management arguments
        memory_args = [
            f"--max-memory-usage={settings.gpu_memory_limit_mb}",
            "--memory-pressure-off",
            f"--max_old_space_size={min(4096, settings.gpu_memory_limit_mb // 2)}",
        ]

        # Optional resource blocking for performance
        block_args: list[str] = []
        if settings.crawl_block_images:
            # Disable image loading at the renderer level
            block_args.append("--blink-settings=imagesEnabled=false")
        # Note: Playwright-level request interception would be more granular, but
        # Crawl4AI 0.7.0 API does not expose routing hooks directly here.
        # These flags provide a safe, broad performance win without breaking features.

        # Combine all arguments
        all_args = base_args + gpu_args + memory_args + block_args

        # Remove empty arguments
        all_args = [arg for arg in all_args if arg.strip()]

        logger.debug(
            "Chrome arguments: %d total (%d GPU-specific)", len(all_args), len(gpu_args)
        )
        return all_args

    def _get_gpu_utilization(self) -> dict:
        """
        Get real-time GPU utilization metrics.

        Note: nvidia-smi reports CUDA compute utilization only.
        Browser GPU acceleration (Vulkan/OpenGL/WebGL) may not be reflected
        in these metrics. GPU memory usage is more accurate for total activity.

        Returns:
            Dictionary with GPU metrics or empty dict if unavailable
        """
        try:
            import shutil
            import subprocess

            smi_path = shutil.which("nvidia-smi") or next(
                (
                    p
                    for p in [
                        "/usr/bin/nvidia-smi",
                        "/usr/local/bin/nvidia-smi",
                        "/bin/nvidia-smi",
                    ]
                    if Path(p).exists()
                ),
                None,
            )

            if not smi_path:
                return {"available": False}

            result = subprocess.run(
                [
                    smi_path,
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "gpu_utilization_percent": int(values[0])
                    if values[0] != "N/A"
                    else 0,
                    "memory_utilization_percent": int(values[1])
                    if values[1] != "N/A"
                    else 0,
                    "memory_used_mb": int(values[2]) if values[2] != "N/A" else 0,
                    "memory_total_mb": int(values[3]) if values[3] != "N/A" else 0,
                    "temperature_c": int(values[4]) if values[4] != "N/A" else 0,
                    "available": True,
                }
        except Exception as e:
            logger.debug("GPU utilization check failed: %s", e)

        return {"available": False}

    async def _monitor_gpu_during_crawl(self) -> None:
        """
        Monitor GPU utilization continuously during crawl operations.
        Logs real-time GPU usage to show actual hardware acceleration activity.
        """
        try:
            peak_utilization = 0
            peak_memory_usage = 0
            sample_count = 0

            while True:
                await asyncio.sleep(2)  # Sample every 2 seconds

                gpu_metrics = self._get_gpu_utilization()
                if gpu_metrics.get("available"):
                    util = gpu_metrics["gpu_utilization_percent"]
                    memory_used = gpu_metrics["memory_used_mb"]
                    temp = gpu_metrics["temperature_c"]

                    # Track peaks
                    if util > peak_utilization:
                        peak_utilization = util
                    if memory_used > peak_memory_usage:
                        peak_memory_usage = memory_used

                    sample_count += 1

                    # Log significant GPU activity during crawl
                    if util > 10:  # Significant GPU usage
                        logger.info(
                            "🔥 RTX 4070 GPU ACTIVE: %d%% util, %dMB used, %d°C (during crawl)",
                            util,
                            memory_used,
                            temp,
                        )
                    elif util > 5:  # Moderate GPU usage
                        logger.info(
                            "⚡ RTX 4070 GPU working: %d%% CUDA util, %dMB used, %d°C (browser 3D not measured)",
                            util,
                            memory_used,
                            temp,
                        )
                    elif sample_count % 5 == 0:  # Periodic low-usage updates
                        logger.info(
                            "🎮 RTX 4070 GPU monitor: %d%% util, %dMB used, %d°C",
                            util,
                            memory_used,
                            temp,
                        )

        except asyncio.CancelledError:
            # Log final peak statistics when monitoring ends
            if peak_utilization > 0:
                logger.info(
                    "🏁 RTX 4070 GPU crawl complete - Peak utilization: %d%%, Peak memory: %dMB",
                    peak_utilization,
                    peak_memory_usage,
                )
            raise

    def _is_gpu_available(self) -> bool:
        """
        Check if NVIDIA GPU is available for acceleration.

        Returns:
            True if GPU is available and functional
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                memory_mb = int(result.stdout.strip())
                logger.debug("GPU detected with %sMB VRAM", memory_mb)
                return memory_mb > 4000  # Ensure sufficient VRAM for GPU acceleration
            return False
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            return False

    def _check_gpu_health(self) -> None:
        """
        Perform comprehensive GPU health check for RTX 4070 optimization.
        """
        try:
            import shutil
            import subprocess

            smi_path = shutil.which("nvidia-smi") or next(
                (
                    p
                    for p in [
                        "/usr/bin/nvidia-smi",
                        "/usr/local/bin/nvidia-smi",
                        "/bin/nvidia-smi",
                    ]
                    if Path(p).exists()
                ),
                None,
            )

            # Check NVIDIA GPU availability
            result = subprocess.run(
                [smi_path] if smi_path else ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_info = result.stdout

                # Log GPU status
                logger.info("🎮 NVIDIA RTX 4070 detected and available")

                # Check for TEI service GPU usage
                if "text-embeddings-inference" in gpu_info:
                    logger.info("✅ TEI embedding service using GPU")

                # Extract and log GPU memory info
                memory_result = subprocess.run(
                    [
                        smi_path or "nvidia-smi",
                        "--query-gpu=memory.total,memory.used,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if memory_result.returncode == 0:
                    memory_info = memory_result.stdout.strip().split(", ")
                    if len(memory_info) == 3:
                        total, used, free = map(int, memory_info)
                        logger.info(
                            "🔋 GPU Memory: %dMB used, %dMB free, %dMB total",
                            used,
                            free,
                            total,
                        )

                        # Warn if memory usage is high
                        usage_percent = (used / total) * 100
                        if usage_percent > 80:
                            logger.warning(
                                "⚠️ High GPU memory usage: %.1f%%", usage_percent
                            )

                # Log browser GPU configuration
                self._log_browser_gpu_status()

            else:
                logger.warning("⚠️ nvidia-smi failed - GPU may not be available")
                if not settings.gpu_fallback_enabled:
                    raise RuntimeError("GPU required but nvidia-smi failed")

        except Exception as e:
            logger.warning("⚠️ GPU health check failed: %s", e)
            if not settings.gpu_fallback_enabled:
                raise RuntimeError(f"GPU health check failed: {e}") from e

    def _log_browser_gpu_status(self) -> None:
        """Log current browser GPU acceleration configuration."""
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            logger.info("🎮 Browser GPU acceleration: ENABLED")
            logger.info(
                "🔧 GPU flags: %d core + %d advanced",
                len(settings.chrome_gpu_flags.split()),
                len(settings.chrome_advanced_gpu_flags.split()),
            )
            logger.info(
                "🏎️ Acceleration: Vulkan + GPU rasterization + WebGL + Zero-copy"
            )
        else:
            logger.info("⚪ Browser GPU acceleration: DISABLED")

    async def scrape_single_page(
        self,
        url: str,
        extraction_strategy: str = "css",
        wait_for: str | None = None,
        custom_config: dict[str, Any] | None = None,
        use_virtual_scroll: bool | None = None,
        virtual_scroll_config: dict[str, Any] | None = None,
    ) -> PageContent:
        """
        Scrape a single web page using Crawl4AI with advanced features.

        Args:
            url: URL to scrape
            extraction_strategy: Strategy for content extraction ("css", "llm", "cosine", "json_css")
            wait_for: CSS selector or JavaScript condition to wait for
            custom_config: Custom crawler configuration overrides
            use_virtual_scroll: Whether to use virtual scroll (auto-detect if None)
            virtual_scroll_config: Virtual scroll configuration

        Returns:
            PageContent with extracted data
        """
        start_time = time.time()

        try:
            # Configure virtual scroll if enabled
            virtual_scroll = None
            if use_virtual_scroll or (
                use_virtual_scroll is None and settings.crawl_virtual_scroll
            ):
                scroll_config = virtual_scroll_config or {}
                virtual_scroll = VirtualScrollConfig(
                    container_selector=scroll_config.get(
                        "container_selector", "body"
                    ),  # More generic selector
                    scroll_count=scroll_config.get(
                        "scroll_count", settings.crawl_scroll_count
                    ),
                    scroll_by=scroll_config.get(
                        "scroll_by", "window_height"
                    ),  # Use window height instead
                    wait_after_scroll=scroll_config.get("wait_after_scroll", 1.0),
                )

            # Configure extraction strategy
            extraction_strategy_obj = None
            if extraction_strategy == "llm":
                # Note: LLM extraction would need API keys configured
                extraction_strategy_obj = LLMExtractionStrategy(
                    provider="openai",  # Would need configuration
                    model="gpt-4",
                )
            elif extraction_strategy == "cosine":
                extraction_strategy_obj = CosineStrategy(
                    semantic_filter="meaningful content"
                )
            elif extraction_strategy == "json_css":
                # Basic JSON CSS extraction
                extraction_strategy_obj = JsonCssExtractionStrategy(
                    {
                        "title": "h1, h2, .title",
                        "content": "p, .content, .article-body",
                        "links": "a[href]",
                    }
                )

            # Prepare run configuration with advanced features
            run_config = CrawlerRunConfig(
                page_timeout=int(settings.crawler_timeout * 1000),
                delay_before_return_html=0.1,
                remove_overlay_elements=settings.crawl_remove_overlays,
                wait_for=wait_for,
                word_count_threshold=settings.crawl_min_words,
                extraction_strategy=extraction_strategy_obj,
                virtual_scroll_config=virtual_scroll,
                **(custom_config or {}),
            )

            # Monitor GPU utilization for single page scrape if GPU acceleration enabled
            gpu_metrics_before = None
            if settings.crawl_gpu_enabled and settings.gpu_acceleration:
                gpu_metrics_before = self._get_gpu_utilization()
                if gpu_metrics_before.get("available"):
                    logger.info(
                        "🎮 GPU-accelerated scraping: %d%% util before scrape",
                        gpu_metrics_before["gpu_utilization_percent"],
                    )

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.debug(f"Starting advanced crawl of: {url}")

                result = await crawler.arun(url=url, config=run_config)

                # Check GPU utilization after scrape
                if (
                    settings.crawl_gpu_enabled
                    and settings.gpu_acceleration
                    and gpu_metrics_before
                ):
                    gpu_metrics_after = self._get_gpu_utilization()
                    if gpu_metrics_after.get("available"):
                        gpu_util_change = (
                            gpu_metrics_after["gpu_utilization_percent"]
                            - gpu_metrics_before["gpu_utilization_percent"]
                        )
                        if gpu_util_change > 0:
                            logger.info(
                                "🎮 RTX 4070 GPU utilized: %d%% increase during page rendering",
                                gpu_util_change,
                            )
                        else:
                            logger.info(
                                "🎮 RTX 4070 GPU: %d%% util (no change during render)",
                                gpu_metrics_after["gpu_utilization_percent"],
                            )

                if not result.success:
                    raise ToolError(f"Failed to crawl {url}: {result.error_message}")

                # Extract links from HTML
                links = []
                if hasattr(result, "links") and result.links:
                    for link in result.links:
                        if isinstance(link, dict):
                            href = link.get("href", "")
                        else:
                            href = str(link)

                        if href:
                            # Convert relative URLs to absolute
                            absolute_url = urljoin(url, href)
                            links.append(absolute_url)

                # Extract images
                images = []
                if hasattr(result, "media") and result.media:
                    for media in result.media:
                        if isinstance(media, dict):
                            src = media.get("src", "")
                            if media.get("type") == "image" and src:
                                absolute_url = urljoin(url, src)
                                images.append(absolute_url)

                processing_time = time.time() - start_time

                # Create page content
                page_content = PageContent(
                    url=url,
                    title=result.metadata.get("title", "") if result.metadata else "",
                    content=result.cleaned_html
                    if hasattr(result, "cleaned_html")
                    else result.markdown,
                    markdown=result.markdown if hasattr(result, "markdown") else "",
                    html=result.html if settings.crawl_extract_media else None,
                    links=links,
                    images=images,
                    metadata={
                        "processing_time": processing_time,
                        "http_status": result.status_code
                        if hasattr(result, "status_code")
                        else None,
                        "content_length": len(result.html) if result.html else 0,
                        "extraction_strategy": extraction_strategy,
                        **(result.metadata or {}),
                    },
                )

                logger.debug(f"Successfully crawled {url} in {processing_time:.2f}s")
                return page_content

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            raise ToolError(f"Failed to scrape {url}: {e!s}") from e

    async def crawl_website(
        self, request: CrawlRequest, progress_callback: Callable | None = None
    ) -> CrawlResult:
        """
        Crawl multiple pages from a website using Crawl4AI 0.7.0+ API.

        Args:
            request: Crawl request configuration
            progress_callback: Optional callback for progress updates

        Returns:
            CrawlResult with all crawled pages and statistics
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        result = CrawlResult(
            request_id=request_id,
            status=CrawlStatus.RUNNING,
            urls=request.url if isinstance(request.url, list) else [request.url],
        )

        try:
            # Prepare initial URLs
            initial_urls = (
                request.url if isinstance(request.url, list) else [request.url]
            )
            start_url = initial_urls[0]

            # Statistics tracking
            stats = CrawlStatistics()
            stats.total_pages_requested = request.max_pages or 0

            if progress_callback:
                await progress_callback(0, 4)

            # Phase 1: Try to discover URLs from sitemap first
            discovered_urls = set(initial_urls)

            if progress_callback:
                await progress_callback(1, 4)

            try:
                sitemap_urls = await self.get_sitemap_urls(start_url)
                if sitemap_urls:
                    # Filter sitemap URLs based on patterns
                    for url in sitemap_urls:
                        if request.exclude_patterns and any(
                            pattern in url for pattern in request.exclude_patterns
                        ):
                            continue
                        if request.include_patterns and not any(
                            pattern in url for pattern in request.include_patterns
                        ):
                            continue
                        discovered_urls.add(url)

                    logger.info(f"Discovered {len(discovered_urls)} URLs from sitemap")
                else:
                    logger.info("No sitemap found, falling back to recursive crawling")
                    discovered_urls.add(start_url)
            except (ValueError, AttributeError, KeyError, ET.ParseError) as e:
                logger.warning(
                    "Sitemap discovery failed: %s, falling back to recursive crawling",
                    e,
                )
                discovered_urls.add(start_url)

            if progress_callback:
                await progress_callback(2, 4)

            # Phase 2: Implement recursive crawling if sitemap not found
            if len(discovered_urls) == 1 and start_url in discovered_urls:
                # Use Crawl4AI's native deep crawling strategy
                pages = await self._native_deep_crawl(
                    start_url=start_url,
                    max_pages=request.max_pages or 100,
                    max_depth=request.max_depth or 3,
                    include_patterns=request.include_patterns,
                    exclude_patterns=request.exclude_patterns,
                    stats=stats,
                    progress_callback=progress_callback,
                )
            else:
                # Use sitemap URLs - batch crawl approach
                urls_to_crawl = list(discovered_urls)[: request.max_pages]
                pages = await self._batch_crawl(urls_to_crawl, stats)

            # Set result pages and update statistics
            result.pages = pages
            stats.total_pages_crawled = len(pages)
            logger.info(
                "Crawled %d pages successfully, %d failed",
                len(pages),
                stats.total_pages_failed,
            )

            if progress_callback:
                await progress_callback(3, 4)

            # Phase 4: Calculate final statistics
            end_time = time.time()
            stats.crawl_duration_seconds = end_time - start_time

            if result.pages:
                stats.total_bytes_downloaded = sum(
                    len(page.content or "") for page in result.pages
                )
                stats.total_links_discovered = sum(
                    len(page.links) for page in result.pages
                )
                stats.total_images_found = sum(
                    len(page.images) for page in result.pages
                )
                stats.unique_domains = len(
                    {urlparse(page.url).netloc for page in result.pages}
                )

                if stats.crawl_duration_seconds > 0:
                    stats.pages_per_second = (
                        stats.total_pages_crawled / stats.crawl_duration_seconds
                    )

                if stats.total_pages_crawled > 0:
                    stats.average_page_size = (
                        stats.total_bytes_downloaded / stats.total_pages_crawled
                    )

            result.statistics = stats
            result.status = CrawlStatus.COMPLETED
            result.end_time = datetime.fromtimestamp(end_time)

            if progress_callback:
                await progress_callback(4, 4)

            logger.info(
                f"Crawl completed: {stats.total_pages_crawled} pages in {stats.crawl_duration_seconds:.2f}s"
            )

            return result

        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Crawl failed: {e!s}")
            result.end_time = datetime.fromtimestamp(time.time())
            logger.error(f"Crawl failed: {e}")
            return result

    # Note: _traditional_crawl method removed as it used deprecated dispatcher.dispatch() API
    # Multi-URL crawling now handled directly in crawl_website() using arun_many()

    async def get_sitemap_urls(self, base_url: str) -> list[str]:
        """
        Extract URLs from sitemap.xml of a website.

        Args:
            base_url: Base URL of the website

        Returns:
            List of URLs found in sitemap
        """
        sitemap_urls = [
            urljoin(base_url, "/sitemap.xml"),
            urljoin(base_url, "/sitemap_index.xml"),
            urljoin(base_url, "/robots.txt"),  # Check robots.txt for sitemap references
        ]

        urls = set()

        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                for sitemap_url in sitemap_urls:
                    try:
                        result = await crawler.arun(
                            url=sitemap_url,
                            config=CrawlerRunConfig(
                                page_timeout=10000, remove_overlay_elements=False
                            ),
                        )

                        # Check both success and HTTP status code (avoid parsing 404 responses)
                        status_code = getattr(result, "status_code", None)
                        is_success = (
                            result.success
                            and result.html
                            and (status_code is None or (200 <= status_code < 300))
                        )

                        if is_success:
                            if "robots.txt" in sitemap_url:
                                # Parse robots.txt for sitemap references
                                lines = result.html.split("\n")
                                for line in lines:
                                    if line.lower().startswith("sitemap:"):
                                        sitemap_ref = line.split(":", 1)[1].strip()
                                        sitemap_urls.append(sitemap_ref)
                            else:
                                # Parse XML sitemap
                                try:
                                    root = ET.fromstring(result.html)

                                    # Handle sitemap index
                                    for sitemap in root.findall(
                                        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"
                                    ):
                                        loc_elem = sitemap.find(
                                            "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                                        )
                                        if loc_elem is not None and loc_elem.text:
                                            sitemap_urls.append(loc_elem.text)

                                    # Handle regular sitemap
                                    for url_elem in root.findall(
                                        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"
                                    ):
                                        loc_elem = url_elem.find(
                                            "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                                        )
                                        if loc_elem is not None and loc_elem.text:
                                            urls.add(loc_elem.text)

                                except ET.ParseError:
                                    logger.warning(
                                        f"Failed to parse XML sitemap: {sitemap_url}"
                                    )

                    except Exception as e:
                        logger.debug(f"Could not fetch sitemap {sitemap_url}: {e}")

        except Exception as e:
            logger.error(f"Error extracting sitemap URLs: {e}")

        return list(urls)

    async def _recursive_crawl(
        self,
        start_url: str,
        max_pages: int,
        max_depth: int,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        stats: CrawlStatistics | None = None,
        progress_callback: Callable | None = None,
    ) -> list[PageContent]:
        """
        Implement breadth-first recursive crawling to discover linked pages.
        """
        pages: list[PageContent] = []
        visited_urls = set()
        url_queue = [(start_url, 0)]  # (url, depth)

        # Extract domain for link filtering
        domain = urlparse(start_url).netloc

        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.1,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            stream=False,
        )

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            while url_queue and len(pages) < max_pages:
                current_url, depth = url_queue.pop(0)

                if current_url in visited_urls or depth > max_depth:
                    continue

                # Apply include/exclude patterns
                if exclude_patterns and any(
                    pattern in current_url for pattern in exclude_patterns
                ):
                    continue
                if include_patterns and not any(
                    pattern in current_url for pattern in include_patterns
                ):
                    continue

                visited_urls.add(current_url)
                logger.info(f"Crawling [{depth}/{max_depth}]: {current_url}")

                try:
                    result = await crawler.arun(url=current_url, config=run_config)

                    if result.success:
                        page_content = self._process_crawl_result(
                            result, extract_raw_html=False
                        )
                        pages.append(page_content)
                        if stats is not None:
                            stats.total_pages_crawled += 1

                        # Extract and queue new links if we haven't reached max depth
                        if depth < max_depth and len(pages) < max_pages:
                            new_links = self._extract_links_from_result(result, domain)
                            for link in new_links:
                                if (
                                    link not in visited_urls
                                    and (link, depth + 1) not in url_queue
                                ):
                                    url_queue.append((link, depth + 1))

                        if progress_callback:
                            progress = min(len(pages) / max_pages, 1.0)
                            await progress_callback(2 + int(progress), 4)

                    else:
                        if stats is not None:
                            stats.total_pages_failed += 1
                        logger.warning(
                            "Failed to crawl %s: %s", current_url, result.error_message
                        )

                except (ConnectionError, TimeoutError, ValueError):
                    if stats is not None:
                        stats.total_pages_failed += 1
                    logger.exception("Error crawling %s", current_url)

        logger.info(
            "Recursive crawl discovered %d pages across %d depth levels",
            len(pages),
            max_depth,
        )
        return pages

    async def _batch_crawl(
        self, urls: list[str], stats: CrawlStatistics
    ) -> list[PageContent]:
        """
        Crawl a batch of URLs concurrently using arun_many with GPU-optimized memory management.
        """
        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.1,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            stream=False,
        )

        # Use GPU-aware concurrent session limits for optimal RTX 4070 performance
        max_sessions = min(settings.max_concurrent_crawls, len(urls))
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            max_sessions = min(settings.gpu_concurrent_browsers, max_sessions)
            logger.info(
                "🎮 GPU-optimized batch crawl: %d concurrent sessions (RTX 4070)",
                max_sessions,
            )

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=settings.crawl_memory_threshold,
            max_session_permit=max(1, max_sessions),
        )

        pages = []

        # Monitor GPU utilization during crawl if GPU acceleration is enabled
        gpu_metrics_before = None
        gpu_metrics_after = None
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            gpu_metrics_before = self._get_gpu_utilization()
            if gpu_metrics_before.get("available"):
                logger.info(
                    "🎮 Pre-crawl GPU: %d%% util, %dMB/%dMB used, %d°C",
                    gpu_metrics_before["gpu_utilization_percent"],
                    gpu_metrics_before["memory_used_mb"],
                    gpu_metrics_before["memory_total_mb"],
                    gpu_metrics_before["temperature_c"],
                )

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            logger.info(
                "Starting GPU-accelerated batch crawl of %d URLs with %d concurrent sessions",
                len(urls),
                max_sessions,
            )

            # Start continuous GPU monitoring during crawl if enabled
            gpu_monitor_task = None
            if (
                settings.crawl_gpu_enabled
                and settings.gpu_acceleration
                and gpu_metrics_before
            ):
                gpu_monitor_task = asyncio.create_task(self._monitor_gpu_during_crawl())

            try:
                crawl_results = await crawler.arun_many(
                    urls=urls, config=run_config, dispatcher=dispatcher
                )

                for crawl_result in crawl_results:
                    crawl_url = getattr(crawl_result, "url", "<unknown>")
                    if crawl_result.success:
                        page_content = self._process_crawl_result(
                            crawl_result, extract_raw_html=False
                        )
                        if page_content:
                            pages.append(page_content)
                            stats.total_pages_crawled += 1
                        else:
                            stats.total_pages_failed += 1
                            logger.warning(
                                "Failed to process crawl result for %s", crawl_url
                            )
                    else:
                        stats.total_pages_failed += 1
                        logger.warning(
                            "Failed to crawl %s: %s",
                            crawl_url,
                            crawl_result.error_message,
                        )

            except Exception:
                logger.exception("Error during batch crawl")
                # Mark all URLs as failed
                stats.total_pages_failed += len(urls)
            finally:
                # Stop GPU monitoring
                if gpu_monitor_task:
                    gpu_monitor_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await gpu_monitor_task

        # Monitor GPU utilization after crawl to show actual usage
        if (
            settings.crawl_gpu_enabled
            and settings.gpu_acceleration
            and gpu_metrics_before
        ):
            gpu_metrics_after = self._get_gpu_utilization()
            if gpu_metrics_after.get("available"):
                gpu_util_change = (
                    gpu_metrics_after["gpu_utilization_percent"]
                    - gpu_metrics_before["gpu_utilization_percent"]
                )
                memory_change = (
                    gpu_metrics_after["memory_used_mb"]
                    - gpu_metrics_before["memory_used_mb"]
                )
                temp_change = (
                    gpu_metrics_after["temperature_c"]
                    - gpu_metrics_before["temperature_c"]
                )

                gpu_util_change_str = (
                    f"+{gpu_util_change}"
                    if gpu_util_change >= 0
                    else str(gpu_util_change)
                )
                memory_change_str = (
                    f"+{memory_change}" if memory_change >= 0 else str(memory_change)
                )
                temp_change_str = (
                    f"+{temp_change}" if temp_change >= 0 else str(temp_change)
                )

                logger.info(
                    "🎮 Post-crawl GPU: %d%% util (%s%% change), %dMB used (%sMB), %d°C (%s°C)",
                    gpu_metrics_after["gpu_utilization_percent"],
                    gpu_util_change_str,
                    gpu_metrics_after["memory_used_mb"],
                    memory_change_str,
                    gpu_metrics_after["temperature_c"],
                    temp_change_str,
                )

                # Log GPU acceleration effectiveness
                if gpu_util_change > 5:  # Significant GPU usage increase
                    logger.info(
                        "✅ RTX 4070 GPU acceleration active: %d%% utilization increase during crawl",
                        gpu_util_change,
                    )
                elif gpu_util_change > 0:
                    logger.info(
                        "⚡ RTX 4070 GPU acceleration detected: %d%% utilization increase",
                        gpu_util_change,
                    )
                else:
                    logger.warning(
                        "⚠️ GPU utilization unchanged - GPU acceleration may not be active"
                    )

        return pages

    def _process_crawl_result(
        self, crawl_result: Any, extract_raw_html: bool = False
    ) -> PageContent:
        """
        Process a single crawl result into PageContent.
        """
        # Extract links
        links = []
        if hasattr(crawl_result, "links") and crawl_result.links:
            for link in crawl_result.links:
                href = link.get("href", "") if isinstance(link, dict) else str(link)

                if href:
                    absolute_url = urljoin(crawl_result.url, href)
                    links.append(absolute_url)

        # Extract images
        images = []
        if hasattr(crawl_result, "media") and crawl_result.media:
            for media in crawl_result.media:
                if isinstance(media, dict):
                    src = media.get("src", "")
                    if media.get("type") == "image" and src:
                        absolute_url = urljoin(crawl_result.url, src)
                        images.append(absolute_url)

        # Simplified attribute access using getattr with fallbacks
        content = getattr(
            crawl_result, "cleaned_html", getattr(crawl_result, "markdown", "")
        )
        markdown = getattr(crawl_result, "markdown", "")
        title = crawl_result.metadata.get("title") if crawl_result.metadata else ""
        html = crawl_result.html if extract_raw_html else None
        content_length = len(getattr(crawl_result, "html", "") or "")

        return PageContent(
            url=crawl_result.url,
            title=title,
            content=content,
            markdown=markdown,
            html=html,
            links=links,
            images=images,
            metadata={
                "http_status": getattr(crawl_result, "status_code", None),
                "content_length": content_length,
                "extraction_method": "recursive_crawl",
                **(crawl_result.metadata or {}),
            },
        )

    def _extract_links_from_result(self, crawl_result: Any, domain: str) -> list[str]:
        """
        Extract same-domain links from a crawl result.
        """
        links = []
        if hasattr(crawl_result, "links") and crawl_result.links:
            for link in crawl_result.links:
                href = link.get("href", "") if isinstance(link, dict) else str(link)

                if href:
                    absolute_url = urljoin(crawl_result.url, href)
                    # Only add same-domain links
                    if urlparse(absolute_url).netloc == domain:
                        links.append(absolute_url)

        return links

    async def _native_deep_crawl(
        self,
        start_url: str,
        max_pages: int,
        max_depth: int,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        stats: CrawlStatistics | None = None,
        progress_callback: Callable | None = None,
    ) -> list[PageContent]:
        """
        Use Crawl4AI's native deep crawling capabilities with proper URL filtering.

        This replaces our custom implementation with Crawl4AI's native strategies
        to avoid crawling invalid URLs like /internal and /external.
        """
        domain = urlparse(start_url).netloc

        # Create comprehensive filter chain to avoid invalid URLs
        filter_chain = FilterChain(
            [
                # Domain filter to stay within same domain
                DomainFilter(allowed_domains=[domain], blocked_domains=[]),
                # Content type filter for HTML pages only
                ContentTypeFilter(allowed_types=["text/html"], check_extension=True),
                # URL pattern filter to exclude invalid patterns
                URLPatternFilter(
                    patterns=[
                        *settings.crawl_exclude_url_patterns,
                        # File extensions are always excluded
                        "*.css",
                        "*.js",
                        "*.jpg",
                        "*.jpeg",
                        "*.png",
                        "*.gif",
                        "*.svg",
                        "*.ico",
                        "*.pdf",
                        "*.zip",
                        "*.xml",
                        "*.json",
                    ],
                    reverse=True,  # Exclude matching patterns
                    use_glob=True,
                ),
            ]
        )

        # Apply user-defined include/exclude patterns if provided
        if include_patterns:
            user_include_filter = URLPatternFilter(
                patterns=include_patterns, reverse=False, use_glob=True
            )
            filter_chain.filters.append(user_include_filter)

        if exclude_patterns:
            user_exclude_filter = URLPatternFilter(
                patterns=exclude_patterns, reverse=True, use_glob=True
            )
            filter_chain.filters.append(user_exclude_filter)

        # Use BFS strategy with comprehensive filtering
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,  # Stay within same domain
            max_pages=max_pages,
            filter_chain=filter_chain,
        )

        # Configure crawler with deep crawl strategy
        crawl_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.1,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            deep_crawl_strategy=deep_crawl_strategy,
            verbose=settings.debug,
        )

        pages = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            logger.info(
                "Starting native Crawl4AI deep crawl from %s (max_pages: %d, max_depth: %d)",
                start_url,
                max_pages,
                max_depth,
            )

            try:
                # Use Crawl4AI's native deep crawling
                result = await crawler.arun(url=start_url, config=crawl_config)

                if result.success:
                    # Process main page
                    page_content = self._process_crawl_result(
                        result, extract_raw_html=False
                    )
                    pages.append(page_content)
                    if stats is not None:
                        stats.total_pages_crawled += 1

                    # Check if deep crawl results are available
                    if (
                        hasattr(result, "deep_crawl_results")
                        and result.deep_crawl_results
                    ):
                        for crawl_result in result.deep_crawl_results:
                            if crawl_result.success:
                                page_content = self._process_crawl_result(
                                    crawl_result, extract_raw_html=False
                                )
                                pages.append(page_content)
                                if stats is not None:
                                    stats.total_pages_crawled += 1

                                if progress_callback:
                                    progress = min(len(pages) / max_pages, 1.0)
                                    await progress_callback(2 + int(progress * 0.6), 4)
                            else:
                                if stats is not None:
                                    stats.total_pages_failed += 1
                                logger.warning(
                                    f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                                )
                    else:
                        # Fallback to manual BFS if deep_crawl_results not available
                        logger.info(
                            "Deep crawl results not available, using manual BFS approach with filtering"
                        )
                        additional_pages = await self._manual_bfs_with_filtering(
                            crawler,
                            start_url,
                            max_pages - 1,
                            max_depth,
                            filter_chain,
                            stats,
                        )
                        pages.extend(additional_pages)

                else:
                    if stats is not None:
                        stats.total_pages_failed += 1
                    logger.error(
                        f"Failed to start deep crawl from {start_url}: {result.error_message}"
                    )

            except Exception:
                if stats is not None:
                    stats.total_pages_failed += 1
                logger.exception("Error during native deep crawl")

        logger.info(
            f"Native deep crawl completed: {len(pages)} pages discovered with proper URL filtering"
        )
        return pages

    async def _manual_bfs_with_filtering(
        self,
        crawler: AsyncWebCrawler,
        start_url: str,
        max_pages: int,
        max_depth: int,
        filter_chain: FilterChain,
        stats: CrawlStatistics | None,
    ) -> list[PageContent]:
        """
        Manual BFS crawling with proper URL filtering as fallback.

        This is used when Crawl4AI's native deep crawl doesn't return deep_crawl_results.
        """
        pages: list[PageContent] = []
        visited_urls: set[str] = {start_url}

        # Compute concurrency based on config and GPU
        max_sessions = settings.max_concurrent_crawls
        if settings.crawl_gpu_enabled and settings.gpu_acceleration:
            max_sessions = min(settings.gpu_concurrent_browsers, max_sessions)

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=settings.crawl_memory_threshold,
            max_session_permit=max(1, max_sessions),
        )

        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.1,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            stream=False,
        )

        # BFS levels with concurrent batches
        current_depth = 0
        current_level_urls: list[str] = [start_url]

        while (
            current_level_urls and current_depth < max_depth and len(pages) < max_pages
        ):
            logger.info(
                "Manual BFS batch crawl depth %d with %d URLs (concurrency: %d)",
                current_depth,
                len(current_level_urls),
                max_sessions,
            )

            # Respect remaining capacity
            remaining_capacity = max_pages - len(pages)
            batch_urls = current_level_urls[:remaining_capacity]

            try:
                crawl_results = await crawler.arun_many(
                    urls=batch_urls, config=run_config, dispatcher=dispatcher
                )
            except Exception:
                logger.exception("Error during BFS batch crawl")
                if stats is not None:
                    stats.total_pages_failed += len(batch_urls)
                break

            next_level_candidates: list[str] = []

            for crawl_result in crawl_results:
                crawl_url = getattr(crawl_result, "url", "")
                if crawl_result.success:
                    page_content = self._process_crawl_result(
                        crawl_result, extract_raw_html=False
                    )
                    pages.append(page_content)
                    if stats is not None:
                        stats.total_pages_crawled += 1

                    if (
                        current_depth + 1 < max_depth
                        and hasattr(crawl_result, "links")
                        and crawl_result.links
                    ):
                        for link in crawl_result.links:
                            href = self._extract_href(link)
                            if not href:
                                continue
                            absolute_url = urljoin(crawl_url or start_url, href)
                            if absolute_url in visited_urls:
                                continue
                            try:
                                allowed = await filter_chain.apply(absolute_url)
                            except Exception:
                                allowed = False
                            if allowed:
                                visited_urls.add(absolute_url)
                                next_level_candidates.append(absolute_url)
                else:
                    if stats is not None:
                        stats.total_pages_failed += 1
                    logger.warning(
                        "Failed to crawl %s: %s",
                        crawl_url or "<unknown>",
                        getattr(crawl_result, "error_message", "unknown error"),
                    )

                if len(pages) >= max_pages:
                    break

            # Deduplicate and cap next-level breadth
            seen: set[str] = set()
            next_level_urls: list[str] = []
            for u in next_level_candidates:
                if u in seen:
                    continue
                seen.add(u)
                next_level_urls.append(u)
                if len(next_level_urls) >= max(1, max_sessions) * 4:
                    break

            current_level_urls = next_level_urls
            current_depth += 1

        return pages

    def _extract_href(self, link: Any) -> str:
        """Extract href from link object."""
        if isinstance(link, dict):
            href = link.get("href", "")
            return str(href) if href is not None else ""
        if isinstance(link, str):
            return link
        return str(link) if link else ""

    async def crawl_repository(
        self,
        repo_url: str,
        clone_path: str | None = None,
        file_patterns: list[str] | None = None,
        progress_callback: Callable | None = None,
    ) -> CrawlResult:
        """
        Clone and analyze a Git repository.

        Args:
            repo_url: URL of the Git repository
            clone_path: Path to clone the repository (optional)
            file_patterns: File patterns to include (e.g., ['*.py', '*.md'])
            progress_callback: Optional progress callback

        Returns:
            CrawlResult with repository content
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        result = CrawlResult(
            request_id=request_id, status=CrawlStatus.RUNNING, urls=[repo_url]
        )

        try:
            # Determine clone path
            if clone_path is None:
                repo_name = Path(repo_url).stem
                clone_path = f"/tmp/crawlerr_repos/{repo_name}_{int(time.time())}"

            clone_dir = Path(clone_path)
            clone_dir.parent.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                await progress_callback(1, 4, f"Cloning repository from {repo_url}")

            # Clone repository
            logger.info(f"Cloning repository: {repo_url}")
            _repo = git.Repo.clone_from(repo_url, clone_dir)

            if progress_callback:
                await progress_callback(2, 4, "Analyzing repository structure")

            # Analyze repository files with adaptive processing
            file_patterns = file_patterns or [
                "*.py",
                "*.js",
                "*.ts",
                "*.md",
                "*.txt",
                "*.json",
                "*.yaml",
                "*.yml",
            ]
            pages = []
            stats = CrawlStatistics()

            # Collect all matching files first
            all_files: list[Path] = []
            for pattern in file_patterns:
                all_files.extend(clone_dir.rglob(pattern))

            # Filter to actual files and sort by importance (prioritize documentation and config files)
            def file_priority(file_path: Path) -> int:
                """Assign priority to files for better processing order."""
                name = file_path.name.lower()
                if name in ["readme.md", "readme.txt", "readme.rst"]:
                    return 0  # Highest priority
                elif name.startswith("readme"):
                    return 1
                elif file_path.suffix.lower() in [".md", ".rst", ".txt"]:
                    return 2  # Documentation files
                elif name in [
                    "package.json",
                    "requirements.txt",
                    "pyproject.toml",
                    "cargo.toml",
                ]:
                    return 3  # Config files
                else:
                    return 4  # Source files

            valid_files = [f for f in all_files if f.is_file()]
            valid_files.sort(key=file_priority)

            # Process files in batches with adaptive filtering
            processed_count = 0
            batch_size = 50  # Process files in batches to avoid memory issues

            for i in range(0, len(valid_files), batch_size):
                batch = valid_files[i : i + batch_size]

                for file_path in batch:
                    try:
                        # Read file content
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        # Skip empty or very small files unless they're important
                        if len(content.strip()) < 50 and file_priority(file_path) > 3:
                            continue

                        # Create relative path from repo root
                        relative_path = file_path.relative_to(clone_dir)
                        file_url = f"{repo_url}/blob/main/{relative_path}"

                        # Enhanced metadata with repository analysis
                        file_metadata = {
                            "file_path": str(relative_path),
                            "file_extension": file_path.suffix,
                            "file_size": file_path.stat().st_size,
                            "repository": repo_url,
                            "source_type": "repository",
                            "file_priority": file_priority(file_path),
                            "processing_method": "adaptive_batch",
                        }

                        # Add language detection for code files
                        if file_path.suffix in [
                            ".py",
                            ".js",
                            ".ts",
                            ".go",
                            ".rs",
                            ".cpp",
                            ".java",
                        ]:
                            file_metadata["language"] = file_path.suffix[
                                1:
                            ]  # Remove dot

                        page_content = PageContent(
                            url=file_url,
                            title=f"{relative_path.name} - {repo_name}",
                            content=content,
                            metadata=file_metadata,
                        )

                        pages.append(page_content)
                        stats.total_pages_crawled += 1
                        stats.total_bytes_downloaded += len(content)
                        processed_count += 1

                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        result.errors.append(f"Could not read {file_path}: {e!s}")

                # Progress update after each batch
                if progress_callback and processed_count % batch_size == 0:
                    await progress_callback(
                        2, 4, f"Processed {processed_count} files from repository"
                    )

            if progress_callback:
                await progress_callback(3, 4, f"Processed {len(pages)} files")

            # Cleanup cloned repository
            try:
                import shutil

                shutil.rmtree(clone_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup repository clone: {e}")

            if progress_callback:
                await progress_callback(4, 4, "Repository crawl completed")

            # Finalize result
            end_time = time.time()
            stats.crawl_duration_seconds = end_time - start_time
            stats.total_pages_requested = len(pages)
            stats.unique_domains = 1

            result.pages = pages
            result.statistics = stats
            result.status = CrawlStatus.COMPLETED
            result.end_time = datetime.fromtimestamp(end_time)

            logger.info(
                f"Repository crawl completed: {len(pages)} files in {stats.crawl_duration_seconds:.2f}s"
            )
            return result

        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Repository crawl failed: {e!s}")
            result.end_time = datetime.fromtimestamp(time.time())
            logger.error(f"Repository crawl failed: {e}")
            return result

    async def crawl_directory(
        self,
        directory_path: str,
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        progress_callback: Callable | None = None,
    ) -> CrawlResult:
        """
        Crawl files in a local directory.

        Args:
            directory_path: Path to the directory to crawl
            file_patterns: File patterns to include
            recursive: Whether to crawl subdirectories
            progress_callback: Optional progress callback

        Returns:
            CrawlResult with directory content
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        result = CrawlResult(
            request_id=request_id,
            status=CrawlStatus.RUNNING,
            urls=[f"file://{directory_path}"],
        )

        try:
            dir_path = Path(directory_path)
            if not dir_path.exists():
                raise ToolError(f"Directory does not exist: {directory_path}")

            file_patterns = file_patterns or [
                "*.py",
                "*.js",
                "*.ts",
                "*.md",
                "*.txt",
                "*.json",
                "*.yaml",
                "*.yml",
            ]
            pages = []
            stats = CrawlStatistics()

            if progress_callback:
                await progress_callback(0, 1, f"Scanning directory: {directory_path}")

            # Collect all matching files
            all_files: list[Path] = []
            for pattern in file_patterns:
                if recursive:
                    all_files.extend(dir_path.rglob(pattern))
                else:
                    all_files.extend(dir_path.glob(pattern))

            # Remove duplicates and ensure they're files
            unique_files = list({f for f in all_files if f.is_file()})

            if progress_callback:
                await progress_callback(
                    0, len(unique_files), f"Processing {len(unique_files)} files"
                )

            # Adaptive file processing with prioritization
            def file_relevance_score(file_path: Path) -> float:
                """Calculate relevance score for file processing priority."""
                score = 0.5  # Base score

                # Higher priority for documentation and config files
                name_lower = file_path.name.lower()
                if name_lower.startswith("readme"):
                    score += 0.4
                elif file_path.suffix.lower() in [
                    ".md",
                    ".rst",
                    ".txt",
                ] or name_lower in [
                    "package.json",
                    "requirements.txt",
                    "pyproject.toml",
                ]:
                    score += 0.3
                elif file_path.suffix.lower() in [".py", ".js", ".ts"]:
                    score += 0.2

                # Penalize very large files or hidden files
                if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                    score -= 0.2
                if name_lower.startswith("."):
                    score -= 0.1

                # Boost files in root directory
                if len(file_path.relative_to(dir_path).parts) == 1:
                    score += 0.1

                return max(0.0, min(1.0, score))

            # Sort files by relevance for better processing order
            unique_files.sort(key=file_relevance_score, reverse=True)

            # Process files in adaptive batches
            batch_size = 20
            processed_count = 0

            for i in range(0, len(unique_files), batch_size):
                batch = unique_files[i : i + batch_size]

                for file_path in batch:
                    try:
                        # Read file content
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        # Skip empty files or very large files (unless high priority)
                        relevance = file_relevance_score(file_path)
                        if (len(content.strip()) < 10 and relevance < 0.7) or (
                            len(content) > 500000 and relevance < 0.8
                        ):  # 500KB threshold
                            continue

                        # Create file URL
                        file_url = file_path.as_uri()

                        # Enhanced metadata with adaptive information
                        file_metadata = {
                            "file_path": str(file_path),
                            "relative_path": str(file_path.relative_to(dir_path)),
                            "file_extension": file_path.suffix,
                            "file_size": file_path.stat().st_size,
                            "source_type": "directory",
                            "relevance_score": relevance,
                            "processing_method": "adaptive_directory",
                        }

                        # Add content type hints
                        if file_path.suffix.lower() in [
                            ".py",
                            ".js",
                            ".ts",
                            ".go",
                            ".rs",
                            ".java",
                        ]:
                            file_metadata["content_type"] = "source_code"
                        elif file_path.suffix.lower() in [".md", ".rst", ".txt"]:
                            file_metadata["content_type"] = "documentation"
                        elif file_path.suffix.lower() in [
                            ".json",
                            ".yaml",
                            ".yml",
                            ".toml",
                        ]:
                            file_metadata["content_type"] = "configuration"

                        page_content = PageContent(
                            url=file_url,
                            title=file_path.name,
                            content=content,
                            metadata=file_metadata,
                        )

                        pages.append(page_content)
                        stats.total_pages_crawled += 1
                        stats.total_bytes_downloaded += len(content)
                        processed_count += 1

                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        result.errors.append(f"Could not read {file_path}: {e!s}")
                        stats.total_pages_failed += 1

                # Progress update after each batch
                if progress_callback:
                    await progress_callback(
                        processed_count,
                        len(unique_files),
                        f"Processed {processed_count}/{len(unique_files)} files",
                    )

            if progress_callback:
                await progress_callback(
                    len(unique_files), len(unique_files), "Directory crawl completed"
                )

            # Finalize result
            end_time = time.time()
            stats.crawl_duration_seconds = end_time - start_time
            stats.total_pages_requested = len(unique_files)
            stats.unique_domains = 1

            result.pages = pages
            result.statistics = stats
            result.status = CrawlStatus.COMPLETED
            result.end_time = datetime.fromtimestamp(end_time)

            logger.info(
                f"Directory crawl completed: {len(pages)} files in {stats.crawl_duration_seconds:.2f}s"
            )
            return result

        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Directory crawl failed: {e!s}")
            result.end_time = datetime.fromtimestamp(time.time())
            logger.error(f"Directory crawl failed: {e}")
            return result
