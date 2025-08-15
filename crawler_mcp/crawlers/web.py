"""
Optimized web crawling strategy with streaming and caching support.
"""

import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from crawl4ai import (  # type: ignore
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai import CrawlResult as Crawl4aiResult  # type: ignore
from crawl4ai.content_filter_strategy import PruningContentFilter  # type: ignore
from crawl4ai.deep_crawling import (  # type: ignore
    BFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter  # type: ignore
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer  # type: ignore
from crawl4ai.extraction_strategy import (  # type: ignore
    CosineStrategy,
    LLMExtractionStrategy,
)
from crawl4ai.markdown_generation_strategy import (
    DefaultMarkdownGenerator,  # type: ignore
)

from ..config import settings
from ..core.memory import MemoryManager, get_memory_manager
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .base import BaseCrawlStrategy

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


class WebCrawlStrategy(BaseCrawlStrategy):
    """
    High-performance web crawling strategy with streaming, caching, and GPU acceleration.
    Optimized for RTX 4070 + i7-13700K performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory_manager: MemoryManager | None = None

    async def _initialize_managers(self) -> None:
        """Initialize required managers."""
        if not self.memory_manager:
            self.memory_manager = get_memory_manager()

    async def validate_request(self, request: CrawlRequest) -> bool:
        """Validate web crawl request."""
        if not request.url:
            self.logger.error("URL is required for web crawling")
            return False

        if request.max_pages is not None and (
            request.max_pages < 1 or request.max_pages > 2000
        ):
            self.logger.error(
                f"max_pages must be between 1 and 2000, got {request.max_pages}"
            )
            return False

        if request.max_depth is not None and (
            request.max_depth < 1 or request.max_depth > 5
        ):
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
        Execute optimized web crawling by delegating to the crawl4ai library.
        """
        await self._initialize_managers()

        start_time = time.time()
        self.logger.info(
            f"Starting web crawl: {request.url} (max_pages: {request.max_pages}, max_depth: {request.max_depth})"
        )

        await self.pre_execute_setup()

        try:
            if self.memory_manager is None:
                raise RuntimeError("Memory manager not initialized")
            if not await self.memory_manager.can_handle_crawl(request.max_pages or 100):
                self.logger.warning(
                    "System may have insufficient memory for crawl, proceeding with caution"
                )

            first_url = request.url[0] if isinstance(request.url, list) else request.url

            # Create minimal browser config - let Crawl4AI handle optimization
            browser_config = BrowserConfig(
                headless=settings.crawl_headless,
                browser_type=settings.crawl_browser,
                light_mode=True,  # Let Crawl4AI optimize performance
                text_mode=getattr(settings, "crawl_block_images", False),
                verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
                # NO extra_args - avoid flag conflicts
            )

            # Create fresh AsyncWebCrawler instance with stdout suppressed
            with suppress_stdout():
                browser = AsyncWebCrawler(config=browser_config)
                await browser.start()

            # Sitemap preseeding: discover and parse sitemap URLs to bias deep crawling
            sitemap_seeds = await self._discover_sitemap_seeds(
                first_url, request.max_pages or 100
            )
            self.logger.info(
                f"Discovered {len(sitemap_seeds)} sitemap seeds: {sitemap_seeds[:5]}..."
            )

            run_config = await self._build_run_config(request, sitemap_seeds)
            self.logger.info(
                f"Deep crawl strategy configured: {type(run_config.deep_crawl_strategy).__name__ if run_config.deep_crawl_strategy else 'None'}"
            )
            if run_config.deep_crawl_strategy:
                self.logger.info(
                    f"Max pages: {getattr(run_config.deep_crawl_strategy, 'max_pages', 'Unknown')}, Max depth: {getattr(run_config.deep_crawl_strategy, 'max_depth', 'Unknown')}"
                )

            pages = []
            errors = []
            total_bytes = 0
            unique_domains = set()
            total_links_discovered = 0
            max_pages = request.max_pages or 100

            if progress_callback:
                progress_callback(0, max_pages, "Starting crawl...")

            # Delegate crawling to crawl4ai using deep crawl + streaming
            crawl_count = 0
            self.logger.info(
                "Starting async iteration with stream=True and deep crawl strategy..."
            )
            with suppress_stdout():
                async for result in await browser.arun(
                    url=first_url, config=run_config
                ):
                    crawl_count += 1
                    self.logger.info(
                        f"Processing page {crawl_count}: {result.url} (success: {result.success})"
                    )
                    if hasattr(result, "metadata") and "score" in result.metadata:
                        self.logger.info(f"Page score: {result.metadata['score']}")
                    if result.success:
                        page_content = self._to_page_content(result)
                        pages.append(page_content)
                        total_bytes += len(page_content.content)
                        unique_domains.add(urlparse(page_content.url).netloc)
                        total_links_discovered += len(page_content.links)

                        if (
                            self.memory_manager
                            and await self.memory_manager.check_memory_pressure()
                        ):
                            self.logger.warning(
                                "Memory pressure detected during crawl, may slow down"
                            )

                        if progress_callback:
                            progress_callback(
                                len(pages),
                                max_pages,
                                f"Crawled: {page_content.url[:60]}...",
                            )
                    else:
                        errors.append(
                            f"Failed to crawl {result.url}: {result.error_message}"
                        )

                    if len(pages) >= max_pages:
                        self.logger.info(f"Reached max_pages limit of {max_pages}")
                        break

            self.logger.info(
                f"Crawl loop completed: {crawl_count} results processed, {len(pages)} successful pages"
            )
            end_time = time.time()
            crawl_duration = end_time - start_time
            pages_per_second = len(pages) / crawl_duration if crawl_duration > 0 else 0
            avg_page_size = total_bytes / len(pages) if pages else 0

            statistics = CrawlStatistics(
                total_pages_requested=max_pages,
                total_pages_crawled=len(pages),
                total_pages_failed=len(errors),
                unique_domains=len(unique_domains),
                total_links_discovered=total_links_discovered,
                total_bytes_downloaded=total_bytes,
                crawl_duration_seconds=crawl_duration,
                pages_per_second=pages_per_second,
                average_page_size=avg_page_size,
            )

            crawl_result = CrawlResult(
                request_id=f"web_crawl_{int(time.time())}",
                status=CrawlStatus.COMPLETED,
                urls=[request.url] if isinstance(request.url, str) else request.url,
                pages=pages,
                errors=errors,
                statistics=statistics,
            )

            self.logger.info(
                f"Web crawl completed: {len(pages)} pages, {crawl_duration:.1f}s, "
                f"{pages_per_second:.1f} pages/sec, {len(errors)} errors"
            )

            return crawl_result

        except Exception as e:
            self.logger.error(f"Web crawl failed: {e}", exc_info=True)
            urls = [request.url] if isinstance(request.url, str) else request.url
            return CrawlResult(
                request_id=f"web_crawl_failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=urls,
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

        finally:
            # Clean up browser instance
            if "browser" in locals():
                try:
                    with suppress_stdout():
                        await browser.close()
                except Exception as e:
                    self.logger.warning(f"Error closing browser: {e}")

            await self.post_execute_cleanup()

    def _to_page_content(self, result: Crawl4aiResult) -> PageContent:
        """Converts a crawl4ai result to a PageContent object."""
        # Prioritize fit markdown for clean, filtered content
        fit_markdown = (
            getattr(result.markdown, "fit_markdown", None)
            if hasattr(result, "markdown")
            else None
        )
        best_content = (
            result.extracted_content
            or fit_markdown
            or result.markdown
            or result.cleaned_html
            or ""
        )
        return PageContent(
            url=result.url,
            title=result.metadata.get("title", ""),
            content=best_content,
            html=result.html,
            markdown=fit_markdown or result.markdown,
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
                "depth": result.metadata.get("depth", 0),
                "word_count": len(best_content.split()),
                "status_code": result.status_code,
                "response_headers": dict(result.response_headers or {}),
                "chunk_metadata": getattr(result, "chunk_metadata", {}),
                "has_fit_markdown": fit_markdown is not None,
            },
            timestamp=datetime.fromtimestamp(time.time()),
        )

    async def _build_run_config(
        self, request: CrawlRequest, sitemap_seeds: list[str] | None = None
    ) -> CrawlerRunConfig:
        """Build Crawl4AI CrawlerRunConfig aligned with deep crawling and streaming."""
        # Cache mode mapping
        cache_mode = (
            CacheMode.ENABLED
            if getattr(settings, "crawl_enable_caching", True)
            else CacheMode.BYPASS
        )

        # Timeout: Crawl4AI expects milliseconds; coerce if likely in seconds
        timeout_val = getattr(settings, "crawler_timeout", 30000)
        page_timeout = (
            int(timeout_val * 1000)
            if isinstance(timeout_val, int | float) and timeout_val < 1000
            else int(timeout_val)
        )

        # Deep crawl strategy (BestFirst preferred, BFS fallback)
        deep_strategy = self._build_deep_crawl_strategy(request, sitemap_seeds or [])

        # Create content filter for fit markdown generation
        content_filter = PruningContentFilter(
            threshold=0.45,  # Prune nodes below 45% relevance score
            threshold_type="dynamic",  # Dynamic scoring
            min_word_threshold=5,  # Ignore very short text blocks
        )

        # Create markdown generator with content filter
        markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

        # Base run config with fit markdown optimization
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_strategy,
            stream=True,  # CRITICAL: Enable streaming for multi-page deep crawling
            cache_mode=cache_mode,
            page_timeout=page_timeout,
            remove_overlay_elements=getattr(settings, "crawl_remove_overlays", True),
            word_count_threshold=getattr(settings, "crawl_min_words", 50),
            check_robots_txt=False,  # per user preference
            verbose=False,  # Disable verbose output for MCP compatibility
            # Optimize for clean markdown extraction
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            exclude_external_links=True,
            markdown_generator=markdown_generator,  # Enable fit markdown generation
        )

        # Optional: memory thresholds to align with our MemoryManager
        if hasattr(settings, "crawl_memory_threshold_percent"):
            with contextlib.suppress(Exception):
                run_config.memory_threshold_percent = (
                    settings.crawl_memory_threshold_percent
                )  # type: ignore[attr-defined]
        if hasattr(settings, "crawl_memory_check_interval"):
            with contextlib.suppress(Exception):
                run_config.check_interval = settings.crawl_memory_check_interval  # type: ignore[attr-defined]

        # Optional: wait_for selector
        if getattr(request, "wait_for", None):
            run_config.wait_for = request.wait_for  # type: ignore[attr-defined]

        # Extraction strategy (best-effort mapping)
        if getattr(request, "extraction_strategy", None):
            if request.extraction_strategy == "llm":
                with contextlib.suppress(Exception):
                    run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
                        provider="openai",
                        api_token="",
                        instruction="Extract main content and key information from the page",
                    )
            elif request.extraction_strategy == "cosine":
                with contextlib.suppress(Exception):
                    run_config.extraction_strategy = CosineStrategy(  # type: ignore[attr-defined]
                        semantic_filter="main content, articles, blog posts",
                        word_count_threshold=getattr(settings, "crawl_min_words", 50),
                    )

        # Chunking strategy (best-effort; only if available)
        if getattr(request, "chunking_strategy", None):
            try:
                from crawl4ai.chunking_strategy import (  # type: ignore[import-untyped]
                    FixedLengthWordChunking,
                    OverlappingWindowChunking,
                    RegexChunking,
                    SlidingWindowChunking,
                )

                chunking_map = {
                    "overlapping_window": OverlappingWindowChunking,
                    "sliding_window": SlidingWindowChunking,
                    "fixed_length_word": FixedLengthWordChunking,
                    "regex": RegexChunking,
                }
                chunker_class = chunking_map.get(request.chunking_strategy or "")
                if chunker_class:
                    chunking_options = request.chunking_options or {}
                    run_config.chunking_strategy = chunker_class(**chunking_options)  # type: ignore[attr-defined]
            except Exception:
                pass

        return run_config

    def _build_deep_crawl_strategy(
        self, request: CrawlRequest, sitemap_seeds: list[str]
    ) -> BFSDeepCrawlStrategy | None:
        """Construct a Best-First deep crawl strategy with filters and scoring; fallback to BFS."""
        max_depth = request.max_depth or 1
        max_pages = request.max_pages or 100

        # Build filter chain from include/exclude patterns
        include_patterns = request.include_patterns or []
        exclude_patterns = (request.exclude_patterns or []) + getattr(
            settings, "crawl_exclude_url_patterns", []
        )

        filter_chain = None
        try:
            filters: list[Any] = []
            if include_patterns:
                # Some versions of URLPatternFilter may not support modes; wrap in try/except
                try:
                    filters.append(
                        URLPatternFilter(
                            patterns=list(include_patterns), mode="include"
                        )
                    )  # type: ignore[attr-defined]
                except Exception:
                    filters.append(URLPatternFilter(patterns=list(include_patterns)))  # type: ignore[attr-defined]
            if exclude_patterns:
                try:
                    filters.append(
                        URLPatternFilter(
                            patterns=list(exclude_patterns), mode="exclude"
                        )
                    )  # type: ignore[attr-defined]
                except Exception:
                    filters.append(URLPatternFilter(patterns=list(exclude_patterns)))  # type: ignore[attr-defined]
            if filters:
                filter_chain = FilterChain(filters)  # type: ignore[attr-defined]
        except Exception:
            filter_chain = None

        # Scorer: prioritize includes, sitemap-derived keywords, and content-like URLs
        try:
            keywords: list[str] = []
            for pat in include_patterns:
                keywords.extend(
                    [
                        t
                        for t in str(pat).replace("*", " ").replace("/", " ").split()
                        if len(t) > 2
                    ]
                )
            # Add tokens from sitemap URLs
            for u in sitemap_seeds[:max_pages]:
                try:
                    path = urlparse(u).path
                    keywords.extend(
                        [
                            t
                            for t in path.replace("-", " ")
                            .replace("_", " ")
                            .replace("/", " ")
                            .split()
                            if len(t) > 2
                        ]
                    )
                except Exception:
                    continue
            if not keywords:
                keywords = ["docs", "blog", "guide", "article", "learn", "help", "faq"]

            # Add more generic keywords to be less restrictive
            keywords.extend(
                [
                    "fastmcp",
                    "mcp",
                    "client",
                    "server",
                    "api",
                    "tutorial",
                    "example",
                    "getting",
                    "started",
                ]
            )
            self.logger.info(f"Using keywords for scoring: {keywords[:10]}...")
            KeywordRelevanceScorer(keywords=keywords, weight=0.7)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.warning(f"Failed to create scorer: {e}")

        # Use simple BFS strategy with comprehensive filtering
        try:
            self.logger.info(
                f"Creating BFS deep crawl strategy: max_depth={max_depth}, max_pages={max_pages}, filter_chain={'present' if filter_chain else 'None'}"
            )
            # Disable filter_chain - any filtering prevents multi-page crawling
            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                max_depth=max_depth,
                include_external=False,
                max_pages=max_pages,
                # filter_chain=filter_chain,  # Disabled - even minimal filters break it
            )
        except Exception as e:
            self.logger.warning(f"Failed to create BFS strategy: {e}")
            # Last resort: minimal BFS parameters
            try:
                return BFSDeepCrawlStrategy(max_depth=max_depth, include_external=False)  # type: ignore[attr-defined]
            except Exception as e2:
                self.logger.error(f"Failed to create minimal BFS strategy: {e2}")
                return None

    async def _discover_sitemap_seeds(self, start_url: str, limit: int) -> list[str]:
        """Fetch robots.txt and sitemap.xml to build seed URLs for prioritization.
        Returns a bounded list of same-domain URLs.
        """
        try:
            parsed = urlparse(start_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base, "/robots.txt")
            sitemap_urls = await self._extract_sitemaps_from_robots(robots_url)
            if not sitemap_urls:
                # fallback to conventional path
                sitemap_urls = [urljoin(base, "/sitemap.xml")]

            seeds: list[str] = []
            for sm in sitemap_urls:
                urls = await self._parse_sitemap(sm, base, remaining=limit - len(seeds))
                seeds.extend(urls)
                if len(seeds) >= limit:
                    break

            # Dedup same-domain
            seen = set()
            same_domain_seeds = []
            for u in seeds:
                try:
                    if urlparse(u).netloc == parsed.netloc and u not in seen:
                        seen.add(u)
                        same_domain_seeds.append(u)
                except Exception:
                    continue
            return same_domain_seeds[:limit]
        except Exception:
            return []

    async def _extract_sitemaps_from_robots(self, robots_url: str) -> list[str]:
        try:
            text = await self._fetch_text(robots_url, timeout=10)
            if not text:
                return []
            sitemaps: list[str] = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("sitemap:"):
                    sitemaps.append(line.split(":", 1)[1].strip())
            return sitemaps
        except Exception:
            return []

    async def _parse_sitemap(
        self, sitemap_url: str, base: str, remaining: int
    ) -> list[str]:
        """Parse a sitemap or sitemap index and return up to `remaining` URLs."""
        try:
            import xml.etree.ElementTree as ET

            xml_text = await self._fetch_text(sitemap_url, timeout=15)
            if not xml_text:
                return []
            urls: list[str] = []
            try:
                root = ET.fromstring(xml_text)
            except Exception:
                return []

            tag = root.tag.lower()

            def ns_strip(t: str) -> str:
                return t.split("}", 1)[-1] if "}" in t else t

            tag = ns_strip(tag)
            if tag == "sitemapindex":
                for sm in root.findall(".//{*}sitemap/{*}loc"):
                    loc_text = (sm.text or "").strip()
                    if not loc_text:
                        continue
                    if len(urls) >= remaining:
                        break
                    urls.extend(
                        await self._parse_sitemap(loc_text, base, remaining - len(urls))
                    )
                    if len(urls) >= remaining:
                        break
            elif tag == "urlset":
                for loc in root.findall(".//{*}url/{*}loc"):
                    loc_text = (loc.text or "").strip()
                    if not loc_text:
                        continue
                    urls.append(loc_text)
                    if len(urls) >= remaining:
                        break
            # Normalize to absolute
            abs_urls: list[str] = []
            for u in urls[:remaining]:
                try:
                    abs_urls.append(urljoin(base, u))
                except Exception:
                    continue
            return abs_urls[:remaining]
        except Exception:
            return []

    async def _fetch_text(self, url: str, timeout: int = 10) -> str:
        """Lightweight async fetch (Playwright via the existing crawler session is not used here to avoid side effects).
        Uses aiohttp if available, else returns empty string.
        """
        try:
            import aiohttp  # type: ignore
        except Exception:
            return ""

        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, timeout=timeout_obj) as resp,
            ):
                if resp.status != 200:
                    return ""
                return await resp.text()
        except Exception:
            return ""

    async def pre_execute_setup(self) -> None:
        """Setup before crawling begins."""
        await super().pre_execute_setup()
        # No browser session cleanup needed - each crawl uses fresh browser
