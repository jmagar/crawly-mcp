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

from ..config import settings
from ..core.memory import MemoryManager, get_memory_manager
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from ..types.crawl4ai_types import DefaultMarkdownGeneratorImpl
from .base import BaseCrawlStrategy

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout() -> Any:
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
                "max_pages must be between 1 and 2000, got %s", request.max_pages
            )
            return False

        if request.max_depth is not None and (
            request.max_depth < 1 or request.max_depth > 5
        ):
            self.logger.error(
                "max_depth must be between 1 and 5, got %s", request.max_depth
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

        self.logger.info("WebCrawlStrategy.execute() started for URL: %s", request.url)
        await self._initialize_managers()

        start_time = time.time()
        self.logger.info(
            "Starting web crawl: %s (max_pages: %s, max_depth: %s)",
            request.url,
            request.max_pages,
            request.max_depth,
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
                "Discovered %s sitemap seeds: %s...",
                len(sitemap_seeds),
                sitemap_seeds[:5],
            )

            run_config = await self._build_run_config(request, sitemap_seeds)
            self.logger.info(
                "Deep crawl strategy configured: %s",
                type(run_config.deep_crawl_strategy).__name__
                if run_config.deep_crawl_strategy
                else "None",
            )
            if run_config.deep_crawl_strategy:
                self.logger.info(
                    "Max pages: %s, Max depth: %s",
                    getattr(run_config.deep_crawl_strategy, "max_pages", "Unknown"),
                    getattr(run_config.deep_crawl_strategy, "max_depth", "Unknown"),
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

            # Collect successful results first
            successful_results = []
            with suppress_stdout():
                try:
                    # Get result from arun - type depends on config.stream setting
                    self.logger.info(
                        "About to call browser.arun with stream=%s", run_config.stream
                    )
                    crawl_result = await browser.arun(url=first_url, config=run_config)
                    self.logger.info("browser.arun completed successfully")

                    # Debug: Log the actual type we received
                    self.logger.info(
                        "CRAWL DEBUG: crawl_result type = %s, stream=%s, deep_crawl=%s",
                        type(crawl_result).__name__,
                        run_config.stream,
                        run_config.deep_crawl_strategy is not None,
                    )

                    # Handle different return types based on deep crawl strategy and stream setting
                    if hasattr(crawl_result, "__aiter__"):
                        # AsyncGenerator case (when stream=True)
                        self.logger.info(
                            "Processing AsyncGenerator results (stream=True mode)"
                        )
                        async for result in crawl_result:
                            crawl_count += 1

                            # Pre-check for unexpected types (defensive programming)
                            if isinstance(result, int):
                                self.logger.warning(
                                    "Received integer %d instead of CrawlResult in streaming mode, skipping",
                                    result,
                                )
                                continue

                            # Ensure result is a CrawlResult object
                            if not hasattr(result, "success"):
                                self.logger.warning(
                                    "Received unexpected type %s in streaming mode, skipping",
                                    type(result).__name__,
                                )
                                continue

                            if result.success:
                                try:
                                    sanitized_result = self._sanitize_crawl_result(
                                        result
                                    )
                                    successful_results.append(sanitized_result)
                                except AttributeError as e:
                                    if (
                                        "'int' object has no attribute 'raw_markdown'"
                                        in str(e)
                                    ):
                                        self.logger.warning(
                                            "Caught integer markdown hash issue for %s, skipping result",
                                            result.url,
                                        )
                                        continue
                                    else:
                                        raise
                            else:
                                errors.append(
                                    f"Failed to crawl {result.url}: {result.error_message}"
                                )
                            if len(successful_results) >= max_pages:
                                break
                    elif isinstance(crawl_result, list):
                        # List case (deep crawl mode returns list even when stream=False)
                        self.logger.info(
                            "Processing List results (deep crawl mode with stream=False)"
                        )
                        for result in crawl_result:
                            crawl_count += 1
                            if result.success:
                                try:
                                    sanitized_result = self._sanitize_crawl_result(
                                        result
                                    )
                                    successful_results.append(sanitized_result)
                                except AttributeError as e:
                                    if (
                                        "'int' object has no attribute 'raw_markdown'"
                                        in str(e)
                                    ):
                                        self.logger.warning(
                                            "Caught integer markdown hash issue for %s, skipping result",
                                            result.url,
                                        )
                                        continue
                                    else:
                                        raise
                            else:
                                errors.append(
                                    f"Failed to crawl {result.url}: {result.error_message}"
                                )
                            if len(successful_results) >= max_pages:
                                break
                    elif isinstance(crawl_result, Crawl4aiResult):
                        # Single result case (when stream=False and no deep crawl)
                        crawl_count = 1
                        self.logger.info(
                            "Processing single CrawlResult (stream=False, no deep crawl)"
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed single result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                            )
                    elif hasattr(crawl_result, "success") and hasattr(
                        crawl_result, "url"
                    ):
                        # Handle CrawlResultContainer and other container types
                        crawl_count = 1
                        self.logger.info(
                            "Processing container result type: %s",
                            type(crawl_result).__name__,
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed container result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                "Failed to crawl {}: {}".format(
                                    crawl_result.url,
                                    getattr(
                                        crawl_result, "error_message", "Unknown error"
                                    ),
                                )
                            )
                    else:
                        raise Exception(
                            f"Unexpected crawl result type: {type(crawl_result)} (deep_crawl={run_config.deep_crawl_strategy is not None})"
                        )

                except AttributeError as e:
                    if "'int' object has no attribute 'raw_markdown'" in str(e):
                        self.logger.error(
                            "Critical: Integer markdown hash issue preventing crawl iteration"
                        )
                        raise Exception(
                            "Crawl failed due to integer markdown hash issue in streaming mode"
                        ) from e
                    else:
                        raise

            # Process results outside suppress_stdout context for debugging
            for result in successful_results:
                self.logger.info("Processing successful result for %s", result.url)
                page_content = self._to_page_content(result)
                self.logger.info(
                    "Created PageContent with %s words for %s",
                    page_content.word_count,
                    result.url,
                )
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

            self.logger.info(
                "Crawl loop completed: %s results processed, %s successful pages",
                crawl_count,
                len(pages),
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
                "Web crawl completed: %s pages, %.1fs, %.1f pages/sec, %s errors",
                len(pages),
                crawl_duration,
                pages_per_second,
                len(errors),
            )

            return crawl_result

        except Exception as e:
            self.logger.error("Web crawl failed: %s", e, exc_info=True)
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
                    self.logger.warning("Error closing browser: %s", e)

            await self.post_execute_cleanup()

    def _extract_text_from_html(self, html: str | None) -> str:
        """Extract plain text from HTML as a final fallback."""
        if not html:
            return ""

        try:
            # Simple HTML tag removal (basic fallback)
            import re

            # Remove script and style tags and their content
            html = re.sub(
                r"<(script|style)[^>]*>.*?</\1>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove all other HTML tags
            text = re.sub(r"<[^>]+>", "", html)
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            self.logger.warning("Failed to extract text from HTML: %s", e)
            return ""

    def _sanitize_crawl_result(self, result: Crawl4aiResult) -> Crawl4aiResult:
        """Sanitize CrawlResult to prevent integer hash issues with markdown field."""
        import sys

        from ..types.crawl4ai_types import (
            MarkdownGenerationResultImpl as MarkdownGenerationResult,
        )

        try:
            # Check if the private _markdown field contains an integer hash
            if hasattr(result, "_markdown") and isinstance(result._markdown, int):
                print(
                    f"CRAWL DEBUG - Found integer _markdown ({result._markdown}), replacing with empty MarkdownGenerationResult",
                    file=sys.stderr,
                    flush=True,
                )
                # Replace the integer hash with an empty MarkdownGenerationResult
                result._markdown = MarkdownGenerationResult(
                    raw_markdown="",
                    markdown_with_citations="",
                    references_markdown="",
                    fit_markdown=None,
                    fit_html=None,
                )

            # Also check if markdown property access would fail
            # This is a defensive check
            if hasattr(result, "markdown"):
                try:
                    # Try to access it to see if it would error
                    _ = result.markdown
                except AttributeError as e:
                    if "'int' object has no attribute" in str(e):
                        print(
                            f"CRAWL DEBUG - Markdown property access failed, force setting safe value for {result.url}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Force set a safe markdown value
                        result._markdown = MarkdownGenerationResult(
                            raw_markdown="",
                            markdown_with_citations="",
                            references_markdown="",
                            fit_markdown=None,
                            fit_html=None,
                        )
        except Exception as e:
            print(
                f"CRAWL DEBUG - Sanitization warning for {getattr(result, 'url', 'unknown')}: {e}",
                file=sys.stderr,
                flush=True,
            )

        return result

    def _safe_get_markdown(self, result: Crawl4aiResult) -> str:
        """Safely get markdown content from result, handling integer hash issues."""
        try:
            # Try to access markdown property
            markdown = result.markdown
            if isinstance(markdown, str):
                return markdown
            elif hasattr(markdown, "raw_markdown"):
                return markdown.raw_markdown
            else:
                return str(markdown) if markdown else ""
        except AttributeError as e:
            if "'int' object has no attribute" in str(e):
                # Return empty string if we hit the integer hash issue
                return ""
            else:
                raise

    def _to_page_content(self, result: Crawl4aiResult) -> PageContent:
        """Converts a crawl4ai result to a PageContent object."""
        import sys

        # STREAMING CRAWL FIX: During streaming, Crawl4AI returns lazy-loaded objects with hash placeholders
        # Sanitize the result first to prevent integer hash issues
        result = self._sanitize_crawl_result(result)

        # Initial inspection of the markdown field - SAFE ACCESS
        try:
            # Only access markdown if it's safe
            if hasattr(result, "_markdown") and not isinstance(
                getattr(result, "_markdown", None), int
            ):
                print(
                    f"CRAWL DEBUG - Markdown field is safe to access for {result.url}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"CRAWL DEBUG - Markdown field contains integer hash for {result.url}, skipping access",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(
                f"CRAWL DEBUG - Cannot inspect markdown for {result.url}: {e}",
                file=sys.stderr,
                flush=True,
            )

        best_content = ""

        # Strategy 1: Try to force content extraction by calling str() on the result itself
        # This might trigger lazy loading
        try:
            if hasattr(result, "__str__"):
                result_str = str(result)
                if result_str and len(result_str) > 100:  # Reasonable content length
                    best_content = result_str
                    print(
                        f"CRAWL DEBUG - Used result.__str__(): {len(best_content)} chars",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception as e:
            print(
                f"CRAWL DEBUG - result.__str__() failed: {e}",
                file=sys.stderr,
                flush=True,
            )

        # Strategy 2: Handle integer hash markdown (streaming mode issue)
        if not best_content:
            try:
                # Check if markdown field contains an integer hash instead of MarkdownGenerationResult
                if hasattr(result, "markdown") and result.markdown is not None:
                    markdown_val = result.markdown

                    # If markdown is an integer (hash), skip direct access and try other strategies
                    if isinstance(markdown_val, int):
                        print(
                            f"CRAWL DEBUG - markdown is integer hash ({markdown_val}), skipping direct access",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        # Try to access the MarkdownGenerationResult attributes safely
                        if hasattr(markdown_val, "fit_markdown"):
                            fit_md = getattr(markdown_val, "fit_markdown", None)
                            if fit_md and len(str(fit_md)) > 100:
                                best_content = str(fit_md)
                                print(
                                    f"CRAWL DEBUG - Used fit_markdown: {len(best_content)} chars",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        elif hasattr(markdown_val, "raw_markdown"):
                            raw_md = getattr(markdown_val, "raw_markdown", None)
                            if raw_md and len(str(raw_md)) > 100:
                                best_content = str(raw_md)
                                print(
                                    f"CRAWL DEBUG - Used raw_markdown: {len(best_content)} chars",
                                    file=sys.stderr,
                                    flush=True,
                                )
            except Exception as e:
                print(
                    f"CRAWL DEBUG - markdown access failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Strategy 3: Try accessing fit_markdown through different paths (legacy)
        if not best_content:
            try:
                # Check if markdown has a fit_markdown attribute that can be forced to load
                if (
                    hasattr(result, "markdown")
                    and result.markdown
                    and not isinstance(result.markdown, int)  # Skip if it's a hash
                    and hasattr(result.markdown, "fit_markdown")
                ):
                    fit_md = result.markdown.fit_markdown
                    if fit_md and len(str(fit_md)) > 100:
                        best_content = str(fit_md)
                        print(
                            f"CRAWL DEBUG - Used legacy fit_markdown: {len(best_content)} chars",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception as e:
                print(
                    f"CRAWL DEBUG - legacy fit_markdown access failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Strategy 4: Try to get the underlying content data from the result
        if not best_content:
            try:
                # Look for content in various result attributes that might hold actual data
                content_attrs = [
                    "content",
                    "text",
                    "body",
                    "data",
                    "_content",
                    "_text",
                    "_data",
                ]
                for attr in content_attrs:
                    if hasattr(result, attr):
                        val = getattr(result, attr)
                        if val and len(str(val)) > 100:
                            best_content = str(val)
                            print(
                                f"CRAWL DEBUG - Used result.{attr}: {len(best_content)} chars",
                                file=sys.stderr,
                                flush=True,
                            )
                            break
            except Exception as e:
                print(
                    f"CRAWL DEBUG - content attribute access failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Strategy 5: Extract from HTML as last resort
        if not best_content and result.html:
            try:
                # Check if html is actually a hash or real content
                html_str = str(result.html)
                if len(html_str) > 100:  # Real HTML content
                    best_content = self._extract_text_from_html(html_str)
                    print(
                        f"CRAWL DEBUG - Used HTML extraction: {len(best_content)} chars",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"CRAWL DEBUG - HTML is only {len(html_str)} chars (likely hash)",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"CRAWL DEBUG - HTML extraction failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Strategy 5: Try to force evaluation of lazy-loaded objects
        if not best_content:
            try:
                # For StringCompatibleMarkdown objects, try different access patterns
                if hasattr(result, "markdown") and result.markdown:
                    markdown_obj = result.markdown

                    # Try calling methods that might force content loading
                    force_methods = ["__call__", "get", "load", "evaluate", "resolve"]
                    for method_name in force_methods:
                        if hasattr(markdown_obj, method_name):
                            try:
                                method = getattr(markdown_obj, method_name)
                                if callable(method):
                                    forced_content = method()
                                    if (
                                        forced_content
                                        and len(str(forced_content)) > 100
                                    ):
                                        best_content = str(forced_content)
                                        print(
                                            f"CRAWL DEBUG - Used {method_name}(): {len(best_content)} chars",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                        break
                            except Exception:
                                continue
            except Exception as e:
                print(
                    f"CRAWL DEBUG - Force loading failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Calculate word count
        word_count = len(best_content.split()) if best_content.strip() else 0

        # Debug output to show what we found
        debug_msg = (
            f"CRAWL DEBUG - Final content extraction for {result.url}: "
            f"content_length={len(best_content)}, "
            f"word_count={word_count}"
        )
        print(debug_msg, file=sys.stderr, flush=True)

        # Legacy fit_markdown extraction for metadata
        import contextlib

        fit_markdown = None
        with contextlib.suppress(Exception):
            fit_markdown = (
                getattr(result.markdown, "fit_markdown", None)
                if hasattr(result, "markdown")
                else None
            )

        return PageContent(
            url=result.url,
            title=result.metadata.get("title", ""),
            content=best_content,
            html=result.html,
            markdown=fit_markdown or self._safe_get_markdown(result),
            word_count=word_count,
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

        # Create content filter for fit markdown generation - align with working scrape settings
        content_filter = PruningContentFilter(
            threshold=0.45,  # Prune nodes below 45% relevance score
            threshold_type="dynamic",  # Dynamic scoring
            min_word_threshold=5,  # Ignore very short text blocks
        )

        # Create markdown generator with content filter
        markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

        # Base run config with fit markdown optimization
        # NOTE: Deep crawl requires streaming mode (stream=True) to work properly
        # The BFS strategy expects an async generator, not a list
        stream_enabled = (
            deep_strategy is not None
        )  # Enable streaming only if deep crawl is used
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_strategy,
            stream=stream_enabled,  # Enable streaming when deep crawl is used
            cache_mode=cache_mode,
            page_timeout=page_timeout,
            remove_overlay_elements=getattr(settings, "crawl_remove_overlays", True),
            word_count_threshold=getattr(settings, "crawl_min_words", 50),
            check_robots_txt=False,  # per user preference
            verbose=False,  # Disable verbose output for MCP compatibility
            # Optimize for clean markdown extraction - ensure content processing
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            exclude_external_links=True,
            markdown_generator=markdown_generator,  # Enable fit markdown generation
            # Force content processing for streaming
            process_iframes=False,  # Disable for performance
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
            self.logger.info("Using keywords for scoring: %s...", keywords[:10])
            KeywordRelevanceScorer(keywords=keywords, weight=0.7)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.warning("Failed to create scorer: %s", e)

        # Use simple BFS strategy with comprehensive filtering
        try:
            self.logger.info(
                "Creating BFS deep crawl strategy: max_depth=%s, max_pages=%s, filter_chain=%s",
                max_depth,
                max_pages,
                "present" if filter_chain else "None",
            )
            # Disable filter_chain - any filtering prevents multi-page crawling
            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                max_depth=max_depth,
                include_external=False,
                max_pages=max_pages,
                # filter_chain=filter_chain,  # Disabled - even minimal filters break it
            )
        except Exception as e:
            self.logger.warning("Failed to create BFS strategy: %s", e)
            # Last resort: minimal BFS parameters
            try:
                return BFSDeepCrawlStrategy(max_depth=max_depth, include_external=False)  # type: ignore[attr-defined]
            except Exception as e2:
                self.logger.error("Failed to create minimal BFS strategy: %s", e2)
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
