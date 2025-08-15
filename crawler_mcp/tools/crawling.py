"""
FastMCP tools for web crawling operations.
"""

import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from ..config import settings
from ..core import CrawlerService, RagService, SourceService
from ..middleware.progress import progress_middleware
from ..models.crawl import CrawlRequest, CrawlResult, CrawlStatus
from ..models.sources import SourceType

logger = logging.getLogger(__name__)


def _detect_crawl_type_and_params(target: str) -> tuple[str, dict[str, Any]]:
    """
    Detect crawl type and normalize parameters based on input string.

    Args:
        target: Input string to analyze

    Returns:
        Tuple of (crawl_type, normalized_params)
        crawl_type: 'directory', 'repository', or 'website'
        normalized_params: Dictionary with type-specific parameters
    """
    from pathlib import Path

    # 1. Local path detection (highest priority)
    if target.startswith(("/", "./", "../", "~")) or Path(target).expanduser().exists():
        return "directory", {"directory_path": str(Path(target).expanduser().resolve())}

    # 2. Git repository detection
    git_hosts = [
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "gitee.com",
        "codeberg.org",
    ]
    if (
        any(host in target for host in git_hosts)
        or target.endswith(".git")
        or target.startswith(("git@", "git://", "ssh://git@"))
    ):
        return "repository", {"repo_url": target}

    # 3. Web URL detection (default fallback)
    return "website", {"url": target}


async def _process_rag_if_requested(
    ctx: Context,
    crawl_result: CrawlResult,
    source_type: SourceType,
    process_with_rag: bool,
    deduplication: bool | None = None,
    force_update: bool = False,
) -> dict[str, Any]:
    """
    Shared RAG processing logic for all crawl types.

    Args:
        ctx: FastMCP context
        crawl_result: Result from crawling operation
        source_type: Type of source for registration
        process_with_rag: Whether to process with RAG
        deduplication: Enable deduplication (defaults to settings)
        force_update: Force update all chunks even if unchanged

    Returns:
        Dictionary with RAG processing results or error info
    """
    rag_info: dict[str, Any] = {}

    if process_with_rag and crawl_result.pages:
        await ctx.info(f"Processing {len(crawl_result.pages)} items for RAG indexing")

        try:
            # Process for RAG with deduplication support
            async with RagService() as rag_service:
                rag_stats = await rag_service.process_crawl_result(
                    crawl_result,
                    deduplication=deduplication,
                    force_update=force_update,
                )
                rag_info["rag_processing"] = rag_stats

            # Register sources
            async with SourceService() as source_service:
                sources = await source_service.register_crawl_result(
                    crawl_result, source_type
                )
                rag_info["sources_registered"] = len(sources)

            await ctx.info(
                f"RAG processing completed: {rag_stats.get('chunks_created', 0)} chunks, "
                f"{rag_stats.get('embeddings_generated', 0)} embeddings"
            )

        except Exception as e:
            await ctx.info(f"RAG processing failed: {e!s}")
            rag_info["rag_processing_error"] = str(e)

    return rag_info


def _process_crawl_result_unified(
    crawl_result: CrawlResult,
    crawl_type: str,
    original_target: str,
) -> dict[str, Any]:
    """
    Process crawl result into unified response format.

    Args:
        crawl_result: Result from crawling operation
        crawl_type: Type of crawl ('directory', 'repository', 'website')
        original_target: Original input target

    Returns:
        Unified result dictionary
    """
    if crawl_type == "website":
        # Web crawling result format
        result: dict[str, Any] = {
            "status": crawl_result.status,
            "pages_crawled": len(crawl_result.pages),
            "pages_requested": crawl_result.statistics.total_pages_requested,
            "success_rate": crawl_result.success_rate,
            "statistics": {
                "total_pages_crawled": crawl_result.statistics.total_pages_crawled,
                "total_pages_failed": crawl_result.statistics.total_pages_failed,
                "unique_domains": crawl_result.statistics.unique_domains,
                "total_links_discovered": crawl_result.statistics.total_links_discovered,
                "total_bytes_downloaded": crawl_result.statistics.total_bytes_downloaded,
                "crawl_duration_seconds": crawl_result.statistics.crawl_duration_seconds,
                "pages_per_second": crawl_result.statistics.pages_per_second,
                "average_page_size": crawl_result.statistics.average_page_size,
            },
            "advanced_features": {
                "adaptive_crawling": True,
                "url_seeding": True,
                "link_scoring": True,
                "memory_adaptive_dispatch": True,
                "virtual_scroll_support": True,
                "crawl4ai_version": "0.7.0+",
                "extraction_methods": ["adaptive_ai", "traditional"],
            },
            "errors": crawl_result.errors[:10],
            "sample_pages": [],
        }

        # Add sample page information
        for page in crawl_result.pages[:5]:
            result["sample_pages"].append(
                {
                    "url": page.url,
                    "title": page.title,
                    "word_count": page.word_count,
                    "links_count": len(page.links),
                    "images_count": len(page.images),
                }
            )

    else:
        # File-based crawling result format (directory or repository)
        file_result: dict[str, Any] = {
            "status": crawl_result.status.value,
            "files_processed": len(crawl_result.pages),
            "total_content_size": sum(
                len(page.content or "") for page in crawl_result.pages
            ),
            "file_types": {},
            "statistics": {
                "total_files": len(crawl_result.pages),
                "total_bytes": crawl_result.statistics.total_bytes_downloaded,
                "processing_time": crawl_result.statistics.crawl_duration_seconds,
            },
            "errors": crawl_result.errors[:10],
            "sample_files": [],
        }

        if crawl_type == "repository":
            file_result["repository_url"] = original_target
            file_result["adaptive_features"] = {
                "file_prioritization": True,
                "content_filtering": True,
                "batch_processing": True,
                "language_detection": True,
                "processing_method": "adaptive_batch",
            }
        else:  # directory
            file_result["directory_path"] = original_target
            file_result["intelligent_features"] = {
                "relevance_scoring": True,
                "adaptive_filtering": True,
                "content_type_detection": True,
                "batch_processing": True,
                "processing_method": "adaptive_directory",
            }

        # Analyze file types
        file_types: dict[str, int] = {}
        for page in crawl_result.pages:
            file_ext = page.metadata.get("file_extension", "unknown")
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
        file_result["file_types"] = file_types

        # Add sample file information
        for page in crawl_result.pages[:10]:
            if crawl_type == "repository":
                sample_info = {
                    "path": page.metadata.get("file_path", page.url),
                    "extension": page.metadata.get("file_extension", ""),
                    "size": page.metadata.get("file_size", len(page.content or "")),
                    "word_count": page.word_count,
                }
            else:  # directory
                sample_info = {
                    "path": page.metadata.get("relative_path", page.title),
                    "extension": page.metadata.get("file_extension", ""),
                    "size": page.metadata.get("file_size", len(page.content or "")),
                    "word_count": page.word_count,
                }
            file_result["sample_files"].append(sample_info)

        result = file_result

    return result


def register_crawling_tools(mcp: FastMCP) -> None:
    """Register all crawling tools with the FastMCP server."""

    @mcp.tool
    async def scrape(
        ctx: Context,
        url: str,
        extraction_strategy: str = "css",
        wait_for: str | None = None,
        include_raw_html: bool = False,
        process_with_rag: bool = True,
        enable_virtual_scroll: bool | None = None,
        virtual_scroll_count: int | None = None,
        deduplication: bool | None = None,
        force_update: bool = False,
    ) -> dict[str, Any]:
        """
        Scrape a single web page using Crawl4AI with advanced features.

        Args:
            url: The URL to scrape
            extraction_strategy: Content extraction strategy ("css", "llm", "cosine", "json_css")
            wait_for: CSS selector or JavaScript condition to wait for (optional)
            include_raw_html: Whether to include raw HTML in the response
            process_with_rag: Whether to process content for RAG indexing
            enable_virtual_scroll: Enable virtual scroll for dynamic content (auto-detect if None)
            virtual_scroll_count: Number of scroll actions (defaults to config setting)
            deduplication: Enable deduplication (defaults to settings)
            force_update: Force update all chunks even if unchanged

        Returns:
            Dictionary with scraped content and metadata
        """
        await ctx.info(f"Starting scrape of: {url}")

        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"scrape_{hash(url)}")

        try:
            await ctx.info("Initializing crawler")
            await ctx.report_progress(progress=1, total=4)

            # Initialize services
            crawler_service = CrawlerService()

            await ctx.info("Scraping webpage")
            await ctx.report_progress(progress=2, total=4)

            # Prepare virtual scroll configuration if specified
            virtual_scroll_config = None
            if virtual_scroll_count:
                virtual_scroll_config = {"scroll_count": virtual_scroll_count}

            # Scrape the page with advanced features
            page_content = await crawler_service.scrape_single_page(
                url=url,
                extraction_strategy=extraction_strategy,
                wait_for=wait_for,
                custom_config={},  # Empty config - include_raw_html is handled by the response object
                use_virtual_scroll=enable_virtual_scroll or False,
                virtual_scroll_config=virtual_scroll_config,
            )

            result: dict[str, Any] = {
                "url": page_content.url,
                "title": page_content.title,
                "content": page_content.content,
                "word_count": page_content.word_count,
                "links_found": len(page_content.links),
                "images_found": len(page_content.images),
                "metadata": page_content.metadata,
                "timestamp": page_content.timestamp.isoformat(),
                "advanced_features": {
                    "extraction_strategy": extraction_strategy,
                    "virtual_scroll_enabled": enable_virtual_scroll,
                    "virtual_scroll_count": virtual_scroll_count,
                    "crawl4ai_version": "0.7.0+",
                },
            }

            if include_raw_html:
                result["html"] = page_content.html
                result["markdown"] = page_content.markdown
                links: list[str] = list(page_content.links)
                images: list[str] = list(page_content.images)
                result["links"] = links
                result["images"] = images

            # Process with RAG if requested
            if process_with_rag:
                await ctx.report_progress(progress=3, total=4)
                # Create a minimal crawl result for RAG processing
                from ..models.crawl import (
                    CrawlResult,
                    CrawlStatistics,
                    CrawlStatus,
                )

                crawl_result = CrawlResult(
                    request_id="single_scrape",
                    status=CrawlStatus.COMPLETED,
                    urls=[url],
                    pages=[page_content],
                    statistics=CrawlStatistics(
                        total_pages_requested=1,
                        total_pages_crawled=1,
                        total_bytes_downloaded=len(page_content.content),
                    ),
                )
                rag_info = await _process_rag_if_requested(
                    ctx,
                    crawl_result,
                    SourceType.WEBPAGE,
                    process_with_rag=True,
                    deduplication=deduplication,
                    force_update=force_update,
                )
                result.update(rag_info)

            await ctx.info("Scraping completed")
            await ctx.report_progress(progress=4, total=4)
            await ctx.info(
                f"Successfully scraped {url}: {page_content.word_count} words, {len(page_content.links)} links"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to scrape {url}: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e

        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)

    @mcp.tool
    async def crawl(
        ctx: Context,
        target: str,
        process_with_rag: bool = True,
        # Deduplication parameters
        deduplication: bool | None = None,
        force_update: bool = False,
        # Web-specific parameters (ignored for file/repo crawling)
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        sitemap_first: bool = True,
        # File-specific parameters (ignored for web crawling)
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        # Repo-specific parameters (ignored for other types)
        clone_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Smart crawl function that automatically detects input type and routes appropriately.

        This unified tool handles all crawling scenarios:
        - Local directories: "/home/user/code", "./src", "../docs"
        - Git repositories: "https://github.com/user/repo", "git@github.com:user/repo.git"
        - Websites: "https://example.com", "http://site.org/page"

        The function automatically detects the input type and uses the appropriate crawling method:
        - Directory crawling with intelligent file processing and relevance scoring
        - Repository cloning and analysis with adaptive batch processing
        - Website crawling with AI-powered site-wide discovery and deep crawling

        Args:
            target: Target to crawl (auto-detected: directory path, repository URL, or website URL)
            process_with_rag: Whether to process content for RAG indexing
            deduplication: Enable deduplication (defaults to settings)
            force_update: Force update all chunks even if unchanged

            # Web crawling parameters (used when target is a website)
            include_patterns: URL patterns to include (optional)
            exclude_patterns: URL patterns to exclude (optional)
            sitemap_first: Whether to check sitemap.xml first

            # File crawling parameters (used when target is a directory)
            file_patterns: File patterns to include (e.g., ['*.py', '*.md'])
            recursive: Whether to crawl subdirectories recursively

            # Repository parameters (used when target is a git repository)
            clone_path: Custom path to clone the repository (optional)

        Returns:
            Dictionary with crawl results and statistics (format varies by detected type)

        Examples:
            crawl("/home/jmagar/code")  # → directory crawling
            crawl("https://github.com/user/repo")  # → repository cloning and crawling
            crawl("https://example.com")  # → website crawling
        """
        # Detect input type and get normalized parameters
        crawl_type, type_params = _detect_crawl_type_and_params(target)

        await ctx.info(f"Detected {crawl_type} crawl for target: {target}")

        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"crawl_{hash(target)}")

        try:
            # Initialize crawler service
            crawler_service = CrawlerService()
            crawl_result = None

            # Route to appropriate crawling method based on detected type
            if crawl_type == "directory":
                await ctx.info(
                    f"Starting directory crawl: {type_params['directory_path']}"
                )

                # Progress callback for directory crawler
                def dir_progress(
                    current: int, total: int, message: str | None = None
                ) -> None:
                    # Note: These callbacks are expected to be sync functions
                    # Progress reporting will be handled at a higher level
                    pass

                crawl_result = await crawler_service.crawl_directory(
                    directory_path=type_params["directory_path"],
                    file_patterns=file_patterns,
                    recursive=recursive,
                    progress_callback=dir_progress,
                )
                source_type = SourceType.DIRECTORY

            elif crawl_type == "repository":
                await ctx.info(f"Starting repository crawl: {type_params['repo_url']}")

                # Progress callback for repository crawler
                def repo_progress(
                    current: int, total: int, message: str | None = None
                ) -> None:
                    # Note: These callbacks are expected to be sync functions
                    # Progress reporting will be handled at a higher level
                    pass

                crawl_result = await crawler_service.crawl_repository(
                    repo_url=type_params["repo_url"],
                    clone_path=clone_path,
                    file_patterns=file_patterns,
                    progress_callback=repo_progress,
                )
                source_type = SourceType.REPOSITORY

            else:  # website
                url = type_params["url"]

                # Use config settings for max_pages and max_depth
                max_pages = settings.crawl_max_pages
                max_depth = settings.crawl_max_depth

                await ctx.info(
                    f"Starting website crawl: {url} (max_pages: {max_pages}, max_depth: {max_depth})"
                )

                # Create crawl request
                request = CrawlRequest(
                    url=url,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )

                # Progress callback for web crawler
                def web_progress(
                    current: int, total: int, message: str | None = None
                ) -> None:
                    # Note: These callbacks are expected to be sync functions
                    # Progress reporting will be handled at a higher level
                    pass

                await ctx.report_progress(progress=1, total=10)
                crawl_result = await crawler_service.crawl_website(
                    request, web_progress
                )
                source_type = SourceType.WEBPAGE

            # Check crawl result status
            if crawl_result.status not in [CrawlStatus.COMPLETED, "completed"]:
                raise ToolError(f"Crawl failed: {', '.join(crawl_result.errors)}")

            await ctx.info(
                f"Crawl completed successfully: {len(crawl_result.pages)} items processed"
            )

            # Process result using unified helper
            result = _process_crawl_result_unified(crawl_result, crawl_type, target)

            # Process with RAG if requested using shared helper
            rag_info = await _process_rag_if_requested(
                ctx,
                crawl_result,
                source_type,
                process_with_rag,
                deduplication=deduplication,
                force_update=force_update,
            )
            result.update(rag_info)

            if crawl_type == "website":
                await ctx.report_progress(progress=10, total=10)

            duration = result.get("statistics", {}).get(
                "processing_time"
            ) or result.get("statistics", {}).get("crawl_duration_seconds", 0)
            items_count = result.get("pages_crawled") or result.get(
                "files_processed", 0
            )
            await ctx.info(f"Crawl completed: {items_count} items in {duration:.1f}s")

            return result

        except Exception as e:
            error_msg = f"{crawl_type.capitalize()} crawl failed: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e

        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)
