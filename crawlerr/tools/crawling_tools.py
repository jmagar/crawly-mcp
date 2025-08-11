"""
FastMCP tools for web crawling operations.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from ..services import CrawlerService, RagService, SourceService
from ..models.crawl_models import CrawlRequest
from ..models.source_models import SourceType
from ..middleware.progress_middleware import progress_middleware


logger = logging.getLogger(__name__)


def register_crawling_tools(mcp: FastMCP):
    """Register all crawling tools with the FastMCP server."""
    
    @mcp.tool
    async def scrape(
        ctx: Context,
        url: str,
        extraction_strategy: str = "css",
        wait_for: Optional[str] = None,
        include_raw_html: bool = False,
        process_with_rag: bool = True,
        enable_virtual_scroll: bool = None,
        virtual_scroll_count: int = None
    ) -> Dict[str, Any]:
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
                virtual_scroll_config = {
                    "scroll_count": virtual_scroll_count
                }
            
            # Scrape the page with advanced features
            page_content = await crawler_service.scrape_single_page(
                url=url,
                extraction_strategy=extraction_strategy,
                wait_for=wait_for,
                custom_config={},  # Empty config - include_raw_html is handled by the response object
                use_virtual_scroll=enable_virtual_scroll,
                virtual_scroll_config=virtual_scroll_config
            )
            
            result = {
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
                    "crawl4ai_version": "0.7.0+"
                }
            }
            
            if include_raw_html:
                result["html"] = page_content.html
                result["markdown"] = page_content.markdown
                result["links"] = page_content.links
                result["images"] = page_content.images
            
            # Process with RAG if requested
            if process_with_rag:
                await ctx.info("Processing for RAG indexing")
                await ctx.report_progress(progress=3, total=4)
                
                try:
                    # Create a minimal crawl result for RAG processing
                    from ..models.crawl_models import CrawlResult, CrawlStatus, CrawlStatistics
                    
                    crawl_result = CrawlResult(
                        request_id="single_scrape",
                        status=CrawlStatus.COMPLETED,
                        urls=[url],
                        pages=[page_content],
                        statistics=CrawlStatistics(
                            total_pages_requested=1,
                            total_pages_crawled=1,
                            total_bytes_downloaded=len(page_content.content)
                        )
                    )
                    
                    # Process for RAG
                    async with RagService() as rag_service:
                        rag_stats = await rag_service.process_crawl_result(crawl_result)
                        result["rag_processing"] = rag_stats
                    
                    # Register with source service
                    async with SourceService() as source_service:
                        sources = await source_service.register_crawl_result(
                            crawl_result, SourceType.WEBPAGE
                        )
                        result["sources_registered"] = len(sources)
                    
                    await ctx.info(f"Processed {rag_stats.get('chunks_created', 0)} chunks for RAG indexing")
                    
                except Exception as e:
                    await ctx.info(f"RAG processing failed: {str(e)}")
                    result["rag_processing_error"] = str(e)
            
            await ctx.info("Scraping completed")
            await ctx.report_progress(progress=4, total=4)
            await ctx.info(f"Successfully scraped {url}: {page_content.word_count} words, {len(page_content.links)} links")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to scrape {url}: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
        
        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)
    
    @mcp.tool
    async def crawl(
        ctx: Context,
        url: str,
        max_pages: int = 1000,
        max_depth: int = 3,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        sitemap_first: bool = True,
        process_with_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Crawl multiple pages from a website with advanced AI-powered site-wide crawling.
        
        This tool now uses Crawl4AI 0.7.0's advanced features including:
        - AI-powered adaptive crawling with confidence scoring
        - Intelligent URL discovery and seeding
        - Link preview and relevance scoring
        - Memory-adaptive resource management
        - Virtual scroll support for modern dynamic sites
        
        Args:
            url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl (1-2000)
            max_depth: Maximum depth to crawl (1-5)
            include_patterns: URL patterns to include (optional)
            exclude_patterns: URL patterns to exclude (optional)
            sitemap_first: Whether to check sitemap.xml first
            process_with_rag: Whether to process content for RAG indexing
            
        Returns:
            Dictionary with crawl results and statistics
        """
        await ctx.info(f"Starting crawl of {url} (max_pages: {max_pages}, max_depth: {max_depth})")
        
        # Validate parameters
        if max_pages > 2000:
            raise ToolError("max_pages cannot exceed 2000")
        if max_depth > 5:
            raise ToolError("max_depth cannot exceed 5")
        
        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"crawl_{hash(url)}")
        
        try:
            # Initialize crawler service
            crawler_service = CrawlerService()
            
            # Try sitemap first if requested
            urls_to_crawl = [url]
            if sitemap_first:
                await ctx.report_progress(progress=1, total=10)
                await ctx.info("Checking for sitemap.xml")
                
                try:
                    sitemap_urls = await crawler_service.get_sitemap_urls(url)
                    if sitemap_urls:
                        urls_to_crawl = sitemap_urls[:max_pages]
                        await ctx.info(f"Found {len(sitemap_urls)} URLs in sitemap, using {len(urls_to_crawl)}")
                    else:
                        await ctx.info("No sitemap found, proceeding with recursive crawl")
                except Exception as e:
                    await ctx.info(f"Sitemap check failed: {str(e)}, proceeding with recursive crawl")
            
            # Create crawl request
            request = CrawlRequest(
                url=urls_to_crawl,
                max_pages=max_pages,
                max_depth=max_depth,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns
            )
            
            # Progress callback for crawler
            async def crawler_progress(current, total, message=None):
                progress = 2 + int((current / total) * 6)  # Progress 2-8
                await ctx.report_progress(progress=progress, total=10)
            
            # Perform the crawl
            await ctx.report_progress(progress=2, total=10)
            crawl_result = await crawler_service.crawl_website(request, crawler_progress)
            
            if crawl_result.status.value != "completed":
                raise ToolError(f"Crawl failed: {', '.join(crawl_result.errors)}")
            
            # Prepare response with advanced features information
            result = {
                "status": crawl_result.status.value,
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
                    "average_page_size": crawl_result.statistics.average_page_size
                },
                "advanced_features": {
                    "adaptive_crawling": True,
                    "url_seeding": True,
                    "link_scoring": True,
                    "memory_adaptive_dispatch": True,
                    "virtual_scroll_support": True,
                    "crawl4ai_version": "0.7.0+",
                    "extraction_methods": ["adaptive_ai", "traditional"]
                },
                "errors": crawl_result.errors[:10],  # Limit errors to first 10
                "sample_pages": []
            }
            
            # Add sample page information
            for page in crawl_result.pages[:5]:  # Show first 5 pages
                result["sample_pages"].append({
                    "url": page.url,
                    "title": page.title,
                    "word_count": page.word_count,
                    "links_count": len(page.links),
                    "images_count": len(page.images)
                })
            
            # Process with RAG if requested
            if process_with_rag and crawl_result.pages:
                await ctx.report_progress(progress=9, total=10)
                await ctx.info(f"Processing {len(crawl_result.pages)} pages for RAG indexing")
                
                try:
                    # Process for RAG
                    async with RagService() as rag_service:
                        rag_stats = await rag_service.process_crawl_result(crawl_result)
                        result["rag_processing"] = rag_stats
                    
                    # Register sources
                    async with SourceService() as source_service:
                        sources = await source_service.register_crawl_result(
                            crawl_result, SourceType.WEBPAGE
                        )
                        result["sources_registered"] = len(sources)
                    
                    await ctx.info(
                        f"RAG processing completed: {rag_stats.get('chunks_created', 0)} chunks, "
                        f"{rag_stats.get('embeddings_generated', 0)} embeddings"
                    )
                    
                except Exception as e:
                    await ctx.info(f"RAG processing failed: {str(e)}")
                    result["rag_processing_error"] = str(e)
            
            await ctx.report_progress(progress=10, total=10)
            await ctx.info(f"Crawl completed: {result['pages_crawled']} pages in {result['statistics']['crawl_duration_seconds']:.1f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Crawl failed: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
        
        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)
    
    @mcp.tool
    async def crawl_repo(
        ctx: Context,
        repo_url: str,
        file_patterns: Optional[List[str]] = None,
        clone_path: Optional[str] = None,
        process_with_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Clone and analyze a Git repository for RAG indexing with adaptive processing.
        
        Enhanced with intelligent file prioritization, content filtering, and batch processing
        for optimal performance and relevance.
        
        Args:
            repo_url: URL of the Git repository to clone
            file_patterns: File patterns to include (e.g., ['*.py', '*.md'])
            clone_path: Custom path to clone the repository (optional)
            process_with_rag: Whether to process content for RAG indexing
            
        Returns:
            Dictionary with repository analysis results
        """
        await ctx.info(f"Starting repository crawl: {repo_url}")
        
        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"repo_{hash(repo_url)}")
        
        try:
            # Initialize crawler service
            crawler_service = CrawlerService()
            
            # Progress callback for crawler
            async def crawler_progress(current, total, message):
                await ctx.report_progress(progress=current, total=total)
            
            # Crawl repository
            crawl_result = await crawler_service.crawl_repository(
                repo_url=repo_url,
                clone_path=clone_path,
                file_patterns=file_patterns,
                progress_callback=crawler_progress
            )
            
            if crawl_result.status.value != "completed":
                raise ToolError(f"Repository crawl failed: {', '.join(crawl_result.errors)}")
            
            # Prepare response with adaptive processing information
            result = {
                "status": crawl_result.status.value,
                "repository_url": repo_url,
                "files_processed": len(crawl_result.pages),
                "total_content_size": sum(len(page.content) for page in crawl_result.pages),
                "file_types": {},
                "statistics": {
                    "total_files": len(crawl_result.pages),
                    "total_bytes": crawl_result.statistics.total_bytes_downloaded,
                    "processing_time": crawl_result.statistics.crawl_duration_seconds
                },
                "adaptive_features": {
                    "file_prioritization": True,
                    "content_filtering": True,
                    "batch_processing": True,
                    "language_detection": True,
                    "processing_method": "adaptive_batch"
                },
                "errors": crawl_result.errors[:10],
                "sample_files": []
            }
            
            # Analyze file types
            for page in crawl_result.pages:
                file_ext = page.metadata.get('file_extension', 'unknown')
                result["file_types"][file_ext] = result["file_types"].get(file_ext, 0) + 1
            
            # Add sample file information
            for page in crawl_result.pages[:10]:  # Show first 10 files
                result["sample_files"].append({
                    "path": page.metadata.get('file_path', page.url),
                    "extension": page.metadata.get('file_extension', ''),
                    "size": page.metadata.get('file_size', len(page.content)),
                    "word_count": page.word_count
                })
            
            # Process with RAG if requested
            if process_with_rag and crawl_result.pages:
                await ctx.info(f"Processing {len(crawl_result.pages)} files for RAG indexing")
                
                try:
                    # Process for RAG
                    async with RagService() as rag_service:
                        rag_stats = await rag_service.process_crawl_result(crawl_result)
                        result["rag_processing"] = rag_stats
                    
                    # Register sources
                    async with SourceService() as source_service:
                        sources = await source_service.register_crawl_result(
                            crawl_result, SourceType.REPOSITORY
                        )
                        result["sources_registered"] = len(sources)
                    
                    await ctx.info(
                        f"RAG processing completed: {rag_stats.get('chunks_created', 0)} chunks, "
                        f"{rag_stats.get('embeddings_generated', 0)} embeddings"
                    )
                    
                except Exception as e:
                    await ctx.info(f"RAG processing failed: {str(e)}")
                    result["rag_processing_error"] = str(e)
            
            await ctx.info(f"Repository crawl completed: {result['files_processed']} files processed")
            
            return result
            
        except Exception as e:
            error_msg = f"Repository crawl failed: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
        
        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)
    
    @mcp.tool
    async def crawl_dir(
        ctx: Context,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        process_with_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Crawl files in a local directory for RAG indexing with intelligent processing.
        
        Enhanced with relevance scoring, adaptive filtering, and content type detection
        for optimal file processing and indexing efficiency.
        
        Args:
            directory_path: Path to the directory to crawl
            file_patterns: File patterns to include (e.g., ['*.py', '*.md'])
            recursive: Whether to crawl subdirectories recursively
            process_with_rag: Whether to process content for RAG indexing
            
        Returns:
            Dictionary with directory crawl results
        """
        await ctx.info(f"Starting directory crawl: {directory_path}")
        
        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"dir_{hash(directory_path)}")
        
        try:
            # Initialize crawler service
            crawler_service = CrawlerService()
            
            # Progress callback for crawler
            async def crawler_progress(current, total, message):
                await ctx.report_progress(progress=current, total=total)
            
            # Crawl directory
            crawl_result = await crawler_service.crawl_directory(
                directory_path=directory_path,
                file_patterns=file_patterns,
                recursive=recursive,
                progress_callback=crawler_progress
            )
            
            if crawl_result.status.value != "completed":
                raise ToolError(f"Directory crawl failed: {', '.join(crawl_result.errors)}")
            
            # Prepare response with intelligent processing information
            result = {
                "status": crawl_result.status.value,
                "directory_path": directory_path,
                "files_processed": len(crawl_result.pages),
                "total_content_size": sum(len(page.content) for page in crawl_result.pages),
                "file_types": {},
                "statistics": {
                    "total_files": len(crawl_result.pages),
                    "total_bytes": crawl_result.statistics.total_bytes_downloaded,
                    "processing_time": crawl_result.statistics.crawl_duration_seconds
                },
                "intelligent_features": {
                    "relevance_scoring": True,
                    "adaptive_filtering": True,
                    "content_type_detection": True,
                    "batch_processing": True,
                    "processing_method": "adaptive_directory"
                },
                "errors": crawl_result.errors[:10],
                "sample_files": []
            }
            
            # Analyze file types
            for page in crawl_result.pages:
                file_ext = page.metadata.get('file_extension', 'unknown')
                result["file_types"][file_ext] = result["file_types"].get(file_ext, 0) + 1
            
            # Add sample file information
            for page in crawl_result.pages[:10]:  # Show first 10 files
                result["sample_files"].append({
                    "path": page.metadata.get('relative_path', page.title),
                    "extension": page.metadata.get('file_extension', ''),
                    "size": page.metadata.get('file_size', len(page.content)),
                    "word_count": page.word_count
                })
            
            # Process with RAG if requested
            if process_with_rag and crawl_result.pages:
                await ctx.info(f"Processing {len(crawl_result.pages)} files for RAG indexing")
                
                try:
                    # Process for RAG
                    async with RagService() as rag_service:
                        rag_stats = await rag_service.process_crawl_result(crawl_result)
                        result["rag_processing"] = rag_stats
                    
                    # Register sources
                    async with SourceService() as source_service:
                        sources = await source_service.register_crawl_result(
                            crawl_result, SourceType.DIRECTORY
                        )
                        result["sources_registered"] = len(sources)
                    
                    await ctx.info(
                        f"RAG processing completed: {rag_stats.get('chunks_created', 0)} chunks, "
                        f"{rag_stats.get('embeddings_generated', 0)} embeddings"
                    )
                    
                except Exception as e:
                    await ctx.info(f"RAG processing failed: {str(e)}")
                    result["rag_processing_error"] = str(e)
            
            await ctx.info(f"Directory crawl completed: {result['files_processed']} files processed")
            
            return result
            
        except Exception as e:
            error_msg = f"Directory crawl failed: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
        
        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)