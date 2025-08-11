"""
Service for web crawling operations using Crawl4AI 0.7.0.
"""
import warnings

# Suppress Crawl4AI Pydantic v1 deprecation warnings before import
warnings.filterwarnings("ignore", message="Support for class-based.*config.*is deprecated.*", category=DeprecationWarning)

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Set, Union
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import xml.etree.ElementTree as ET
import git
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy, CosineStrategy, JsonCssExtractionStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import VirtualScrollConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter, ContentTypeFilter
from ..config import settings
from ..models.crawl_models import (
    CrawlRequest, CrawlResult, CrawlStatus, PageContent, CrawlStatistics
)
from ..models.source_models import SourceType, SourceInfo, SourceMetadata
from fastmcp.exceptions import ToolError


logger = logging.getLogger(__name__)


class CrawlerService:
    """
    Service for web crawling operations using advanced Crawl4AI 0.7.0 features.
    """
    
    def __init__(self):
        self.browser_config = BrowserConfig(
            browser_type=settings.crawl_browser,
            headless=settings.crawl_headless,
            viewport_width=1920,
            viewport_height=1080,
            user_agent=settings.crawl_user_agent,
            accept_downloads=False,
            chrome_channel="chromium" if settings.crawl_browser == "chromium" else None
        )
        
        # Memory-adaptive dispatcher for intelligent resource management
        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=settings.crawl_memory_threshold,
            max_session_permit=settings.max_concurrent_crawls
        )
        
        # Default crawl configuration - no extraction strategy by default
        self.default_run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),  # Convert to milliseconds
            delay_before_return_html=0.5,
            remove_overlay_elements=settings.crawl_remove_overlays,
            process_iframes=False,
            word_count_threshold=settings.crawl_min_words
        )
        
        # Note: Advanced features like AsyncUrlSeeder, LinkPreview, AdaptiveCrawler 
        # are not available in Crawl4AI 0.7.0+. Using standard crawling approaches.
    
    async def scrape_single_page(
        self,
        url: str,
        extraction_strategy: str = "css",
        wait_for: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        use_virtual_scroll: bool = None,
        virtual_scroll_config: Optional[Dict[str, Any]] = None
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
            if use_virtual_scroll or (use_virtual_scroll is None and settings.crawl_virtual_scroll):
                scroll_config = virtual_scroll_config or {}
                virtual_scroll = VirtualScrollConfig(
                    container_selector=scroll_config.get("container_selector", "body"),  # More generic selector
                    scroll_count=scroll_config.get("scroll_count", settings.crawl_scroll_count),
                    scroll_by=scroll_config.get("scroll_by", "window_height"),  # Use window height instead
                    wait_after_scroll=scroll_config.get("wait_after_scroll", 1.0)
                )
            
            # Configure extraction strategy
            extraction_strategy_obj = None
            if extraction_strategy == "llm":
                # Note: LLM extraction would need API keys configured
                extraction_strategy_obj = LLMExtractionStrategy(
                    provider="openai",  # Would need configuration
                    model="gpt-4"
                )
            elif extraction_strategy == "cosine":
                extraction_strategy_obj = CosineStrategy(
                    semantic_filter="meaningful content"
                )
            elif extraction_strategy == "json_css":
                # Basic JSON CSS extraction
                extraction_strategy_obj = JsonCssExtractionStrategy({
                    "title": "h1, h2, .title",
                    "content": "p, .content, .article-body",
                    "links": "a[href]"
                })
            
            # Prepare run configuration with advanced features
            run_config = CrawlerRunConfig(
                page_timeout=int(settings.crawler_timeout * 1000),
                delay_before_return_html=0.5,
                remove_overlay_elements=settings.crawl_remove_overlays,
                wait_for=wait_for,
                word_count_threshold=settings.crawl_min_words,
                extraction_strategy=extraction_strategy_obj,
                virtual_scroll_config=virtual_scroll,
                **(custom_config or {})
            )
            
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.debug(f"Starting advanced crawl of: {url}")
                
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )
                
                if not result.success:
                    raise ToolError(f"Failed to crawl {url}: {result.error_message}")
                
                # Extract links from HTML
                links = []
                if hasattr(result, 'links') and result.links:
                    for link in result.links:
                        if isinstance(link, dict):
                            href = link.get('href', '')
                        else:
                            href = str(link)
                        
                        if href:
                            # Convert relative URLs to absolute
                            absolute_url = urljoin(url, href)
                            links.append(absolute_url)
                
                # Extract images
                images = []
                if hasattr(result, 'media') and result.media:
                    for media in result.media:
                        if isinstance(media, dict):
                            src = media.get('src', '')
                            if media.get('type') == 'image' and src:
                                absolute_url = urljoin(url, src)
                                images.append(absolute_url)
                
                processing_time = time.time() - start_time
                
                # Create page content
                page_content = PageContent(
                    url=url,
                    title=result.metadata.get('title', '') if result.metadata else '',
                    content=result.cleaned_html if hasattr(result, 'cleaned_html') else result.markdown,
                    markdown=result.markdown if hasattr(result, 'markdown') else '',
                    html=result.html if settings.crawl_extract_media else None,
                    links=links,
                    images=images,
                    metadata={
                        'processing_time': processing_time,
                        'http_status': result.status_code if hasattr(result, 'status_code') else None,
                        'content_length': len(result.html) if result.html else 0,
                        'extraction_strategy': extraction_strategy,
                        **(result.metadata or {})
                    }
                )
                
                logger.debug(f"Successfully crawled {url} in {processing_time:.2f}s")
                return page_content
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            raise ToolError(f"Failed to scrape {url}: {str(e)}")
    
    async def crawl_website(
        self,
        request: CrawlRequest,
        progress_callback: Optional[callable] = None
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
            urls=request.url if isinstance(request.url, list) else [request.url]
        )
        
        try:
            # Prepare initial URLs
            initial_urls = request.url if isinstance(request.url, list) else [request.url]
            start_url = initial_urls[0]
            
            # Statistics tracking
            stats = CrawlStatistics()
            stats.total_pages_requested = request.max_pages
            
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
                        if request.exclude_patterns and any(pattern in url for pattern in request.exclude_patterns):
                            continue
                        if request.include_patterns and not any(pattern in url for pattern in request.include_patterns):
                            continue
                        discovered_urls.add(url)
                    
                    logger.info(f"Discovered {len(discovered_urls)} URLs from sitemap")
                else:
                    logger.info("No sitemap found, falling back to recursive crawling")
                    discovered_urls.add(start_url)
            except Exception as e:
                logger.warning(f"Sitemap discovery failed: {e}, falling back to recursive crawling")
                discovered_urls.add(start_url)
            
            if progress_callback:
                await progress_callback(2, 4)
            
            # Phase 2: Implement recursive crawling if sitemap not found
            if len(discovered_urls) == 1 and start_url in discovered_urls:
                # Use Crawl4AI's native deep crawling strategy
                pages = await self._native_deep_crawl(
                    start_url=start_url,
                    max_pages=request.max_pages,
                    max_depth=request.max_depth,
                    include_patterns=request.include_patterns,
                    exclude_patterns=request.exclude_patterns,
                    stats=stats,
                    progress_callback=progress_callback
                )
            else:
                # Use sitemap URLs - batch crawl approach
                urls_to_crawl = list(discovered_urls)[:request.max_pages]
                pages = await self._batch_crawl(urls_to_crawl, stats)
            
            # Set result pages and update statistics  
            result.pages = pages
            stats.total_pages_crawled = len(pages)
            logger.info(f"Crawled {len(pages)} pages successfully, {stats.total_pages_failed} failed")
            
            if progress_callback:
                await progress_callback(3, 4)
            
            # Phase 4: Calculate final statistics
            end_time = time.time()
            stats.crawl_duration_seconds = end_time - start_time
            
            if result.pages:
                stats.total_bytes_downloaded = sum(len(page.content or '') for page in result.pages)
                stats.total_links_discovered = sum(len(page.links) for page in result.pages)
                stats.total_images_found = sum(len(page.images) for page in result.pages)
                stats.unique_domains = len(set(urlparse(page.url).netloc for page in result.pages))
                
                if stats.crawl_duration_seconds > 0:
                    stats.pages_per_second = stats.total_pages_crawled / stats.crawl_duration_seconds
                
                if stats.total_pages_crawled > 0:
                    stats.average_page_size = stats.total_bytes_downloaded / stats.total_pages_crawled
            
            result.statistics = stats
            result.status = CrawlStatus.COMPLETED
            result.end_time = end_time
            
            if progress_callback:
                await progress_callback(4, 4)
            
            logger.info(f"Crawl completed: {stats.total_pages_crawled} pages in {stats.crawl_duration_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Crawl failed: {str(e)}")
            result.end_time = time.time()
            logger.error(f"Crawl failed: {e}")
            return result
    
    # Note: _traditional_crawl method removed as it used deprecated dispatcher.dispatch() API
    # Multi-URL crawling now handled directly in crawl_website() using arun_many()
    
    async def get_sitemap_urls(self, base_url: str) -> List[str]:
        """
        Extract URLs from sitemap.xml of a website.
        
        Args:
            base_url: Base URL of the website
            
        Returns:
            List of URLs found in sitemap
        """
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')  # Check robots.txt for sitemap references
        ]
        
        urls = set()
        
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                for sitemap_url in sitemap_urls:
                    try:
                        result = await crawler.arun(
                            url=sitemap_url,
                            config=CrawlerRunConfig(
                                page_timeout=10000,
                                remove_overlay_elements=False
                            )
                        )
                        
                        # Check both success and HTTP status code (avoid parsing 404 responses)
                        status_code = getattr(result, 'status_code', None)
                        is_success = result.success and result.html and (status_code is None or (200 <= status_code < 300))
                        
                        if is_success:
                            if 'robots.txt' in sitemap_url:
                                # Parse robots.txt for sitemap references
                                lines = result.html.split('\n')
                                for line in lines:
                                    if line.lower().startswith('sitemap:'):
                                        sitemap_ref = line.split(':', 1)[1].strip()
                                        sitemap_urls.append(sitemap_ref)
                            else:
                                # Parse XML sitemap
                                try:
                                    root = ET.fromstring(result.html)
                                    
                                    # Handle sitemap index
                                    for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                                        loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                        if loc_elem is not None:
                                            sitemap_urls.append(loc_elem.text)
                                    
                                    # Handle regular sitemap
                                    for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                                        loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                        if loc_elem is not None:
                                            urls.add(loc_elem.text)
                                            
                                except ET.ParseError:
                                    logger.warning(f"Failed to parse XML sitemap: {sitemap_url}")
                                    
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
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        stats: CrawlStatistics = None,
        progress_callback: Optional[callable] = None
    ) -> List[PageContent]:
        """
        Implement breadth-first recursive crawling to discover linked pages.
        """
        pages = []
        visited_urls = set()
        url_queue = [(start_url, 0)]  # (url, depth)
        domain = urlparse(start_url).netloc
        
        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.5,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            stream=False
        )
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            while url_queue and len(pages) < max_pages:
                current_url, depth = url_queue.pop(0)
                
                if current_url in visited_urls or depth > max_depth:
                    continue
                    
                # Apply include/exclude patterns
                if exclude_patterns and any(pattern in current_url for pattern in exclude_patterns):
                    continue
                if include_patterns and not any(pattern in current_url for pattern in include_patterns):
                    continue
                
                visited_urls.add(current_url)
                logger.info(f"Crawling [{depth}/{max_depth}]: {current_url}")
                
                try:
                    result = await crawler.arun(url=current_url, config=run_config)
                    
                    if result.success:
                        page_content = self._process_crawl_result(result, extract_raw_html=False)
                        pages.append(page_content)
                        stats.total_pages_crawled += 1
                        
                        # Extract and queue new links if we haven't reached max depth
                        if depth < max_depth and len(pages) < max_pages:
                            new_links = self._extract_links_from_result(result, domain)
                            for link in new_links:
                                if link not in visited_urls and (link, depth + 1) not in url_queue:
                                    url_queue.append((link, depth + 1))
                                    
                        if progress_callback:
                            progress = min(len(pages) / max_pages, 1.0)
                            await progress_callback(2 + int(progress), 4)
                            
                    else:
                        stats.total_pages_failed += 1
                        logger.warning(f"Failed to crawl {current_url}: {result.error_message}")
                        
                except Exception as e:
                    stats.total_pages_failed += 1
                    logger.error(f"Error crawling {current_url}: {e}")
                    
        logger.info(f"Recursive crawl discovered {len(pages)} pages across {max_depth} depth levels")
        return pages
    
    async def _batch_crawl(self, urls: List[str], stats: CrawlStatistics) -> List[PageContent]:
        """
        Crawl a batch of URLs concurrently using arun_many.
        """
        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.5,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            stream=False
        )
        
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=settings.crawl_memory_threshold,
            max_session_permit=min(settings.max_concurrent_crawls, len(urls))
        )
        
        pages = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            logger.info(f"Starting batch crawl of {len(urls)} URLs")
            
            crawl_results = await crawler.arun_many(
                urls=urls,
                config=run_config,
                dispatcher=dispatcher
            )
            
            for crawl_result in crawl_results:
                if crawl_result.success:
                    page_content = self._process_crawl_result(crawl_result, extract_raw_html=False)
                    pages.append(page_content)
                    stats.total_pages_crawled += 1
                else:
                    stats.total_pages_failed += 1
                    logger.warning(f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}")
                    
        return pages
    
    def _process_crawl_result(self, crawl_result, extract_raw_html: bool = False) -> PageContent:
        """
        Process a single crawl result into PageContent.
        """
        # Extract links
        links = []
        if hasattr(crawl_result, 'links') and crawl_result.links:
            for link in crawl_result.links:
                if isinstance(link, dict):
                    href = link.get('href', '')
                else:
                    href = str(link)
                
                if href:
                    absolute_url = urljoin(crawl_result.url, href)
                    links.append(absolute_url)
        
        # Extract images
        images = []
        if hasattr(crawl_result, 'media') and crawl_result.media:
            for media in crawl_result.media:
                if isinstance(media, dict):
                    src = media.get('src', '')
                    if media.get('type') == 'image' and src:
                        absolute_url = urljoin(crawl_result.url, src)
                        images.append(absolute_url)
        
        return PageContent(
            url=crawl_result.url,
            title=crawl_result.metadata.get('title', '') if crawl_result.metadata else '',
            content=crawl_result.cleaned_html if hasattr(crawl_result, 'cleaned_html') else crawl_result.markdown,
            markdown=crawl_result.markdown if hasattr(crawl_result, 'markdown') else '',
            html=crawl_result.html if extract_raw_html else None,
            links=links,
            images=images,
            metadata={
                'http_status': crawl_result.status_code if hasattr(crawl_result, 'status_code') else None,
                'content_length': len(crawl_result.html) if crawl_result.html else 0,
                'extraction_method': 'recursive_crawl',
                **(crawl_result.metadata or {})
            }
        )
    
    def _extract_links_from_result(self, crawl_result, domain: str) -> List[str]:
        """
        Extract same-domain links from a crawl result.
        """
        links = []
        if hasattr(crawl_result, 'links') and crawl_result.links:
            for link in crawl_result.links:
                if isinstance(link, dict):
                    href = link.get('href', '')
                else:
                    href = str(link)
                
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
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        stats: CrawlStatistics = None,
        progress_callback: Optional[callable] = None
    ) -> List[PageContent]:
        """
        Use Crawl4AI's native deep crawling capabilities with proper URL filtering.
        
        This replaces our custom implementation with Crawl4AI's native strategies
        to avoid crawling invalid URLs like /internal and /external.
        """
        domain = urlparse(start_url).netloc
        
        # Create comprehensive filter chain to avoid invalid URLs
        filter_chain = FilterChain([
            # Domain filter to stay within same domain
            DomainFilter(
                allowed_domains=[domain],
                blocked_domains=[]
            ),
            
            # Content type filter for HTML pages only
            ContentTypeFilter(
                allowed_types=["text/html"],
                check_extension=True
            ),
            
            # URL pattern filter to exclude invalid patterns
            URLPatternFilter(
                patterns=settings.crawl_exclude_url_patterns + [
                    # File extensions are always excluded
                    "*.css", "*.js", "*.jpg", "*.jpeg", "*.png", "*.gif", 
                    "*.svg", "*.ico", "*.pdf", "*.zip", "*.xml", "*.json"
                ],
                reverse=True,  # Exclude matching patterns
                use_glob=True
            )
        ])
        
        # Apply user-defined include/exclude patterns if provided
        if include_patterns:
            user_include_filter = URLPatternFilter(
                patterns=include_patterns,
                reverse=False,
                use_glob=True
            )
            filter_chain.filters.append(user_include_filter)
            
        if exclude_patterns:
            user_exclude_filter = URLPatternFilter(
                patterns=exclude_patterns,
                reverse=True,
                use_glob=True
            )
            filter_chain.filters.append(user_exclude_filter)
        
        # Use BFS strategy with comprehensive filtering
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,  # Stay within same domain
            max_pages=max_pages,
            filter_chain=filter_chain
        )
        
        # Configure crawler with deep crawl strategy
        crawl_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.5,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words,
            deep_crawl_strategy=deep_crawl_strategy,
            verbose=settings.debug
        )
        
        pages = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            logger.info(f"Starting native Crawl4AI deep crawl from {start_url} (max_pages: {max_pages}, max_depth: {max_depth})")
            
            try:
                # Use Crawl4AI's native deep crawling
                result = await crawler.arun(url=start_url, config=crawl_config)
                
                if result.success:
                    # Process main page
                    page_content = self._process_crawl_result(result, extract_raw_html=False)
                    pages.append(page_content)
                    stats.total_pages_crawled += 1
                    
                    # Check if deep crawl results are available
                    if hasattr(result, 'deep_crawl_results') and result.deep_crawl_results:
                        for crawl_result in result.deep_crawl_results:
                            if crawl_result.success:
                                page_content = self._process_crawl_result(crawl_result, extract_raw_html=False)
                                pages.append(page_content)
                                stats.total_pages_crawled += 1
                                
                                if progress_callback:
                                    progress = min(len(pages) / max_pages, 1.0)
                                    await progress_callback(2 + int(progress * 0.6), 4)
                            else:
                                stats.total_pages_failed += 1
                                logger.warning(f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}")
                    else:
                        # Fallback to manual BFS if deep_crawl_results not available
                        logger.info("Deep crawl results not available, using manual BFS approach with filtering")
                        additional_pages = await self._manual_bfs_with_filtering(
                            crawler, start_url, max_pages - 1, max_depth, filter_chain, stats
                        )
                        pages.extend(additional_pages)
                        
                else:
                    stats.total_pages_failed += 1
                    logger.error(f"Failed to start deep crawl from {start_url}: {result.error_message}")
                    
            except Exception as e:
                stats.total_pages_failed += 1
                logger.error(f"Error during native deep crawl: {e}")
                
        logger.info(f"Native deep crawl completed: {len(pages)} pages discovered with proper URL filtering")
        return pages
    
    async def _manual_bfs_with_filtering(
        self,
        crawler: AsyncWebCrawler,
        start_url: str,
        max_pages: int,
        max_depth: int,
        filter_chain: FilterChain,
        stats: CrawlStatistics
    ) -> List[PageContent]:
        """
        Manual BFS crawling with proper URL filtering as fallback.
        
        This is used when Crawl4AI's native deep crawl doesn't return deep_crawl_results.
        """
        pages = []
        visited_urls = {start_url}
        url_queue = [(start_url, 0)]
        
        run_config = CrawlerRunConfig(
            page_timeout=int(settings.crawler_timeout * 1000),
            delay_before_return_html=0.5,
            remove_overlay_elements=settings.crawl_remove_overlays,
            word_count_threshold=settings.crawl_min_words
        )
        
        while url_queue and len(pages) < max_pages:
            current_url, depth = url_queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            logger.info(f"Manual BFS crawl [{depth}/{max_depth}] ({len(pages)+1}/{max_pages}): {current_url}")
            
            try:
                result = await crawler.arun(url=current_url, config=run_config)
                
                if result.success:
                    page_content = self._process_crawl_result(result, extract_raw_html=False)
                    pages.append(page_content)
                    stats.total_pages_crawled += 1
                    
                    # Extract links using crawl4ai's built-in link extraction
                    if hasattr(result, 'links') and result.links and depth < max_depth:
                        for link in result.links:
                            href = self._extract_href(link)
                            if href:
                                absolute_url = urljoin(current_url, href)
                                
                                # Apply filter chain to validate URL
                                if absolute_url not in visited_urls and await filter_chain.apply(absolute_url):
                                    visited_urls.add(absolute_url)
                                    url_queue.append((absolute_url, depth + 1))
                                    
                else:
                    stats.total_pages_failed += 1
                    logger.warning(f"Failed to crawl {current_url}: {result.error_message}")
                    
            except Exception as e:
                stats.total_pages_failed += 1
                logger.error(f"Error crawling {current_url}: {e}")
                
        return pages
    
    def _extract_href(self, link) -> str:
        """Extract href from link object."""
        if isinstance(link, dict):
            return link.get('href', '')
        if isinstance(link, str):
            return link
        else:
            return str(link) if link else ''
    
    async def crawl_repository(
        self,
        repo_url: str,
        clone_path: Optional[str] = None,
        file_patterns: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
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
            request_id=request_id,
            status=CrawlStatus.RUNNING,
            urls=[repo_url]
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
            repo = git.Repo.clone_from(repo_url, clone_dir)
            
            if progress_callback:
                await progress_callback(2, 4, "Analyzing repository structure")
            
            # Analyze repository files with adaptive processing
            file_patterns = file_patterns or ['*.py', '*.js', '*.ts', '*.md', '*.txt', '*.json', '*.yaml', '*.yml']
            pages = []
            stats = CrawlStatistics()
            
            # Collect all matching files first
            all_files = []
            for pattern in file_patterns:
                all_files.extend(clone_dir.rglob(pattern))
            
            # Filter to actual files and sort by importance (prioritize documentation and config files)
            def file_priority(file_path: Path) -> int:
                """Assign priority to files for better processing order."""
                name = file_path.name.lower()
                if name in ['readme.md', 'readme.txt', 'readme.rst']:
                    return 0  # Highest priority
                elif name.startswith('readme'):
                    return 1
                elif file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                    return 2  # Documentation files
                elif name in ['package.json', 'requirements.txt', 'pyproject.toml', 'cargo.toml']:
                    return 3  # Config files
                else:
                    return 4  # Source files
            
            valid_files = [f for f in all_files if f.is_file()]
            valid_files.sort(key=file_priority)
            
            # Process files in batches with adaptive filtering
            processed_count = 0
            batch_size = 50  # Process files in batches to avoid memory issues
            
            for i in range(0, len(valid_files), batch_size):
                batch = valid_files[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Skip empty or very small files unless they're important
                        if len(content.strip()) < 50 and file_priority(file_path) > 3:
                            continue
                        
                        # Create relative path from repo root
                        relative_path = file_path.relative_to(clone_dir)
                        file_url = f"{repo_url}/blob/main/{relative_path}"
                        
                        # Enhanced metadata with repository analysis
                        file_metadata = {
                            'file_path': str(relative_path),
                            'file_extension': file_path.suffix,
                            'file_size': file_path.stat().st_size,
                            'repository': repo_url,
                            'source_type': 'repository',
                            'file_priority': file_priority(file_path),
                            'processing_method': 'adaptive_batch'
                        }
                        
                        # Add language detection for code files
                        if file_path.suffix in ['.py', '.js', '.ts', '.go', '.rs', '.cpp', '.java']:
                            file_metadata['language'] = file_path.suffix[1:]  # Remove dot
                        
                        page_content = PageContent(
                            url=file_url,
                            title=f"{relative_path.name} - {repo_name}",
                            content=content,
                            metadata=file_metadata
                        )
                        
                        pages.append(page_content)
                        stats.total_pages_crawled += 1
                        stats.total_bytes_downloaded += len(content)
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        result.errors.append(f"Could not read {file_path}: {str(e)}")
                
                # Progress update after each batch
                if progress_callback and processed_count % batch_size == 0:
                    await progress_callback(
                        2, 4, 
                        f"Processed {processed_count} files from repository"
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
            result.end_time = end_time
            
            logger.info(f"Repository crawl completed: {len(pages)} files in {stats.crawl_duration_seconds:.2f}s")
            return result
            
        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Repository crawl failed: {str(e)}")
            result.end_time = time.time()
            logger.error(f"Repository crawl failed: {e}")
            return result
    
    async def crawl_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        progress_callback: Optional[callable] = None
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
            urls=[f"file://{directory_path}"]
        )
        
        try:
            dir_path = Path(directory_path)
            if not dir_path.exists():
                raise ToolError(f"Directory does not exist: {directory_path}")
            
            file_patterns = file_patterns or ['*.py', '*.js', '*.ts', '*.md', '*.txt', '*.json', '*.yaml', '*.yml']
            pages = []
            stats = CrawlStatistics()
            
            if progress_callback:
                await progress_callback(0, 1, f"Scanning directory: {directory_path}")
            
            # Collect all matching files
            all_files = []
            for pattern in file_patterns:
                if recursive:
                    all_files.extend(dir_path.rglob(pattern))
                else:
                    all_files.extend(dir_path.glob(pattern))
            
            # Remove duplicates and ensure they're files
            unique_files = list(set(f for f in all_files if f.is_file()))
            
            if progress_callback:
                await progress_callback(0, len(unique_files), f"Processing {len(unique_files)} files")
            
            # Adaptive file processing with prioritization
            def file_relevance_score(file_path: Path) -> float:
                """Calculate relevance score for file processing priority."""
                score = 0.5  # Base score
                
                # Higher priority for documentation and config files
                name_lower = file_path.name.lower()
                if name_lower.startswith('readme'):
                    score += 0.4
                elif file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                    score += 0.3
                elif name_lower in ['package.json', 'requirements.txt', 'pyproject.toml']:
                    score += 0.3
                elif file_path.suffix.lower() in ['.py', '.js', '.ts']:
                    score += 0.2
                
                # Penalize very large files or hidden files
                if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                    score -= 0.2
                if name_lower.startswith('.'):
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
                batch = unique_files[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Skip empty files or very large files (unless high priority)
                        relevance = file_relevance_score(file_path)
                        if (len(content.strip()) < 10 and relevance < 0.7) or \
                           (len(content) > 500000 and relevance < 0.8):  # 500KB threshold
                            continue
                        
                        # Create file URL
                        file_url = file_path.as_uri()
                        
                        # Enhanced metadata with adaptive information
                        file_metadata = {
                            'file_path': str(file_path),
                            'relative_path': str(file_path.relative_to(dir_path)),
                            'file_extension': file_path.suffix,
                            'file_size': file_path.stat().st_size,
                            'source_type': 'directory',
                            'relevance_score': relevance,
                            'processing_method': 'adaptive_directory'
                        }
                        
                        # Add content type hints
                        if file_path.suffix.lower() in ['.py', '.js', '.ts', '.go', '.rs', '.java']:
                            file_metadata['content_type'] = 'source_code'
                        elif file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                            file_metadata['content_type'] = 'documentation'
                        elif file_path.suffix.lower() in ['.json', '.yaml', '.yml', '.toml']:
                            file_metadata['content_type'] = 'configuration'
                        
                        page_content = PageContent(
                            url=file_url,
                            title=file_path.name,
                            content=content,
                            metadata=file_metadata
                        )
                        
                        pages.append(page_content)
                        stats.total_pages_crawled += 1
                        stats.total_bytes_downloaded += len(content)
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        result.errors.append(f"Could not read {file_path}: {str(e)}")
                        stats.total_pages_failed += 1
                
                # Progress update after each batch
                if progress_callback:
                    await progress_callback(
                        processed_count, 
                        len(unique_files), 
                        f"Processed {processed_count}/{len(unique_files)} files"
                    )
            
            if progress_callback:
                await progress_callback(len(unique_files), len(unique_files), "Directory crawl completed")
            
            # Finalize result
            end_time = time.time()
            stats.crawl_duration_seconds = end_time - start_time
            stats.total_pages_requested = len(unique_files)
            stats.unique_domains = 1
            
            result.pages = pages
            result.statistics = stats
            result.status = CrawlStatus.COMPLETED
            result.end_time = end_time
            
            logger.info(f"Directory crawl completed: {len(pages)} files in {stats.crawl_duration_seconds:.2f}s")
            return result
            
        except Exception as e:
            result.status = CrawlStatus.FAILED
            result.errors.append(f"Directory crawl failed: {str(e)}")
            result.end_time = time.time()
            logger.error(f"Directory crawl failed: {e}")
            return result