# Web Crawler Refactoring Plan

## Overview
Refactor `crawler_mcp/crawlers/web.py` (1116 lines) into smaller, focused modules while keeping the original file intact for rollback safety.

## Current File Analysis
The current `web.py` file handles multiple responsibilities:
- Browser configuration and GPU optimization
- Sitemap discovery and URL seeding
- Deep crawl strategy construction
- Result processing and sanitization
- Main crawl execution with multiple strategies
- Progress tracking and error handling

## Proposed Module Structure

### Directory: `crawler_mcp/crawlers/web/`

#### 1. `browser.py` (~200 lines)
**Responsibility**: Browser configuration, GPU optimization, and session management

**Extracted Code:**
- Browser configuration logic (lines 133-178)
- GPU settings and optimization
- `suppress_stdout()` context manager (lines 47-57)

**New Functionality:**
```python
class BrowserManager:
    """Manages browser configuration and session lifecycle."""

    def create_browser_config(self) -> BrowserConfig
    def get_gpu_optimized_args(self) -> list[str]
    def get_performance_args(self) -> list[str]
    def validate_gpu_availability(self) -> bool
    def create_browser_session(self) -> AsyncWebCrawler
    async def close_browser_session(self, browser: AsyncWebCrawler) -> None

    @contextmanager
    def suppress_stdout(self) -> Any

    def get_optimized_browser_config(
        self,
        enable_gpu: bool | None = None,
        light_mode: bool = True,
        text_mode: bool = False,
        headless: bool = True,
    ) -> BrowserConfig
```

**Browser Configuration Features:**
- **GPU Acceleration**: Conditional GPU flags based on hardware availability
- **Performance Optimization**: Aggressive caching, memory management, network optimization
- **Environment Detection**: Different configs for CI/containers vs local development
- **Resource Management**: Memory limits, connection pooling, timeout configuration

**Optimized Settings:**
```python
# High-performance config for i7-13700K + RTX 4070
extra_args = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    # GPU acceleration (conditional)
    "--enable-gpu",
    "--enable-accelerated-2d-canvas",
    "--enable-gpu-compositing",
    "--enable-gpu-rasterization",
    # Network optimizations
    "--max-connections-per-host=30",
    "--enable-quic",
    "--enable-tcp-fast-open",
    # Memory optimizations
    "--aggressive-cache-discard",
    "--memory-pressure-off",
]
```

#### 2. `sitemap.py` (~150 lines)
**Responsibility**: Sitemap discovery, robots.txt parsing, and URL seeding

**Extracted Methods:**
- `_discover_sitemap_seeds()` (lines 786-818)
- `_extract_sitemaps_from_robots()` (lines 820-834)
- `_parse_sitemap()` (lines 836-887)
- `_fetch_text()` (lines 889-908)

**New Functionality:**
```python
class SitemapDiscovery:
    """Handles sitemap discovery and URL seeding for crawl optimization."""

    async def discover_sitemap_seeds(self, start_url: str, limit: int) -> list[str]
    async def extract_sitemaps_from_robots(self, robots_url: str) -> list[str]
    async def parse_sitemap(
        self, sitemap_url: str, base: str, remaining: int
    ) -> list[str]
    async def parse_sitemap_index(self, root: ET.Element, base: str, remaining: int) -> list[str]
    async def parse_urlset(self, root: ET.Element, remaining: int) -> list[str]
    async def fetch_text(self, url: str, timeout: int = 10) -> str
    async def validate_sitemap_urls(self, urls: list[str], base_domain: str) -> list[str]
    def normalize_sitemap_urls(self, urls: list[str], base: str) -> list[str]
    async def discover_alternative_sitemaps(self, base_url: str) -> list[str]
```

**Enhanced Features:**
- **Recursive Sitemap Parsing**: Handle sitemap indexes with nested sitemaps
- **Error Recovery**: Graceful handling of malformed XML, network errors
- **URL Validation**: Domain validation, duplicate removal, URL normalization
- **Alternative Discovery**: Check common sitemap locations beyond robots.txt
- **Caching**: Cache sitemap responses to avoid re-fetching

**Sitemap Discovery Strategy:**
1. Check robots.txt for sitemap declarations
2. Try common sitemap locations (/sitemap.xml, /sitemap_index.xml)
3. Parse sitemap indexes recursively
4. Validate and normalize discovered URLs
5. Return prioritized list based on URL patterns

#### 3. `strategies.py` (~250 lines)
**Responsibility**: Deep crawl strategy construction and configuration

**Extracted Methods:**
- `_build_deep_crawl_strategy()` (lines 666-784)
- `_build_run_config()` (lines 519-664)
- Filter and scorer configuration logic

**New Functionality:**
```python
class CrawlStrategyBuilder:
    """Builds and configures deep crawling strategies."""

    def build_deep_crawl_strategy(
        self, request: CrawlRequest, sitemap_seeds: list[str]
    ) -> BFSDeepCrawlStrategy | None

    def build_filter_chain(
        self,
        include_patterns: list[str],
        exclude_patterns: list[str]
    ) -> FilterChain | None

    def build_keyword_scorer(
        self,
        include_patterns: list[str],
        sitemap_seeds: list[str]
    ) -> KeywordRelevanceScorer | None

    def build_run_config(
        self, request: CrawlRequest, sitemap_seeds: list[str] | None = None
    ) -> CrawlerRunConfig

    def get_extraction_strategy(self, strategy_name: str) -> Any | None
    def get_chunking_strategy(self, strategy_name: str, options: dict) -> Any | None
    def create_content_filter(self) -> PruningContentFilter
    def create_markdown_generator(self) -> Any
    def get_scraping_strategy(self) -> Any | None
```

**Strategy Configuration:**
- **BFS Deep Crawl**: Optimized breadth-first search with intelligent filtering
- **Filter Chains**: URL pattern filtering with include/exclude rules
- **Keyword Scoring**: Relevance scoring based on URL patterns and content
- **Extraction Strategies**: LLM, Cosine, and default content extraction
- **Content Filtering**: Intelligent content pruning for quality

**Advanced Features:**
```python
# Intelligent keyword extraction from sitemap URLs
keywords = []
for url in sitemap_seeds:
    path_tokens = urlparse(url).path.replace("-", " ").replace("_", " ").split()
    keywords.extend([t for t in path_tokens if len(t) > 2])

# Dynamic ef parameter for HNSW search optimization
ef_value = min(256, max(64, limit * 4))

# Memory-adaptive configuration
if memory_threshold_percent:
    run_config.memory_threshold_percent = memory_threshold_percent
```

#### 4. `results.py` (~200 lines)
**Responsibility**: Result processing, sanitization, and conversion

**Extracted Methods:**
- `_sanitize_crawl_result()` (lines 356-409)
- `_safe_get_markdown()` (lines 411-467)
- `_to_page_content()` (lines 469-517)
- `_extract_text_from_html()` (lines 331-354)

**New Functionality:**
```python
class ResultProcessor:
    """Processes and sanitizes crawl results from Crawl4AI."""

    def sanitize_crawl_result(self, result: Crawl4aiResult) -> Crawl4aiResult
    def safe_get_markdown(self, result: Crawl4aiResult) -> str
    def to_page_content(self, result: Crawl4aiResult) -> PageContent
    def extract_text_from_html(self, html: str | None) -> str
    def validate_content_quality(self, content: str) -> bool
    def extract_metadata(self, result: Crawl4aiResult) -> dict[str, Any]
    def process_links_and_media(self, result: Crawl4aiResult) -> tuple[list[str], list[str]]
    def calculate_content_metrics(self, content: str) -> dict[str, int]

    def process_batch_results(
        self, results: list[Crawl4aiResult]
    ) -> tuple[list[PageContent], list[str]]
```

**Result Processing Features:**
- **Hash Issue Resolution**: Fix Crawl4AI integer hash placeholder issues
- **Content Validation**: Ensure content quality and completeness
- **Metadata Extraction**: Rich metadata from crawl results
- **Link Processing**: Extract and validate internal/external links
- **Media Processing**: Extract and process image references
- **Error Recovery**: Graceful handling of corrupted or incomplete results

**Sanitization Logic:**
```python
# Fix integer hash issues with markdown field
if hasattr(result, "_markdown") and isinstance(result._markdown, int):
    result._markdown = MarkdownGenerationResult(
        raw_markdown="",
        markdown_with_citations="",
        references_markdown="",
        fit_markdown=None,
        fit_html=None,
    )

# Validate markdown content quality
content = result.markdown.fit_markdown or result.markdown.raw_markdown
if isinstance(content, str) and len(content) > 16:  # Avoid hash placeholders
    return content
```

#### 5. `crawler.py` (~316 lines)
**Responsibility**: Main web crawling orchestration and execution

**Extracted Methods:**
- `WebCrawlStrategy` class core methods (lines 60-329)
- `_crawl_using_deep_strategy()` (lines 910-1011)
- `_crawl_using_arun_many()` (lines 1013-1111)

**New Functionality:**
```python
class WebCrawlStrategy(BaseCrawlStrategy):
    """High-performance web crawling strategy with modular components."""

    def __init__(self) -> None:
        super().__init__()
        self.browser_manager = BrowserManager()
        self.sitemap_discovery = SitemapDiscovery()
        self.strategy_builder = CrawlStrategyBuilder()
        self.result_processor = ResultProcessor()

    async def validate_request(self, request: CrawlRequest) -> bool
    async def execute(
        self,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult

    async def crawl_using_deep_strategy(
        self, browser: Any, first_url: str, run_config: Any, max_pages: int
    ) -> tuple[list[Any], list[str]]

    async def crawl_using_arun_many(
        self,
        browser: Any,
        sitemap_urls: list[str],
        run_config: Any,
        request: Any,
        progress_callback: Any,
    ) -> tuple[list[Any], list[str]]

    async def process_crawl_results(
        self, successful_results: list[Any], max_pages: int, progress_callback: Any
    ) -> tuple[list[PageContent], dict[str, Any]]
```

**Crawl Execution Features:**
- **Dual Strategy Support**: BFS deep crawling and sitemap-based arun_many
- **Progress Tracking**: Detailed progress reporting throughout the crawl
- **Memory Management**: Integration with memory manager for pressure monitoring
- **Error Collection**: Comprehensive error tracking and reporting
- **Statistics Generation**: Detailed crawl statistics and performance metrics

**Execution Flow:**
1. **Setup**: Initialize browser, discover sitemaps, build strategy
2. **Strategy Selection**: Choose between deep crawl or arun_many based on context
3. **Execution**: Run selected crawl strategy with progress tracking
4. **Processing**: Process results, handle errors, generate statistics
5. **Cleanup**: Close browser, perform cleanup, return results

#### 6. `__init__.py` (~50 lines)
**Responsibility**: Package initialization and public API

```python
"""Web crawler modules for high-performance website crawling."""

from .browser import BrowserManager
from .sitemap import SitemapDiscovery
from .strategies import CrawlStrategyBuilder
from .results import ResultProcessor
from .crawler import WebCrawlStrategy

class WebCrawlerService:
    """Unified web crawling service using modular components."""

    def __init__(self):
        self.browser_manager = BrowserManager()
        self.sitemap_discovery = SitemapDiscovery()
        self.strategy_builder = CrawlStrategyBuilder()
        self.result_processor = ResultProcessor()
        self.crawler = WebCrawlStrategy()

    async def crawl_website(
        self,
        url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        progress_callback=None,
    ) -> CrawlResult:
        """High-level web crawling interface."""
        request = CrawlRequest(
            url=url,
            max_pages=max_pages,
            max_depth=max_depth,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        return await self.crawler.execute(request, progress_callback)

__all__ = [
    "WebCrawlStrategy",
    "WebCrawlerService",
    "BrowserManager",
    "SitemapDiscovery",
    "CrawlStrategyBuilder",
    "ResultProcessor"
]
```

## Migration Strategy

### Phase 1: Create Modular Structure (No Breaking Changes)
1. Create `crawler_mcp/crawlers/web/` directory
2. Extract code into new modules without changing original file
3. Ensure WebCrawlStrategy uses the new modular components internally
4. Add comprehensive tests for each module

### Phase 2: Enhanced Functionality
1. Improve browser configuration with better hardware detection
2. Enhanced sitemap discovery with alternative strategies
3. More sophisticated crawl strategies with intelligent routing
4. Better result processing with quality validation

### Phase 3: Integration Testing
1. Test complete web crawling pipeline with various websites
2. Performance benchmarks comparing old vs new implementation
3. Validate all browser configurations and optimizations
4. Memory usage profiling during large crawls

### Phase 4: Gradual Migration (Optional)
1. Add feature flag: `use_modular_web_crawler = Field(default=False)`
2. Update original `web.py` to conditionally use new modules:
   ```python
   # At top of web.py
   from ..config import settings
   if settings.use_modular_web_crawler:
       from .web import WebCrawlStrategy as ModularWebCrawlStrategy
       WebCrawlStrategy = ModularWebCrawlStrategy
   ```

## Benefits

### Code Organization
- **Clear Separation**: Browser, sitemap, strategy, and result concerns are isolated
- **Testable Components**: Each module can be thoroughly tested independently
- **Reusable Logic**: Browser and sitemap modules can be used elsewhere
- **Single Responsibility**: Each module has one focused purpose

### Performance Benefits
- **Optimized Browser Config**: Hardware-specific optimizations can be tuned independently
- **Intelligent Sitemap Discovery**: Enhanced URL seeding for better crawl efficiency
- **Strategy Optimization**: Crawl strategies can be fine-tuned without affecting other logic
- **Result Processing**: Sanitization and processing can be optimized separately

### Developer Experience
- **Easier Debugging**: Issues can be isolated to specific modules
- **Clear Interfaces**: Well-defined APIs between components
- **Better Testing**: Isolated functionality allows for comprehensive unit tests
- **Modular Development**: Different developers can work on different modules

## Dependencies and Imports

### Shared Dependencies
All modules will import:
```python
import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai import CrawlResult as Crawl4aiResult
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from ...config import settings
from ...models.crawl import CrawlRequest, CrawlResult, CrawlStatistics, CrawlStatus, PageContent
from ...types.crawl4ai_types import DefaultMarkdownGeneratorImpl
from ..base import BaseCrawlStrategy
```

### Inter-Module Dependencies
- `crawler.py` depends on all other modules (browser, sitemap, strategies, results)
- `strategies.py` may use `sitemap.py` for URL-based keyword extraction
- `results.py` is independent and can be used with any crawl strategy
- `browser.py` and `sitemap.py` are independent of each other

## Testing Strategy

### Unit Tests
Each module will have comprehensive unit tests:

**test_browser.py**:
- Browser configuration generation
- GPU detection and optimization
- Session lifecycle management
- Error handling for browser failures

**test_sitemap.py**:
- Sitemap discovery accuracy
- XML parsing robustness
- URL validation and normalization
- Error handling for malformed sitemaps

**test_strategies.py**:
- Strategy configuration correctness
- Filter chain construction
- Keyword extraction from URLs
- Run configuration generation

**test_results.py**:
- Result sanitization effectiveness
- Content quality validation
- Metadata extraction accuracy
- Error handling for corrupted results

**test_crawler.py**:
- Complete crawl pipeline integration
- Strategy selection logic
- Progress reporting accuracy
- Error aggregation and statistics

### Integration Tests
- End-to-end website crawling workflows
- Performance comparison with original implementation
- Memory usage during large crawls
- Browser optimization effectiveness

### Performance Tests
- Browser startup and configuration time
- Sitemap discovery speed
- Strategy execution performance
- Result processing throughput

## Risk Mitigation

### Rollback Strategy
- Original `web.py` remains completely unchanged
- Can instantly disable modular implementation with feature flag
- No breaking changes to existing APIs

### Error Handling
- All modules maintain same error handling patterns as original
- Browser session errors are isolated and handled gracefully
- Comprehensive logging maintained throughout

### Performance Validation
- Benchmark all operations before and after refactoring
- Validate browser optimization effectiveness
- Memory usage profiling
- Crawl speed and accuracy validation

## Configuration Enhancement

### New Configuration Options
```python
# In config.py
class Settings:
    # Web crawler settings
    web_enable_gpu_detection: bool = Field(default=True, description="Auto-detect GPU availability")
    web_browser_pool_size: int = Field(default=1, description="Number of concurrent browser sessions")
    web_sitemap_cache_ttl: int = Field(default=3600, description="Sitemap cache TTL in seconds")
    web_result_quality_threshold: float = Field(default=0.1, description="Minimum content quality score")
    web_max_sitemap_urls: int = Field(default=1000, description="Maximum URLs to extract from sitemaps")
    web_strategy_selection_auto: bool = Field(default=True, description="Auto-select crawl strategy")
```

## Future Enhancements

### Advanced Browser Management
- **Browser Pool**: Multiple concurrent browser sessions
- **Session Persistence**: Reuse browser sessions across crawls
- **Resource Monitoring**: Real-time memory and CPU usage tracking
- **Adaptive Configuration**: Dynamic browser config based on target website

### Enhanced Sitemap Discovery
- **Intelligent Caching**: Smart sitemap caching with invalidation
- **Alternative Sources**: RSS feeds, API endpoints for URL discovery
- **Content Analysis**: Analyze existing content to guide crawl strategy
- **Parallel Discovery**: Concurrent sitemap discovery and parsing

### Advanced Strategies
- **ML-Based Routing**: Machine learning for crawl strategy selection
- **Adaptive Depth**: Dynamic depth adjustment based on content quality
- **Priority Queues**: Intelligent URL prioritization during crawling
- **Real-time Optimization**: Strategy adjustment based on crawl progress

This refactoring transforms a single 1116-line file into 5 focused modules of ~150-300 lines each, significantly improving maintainability while enhancing the sophisticated web crawling capabilities.
