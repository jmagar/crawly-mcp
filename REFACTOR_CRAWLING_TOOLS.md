# Crawling Tools Refactoring Plan

## Overview
Refactor `crawler_mcp/tools/crawling.py` (556 lines) into smaller, focused modules while keeping the original file intact for rollback safety.

## Current File Analysis
The current `crawling.py` file handles multiple responsibilities:
- Input type detection (directory, repository, website)
- RAG processing coordination
- Result formatting and unification
- FastMCP tool implementations (scrape, crawl)
- Progress tracking and error handling

## Proposed Module Structure

### Directory: `crawler_mcp/tools/crawling/`

#### 1. `detection.py` (~100 lines)
**Responsibility**: Input type detection and parameter normalization

**Extracted Functions:**
- `_detect_crawl_type_and_params()` (lines 20-54)
- Input validation and normalization logic

**New Functionality:**
```python
class CrawlTypeDetector:
    """Intelligent input type detection for unified crawling interface."""

    def detect_crawl_type_and_params(self, target: str) -> tuple[str, dict[str, Any]]
    def is_local_path(self, target: str) -> bool
    def is_git_repository(self, target: str) -> bool
    def is_web_url(self, target: str) -> bool
    def normalize_local_path(self, path: str) -> dict[str, Any]
    def normalize_repository_url(self, url: str) -> dict[str, Any]
    def normalize_web_url(self, url: str) -> dict[str, Any]
    def validate_target_accessibility(self, target: str, crawl_type: str) -> bool
```

**Detection Logic:**
1. **Local Path Detection** (highest priority):
   - Starts with `/`, `./`, `../`, `~`
   - Path exists on filesystem
   - Returns normalized absolute path

2. **Git Repository Detection**:
   - Known git hosts (github.com, gitlab.com, etc.)
   - Ends with `.git`
   - Starts with `git@`, `git://`, `ssh://git@`
   - Returns repository URL

3. **Web URL Detection** (default fallback):
   - Any other input treated as web URL
   - Returns normalized URL

**Enhanced Features:**
- Input validation and sanitization
- Error handling for inaccessible targets
- Support for additional git hosting services
- URL normalization and validation

#### 2. `processing.py` (~150 lines)
**Responsibility**: RAG processing coordination and result formatting

**Extracted Functions:**
- `_process_rag_if_requested()` (lines 57-107)
- `_process_crawl_result_unified()` (lines 110-232)

**New Functionality:**
```python
class CrawlResultProcessor:
    """Handles RAG processing and result formatting for all crawl types."""

    async def process_rag_if_requested(
        self,
        ctx: Context,
        crawl_result: CrawlResult,
        source_type: SourceType,
        process_with_rag: bool,
        deduplication: bool | None = None,
        force_update: bool = False,
    ) -> dict[str, Any]

    def process_crawl_result_unified(
        self,
        crawl_result: CrawlResult,
        crawl_type: str,
        original_target: str,
    ) -> dict[str, Any]

    def format_website_result(self, crawl_result: CrawlResult) -> dict[str, Any]
    def format_repository_result(self, crawl_result: CrawlResult, repo_url: str) -> dict[str, Any]
    def format_directory_result(self, crawl_result: CrawlResult, directory_path: str) -> dict[str, Any]

    async def register_source_automatically(
        self, crawl_result: CrawlResult, source_type: SourceType
    ) -> int

    def calculate_advanced_statistics(self, crawl_result: CrawlResult) -> dict[str, Any]
```

**RAG Processing Features:**
- Automatic source registration through RAG processing
- Deduplication support with configurable options
- Force update functionality for content refresh
- Error handling and graceful degradation
- Progress reporting for long-running operations

**Result Formatting:**
- **Website Results**: Pages crawled, success rate, crawl statistics, advanced features info
- **Repository Results**: Files processed, repository URL, adaptive features, file type analysis
- **Directory Results**: Files processed, directory path, intelligent features, content analysis
- **Unified Statistics**: Processing time, success rates, error summaries

#### 3. `tools.py` (~306 lines)
**Responsibility**: FastMCP tool implementations

**Extracted Functions:**
- `register_crawling_tools()` (lines 235-556)
- `scrape()` tool implementation
- `crawl()` tool implementation

**New Functionality:**
```python
class CrawlingTools:
    """FastMCP tool implementations for web crawling operations."""

    def __init__(self):
        self.detector = CrawlTypeDetector()
        self.processor = CrawlResultProcessor()

    async def scrape_tool(
        self,
        ctx: Context,
        url: str,
        extraction_strategy: str | None = None,
        wait_for: str | None = None,
        include_raw_html: bool = False,
        process_with_rag: bool = True,
        enable_virtual_scroll: bool | None = None,
        virtual_scroll_count: int | None = None,
        deduplication: bool | None = None,
        force_update: bool = False,
    ) -> dict[str, Any]

    async def crawl_tool(
        self,
        ctx: Context,
        target: str,
        process_with_rag: bool = True,
        deduplication: bool | None = None,
        force_update: bool = False,
        # Web-specific parameters
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        sitemap_first: bool = True,
        # File-specific parameters
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        # Repo-specific parameters
        clone_path: str | None = None,
    ) -> dict[str, Any]

    def register_tools(self, mcp: FastMCP) -> None

def register_crawling_tools(mcp: FastMCP) -> None:
    """Register all crawling tools with the FastMCP server."""
    tools = CrawlingTools()
    tools.register_tools(mcp)
```

**Enhanced Tool Features:**
- **Scrape Tool**:
  - Advanced Crawl4AI features (virtual scroll, extraction strategies)
  - Comprehensive result formatting with metadata
  - RAG processing integration
  - Progress tracking and error handling

- **Crawl Tool**:
  - Automatic input type detection
  - Unified interface for all crawl types
  - Type-specific parameter handling
  - Intelligent progress reporting
  - Comprehensive result statistics

#### 4. `__init__.py` (~50 lines)
**Responsibility**: Package initialization and public API

```python
"""Crawling tools modules for FastMCP integration."""

from .detection import CrawlTypeDetector
from .processing import CrawlResultProcessor
from .tools import CrawlingTools, register_crawling_tools

class CrawlingService:
    """Unified crawling service with automatic type detection."""

    def __init__(self):
        self.detector = CrawlTypeDetector()
        self.processor = CrawlResultProcessor()
        self.tools = CrawlingTools()

    async def auto_crawl(
        self,
        target: str,
        process_with_rag: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """High-level auto-detecting crawl interface."""
        crawl_type, params = self.detector.detect_crawl_type_and_params(target)
        # Implementation would route to appropriate crawler

    def get_supported_crawl_types(self) -> list[str]:
        """Get list of supported crawl types."""
        return ["directory", "repository", "website"]

__all__ = [
    "CrawlTypeDetector",
    "CrawlResultProcessor",
    "CrawlingTools",
    "CrawlingService",
    "register_crawling_tools"
]
```

## Migration Strategy

### Phase 1: Create Modular Structure (No Breaking Changes)
1. Create `crawler_mcp/tools/crawling/` directory
2. Extract code into new modules without changing original file
3. Ensure `register_crawling_tools()` function uses new modular components
4. Maintain complete backward compatibility

### Phase 2: Enhanced Functionality
1. Add enhanced detection logic for edge cases
2. Improve result formatting with additional metadata
3. Add comprehensive error handling and recovery
4. Enhance progress reporting granularity

### Phase 3: Integration Testing
1. Test all tool combinations with FastMCP
2. Validate auto-detection accuracy across various inputs
3. Performance benchmarks for tool execution times
4. Error handling validation

### Phase 4: Gradual Migration (Optional)
1. Add feature flag: `use_modular_crawling_tools = Field(default=False)`
2. Update original `crawling.py` to conditionally use new modules:
   ```python
   # At top of crawling.py
   from ..config import settings
   if settings.use_modular_crawling_tools:
       from .crawling.tools import register_crawling_tools as modular_register
       register_crawling_tools = modular_register
   ```

## Benefits

### Code Organization
- **Clear Separation**: Detection, processing, and tool implementation are isolated
- **Reusable Components**: Detection and processing can be used outside of tools
- **Testable Units**: Each component can be thoroughly unit tested
- **Single Responsibility**: Each module has one clear purpose

### Enhanced Functionality
- **Better Detection**: More sophisticated input type detection
- **Improved Processing**: Enhanced RAG integration and result formatting
- **Advanced Tools**: More configurable and feature-rich tool implementations
- **Error Resilience**: Better error handling and recovery mechanisms

### Developer Experience
- **Easier Debugging**: Issues can be traced to specific modules
- **Clear APIs**: Well-defined interfaces between components
- **Better Documentation**: Each module can have focused documentation
- **Simplified Testing**: Isolated functionality is easier to test

## Dependencies and Imports

### Shared Dependencies
All modules will import:
```python
import logging
from typing import Any
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from ...config import settings
from ...core import CrawlerService, RagService
from ...models.crawl import CrawlRequest, CrawlResult, CrawlStatus
from ...models.sources import SourceType
```

### Inter-Module Dependencies
- `tools.py` depends on both `detection.py` and `processing.py`
- `detection.py` and `processing.py` are independent of each other
- All modules can be used independently for specialized use cases

## Testing Strategy

### Unit Tests
Each module will have comprehensive unit tests:

**test_detection.py**:
- Input type detection accuracy
- Parameter normalization validation
- Edge case handling (malformed URLs, invalid paths)
- Error handling for inaccessible targets

**test_processing.py**:
- RAG processing pipeline validation
- Result formatting accuracy for all crawl types
- Error handling during RAG operations
- Statistics calculation correctness

**test_tools.py**:
- Tool parameter validation
- Progress reporting functionality
- Error propagation and handling
- Integration with FastMCP framework

### Integration Tests
- End-to-end tool execution workflows
- Cross-module interaction validation
- Performance comparison with original implementation
- FastMCP integration testing

### Tool-Specific Tests
- **Scrape Tool**: Single page extraction, various strategies, error handling
- **Crawl Tool**: Multi-type input handling, parameter routing, result consistency

## Risk Mitigation

### Rollback Strategy
- Original `crawling.py` remains completely unchanged
- Can instantly disable modular implementation with feature flag
- No breaking changes to existing tool APIs

### Error Handling
- All modules maintain same error patterns as original
- Tool errors are properly propagated to FastMCP
- Comprehensive logging maintained throughout

### Performance Validation
- Benchmark tool execution times
- Validate auto-detection performance
- Memory usage profiling
- FastMCP integration performance

## Configuration Enhancement

### New Configuration Options
```python
# In config.py
class Settings:
    # Crawling tools settings
    crawling_enable_auto_detection: bool = Field(default=True, description="Enable automatic crawl type detection")
    crawling_validation_timeout: int = Field(default=10, description="Timeout for target validation in seconds")
    crawling_max_result_size: int = Field(default=1000, description="Maximum number of items in result samples")
    crawling_enable_enhanced_stats: bool = Field(default=True, description="Enable detailed statistics collection")
    crawling_rag_batch_size: int = Field(default=50, description="Batch size for RAG processing")
```

## Future Enhancements

### Additional Detection Types
- **Cloud Storage**: S3 buckets, Google Drive, OneDrive
- **Database URLs**: PostgreSQL, MongoDB connection strings
- **Archive Files**: ZIP, TAR archives for extraction
- **API Endpoints**: REST API discovery and crawling

### Enhanced Processing
- **Content Analysis**: Language detection, content categorization
- **Quality Metrics**: Content quality scoring, relevance assessment
- **Batch Operations**: Multi-target crawling with unified results
- **Streaming Results**: Real-time result streaming for large operations

This refactoring transforms a single 556-line file into 3 focused modules of ~100-300 lines each, significantly improving maintainability while adding enhanced functionality and better separation of concerns.
