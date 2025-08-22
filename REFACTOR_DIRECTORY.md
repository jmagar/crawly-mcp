# Directory Crawler Refactoring Plan

## Overview
Refactor `crawler_mcp/crawlers/directory.py` (506 lines) into smaller, focused modules while keeping the original file intact for rollback safety.

## Current File Analysis
The current `directory.py` file handles multiple responsibilities:
- File discovery with pattern matching
- File validation and filtering
- Intelligent file scoring and prioritization
- High-performance concurrent file processing
- Directory crawl orchestration

## Proposed Module Structure

### Directory: `crawler_mcp/crawlers/directory/`

#### 1. `discovery.py` (~150 lines)
**Responsibility**: File discovery, pattern matching, and validation

**Extracted Methods from DirectoryCrawlStrategy:**
- `_discover_files()` (lines 196-232)
- `_is_valid_file()` (lines 234-299)
- File pattern matching logic

**New Functionality:**
```python
class FileDiscovery:
    """Handles file discovery and validation for directory crawling."""

    async def discover_files(self, directory: Path, request: DirectoryRequest) -> list[Path]
    def is_valid_file(self, file_path: Path) -> bool
    def filter_by_patterns(self, files: list[Path], patterns: list[str]) -> list[Path]
    def apply_size_limits(self, files: list[Path], max_size_mb: int = 10) -> list[Path]
    def check_file_readability(self, file_path: Path) -> bool
    def get_binary_extensions(self) -> set[str]
```

**Key Features:**
- Recursive and non-recursive pattern matching
- Binary file detection and filtering
- Size-based filtering (configurable limits)
- Readability validation with encoding detection
- Duplicate removal and path normalization

**File Validation Logic:**
- Skip empty files and directories
- Filter out binary files by extension
- Size limits to prevent memory issues
- Basic readability test with UTF-8 fallback

#### 2. `scoring.py` (~150 lines)
**Responsibility**: File relevance scoring and intelligent prioritization

**Extracted Methods from DirectoryCrawlStrategy:**
- `_score_files()` (lines 300-408)
- File prioritization algorithms

**New Functionality:**
```python
class FileScorer:
    """Intelligent file scoring and prioritization system."""

    async def score_files(
        self, files: list[Path], base_directory: Path
    ) -> list[tuple[Path, float]]

    def calculate_extension_score(self, file_path: Path) -> float
    def calculate_size_score(self, file_path: Path) -> float
    def calculate_depth_score(self, file_path: Path, base_directory: Path) -> float
    def calculate_name_pattern_score(self, file_path: Path) -> float
    def calculate_modification_time_score(self, file_path: Path) -> float
    def get_extension_priorities(self) -> dict[str, int]
    def get_important_name_patterns(self) -> dict[str, int]
```

**Scoring Criteria:**
1. **Extension Score** (0-100 points):
   - Python, JavaScript, TypeScript: 90-100 points
   - Java, C++, Go, Rust: 80-90 points
   - HTML, CSS, Markdown: 60-70 points
   - Config files: 50-60 points
   - Logs, temporary files: 10-20 points

2. **Size Score** (0-20 points):
   - 1KB-100KB: 20 points (ideal size)
   - 100KB-500KB: 10 points (good size)
   - >1MB: -10 points (less preferred)

3. **Depth Score** (0-10 points):
   - Root level: 10 points
   - Each level deeper: -2 points

4. **Name Pattern Score** (-10 to +15 points):
   - readme, main, index, app: +15 points
   - config, settings, env: +10 points
   - test, spec, example: +5 points
   - temp, tmp, cache, backup: -10 points

5. **Modification Time Score** (0-5 points):
   - Modified within 7 days: +5 points
   - Modified within 30 days: +2 points

#### 3. `crawler.py` (~206 lines)
**Responsibility**: Main directory crawling orchestration and file processing

**Extracted Methods from DirectoryCrawlStrategy:**
- `__init__()`, `validate_request()`, `execute()` (lines 24-194)
- `_process_files_highly_concurrent()` (lines 410-448)
- `_process_single_file_sync()` and `_process_single_file_sync_impl()` (lines 450-506)

**New Functionality:**
```python
class DirectoryCrawlStrategy(BaseCrawlStrategy):
    """Main directory crawling strategy with intelligent processing."""

    def __init__(self) -> None
    async def validate_request(self, request: DirectoryRequest) -> bool
    async def execute(
        self,
        request: DirectoryRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult

    # High-performance processing
    async def process_files_concurrent(
        self, file_paths: list[Path], base_directory: Path
    ) -> list[PageContent | Exception]

    def process_single_file_sync(
        self, file_path: Path, base_directory: Path
    ) -> PageContent | Exception

    # Integration with discovery and scoring
    async def discover_and_score_files(
        self, directory: Path, request: DirectoryRequest
    ) -> list[Path]
```

**Key Features:**
- Integration with FileDiscovery and FileScorer
- High-performance concurrent processing with ThreadPoolExecutor
- Memory pressure monitoring during processing
- Comprehensive error handling and statistics
- Progress reporting throughout the crawl

**Concurrent Processing:**
- Uses configured thread count (default 16 for i7-13700K)
- ThreadPoolExecutor for maximum CPU utilization
- asyncio.gather for true parallelism
- Exception handling for individual file failures

#### 4. `__init__.py` (~50 lines)
**Responsibility**: Package initialization and public API

```python
"""Directory crawler modules for intelligent file processing."""

from .discovery import FileDiscovery
from .scoring import FileScorer
from .crawler import DirectoryCrawlStrategy, DirectoryRequest

class DirectoryService:
    """Unified directory crawling service using modular components."""

    def __init__(self):
        self.discovery = FileDiscovery()
        self.scorer = FileScorer()
        self.crawler = DirectoryCrawlStrategy()

    async def crawl_directory(
        self,
        directory_path: str,
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        max_files: int = 1000,
        progress_callback=None,
    ) -> CrawlResult:
        """High-level directory crawling interface."""
        request = DirectoryRequest(
            directory_path=directory_path,
            file_patterns=file_patterns,
            recursive=recursive,
            max_files=max_files,
        )
        return await self.crawler.execute(request, progress_callback)

__all__ = [
    "DirectoryCrawlStrategy",
    "DirectoryRequest",
    "DirectoryService",
    "FileDiscovery",
    "FileScorer"
]
```

## Migration Strategy

### Phase 1: Create Modular Structure (No Breaking Changes)
1. Create `crawler_mcp/crawlers/directory/` directory
2. Extract code into new modules without changing original file
3. Ensure DirectoryCrawlStrategy uses the new modular components internally
4. Add comprehensive tests for each module

### Phase 2: Integration and Testing
1. Create integration tests for the complete directory crawling pipeline
2. Performance benchmarks comparing old vs new implementation
3. Validate all edge cases and error conditions
4. Memory usage profiling during concurrent processing

### Phase 3: Gradual Migration (Optional)
1. Add feature flag: `use_modular_directory_crawler = Field(default=False)`
2. Update original `directory.py` to conditionally use new modules:
   ```python
   # At top of directory.py
   from ..config import settings
   if settings.use_modular_directory_crawler:
       from .directory import DirectoryCrawlStrategy as ModularDirectoryCrawlStrategy
       DirectoryCrawlStrategy = ModularDirectoryCrawlStrategy
   ```

## Benefits

### Code Organization
- **Separation of Concerns**: Discovery, scoring, and processing are clearly separated
- **Testability**: Each component can be unit tested independently
- **Configurability**: Scoring algorithms can be easily modified or extended
- **Reusability**: Discovery and scoring modules can be used elsewhere

### Performance Benefits
- **Optimized Discovery**: File discovery can be optimized independently
- **Tunable Scoring**: Scoring algorithms can be fine-tuned without affecting processing
- **Isolated Processing**: File processing optimization doesn't impact discovery logic
- **Memory Efficiency**: Smaller modules with focused responsibilities

### Developer Experience
- **Easier Debugging**: Issues can be isolated to specific modules
- **Clear Interfaces**: Well-defined APIs between components
- **Extensible Scoring**: Easy to add new scoring criteria
- **Better Documentation**: Each module can have focused documentation

## Dependencies and Imports

### Shared Dependencies
All modules will import:
```python
import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ...models.crawl import CrawlResult, CrawlStatistics, CrawlStatus, PageContent
from ..base import BaseCrawlStrategy
```

### Inter-Module Dependencies
- `crawler.py` depends on both `discovery.py` and `scoring.py`
- `discovery.py` and `scoring.py` are independent of each other
- All modules can be used independently for specialized use cases

## Testing Strategy

### Unit Tests
Each module will have comprehensive unit tests:

**test_discovery.py**:
- File pattern matching accuracy
- Binary file detection
- Size filtering
- Readability validation
- Error handling for permissions/access issues

**test_scoring.py**:
- Scoring algorithm accuracy
- Extension priority verification
- Size scoring validation
- Depth calculation correctness
- Name pattern matching

**test_crawler.py**:
- Complete crawl pipeline
- Concurrent processing performance
- Memory pressure handling
- Progress reporting
- Error aggregation and statistics

### Integration Tests
- End-to-end directory crawling workflows
- Performance comparison with original implementation
- Memory usage under various loads
- Large directory handling (1000+ files)

### Performance Tests
- Threading efficiency with different worker counts
- Memory usage during concurrent processing
- File discovery speed on large directories
- Scoring performance with various file sets

## Risk Mitigation

### Rollback Strategy
- Original `directory.py` remains completely unchanged
- Can instantly disable modular implementation with feature flag
- No breaking changes to existing APIs

### Error Handling
- All modules maintain same error handling patterns as original
- File processing errors are isolated and don't affect other files
- Comprehensive logging maintained throughout

### Performance Validation
- Benchmark file discovery performance
- Validate concurrent processing efficiency
- Memory usage profiling
- Ensure no regression in crawl speed

## Configuration Enhancement

### New Configuration Options
```python
# In config.py
class Settings:
    # Directory crawler settings
    directory_max_file_size_mb: int = Field(default=10, description="Max file size in MB")
    directory_processing_threads: int = Field(default=16, description="Thread count for file processing")
    directory_enable_scoring: bool = Field(default=True, description="Enable intelligent file scoring")
    directory_score_weights: dict = Field(
        default={
            "extension": 0.4,
            "size": 0.2,
            "depth": 0.1,
            "name_pattern": 0.2,
            "modification_time": 0.1
        },
        description="Scoring weight distribution"
    )
```

This refactoring transforms a single 506-line file into 3 focused modules of ~150-200 lines each, significantly improving maintainability while enhancing the intelligent file processing capabilities.
