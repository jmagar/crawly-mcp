# Vector Service Refactoring Plan

## Overview
Refactor `crawler_mcp/core/vectors.py` (805 lines) into smaller, focused modules while keeping the original file intact for rollback safety.

## Current File Analysis
The current `vectors.py` file handles multiple responsibilities:
- Collection management and configuration
- Document CRUD operations (upsert, retrieve, delete)
- Vector similarity search with filtering
- Statistics and analytics operations
- Client connection management

## Proposed Module Structure

### Directory: `crawler_mcp/core/vectors/`

#### 1. `collections.py` (~200 lines)
**Responsibility**: Qdrant collection lifecycle management

**Extracted Methods from VectorService:**
- `ensure_collection()` (lines 88-166)
- `get_collection_info()` (lines 168-202)
- `_recreate_client()` (lines 67-74)
- Collection configuration logic

**New Functionality:**
```python
class CollectionManager:
    """Manages Qdrant collection lifecycle and configuration."""

    async def ensure_collection_exists(self) -> bool
    async def get_collection_info(self) -> dict[str, Any]
    async def create_collection_with_config(self) -> bool
    async def delete_collection(self) -> bool
    async def recreate_client(self) -> None
```

**Key Features:**
- HNSW configuration management
- Optimized collection parameters
- Client recreation on connection issues
- Collection health monitoring

#### 2. `operations.py` (~250 lines)
**Responsibility**: Document CRUD operations

**Extracted Methods from VectorService:**
- `upsert_documents()` (lines 204-289)
- `get_document_by_id()` (lines 416-471)
- `delete_documents_by_source()` (lines 473-509)
- `get_chunks_by_source()` (lines 511-556)
- `delete_chunks_by_ids()` (lines 558-589)

**New Functionality:**
```python
class DocumentOperations:
    """Handles all document CRUD operations."""

    async def upsert_documents(self, documents: list[DocumentChunk], batch_size: int = 100) -> int
    async def get_document_by_id(self, document_id: str) -> DocumentChunk | None
    async def delete_documents_by_source(self, source_url: str) -> int
    async def get_chunks_by_source(self, source_url: str) -> list[dict[str, Any]]
    async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int
    async def bulk_update_documents(self, operations: list[dict]) -> int
```

**Key Features:**
- Batch processing optimization
- Error handling with partial failures
- Client recreation on connection errors
- Progress tracking for large operations

#### 3. `search.py` (~200 lines)
**Responsibility**: Vector similarity search and filtering

**Extracted Methods from VectorService:**
- `search_similar()` (lines 291-414)
- Query optimization logic
- Filter construction

**New Functionality:**
```python
class SearchEngine:
    """Handles vector similarity search operations."""

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[SearchMatch]

    async def search_with_reranking(self, query_vector: list[float], **kwargs) -> list[SearchMatch]
    async def build_search_filters(self, **filter_params) -> Filter | None
    async def optimize_search_params(self, query_complexity: str) -> SearchParams
```

**Key Features:**
- Dynamic ef parameter optimization
- Advanced filtering capabilities
- Query result reranking
- Search performance optimization

#### 4. `statistics.py` (~155 lines)
**Responsibility**: Analytics and source statistics

**Extracted Methods from VectorService:**
- `get_sources_stats()` (lines 591-659)
- `get_unique_sources()` (lines 661-805)

**New Functionality:**
```python
class StatisticsCollector:
    """Collects and analyzes vector database statistics."""

    async def get_sources_stats(self) -> dict[str, Any]
    async def get_unique_sources(
        self,
        domains: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]

    async def get_collection_health(self) -> dict[str, Any]
    async def analyze_content_distribution(self) -> dict[str, Any]
    async def get_embedding_quality_metrics(self) -> dict[str, Any]
```

**Key Features:**
- Paginated source listing
- Advanced filtering by domain and search terms
- Content distribution analysis
- Performance metrics collection

#### 5. `__init__.py` (~50 lines)
**Responsibility**: Package initialization and public API

```python
"""Vector service modules for Qdrant operations."""

from .collections import CollectionManager
from .operations import DocumentOperations
from .search import SearchEngine
from .statistics import StatisticsCollector

class VectorService:
    """Unified vector service using modular components."""

    def __init__(self):
        self.collections = CollectionManager()
        self.operations = DocumentOperations()
        self.search = SearchEngine()
        self.statistics = StatisticsCollector()

    # Backward compatibility methods that delegate to modules
    async def ensure_collection(self) -> bool:
        return await self.collections.ensure_collection_exists()

    async def upsert_documents(self, documents, batch_size=100) -> int:
        return await self.operations.upsert_documents(documents, batch_size)

    # ... other compatibility methods

__all__ = [
    "VectorService",
    "CollectionManager",
    "DocumentOperations",
    "SearchEngine",
    "StatisticsCollector"
]
```

## Migration Strategy

### Phase 1: Create Modular Structure (No Breaking Changes)
1. Create `crawler_mcp/core/vectors/` directory
2. Extract code into new modules without changing original file
3. Add comprehensive tests for each module
4. Ensure 100% backward compatibility

### Phase 2: Integration Testing
1. Create integration tests that use both old and new implementations
2. Performance benchmarks to ensure no regression
3. Test all edge cases and error conditions

### Phase 3: Gradual Migration (Optional)
1. Add feature flag in config: `use_modular_vectors = Field(default=False)`
2. Update original `vectors.py` to conditionally use new modules:
   ```python
   # At top of vectors.py
   from ..config import settings
   if settings.use_modular_vectors:
       from .vectors import VectorService as ModularVectorService
       VectorService = ModularVectorService
   ```

## Benefits

### Code Organization
- **Single Responsibility**: Each module has one clear purpose
- **Easier Testing**: Isolated functionality can be unit tested independently
- **Better Maintainability**: Changes to search logic don't affect collection management
- **Clearer Dependencies**: Import only what's needed

### Performance Benefits
- **Lazy Loading**: Only load modules that are actually used
- **Memory Efficiency**: Smaller modules have lower memory footprint
- **Parallel Development**: Multiple developers can work on different modules

### Developer Experience
- **Easier Navigation**: Find relevant code faster
- **Clearer APIs**: Module interfaces are more focused
- **Better Documentation**: Each module can have detailed docs
- **Simplified Debugging**: Smaller scope for investigating issues

## Dependencies and Imports

### Shared Dependencies
All modules will import:
```python
import logging
from typing import Any
from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import *
from ...config import settings
from ...models.rag import DocumentChunk, SearchMatch
```

### Inter-Module Dependencies
- `operations.py` depends on `collections.py` for collection existence
- `search.py` depends on `collections.py` for collection validation
- `statistics.py` depends on `collections.py` for collection info
- All modules share the same Qdrant client instance

## Testing Strategy

### Unit Tests
Each module will have comprehensive unit tests:
- `test_collections.py`: Collection lifecycle, configuration, error handling
- `test_operations.py`: CRUD operations, batch processing, error recovery
- `test_search.py`: Search accuracy, filtering, performance
- `test_statistics.py`: Analytics accuracy, pagination, filtering

### Integration Tests
- Cross-module interactions
- End-to-end workflows
- Performance comparisons with original implementation

### Backward Compatibility Tests
- Ensure all existing code continues to work unchanged
- API compatibility validation
- Error message consistency

## Risk Mitigation

### Rollback Strategy
- Original `vectors.py` remains completely unchanged
- Can instantly rollback by not using modular implementation
- Feature flag allows gradual rollout and quick disable

### Error Handling
- All modules maintain same error handling patterns as original
- Client recreation logic preserved in all modules
- Comprehensive logging maintained

### Performance Validation
- Benchmark all operations before and after refactoring
- Memory usage profiling
- Connection pooling efficiency testing

This refactoring transforms a single 805-line file into 4 focused modules of ~150-250 lines each, significantly improving maintainability while preserving all existing functionality.
