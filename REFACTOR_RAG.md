# RAG Service Refactoring Plan

## Overview
Refactor `crawler_mcp/core/rag.py` (1560 lines) into smaller, focused modules while keeping the original file intact for rollback safety.

## Current File Analysis
The current `rag.py` file handles multiple responsibilities:
- Text chunking with token counting and optimization
- Embedding generation with parallel processing pipelines
- Content deduplication with hash management
- Main processing pipeline coordination
- Service orchestration and context management
- Health monitoring and statistics

## Proposed Module Structure

### Directory: `crawler_mcp/core/rag/`

#### 1. `chunking.py` (~400 lines)
**Responsibility**: Text chunking strategies, token counting, and chunk optimization

**Extracted Code:**
- `ChunkingStrategy` class and implementations
- `_chunk_content()` method (lines 200-350)
- `_calculate_accurate_token_count()` method (lines 180-200)
- Token counting logic and optimization

**New Functionality:**
```python
class ChunkingStrategy:
    """Base class for text chunking strategies."""

    def chunk_text(self, text: str, metadata: dict) -> list[dict[str, Any]]
    def calculate_token_count(self, text: str) -> int
    def validate_chunk_size(self, chunk: str) -> bool
    def optimize_chunk_boundaries(self, chunks: list[str]) -> list[str]

class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: dict) -> list[dict[str, Any]]

class SemanticChunker(ChunkingStrategy):
    """Semantic boundary-aware chunking."""

    def chunk_text(self, text: str, metadata: dict) -> list[dict[str, Any]]
    def find_semantic_boundaries(self, text: str) -> list[int]
    def split_on_sentences(self, text: str) -> list[str]
    def split_on_paragraphs(self, text: str) -> list[str]

class AdaptiveChunker(ChunkingStrategy):
    """Adaptive chunking based on content type and structure."""

    def chunk_text(self, text: str, metadata: dict) -> list[dict[str, Any]]
    def detect_content_structure(self, text: str) -> str
    def chunk_code_content(self, text: str) -> list[str]
    def chunk_markdown_content(self, text: str) -> list[str]
    def chunk_plain_text(self, text: str) -> list[str]

class TokenCounter:
    """Accurate token counting with multiple tokenizer support."""

    def __init__(self):
        self.tokenizer = self._load_tokenizer()
        self.word_to_token_ratio = 1.3

    def count_tokens(self, text: str) -> int
    def estimate_tokens_from_words(self, text: str) -> int
    def _load_tokenizer(self) -> Any
    def calculate_chunk_token_distribution(self, chunks: list[str]) -> dict[str, float]
```

**Enhanced Chunking Features:**
- **Multiple Strategies**: Fixed-size, semantic, and adaptive chunking
- **Content-Aware**: Different strategies for code, markdown, and plain text
- **Token Optimization**: Accurate token counting with fallback estimation
- **Boundary Detection**: Smart boundary detection for better chunk quality
- **Overlap Management**: Configurable overlap for context preservation

**Chunk Quality Metrics:**
```python
def assess_chunk_quality(self, chunk: str, metadata: dict) -> dict[str, float]:
    """Assess chunk quality across multiple dimensions."""
    return {
        "completeness": self._assess_completeness(chunk),
        "coherence": self._assess_coherence(chunk),
        "information_density": self._assess_information_density(chunk),
        "context_preservation": self._assess_context_preservation(chunk, metadata),
    }
```

#### 2. `embedding.py` (~300 lines)
**Responsibility**: Embedding generation, batch processing, and parallel pipelines

**Extracted Code:**
- Embedding service management
- Parallel pipeline implementation (lines 400-600)
- Batch processing optimization
- Worker pool management

**New Functionality:**
```python
class EmbeddingPipeline:
    """High-performance embedding generation pipeline."""

    def __init__(self):
        self.embedding_service = self._initialize_embedding_service()
        self.worker_pool = None
        self.batch_size = settings.embedding_batch_size
        self.max_workers = settings.embedding_workers

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]

    async def process_chunks_parallel(
        self, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]

    async def start_pipeline(self) -> None
    async def stop_pipeline(self) -> None

    def create_worker_pool(self, max_workers: int) -> None
    async def process_batch(self, batch: list[str]) -> list[list[float]]
    def optimize_batch_size(self, total_items: int) -> int

    async def health_check(self) -> dict[str, Any]

class EmbeddingWorker:
    """Individual embedding worker for parallel processing."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.embedding_service = None
        self.processed_count = 0
        self.error_count = 0

    async def initialize(self) -> None
    async def process_batch(self, texts: list[str]) -> list[list[float]]
    async def shutdown(self) -> None
    def get_statistics(self) -> dict[str, int]

class EmbeddingCache:
    """Caching layer for embedding operations."""

    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cached_embedding(self, text_hash: str) -> list[float] | None
    def cache_embedding(self, text_hash: str, embedding: list[float]) -> None
    def clear_cache(self) -> None
    def get_cache_statistics(self) -> dict[str, int]
```

**Parallel Processing Features:**
- **Worker Pool Management**: Dynamic worker scaling based on load
- **Batch Optimization**: Intelligent batch sizing for optimal throughput
- **Error Recovery**: Robust error handling with retry logic
- **Performance Monitoring**: Real-time metrics and health monitoring
- **Caching Layer**: Embedding caching to avoid recomputation

**Pipeline Configuration:**
```python
# Optimized for i7-13700K (16 cores, 24 threads)
embedding_workers = 12  # Leave headroom for other processes
embedding_batch_size = 32  # Optimal batch size for Qwen3-Embedding-0.6B
embedding_cache_size = 10000  # Cache frequent embeddings
```

#### 3. `deduplication.py` (~250 lines)
**Responsibility**: Content deduplication, hash management, and orphan detection

**Extracted Code:**
- Content deduplication logic (lines 600-800)
- Hash generation and comparison
- Orphan chunk detection and cleanup

**New Functionality:**
```python
class DeduplicationManager:
    """Manages content deduplication across the RAG system."""

    def __init__(self):
        self.hash_cache = {}
        self.similarity_threshold = 0.95

    async def deduplicate_chunks(
        self, chunks: list[DocumentChunk], existing_chunks: list[dict] | None = None
    ) -> tuple[list[DocumentChunk], list[DocumentChunk]]

    def generate_content_hash(self, content: str) -> str
    def calculate_content_similarity(self, content1: str, content2: str) -> float
    async def find_existing_chunks(self, source_url: str) -> list[dict[str, Any]]
    async def identify_orphaned_chunks(self, source_url: str) -> list[str]
    async def cleanup_orphaned_chunks(self, chunk_ids: list[str]) -> int

    def normalize_content_for_comparison(self, content: str) -> str
    def extract_content_fingerprint(self, content: str) -> dict[str, Any]
    async def bulk_deduplication(
        self, source_chunks: dict[str, list[DocumentChunk]]
    ) -> dict[str, tuple[list[DocumentChunk], list[DocumentChunk]]]

class ContentHasher:
    """Content hashing utilities for deduplication."""

    @staticmethod
    def hash_content(content: str) -> str:
        """Generate SHA-256 hash of normalized content."""

    @staticmethod
    def hash_chunk_metadata(chunk: DocumentChunk) -> str:
        """Generate hash including content and key metadata."""

    @staticmethod
    def normalize_whitespace(content: str) -> str:
        """Normalize whitespace for consistent hashing."""

    @staticmethod
    def extract_text_features(content: str) -> dict[str, Any]:
        """Extract features for similarity comparison."""

class SimilarityDetector:
    """Advanced similarity detection for near-duplicate content."""

    def __init__(self):
        self.min_similarity = 0.85

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float
    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float
    def detect_near_duplicates(
        self, new_chunks: list[DocumentChunk], existing_chunks: list[dict]
    ) -> list[tuple[DocumentChunk, dict, float]]
```

**Deduplication Strategy:**
1. **Content Hashing**: SHA-256 hashes of normalized content
2. **Similarity Detection**: Multiple algorithms for near-duplicate detection
3. **Metadata Consideration**: Include source, title, and timestamp in uniqueness
4. **Orphan Management**: Identify and clean up orphaned chunks
5. **Bulk Operations**: Efficient batch deduplication for large datasets

#### 4. `processing.py` (~350 lines)
**Responsibility**: Main processing pipeline coordination and workflow management

**Extracted Code:**
- `process_crawl_result()` main method (lines 800-1100)
- Pipeline orchestration logic
- Progress tracking and error handling

**New Functionality:**
```python
class ProcessingPipeline:
    """Main RAG processing pipeline coordinator."""

    def __init__(self):
        self.chunker = AdaptiveChunker()
        self.embedding_pipeline = EmbeddingPipeline()
        self.deduplication_manager = DeduplicationManager()
        self.vector_service = None

    async def process_crawl_result(
        self,
        crawl_result: CrawlResult,
        deduplication: bool = True,
        force_update: bool = False,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]

    async def process_pages_batch(
        self, pages: list[PageContent], batch_size: int = 50
    ) -> tuple[list[DocumentChunk], dict[str, Any]]

    async def process_single_page(self, page: PageContent) -> list[DocumentChunk]

    async def store_chunks_batch(
        self, chunks: list[DocumentChunk], batch_size: int = 100
    ) -> int

    def calculate_processing_statistics(
        self,
        total_pages: int,
        total_chunks: int,
        processing_time: float,
        **kwargs
    ) -> dict[str, Any]

    async def validate_processing_result(
        self, result: dict[str, Any]
    ) -> tuple[bool, list[str]]

class WorkflowManager:
    """Manages complex RAG workflow execution."""

    async def execute_full_pipeline(
        self,
        crawl_result: CrawlResult,
        config: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]

    async def execute_incremental_update(
        self, source_url: str, new_pages: list[PageContent]
    ) -> dict[str, Any]

    async def execute_reprocessing(
        self, source_url: str, force_rechunk: bool = False
    ) -> dict[str, Any]

    def create_processing_plan(
        self, crawl_result: CrawlResult, config: dict[str, Any]
    ) -> dict[str, Any]

    async def monitor_workflow_health(self) -> dict[str, Any]

class ProgressTracker:
    """Advanced progress tracking for long-running operations."""

    def __init__(self):
        self.current_stage = "initialization"
        self.progress_data = {}

    def start_stage(self, stage_name: str, total_items: int) -> None
    def update_progress(self, completed: int, message: str = "") -> None
    def complete_stage(self, stage_name: str) -> None
    def get_overall_progress(self) -> dict[str, Any]
    async def report_progress(self, callback: Callable | None) -> None
```

**Processing Workflow:**
1. **Input Validation**: Validate crawl result and configuration
2. **Content Chunking**: Apply appropriate chunking strategy based on content type
3. **Deduplication**: Remove duplicate and near-duplicate content
4. **Embedding Generation**: Generate embeddings using parallel pipeline
5. **Vector Storage**: Store chunks in vector database with batch optimization
6. **Statistics Calculation**: Generate comprehensive processing statistics
7. **Quality Validation**: Validate processing result quality

#### 5. `service.py` (~260 lines)
**Responsibility**: Service orchestration, context management, and health monitoring

**Extracted Code:**
- `RagService` class main structure (lines 50-180)
- Context management and lifecycle
- Health checking and monitoring

**New Functionality:**
```python
class RagService:
    """Main RAG service with modular architecture."""

    def __init__(self):
        self.processing_pipeline = ProcessingPipeline()
        self.workflow_manager = WorkflowManager()
        self.vector_service = None
        self.embedding_pipeline = None
        self._context_count = 0
        self._initialized = False

    async def __aenter__(self) -> "RagService":
        """Async context manager entry."""
        await self.initialize()
        self._context_count += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._context_count -= 1
        if self._context_count <= 0:
            await self.shutdown()

    async def initialize(self) -> None:
        """Initialize all RAG service components."""
        if self._initialized:
            return

        # Initialize vector service
        self.vector_service = VectorService()
        await self.vector_service.ensure_collection()

        # Initialize embedding pipeline
        await self.processing_pipeline.embedding_pipeline.start_pipeline()

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown all RAG service components."""
        if not self._initialized:
            return

        # Shutdown embedding pipeline
        await self.processing_pipeline.embedding_pipeline.stop_pipeline()

        # Close vector service
        if self.vector_service:
            await self.vector_service.close()

        self._initialized = False

    async def process_crawl_result(
        self, crawl_result: CrawlResult, **kwargs
    ) -> dict[str, Any]:
        """Process crawl result through the complete RAG pipeline."""
        return await self.processing_pipeline.process_crawl_result(
            crawl_result, **kwargs
        )

    async def query(self, query: RagQuery) -> RagQueryResult:
        """Execute RAG query with reranking and filtering."""
        # Implementation delegates to appropriate services

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of all RAG components."""
        health_status = {
            "service": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check vector service
        if self.vector_service:
            vector_healthy = await self.vector_service.health_check()
            health_status["components"]["vector_service"] = (
                "healthy" if vector_healthy else "unhealthy"
            )

        # Check embedding pipeline
        embedding_health = await self.processing_pipeline.embedding_pipeline.health_check()
        health_status["components"]["embedding_pipeline"] = embedding_health

        # Check overall service health
        unhealthy_components = [
            k for k, v in health_status["components"].items()
            if v != "healthy" and not isinstance(v, dict)
        ]
        if unhealthy_components:
            health_status["service"] = "degraded"
            health_status["issues"] = unhealthy_components

        return health_status

class ServiceMetrics:
    """Service-level metrics and monitoring."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.processing_times = []

    def record_request(self, processing_time: float) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.processing_times.append(processing_time)

    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current service metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )

        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "average_processing_time": avg_processing_time,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
        }
```

#### 6. `__init__.py` (~50 lines)
**Responsibility**: Package initialization and unified API

```python
"""RAG service modules for intelligent content processing."""

from .chunking import ChunkingStrategy, FixedSizeChunker, SemanticChunker, AdaptiveChunker, TokenCounter
from .embedding import EmbeddingPipeline, EmbeddingWorker, EmbeddingCache
from .deduplication import DeduplicationManager, ContentHasher, SimilarityDetector
from .processing import ProcessingPipeline, WorkflowManager, ProgressTracker
from .service import RagService, ServiceMetrics

class RagSystem:
    """Unified RAG system with modular architecture."""

    def __init__(self):
        self.service = RagService()
        self.metrics = ServiceMetrics()

    async def process(self, crawl_result: CrawlResult, **kwargs) -> dict[str, Any]:
        """High-level processing interface."""
        start_time = time.time()
        try:
            result = await self.service.process_crawl_result(crawl_result, **kwargs)
            processing_time = time.time() - start_time
            self.metrics.record_request(processing_time)
            return result
        except Exception as e:
            self.metrics.record_error()
            raise

    async def query(self, query: RagQuery) -> RagQueryResult:
        """High-level query interface."""
        return await self.service.query(query)

    async def health_check(self) -> dict[str, Any]:
        """Complete system health check."""
        service_health = await self.service.health_check()
        service_metrics = self.metrics.get_metrics()

        return {
            **service_health,
            "metrics": service_metrics,
        }

__all__ = [
    "RagService",
    "RagSystem",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SemanticChunker",
    "AdaptiveChunker",
    "EmbeddingPipeline",
    "DeduplicationManager",
    "ProcessingPipeline",
    "WorkflowManager"
]
```

## Migration Strategy

### Phase 1: Create Modular Structure (No Breaking Changes)
1. Create `crawler_mcp/core/rag/` directory
2. Extract code into new modules without changing original file
3. Ensure RagService uses the new modular components internally
4. Add comprehensive tests for each module

### Phase 2: Enhanced Functionality
1. Implement advanced chunking strategies with content-aware logic
2. Optimize embedding pipeline with better parallel processing
3. Enhance deduplication with similarity detection
4. Add comprehensive monitoring and health checking

### Phase 3: Integration Testing
1. Test complete RAG pipeline with various content types
2. Performance benchmarks comparing old vs new implementation
3. Validate all chunking strategies and embedding quality
4. Memory usage profiling during large processing jobs

### Phase 4: Gradual Migration (Optional)
1. Add feature flag: `use_modular_rag_service = Field(default=False)`
2. Update original `rag.py` to conditionally use new modules:
   ```python
   # At top of rag.py
   from ..config import settings
   if settings.use_modular_rag_service:
       from .rag import RagService as ModularRagService
       RagService = ModularRagService
   ```

## Benefits

### Code Organization
- **Clear Separation**: Chunking, embedding, deduplication, and orchestration are isolated
- **Testable Components**: Each module can be thoroughly tested independently
- **Reusable Logic**: Chunking and embedding modules can be used elsewhere
- **Single Responsibility**: Each module has one focused purpose

### Performance Benefits
- **Optimized Chunking**: Content-aware chunking strategies for better quality
- **Parallel Embeddings**: Enhanced parallel processing with worker pools
- **Smart Deduplication**: Advanced deduplication with similarity detection
- **Pipeline Optimization**: Each stage can be optimized independently

### Developer Experience
- **Easier Debugging**: Issues can be isolated to specific modules
- **Clear Interfaces**: Well-defined APIs between components
- **Better Testing**: Isolated functionality allows for comprehensive unit tests
- **Modular Development**: Different developers can work on different modules

## Dependencies and Imports

### Shared Dependencies
All modules will import:
```python
import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ...config import settings
from ...models.crawl import CrawlResult, PageContent
from ...models.rag import DocumentChunk, RagQuery, RagQueryResult
from ...core.vectors import VectorService
from ...core.embeddings import EmbeddingService
```

### Inter-Module Dependencies
- `service.py` orchestrates all other modules
- `processing.py` depends on chunking, embedding, and deduplication modules
- `chunking.py`, `embedding.py`, and `deduplication.py` are largely independent
- All modules can be used independently for specialized use cases

## Testing Strategy

### Unit Tests
Each module will have comprehensive unit tests:

**test_chunking.py**:
- Chunking strategy accuracy across content types
- Token counting precision with different tokenizers
- Boundary detection and optimization
- Chunk quality assessment

**test_embedding.py**:
- Embedding generation accuracy and consistency
- Parallel processing performance and reliability
- Worker pool management and scaling
- Caching effectiveness and consistency

**test_deduplication.py**:
- Content deduplication accuracy
- Similarity detection precision and recall
- Hash generation consistency
- Orphan detection and cleanup

**test_processing.py**:
- Complete pipeline integration
- Progress tracking accuracy
- Error handling and recovery
- Statistics calculation correctness

**test_service.py**:
- Service lifecycle management
- Health monitoring accuracy
- Context management correctness
- Metrics collection and reporting

### Integration Tests
- End-to-end RAG processing workflows
- Performance comparison with original implementation
- Memory usage during large processing jobs
- Cross-module interaction validation

### Performance Tests
- Chunking speed across different content types
- Embedding generation throughput with parallel processing
- Deduplication performance with large datasets
- Overall pipeline performance and scalability

## Risk Mitigation

### Rollback Strategy
- Original `rag.py` remains completely unchanged
- Can instantly disable modular implementation with feature flag
- No breaking changes to existing APIs

### Error Handling
- All modules maintain same error handling patterns as original
- Pipeline errors are isolated and don't cascade
- Comprehensive logging maintained throughout

### Performance Validation
- Benchmark all operations before and after refactoring
- Validate embedding quality and consistency
- Memory usage profiling
- Processing speed and accuracy validation

## Configuration Enhancement

### New Configuration Options
```python
# In config.py
class Settings:
    # RAG service settings
    rag_chunking_strategy: str = Field(default="adaptive", description="Default chunking strategy")
    rag_chunk_size: int = Field(default=1000, description="Default chunk size for fixed chunking")
    rag_chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    rag_max_chunk_size: int = Field(default=2000, description="Maximum chunk size")
    rag_min_chunk_size: int = Field(default=100, description="Minimum chunk size")

    # Embedding settings
    rag_embedding_workers: int = Field(default=12, description="Number of embedding workers")
    rag_embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    rag_embedding_cache_size: int = Field(default=10000, description="Embedding cache size")

    # Deduplication settings
    rag_deduplication_enabled: bool = Field(default=True, description="Enable content deduplication")
    rag_similarity_threshold: float = Field(default=0.95, description="Similarity threshold for deduplication")
    rag_cleanup_orphans: bool = Field(default=True, description="Auto-cleanup orphaned chunks")

    # Processing settings
    rag_processing_batch_size: int = Field(default=50, description="Processing batch size")
    rag_max_processing_time: int = Field(default=3600, description="Maximum processing time in seconds")
    rag_enable_quality_checks: bool = Field(default=True, description="Enable chunk quality validation")
```

This refactoring transforms a single 1560-line file into 5 focused modules of ~250-400 lines each, significantly improving maintainability while enhancing the sophisticated RAG processing capabilities with advanced chunking, parallel embedding generation, and intelligent deduplication.
