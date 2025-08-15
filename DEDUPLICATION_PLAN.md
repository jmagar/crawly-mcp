# Duplicate Content Handling & Change Detection Plan for Crawlerr MCP

## Current State Analysis

### Problem Statement
When URLs are re-crawled, the system creates duplicate content in Qdrant because:
1. **Random UUID Generation**: Each chunk gets a new UUID via `uuid.uuid4()` (crawler_mcp/core/rag.py:436)
2. **No Duplicate Checking**: System doesn't check if content already exists before inserting
3. **No Automatic Cleanup**: Old content isn't deleted when re-crawling the same URL
4. **No Change Detection**: Can't determine if content has actually changed

### Current Architecture
```
Crawl Flow: URL → Crawler → CrawlResult → process_crawl_result → generate_embeddings → upsert to Qdrant
                                                    ↓
                                            Creates new UUID for each chunk
                                                    ↓
                                            Duplicates accumulate in Qdrant
```

### Existing Capabilities
- ✅ Delete mechanism exists: `delete_documents_by_source()` in vectors.py:443
- ✅ Source tracking exists: SourceService tracks URLs with hash-based IDs
- ✅ Qdrant supports upsert operations (updates if ID exists)
- ❌ No content hashing
- ❌ No deterministic ID generation
- ❌ Delete mechanism never called automatically

## Proposed Solution

### Core Strategy: Content-Based Deduplication with Change Detection

#### 1. Deterministic ID Generation
Replace random UUIDs with deterministic IDs based on URL and chunk position:
```python
# Instead of: chunk_id = f"{uuid.uuid4()}"
chunk_id = hashlib.sha256(f"{source_url}:{chunk_index}".encode()).hexdigest()[:16]
```

#### 2. Content Hashing
Add content fingerprinting to detect changes:
```python
content_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
```

#### 3. Smart Upsert Strategy
Before processing new crawl results:
1. Query existing documents by source_url
2. Build a map of existing chunk_id → content_hash
3. For each new chunk:
   - Calculate deterministic ID and content hash
   - Compare with existing chunks
   - Only upsert if content changed or chunk is new
4. Track which chunks should exist vs what exists
5. Delete orphaned chunks (existed before but not in new crawl)

## Implementation Plan

### Phase 1: Data Model Changes

#### 1.1 Update DocumentChunk Model (models/rag.py)
```python
class DocumentChunk:
    id: str  # Will become deterministic
    content_hash: str  # New field for change detection
    previous_hash: str | None  # Track content changes
    first_seen: datetime  # When first crawled
    last_modified: datetime  # When content changed
```

#### 1.2 Update SourceInfo Model (models/sources.py)
```python
class SourceInfo:
    content_hash: str  # Overall source content hash
    chunk_hashes: dict[str, str]  # Map of chunk_id → content_hash
    last_crawl_status: str  # "unchanged", "modified", "new"
```

### Phase 2: Core Logic Changes

#### 2.1 Modify process_crawl_result (core/rag.py:396)
```python
async def process_crawl_result(self, crawl_result, progress_callback=None):
    # Step 1: Check if source exists
    existing_chunks = await self.vector_service.get_chunks_by_source(page.url)
    existing_map = {chunk.id: chunk.content_hash for chunk in existing_chunks}

    # Step 2: Process new chunks with deterministic IDs
    new_chunks = []
    updated_chunks = []
    unchanged_chunks = []

    for i, page in enumerate(crawl_result.pages):
        chunk_id = self._generate_deterministic_id(page.url, i)
        content_hash = self._calculate_content_hash(page.content)

        if chunk_id in existing_map:
            if existing_map[chunk_id] != content_hash:
                updated_chunks.append(chunk)  # Content changed
            else:
                unchanged_chunks.append(chunk_id)  # No change
        else:
            new_chunks.append(chunk)  # New chunk

    # Step 3: Find orphaned chunks
    new_chunk_ids = {self._generate_deterministic_id(page.url, i)
                     for i, page in enumerate(crawl_result.pages)}
    orphaned_ids = set(existing_map.keys()) - new_chunk_ids

    # Step 4: Apply changes
    if new_chunks or updated_chunks:
        await self._upsert_chunks(new_chunks + updated_chunks)

    if orphaned_ids:
        await self.vector_service.delete_chunks_by_ids(orphaned_ids)
```

#### 2.2 Add Helper Methods (core/rag.py)
```python
def _generate_deterministic_id(self, url: str, chunk_index: int) -> str:
    """Generate deterministic ID from URL and chunk index."""
    normalized_url = self._normalize_url(url)
    return hashlib.sha256(
        f"{normalized_url}:{chunk_index}".encode()
    ).hexdigest()[:16]

def _calculate_content_hash(self, content: str) -> str:
    """Calculate SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _normalize_url(self, url: str) -> str:
    """Normalize URL for consistent hashing."""
    # Remove trailing slashes, normalize protocol, etc.
    parsed = urlparse(url)
    # ... normalization logic
    return normalized_url
```

### Phase 3: Vector Service Enhancements

#### 3.1 Add Query Methods (core/vectors.py)
```python
async def get_chunks_by_source(self, source_url: str) -> list[DocumentChunk]:
    """Get all existing chunks for a source URL."""
    # Query Qdrant for all chunks with matching source_url

async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
    """Delete specific chunks by their IDs."""
    # Batch delete operation
```

### Phase 4: Configuration & Options

#### 4.1 Add Settings (config.py)
```python
# Deduplication settings
deduplication_enabled: bool = True
deduplication_strategy: str = "content_hash"  # "content_hash", "timestamp", "none"
delete_orphaned_chunks: bool = True
preserve_unchanged_metadata: bool = True
```

#### 4.2 Add Tool Parameters
```python
@mcp.tool
async def crawl(
    ctx: Context,
    target: str,
    process_with_rag: bool = True,
    deduplication: bool = True,  # New parameter
    force_update: bool = False,  # Force update even if unchanged
):
```

### Phase 5: Optimization Strategies

#### 5.1 Batch Operations
- Query all existing chunks for a source in single operation
- Batch upsert operations for better performance
- Use streaming hash calculation for large documents

#### 5.2 Incremental Updates
```python
class IncrementalCrawlStrategy:
    def should_recrawl(self, source_info: SourceInfo) -> bool:
        # Check last modified headers, ETags, etc.

    def process_partial_update(self, changed_pages: list[PageContent]):
        # Only process pages that actually changed
```

#### 5.3 Performance Options
```python
# Skip deduplication for first-time crawls
if not await self.source_exists(url):
    # Fast path: no deduplication needed
    return await self.direct_upsert(chunks)
```

## Edge Cases & Considerations

### 1. URL Normalization
- Handle trailing slashes consistently
- Normalize query parameters
- Handle URL redirects

### 2. Large Documents
- Stream content for hash calculation
- Chunk processing in batches
- Memory-efficient comparison

### 3. Transaction Safety
- Ensure atomic updates (delete + insert)
- Handle partial failures gracefully
- Implement rollback mechanism

### 4. Backwards Compatibility
- Migration strategy for existing data
- Support both old (UUID) and new (deterministic) IDs
- Gradual rollout option

## Benefits

1. **No Duplicates**: Same URL won't create duplicate entries
2. **Change Detection**: Only update when content actually changes
3. **Storage Efficiency**: Reduce Qdrant storage usage
4. **Better Performance**: Skip processing unchanged content
5. **Accurate Timestamps**: Track when content actually changed
6. **Orphan Cleanup**: Remove chunks that no longer exist

## Metrics & Monitoring

Track the following metrics:
- Deduplication rate (% of chunks skipped)
- Storage saved (GB)
- Processing time saved (seconds)
- Change detection accuracy
- Orphaned chunks cleaned

## Testing Strategy

1. **Unit Tests**: Test ID generation, hashing, comparison logic
2. **Integration Tests**: Test full crawl → dedupe → store flow
3. **Performance Tests**: Measure improvement with large datasets
4. **Edge Case Tests**: URL variations, content encoding issues

## Rollout Plan

1. **Phase 1**: Implement deterministic IDs (backward compatible)
2. **Phase 2**: Add content hashing and change detection
3. **Phase 3**: Enable orphan cleanup
4. **Phase 4**: Add configuration options
5. **Phase 5**: Performance optimizations

## Alternative Approaches Considered

1. **Timestamp-based**: Check last_modified headers (rejected: not reliable)
2. **Delete-before-insert**: Delete all, then insert (rejected: inefficient)
3. **Version tracking**: Keep all versions (rejected: storage intensive)
4. **External dedup service**: Use separate service (rejected: complexity)

## Conclusion

This plan provides a comprehensive solution to prevent duplicate content storage while enabling efficient change detection. The deterministic ID approach ensures consistency, while content hashing enables smart updates only when content actually changes.
