# 🔧 Crawly MCP Configuration Guide

This guide covers comprehensive configuration options, best practices, and optimization strategies for the Crawly MCP server.

## 📋 Table of Contents

- [Environment Variables](#-environment-variables)
- [Deduplication Configuration](#-deduplication-configuration)
- [Performance Tuning](#-performance-tuning)
- [Docker Configuration](#-docker-configuration)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)

## 🌍 Environment Variables

### Core Service Configuration

```bash
# Vector Database (Qdrant)
QDRANT_URL=http://localhost:7000
QDRANT_COLLECTION=crawlerr_documents
QDRANT_API_KEY=                        # Optional for secured instances

# Text Embeddings Inference (HF TEI)
TEI_URL=http://localhost:8080
TEI_BATCH_SIZE=64                      # Batch size for embedding requests
TEI_MAX_CONCURRENT_REQUESTS=128        # Max concurrent embedding requests
TEI_TIMEOUT=30                         # Request timeout in seconds

# Crawling Behavior
CRAWL_MAX_PAGES=1000                   # Max pages per crawl session
CRAWL_MAX_DEPTH=3                      # Max crawl depth for recursive crawls
MAX_CONCURRENT_CRAWLS=25               # Max concurrent crawler instances
REQUEST_DELAY=1.0                      # Delay between requests (seconds)
USER_AGENT="Crawly-MCP/1.0"          # User agent string

# Server Configuration
SERVER_HOST=127.0.0.1                  # Server host
SERVER_PORT=8010                       # Server port
LOG_LEVEL=INFO                         # Logging level (DEBUG, INFO, WARN, ERROR)
```

### Deduplication Configuration

```bash
# Content Deduplication
DEDUPLICATION_ENABLED=true             # Enable smart deduplication
DELETE_ORPHANED_CHUNKS=true            # Auto-delete orphaned chunks
CONTENT_HASH_ALGORITHM=sha256          # Hash algorithm for content detection

# Chunking Configuration
CHUNK_SIZE=1024                        # Token-based chunk size
CHUNK_OVERLAP=200                      # Token overlap between chunks
MIN_CHUNK_SIZE=100                     # Minimum chunk size threshold
MAX_CHUNK_SIZE=2048                    # Maximum chunk size limit

# Performance Optimization
FAST_PATH_ENABLED=true                 # Enable fast path for first crawls
BATCH_UPSERT_SIZE=100                  # Batch size for vector upserts
PARALLEL_PROCESSING=true               # Enable parallel chunk processing
```

### Advanced Configuration

```bash
# Memory Management
MAX_MEMORY_USAGE_MB=4096               # Max memory usage limit
MEMORY_CHECK_INTERVAL=60               # Memory check interval (seconds)
GC_THRESHOLD=0.8                       # Garbage collection threshold

# Crawler Timeouts
HTTP_TIMEOUT=30                        # HTTP request timeout
CONNECT_TIMEOUT=10                     # Connection timeout
READ_TIMEOUT=30                        # Read timeout

# Rate Limiting
REQUESTS_PER_SECOND=10                 # Max requests per second per domain
BURST_SIZE=20                          # Burst request allowance
BACKOFF_FACTOR=2                       # Exponential backoff factor

# Content Filtering
MIN_CONTENT_LENGTH=100                 # Minimum content length to process
MAX_CONTENT_LENGTH=1000000             # Maximum content length to process
SKIP_BINARY_FILES=true                 # Skip binary file processing
```

## 🎯 Deduplication Configuration

### Basic Setup

Deduplication is enabled by default for optimal performance. Here's how to configure it:

```bash
# Enable deduplication (recommended)
DEDUPLICATION_ENABLED=true

# Enable orphaned chunk cleanup (recommended)
DELETE_ORPHANED_CHUNKS=true

# Content change detection
CONTENT_HASH_ALGORITHM=sha256
```

### Performance Modes

#### High Performance Mode (Default)
Best for most use cases with good balance of speed and accuracy:

```bash
DEDUPLICATION_ENABLED=true
FAST_PATH_ENABLED=true
BATCH_UPSERT_SIZE=100
PARALLEL_PROCESSING=true
CONTENT_HASH_ALGORITHM=sha256
```

#### Memory Optimized Mode
For systems with limited RAM:

```bash
DEDUPLICATION_ENABLED=true
BATCH_UPSERT_SIZE=50
PARALLEL_PROCESSING=false
MAX_MEMORY_USAGE_MB=2048
CHUNK_SIZE=512
```

#### Maximum Accuracy Mode
For critical applications requiring highest accuracy:

```bash
DEDUPLICATION_ENABLED=true
FAST_PATH_ENABLED=false
CONTENT_HASH_ALGORITHM=sha256
MIN_CHUNK_SIZE=200
CHUNK_OVERLAP=400
```

### Chunking Strategy

Configure chunking based on your content type:

#### General Web Content
```bash
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
MIN_CHUNK_SIZE=100
MAX_CHUNK_SIZE=2048
```

#### Code Documentation
```bash
CHUNK_SIZE=768
CHUNK_OVERLAP=150
MIN_CHUNK_SIZE=200
MAX_CHUNK_SIZE=1536
```

#### Academic Papers
```bash
CHUNK_SIZE=1536
CHUNK_OVERLAP=300
MIN_CHUNK_SIZE=300
MAX_CHUNK_SIZE=3072
```

## ⚡ Performance Tuning

### CPU Optimization

#### Multi-core Systems (8+ cores)
```bash
MAX_CONCURRENT_CRAWLS=32
PARALLEL_PROCESSING=true
TEI_MAX_CONCURRENT_REQUESTS=128
BATCH_UPSERT_SIZE=150
```

#### Limited CPU Resources (2-4 cores)
```bash
MAX_CONCURRENT_CRAWLS=8
PARALLEL_PROCESSING=false
TEI_MAX_CONCURRENT_REQUESTS=32
BATCH_UPSERT_SIZE=50
```

### Memory Optimization

#### High Memory Systems (16GB+)
```bash
MAX_MEMORY_USAGE_MB=8192
CHUNK_SIZE=1536
BATCH_UPSERT_SIZE=200
TEI_BATCH_SIZE=128
```

#### Limited Memory Systems (4-8GB)
```bash
MAX_MEMORY_USAGE_MB=2048
CHUNK_SIZE=512
BATCH_UPSERT_SIZE=25
TEI_BATCH_SIZE=32
```

### Network Optimization

#### High-Bandwidth Networks
```bash
REQUESTS_PER_SECOND=20
BURST_SIZE=50
MAX_CONCURRENT_CRAWLS=40
HTTP_TIMEOUT=60
```

#### Limited Bandwidth
```bash
REQUESTS_PER_SECOND=5
BURST_SIZE=10
MAX_CONCURRENT_CRAWLS=10
REQUEST_DELAY=2.0
```

## 🐳 Docker Configuration

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:v1.15.1
    ports:
      - "7000:7000"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # HuggingFace Text Embeddings Inference
  text-embeddings-inference:
    image: ghcr.io/huggingface/text-embeddings-inference:89-latest
    ports:
      - "8080:80"
    volumes:
      - hf_cache:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - HUGGINGFACE_HUB_CACHE=/data
    command:
      - --model-id=Qwen/Qwen3-Embedding-0.6B
      - --max-concurrent-requests=128
      - --max-batch-tokens=32768
      - --max-batch-requests=64
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_storage:
  hf_cache:
```

### GPU-Optimized Configuration

For systems with NVIDIA GPUs:

```yaml
text-embeddings-inference:
  image: ghcr.io/huggingface/text-embeddings-inference:89-latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    - NVIDIA_VISIBLE_DEVICES=0
  command:
    - --model-id=Qwen/Qwen3-Embedding-0.6B
    - --max-concurrent-requests=256
    - --max-batch-tokens=65536
    - --max-batch-requests=128
```

### Production Configuration

For production deployments:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.15.1
    ports:
      - "7000:7000"
    volumes:
      - /opt/crawly/qdrant:/qdrant/storage
    environment:
      - QDRANT_LOG_LEVEL=WARN
      - QDRANT_SERVICE__HTTP_PORT=7000
      - QDRANT_SERVICE__GRPC_PORT=7001
      - QDRANT_STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8
      - QDRANT_STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=4
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  text-embeddings-inference:
    image: ghcr.io/huggingface/text-embeddings-inference:89-latest
    ports:
      - "8080:80"
    volumes:
      - /opt/crawly/hf_cache:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    command:
      - --model-id=Qwen/Qwen3-Embedding-0.6B
      - --max-concurrent-requests=256
      - --max-batch-tokens=65536
      - --max-batch-requests=128
      - --max-client-batch-size=32
      - --auto-truncate
    restart: always
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## 🌟 Best Practices

### Deduplication Best Practices

#### 1. Enable for Production
```bash
# Always enable deduplication in production
DEDUPLICATION_ENABLED=true
DELETE_ORPHANED_CHUNKS=true
```

#### 2. Optimize Chunk Size
```bash
# Choose chunk size based on content type
# General web content: 1024 tokens
# Technical docs: 768-1024 tokens
# Academic papers: 1536+ tokens
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
```

#### 3. Monitor Performance
Track these metrics for optimal performance:
- Re-crawl speed improvement (target: >3x for unchanged content)
- Storage reduction (typical: 30-70% for repeated crawls)
- Processing time per chunk (<10ms average)

#### 4. Regular Maintenance
```bash
# Run migration script for existing installations
python scripts/migrate_to_deterministic_ids.py --dry-run

# Monitor orphaned chunks
# Check logs for deduplication statistics
```

### Crawling Best Practices

#### 1. Respectful Crawling
```bash
# Be respectful to target sites
REQUEST_DELAY=1.0
REQUESTS_PER_SECOND=10
USER_AGENT="Crawly-MCP/1.0 (+https://your-domain.com/bot)"
```

#### 2. Content Quality
```bash
# Filter low-quality content
MIN_CONTENT_LENGTH=100
MAX_CONTENT_LENGTH=1000000
SKIP_BINARY_FILES=true
```

#### 3. Resource Management
```bash
# Prevent resource exhaustion
MAX_MEMORY_USAGE_MB=4096
HTTP_TIMEOUT=30
MAX_CONCURRENT_CRAWLS=25
```

### Vector Database Best Practices

#### 1. Collection Configuration
```python
# Optimal Qdrant collection settings
{
    "vectors": {
        "size": 1024,  # Match embedding model dimensions
        "distance": "Cosine"  # Best for text similarity
    },
    "optimizers_config": {
        "max_optimization_threads": 4,
        "max_segment_size": 200000
    },
    "hnsw_config": {
        "m": 16,  # Balance between accuracy and speed
        "ef_construct": 128,  # Build-time accuracy
        "full_scan_threshold": 10000
    }
}
```

#### 2. Performance Monitoring
```bash
# Monitor vector database performance
# - Query latency should be <100ms
# - Memory usage should be stable
# - Index build time should be reasonable
```

### Embedding Best Practices

#### 1. Model Selection
```bash
# Use Qwen3-Embedding-0.6B for multilingual support
# Alternative models:
# - sentence-transformers/all-MiniLM-L6-v2 (English-focused, faster)
# - intfloat/e5-large-v2 (Higher accuracy, slower)
```

#### 2. Batch Processing
```bash
# Optimize batch sizes for your GPU
# RTX 4070: TEI_BATCH_SIZE=64-128
# RTX 3080: TEI_BATCH_SIZE=32-64
# CPU only: TEI_BATCH_SIZE=16-32
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Slow Re-crawls Despite Deduplication
**Symptoms**: Re-crawls taking similar time as first crawl
**Solutions**:
```bash
# Check if deduplication is enabled
DEDUPLICATION_ENABLED=true

# Verify fast path is working
# Look for "Fast path: No existing chunks found" in logs

# Check chunk ID consistency
# Run migration script if needed
python scripts/migrate_to_deterministic_ids.py --dry-run
```

#### 2. High Memory Usage
**Symptoms**: Server consuming excessive memory
**Solutions**:
```bash
# Reduce batch sizes
BATCH_UPSERT_SIZE=50
TEI_BATCH_SIZE=32

# Limit concurrent operations
MAX_CONCURRENT_CRAWLS=10
PARALLEL_PROCESSING=false

# Set memory limits
MAX_MEMORY_USAGE_MB=2048
```

#### 3. Vector Database Connection Issues
**Symptoms**: Connection refused or timeout errors
**Solutions**:
```bash
# Check service status
docker-compose ps

# Verify connectivity
curl http://localhost:7000/health

# Check firewall/network configuration
# Ensure ports 7000 (Qdrant) and 8080 (TEI) are open
```

#### 4. Embedding Generation Failures
**Symptoms**: TEI service errors or timeouts
**Solutions**:
```bash
# Check TEI service logs
docker-compose logs text-embeddings-inference

# Reduce request load
TEI_MAX_CONCURRENT_REQUESTS=64
TEI_TIMEOUT=60

# Verify GPU availability (if using GPU)
nvidia-smi
```

#### 5. Poor Deduplication Performance
**Symptoms**: Expected duplicates not being detected
**Solutions**:
```bash
# Check content hash algorithm
CONTENT_HASH_ALGORITHM=sha256

# Verify URL normalization
# Check logs for consistent chunk IDs

# Run performance tests
python -m pytest tests/test_deduplication_performance.py -v
```

### Debug Mode Configuration

For troubleshooting, enable debug mode:

```bash
# Comprehensive debug logging
LOG_LEVEL=DEBUG

# Enable detailed deduplication logging
# This will show chunk-by-chunk processing details

# Disable fast path to see full processing
FAST_PATH_ENABLED=false

# Enable memory debugging
MEMORY_CHECK_INTERVAL=30
```

### Performance Profiling

To profile performance:

```bash
# Enable timing logs
LOG_LEVEL=DEBUG

# Run performance tests
python -m pytest tests/test_deduplication_performance.py -v -s

# Monitor system resources
htop  # CPU and memory usage
iotop  # Disk I/O
nethogs  # Network usage
```

### Log Analysis

Key log patterns to monitor:

```bash
# Successful deduplication
"Skipping unchanged chunk" - Good, deduplication working
"Fast path: No existing chunks found" - Good, optimization working

# Performance indicators
"Processing X pages for RAG indexing (dedup=true)" - Check processing time
"Found X existing chunks for URL" - Verify chunk retrieval

# Warning signs
"Could not retrieve existing chunks" - Investigate connectivity
"Migration script recommended" - Run UUID migration
"High memory usage detected" - Tune memory settings
```

---

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check Logs**: Enable DEBUG logging and examine the output
2. **Performance Tests**: Run the test suite to verify functionality
3. **Community Support**: Open an issue on GitHub with logs and configuration
4. **Documentation**: Refer to the main README.md and implementation plans

Remember: Deduplication is designed to improve performance automatically. Most users should stick with the default configuration unless specific optimization is needed.
