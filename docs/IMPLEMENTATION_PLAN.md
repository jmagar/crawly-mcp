# Crawl4AI FastMCP RAG MCP Server Implementation Plan

## Project Overview
Create a Python FastMCP MCP server with streamable-HTTP transport that provides advanced web crawling and RAG capabilities using Crawl4AI 0.7.0, Qdrant vector database, and Qwen3-Embedding-0.6B via HF TEI.

## Architecture Components

### 1. Core Infrastructure
- **FastMCP 2.0 Server** with streamable-HTTP transport
- **Qdrant Vector Database** deployed via Docker Compose
- **HF TEI Service** with Qwen3-Embedding-0.6B (already configured)
- **Crawl4AI 0.7.0** with advanced features integration

### 2. Docker Compose Enhancement
```yaml
# Add to existing docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.6.1
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: "INFO"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  qdrant_data:
```

### 3. MCP Server Structure
```
src/
├── server.py              # FastMCP server with streamable-HTTP
├── crawlers/
│   ├── __init__.py
│   ├── web_crawler.py      # Crawl4AI integration
│   ├── repo_crawler.py     # GitHub repo crawling
│   └── file_crawler.py     # Local directory crawling
├── rag/
│   ├── __init__.py
│   ├── embedding_client.py # HF TEI client
│   ├── vector_store.py     # Qdrant integration
│   └── search.py          # RAG query implementation
├── middleware/
│   ├── __init__.py
│   ├── logging.py         # Request/response logging
│   ├── error_handler.py   # Error handling middleware
│   └── progress.py        # Progress tracking
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models for data structures
└── utils/
    ├── __init__.py
    ├── config.py          # Configuration management
    └── helpers.py         # Utility functions
```

## Tool Implementations

### 1. scrape
**Purpose**: Single page crawling using Crawl4AI AsyncWebCrawler

**Features**:
- Extract markdown, metadata, images, links
- Store in Qdrant with comprehensive metadata
- Support for LLM extraction strategies
- Adaptive crawling with intelligent stopping
- Virtual scroll support for infinite pages

**Parameters**:
- `url` (required): Target URL to scrape
- `extraction_strategy` (optional): "css", "llm", "regex"
- `deep_crawl` (optional): Boolean for deep crawling
- `max_depth` (optional): Maximum crawling depth
- `include_images` (optional): Extract and store images
- `llm_provider` (optional): LLM provider for extraction

### 2. crawl
**Purpose**: Comprehensive site crawling via sitemap.xml or recursive crawling

**Features**:
- Sitemap.xml parsing (primary method)
- Fallback to recursive crawling (3 levels, configurable)
- Use BFS/DFS deep crawling strategies
- Batch processing with progress reporting
- Link preview with 3-layer scoring
- Async URL seeder for massive discovery

**Parameters**:
- `url` (required): Starting URL or sitemap URL
- `strategy` (optional): "sitemap", "bfs", "dfs", "best_first"
- `max_depth` (optional): Maximum crawling depth (default: 3)
- `max_pages` (optional): Maximum pages to crawl
- `include_external` (optional): Include external domains
- `domain_filter` (optional): Domain filtering patterns

### 3. crawl_repo
**Purpose**: Clone GitHub repositories and crawl the files

**Features**:
- Clone GitHub repositories to temp directory
- Crawl code files with language detection
- Extract documentation, README files
- Store with repository metadata
- Support for private repositories with authentication

**Parameters**:
- `repo_url` (required): GitHub repository URL
- `branch` (optional): Specific branch to crawl
- `file_patterns` (optional): File patterns to include/exclude
- `include_docs` (optional): Include documentation files
- `auth_token` (optional): GitHub authentication token

### 4. crawl_dir
**Purpose**: Crawl local directory structures

**Features**:
- Local directory traversal
- File type filtering and processing
- Support for various document formats
- Maintain directory structure metadata
- PDF processing and text extraction

**Parameters**:
- `directory_path` (required): Local directory path
- `file_extensions` (optional): File extensions to include
- `recursive` (optional): Recursive directory traversal
- `max_files` (optional): Maximum files to process
- `exclude_patterns` (optional): Patterns to exclude

### 5. rag_query
**Purpose**: Semantic search using Qdrant vector database

**Features**:
- Semantic search using Qdrant
- Query expansion and reranking
- Context-aware results with metadata
- Similarity scoring and filtering
- Hybrid search (semantic + keyword)

**Parameters**:
- `query` (required): Search query text
- `limit` (optional): Number of results to return
- `threshold` (optional): Similarity threshold
- `source_filter` (optional): Filter by source type
- `date_filter` (optional): Filter by date range

### 6. list_sources
**Purpose**: List all crawled sources with metadata

**Features**:
- List all crawled sources with metadata
- Filter by source type, date, domain
- Pagination support
- Export capabilities
- Statistics and analytics

**Parameters**:
- `source_type` (optional): Filter by source type
- `domain` (optional): Filter by domain
- `date_from` (optional): Start date filter
- `date_to` (optional): End date filter
- `limit` (optional): Number of results per page
- `offset` (optional): Pagination offset

## Advanced Features Integration

### FastMCP 2.0 Features
- **Context Object**: Logging, progress reporting, user elicitation
- **Middleware**: Request/response logging, error handling, authentication
- **Resources**: Expose crawled data as readable resources
- **Prompts**: Pre-defined crawling and analysis templates
- **Streamable HTTP**: Real-time progress updates and streaming responses
- **Tool Transformation**: Dynamic tool modification
- **Component Management**: Enable/disable tools at runtime

### Crawl4AI 0.7.0 Features
- **Adaptive Crawling**: Smart stopping when sufficient data gathered
- **Virtual Scroll**: Support for infinite scroll pages
- **Link Preview**: 3-layer intelligent link scoring
- **LLM Strategies**: Provider-agnostic extraction (OpenAI, Ollama, etc.)
- **Deep Crawling**: BFS/DFS strategies with advanced filtering
- **Session Management**: Persistent browser sessions
- **Content Filters**: LLM-powered content filtering
- **Schema Generation**: Automatic extraction schema generation

### Qdrant Integration Features
- **Collections**: Organized storage by content type
- **Filtering**: Advanced metadata filtering
- **Hybrid Search**: Vector + keyword search
- **Batch Operations**: Efficient bulk operations
- **Clustering**: Content clustering and organization

### Qwen3-Embedding Features
- **Multilingual Support**: 100+ languages
- **High Performance**: Optimized for speed and accuracy
- **Batch Processing**: Efficient embedding generation
- **TEI Integration**: Seamless HF TEI integration

## Metadata Embedding Strategy

### Content Metadata
- `url`: Source URL
- `title`: Page/document title
- `description`: Meta description or summary
- `keywords`: Extracted keywords and tags
- `language`: Detected content language
- `content_type`: MIME type or content category

### Technical Metadata
- `crawl_timestamp`: When content was crawled
- `file_size`: Content size in bytes
- `response_status`: HTTP response status
- `headers`: Relevant HTTP headers
- `encoding`: Character encoding
- `hash`: Content hash for deduplication

### Crawl Context
- `source_tool`: Which tool was used for crawling
- `depth_level`: Crawl depth from starting point
- `parent_url`: Parent URL in crawl hierarchy
- `crawl_session_id`: Session identifier
- `user_agent`: User agent used for crawling

### Quality Metrics
- `extraction_confidence`: Confidence score for extracted data
- `content_completeness`: Percentage of content successfully extracted
- `processing_time`: Time taken to process content
- `error_count`: Number of errors encountered
- `retry_count`: Number of retries performed

### Semantic Tags
- `topics`: Auto-generated topic classifications
- `entities`: Named entities extracted
- `sentiment`: Content sentiment analysis
- `complexity_score`: Content complexity rating
- `relevance_score`: Relevance to query/context

## Error Handling & Monitoring

### Exception Handling
- Comprehensive exception handling with context preservation
- Custom exception classes for different error types
- Graceful degradation for failed crawl attempts
- Error logging with full context and stack traces

### Retry Mechanisms
- Exponential backoff for transient failures
- Circuit breaker pattern for persistent failures
- Dead letter queue for failed operations
- Retry policies based on error type

### Logging Strategy
- Structured logging with correlation IDs
- Different log levels for different components
- Performance metrics logging
- Security event logging

### Monitoring & Observability
- Progress tracking for long-running operations
- Real-time status updates via streamable HTTP
- Health checks for all components
- Metrics collection and aggregation

## Implementation Steps

### Phase 1: Foundation Setup
1. **Docker Environment**
   - Extend docker-compose.yml with Qdrant service
   - Verify HF TEI service integration
   - Set up networking between services

2. **Project Structure**
   - Create directory structure
   - Set up Python package structure
   - Install dependencies (FastMCP, Crawl4AI, Qdrant client)

3. **Basic MCP Server**
   - Create FastMCP server with streamable-HTTP transport
   - Implement basic server configuration
   - Add health check endpoints

### Phase 2: Core Components
4. **Embedding Client**
   - Implement HF TEI client for Qwen3-Embedding-0.6B
   - Add embedding generation and batching
   - Test embedding quality and performance

5. **Vector Store Integration**
   - Implement Qdrant client wrapper
   - Create collection management
   - Add CRUD operations for vectors

6. **Crawl4AI Integration**
   - Set up AsyncWebCrawler with advanced configuration
   - Implement extraction strategies
   - Add session management

### Phase 3: Tool Implementation
7. **Scrape Tool**
   - Single page crawling implementation
   - Metadata extraction and storage
   - Error handling and validation

8. **Crawl Tool**
   - Sitemap parsing implementation
   - Deep crawling strategies
   - Progress reporting integration

9. **Repository Crawler**
   - GitHub API integration
   - File processing pipeline
   - Code-specific metadata extraction

10. **Directory Crawler**
    - Local file system traversal
    - Document processing
    - File type detection

11. **RAG Query Tool**
    - Semantic search implementation
    - Result ranking and filtering
    - Context-aware responses

12. **List Sources Tool**
    - Source enumeration and filtering
    - Metadata aggregation
    - Export functionality

### Phase 4: Advanced Features
13. **Middleware Implementation**
    - Logging middleware with correlation IDs
    - Error handling middleware
    - Progress tracking middleware

14. **Resource Endpoints**
    - Expose crawled data as MCP resources
    - Implement resource discovery
    - Add resource streaming

15. **Prompt Templates**
    - Create reusable crawling prompts
    - Implement prompt parameterization
    - Add prompt validation

### Phase 5: Testing & Optimization
16. **Comprehensive Testing**
    - Unit tests for all components
    - Integration tests for end-to-end workflows
    - Performance testing and benchmarking

17. **Performance Optimization**
    - Concurrent processing optimization
    - Memory usage optimization
    - Database query optimization

18. **Documentation**
    - API documentation
    - Usage examples
    - Deployment guides

### Phase 6: Production Readiness
19. **Security Implementation**
    - Authentication and authorization
    - Input validation and sanitization
    - Rate limiting and abuse prevention

20. **Monitoring & Observability**
    - Metrics collection
    - Alerting configuration
    - Performance dashboards

21. **Deployment Configuration**
    - Production Docker configuration
    - Environment variable management
    - CI/CD pipeline setup

## Technology Stack Summary

- **FastMCP 2.0**: MCP server framework with advanced features
- **Crawl4AI 0.7.0**: Advanced web crawling and extraction
- **Qdrant**: Vector database for semantic search
- **HF TEI**: Text embeddings inference server
- **Qwen3-Embedding-0.6B**: Multilingual embedding model
- **Docker Compose**: Container orchestration
- **Python 3.9+**: Primary development language

## Success Criteria

1. **Functionality**: All tools working with comprehensive error handling
2. **Performance**: Efficient crawling and embedding generation
3. **Scalability**: Handle large-scale crawling operations
4. **Reliability**: Robust error handling and recovery
5. **Usability**: Intuitive API with good documentation
6. **Maintainability**: Clean, well-structured codebase

This implementation plan provides a comprehensive roadmap for building a production-ready RAG-enabled web crawling MCP server that leverages the full capabilities of FastMCP, Crawl4AI, Qdrant, and Qwen3 embeddings.
