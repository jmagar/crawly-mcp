# Crawler MCP Testing Guide

This directory contains comprehensive tests for the Crawler MCP FastMCP server using in-memory testing with real services.

## Test Structure

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast tests that don't require external services
- **Integration Tests** (`@pytest.mark.integration`): Tests with real Qdrant and TEI services
- **Slow Tests** (`@pytest.mark.slow`): Tests that take longer (website crawling, etc.)
- **Service-Dependent Tests** (`@pytest.mark.requires_services`): Tests requiring Qdrant and TEI

### Test Files

- `conftest.py`: Fixtures and test configuration
- `test_server_health.py`: Server health and basic functionality
- `test_crawling_tools.py`: Web crawling and scraping tools
- `test_rag_tools.py`: RAG functionality and vector search
- `test_core_services.py`: Direct testing of core services

## Running Tests

### Prerequisites

1. **Start Required Services**:

   ```bash
   docker-compose up -d  # Start Qdrant and TEI services
   ```

2. **Environment Variables**:
   Copy `tests/.env.test` to `.env` and adjust as needed:

   ```bash
   cp tests/.env.test .env
   ```

   Note: The examples below assume default ports. If you've configured custom ports, replace:
   - `7000` with the value of `${QDRANT_HTTP_PORT}`
   - `8080` with the value of `${TEI_HTTP_PORT}`

### Test Commands

```bash
# Run all tests
uv run pytest

# Run only unit tests (no external services needed)
uv run pytest -m "unit"

# Run integration tests (requires services)
uv run pytest -m "integration"

# Run tests excluding slow ones
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/test_server_health.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=crawler_mcp
```

## In-Memory Testing Approach

Our tests use FastMCP's in-memory testing capabilities:

- **Direct Server Instance**: Tests pass the server instance directly to the FastMCP Client
- **No Network Overhead**: Everything runs in the same Python process
- **Real Services**: Uses actual Qdrant and TEI services for realistic testing
- **Debuggable**: Set breakpoints in tests or server code and step through

### Example Test Pattern

```python
@pytest.mark.integration
@pytest.mark.requires_services
async def test_example(mcp_client: Client):
    """Test using in-memory client with real services."""
    # Call tool directly through in-memory connection
    result = await mcp_client.call_tool("scrape", {
        "url": "https://example.com",
        "auto_index": True,
    })

    # Verify results
    assert result.data["success"] is True
    assert "content" in result.data
```

## Test Environment

### Fixtures

- `mcp_client`: In-memory FastMCP client connected to server
- `embedding_service`: Direct embedding service instance
- `vector_service`: Direct vector service instance
- `rag_service`: Direct RAG service instance
- `clean_test_collection`: Ensures clean test database state
- `sample_text_files`: Creates test files for directory crawling

### Service Management

- Tests use a separate Qdrant collection (`test_crawler_mcp`)
- Collection is cleaned before each test requiring it
- Services are checked for availability before running service-dependent tests
- Tests are skipped if required services are not available

## Debugging Tests

### Breakpoints

Since everything runs in-memory, you can set breakpoints:

```python
async def test_debug_example(mcp_client: Client):
    breakpoint()  # Debugger will stop here
    result = await mcp_client.call_tool("scrape", {"url": "..."})
    breakpoint()  # And here
```

### Logging

Adjust log levels in `.env` for more detailed output:

```env
LOG_LEVEL=DEBUG  # For detailed logging
LOG_TO_FILE=true  # To capture logs in files
```

### Test Data Inspection

Access services directly in tests:

```python
async def test_inspect_data(vector_service: VectorService):
    info = await vector_service.get_collection_info()
    print(f"Collection has {info['points_count']} points")
```

## Performance Considerations

- **Fast Execution**: In-memory tests run in milliseconds
- **Service Startup**: Allow time for Qdrant/TEI to be fully ready
- **Resource Cleanup**: Tests clean up after themselves
- **Parallel Execution**: Tests can run in parallel (use `pytest-xdist`)

## Continuous Integration

For CI environments:

1. Start services with Docker Compose
2. Wait for services to be healthy
3. Run tests with appropriate markers
4. Use test-specific timeouts and retries

Example GitHub Actions:

```yaml
- name: Start services
  run: docker-compose up -d

- name: Wait for services
  run: |
    # Using environment variables for port configuration
    timeout 60 bash -c 'until curl -f http://localhost:${QDRANT_HTTP_PORT:-7000}/health; do sleep 2; done'
    timeout 60 bash -c 'until curl -f http://localhost:${TEI_HTTP_PORT:-8080}/health; do sleep 2; done'

- name: Run tests
  run: uv run pytest -m "not slow"
```
