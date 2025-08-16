# Crawl Tool Fix Plan - Integer Hash Markdown Issue

## Problem Summary
The crawl tool fails with `'int' object has no attribute 'raw_markdown'` when performing multi-page crawls. Single-page scraping works fine. The error occurs at line 509 in `crawler_mcp/crawlers/web.py` when accessing `result.markdown` in a debug print statement.

## Root Cause Analysis

### 1. The Error Location
**File:** `/home/jmagar/code/crawl-mcp/crawler_mcp/crawlers/web.py`
**Line:** 509 (debug print accessing `result.markdown`)
**Traceback:**
```
File "/home/jmagar/code/crawl-mcp/.venv/lib/python3.11/site-packages/crawl4ai/models.py", line 263, in __new__
    return super().__new__(cls, markdown_result.raw_markdown)
AttributeError: 'int' object has no attribute 'raw_markdown'
```

### 2. The Real Issue
- The `_sanitize_crawl_result()` method (line 505) is supposed to fix integer hashes in the `_markdown` field
- However, immediately after sanitization, line 509 tries to access `result.markdown` in a debug print
- This access triggers the `StringCompatibleMarkdown.__new__()` method in crawl4ai which expects a `MarkdownGenerationResult` but gets an integer

### 3. Why It Happens
- During streaming mode with deep crawl (multi-page), crawl4ai may set `_markdown` to an integer hash for deduplication
- Our sanitization checks `_markdown` (private field) but the property accessor `markdown` still triggers the error
- The debug print at line 509 happens BEFORE we've had a chance to properly handle the result

### 4. Configuration Issue
- Multi-page crawls automatically enable `BFSDeepCrawlStrategy` (line 952-957)
- Deep crawl strategy forces `stream=True` (line 776-783)
- Streaming mode is where the integer hash issue manifests

## Fix Implementation Plan

### Phase 1: Immediate Critical Fixes (Priority: HIGH)

#### Fix 1: Remove Dangerous Debug Statement
**Location:** `crawler_mcp/crawlers/web.py:508-512`
```python
# REMOVE or GUARD this debug print that causes the crash
# OLD:
print(
    f"CRAWL DEBUG - Inspecting result.markdown: type={type(result.markdown)}, value={result.markdown}",
    file=sys.stderr,
    flush=True,
)

# NEW:
try:
    # Only access markdown if it's safe
    if hasattr(result, '_markdown') and not isinstance(getattr(result, '_markdown', None), int):
        print(
            f"CRAWL DEBUG - Markdown field is safe to access",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"CRAWL DEBUG - Markdown field contains integer hash, skipping access",
            file=sys.stderr,
            flush=True,
        )
except Exception as e:
    print(f"CRAWL DEBUG - Cannot inspect markdown: {e}", file=sys.stderr, flush=True)
```

#### Fix 2: Improve Sanitization Method
**Location:** `crawler_mcp/crawlers/web.py:475-497`
```python
def _sanitize_crawl_result(self, result: Crawl4aiResult) -> Crawl4aiResult:
    """Sanitize CrawlResult to prevent integer hash issues with markdown field."""
    import sys
    from crawl4ai.models import MarkdownGenerationResult

    # Check both _markdown and handle the property access issue
    try:
        # First check if _markdown is an integer
        if hasattr(result, "_markdown") and isinstance(result._markdown, int):
            print(
                f"CRAWL DEBUG - Found integer _markdown ({result._markdown}), replacing with empty MarkdownGenerationResult",
                file=sys.stderr,
                flush=True,
            )
            # Replace the integer hash with an empty MarkdownGenerationResult
            result._markdown = MarkdownGenerationResult(
                raw_markdown="",
                markdown_with_citations="",
                references_markdown="",
                fit_markdown=None,
                fit_html=None,
            )

        # Also check if markdown property access would fail
        # This is a defensive check
        if hasattr(result, 'markdown'):
            try:
                # Try to access it to see if it would error
                _ = result.markdown
            except AttributeError as e:
                if "'int' object has no attribute" in str(e):
                    # Force set a safe markdown value
                    result._markdown = MarkdownGenerationResult(
                        raw_markdown="",
                        markdown_with_citations="",
                        references_markdown="",
                        fit_markdown=None,
                        fit_html=None,
                    )
    except Exception as e:
        print(f"CRAWL DEBUG - Sanitization warning: {e}", file=sys.stderr, flush=True)

    return result
```

#### Fix 3: Add Pre-Iteration Type Checking
**Location:** `crawler_mcp/crawlers/web.py:214` (in async iteration loop)
```python
async for result in crawl_result:
    crawl_count += 1

    # NEW: Pre-check for integer results (shouldn't happen according to crawl4ai docs, but we see it)
    if isinstance(result, int):
        self.logger.warning(
            "Received integer %d instead of CrawlResult in streaming mode, skipping",
            result
        )
        continue

    # NEW: Ensure result is a CrawlResult object
    if not hasattr(result, 'success'):
        self.logger.warning(
            "Received unexpected type %s in streaming mode, skipping",
            type(result).__name__
        )
        continue

    if result.success:
        try:
            sanitized_result = self._sanitize_crawl_result(result)
            successful_results.append(sanitized_result)
        except AttributeError as e:
            # ... existing error handling ...
```

### Phase 2: Configuration Improvements (Priority: MEDIUM)

#### Fix 4: Add Streaming Control
**Location:** `crawler_mcp/config.py`
```python
# Add new configuration option
crawl_force_streaming: bool = Field(
    default=False,
    description="Force streaming mode even for single-page crawls"
)
crawl_disable_streaming: bool = Field(
    default=False,
    description="Disable streaming mode even for multi-page crawls (may use more memory)"
)
```

#### Fix 5: Conditional Streaming Based on Config
**Location:** `crawler_mcp/crawlers/web.py:776-783`
```python
# Improve streaming decision logic
if settings.crawl_disable_streaming:
    stream_enabled = False
elif settings.crawl_force_streaming:
    stream_enabled = True
else:
    # Default: Enable streaming only if deep crawl is used
    stream_enabled = (deep_strategy is not None)

self.logger.info(
    "Streaming mode: %s (deep_crawl=%s, force=%s, disable=%s)",
    stream_enabled,
    deep_strategy is not None,
    settings.crawl_force_streaming,
    settings.crawl_disable_streaming
)
```

### Phase 3: Enhanced Error Recovery (Priority: LOW)

#### Fix 6: Add Retry Logic
```python
# If streaming mode fails with integer hash issue, retry without streaming
if "'int' object has no attribute 'raw_markdown'" in str(error):
    self.logger.warning("Retrying crawl without streaming mode due to integer hash issue")
    run_config.stream = False
    # Retry the crawl
```

#### Fix 7: Add Type Hints
- Add proper type stubs for crawl4ai imports to satisfy mypy
- Create a `py.typed` marker or ignore the import warnings

## Testing Plan

### Test Cases
1. **Single-page crawl** - Should continue working (already works)
2. **Multi-page crawl with streaming** - Should handle integer hashes gracefully
3. **Multi-page crawl without streaming** - Test with `crawl_disable_streaming=True`
4. **Deduplication scenario** - Crawl the same URLs twice to trigger deduplication

### Test Commands
```bash
# Test single page (should work)
curl -X POST http://localhost:8010/call_tool \
  -H "Content-Type: application/json" \
  -d '{"name": "scrape", "arguments": {"url": "https://example.com"}}'

# Test multi-page (currently fails, should work after fix)
curl -X POST http://localhost:8010/call_tool \
  -H "Content-Type: application/json" \
  -d '{"name": "crawl", "arguments": {"target": "https://modelcontextprotocol.io"}}'
```

## Implementation Order

1. **First**: Remove/fix the debug print at line 509 (Fix 1) - This alone might resolve the immediate issue
2. **Second**: Improve sanitization method (Fix 2)
3. **Third**: Add pre-iteration type checking (Fix 3)
4. **Fourth**: Test thoroughly
5. **Optional**: Implement configuration controls if needed (Fixes 4-5)

## Expected Outcome

After implementing these fixes:
- Multi-page crawls should work without the integer hash error
- The tool should gracefully handle any deduplication scenarios
- Debug output should be safer and more informative
- Users can control streaming behavior if needed

## Notes for Implementation

- The core issue is the unsafe access to `result.markdown` at line 509
- Crawl4ai documentation claims `_markdown` is never an integer, but empirical evidence shows otherwise
- The fix should be defensive - handle both expected and unexpected types
- Preserve existing functionality while adding safety checks
