
Skip to content
Navigation Menu
jmagar
crawler-mcp

Code
Issues
Pull requests 1
Actions
Projects
Wiki
Security
Insights

    Settings

Feat/new development #3
Open
jmagar wants to merge 5 commits into main from feat/new-development
Open
Feat/new development
#3
jmagar wants to merge 5 commits into main from feat/new-development
+9,657 −2,262
Conversation 147
Commits 5
Checks 1
Files changed 46
Conversation
jmagar
Owner
jmagar commented Aug 20, 2025 •
Summary by CodeRabbit

    New Features
        Safer scraping with improved content extraction/sanitization, optional extraction strategies, concurrency/performance knobs, and a high-performance browser pool.
        RAG indexing upgraded to subchunk-level processing with per-subchunk deduplication, parallel embedding pipeline, richer metadata, and orphan cleanup.
        Typed integration layer for crawl components and embedding throughput metrics.

    Bug Fixes
        More robust error handling and corrected crawl success-rate; suppresses noisy deprecation warnings.

    Documentation
        Added middleware and testing guides plus tests README.

    Tests
        Large new test suites added; some obsolete tests removed.

    Chores
        .gitignore updated, dev script removed, docker and dependency/tooling updates.

jmagar and others added 2 commits August 16, 2025 19:30
@jmagar
@claude
fix: resolve crawl4ai integer hash issues and improve type safety
97eee2d
@jmagar
@claude
fix: implement proper text chunking to resolve embedding timeouts
c57d379
@Copilot Copilot AI review requested due to automatic review settings August 20, 2025 15:37
@coderabbitai coderabbitai
coderabbitai bot commented Aug 20, 2025 •

Note
Other AI code review bot(s) detected

CodeRabbit has detected other AI code review bot(s) in this pull request and will avoid duplicating their findings in the review comments. This may lead to a less comprehensive review.
Walkthrough

Adds a crawl4ai typing facade, hardens markdown extraction and sanitization, migrates RAG deduplication to subchunk granularity with richer metadata and orphan cleanup, makes extraction_strategy optional, introduces high-performance browser/file pools, tightens typing and test infra, adjusts dev/docker configs, and adds many docs and tests.
Changes
Cohort / File(s) 	Summary of changes
Type facade for crawl4ai
crawler_mcp/types/crawl4ai_types.py 	New protocol-based typed contract layer and runtime-safe aliases for crawl4ai components (results, markdown, filters, extraction strategies, run configs, enums); guarded runtime imports and exported implementation aliases.
Crawling content handling & APIs
crawler_mcp/crawlers/web.py, crawler_mcp/core/orchestrator.py, crawler_mcp/crawlers/base.py, crawler_mcp/models/crawl.py, crawler_mcp/tools/crawling.py 	Replace markdown/filter imports with typed impls; add robust sanitization and _safe_get_markdown, best_content selection and length/object checks, set PageContent.markdown to string, compute metadata.word_count from sanitized content; change extraction_strategy to `str
RAG processing, deduplication & cache
crawler_mcp/core/rag.py 	Move deduplication to per-subchunk granularity with deterministic subchunk IDs and content_hashes, legacy-upgrade logic, orphaned-chunk deletion, richer chunk metadata (sub_chunk_index, start_pos, end_pos), improved error handling, QueryCache typed to RagResult, and new parallel _process_embeddings_pipeline; return shape includes chunks_stored.
Vector search params
crawler_mcp/core/vectors.py 	Accept typed SearchParams (hnsw_ef) instead of raw dict for dynamic EF configuration in search_similar.
Embedding throughput logging
crawler_mcp/core/embeddings.py 	Add batch timing instrumentation and embeddings-per-second throughput logging for true-batch and batched generators.
Browser & file pools (high-performance)
crawler_mcp/core/browser_pool.py, crawler_mcp/crawlers/directory.py 	New HighPerformanceBrowserPool for pooled AsyncWebCrawler instances; DirectoryCrawlStrategy switched to thread-pooled high-concurrency file processing (sync per-file workers via ThreadPoolExecutor) with post-processing memory checks.
Middleware typing
crawler_mcp/middleware/progress.py 	Refine cleanup task typing to `asyncio.Task[None]
Server warning suppression
crawler_mcp/server.py 	Add warnings.filterwarnings to ignore BeautifulSoup 'text' argument DeprecationWarning patterns.
Dev environment / compose
dev.sh (deleted), docker-compose.yml 	Remove dev.sh; change qdrant host ports to 7000:6333, 7001:6334; adjust text-embeddings-inference startup args and environment (batch sizes, workers, PYTORCH_CUDA_ALLOC_CONF).
Project config & gitignore
pyproject.toml, .gitignore 	Add coverage dependency, stricter mypy and pytest configs (with overrides for crawl4ai); update test tooling groups; add .serena/ ignore rule.
Configuration additions
crawler_mcp/config.py, .env.example 	Add many new CrawlerrSettings fields (qdrant pool/batch/prefetch, embedding_workers, file/thread/pool/concurrency/gpu/cache knobs, arun_many toggle, lxml strategy toggle); add EMBEDDING_WORKERS to .env.example.
Docs
docs/middleware.md, docs/testing.md 	Add middleware guide (hooks, lifecycle, examples) and testing guide (in-memory and deployed testing patterns).
Tests — additions & fixtures
tests/* (many new files: conftest.py, test_*.py, README.md, __init__.py) 	Add extensive unit/integration tests, fixtures, in-memory FastMCP test server/client, and test guidance covering config, crawlers, core services (embeddings/vectors/rag), RAG tools, server, middleware, models, sources, and end-to-end flows.
Tests — removals
tests/test_backwards_compatibility.py, tests/test_deduplication.py, tests/test_deduplication_performance.py, tests/test_migration.py 	Remove several prior deduplication/backwards-compatibility/migration/performance test modules.
Sequence Diagram(s)
Estimated code review effort

🎯 5 (Critical) | ⏱️ ~120 minutes
Possibly related PRs

feat: implement fit markdown extraction for clean content #2 — Overlaps crawling/markdown extraction and RAG deduplication changes (best_content/pruning/content-filter imports and usage).

    🚀 Qwen3 Reranker Integration & Token-Aware Chunking Optimization #1 — Related to RAG/chunking, reranking/tokenization, and deduplication helpers; touches similar regions in RagService.

Poem

    A rabbit nibbled through the docs and code,
    Tuned chunk IDs down a tidy road.
    It cleaned the markdown, split each chunk,
    Pooled fast browsers and threaded each trunk.
    Ports rebound and tests come spring—hop, deploy, behold! 🐇✨

Tip
🔌 Remote MCP (Model Context Protocol) integration is now available!

✨ Finishing Touches

Thanks for using CodeRabbit! It's free for OSS, and your support helps us grow. If you like it, consider giving us a shout-out.
❤️ Share

🪧 Tips

Copilot
Copilot AI reviewed Aug 20, 2025
Copilot AI left a comment
Pull Request Overview

This PR introduces comprehensive test coverage for the crawler MCP server to achieve high code coverage and ensure robust functionality. The tests follow FastMCP best practices with in-memory testing, direct service integration, and behavior-focused testing patterns.

Key changes:

    Adds comprehensive test suites for all major components (RAG, vector services, crawling, sources)
    Implements edge case testing with import error handling and failover mechanisms
    Creates integration tests for complete end-to-end workflows

Reviewed Changes

Copilot reviewed 39 out of 41 changed files in this pull request and generated 4 comments.
Show a summary per file













Comments suppressed due to low confidence (1)

Tip: Customize your code reviews with copilot-instructions.md. Create the file or learn how to get started.
tests/test_vectors_comprehensive.py
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content",
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
Copilot AI
Aug 20, 2025

Consider using a constant for the embedding dimension (384) instead of magic numbers. This makes the test more maintainable and clearer about the expected vector size.
Suggested change
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
                        embedding=[0.1, 0.2, 0.3] * (EMBEDDING_DIM // 3),  # 384 dimensions

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_rag_edge_cases.py
                raise ImportError("tiktoken not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
Copilot AI
Aug 20, 2025

The import mocking pattern is repeated multiple times. Consider extracting this into a reusable test fixture or helper function to reduce code duplication.
Suggested change
        with patch("builtins.__import__", side_effect=mock_import):
        # Patch import to raise ImportError for tiktoken
        with import_blocker("tiktoken"):

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_rag_comprehensive.py
        assert result == mock_result

        # Wait for expiration and test miss
        time.sleep(1)
Copilot AI
Aug 20, 2025

Using time.sleep() in async tests can make tests slower and less reliable. Consider using asyncio.sleep() or mocking time-dependent operations instead.
Suggested change
        time.sleep(1)
        await asyncio.sleep(1)

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_integration_flow.py
        # Step 2: Verify content was indexed by checking stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data

Copilot AI
Aug 20, 2025

The hard-coded sleep delays in integration tests make them slower and potentially flaky. Consider implementing proper wait conditions or using event-driven synchronization instead.
Suggested change
        # Wait for indexing to complete by polling stats
        async def wait_for_indexing(client, min_sources=1, min_chunks=1, timeout=15, poll_interval=0.5):
            start = asyncio.get_event_loop().time()
            while True:
                stats_result = await client.call_tool("get_rag_stats", {})
                stats = stats_result.data
                if stats["total_sources"] >= min_sources and stats["total_chunks"] >= min_chunks:
                    return stats
                if asyncio.get_event_loop().time() - start > timeout:
                    raise TimeoutError("Indexing did not complete in time")
                await asyncio.sleep(poll_interval)
        stats = await wait_for_indexing(mcp_client, min_sources=1, min_chunks=1)

Copilot uses AI. Check for mistakes.
@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Actionable comments posted: 101

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (31)











.gitignore
@@ -142,3 +142,4 @@ ipython_config.py
*.key
*.crt
*.p12
.serena/
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Add context and mirror ignore where needed (.dockerignore, packaging).

The new rule looks fine. Two quick checks:

    If you intend to ignore only the repo-root directory, prefer "/.serena/". If any nested ".serena/" folders should be ignored anywhere, current form is correct.
    If ".serena/" may contain large artifacts or secrets, mirror this in .dockerignore (to keep it out of Docker build context) and in packaging config (e.g., MANIFEST.in) if you publish wheels/sdists.

Optional tweak if root-only is desired:

-.serena/
+/.serena/

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +786 to +795
                # Split page content into optimal chunks for Qwen3-Embedding-0.6B
                text_chunks = self._chunk_text_character_based(
                    page.content,
                    metadata={
                        "source_url": page.url,
                        "source_title": page.title,
                        "page_index": i,
                        **page.metadata.get("chunk_metadata", {}),
                    },
                )
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Prefer token-based chunking when tokenizer is available

The pipeline unconditionally uses character-based chunking even when a tokenizer is initialized. This loses boundary quality and can increase fragmentation vs. token-aware chunking.

Switch to token-based when self.tokenizer is set, with character-based as fallback.

Apply:

-                # Split page content into optimal chunks for Qwen3-Embedding-0.6B
-                text_chunks = self._chunk_text_character_based(
+                # Split page content into optimal chunks
+                # Prefer token-based when tokenizer is available; fallback to character-based.
+                chunk_fn = (
+                    self._chunk_text_token_based if self.tokenizer else self._chunk_text_character_based
+                )
+                text_chunks = chunk_fn(
                     page.content,
                     metadata={
                         "source_url": page.url,
                         "source_title": page.title,
                         "page_index": i,
                         **page.metadata.get("chunk_metadata", {}),
                     },
                 )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +895 to +901
                        metadata={
                            **page.metadata,
                            "sub_chunk_index": sub_chunk_idx,
                            "page_index": i,
                            "start_pos": chunk_data["start_pos"],
                            "end_pos": chunk_data["end_pos"],
                        },
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Preserve user-provided per-chunk metadata

You accept page.metadata.get("chunk_metadata", {}) during chunking but don’t include it in the stored chunk metadata. This can drop important, caller-supplied annotations at write time.

Fold the original chunk_metadata back into the payload.

Apply:

                         metadata={
                             **page.metadata,
+                            "chunk_metadata": page.metadata.get("chunk_metadata", {}),
                             "sub_chunk_index": sub_chunk_idx,
                             "page_index": i,
                             "start_pos": chunk_data["start_pos"],
                             "end_pos": chunk_data["end_pos"],
                         },

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/vectors.py
Comment on lines +349 to 352
                search_params=SearchParams(
                    hnsw_ef=ef_value
                ),  # Dynamic ef for optimal speed/accuracy
            )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Per-query hnsw_ef is correct; consider allowing caller override.

Current heuristic ef = min(256, max(64, limit*4)) is sensible. To improve control, allow an optional ef_override param (or part of a typed options object) to override the heuristic when needed.

Example signature tweak:

-    async def search_similar(
+    async def search_similar(
         self,
         query_vector: list[float],
         limit: int = 10,
         score_threshold: float = 0.0,
         source_filter: list[str] | None = None,
         date_range: tuple[datetime, datetime] | None = None,
+        ef_override: int | None = None,
     ) -> list[SearchMatch]:
@@
-            ef_value = min(256, max(64, limit * 4))  # 4x limit, capped at 256
+            ef_value = ef_override if ef_override is not None else min(256, max(64, limit * 4))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/server.py
Comment on lines +19 to +23
warnings.filterwarnings(
    "ignore",
    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
    category=DeprecationWarning,
)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Scope the deprecation warning filter to BeautifulSoup module

The message matches bs4’s deprecation on text=. Constrain by module to avoid accidentally suppressing unrelated libraries’ warnings.

Apply:

 warnings.filterwarnings(
     "ignore",
-    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
-    category=DeprecationWarning,
+    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
+    category=DeprecationWarning,
+    module="bs4\\..*",
 )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/tools/crawling.py
Comment on lines +242 to 243
        extraction_strategy: str | None = None,
        wait_for: str | None = None,
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Refactor Required: Honor extraction_strategy and wait_for in the crawler implementation

While both the FastMCP tool (scrape) and CrawlerService.scrape_single_page signatures correctly accept extraction_strategy: str | None = None and wait_for: str | None = None, the current implementation never actually uses those values. To complete this API improvement, you should:

• In crawler_mcp/core/orchestrator.py (method scrape_single_page, ~lines 212–260):
– Before calling browser.arun(…), inject logic to
• pass wait_for as a selector or JS condition (e.g. via await browser.wait_for(wait_for) or a wait_for argument to arun).
• apply extraction_strategy (e.g. map it to an extraction_method parameter on arun, or merge into custom_config).

• In crawler_mcp/tools/crawling.py (the FastMCP tool scrape, ~lines 242–260):
– Ensure the docstring reflects the behavior when extraction_strategy=None or wait_for=None.
– Confirm that what you pass into crawler_service.scrape_single_page(...) is actually consumed downstream.

Until these parameters are surfaced to the underlying AsyncWebCrawler, the default strategy will always apply and any supplied string will be ignored. Please implement the missing pass-through and handling logic so that None truly invokes the default and non-None values are respected.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/types/crawl4ai_types.py
Comment on lines +216 to +231
# Updated CrawlerRunConfig with better typing
class CrawlerRunConfigAdvanced(Protocol):
    """Protocol for crawl4ai CrawlerRunConfig."""

    def __init__(
        self,
        deep_crawl_strategy: DeepCrawlStrategy | None = None,
        stream: bool = False,
        cache_mode: CacheMode | str | None = None,
        page_timeout: float | None = None,
        markdown_generator: MarkdownGenerationStrategy | None = None,
        content_filter: ContentFilterStrategy | None = None,
        extraction_strategy: ExtractionStrategy | None = None,
        **kwargs: Any,
    ) -> None: ...

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

You introduced CrawlerRunConfigAdvanced but didn’t export it

If downstream modules import from this types layer, they won’t see CrawlerRunConfigAdvanced via __all__. Export it for consistency with other protocol exports.

Apply:

 __all__ = [
@@
-    "CrawlerRunConfig",
+    "CrawlerRunConfig",
+    "CrawlerRunConfigAdvanced",

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/types/crawl4ai_types.py
Comment on lines +236 to +259
# For actual runtime usage, we just cast the imports to our protocols
# This maintains type safety while allowing runtime flexibility
try:
    from crawl4ai.content_filter_strategy import (
        PruningContentFilter as _PruningContentFilter,  # type: ignore
    )
    from crawl4ai.markdown_generation_strategy import (
        DefaultMarkdownGenerator as _DefaultMarkdownGenerator,  # type: ignore
    )
    from crawl4ai.models import (
        MarkdownGenerationResult as _MarkdownGenerationResult,  # type: ignore
    )

    # Type-safe aliases that satisfy our protocols
    DefaultMarkdownGeneratorImpl = _DefaultMarkdownGenerator
    PruningContentFilterImpl = _PruningContentFilter
    MarkdownGenerationResultImpl = _MarkdownGenerationResult

except ImportError:
    # Fallback for when crawl4ai is not available
    DefaultMarkdownGeneratorImpl = None  # type: ignore
    PruningContentFilterImpl = None  # type: ignore
    MarkdownGenerationResultImpl = None  # type: ignore

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Handle missing crawl4ai implementations at runtime

The aliases DefaultMarkdownGeneratorImpl, PruningContentFilterImpl, and MarkdownGenerationResultImpl may be set to None when the crawl4ai package isn’t installed. Instantiating them unguarded will raise
TypeError: 'NoneType' object is not callable.
Please add explicit checks or fallbacks at each call site.

• In crawler_mcp/crawlers/web.py:688

markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

– guard before calling, e.g.:

if DefaultMarkdownGeneratorImpl is None:
    raise ImportError("crawl4ai package is required for markdown generation")
markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

• Any other direct uses (e.g. orchestrator imports) should similarly verify the alias isn’t None before instantiation or method calls.

For stricter safety, consider exporting stub classes in crawl4ai_types.py that raise a clear ImportError on construction instead of defaulting to None.
🤖 Prompt for AI Agents

@jmagar
docker-compose.yml
Comment on lines +6 to +7
      - "7000:6333"
      - "7001:6334"  # gRPC port
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Mandatory: Parameterize and bind Qdrant host ports consistently across the codebase

Changing the host ports from 6333/6334 to 7000/7001 without updating every reference will break existing clients, tests, CI workflows, and documentation. To contain exposure, bind only to the loopback interface and make the host ports configurable via environment variables.

Please update the following locations:

    docker-compose.yml (root)
        Lines 6–7: replace hard-coded mappings
        Lines 12–13: keep container ports at 6333/6334 but expose via ${QDRANT_HTTP_PORT_HOST}/${QDRANT_GRPC_PORT_HOST}
    .env (root)
        Update QDRANT_URL default from port 6333 to use ${QDRANT_HTTP_PORT_HOST:-7000}
    Configuration model (e.g., src/config.py)
        Field alias QDRANT_URL default should reflect the new host port
    Tests (e.g., tests/test_config.py, around the assertion on line 34)
        Change assert config.qdrant_url == "http://localhost:6333" to use the parameterized port
    CI/workflow scripts (e.g., .github/workflows/*.yml, around line 160)
        Update health-check URLs in curl -f http://localhost:6333/health to use ${QDRANT_HTTP_PORT_HOST}
    README.md and any documentation snippets
        Replace examples pointing at port 6333/6334 with the new defaults or environment-driven values
    Additional Docker‐Compose profiles (lines 235, 311 in the same file)
        Align any other Qdrant service sections to use the same env-driven port mappings and 127.0.0.1 binding

Suggested refactor diff in docker-compose.yml:

-      - "7000:6333"
-      - "7001:6334"  # gRPC port
+      - "127.0.0.1:${QDRANT_HTTP_PORT_HOST:-7000}:6333"
+      - "127.0.0.1:${QDRANT_GRPC_PORT_HOST:-7001}:6334"  # gRPC port

And in .env:

- QDRANT_URL=http://localhost:6333
+ QDRANT_URL=http://localhost:${QDRANT_HTTP_PORT_HOST:-7000}

🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +1 to +16
# MCP Middleware

> Add cross-cutting functionality to your MCP server with middleware that can inspect, modify, and respond to all MCP requests and responses.
export const VersionBadge = ({version}) => {
  return <code className="version-badge-container">
            <p className="version-badge">
                <span className="version-badge-label">New in version:</span>
                <code className="version-badge-version">{version}</code>
            </p>
        </code>;
};

<VersionBadge version="2.9.0" />

MCP middleware is a powerful concept that allows you to add cross-cutting functionality to your FastMCP server. Unlike traditional web middleware, MCP middleware is designed specifically for the Model Context Protocol, providing hooks for different types of MCP operations like tool calls, resource reads, and prompt requests.
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

MDX components in .md: convert to .mdx or use admonitions to avoid lint/build issues

This file uses React/MDX components (VersionBadge, Tip, Warning) and inline HTML in a .md file. This will trigger MD033 and may fail non-MDX doc builds.

Options:

    Rename to docs/middleware.mdx and ensure the docs pipeline supports MDX.
    Or replace components with admonitions:

-<Tip>
-  ...
-</Tip>
+:::tip
+...
+:::

-<Warning>
-  ...
-</Warning>
+:::warning
+...
+:::

As a quick stopgap, add a markdownlint suppression at the top:

+# markdownlint-disable MD033

🧰 Tools
















🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +220 to +234
```python
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class AuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name

        # Deny access to restricted tools
        if tool_name.lower() in ["delete", "admin_config"]:
            raise ToolError("Access denied: tool requires admin privileges")

        # Allow other tools to proceed
        return await call_next(context)
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Unify error type usage: prefer ToolError for tool denial

Earlier sections correctly recommend ToolError for access denial. This section introduces McpError/ErrorData. For consistency and to avoid leaking protocol-layer types, prefer ToolError here as well.

-from mcp import McpError
-from mcp.types import ErrorData
+from fastmcp.exceptions import ToolError
@@
-        if len(self.client_requests[client_id]) >= self.requests_per_minute:
-            raise McpError(ErrorData(code=-32000, message="Rate limit exceeded"))
+        if len(self.client_requests[client_id]) >= self.requests_per_minute:
+            raise ToolError("Rate limit exceeded")

If you intend to show both patterns, clarify when protocol errors are appropriate.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +315 to +323
<VersionBadge version="2.11.0" />

In addition to modifying the request and response, you can also store state data that your tools can (optionally) access later. To do so, use the FastMCP Context to either `set_state` or `get_state` as appropriate. For more information, see the [Context State Management](/servers/context#state-management) docs.

## Creating Middleware

FastMCP middleware is implemented by subclassing the `Middleware` base class and overriding the hooks you need. You only need to implement the hooks that are relevant to your use case.

```python
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

VersionBadge render note

If you keep VersionBadge, document that it is a site component and include its implementation in the MDX runtime, or replace with plain text to avoid broken renders.

Replace with inline text:

-<VersionBadge version="2.11.0" />
+New in version 2.11.0

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
docs/testing.md
Comment on lines +13 to +40
```python
from fastmcp import FastMCP, Client

# Create your server
server = FastMCP("WeatherServer")

@server.tool
def get_temperature(city: str) -> dict:
    """Get current temperature for a city"""
    temps = {"NYC": 72, "LA": 85, "Chicago": 68}
    return {"city": city, "temp": temps.get(city, 70)}

@server.resource("weather://forecast")
def get_forecast() -> dict:
    """Get 5-day forecast"""
    return {"days": 5, "conditions": "sunny"}

async def test_weather_operations():
    # Pass server directly - no deployment needed
    async with Client(server) as client:
        # Test tool execution
        result = await client.call_tool("get_temperature", {"city": "NYC"})
        assert result.data == {"city": "NYC", "temp": 72}

        # Test resource retrieval
        forecast = await client.read_resource("weather://forecast")
        assert forecast.contents[0].data == {"days": 5, "conditions": "sunny"}
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Example correctness: Client(server) pattern is good; add timeout to avoid hanging in examples

The in-memory example is solid and aligns with our guideline to pass the server instance directly. To keep samples robust for readers who will copy/paste, consider wrapping networked awaits with a timeout (esp. in the HTTP section) to avoid hanging tests.

Apply this minimal change to the HTTP example:

 from fastmcp import Client
+import asyncio

 async def test_deployed_server():
     # Connect to a running server
-    async with Client("http://localhost:8000/mcp/") as client:
-        await client.ping()
+    async with Client("http://localhost:8000/mcp/") as client:
+        await asyncio.wait_for(client.ping(), timeout=5)

🤖 Prompt for AI Agents

@jmagar
docs/testing.md
Comment on lines +112 to +146
```python
from fastmcp import Client

async def test_deployed_server():
    # Connect to a running server
    async with Client("http://localhost:8000/mcp/") as client:
        await client.ping()

        # Test with real network transport
        tools = await client.list_tools()
        assert len(tools) > 0

        result = await client.call_tool("greet", {"name": "World"})
        assert "Hello" in result.data
```

### Testing Authentication

The FastMCP Client handles authentication transparently, making it easy to test secured servers:

```python
async def test_authenticated_server():
    # Bearer token authentication
    async with Client(
        "https://api.example.com/mcp",
        headers={"Authorization": "Bearer test-token"}
    ) as client:
        await client.ping()
        tools = await client.list_tools()

    # OAuth flow (opens browser for authorization)
    async with Client("https://api.example.com/mcp", auth="oauth") as client:
        result = await client.call_tool("protected_tool", {})
        assert result.data is not None
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Prefer pytest markers for deployed-server tests in docs

These examples hit a running server. Suggest showing pytest markers so readers can categorize/skip them in CI.

Add a marker in the snippet header:

-from fastmcp import Client
+import pytest
+from fastmcp import Client
@@
-async def test_deployed_server():
+@pytest.mark.integration
+@pytest.mark.requires_services
+async def test_deployed_server():

🧰 Tools



🤖 Prompt for AI Agents

@jmagar
pyproject.toml
Comment on lines +50 to 52
    "torch>=2.8.0",
    "coverage>=7.10.3",
]
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Move coverage to test dependencies, not runtime

coverage should not be a runtime dependency of the library/server.

-    "torch>=2.8.0",
-    "coverage>=7.10.3",
+    "torch>=2.8.0",

Then add it to the test group below (see next comment).
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +24 to +36
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

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Add a short note about uv (or alternative) in Prerequisites to avoid confusion.

All commands use “uv run …”. Clarify how to install/use uv or how to substitute with Python/pip so new contributors aren’t blocked.

Proposed addition:

 ## Running Tests

 ### Prerequisites

+0. **Tooling**
+   - Ensure Python 3.11+ is available.
+   - Ensure `uv` is installed (or replace `uv run pytest` with `python -m pytest` in the commands below).
+
 1. **Start Required Services**:

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +26 to +35
1. **Start Required Services**:
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services
   ```

2. **Environment Variables**:
   Copy `tests/.env.test` to `.env` and adjust as needed:
   ```bash
   cp tests/.env.test .env
   ```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix markdownlint MD031: add blank lines around fenced code blocks in Prerequisites.

Current fences begin immediately after list items, triggering MD031. Insert blank lines before the opening and after the closing fences for both steps.

Apply this diff:

 1. **Start Required Services**:
+
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services

    Environment Variables:

    Copy tests/.env.test to .env and adjust as needed:

    Copy tests/.env.test to .env and adjust as needed:

    cp tests/.env.test .env


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~26-~26: Use correct spacing
Context: ...uisites  1. **Start Required Services**:    ```bash    docker-compose up -d  # Start Qdrant and TEI services    ```  2. **Environment Variables**:    Copy `tests/...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~31-~31: There might be a mistake here.
Context: ...es    ```  2. **Environment Variables**:    Copy `tests/.env.test` to `.env` and adj...

(QB_NEW_EN)

---

[grammar] ~32-~32: Use correct spacing
Context: ...nv.test` to `.env` and adjust as needed:    ```bash    cp tests/.env.test .env    ```  ### Test Commands  ```bash # Run all tests u...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

27-27: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

---

33-33: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 26 to 35, the fenced code blocks for the "Start
Required Services" and "Environment Variables" list items lack blank lines
before the opening bash and after the closing , causing markdownlint
MD031; fix by inserting a blank line immediately before each opening fenced code
block and a blank line immediately after each closing fence so each fenced block
is separated from the list item text.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +26 to +29
1. **Start Required Services**:
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services
   ```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Prefer “docker compose” over “docker-compose” in docs.

Modern Docker recommends the space-separated subcommand. Keep if your team standardizes on the hyphenated plugin, but consider updating for consistency with current docs.
🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +37 to +61
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

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Register custom pytest markers to avoid PytestUnknownMarkWarning.

You use custom markers (unit, integration, slow, requires_services). Add a pytest.ini snippet to document/avoid warnings.

Proposed snippet (documentation-only):

# pytest.ini (at repo root)
[pytest]
markers =
  unit: fast tests without external services
  integration: tests using real services
  slow: long-running tests
  requires_services: requires Qdrant and TEI

🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +39 to +60
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
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Minor: document combined marker usage and coverage targets.

Add examples that reflect typical usage patterns:

    Run integration tests that also require services: uv run pytest -m "integration and requires_services".
    Produce HTML/XML coverage for CI: uv run pytest --cov=crawler_mcp --cov-report=term-missing --cov-report=xml --cov-report=html.

🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +71 to +88
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

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Async test requires an async test runner; mark with anyio/asyncio or make the example sync.

As written, async def tests will require pytest-anyio (preferred with FastMCP) or pytest-asyncio. Add a marker to the example to prevent copy-paste failures.

Apply this diff:

 @pytest.mark.integration
 @pytest.mark.requires_services
+@pytest.mark.anyio
 async def test_example(mcp_client: Client):

Optionally document in README: “Install pytest-anyio and enable it via @pytest.mark.anyio or configure asyncio_mode in pytest.ini.”
📝 Committable suggestion


🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +100 to +106
### Service Management

- Tests use a separate Qdrant collection (`test_crawler_mcp`)
- Collection is cleaned before each test requiring it
- Services are checked for availability before running service-dependent tests
- Tests are skipped if required services are not available

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Document isolation strategy for parallel tests with a shared Qdrant instance.

Parallel runs (xdist) can collide on a single collection. Either use a per-test/worker suffix or a unique temporary collection per test in fixtures.

Suggested doc note:

    Use a unique collection name per test/worker (e.g., f"test_crawler_mcp_{request.node.nodeid_hash}" or append PYTEST_XDIST_WORKER) and drop it in teardown to avoid cross-test interference.

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +120 to +126
### Logging

Adjust log levels in `.env` for more detailed output:
```env
LOG_LEVEL=DEBUG  # For detailed logging
LOG_TO_FILE=true  # To capture logs in files
```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix MD031 and minor wording in Logging section.

Add blank lines around the env fenced block to satisfy MD031 and improve readability.

Apply this diff:

 ### Logging

 Adjust log levels in `.env` for more detailed output:
+
 ```env
 LOG_LEVEL=DEBUG  # For detailed logging
 LOG_TO_FILE=true  # To capture logs in files


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~120-~120: Use correct spacing
Context: ...reakpoint()  # And here ```  ### Logging  Adjust log levels in `.env` for more det...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~122-~122: Use correct spacing
Context: ...vels in `.env` for more detailed output: ```env LOG_LEVEL=DEBUG  # For detailed logging LOG_TO_FILE=true  # To capture logs in files ```  ### Test Data Inspection  Access services di...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

123-123: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 120 to 126, the fenced env code block needs
blank lines before the opening env and after the closing to satisfy MD031
and improve readability; edit the Logging section to insert one blank line
immediately above the env line and one blank line immediately below the closing and keep the env content unchanged (LOG_LEVEL=DEBUG and
LOG_TO_FILE=true), also tidy the surrounding sentence wording if necessary for
clarity.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +128 to +135
### Test Data Inspection

Access services directly in tests:
```python
async def test_inspect_data(vector_service: VectorService):
    info = await vector_service.get_collection_info()
    print(f"Collection has {info['points_count']} points")
```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix MD031 around the “Test Data Inspection” example.

Add blank lines around the code fence.

Apply this diff:

 Access services directly in tests:
+
 ```python
 async def test_inspect_data(vector_service: VectorService):
     info = await vector_service.get_collection_info()
     print(f"Collection has {info['points_count']} points")


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~128-~128: Use correct spacing
Context: ...s in files ```  ### Test Data Inspection  Access services directly in tests: ```py...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~130-~130: Use correct spacing
Context: ...tion  Access services directly in tests: ```python async def test_inspect_data(vector_service: VectorService):     info = await vector_service.get_collection_info()     print(f"Collection has {info['points_count']} points") ```  ## Performance Considerations  - **Fast Exe...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

131-131: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 128 to 135, the "Test Data Inspection" example
violates MD031 by lacking blank lines before and after the fenced code block;
fix it by adding a blank line immediately before the opening and a blank line immediately after the closing so the code fence is separated from
surrounding text, preserving the existing code content and spacing inside the
fence.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +139 to +143
- **Fast Execution**: In-memory tests run in milliseconds
- **Service Startup**: Allow time for Qdrant/TEI to be fully ready
- **Resource Cleanup**: Tests clean up after themselves
- **Parallel Execution**: Tests can run in parallel (use `pytest-xdist`)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Set expectations for service readiness to reduce flakes.

Add recommended readiness checks/timeouts for Qdrant/TEI in local runs, not only CI. Mention typical warm-up for TEI model load.

You already show health checks in CI; mirror those in a short “Before running integration tests locally” snippet.
🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +154 to +165
```yaml
- name: Start services
  run: docker-compose up -d

- name: Wait for services
  run: |
    timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'
    timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'
- name: Run tests
  run: uv run pytest -m "not slow"
```
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Update README example: surround YAML block with blank lines and correct Qdrant port

Please update the tests/README.md example to both satisfy MD031 and align the Qdrant health-check port with your docker-compose.yml mapping:

    Add a blank line before and after the fenced yaml block.
    Change the Qdrant health-check from port 6333 (container port) to 7000 (host port mapped in docker-compose.yml), so the example actually reaches the service.

Suggested diff:

 Example GitHub Actions:

+
 ```yaml
 - name: Start services
   run: docker-compose up -d

 - name: Wait for services
   run: |
-    timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'
+    timeout 60 bash -c 'until curl -f http://localhost:7000/health; do sleep 2; done'
     timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'

 - name: Run tests
   run: uv run pytest -m "not slow"


Tagging this as a critical fix to avoid broken health checks in CI.

<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

154-154: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 154 to 165, the YAML example block needs a blank
line before and after the fenced yaml to satisfy MD031, and the Qdrant
health-check URL should use host port 7000 (mapped in docker-compose.yml)
instead of 6333 so CI contacts the service correctly; update the example by
adding an empty line above the opening yaml fence and below the closing
fence, and change the curl URL from http://localhost:6333/health to
http://localhost:7000/health in the "Wait for services" step.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/test_config.py
Comment on lines +22 to +53
        # Test that values are loaded from .env file (not hardcoded defaults)
        assert config.server_host == "0.0.0.0"  # From .env SERVER_HOST
        assert config.server_port == 8010  # From .env SERVER_PORT
        assert config.debug is True  # From .env DEBUG=true
        assert config.production is False  # From .env PRODUCTION=false

        # Logging configuration from .env
        assert config.log_level == "INFO"  # From .env LOG_LEVEL
        assert config.log_format == "console"  # From .env LOG_FORMAT
        assert config.log_to_file is True  # From .env LOG_TO_FILE=true

        # Service endpoints from .env
        assert config.qdrant_url == "http://localhost:6333"  # From .env QDRANT_URL
        assert config.qdrant_collection == "crawlerr_documents"  # From .env
        assert config.qdrant_vector_size == 1024  # From .env
        assert config.qdrant_distance == "cosine"  # From .env

        # TEI configuration from .env
        assert config.tei_url == "http://localhost:8080"  # From .env TEI_URL
        assert config.tei_model == "Qwen/Qwen3-Embedding-0.6B"  # From .env TEI_MODEL
        assert config.tei_batch_size == 64  # From .env TEI_BATCH_SIZE

        # Crawling settings from .env
        assert config.chunk_size == 1024  # From .env CHUNK_SIZE
        assert config.chunk_overlap == 200  # From .env CHUNK_OVERLAP
        assert (
            config.reranker_model == "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
        )  # From .env
        assert (
            config.deduplication_strategy == "content_hash"
        )  # From .env DEDUPLICATION_STRATEGY

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Tests rely on a .env in the repo; make them hermetic with monkeypatch

Hardcoding expectations from .env makes tests environment-dependent and flaky in CI. Use monkeypatch to set env vars or pass values directly to CrawlerrSettings in each test.

Example change:

-        config = CrawlerrSettings()
+        monkeypatch = pytest.MonkeyPatch()
+        monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
+        monkeypatch.setenv("SERVER_PORT", "8010")
+        monkeypatch.setenv("DEBUG", "true")
+        monkeypatch.setenv("PRODUCTION", "false")
+        try:
+            config = CrawlerrSettings()
+        finally:
+            monkeypatch.undo()

Repeat pattern where specific .env-driven assertions are made. Alternatively, assert defaults without attributing them to .env.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_config.py
Comment on lines +146 to +157
    @pytest.mark.unit
    def test_log_file_directory_creation(self):
        """Test that log file directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "test.log"

            config = CrawlerrSettings(log_file=str(log_path), log_to_file=True)

            # Directory should be created
            assert log_path.parent.exists()
            assert config.log_file == str(log_path)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Log directory creation: also assert file path normalization

Good coverage for directory creation. Consider also asserting that absolute paths are preserved and that expanduser (~) is handled.

-            assert config.log_file == str(log_path)
+            assert Path(config.log_file) == log_path

🤖 Prompt for AI Agents

@jmagar
tests/test_config.py
Comment on lines +297 to +313
    def test_field_constraints(self):
        """Test field constraints work correctly."""
        # Test max_pages constraints
        config = CrawlerrSettings(crawl_max_pages=1)  # Minimum value
        assert config.crawl_max_pages == 1

        config = CrawlerrSettings(crawl_max_pages=1000)  # Maximum value
        assert config.crawl_max_pages == 1000

        # Invalid: below minimum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=1001)

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Enforce min/max constraints for crawl_max_pages
The CrawlerrSettings model currently defines crawl_max_pages without any ge/le bounds, but the tests in tests/test_config.py (lines 297–313) expect values below 1 and above 1000 to raise a ValueError. Add the missing constraints so that the implementation matches the tests:

• File: crawler_mcp/config.py (around line 92)

-    crawl_max_pages: int = Field(default=1000, alias="CRAWL_MAX_PAGES")
+    crawl_max_pages: int = Field(
+        default=1000,
+        alias="CRAWL_MAX_PAGES",
+        ge=1,
+        le=1000,
+    )

This change enforces the minimum of 1 and maximum of 1000, ensuring the existing tests pass without modification.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_edge_cases.py
Comment on lines +291 to +294
            # The implementation creates adjacent chunks (no gaps, but also no overlap)
            # when the calculated overlap would cause the next chunk to start before current end
            assert next_start <= current_end  # No gaps between chunks

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Tighten overlap assertion to match docstring intent

Comment says “no gaps, but also no overlap,” yet the assertion allows overlap (<=). Use equality to assert adjacency.

Apply:

-            assert next_start <= current_end  # No gaps between chunks
+            assert next_start == current_end  # Adjacent chunks: no gaps, no overlap

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_edge_cases.py
Comment on lines +686 to +691
        # Process crawl result (should detect and delete orphans)
        result = await service.process_crawl_result(
            crawl_result, progress_callback=progress_callback
        )

        # Should have called delete_chunks_by_ids
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused local variable result (ruff F841)

The test doesn’t assert on the returned value. Drop the assignment to avoid the lint violation.

Apply:

-        result = await service.process_crawl_result(
-            crawl_result, progress_callback=progress_callback
-        )
+        await service.process_crawl_result(
+            crawl_result, progress_callback=progress_callback
+        )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +23 to +36
@pytest.fixture
async def rag_service():
    """Create a real RagService instance with live services."""
    service = RagService()
    async with service:
        yield service


@pytest.fixture
async def test_client():
    """Create FastMCP test client with live services."""
    async with Client(mcp) as client:
        yield client

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Fixture duplication with conftest; prefer shared fixtures or unique names

This fixture shadows a likely identical rag_service fixture in tests/conftest.py. Prefer reusing the shared fixture to avoid divergent configuration or rename to make intent explicit (e.g., rag_service_live).

-@pytest.fixture
-async def rag_service():
+@pytest.fixture(name="rag_service_live")
+async def rag_service_live():
     """Create a real RagService instance with live services."""
     service = RagService()
     async with service:
         yield service

Then adjust test signatures accordingly. Alternatively, delete this fixture and import the one from conftest.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +41 to +47
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self, rag_service):
        """Test RAG service initializes properly with live services."""
        assert rag_service.vector_service is not None
        assert rag_service.embedding_service is not None
        assert rag_service._context_count > 0

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Avoid asserting on private attributes

Asserting on _context_count couples tests to internals and may break with harmless refactors.

Remove Line 46 and rely on public behaviors (health_check, context manager entry/exit tests already cover this).

-        assert rag_service._context_count > 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +112 to +118
        # Query for the content
        query = RagQuery(
            query="FastMCP testing capabilities",
            sources=["https://example.com/query-test"],
            limit=5,
            score_threshold=0.1,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix RagQuery field names: use source_filters and min_score

The RagQuery model defines source_filters and min_score, not sources and score_threshold.

Apply:

-        query = RagQuery(
-            query="FastMCP testing capabilities",
-            sources=["https://example.com/query-test"],
-            limit=5,
-            score_threshold=0.1,
-        )
+        query = RagQuery(
+            query="FastMCP testing capabilities",
+            source_filters=["https://example.com/query-test"],
+            limit=5,
+            min_score=0.1,
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +154 to +161
        deleted_count = await rag_service.delete_source(
            "https://example.com/delete-test"
        )

        # Verify deletion
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

delete_source returns bool, not int

According to RagService.delete_source, the return type is bool (True if any documents were deleted). The current assertions expect int and will misinterpret the result.

Use boolean semantics:

-        deleted_count = await rag_service.delete_source(
+        deleted = await rag_service.delete_source(
             "https://example.com/delete-test"
         )

-        # Verify deletion
-        assert isinstance(deleted_count, int)
-        assert deleted_count >= 0
+        # Verify deletion
+        assert isinstance(deleted, bool)

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +257 to +314
    @pytest.mark.asyncio
    async def test_rag_query_tool_integration(self, test_client):
        """Test rag_query tool with live services."""
        # First add some content using scrape tool
        scrape_result = await test_client.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Verify scrape worked
        assert isinstance(scrape_result.data, dict)

        # Query the scraped content
        query_result = await test_client.call_tool(
            "rag_query", {"query": "example content", "limit": 5, "min_score": 0.0}
        )

        # Verify query results
        assert isinstance(query_result.data, dict)
        assert "query" in query_result.data
        assert "matches" in query_result.data
        assert "total_matches" in query_result.data

    @pytest.mark.asyncio
    async def test_list_sources_integration(self, test_client):
        """Test list_sources tool with live services."""
        result = await test_client.call_tool("list_sources", {})

        assert isinstance(result.data, dict)
        assert "sources" in result.data
        assert "pagination" in result.data
        assert isinstance(result.data["sources"], list)

    @pytest.mark.asyncio
    async def test_get_rag_stats_integration(self, test_client):
        """Test get_rag_stats tool with live services."""
        result = await test_client.call_tool("get_rag_stats", {})

        assert isinstance(result.data, dict)
        assert "rag_system" in result.data or "health_summary" in result.data

    @pytest.mark.asyncio
    async def test_delete_source_tool_integration(self, test_client):
        """Test delete_source tool with live services."""
        # First add content to delete
        await test_client.call_tool(
            "scrape",
            {"url": "https://example.com/delete-integration", "process_with_rag": True},
        )

        # Delete the source
        result = await test_client.call_tool(
            "delete_source",
            {"source_url": "https://example.com/delete-integration", "confirm": True},
        )

        assert isinstance(result.data, dict)
        assert "success" in result.data or "documents_deleted" in result.data

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Mark MCP/live-service tests to be skippable in CI without services

These tests depend on running services and network; mark them to allow selective execution.

At module top add:

+pytestmark = [pytest.mark.integration, pytest.mark.requires_services]

And optionally skip when health checks fail:

if not (await rag_service.health_check()).get("vector_service", {}).get("ok"):
    pytest.skip("Vector service unavailable", allow_module_level=False)

@jmagar
tests/test_rag_integration.py
Comment on lines +401 to +404
        query = RagQuery(
            query="FastMCP testing practices", limit=3, score_threshold=0.0
        )

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix RagQuery param name in reranking test

Use min_score instead of score_threshold.

-        query = RagQuery(
-            query="FastMCP testing practices", limit=3, score_threshold=0.0
-        )
+        query = RagQuery(
+            query="FastMCP testing practices", limit=3, min_score=0.0
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +455 to +459
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for result in results:
            assert isinstance(result, dict)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Harden concurrent gather to surface exceptions explicitly

Return exceptions aids debugging and prevents cancellations masking an error.

-        results = await asyncio.gather(*tasks)
+        results = await asyncio.gather(*tasks, return_exceptions=True)
@@
-        for result in results:
-            assert isinstance(result, dict)
-            assert "chunks_created" in result
+        for result in results:
+            if isinstance(result, Exception):
+                raise result
+            assert isinstance(result, dict)
+            assert "chunks_created" in result

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +10 to +16
import pytest

from crawler_mcp.models.sources import (
    SourceInfo,
    SourceMetadata,
    SourceType,
)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Target Pydantic's ValidationError instead of broad exception tuples

Catching (ValueError, TypeError) is imprecise for Pydantic v2; use ValidationError.

@@
-import pytest
+import pytest
+from pydantic import ValidationError
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo()  # Missing required fields
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo(id="test")  # Missing url and source_type
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo(id="test", url="https://test.com")  # Missing source_type

Also applies to: 359-367, 362-367, 365-367
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +301 to +304
            total_content_length=15000,
            average_chunk_size=1000.0,
            metadata=complex_metadata,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" here

Avoid passing unsupported fields.

-            average_chunk_size=1000.0,

🤖 Prompt for AI Agents

@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Review continued from previous batch...
crawler_mcp/core/orchestrator.py
Comment on lines +296 to +342
            # Extract content using proper crawl4ai patterns
            # result.markdown is a MarkdownGenerationResult object with raw_markdown and fit_markdown attributes
            best_content = ""
            if result.markdown:
                try:
                    # Check if result.markdown is an integer (hash ID issue)
                    if isinstance(result.markdown, int):
                        self.logger.warning(
                            f"Detected integer markdown ({result.markdown}) for {url}, using empty content"
                        )
                        best_content = ""
                    else:
                        # First try fit_markdown (filtered content) if available
                        if (
                            hasattr(result.markdown, "fit_markdown")
                            and result.markdown.fit_markdown
                        ):
                            content = result.markdown.fit_markdown.strip()
                            if len(content) > 16:  # Avoid hash placeholders
                                best_content = content
                        # Fall back to raw_markdown (full content)
                        elif (
                            hasattr(result.markdown, "raw_markdown")
                            and result.markdown.raw_markdown
                        ):
                            content = result.markdown.raw_markdown.strip()
                            if len(content) > 16:  # Avoid hash placeholders
                                best_content = content
                        else:
                            best_content = ""
                except (AttributeError, TypeError) as e:
                    if "'int' object has no attribute" in str(e):
                        self.logger.warning(
                            f"Caught integer markdown issue for {url}, using empty content"
                        )
                        best_content = ""
                    else:
                        self.logger.warning(
                            f"Failed to extract markdown content for {url}: {e}"
                        )
                        best_content = ""
                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract markdown content for {url}: {e}"
                    )
                    best_content = ""

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Content extraction heuristics: guardrails are helpful but some thresholds are brittle

    Hard-coding a 16-character threshold can discard valid short pages/snippets.
    The “integer markdown” branch is good defensive coding; keep it.
    Consider preferring word-count thresholds aligned with settings.crawl_min_words and only then falling back to character count.

-                            if len(content) > 16:  # Avoid hash placeholders
-                                best_content = content
+                            words = content.split()
+                            if len(words) >= max(1, settings.crawl_min_words):
+                                best_content = content
+                            elif len(content) > 16:
+                                # last-resort guard to avoid hash placeholders
+                                best_content = content

Also consider logging when raw content is dropped for being “too short” to aid debugging with real sites.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/orchestrator.py
Comment on lines +345 to +364
                object_indicators = [
                    "CrawlResultContainer(",
                    "CrawlResult(",
                    "<property object at",
                    "__dict__",
                    "object at 0x",
                    "MarkdownGenerationResult(",
                ]

                if any(indicator in best_content for indicator in object_indicators):
                    self.logger.warning(
                        f"Detected object representation in content for {url}, clearing"
                    )
                    best_content = ""

                # Validate content quality - avoid storing trivial content
                if best_content and len(best_content.strip()) < 10:
                    self.logger.debug(f"Content too short for {url}, clearing")
                    best_content = ""

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Object-representation filters are good, but consider a single utility with tests

You’re checking multiple “object-like” indicators to sanitize content. Encapsulate this into a reusable helper and add unit tests so we don’t accidentally regress or miss cases.

-                object_indicators = [
+                object_indicators = [
                     "CrawlResultContainer(",
                     "CrawlResult(",
                     "<property object at",
                     "__dict__",
                     "object at 0x",
                     "MarkdownGenerationResult(",
                 ]

Follow-up: I can extract this list and the detection into utils/sanitization.py and include tests covering true positives/negatives. Want me to draft it?

    Committable suggestion skipped: line range outside the PR's diff.

@jmagar
tests/conftest.py
Comment on lines +37 to +38
from crawler_mcp.config import CrawlerrSettings, settings
from crawler_mcp.core import EmbeddingService, RagService, VectorService
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Imports not at the top (E402); consider moving to the main import block

While harmless at runtime, moving these imports to the top makes linters happy and avoids mixed top-level logic before imports.

-# (move these up near other imports)
-from crawler_mcp.config import CrawlerrSettings, settings
-from crawler_mcp.core import EmbeddingService, RagService, VectorService
+from crawler_mcp.config import CrawlerrSettings, settings
+from crawler_mcp.core import EmbeddingService, RagService, VectorService

    Committable suggestion skipped: line range outside the PR's diff.

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +101 to +108
    # Cleanup: Delete test collection after tests
    try:
        async with VectorService() as vector_service:
            await vector_service._client.delete_collection(
                test_settings.qdrant_collection
            )
    except Exception:
        pass  # Cleanup is best-effort
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Tests reach into VectorService internals via _client; avoid private attribute access

Both the session teardown and per-test cleanup call vector_service._client.delete_collection(...). That’s brittle and couples tests to implementation details.

Option A (preferred): add a public drop_collection(collection_name: str) method on VectorService and call that in tests.

Option B: in tests, resolve a public client if available, then fall back:

-        async with VectorService() as vector_service:
-            await vector_service._client.delete_collection(
-                test_settings.qdrant_collection
-            )
+        async with VectorService() as vector_service:
+            client = getattr(vector_service, "client", None) or getattr(vector_service, "_client", None)
+            if client is not None:
+                await client.delete_collection(test_settings.qdrant_collection)

Also apply the same change in clean_test_collection().

Also applies to: 121-132
@jmagar
tests/test_core_services.py
Comment on lines +148 to +150
        success = await vector_service.upsert_documents(documents)
        assert isinstance(success, (bool, int))  # May return count or boolean

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Ruff UP038: tuple-of-types in isinstance. Consider suppressing or targeting union if supported.

Ruff suggests int | bool, but isinstance(x, int | bool) is only valid in newer Python versions. To keep compatibility, either suppress UP038 or split the check.

Option A (suppress):

-        assert isinstance(success, (bool, int))  # May return count or boolean
+        assert isinstance(success, (bool, int))  # May return count or boolean  # noqa: UP038

Option B (portable split):

-        assert isinstance(success, (bool, int))  # May return count or boolean
+        assert isinstance(success, bool) or isinstance(success, int)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_core_services.py
Comment on lines +171 to +178
            # Results should be SearchMatch objects with document and score properties
            assert hasattr(result, "document") or hasattr(result, "score")
            if hasattr(result, "document"):
                assert hasattr(result.document, "content")
            if hasattr(result, "score"):
                assert isinstance(result.score, (int, float))
                assert 0 <= result.score <= 1

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same UP038 for score type check.

-                assert isinstance(result.score, (int, float))
+                assert isinstance(result.score, (int, float))  # noqa: UP038

Or split into two isinstance checks as above.
📝 Committable suggestion


🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +35 to +46
        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_pages=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_pages=3000)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_depth=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_depth=11)

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Catch Pydantic ValidationError instead of blanket Exception

Be precise about the expected failure type; aligns with Ruff B017 hint.

-import pytest
+import pytest
+from pydantic import ValidationError
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_pages=0)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_pages=3000)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_depth=0)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_depth=11)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +299 to +305
            # Mock pathlib.Path.iterdir to raise an exception
            with patch(
                "pathlib.Path.iterdir", side_effect=PermissionError("Access denied")
            ):
                result = await crawler.execute(request)

                # Should handle error gracefully - directory crawler may just log warnings
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Patch the correct API to simulate permission errors

execute() uses rglob/glob in _discover_files; patching Path.iterdir won't affect it. Patch Path.rglob instead.

-            # Mock pathlib.Path.iterdir to raise an exception
-            with patch(
-                "pathlib.Path.iterdir", side_effect=PermissionError("Access denied")
-            ):
+            # Mock recursive glob to raise an exception during discovery
+            with patch(
+                "pathlib.Path.rglob", side_effect=PermissionError("Access denied")
+            ):

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +369 to +381
            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=mock_create_subprocess_exec,
                ),
                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
            ):
                # Mock the clone directory to return our test structure
                with patch.object(crawler, "_clone_repository", return_value=repo_dir):
                    result = await crawler.execute(request)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Combine nested with blocks (SIM117) for clarity

Flatten the nested context managers to a single with.

-            with (
-                patch(
-                    "asyncio.create_subprocess_exec",
-                    side_effect=mock_create_subprocess_exec,
-                ),
-                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
-                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
-                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
-            ):
-                # Mock the clone directory to return our test structure
-                with patch.object(crawler, "_clone_repository", return_value=repo_dir):
-                    result = await crawler.execute(request)
+            with (
+                patch(
+                    "asyncio.create_subprocess_exec",
+                    side_effect=mock_create_subprocess_exec,
+                ),
+                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
+                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
+                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
+                patch.object(crawler, "_clone_repository", return_value=repo_dir),
+            ):
+                result = await crawler.execute(request)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +32 to +96
    async def test_validate_request_valid_urls(self):
        """Test validation with valid web crawl requests."""
        crawler = WebCrawlStrategy()

        # Valid single URL
        request = CrawlRequest(url="https://example.com", max_pages=50, max_depth=2)
        assert await crawler.validate_request(request) is True

        # Valid multiple URLs
        request_multi = CrawlRequest(
            url=["https://example.com", "https://test.com"], max_pages=100, max_depth=3
        )
        assert await crawler.validate_request(request_multi) is True

    @pytest.mark.unit
    async def test_validate_request_invalid_urls(self):
        """Test validation with invalid web crawl requests."""
        crawler = WebCrawlStrategy()

        # No URL provided (empty list)
        request_no_url = CrawlRequest(url=[])
        assert await crawler.validate_request(request_no_url) is False

        # Test with values that exceed crawler's custom validation limits
        # (WebCrawlStrategy checks max_pages <= 2000, but Pydantic allows <= 1000)
        request_high_pages = CrawlRequest(
            url="https://example.com",
            max_pages=1000,  # At Pydantic limit but valid for Pydantic
        )
        # This should pass since 1000 is within both Pydantic and crawler limits
        assert await crawler.validate_request(request_high_pages) is True

        # Test max_depth boundary (crawler allows <= 5, Pydantic allows <= 10)
        request_high_depth = CrawlRequest(
            url="https://example.com",
            max_depth=6,  # Exceeds crawler limit but within Pydantic limit
        )
        assert await crawler.validate_request(request_high_depth) is False

    @pytest.mark.unit
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()

            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

    @pytest.mark.unit
    async def test_validate_request_none_values(self):
        """Test validation with None values."""
        crawler = WebCrawlStrategy()

        # None values should be allowed (defaults will be used)
        request = CrawlRequest(
            url="https://example.com", max_pages=None, max_depth=None
        )
        assert await crawler.validate_request(request) is True


@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Mark async tests with pytest-asyncio (or enable asyncio_mode=auto).

Several tests are async but lack @pytest.mark.asyncio. Unless pytest-asyncio’s auto mode is enabled in configuration, these will be collected but not awaited, leading to false positives/failures. Prefer a module-level marker to keep noise low.

Apply this minimal change near the imports:

 import pytest
+pytestmark = pytest.mark.asyncio

Also applies to: 109-144, 158-204, 227-264
🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +55 to +70
        # Test with values that exceed crawler's custom validation limits
        # (WebCrawlStrategy checks max_pages <= 2000, but Pydantic allows <= 1000)
        request_high_pages = CrawlRequest(
            url="https://example.com",
            max_pages=1000,  # At Pydantic limit but valid for Pydantic
        )
        # This should pass since 1000 is within both Pydantic and crawler limits
        assert await crawler.validate_request(request_high_pages) is True

        # Test max_depth boundary (crawler allows <= 5, Pydantic allows <= 10)
        request_high_depth = CrawlRequest(
            url="https://example.com",
            max_depth=6,  # Exceeds crawler limit but within Pydantic limit
        )
        assert await crawler.validate_request(request_high_depth) is False

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Boundary assertions are good; add a positive case for max_depth=5.

You test that 6 fails; add an assertion that 5 passes to pin the upper bound.

         request_high_depth = CrawlRequest(
             url="https://example.com",
             max_depth=6,  # Exceeds crawler limit but within Pydantic limit
         )
         assert await crawler.validate_request(request_high_depth) is False
+
+        # Upper boundary should pass
+        request_depth_edge = CrawlRequest(url="https://example.com", max_depth=5)
+        assert await crawler.validate_request(request_depth_edge) is True

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +72 to +84
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()

            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Private method test is acceptable; assert idempotency and no re-init.

Since _initialize_managers is private, keep the test but also assert it doesn’t reinitialize on subsequent calls.

             await crawler._initialize_managers()

             assert crawler.memory_manager is mock_manager
             mock_get_manager.assert_called_once()
+
+            # Calling again should not fetch a new manager
+            await crawler._initialize_managers()
+            mock_get_manager.assert_called_once()

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +206 to +225
class TestCrawlerEdgeCases:
    """Test edge cases and error conditions across crawlers."""

    @pytest.mark.unit
    def test_crawler_inheritance(self):
        """Test that all crawlers inherit from base properly."""
        web_crawler = WebCrawlStrategy()
        dir_crawler = DirectoryCrawlStrategy()
        repo_crawler = RepositoryCrawlStrategy()

        # All should have logger from base class
        assert hasattr(web_crawler, "logger")
        assert hasattr(dir_crawler, "logger")
        assert hasattr(repo_crawler, "logger")

        # All should be callable instances
        assert callable(web_crawler.validate_request)
        assert callable(dir_crawler.validate_request)
        assert callable(repo_crawler.validate_request)

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Tighten edge‐case validation tests to avoid catch-all exceptions

The test_empty_request_handling currently wraps every validate_request call in a broad try / except Exception, which risks silently hiding regressions. Instead, for truly minimal requests you should:

    Remove the try { … } catch Exception: pass around each call to await …validate_request.

    Assert that the result is always a boolean. For example, in tests/test_crawlers.py::TestCrawlerEdgeCases::test_empty_request_handling replace:

  try:

      result = await web_crawler.validate_request(web_request)

      assert result in [True, False]

  except Exception:

      # Some crawlers might throw exceptions for invalid URLs

      # This is acceptable behavior

      pass


with

```diff

  result = await web_crawler.validate_request(web_request)

  assert isinstance(result, bool)


And do the same for the directory and repository sections.

If a specific strategy is expected to raise a known exception for truly invalid inputs (for example, a parsing error or a ValueError when the repo URL is grossly malformed), catch only that exception type:

try:
    result = await repo_crawler.validate_request(bad_repo_request)
    assert isinstance(result, bool)
except ValueError:
    # Explicit, documented failure mode for malformed URLs
    pass

This change makes sure:

    You never hide unexpected errors.
    You verify that every “minimal” request path yields a boolean.
    You still allow for documented, intentional exceptions on truly invalid inputs.

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +270 to +282
    def test_web_crawler_uses_settings(self):
        """Test that web crawler accesses configuration settings."""
        crawler = WebCrawlStrategy()

        # This will exercise settings access during validation
        with patch("crawler_mcp.crawlers.web.settings") as mock_settings:
            mock_settings.crawl_headless = True
            mock_settings.crawl_browser = "chromium"

            # Just instantiating should exercise some settings access
            assert crawler is not None

    @pytest.mark.unit
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

This test doesn’t exercise settings; convert to async and call validate_request.

The test name and comment imply we exercise settings access “during validation,” but no validation is performed. Convert to async and invoke validate_request to actually touch the settings path.

-    def test_web_crawler_uses_settings(self):
+    async def test_web_crawler_uses_settings(self):
         """Test that web crawler accesses configuration settings."""
         crawler = WebCrawlStrategy()

         # This will exercise settings access during validation
         with patch("crawler_mcp.crawlers.web.settings") as mock_settings:
             mock_settings.crawl_headless = True
             mock_settings.crawl_browser = "chromium"

-            # Just instantiating should exercise some settings access
-            assert crawler is not None
+            # Call a method that consults settings
+            req = CrawlRequest(url="https://example.com")
+            result = await crawler.validate_request(req)
+            assert isinstance(result, bool)

🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +40 to +50
        await asyncio.sleep(2)

        # Step 2: Verify content was indexed by checking stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data

        assert stats["total_sources"] > 0
        assert stats["total_chunks"] > 0
        assert "directory" in stats["source_types"]

        # Step 3: Query the indexed content
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Fixed sleeps make tests flaky; prefer polling for eventual consistency.

Replace static asyncio.sleep(2) with a short polling loop that checks get_rag_stats for the expected increase or times out (e.g., within ~10s). This reduces flakiness on slow CI.

Example helper (inlined for brevity):

-        await asyncio.sleep(2)
+        async def wait_for_nonzero_sources(client: Client, timeout_s: float = 10.0):
+            import time
+            end = time.time() + timeout_s
+            while time.time() < end:
+                stats = (await client.call_tool("get_rag_stats", {})).data
+                if stats.get("total_sources", 0) > 0:
+                    return stats
+                await asyncio.sleep(0.25)
+            return stats
+
+        stats_result = await wait_for_nonzero_sources(mcp_client)
+        stats = stats_result

Apply analogous polling in other spots waiting for indexing.

Also applies to: 120-129, 299-311
🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +140 to +143
        rag_data = query_result.data
        # Should find some results (may be 0 if content is minimal)
        assert rag_data["total_results"] >= 0

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Trivial assertion: total_results >= 0 is tautological.

Consider asserting types and shape instead:

-        assert rag_data["total_results"] >= 0
+        assert isinstance(rag_data.get("total_results"), int)
+        assert isinstance(rag_data.get("results"), list)

🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +239 to +241
        directory_path = sample_text_files[0].parent

        # Create multiple concurrent operations
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused variable (Ruff F841).

directory_path is assigned but never used in this test.

-        directory_path = sample_text_files[0].parent

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_models.py
Comment on lines +199 to +206
        source = SourceInfo(
            source_id="test_source_123",
            source_type=SourceType.WEBPAGE,
            url="https://test.com/page",
            title="Test Page",
            metadata=metadata,
        )

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix SourceInfo field name: use id (not source_id)

The model defines id: str, not source_id. Adjust constructor and assertions.

-        source = SourceInfo(
-            source_id="test_source_123",
-            source_type=SourceType.WEBPAGE,
-            url="https://test.com/page",
-            title="Test Page",
-            metadata=metadata,
-        )
+        source = SourceInfo(
+            id="test_source_123",
+            source_type=SourceType.WEBPAGE,
+            url="https://test.com/page",
+            title="Test Page",
+            metadata=metadata,
+        )
@@
-        assert source.source_id == "test_source_123"
+        assert source.id == "test_source_123"

Also applies to: 207-212
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +31 to +36
@pytest.fixture
async def test_server():
    """Create a test server instance for in-memory testing."""
    async with Client(mcp) as client:
        yield client

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Optional: mark MCP integration tests as integration to avoid flakiness when external services are unavailable

If CI lacks live vector/embedding backends, consider pytest markers to allow selective runs.

I can add pytest markers and CI matrix guards if you want.

Also applies to: 623-676
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +86 to +99
        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("test query", 10, 0.7, ["source1"], True, mock_result)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Instantiate RagResult with only declared fields

average_score, best_match_score, has_high_confidence_matches appear to be computed properties, not constructor fields. Keep to the model surface to avoid silent drops.

         mock_result = RagResult(
             query="test query",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +109 to +121
        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for TTL test

Mirror the change above.

         mock_result = RagResult(
             query="test query",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +140 to +178
        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result3 = RagResult(
            query="query3",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for size-limit test (all three instances)

Keep RagResult init minimal and consistent.

         result1 = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
@@
         result2 = RagResult(
             query="query2",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
@@
         result3 = RagResult(
             query="query3",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +202 to +227
        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for clear() test

Remove non-declared fields.

         result1 = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
         result2 = RagResult(
             query="query2",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +253 to +266
        result = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("query1", 10, 0.7, ["source1"], True, result)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for stats() test

Align with declared fields only.

         result = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +530 to +535
        query = RagQuery(
            query="FastMCP testing capabilities",
            sources=["https://example.com/test-query"],
            limit=5,
            score_threshold=0.1,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Use RagQuery’s actual field names: source_filters, min_score

The model exposes source_filters and min_score (not sources/score_threshold).

-        query = RagQuery(
-            query="FastMCP testing capabilities",
-            sources=["https://example.com/test-query"],
-            limit=5,
-            score_threshold=0.1,
-        )
+        query = RagQuery(
+            query="FastMCP testing capabilities",
+            source_filters=["https://example.com/test-query"],
+            limit=5,
+            min_score=0.1,
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +609 to +612
        # delete_source returns an integer count, not a dict
        assert isinstance(result, int)
        assert result >= 0  # Should return number of deleted documents

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Align test_delete_source with actual return type of delete_source

The RagService.delete_source method in crawler_mcp/core/rag.py currently returns a boolean indicating success, not an integer count. The existing test in tests/test_rag_comprehensive.py (around lines 609–612) assumes an integer and will fail. Update the test to accept either a bool or an int, ensuring backward compatibility if the implementation changes in the future.

Required changes in tests/test_rag_comprehensive.py:

• File: tests/test_rag_comprehensive.py
Lines: around 609–612

-        # delete_source returns an integer count, not a dict
-        assert isinstance(result, int)
-        assert result >= 0  # Should return number of deleted documents
+        # delete_source may return a bool (success) or an integer count
+        assert isinstance(result, (bool, int))
+        if isinstance(result, int):
+            assert result >= 0  # Number of deleted documents must be non-negative

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +630 to +637
        scrape_result = await test_server.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Now test rag_query
        query_result = await test_server.call_tool(
            "rag_query", {"query": "test content", "limit": 5}
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused variable assignment (Ruff F841)

No need to bind the scrape result.

-        scrape_result = await test_server.call_tool(
-            "scrape", {"url": "https://example.com", "process_with_rag": True}
-        )
+        await test_server.call_tool(
+            "scrape", {"url": "https://example.com", "process_with_rag": True}
+        )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_server_health.py
Comment on lines +55 to +59
        server = info_data["server"]
        assert server["name"] == "Crawlerr"
        assert server["framework"] == "FastMCP 2.0+"
        assert server["version"] == "0.1.0"

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: avoid hard-coding server name/framework/version unless guaranteed.

If these values are configurable, prefer presence/type checks or fetch from settings.

-        assert server["name"] == "Crawlerr"
-        assert server["framework"] == "FastMCP 2.0+"
-        assert server["version"] == "0.1.0"
+        assert isinstance(server.get("name"), str) and server["name"]
+        assert isinstance(server.get("framework"), str) and server["framework"]
+        assert isinstance(server.get("version"), str) and server["version"]

🤖 Prompt for AI Agents

@jmagar
tests/test_server.py
Comment on lines +124 to +150
            patch("crawler_mcp.server.EmbeddingService") as mock_embedding,
            patch("crawler_mcp.server.VectorService") as mock_vector,
            patch("crawler_mcp.server.RagService") as mock_rag,
        ):
            # Mock embedding service
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.health_check.return_value = True
            mock_embedding_instance.get_model_info.return_value = {
                "model": "test-model"
            }
            mock_embedding().__aenter__.return_value = mock_embedding_instance

            # Mock vector service
            mock_vector_instance = AsyncMock()
            mock_vector_instance.health_check.return_value = True
            mock_vector_instance.get_collection_info.return_value = {"points": 100}
            mock_vector().__aenter__.return_value = mock_vector_instance

            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_instance.health_check.return_value = {
                "embedding": True,
                "vector": True,
            }
            mock_rag_instance.get_stats.return_value = {"documents": 100}
            mock_rag().__aenter__.return_value = mock_rag_instance

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Patching server services is redundant in this test fixture context

Because you define a local health_check tool within the test_server fixture, the patches to crawler_mcp.server.* services are not exercised in this test. This isn’t harmful but adds noise.

    Remove the patch("crawler_mcp.server.*") context from test_health_check_tool_mocked, or alternatively, add a separate test that targets the real server module’s health_check.

🤖 Prompt for AI Agents

@jmagar
tests/test_server.py
Comment on lines +711 to +757
class TestServerConfiguration:
    """Test server configuration and startup functionality."""

    def test_server_settings_loaded(self):
        """Test that server settings are properly loaded."""
        # Check key settings are available
        assert hasattr(settings, "server_host")
        assert hasattr(settings, "server_port")
        assert hasattr(settings, "debug")
        assert hasattr(settings, "tei_url")
        assert hasattr(settings, "qdrant_url")

        # Check types
        assert isinstance(settings.server_host, str)
        assert isinstance(settings.server_port, int)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.tei_url, str)
        assert isinstance(settings.qdrant_url, str)

    @pytest.mark.asyncio
    async def test_mcp_instance_tools(self, test_server):
        """Test that MCP server instance has correct tools registered."""
        # Check test server is created
        assert test_server is not None
        assert hasattr(test_server, "get_tools")
        assert callable(test_server.get_tools)

        # Check that tools are registered
        tools = await test_server.get_tools()
        assert len(tools) > 0

        # Check for expected tools
        tool_names = list(tools.keys())
        expected_tools = [
            "health_check",
            "get_server_info",
            "scrape",
            "crawl",
            "rag_query",
            "list_sources",
            "get_rag_stats",
            "delete_source",
        ]

        for expected in expected_tools:
            assert expected in tool_names

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Tool registration assertions are appropriate and future-proof

Asserting get_tools() and expected tool names helps catch accidental deregistration. Consider also asserting that help/params for one tool are well-formed to detect schema regressions.
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +122 to +128
            average_chunk_size=1000.0,
            created_at=created_time,
            updated_at=updated_time,
            last_crawled=last_crawled,
            is_stale=False,
            metadata=metadata,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix unsupported fields and wrong attribute access (average_chunk_size/is_stale)

    Remove unsupported "average_chunk_size" and "is_stale" kwargs from SourceInfo().
    Assert via the computed property avg_chunk_size, not a non-existent "average_chunk_size" attribute.

         source = SourceInfo(
             id="source-123",
             url="https://example.com/page",
             title="Test Page",
             source_type=SourceType.WEBPAGE,
             status="active",
             chunk_count=5,
             total_content_length=5000,
-            average_chunk_size=1000.0,
             created_at=created_time,
             updated_at=updated_time,
             last_crawled=last_crawled,
-            is_stale=False,
             metadata=metadata,
         )
@@
-        assert source.average_chunk_size == 1000.0
+        assert source.avg_chunk_size == 1000.0
-        assert source.is_stale is False
+        assert source.is_stale is False

Also applies to: 137-143
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +150 to +160
        assert source.id == "min-source"
        assert source.url == "https://minimal.com"
        assert source.source_type == SourceType.WEBPAGE
        assert source.title is None
        assert source.status == "active"  # Default
        assert source.chunk_count == 0
        assert source.total_content_length == 0
        assert source.average_chunk_size == 0.0
        assert source.is_stale is False
        assert isinstance(source.metadata, SourceMetadata)

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Defaults test: use avg_chunk_size property

The model exposes avg_chunk_size; "average_chunk_size" isn't a field.

-        assert source.average_chunk_size == 0.0
+        assert source.avg_chunk_size == 0.0

Also applies to: 155-159, 157-157
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +161 to +173
    def test_source_info_time_fields(self):
        """Test automatic time field population."""
        before_creation = datetime.utcnow()
        source = SourceInfo(
            id="time-test", url="https://time.com", source_type=SourceType.WEBPAGE
        )
        after_creation = datetime.utcnow()

        # Should have created_at and updated_at set automatically
        assert before_creation <= source.created_at <= after_creation
        assert before_creation <= source.updated_at <= after_creation
        assert source.created_at == source.updated_at  # Should be same initially

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Relax equality of created_at/updated_at

They are set via separate default factories; strict equality is brittle.

-        assert source.created_at == source.updated_at  # Should be same initially
+        # Should be very close initially
+        assert abs((source.created_at - source.updated_at).total_seconds()) < 0.5

Also applies to: 172-172
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
            source_type=SourceType.WEBPAGE,
            chunk_count=5,
            total_content_length=5000,
            average_chunk_size=1000.0,
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unsupported parameter "average_chunk_size" from SourceInfo construction

SourceInfo doesn't define an "average_chunk_size" field (it exposes avg_chunk_size as a computed property). Passing unknown kwargs is ignored by Pydantic but is misleading. Rely on chunk_count and total_content_length; assert via avg_chunk_size.

-            average_chunk_size=1000.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +200 to +201
        time_diff = abs((source.created_at - source.updated_at).total_seconds())
        assert time_diff < 0.01  # Should be within 10ms
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Relax timing tolerance to reduce flakiness on slower CI runners

Strict 10ms equality is brittle; factory defaults may differ by tens/hundreds of ms.

-        time_diff = abs((source.created_at - source.updated_at).total_seconds())
-        assert time_diff < 0.01  # Should be within 10ms
+        time_diff = abs((source.created_at - source.updated_at).total_seconds())
+        assert time_diff < 0.5  # Allow up to 500ms

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +323 to +324
            average_chunk_size=0.0,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" in zero-source case

Not a model field; the property will already compute 0.0 when chunk_count == 0.

-            average_chunk_size=0.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +336 to +337
            average_chunk_size=5000.0,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" in large-source case

Keep only chunk_count and total_content_length.

-            average_chunk_size=5000.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +129 to +132
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Use UpdateStatus enum for upsert status; strings won’t match code.

VectorService compares result.status to UpdateStatus.COMPLETED. Returning "completed" (str) will not match and will make the test fail spuriously.

-                mock_qdrant_client.upsert.return_value = MagicMock(
-                    operation_id=0, status="completed"
-                )
+                from qdrant_client.models import UpdateStatus
+                mock_qdrant_client.upsert.return_value = MagicMock(
+                    operation_id=0, status=UpdateStatus.COMPLETED
+                )

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +160 to +163
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Same issue in batch test—return UpdateStatus.COMPLETED.

-                mock_qdrant_client.upsert.return_value = MagicMock(
-                    operation_id=0, status="completed"
-                )
+                from qdrant_client.models import UpdateStatus
+                mock_qdrant_client.upsert.return_value = MagicMock(
+                    operation_id=0, status=UpdateStatus.COMPLETED
+                )

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +213 to +222
                query_vector = [0.1, 0.2, 0.3] * 128
                results = await service.search_similar(
                    query_vector, limit=10, score_threshold=0.7
                )

                assert len(results) == 1
                assert results[0].score == 0.95
                assert results[0].document.content == "Test content"
                mock_qdrant_client.query_points.assert_called_once()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Optionally assert dynamic search_params (ef) wiring.

To lock in the “typed SearchParams” change, assert that query_points received a SearchParams with expected hnsw_ef.

                 results = await service.search_similar(
                     query_vector, limit=10, score_threshold=0.7
                 )

                 assert len(results) == 1
                 assert results[0].score == 0.95
                 assert results[0].document.content == "Test content"
                 mock_qdrant_client.query_points.assert_called_once()
+                args, kwargs = mock_qdrant_client.query_points.call_args
+                assert "search_params" in kwargs
+                assert getattr(kwargs["search_params"], "hnsw_ef", None) == 40  # min(256, max(64, 10*4)) = 64 → wait, code caps at min 256 and max 64; for limit=10, ef=64
+                assert kwargs["with_vectors"] is False

Note: For limit=10, ef_value = max(64, 40) → 64.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Review continued from previous batch...
crawler_mcp/crawlers/web.py
Comment on lines 66 to 76
    def __init__(self) -> None:
        super().__init__()
        self.memory_manager: MemoryManager | None = None
        self.memory_manager = None

    async def _initialize_managers(self) -> None:
        """Initialize required managers."""
        if not self.memory_manager:
            from ..core.memory import get_memory_manager

            self.memory_manager = get_memory_manager()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Minor: keep memory_manager typing consistent with BaseCrawlStrategy

You set self.memory_manager = None without an annotation. Consider keeping the same Optional[...] type annotation as the base to help static checkers.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Outdated
Comment on lines 499 to 537
        try:
            # Check if the private _markdown field contains an integer hash
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
            if hasattr(result, "markdown"):
                try:
                    # Try to access it to see if it would error
                    _ = result.markdown
                except AttributeError as e:
                    if "'int' object has no attribute" in str(e):
                        print(
                            f"CRAWL DEBUG - Markdown property access failed, force setting safe value for {result.url}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Force set a safe markdown value
                        result._markdown = MarkdownGenerationResult(
                            raw_markdown="",
                            markdown_with_citations="",
                            references_markdown="",
                            fit_markdown=None,
                            fit_html=None,
                        )
        except Exception as e:
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Avoid mutating third‑party private attribute _markdown; rely on safe access instead

Directly setting result._markdown is brittle and may break across crawl4ai versions. Since _safe_get_markdown already handles int/hash sentinel values and exceptions, prefer not to mutate the result object. If you must sanitize, store a local fallback rather than overwriting private fields.

Option A (preferred): Remove the mutation and let _safe_get_markdown handle all cases.

-                # Replace the integer hash with an empty MarkdownGenerationResult
-                result._markdown = MarkdownGenerationResult(
-                    raw_markdown="",
-                    markdown_with_citations="",
-                    references_markdown="",
-                    fit_markdown=None,
-                    fit_html=None,
-                )
+                # Defer to _safe_get_markdown which handles integer sentinel cases
+                pass
@@
-                        # Force set a safe markdown value
-                        result._markdown = MarkdownGenerationResult(
-                            raw_markdown="",
-                            markdown_with_citations="",
-                            references_markdown="",
-                            fit_markdown=None,
-                            fit_html=None,
-                        )
+                        # Defer to _safe_get_markdown to provide an empty-safe value
+                        pass

Option B: If you keep the mutation for legacy versions, gate it behind a feature flag (settings.crawl_force_markdown_sanitize) and log a one-time warning.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
crawler_mcp/crawlers/web.py
Outdated
Comment on lines 681 to 689
        content_filter = PruningContentFilter(
            threshold=0.45,  # Prune nodes below 45% relevance score
            threshold=0.48,  # Prune nodes below 48% relevance score for better quality
            threshold_type="dynamic",  # Dynamic scoring
            min_word_threshold=5,  # Ignore very short text blocks
            min_word_threshold=10,  # Higher threshold for quality content blocks
        )

        # Create markdown generator with content filter
        markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Be defensive around PruningContentFilter signature across versions

Some crawl4ai versions may not accept threshold_type="dynamic". Add a fallback.

-        content_filter = PruningContentFilter(
-            threshold=0.48,  # Prune nodes below 48% relevance score for better quality
-            threshold_type="dynamic",  # Dynamic scoring
-            min_word_threshold=10,  # Higher threshold for quality content blocks
-        )
+        try:
+            content_filter = PruningContentFilter(
+                threshold=0.48,
+                threshold_type="dynamic",
+                min_word_threshold=10,
+            )
+        except Exception:
+            content_filter = PruningContentFilter(
+                threshold=0.48,
+                min_word_threshold=10,
+            )

If you want me to confirm the current crawl4ai signature, I can run a quick check against the installed version.
📝 Committable suggestion


@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +728 to +736
        extraction_strategy = getattr(request, "extraction_strategy", None)
        if extraction_strategy == "llm":
            with contextlib.suppress(Exception):
                run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
                    provider="openai",
                    api_token="",
                    instruction="Extract main content and key information from the page",
                )
        elif extraction_strategy == "cosine":
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

LLM extraction: avoid empty API token; gate on configured credentials

Creating an LLMExtractionStrategy with api_token="" is likely to fail at runtime or pull from implicit env. Respect settings and warn if missing.

-        if extraction_strategy == "llm":
-            with contextlib.suppress(Exception):
-                run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
-                    provider="openai",
-                    api_token="",
-                    instruction="Extract main content and key information from the page",
-                )
+        if extraction_strategy == "llm":
+            token = getattr(settings, "openai_api_key", None)
+            if token:
+                with contextlib.suppress(Exception):
+                    run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
+                        provider="openai",
+                        api_token=token,
+                        instruction="Extract main content and key information from the page",
+                    )
+            else:
+                self.logger.warning("LLM extraction requested but no API token configured; skipping LLM strategy.")

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +18 to +29
# Fix NumPy 2.x + SciPy 1.16.1 compatibility issue for coverage measurement
# Force NumPy and SciPy imports before coverage instrumentation to prevent conflicts
try:
    import numpy as np
    import scipy.stats

    # Force initialization to happen before pytest-cov instruments the code
    np.__version__
    scipy.stats.__version__
except Exception:
    # If there are import issues, proceed without the imports
    pass
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Useless expressions for version access; remove or use a harmless reference

The np.__version__ and scipy.stats.__version__ lines are flagged as useless expressions (Ruff B018). If the goal is import priming, the import itself suffices.

-    # Force initialization to happen before pytest-cov instruments the code
-    np.__version__
-    scipy.stats.__version__
+    # Imports above are enough to prime coverage/instrumentation
+    pass

Alternatively, log versions if you want a side effect.
📝 Committable suggestion


🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +123 to +129
            try:
                await vector_service._client.delete_collection(
                    test_settings.qdrant_collection
                )
            except Exception:
                pass  # Collection might not exist

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Use contextlib.suppress instead of try/except/pass

Cleaner and linter-friendly.

-            try:
-                await vector_service._client.delete_collection(
-                    test_settings.qdrant_collection
-                )
-            except Exception:
-                pass  # Collection might not exist
+            from contextlib import suppress
+            with suppress(Exception):
+                client = getattr(vector_service, "client", None) or getattr(vector_service, "_client", None)
+                if client is not None:
+                    await client.delete_collection(test_settings.qdrant_collection)

    Committable suggestion skipped: line range outside the PR's diff.

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +7 to +11
import pytest
from fastmcp import Client


class TestCrawlingTools:
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Mark async tests to avoid event-loop detection flakiness

These tests are async and rely on an async fixture. Add a file-level marker so they run under pytest-asyncio/anyio regardless of global config.

 from fastmcp import Client


+pytestmark = pytest.mark.anyio
+
 class TestCrawlingTools:

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +50 to +78
    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_scrape_with_auto_rag(self, mcp_client: Client):
        """Test scraping with automatic RAG indexing."""
        test_url = "https://httpbin.org/html"

        result = await mcp_client.call_tool(
            "scrape",
            {
                "url": test_url,
                "auto_index": True,  # Enable automatic RAG indexing
            },
        )

        assert result.data is not None
        scrape_data = result.data

        assert scrape_data["success"] is True

        # Should have indexing information
        assert "indexed" in scrape_data
        assert "source_id" in scrape_data

        # If indexing succeeded
        if scrape_data["indexed"]:
            assert scrape_data["source_id"] is not None
            assert "chunks_created" in scrape_data
            assert isinstance(scrape_data["chunks_created"], int)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Assert minimal indexing contract in auto_index path

Good coverage. Consider asserting that source_id is a non-empty string (when indexed) to catch accidental None/empty IDs.

         if scrape_data["indexed"]:
             assert scrape_data["source_id"] is not None
             assert "chunks_created" in scrape_data
             assert isinstance(scrape_data["chunks_created"], int)
+            assert isinstance(scrape_data["source_id"], str) and scrape_data["source_id"].strip()

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +147 to +175
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_crawl_small_website(self, mcp_client: Client):
        """Test crawling a small website (marked as slow)."""
        # Use a simple, reliable test site
        test_url = "https://httpbin.org/"

        result = await mcp_client.call_tool(
            "crawl",
            {
                "target": test_url,
                "max_pages": 3,  # Limit to avoid long test times
                "max_depth": 1,
                "auto_index": False,  # Don't index for this test
            },
        )

        assert result.data is not None
        crawl_data = result.data

        # Should detect as website crawl
        assert crawl_data["crawl_type"] == "website"
        assert crawl_data["success"] is True

        # Should have crawled some pages
        results = crawl_data["results"]
        assert isinstance(results, list)
        assert len(results) >= 1  # At least the main page

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Network-bound integration test: add a defensive skip for offline CI

httpbin is usually reliable, but CI may run without egress. Consider skipping when network is disabled.

 @pytest.mark.slow
 @pytest.mark.integration
 async def test_crawl_small_website(self, mcp_client: Client):
@@
-        result = await mcp_client.call_tool(
+        result = await mcp_client.call_tool(
             "crawl",
             {
                 "target": test_url,
                 "max_pages": 3,  # Limit to avoid long test times
                 "max_depth": 1,
                 "auto_index": False,  # Don't index for this test
             },
         )

And add a small helper at top of file if you want an automatic skip:

+def _has_network() -> bool:
+    import socket
+    try:
+        socket.gethostbyname("httpbin.org")
+        return True
+    except Exception:
+        return False
+
@@
-    async def test_crawl_small_website(self, mcp_client: Client):
+    async def test_crawl_small_website(self, mcp_client: Client):
+        if not _has_network():
+            pytest.skip("No network available in CI")

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +179 to +186
        with pytest.raises(Exception):  # Should raise ToolError
            await mcp_client.call_tool(
                "scrape",
                {
                    "url": "not-a-valid-url",
                },
            )

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Use specific ToolError (and ValueError) in crawling tests

Catching the base Exception silences real errors and violates Ruff B017. In tests/test_crawling_tools.py, replace each with pytest.raises(Exception): in your crawling tests with an explicit tuple of the exceptions your code actually raises—namely ToolError (from fastmcp.exceptions) and ValueError for invalid parameters.

Key locations to update:

    tests/test_crawling_tools.py, test_invalid_url_handling (lines 179–186)
    tests/test_crawling_tools.py, test_nonexistent_directory (lines 190–196)
    The remaining invalid-parameter tests at lines 202–204, 206–214, 219–221, and 223–230

Example diff (apply analogous changes throughout):

--- a/tests/test_crawling_tools.py
+++ b/tests/test_crawling_tools.py
@@
 from fastmcp import Client
+from fastmcp.exceptions import ToolError

@@ -179,7 +180,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):  # Should raise ToolError
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "scrape",
                 {"url": "not-a-valid-url"},
             )
@@ -190,7 +191,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):  # Should raise ToolError
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "crawl",
                 {"target": "/nonexistent/directory/path"},
             )
@@ -202,7 +203,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool("crawl", {})

@@ -206,7 +207,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "crawl",
                 {"target": "https://httpbin.org", "max_pages": -1},
             )
@@ -219,7 +220,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool("scrape", {})

@@ -223,7 +224,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "scrape",
                 {"url": "https://httpbin.org", "extraction_strategy": "invalid_strategy"},
             )

Be sure to add at the top alongside your other imports:

from fastmcp.exceptions import ToolError

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +29 to +33
    @pytest.fixture
    def middleware(self):
        """Create ErrorHandlingMiddleware instance."""
        return ErrorHandlingMiddleware()

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Tests for middleware constructors and methods need fixing

The ErrorHandlingMiddleware, LoggingMiddleware, and ProgressMiddleware classes all require an app: Callable[[Request], Awaitable[Response]] in their constructors and expose a __call__(request: Request) -> Response method. There is no .process(ctx, handler) helper. As written, the tests will fail with:

    TypeError on instantiation (missing required app argument)
    AttributeError when calling a nonexistent .process method

Please update the tests (in tests/test_middleware_comprehensive.py at lines 29–33, 135–138, and 210–214) to either:

    Pass a minimal dummy app into each middleware’s constructor and invoke the middleware via its await middleware(request) interface, or
    Add a .process(ctx, handler) shim in each middleware class that wraps __call__, if you intend to keep the existing test APIs

🤖 Prompt for AI Agents

@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +222 to +233
    def test_progress_tracker_creation(self):
        """Test ProgressTracker creation and properties."""
        tracker = ProgressTracker("test-op-123", "Test Operation")

        assert tracker.operation_id == "test-op-123"
        assert tracker.name == "Test Operation"
        assert tracker.current_step == 0
        assert tracker.total_steps == 0
        assert tracker.status == "pending"
        assert tracker.start_time is not None
        assert tracker.end_time is None

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

🛠️ Refactor suggestion

ProgressTracker/ProgressMiddleware expectations diverge from implementation

The tests assume:

    ProgressTracker has a name, end_time, .update_progress(), .complete(), .error()
    Initial status == "pending"
    ProgressMiddleware exposes .active_operations (public), .list_operations(), and .cleanup_completed()

The provided implementation shows different attributes/method names (update(...), status starts as "starting", internal _active_operations, .list_active_operations()), so these tests will fail.

Either:

    Update tests to align with the existing API (prefer public helpers and avoid touching internals), or
    Adjust progress.py to add the missing convenience methods and public views used in tests.

Example test-side adjustments (if aligning to existing API):

- tracker = ProgressTracker("test-op-123", "Test Operation")
+ tracker = ProgressTracker("test-op-123")
- assert tracker.name == "Test Operation"
- assert tracker.status == "pending"
+ assert tracker.status in {"starting", "running", "completed", "error"}

- tracker.update_progress(5, 10, "Processing...")
+ tracker.update(5, 10, status="running", message="Processing...")

- tracker.complete("All done!")
+ tracker.update(tracker.total_steps, tracker.total_steps, status="completed", message="All done!")

- operations = middleware.list_operations()
+ operations = list(middleware.list_active_operations().values())

If you prefer the test-facing API, I can add a thin adapter layer to progress.py that preserves current behavior while exposing the test-friendly façade. Want me to draft it?

Also applies to: 235-250, 252-259, 261-274, 276-305, 309-345, 347-373, 387-407, 409-437
@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +297 to +304
        """Test removing trackers."""
        # Create a tracker
        tracker = middleware.create_tracker("removable", "Removable Op")
        assert "removable" in middleware.active_operations

        # Remove it
        middleware.remove_tracker("removable")
        assert "removable" not in middleware.active_operations
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Unused local variable (tracker) — remove assignment

tracker isn’t used after creation.

-        tracker = middleware.create_tracker("removable", "Removable Op")
+        middleware.create_tracker("removable", "Removable Op")

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +26 to +39
        assert result.data is not None
        rag_data = result.data

        # Check basic structure
        assert "query" in rag_data
        assert "results" in rag_data
        assert "total_results" in rag_data
        assert "search_metadata" in rag_data

        assert rag_data["query"] == "test query with no results"
        assert rag_data["results"] == []
        assert rag_data["total_results"] == 0

    @pytest.mark.integration
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Result shape assertions don’t match the server tool contract

rag_query returns {"query","total_matches","matches","performance","quality_metrics","timestamp"} per current tools, not results/total_results/search_metadata.

Apply this update:

-        # Check basic structure
-        assert "query" in rag_data
-        assert "results" in rag_data
-        assert "total_results" in rag_data
-        assert "search_metadata" in rag_data
+        # Check basic structure
+        assert "query" in rag_data
+        assert "matches" in rag_data
+        assert "total_matches" in rag_data
+        assert "performance" in rag_data
...
-        assert rag_data["results"] == []
-        assert rag_data["total_results"] == 0
+        assert rag_data["matches"] == []
+        assert rag_data["total_matches"] == 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +47 to +59
        # Step 1: Index some content
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        assert crawl_result.data is not None
        assert crawl_result.data["success"] is True

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Crawl tool parameters don’t exist; use process_with_rag instead and align follow-up query checks

crawl doesn’t accept auto_index or max_files. Use process_with_rag=True to index results.

-        crawl_result = await mcp_client.call_tool(
+        crawl_result = await mcp_client.call_tool(
             "crawl",
             {
                 "target": str(directory_path),
-                "auto_index": True,
-                "max_files": 3,
+                "process_with_rag": True,
             },
         )
...
-        # Should find some results now
-        assert rag_data["total_results"] > 0
-        assert len(rag_data["results"]) > 0
+        # Should find some results now
+        assert rag_data["total_matches"] > 0
+        assert len(rag_data["matches"]) > 0
...
-        for result in rag_data["results"]:
-            assert "content" in result
-            assert "metadata" in result
-            assert "score" in result
-            assert "source_id" in result
+        for match in rag_data["matches"]:
+            assert "score" in match
+            assert "relevance" in match
+            assert "document" in match
+            doc = match["document"]
+            assert "content" in doc
+            assert "metadata" in doc
+            assert "source_url" in doc

Also applies to: 65-93
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +102 to +117
        # Index content first
        await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # Query with directory filter
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Filter params don’t match; use source_filters and updated keys

rag_query takes source_filters, not source_types, and results are under matches with per-match document.metadata.

-        result = await mcp_client.call_tool(
+        result = await mcp_client.call_tool(
             "rag_query",
             {
                 "query": "document",
-                "source_types": ["directory"],
+                "source_filters": ["directory"],
                 "limit": 10,
             },
         )
...
-        if rag_data["total_results"] > 0:
-            for result in rag_data["results"]:
-                metadata = result["metadata"]
+        if rag_data["total_matches"] > 0:
+            for match in rag_data["matches"]:
+                metadata = match["document"]["metadata"]
                 # Should be from directory source
                 assert metadata.get("source_type") == "directory"

Also applies to: 127-136
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +139 to +161
    async def test_get_rag_stats(self, mcp_client: Client):
        """Test getting RAG system statistics."""
        result = await mcp_client.call_tool("get_rag_stats", {})

        assert result.data is not None
        stats_data = result.data

        # Check basic structure
        assert "total_sources" in stats_data
        assert "total_chunks" in stats_data
        assert "source_types" in stats_data
        assert "collection_info" in stats_data

        # Values should be non-negative integers
        assert isinstance(stats_data["total_sources"], int)
        assert isinstance(stats_data["total_chunks"], int)
        assert stats_data["total_sources"] >= 0
        assert stats_data["total_chunks"] >= 0

        # Source types should be a dict
        assert isinstance(stats_data["source_types"], dict)

    @pytest.mark.integration
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

get_rag_stats result shape mismatches

Current tool returns rag_system, source_management, health_summary, quick_facts, timestamp. Update assertions accordingly.

-        # Check basic structure
-        assert "total_sources" in stats_data
-        assert "total_chunks" in stats_data
-        assert "source_types" in stats_data
-        assert "collection_info" in stats_data
+        # Check basic structure
+        assert "rag_system" in stats_data
+        assert "source_management" in stats_data
+        assert "health_summary" in stats_data
+        assert "quick_facts" in stats_data
+        assert "timestamp" in stats_data

Then adapt value checks to quick_facts and rag_system fields (e.g., vector_dimension lives under quick_facts).
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +163 to +188
    async def test_list_sources(self, mcp_client: Client):
        """Test listing sources in the database."""
        result = await mcp_client.call_tool(
            "list_sources",
            {
                "limit": 10,
                "offset": 0,
            },
        )

        assert result.data is not None
        sources_data = result.data

        # Check structure
        assert "sources" in sources_data
        assert "total_count" in sources_data
        assert "limit" in sources_data
        assert "offset" in sources_data

        # Should be valid values
        assert isinstance(sources_data["sources"], list)
        assert isinstance(sources_data["total_count"], int)
        assert sources_data["total_count"] >= 0
        assert sources_data["limit"] == 10
        assert sources_data["offset"] == 0

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

list_sources result shape mismatches

list_sources returns sources, pagination, statistics, filters_applied.

-        # Check structure
-        assert "sources" in sources_data
-        assert "total_count" in sources_data
-        assert "limit" in sources_data
-        assert "offset" in sources_data
+        # Check structure
+        assert "sources" in sources_data
+        assert "pagination" in sources_data
+        assert "statistics" in sources_data
+        assert "filters_applied" in sources_data
...
-        assert isinstance(sources_data["total_count"], int)
-        assert sources_data["total_count"] >= 0
-        assert sources_data["limit"] == 10
-        assert sources_data["offset"] == 0
+        p = sources_data["pagination"]
+        assert isinstance(p["total"], int)
+        assert p["total"] >= 0
+        assert p["limit"] == 10
+        assert p["offset"] == 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +191 to +233
    async def test_delete_source(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test deleting a source from the database."""
        directory_path = sample_text_files[0].parent

        # Index content first to have something to delete
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 1,
            },
        )

        assert crawl_result.data["success"] is True

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # List sources to get a source_id
        list_result = await mcp_client.call_tool("list_sources", {"limit": 1})
        sources = list_result.data["sources"]

        if len(sources) > 0:
            source_id = sources[0]["source_id"]

            # Delete the source
            delete_result = await mcp_client.call_tool(
                "delete_source", {"source_id": source_id}
            )

            assert delete_result.data is not None
            delete_data = delete_result.data

            assert "success" in delete_data
            assert delete_data["success"] is True
            assert "source_id" in delete_data
            assert delete_data["source_id"] == source_id

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

delete_source tool param/result mismatches

The tool expects source_url and returns {"source_url","documents_deleted","success","timestamp"}.

-            source_id = sources[0]["source_id"]
+            source_url = sources[0]["url"]
...
-            delete_result = await mcp_client.call_tool(
-                "delete_source", {"source_id": source_id}
-            )
+            delete_result = await mcp_client.call_tool(
+                "delete_source", {"source_url": source_url, "confirm": True}
+            )
...
-            assert "source_id" in delete_data
-            assert delete_data["source_id"] == source_id
+            assert "source_url" in delete_data
+            assert delete_data["source_url"] == source_url

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +235 to +260
    async def test_rag_query_parameter_validation(self, mcp_client: Client):
        """Test RAG query parameter validation."""
        # Test missing query
        with pytest.raises(Exception):
            await mcp_client.call_tool("rag_query", {})

        # Test invalid limit
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "limit": 0,
                },
            )

        # Test negative offset
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "offset": -1,
                },
            )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Parameter validation checks: assert specific ToolError and match current validation rules

    rag_query doesn’t take offset; use min_score bounds and high limit instead.
    Avoid catching broad Exception (Ruff B017).

-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool("rag_query", {})
...
-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool(
                 "rag_query",
                 {
                     "query": "test",
-                    "limit": 0,
+                    "limit": 1000,  # Exceeds max 100
                 },
             )
...
-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool(
                 "rag_query",
                 {
-                    "query": "test",
-                    "offset": -1,
+                    "query": "test",
+                    "min_score": 2.0,  # Out of range
                 },
             )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
feat: implement parallel pipeline architecture for extreme embedding …
c17894d
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Actionable comments posted: 9

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (4)

♻️ Duplicate comments (3)

📜 Review details



crawler_mcp/core/embeddings.py
Comment on lines 237 to 239
            logger.info(
                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch)"
                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {len(results) / processing_time:.1f} embeddings/sec"
            )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

🛠️ Refactor suggestion

Avoid divide-by-zero in throughput log; consider a monotonic clock.

If the request returns “instantaneously” (cached/local) processing_time can be 0, causing a ZeroDivisionError in the f-string. Also, prefer time.perf_counter() for timing.

Apply this diff to harden the log:

-            logger.info(
-                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {len(results) / processing_time:.1f} embeddings/sec"
-            )
+            logger.info(
+                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {(len(results) / (processing_time or 1e-6)):.1f} embeddings/sec"
+            )

Optional: switch this to debug if it’s noisy in prod.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/embeddings.py
Comment on lines +289 to +291
        # Start timing the entire batch process
        batch_start_time = time.time()

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Use a monotonic timer for batch measurements.

time.perf_counter() is the recommended monotonic high‑resolution clock for durations.

-        # Start timing the entire batch process
-        batch_start_time = time.time()
+        # Start timing the entire batch process
+        batch_start_time = time.perf_counter()

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/embeddings.py
Comment on lines +339 to +345
        # Log total batch processing time
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time
        logger.info(
            f"Completed embedding generation for {len(valid_texts)} texts in {total_batch_time:.2f}s - {len(valid_texts) / total_batch_time:.1f} embeddings/sec"
        )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Make batch throughput computation robust and reflect actual outputs.

    Guard against zero elapsed time.
    Use len(results) (produced items) rather than len(valid_texts) to avoid mismatch if something fails mid-batch.

-        # Log total batch processing time
-        batch_end_time = time.time()
-        total_batch_time = batch_end_time - batch_start_time
-        logger.info(
-            f"Completed embedding generation for {len(valid_texts)} texts in {total_batch_time:.2f}s - {len(valid_texts) / total_batch_time:.1f} embeddings/sec"
-        )
+        # Log total batch processing time
+        batch_end_time = time.perf_counter()
+        total_batch_time = batch_end_time - batch_start_time
+        elapsed = total_batch_time or 1e-6
+        logger.info(
+            f"Completed embedding generation for {len(results)} texts in {total_batch_time:.2f}s - {(len(results) / elapsed):.1f} embeddings/sec"
+        )

If you prefer to report “attempted texts per second,” keep len(valid_texts) but rename the metric accordingly to avoid confusion.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +803 to +809
                    # Generate deterministic ID if deduplication is enabled
                    if deduplication:
                        chunk_id = self._generate_deterministic_id(
                            page.url, composite_chunk_index
                        )
                        if legacy_chunk:
                            legacy_chunk_hash = legacy_chunk.get("content_hash", "")
                            if legacy_chunk_hash == content_hash:
                                # Same content, different ID format - skip but note the legacy chunk for replacement
                                chunks_skipped += 1
                                legacy_chunk_to_replace = legacy_chunk
                                legacy_chunks_to_delete.append(legacy_chunk["id"])
                                logger.debug(
                                    f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
                                )
                                should_skip = True
                        content_hash = self._calculate_content_hash(chunk_data["text"])
                    else:
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Type mismatch: deterministic ID generator expects int but receives str

composite_chunk_index is a string (e.g., "pageIdx_subIdx") but _generate_deterministic_id is typed to accept int for chunk_index. Update the signature to accept str | int and cast to str for hashing.

Apply:

-def _generate_deterministic_id(self, url: str, chunk_index: int) -> str:
+def _generate_deterministic_id(self, url: str, chunk_index: str | int) -> str:
@@
-        normalized_url = self._normalize_url(url)
-        id_string = f"{normalized_url}:{chunk_index}"
+        normalized_url = self._normalize_url(url)
+        id_string = f"{normalized_url}:{str(chunk_index)}"

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +830 to +846
                        # Backwards compatibility: check for legacy chunks with same content
                        elif use_backwards_compatibility and existing_chunks_list:
                            legacy_chunk = self._find_legacy_chunk_by_content(
                                existing_chunks_list, chunk_data["text"]
                            )
                            if legacy_chunk:
                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
                                if legacy_chunk_hash == content_hash:
                                    # Same content, different ID format - skip but note the legacy chunk for replacement
                                    chunks_skipped += 1
                                    legacy_chunk_to_replace = legacy_chunk
                                    legacy_chunks_to_delete.append(legacy_chunk["id"])
                                    logger.debug(
                                        f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
                                    )
                                    should_skip = True

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Bug: “skip unchanged content” deletes legacy chunks without creating replacements

When a legacy chunk with identical content is found, you append the legacy ID to legacy_chunks_to_delete and set should_skip = True, then continue. This deletes the legacy chunk but never creates/upserts a new chunk with the deterministic ID, causing data loss.

Fix by treating this as an upgrade (update), not a skip. Don’t delete immediately and don’t set should_skip. Let the later “update/upgrade” branch handle deletion after successfully creating the deterministic chunk.

-                            if legacy_chunk:
-                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
-                                if legacy_chunk_hash == content_hash:
-                                    # Same content, different ID format - skip but note the legacy chunk for replacement
-                                    chunks_skipped += 1
-                                    legacy_chunk_to_replace = legacy_chunk
-                                    legacy_chunks_to_delete.append(legacy_chunk["id"])
-                                    logger.debug(
-                                        f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
-                                    )
-                                    should_skip = True
+                            if legacy_chunk:
+                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
+                                if legacy_chunk_hash == content_hash:
+                                    # Same content, different ID format - perform upgrade (create new deterministic chunk)
+                                    legacy_chunk_to_replace = legacy_chunk
+                                    chunks_updated += 1
+                                    logger.debug(
+                                        f"Upgrading legacy chunk {legacy_chunk['id']} to deterministic ID {chunk_id} (content unchanged)"
+                                    )
+                                    # Do not skip; proceed to create doc_chunk and mark legacy for deletion later

@jmagar
crawler_mcp/core/rag.py
Comment on lines +1467 to +1476
                    async with storage_semaphore:
                        batch_id, chunk_batch = await storage_queue.get()

                        # Filter chunks with valid embeddings
                        valid_chunks = [
                            chunk
                            for chunk in chunk_batch
                            if chunk.embedding is not None
                        ]

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Semaphore held during awaited queue.get() reduces concurrency

You acquire storage_semaphore before awaiting storage_queue.get(). This holds a permit while idling, throttling concurrency. Get the batch from the queue first, then acquire the semaphore only around the upsert call.

Apply:

-                try:
-                    async with storage_semaphore:
-                        batch_id, chunk_batch = await storage_queue.get()
+                try:
+                    batch_id, chunk_batch = await storage_queue.get()
+                    async with storage_semaphore:
                         # Filter chunks with valid embeddings

@jmagar
crawler_mcp/core/rag.py
Outdated
docker-compose.yml
Outdated
Comment on lines 39 to 47
      - "131072"
      - --max-batch-requests
      - "512"
      - "128"
      - --max-client-batch-size
      - "512"
      - "128"
      - --pooling
      - "last-token"
      - --tokenization-workers
      - "12"
      - "8"
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Parameterize aggressive TEI batching knobs; leave room to tune per environment.

The current values (e.g., max-batch-tokens=131072, max-*-size=128) can amplify peak memory and tail latencies under load. Make them env-driven with sane defaults so CI/dev/prod can tune safely.

-      - "131072"
+      - "${TEI_MAX_BATCH_TOKENS:-131072}"
@@
-      - "128"
+      - "${TEI_MAX_BATCH_REQUESTS:-128}"
@@
-      - "128"
+      - "${TEI_MAX_CLIENT_BATCH_SIZE:-128}"
@@
-      - "8"
+      - "${TEI_TOKENIZATION_WORKERS:-8}"

Action to verify: confirm these defaults fit your GPU memory and target tail latency under expected sequence lengths and traffic.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
docker-compose.yml
Comment on lines +52 to +53
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
      - OMP_NUM_THREADS=8
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Make CUDA/OMP knobs configurable to avoid CPU/GPU over/under-subscription.

Hard-coding PYTORCH_CUDA_ALLOC_CONF and OMP_NUM_THREADS may be suboptimal across machines. Parameterize with env defaults.

-      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
-      - OMP_NUM_THREADS=8
+      - PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:1024}
+      - OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

Consider validating that OMP_NUM_THREADS aligns with available vCPUs and that allocator settings don’t increase fragmentation for your model’s peak batch.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
feat: add browser pool management and optimize crawler performance
9543105
coderabbitai[bot]
coderabbitai bot reviewed Aug 21, 2025
coderabbitai bot left a comment

Actionable comments posted: 27

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (8)

♻️ Duplicate comments (14)

📜 Review details

🔇 Additional comments (4)

.env.example
@@ -34,6 +34,7 @@ EMBEDDING_MAX_LENGTH=32000
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_RETRIES=2
EMBEDDING_WORKERS=4
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Good addition; document bounds and interplay with TEI/client concurrency.

EMBEDDING_WORKERS=4 mirrors the new settings field (ge=1, le=16). Add a short comment in this file noting the valid range [1–16] and that raising this above the TEI server’s concurrency and/or CPU cores can increase tail latencies.

Apply this minimal diff to self-document the knob:

 EMBEDDING_MAX_RETRIES=2
-EMBEDDING_WORKERS=4
+EMBEDDING_WORKERS=4  # [1-16] Parallel embedding workers; keep <= CPU cores and TEI concurrency

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
Comment on lines +45 to +53
    qdrant_connection_pool_size: int = Field(
        default=16, alias="QDRANT_CONNECTION_POOL_SIZE", ge=1, le=32
    )
    qdrant_batch_size: int = Field(
        default=256, alias="QDRANT_BATCH_SIZE", ge=64, le=512
    )
    qdrant_prefetch_size: int = Field(
        default=1024, alias="QDRANT_PREFETCH_SIZE", ge=256, le=2048
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Qdrant client tunables: validate combinations and document memory impact.

Pool size, batch size, and prefetch size interplay directly with client memory and open connections. Add a validator that warns when qdrant_batch_size * qdrant_prefetch_size is unusually high (e.g., > 512k vectors per page) to avoid unexpected spikes.

Example:

@@
     qdrant_prefetch_size: int = Field(
         default=1024, alias="QDRANT_PREFETCH_SIZE", ge=256, le=2048
     )
+
+    @model_validator(mode="after")
+    def _validate_qdrant_tunables(self) -> "CrawlerrSettings":
+        max_inflight = self.qdrant_batch_size * self.qdrant_prefetch_size
+        if max_inflight > 512 * 1024:
+            import logging
+            logging.getLogger(__name__).warning(
+                "Qdrant inflight capacity (%d) is very high; expect elevated memory/network usage",
+                max_inflight,
+            )
+        return self

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
@@ -58,6 +67,7 @@ class CrawlerrSettings(BaseSettings):
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(default=2, alias="EMBEDDING_MAX_RETRIES")
    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Cap embedding_workers by available CPU cores; prevent oversubscription.

At startup, clamp to os.cpu_count() and optionally to TEI_MAX_CONCURRENT_REQUESTS to avoid excessive context-switching and backpressure.

@@
-    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
+    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
+
+    @model_validator(mode="after")
+    def _validate_embedding_workers(self) -> "CrawlerrSettings":
+        import os, logging
+        cores = max(1, os.cpu_count() or 1)
+        if self.embedding_workers > cores:
+            logging.getLogger(__name__).warning(
+                "EMBEDDING_WORKERS=%d exceeds CPU cores=%d; clamping.",
+                self.embedding_workers, cores,
+            )
+            self.embedding_workers = cores
+        return self

@jmagar
crawler_mcp/config.py
Comment on lines +134 to +137
    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
    crawl_virtual_scroll_batch_size: int = Field(
        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE"
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Scroll tuning defaults: ensure consumers actually use these fields.

These are newly exposed but I don’t see validations or bounds. Consider ge/le constraints (e.g., delay >= 0, batch_size >= 1) to prevent negative values via env overrides.

-    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
-    crawl_virtual_scroll_batch_size: int = Field(
-        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE"
-    )
+    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY", ge=0, le=1000)
+    crawl_virtual_scroll_batch_size: int = Field(
+        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE", ge=1, le=100
+    )

@jmagar
crawler_mcp/config.py
Comment on lines +144 to 156
    # High-Performance Configuration (i7-13700k + RTX 4070)
    browser_pool_size: int = Field(default=8, alias="BROWSER_POOL_SIZE", ge=1, le=16)
    file_processing_threads: int = Field(
        default=16, alias="FILE_PROCESSING_THREADS", ge=1, le=24
    )
    crawl_concurrency: int = Field(default=12, alias="CRAWL_CONCURRENCY", ge=1, le=50)
    content_cache_size_gb: int = Field(
        default=8, alias="CONTENT_CACHE_SIZE_GB", ge=1, le=16
    )
    gpu_memory_fraction: float = Field(
        default=0.95, alias="GPU_MEMORY_FRACTION", ge=0.1, le=1.0
    )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Hardware-specific defaults: avoid locking config to a single SKU.

The “i7-13700k + RTX 4070” comment plus high defaults (e.g., 16 file threads) can be aggressive on smaller machines. Either (a) lower defaults to safe values and document scaling, or (b) auto-scale by cores and memory.

Minimal auto-scaling:

-    browser_pool_size: int = Field(default=8, alias="BROWSER_POOL_SIZE", ge=1, le=16)
-    file_processing_threads: int = Field(
-        default=16, alias="FILE_PROCESSING_THREADS", ge=1, le=24
-    )
+    browser_pool_size: int = Field(default= max(1, (os.cpu_count() or 8)//2 ),
+                                   alias="BROWSER_POOL_SIZE", ge=1, le=16)
+    file_processing_threads: int = Field(default= max(4, (os.cpu_count() or 8) ),
+                                         alias="FILE_PROCESSING_THREADS", ge=1, le=24)

Note: import os at module top if adopting this.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
Comment on lines +171 to +183
    # Alternative crawling approach settings
    use_arun_many_for_sitemaps: bool = Field(
        default=False,
        alias="USE_ARUN_MANY_FOR_SITEMAPS",
        description="Use arun_many() with sitemap URLs instead of BFSDeepCrawlStrategy",
    )
    max_concurrent_sessions: int = Field(
        default=20,
        alias="CRAWL_MAX_CONCURRENT_SESSIONS",
        ge=1,
        le=50,
        description="Maximum concurrent sessions for arun_many() approach",
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Mutual exclusivity and defaults for sitemap crawling path.

When USE_ARUN_MANY_FOR_SITEMAPS=True, clarify precedence vs the default BFS strategy. Consider a validator to warn if max_concurrent_sessions is set but the feature is disabled, or if both strategies are toggled on by mistake upstream.
@jmagar
crawler_mcp/config.py
Comment on lines +185 to +201
    # Crawl4AI Performance Optimizations
    crawl_text_mode: bool = Field(
        default=False,
        alias="CRAWL_TEXT_MODE",
        description="Enable text-only mode for 3-4x faster crawling (disables images)",
    )
    crawl_light_mode: bool = Field(
        default=True,
        alias="CRAWL_LIGHT_MODE",
        description="Enable light mode to optimize browser performance",
    )
    use_lxml_strategy: bool = Field(
        default=True,
        alias="USE_LXML_STRATEGY",
        description="Use LXMLWebScrapingStrategy for 20x faster parsing",
    )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Light/text/LXML feature toggles: add a compatibility validator.

Enabling crawl_text_mode=True with crawl_extract_media=True would be contradictory. Add a post-model validator to warn or auto-disable the conflicting option. Similarly, LXML requires specific parsers installed—validate availability at startup and downgrade gracefully.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +24 to +33
    async def initialize(self) -> None:
        """Initialize browser pool with hardware-optimized config."""
        if self.is_initialized:
            return

        try:
            # Import here to avoid circular imports
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            # Hardware-optimized browser configuration
@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Guard initialization with a lock; fix race when two coroutines call acquire() concurrently.

Without synchronization, two tasks can run initialize() simultaneously and create duplicate browser sets.

Apply:

 class HighPerformanceBrowserPool:
@@
     def __init__(self, pool_size: int = 8):
         self.pool_size = pool_size
         self.browsers: list[Any] = []  # Will be AsyncWebCrawler instances
         self.available_browsers: asyncio.Queue[Any] = asyncio.Queue(maxsize=pool_size)
         self.is_initialized = False
+        self._init_lock = asyncio.Lock()
         self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
@@
     async def initialize(self) -> None:
         """Initialize browser pool with hardware-optimized config."""
-        if self.is_initialized:
-            return
+        if self.is_initialized:
+            return
+        async with self._init_lock:
+            if self.is_initialized:
+                return

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +31 to +33
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            # Hardware-optimized browser configuration
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Safer imports and clearer error if crawl4ai is missing.

Importing from crawl4ai inside the method is good to avoid cycles, but improve diagnostics on ImportError and prefer local Protocols for type hints under TYPE_CHECKING.

-            from crawl4ai import AsyncWebCrawler, BrowserConfig
+            try:
+                from crawl4ai import AsyncWebCrawler, BrowserConfig
+            except Exception as e:
+                self.logger.error("crawl4ai is not installed or incompatible: %s", e, exc_info=True)
+                raise

For better typing without runtime dependency:

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..types.crawl4ai_types import AsyncWebCrawler as _AsyncWebCrawler, BrowserConfig as _BrowserConfig

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +34 to +63
            browser_config = BrowserConfig(
                headless=True,
                browser_type="chromium",
                verbose=False,
                # RTX 4070 + i7-13700k optimized Chrome flags
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu-sandbox",
                    "--max_old_space_size=4096",  # 4GB per browser instance
                    "--js-flags=--max-old-space-size=4096",
                    "--renderer-process-limit=4",  # Limit renderer processes
                    "--process-per-site",
                    "--aggressive-cache-discard",
                    "--memory-pressure-off",
                    "--enable-gpu-rasterization",
                    "--enable-zero-copy",
                    "--enable-oop-rasterization",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--no-zygote",  # Better for concurrent instances
                    # Performance optimizations for high-end hardware
                    "--max-renderer-processes=4",
                    "--renderer-process-limit=4",
                    "--enable-accelerated-2d-canvas",
                    "--enable-gpu-compositing",
                ],
            )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Chrome flags are overly aggressive/hardware-specific; make them settings-driven and drop risky ones.

Flags like --no-sandbox and duplicate renderer limits can hurt stability; memory V8 flags may be ineffective. Honor settings (light/text mode) and keep a minimal, portable set.

-            browser_config = BrowserConfig(
+            # TODO: derive from settings (light_mode, block images/media) if available
+            browser_config = BrowserConfig(
                 headless=True,
                 browser_type="chromium",
                 verbose=False,
                 # RTX 4070 + i7-13700k optimized Chrome flags
                 extra_args=[
-                    "--no-sandbox",
-                    "--disable-dev-shm-usage",
-                    "--disable-gpu-sandbox",
-                    "--max_old_space_size=4096",  # 4GB per browser instance
-                    "--js-flags=--max-old-space-size=4096",
-                    "--renderer-process-limit=4",  # Limit renderer processes
-                    "--process-per-site",
-                    "--aggressive-cache-discard",
-                    "--memory-pressure-off",
-                    "--enable-gpu-rasterization",
-                    "--enable-zero-copy",
-                    "--enable-oop-rasterization",
-                    "--disable-background-timer-throttling",
-                    "--disable-backgrounding-occluded-windows",
-                    "--disable-renderer-backgrounding",
-                    "--disable-features=TranslateUI",
-                    "--no-zygote",  # Better for concurrent instances
-                    # Performance optimizations for high-end hardware
-                    "--max-renderer-processes=4",
-                    "--renderer-process-limit=4",
-                    "--enable-accelerated-2d-canvas",
-                    "--enable-gpu-compositing",
+                    "--disable-dev-shm-usage",
+                    "--disable-background-timer-throttling",
+                    "--disable-renderer-backgrounding",
+                    "--disable-backgrounding-occluded-windows",
+                    "--enable-zero-copy",
                 ],
             )

If sandbox must be disabled for containers, gate --no-sandbox behind an env/settings flag and document the risk.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +69 to +81
            # Create browser instances
            for i in range(self.pool_size):
                try:
                    browser = AsyncWebCrawler(config=browser_config)
                    await browser.__aenter__()
                    self.browsers.append(browser)
                    await self.available_browsers.put(browser)
                    self.logger.debug(f"Initialized browser {i + 1}/{self.pool_size}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize browser {i + 1}: {e}")
                    # Continue with fewer browsers rather than failing completely

            if not self.browsers:
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Initialize browsers with partial failure handling and metrics; skip queue put if aenter fails.

You already catch and continue; add exc_info=True and count successes/failures.

-                except Exception as e:
-                    self.logger.error(f"Failed to initialize browser {i + 1}: {e}")
+                except Exception as e:
+                    self.logger.error("Failed to initialize browser %d: %s", i + 1, e, exc_info=True)

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +94 to +107
    async def acquire(self) -> Any:
        """Get browser from pool."""
        if not self.is_initialized:
            await self.initialize()

        browser = await self.available_browsers.get()
        self.logger.debug("Browser acquired from pool")
        return browser

    async def release(self, browser: Any) -> None:
        """Return browser to pool."""
        try:
            await self.available_browsers.put(browser)
            self.logger.debug("Browser returned to pool")
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Ergonomics: provide an async lease contextmanager to ensure release on exceptions.

Avoids borrowers forgetting to call release().

Add:

@@
     async def acquire(self) -> Any:
@@
         return browser
@@
     async def release(self, browser: Any) -> None:
@@
         except Exception as e:
             self.logger.error(f"Failed to return browser to pool: {e}")
+
+    async def lease(self):
+        """Async context manager yielding a browser and auto-releasing it."""
+        browser = await self.acquire()
+        try:
+            yield browser
+        finally:
+            await self.release(browser)

Usage:

from contextlib import asynccontextmanager
HighPerformanceBrowserPool.lease = asynccontextmanager(HighPerformanceBrowserPool.lease)

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +111 to +133
    async def cleanup(self) -> None:
        """Cleanup all browsers."""
        self.logger.info("Cleaning up browser pool")

        # Close all browsers
        for browser in self.browsers:
            try:
                await browser.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}")

        self.browsers.clear()

        # Clear the queue
        while not self.available_browsers.empty():
            try:
                self.available_browsers.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.is_initialized = False
        self.logger.info("Browser pool cleanup completed")

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Close browsers concurrently and drain queue robustly.

Speed up shutdown and avoid leaving stale references in the queue.

-        for browser in self.browsers:
-            try:
-                await browser.__aexit__(None, None, None)
-            except Exception as e:
-                self.logger.error(f"Error closing browser: {e}")
+        await asyncio.gather(
+            *[
+                browser.__aexit__(None, None, None)
+                for browser in list(self.browsers)
+            ],
+            return_exceptions=True,
+        )
+        # Log any close errors
+        # (gather already returned exceptions; consider iterating for logging if needed)
@@
-        while not self.available_browsers.empty():
-            try:
-                self.available_browsers.get_nowait()
-            except asyncio.QueueEmpty:
-                break
+        try:
+            while True:
+                self.available_browsers.get_nowait()
+        except asyncio.QueueEmpty:
+            pass

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +158 to +165
async def get_browser_pool(pool_size: int = 8) -> HighPerformanceBrowserPool:
    """Get or create the global browser pool."""
    global _browser_pool

    if _browser_pool is None:
        _browser_pool = HighPerformanceBrowserPool(pool_size=pool_size)

    return _browser_pool
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Pool size argument is ignored after first creation; warn on mismatch.

If a caller requests a different size later, log it to avoid confusion.

     if _browser_pool is None:
         _browser_pool = HighPerformanceBrowserPool(pool_size=pool_size)
+    elif _browser_pool.pool_size != pool_size:
+        logger.warning(
+            "Browser pool already initialized with size=%d (requested %d); using existing pool.",
+            _browser_pool.pool_size, pool_size
+        )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines 1016 to +1023
            stored_count = await self.vector_service.upsert_documents(document_chunks)
            logger.info(f"Stored {stored_count} document chunks in vector database")

            # Log vector storage time
            storage_end_time = time.time()
            storage_duration = storage_end_time - storage_start_time
            logger.info(
                f"Stored {stored_count} document chunks in vector database in {storage_duration:.2f}s"
            )
@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Duplicate storage operation after pipeline already stored chunks

After the parallel pipeline completes (which already stores chunks via upsert_documents in the storage worker), this code calls upsert_documents again on the same chunks, causing duplicate storage operations.

Remove the duplicate storage operation since the pipeline already handles it:

-        # Store in vector database
-        if progress_callback:
-            progress_callback(
-                total_pages + 1,
-                total_pages + 2,
-                f"Storing {len(document_chunks)} embeddings in vector database",
-            )
-
-        try:
-            # Start timing vector storage
-            storage_start_time = time.time()
-
-            stored_count = await self.vector_service.upsert_documents(document_chunks)
-
-            # Log vector storage time
-            storage_end_time = time.time()
-            storage_duration = storage_end_time - storage_start_time
-            logger.info(
-                f"Stored {stored_count} document chunks in vector database in {storage_duration:.2f}s"
-            )
-
-        except Exception as e:
-            logger.error(f"Error storing embeddings: {e}")
-            raise ToolError(f"Failed to store embeddings: {e!s}") from e
+        # Storage is handled by the pipeline's storage worker
+        stored_count = len(document_chunks)

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +119 to 124
            # Use ThreadPoolExecutor for maximum CPU utilization
            if progress_callback:
                progress_callback(
                    0, len(files), "Starting high-performance file processing..."
                )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Progress reporting: consider emitting progress as files complete.

You set an initial message but never update the numerator. If the callback supports it, increment on each processed file to reflect true progress.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +125 to +135
            batch_results = await self._process_files_highly_concurrent(
                files, directory
            )

                # Small delay to prevent overwhelming the system
                if i + batch_size < len(files):
                    await asyncio.sleep(0.01)
            for batch_result in batch_results:
                if isinstance(batch_result, PageContent):
                    pages.append(batch_result)
                    total_bytes += len(batch_result.content)
                elif isinstance(batch_result, Exception):
                    errors.append(str(batch_result))

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Count bytes, not characters, for total_bytes.

len(string) counts code points; use UTF-8 byte length for network/storage stats.

-                    total_bytes += len(batch_result.content)
+                    total_bytes += len(batch_result.content.encode("utf-8", errors="ignore"))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +136 to 144
            # Memory pressure check after processing
            if (
                self.memory_manager
                and await self.memory_manager.check_memory_pressure()
            ):
                self.logger.warning(
                    "Memory pressure detected during directory processing"
                )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Memory pressure check is post hoc only; consider occasional mid-flight checks for very large directories.

For very large file sets, a periodic check (e.g., every N files) can prevent overshooting memory thresholds. Optional, depending on expected workloads.
@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +410 to 449
    async def _process_files_highly_concurrent(
        self, file_paths: list[Path], base_directory: Path
    ) -> list[PageContent | Exception]:
        """Process a batch of files concurrently."""
        semaphore = asyncio.Semaphore(5)  # Limit concurrent file operations
        """Process files with full CPU utilization using ThreadPoolExecutor."""
        from ..config import settings

        async def process_single_file(file_path: Path) -> PageContent | Exception:
            async with semaphore:
                try:
                    return await self._process_single_file(file_path, base_directory)
                except Exception as e:
                    return Exception(f"Error processing {file_path}: {e}")
        # Use configured thread count (default 16 for i7-13700k)
        max_workers = getattr(settings, "file_processing_threads", 16)

        self.logger.info(
            f"Processing {len(file_paths)} files with {max_workers} threads"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()

            # Process files in parallel using all available threads
            tasks = [
                loop.run_in_executor(
                    executor, self._process_single_file_sync, file_path, base_directory
                )
                for file_path in file_paths
            ]

            # Use asyncio.gather for true parallelism
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter and return results
            processed_results = []
            for result in results:
                if isinstance(result, PageContent | Exception):
                    processed_results.append(result)
                else:
                    # Handle any unexpected return types
                    processed_results.append(
                        Exception(f"Unexpected result type: {type(result)}")
                    )

            return processed_results

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Tests: add coverage for error returns and mixed results from the thread pool.

Ensure that Exceptions returned by _process_single_file_sync are collected into errors and that stats are computed correctly when some files fail.

Would you like me to add unit tests targeting:

    isinstance fix behavior (no TypeError)
    byte-accurate total_bytes
    mixed success/failure aggregation?

🤖 Prompt for AI Agents

⚠️ Potential issue

Bug: isinstance with PEP 604 union is invalid; use a tuple. Also use get_running_loop().

isinstance(result, PageContent | Exception) raises TypeError at runtime. And inside an async context, prefer asyncio.get_running_loop().

Apply:

-        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
-            loop = asyncio.get_event_loop()
+        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
+            loop = asyncio.get_running_loop()
@@
-            processed_results = []
-            for result in results:
-                if isinstance(result, PageContent | Exception):
-                    processed_results.append(result)
-                else:
-                    # Handle any unexpected return types
-                    processed_results.append(
-                        Exception(f"Unexpected result type: {type(result)}")
-                    )
+            processed_results: list[PageContent | Exception] = []
+            for result in results:
+                if isinstance(result, (PageContent, Exception)):
+                    processed_results.append(result)
+                else:
+                    processed_results.append(Exception(f"Unexpected result type: {type(result)}"))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +133 to 168
            # High-performance browser config optimized for i7-13700k + RTX 4070
            browser_config = BrowserConfig(
                headless=settings.crawl_headless,
                browser_type=settings.crawl_browser,
                light_mode=True,  # Let Crawl4AI optimize performance
                text_mode=getattr(settings, "crawl_block_images", False),
                light_mode=getattr(
                    settings, "crawl_light_mode", True
                ),  # Optimized performance mode
                text_mode=getattr(
                    settings, "crawl_text_mode", False
                ),  # 3-4x faster when enabled
                verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
                # NO extra_args - avoid flag conflicts
                # Aggressive performance settings
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    # GPU Acceleration for RTX 4070
                    "--enable-gpu",  # Enable GPU acceleration
                    "--enable-accelerated-2d-canvas",  # GPU for 2D canvas
                    "--enable-gpu-compositing",  # GPU for compositing
                    "--enable-gpu-rasterization",  # GPU for rasterization
                    "--ignore-gpu-blocklist",  # Use GPU even if blocklisted
                    "--disable-gpu-sandbox",  # Remove GPU sandbox restrictions
                    "--enable-zero-copy",  # Zero-copy GPU textures
                    "--use-gl=egl",  # Use EGL for headless GPU
                    "--max_old_space_size=4096",  # 4GB memory per browser
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--aggressive-cache-discard",
                    "--memory-pressure-off",
                    # Network optimizations for faster page loads
                    "--max-connections-per-host=30",  # Match semaphore_count
                    "--enable-quic",
                    "--enable-tcp-fast-open",
                ],
            )
@coderabbitai coderabbitai bot Aug 21, 2025

💡 Verification agent
❓ Verification inconclusive

Harden BrowserConfig GPU flags and remove non-Chromium switch

We verified that headless Chromium always disables GPU by default—forcing real-GPU flags on a GPU-less CI or container will cause the GPU process to crash (e.g. “Exiting GPU process due to errors during initialization”) (github.com, reddit.com).
We also confirmed that --max_old_space_size is a V8/Node.js heap flag (passed via --js-flags) and is not a recognized Chromium switch; using it directly will be ignored or emit warnings (peter.sh, bigbinary.com).

Please update crawler_mcp/crawlers/web.py (around lines 133–168) as follows:

• Gate all GPU-specific switches behind a new crawl_enable_gpu setting
• Remove the standalone --max_old_space_size=4096 argument

             browser_config = BrowserConfig(
                 headless=settings.crawl_headless,
                 browser_type=settings.crawl_browser,
                 light_mode=getattr(
                     settings, "crawl_light_mode", True
                 ),  # Optimized performance mode
                 text_mode=getattr(
                     settings, "crawl_text_mode", False
                 ),  # 3-4x faster when enabled
                 verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
-                # Aggressive performance settings
-                extra_args=[
+                # Aggressive performance settings
+                extra_args=[
                     "--no-sandbox",
                     "--disable-dev-shm-usage",
-                    # GPU Acceleration for RTX 4070
-                    "--enable-gpu",  # Enable GPU acceleration
-                    "--enable-accelerated-2d-canvas",  # GPU for 2D canvas
-                    "--enable-gpu-compositing",  # GPU for compositing
-                    "--enable-gpu-rasterization",  # GPU for rasterization
-                    "--ignore-gpu-blocklist",  # Use GPU even if blocklisted
-                    "--disable-gpu-sandbox",  # Remove GPU sandbox restrictions
-                    "--enable-zero-copy",  # Zero-copy GPU textures
-                    "--use-gl=egl",  # Use EGL for headless GPU
-                    "--max_old_space_size=4096",  # 4GB memory per browser
+                    # Conditionally enable GPU flags only when a GPU is available
+                    *(
+                        [
+                            "--enable-gpu",
+                            "--enable-accelerated-2d-canvas",
+                            "--enable-gpu-compositing",
+                            "--enable-gpu-rasterization",
+                            "--ignore-gpu-blocklist",
+                            "--disable-gpu-sandbox",
+                            "--enable-zero-copy",
+                            "--use-gl=egl",
+                        ]
+                        if getattr(settings, "crawl_enable_gpu", False)
+                        else []
+                    ),
                     "--disable-background-timer-throttling",
                     "--disable-backgrounding-occluded-windows",
                     "--disable-renderer-backgrounding",
                     "--aggressive-cache-discard",
                     "--memory-pressure-off",
                     # Network optimizations for faster page loads
                     "--max-connections-per-host=30",  # Match semaphore_count
                     "--enable-quic",
                     "--enable-tcp-fast-open",
                 ],
             )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +209 to 229
                                        raise
                            else:
                                errors.append(
                                    f"Failed to crawl {result.url}: {result.error_message}"
                                )
                            if len(successful_results) >= max_pages:
                                break
                    elif isinstance(crawl_result, Crawl4aiResult):
                        # Single result case (when stream=False and no deep crawl)
                        crawl_count = 1
                        self.logger.info(
                            "Processing single CrawlResult (stream=False, no deep crawl)"
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed single result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                            )
                    elif hasattr(crawl_result, "success") and hasattr(
                        crawl_result, "url"
                    ):
                        # Handle CrawlResultContainer and other container types
                        crawl_count = 1
                        self.logger.info(
                            "Processing container result type: %s",
                            type(crawl_result).__name__,
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed container result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                "Failed to crawl {}: {}".format(
                                    crawl_result.url,
                                    getattr(
                                        crawl_result, "error_message", "Unknown error"
                                    ),
                                )
                            )
                    else:
                        raise Exception(
                            f"Unexpected crawl result type: {type(crawl_result)} (deep_crawl={run_config.deep_crawl_strategy is not None})"
                        )
            if (
                getattr(settings, "use_arun_many_for_sitemaps", False)
                and sitemap_seeds
                and len(sitemap_seeds) > 1
            ):
                self.logger.info(
                    f"Using arun_many() approach with {len(sitemap_seeds)} sitemap URLs (max_concurrent_sessions={getattr(settings, 'max_concurrent_sessions', 20)})"
                )
                successful_results = await self._crawl_using_arun_many(
                    browser, sitemap_seeds, run_config, request, progress_callback
                )
            else:
                self.logger.info(
                    "Using BFSDeepCrawlStrategy approach with async iteration..."
                )
                successful_results = await self._crawl_using_deep_strategy(
                    browser, first_url, run_config, max_pages
                )

@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Fix crawl_count and logging; remove unused counter

crawl_count is initialized (Line 210) but never incremented, so the final log is misleading.

Apply:

-            crawl_count = 0
+            # Defer computing crawl_count until after we have results

-                successful_results = await self._crawl_using_arun_many(
+                successful_results, errors = await self._crawl_using_arun_many(
                     browser, sitemap_seeds, run_config, request, progress_callback
                 )

-                successful_results = await self._crawl_using_deep_strategy(
+                successful_results, errors = await self._crawl_using_deep_strategy(
                     browser, first_url, run_config, max_pages
                 )

-            self.logger.info(
-                "Crawl loop completed: %s results processed, %s successful pages",
-                crawl_count,
-                len(pages),
-            )
+            self.logger.info(
+                "Crawl loop completed: %s results processed, %s successful pages",
+                len(successful_results),
+                len(pages),
+            )

Also applies to: 263-267
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +219 to +221
                successful_results = await self._crawl_using_arun_many(
                    browser, sitemap_seeds, run_config, request, progress_callback
                )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Wire returned errors into execute() statistics

After propagating errors from the two crawl paths, use them to compute totals.

-                successful_results = await self._crawl_using_arun_many(
+                successful_results, errors = await self._crawl_using_arun_many(
                     browser, sitemap_seeds, run_config, request, progress_callback
                 )
@@
-                successful_results = await self._crawl_using_deep_strategy(
+                successful_results, errors = await self._crawl_using_deep_strategy(
                     browser, first_url, run_config, max_pages
                 )

No code diff needed in 234-247; you already populate pages from successful_results. Ensure statistics uses len(errors) below.

Also applies to: 226-228, 234-247, 273-283
@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +231 to 233
            pages = []
            errors = []

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Remove redundant reinitialization of pages/errors

pages and errors were already initialized before the crawl; re-initializing here serves no purpose and risks losing error state if you ever populate it earlier.

-            # Process crawling results
-            pages = []
-            errors = []
+            # Process crawling results

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +760 to 768
            # Minimal BFS strategy configuration for maximum crawling
            # Omit filter_chain entirely (don't set to None) as per crawl4ai docs
            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                max_depth=max_depth,
                include_external=False,
                max_pages=max_pages,
                # filter_chain=filter_chain,  # Disabled - even minimal filters break it
                # Omit filter_chain - it defaults to empty FilterChain() which allows all URLs
                # Omit score_threshold - it defaults to -infinity which allows all URLs
            )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Include user include/exclude patterns; filter_chain currently ignored

You build filter_chain but don’t pass it to BFSDeepCrawlStrategy, so include/exclude patterns are ignored. Wire it in with a safe fallback.

-            # Minimal BFS strategy configuration for maximum crawling
-            # Omit filter_chain entirely (don't set to None) as per crawl4ai docs
-            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
-                max_depth=max_depth,
-                include_external=False,
-                max_pages=max_pages,
-                # Omit filter_chain - it defaults to empty FilterChain() which allows all URLs
-                # Omit score_threshold - it defaults to -infinity which allows all URLs
-            )
+            # Prefer passing filter_chain when available (with defensive fallback)
+            try:
+                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                    filter_chain=filter_chain if filter_chain else None,
+                )
+            except TypeError:
+                # Older versions may not accept filter_chain kwarg
+                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                )

Also applies to: 773-777
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +906 to +1008
        """Crawl using BFSDeepCrawlStrategy with async generator."""
        successful_results = []
        errors = []

        with suppress_stdout():
            try:
                # Get result from arun - type depends on config.stream setting
                self.logger.info(
                    "About to call browser.arun with stream=%s", run_config.stream
                )
                crawl_result = await browser.arun(url=first_url, config=run_config)
                self.logger.info("browser.arun completed successfully")

                # Debug: Log the actual type we received
                self.logger.info(
                    "CRAWL DEBUG: crawl_result type = %s, stream=%s, deep_crawl=%s",
                    type(crawl_result).__name__,
                    run_config.stream,
                    run_config.deep_crawl_strategy is not None,
                )

                # Handle different return types based on deep crawl strategy and stream setting
                if hasattr(crawl_result, "__aiter__"):
                    # AsyncGenerator case (when stream=True)
                    self.logger.info(
                        "Processing AsyncGenerator results (stream=True mode) - starting iteration"
                    )
                    generator_count = 0
                    async for result in crawl_result:
                        generator_count += 1

                        self.logger.info(
                            f"AsyncGenerator yielded result #{generator_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                        )

                        # Pre-check for unexpected types (defensive programming)
                        if isinstance(result, int):
                            self.logger.warning(
                                "Received integer %d instead of CrawlResult in streaming mode, skipping",
                                result,
                            )
                            continue

                        # Ensure result is a CrawlResult object
                        if not hasattr(result, "success"):
                            self.logger.warning(
                                "Received unexpected type %s in streaming mode, skipping",
                                type(result).__name__,
                            )
                            continue

                        if result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(result)
                                successful_results.append(sanitized_result)
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        result.url,
                                    )
                                    continue
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {result.url}: {result.error_message}"
                            )
                        if len(successful_results) >= max_pages:
                            self.logger.info(
                                f"Breaking from AsyncGenerator loop: reached max_pages ({max_pages})"
                            )
                            break

                    self.logger.info(
                        f"AsyncGenerator iteration completed: yielded {generator_count} results, {len(successful_results)} successful"
                    )
                else:
                    # Handle single result or list cases
                    self.logger.info(
                        f"Received non-async result: {type(crawl_result).__name__}"
                    )
                    if hasattr(crawl_result, "success"):
                        if crawl_result.success:
                            sanitized_result = self._sanitize_crawl_result(crawl_result)
                            successful_results.append(sanitized_result)
                        else:
                            errors.append(
                                f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                            )

            except Exception as e:
                self.logger.error(f"Deep crawl strategy failed: {e}", exc_info=True)
                errors.append(str(e))

        return successful_results

@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Propagate errors up to execute(); currently dropped

_errors collected here are never surfaced; execute() always reports 0 failures. Return both successes and errors so statistics.total_pages_failed is accurate.

-    async def _crawl_using_deep_strategy(
-        self, browser: Any, first_url: str, run_config: Any, max_pages: int
-    ) -> list[Any]:
+    async def _crawl_using_deep_strategy(
+        self, browser: Any, first_url: str, run_config: Any, max_pages: int
+    ) -> tuple[list[Any], list[str]]:
@@
-        return successful_results
+        return successful_results, errors

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +1010 to +1099
        self,
        browser: Any,
        sitemap_urls: list[str],
        run_config: Any,
        request: Any,
        progress_callback: Any,
    ) -> list[Any]:
        """Crawl using arun_many() with discovered sitemap URLs."""
        from crawl4ai import MemoryAdaptiveDispatcher  # type: ignore

        successful_results = []
        max_pages = request.max_pages or len(sitemap_urls)
        max_concurrent = getattr(settings, "max_concurrent_sessions", 20)

        # Limit sitemap URLs to max_pages
        urls_to_crawl = sitemap_urls[:max_pages]

        self.logger.info(
            f"Creating MemoryAdaptiveDispatcher with max_session_permit={max_concurrent}"
        )

        # Create dispatcher for memory-adaptive concurrency
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=getattr(settings, "crawl_memory_threshold", 80.0),
            check_interval=0.5,
            max_session_permit=max_concurrent,
        )

        # Remove deep_crawl_strategy to avoid recursion and set streaming
        batch_config = (
            run_config.clone() if hasattr(run_config, "clone") else run_config
        )
        if hasattr(batch_config, "deep_crawl_strategy"):
            batch_config.deep_crawl_strategy = None
        batch_config.stream = True

        self.logger.info(f"Starting arun_many with {len(urls_to_crawl)} URLs")

        with suppress_stdout():
            try:
                # Use arun_many for concurrent crawling
                results_generator = await browser.arun_many(
                    urls=urls_to_crawl, config=batch_config, dispatcher=dispatcher
                )

                processed_count = 0
                async for result in results_generator:
                    processed_count += 1
                    self.logger.info(
                        f"arun_many result #{processed_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                    )

                    if hasattr(result, "success") and result.success:
                        try:
                            sanitized_result = self._sanitize_crawl_result(result)
                            successful_results.append(sanitized_result)

                            if progress_callback:
                                progress_callback(
                                    len(successful_results),
                                    max_pages,
                                    f"Crawled {result.url}",
                                )

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to process result for {result.url}: {e}"
                            )

                    if len(successful_results) >= max_pages:
                        self.logger.info(f"Reached max_pages limit ({max_pages})")
                        break

                self.logger.info(
                    f"arun_many completed: {processed_count} processed, {len(successful_results)} successful"
                )

            except Exception as e:
                self.logger.error(f"arun_many approach failed: {e}", exc_info=True)
                # Fallback to single URL if arun_many fails
                if urls_to_crawl:
                    self.logger.info("Falling back to single URL crawl")
                    single_result = await browser.arun(
                        url=urls_to_crawl[0], config=batch_config
                    )
                    if hasattr(single_result, "success") and single_result.success:
                        sanitized_result = self._sanitize_crawl_result(single_result)
                        successful_results.append(sanitized_result)

        return successful_results
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Propagate errors for arun_many path too; update execute callers

Same issue as above: errors are collected but never returned to execute().

-    ) -> list[Any]:
+    ) -> tuple[list[Any], list[str]]:
@@
-        successful_results = []
+        successful_results: list[Any] = []
+        errors: list[str] = []
@@
-                            self.logger.warning(
-                                f"Failed to process result for {result.url}: {e}"
-                            )
+                            self.logger.warning(
+                                "Failed to process result for %s: %s", getattr(result, "url", "unknown"), e
+                            )
+                            errors.append(str(e))
@@
-                self.logger.error(f"arun_many approach failed: {e}", exc_info=True)
+                self.logger.error("arun_many approach failed: %s", e, exc_info=True)
                 # Fallback to single URL if arun_many fails
                 if urls_to_crawl:
                     self.logger.info("Falling back to single URL crawl")
                     single_result = await browser.arun(
                         url=urls_to_crawl[0], config=batch_config
                     )
                     if hasattr(single_result, "success") and single_result.success:
                         sanitized_result = self._sanitize_crawl_result(single_result)
                         successful_results.append(sanitized_result)
+                    else:
+                        errors.append(f"Failed to crawl {getattr(single_result, 'url', urls_to_crawl[0])}")
@@
-        return successful_results
+        return successful_results, errors

🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
fix: suppress BeautifulSoup 'text' argument deprecation warnings
0272afc
coderabbitai[bot]
coderabbitai bot reviewed Aug 21, 2025
coderabbitai bot left a comment

Actionable comments posted: 3

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (3)

♻️ Duplicate comments (3)

📜 Review details


🔇 Additional comments (3)

















pyproject.toml
    "types-psutil>=7.0.0.20250801",
    "torch>=2.8.0",
@coderabbitai coderabbitai bot Aug 21, 2025

💡 Verification agent
🧩 Analysis chain

Make torch an optional dependency group

The torch import in crawler_mcp/core/rag.py is gated by settings.reranker_enabled—but since reranker_enabled defaults to True (see crawler_mcp/config.py:81), every installation pulls in torch (and sentence-transformers) by default. To avoid forcing heavy, platform-specific installs for users who don’t need local embeddings at runtime, move these packages into an optional “ml” extras group.

Locations requiring attention:

    crawler_mcp/config.py:81 — reranker_enabled: bool = Field(default=True, ...)
    crawler_mcp/core/rag.py:224–225 — import torch and from sentence_transformers import CrossEncoder under the if settings.reranker_enabled guard

Suggested changes:

 pyproject.toml
 [project]
-dependencies = [
-    "torch>=2.8.0",
-    …
-]
+dependencies = [
+    # other mandatory deps, but no torch or sentence-transformers
+    …
+]

+[project.optional-dependencies]
+ml = [
+    "torch>=2.8.0",
+    "transformers>=4.55.0",
+    "sentence-transformers>=5.1.0",
+]

Please also update documentation to instruct users to install pip install crawler-mcp[ml] if they need reranking with local models.
📝 Committable suggestion


@jmagar
pyproject.toml
Comment on lines +112 to +129
strict_optional = true
show_error_codes = true
disallow_any_generics = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true

# Allow protocols to be mixed with runtime implementations
allow_subclassing_any = true
allow_untyped_calls = false

# Disable specific checks for dynamic imports
disable_error_code = ["no-any-return"]

# Handle missing imports gracefully for optional dependencies
ignore_missing_imports = false
follow_imports = "silent"

@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Resolve mypy config contradiction: you enable warn_return_any but disable no-any-return

warn_return_any is rendered toothless if you disable the “no-any-return” error. Either keep the warning or drop the disable. Recommend enforcing the check.

Apply this diff to re-enable the check and remove the stale comment:

-# Disable specific checks for dynamic imports
-disable_error_code = ["no-any-return"]

🤖 Prompt for AI Agents

@jmagar
pyproject.toml
Comment on lines +140 to +147
minversion = "7.4"
addopts = [
    "--strict-config",
    "--strict-markers",
    "--disable-warnings",
    "--tb=short",
    "-ra",
]
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Align pytest minversion with your dev tooling and drop redundant warning suppression

    dev dependency group pins pytest>=8.4.1; set minversion accordingly.
    --disable-warnings conflicts with filterwarnings=error/ignore rules and obscures useful info. Remove it.

Apply this diff:

-minversion = "7.4"
+minversion = "8.4"
@@
 addopts = [
     "--strict-config",
     "--strict-markers",
-    "--disable-warnings",
     "--tb=short",
     "-ra",
 ]

🤖 Prompt for AI Agents

@jmagar
Merge info
All checks have passed

2 successful checks
successful checks

CodeRabbit
CodeRabbit — Review completed
GitGuardian Security Checks
GitGuardian Security ChecksSuccessful in 2s — No secrets detected ✅

No conflicts with base branch

Merging can be performed automatically.
You can also merge this with the command line.
@jmagar
Add a comment
Comment

Add your comment here...
Remember, contributions to this repository should follow our GitHub Community Guidelines.
ProTip! Add comments to specific lines under Files changed.
Reviewers

@coderabbitai
coderabbitai[bot]

Copilot code review
Copilot

Still in progress?
Assignees
No one—
Labels
None yet
Projects
None yet
Milestone
No milestone
Development

Successfully merging this pull request may close these issues.

None yet

You’re receiving notifications because you authored the thread.
1 participant
@jmagar
Footer
© 2025 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Docs
    Contact

coderabbitai bot reviewed View reviewed changes coderabbitai bot left a comment Actionable comments posted: 3 Caution Some comments are outside the diff and can’t be posted inline due to platform limitations. ⚠️ Outside diff range comments (3) ♻️ Duplicate comments (3) 📜 Review details 🔇 Additional comments (3) pyproject.toml "types-psutil>=7.0.0.20250801", "torch>=2.8.0", coderabbitai bot 💡 Verification agent 🧩 Analysis chain Make torch an optional dependency group The torch import in crawler_mcp/core/rag.py is gated by settings.reranker_enabled—but since reranker_enabled defaults to True (see crawler_mcp/config.py:81), every installation pulls in torch (and sentence-transformers) by default. To avoid forcing heavy, platform-specific installs for users who don’t need local embeddings at runtime, move these packages into an optional “ml” extras group. Locations requiring attention: crawler_mcp/config.py:81 — reranker_enabled: bool = Field(default=True, ...) crawler_mcp/core/rag.py:224–225 — import torch and from sentence_transformers import CrossEncoder under the if settings.reranker_enabled guard Suggested changes: pyproject.toml [project] -dependencies = [ - "torch>=2.8.0", - … -] +dependencies = [ + # other mandatory deps, but no torch or sentence-transformers + … +] +[project.optional-dependencies] +ml = [ + "torch>=2.8.0", + "transformers>=4.55.0", + "sentence-transformers>=5.1.0", +] Please also update documentation to instruct users to install pip install crawler-mcp[ml] if they need reranking with local models. 📝 Committable suggestion Reply... Resolve conversation pyproject.toml Comment on lines +112 to +129 strict_optional = true show_error_codes = true disallow_any_generics = true disallow_incomplete_defs = true check_untyped_defs = true disallow_untyped_decorators = true # Allow protocols to be mixed with runtime implementations allow_subclassing_any = true allow_untyped_calls = false # Disable specific checks for dynamic imports disable_error_code = ["no-any-return"] # Handle missing imports gracefully for optional dependencies ignore_missing_imports = false follow_imports = "silent" coderabbitai bot 🛠️ Refactor suggestion Resolve mypy config contradiction: you enable warn_return_any but disable no-any-return warn_return_any is rendered toothless if you disable the “no-any-return” error. Either keep the warning or drop the disable. Recommend enforcing the check. Apply this diff to re-enable the check and remove the stale comment: -# Disable specific checks for dynamic imports -disable_error_code = ["no-any-return"] 📝 Committable suggestion 🤖 Prompt for AI Agents Reply... Resolve conversation pyproject.toml Comment on lines +140 to +147 minversion = "7.4" addopts = [ "--strict-config", "--strict-markers", "--disable-warnings", "--tb=short", "-ra", ] coderabbitai bot 🛠️ Refactor suggestion Align pytest minversion with your dev tooling and drop redundant warning suppression dev dependency group pins pytest>=8.4.1; set minversion accordingly. --disable-warnings conflicts with filterwarnings=error/ignore rules and obscures useful info. Remove it. Apply this diff: -minversion = "7.4" +minversion = "8.4" @@ addopts = [ "--strict-config", "--strict-markers", - "--disable-warnings", "--tb=short", "-ra", ] 📝 Committable suggestion 🤖 Prompt for AI Agents Reply... Resolve conversation

Skip to content
Navigation Menu
jmagar
crawler-mcp

Code
Issues
Pull requests 1
Actions
Projects
Wiki
Security
Insights

    Settings

Feat/new development #3
Open
jmagar wants to merge 5 commits into main from feat/new-development
Open
Feat/new development
#3
jmagar wants to merge 5 commits into main from feat/new-development
+9,657 −2,262
Conversation 147
Commits 5
Checks 1
Files changed 46
Conversation
jmagar
Owner
jmagar commented Aug 20, 2025 •
Summary by CodeRabbit

    New Features
        Safer scraping with improved content extraction/sanitization, optional extraction strategies, concurrency/performance knobs, and a high-performance browser pool.
        RAG indexing upgraded to subchunk-level processing with per-subchunk deduplication, parallel embedding pipeline, richer metadata, and orphan cleanup.
        Typed integration layer for crawl components and embedding throughput metrics.

    Bug Fixes
        More robust error handling and corrected crawl success-rate; suppresses noisy deprecation warnings.

    Documentation
        Added middleware and testing guides plus tests README.

    Tests
        Large new test suites added; some obsolete tests removed.

    Chores
        .gitignore updated, dev script removed, docker and dependency/tooling updates.

jmagar and others added 2 commits August 16, 2025 19:30
@jmagar
@claude
fix: resolve crawl4ai integer hash issues and improve type safety
97eee2d
@jmagar
@claude
fix: implement proper text chunking to resolve embedding timeouts
c57d379
@Copilot Copilot AI review requested due to automatic review settings August 20, 2025 15:37
@coderabbitai coderabbitai
coderabbitai bot commented Aug 20, 2025 •

Note
Other AI code review bot(s) detected

CodeRabbit has detected other AI code review bot(s) in this pull request and will avoid duplicating their findings in the review comments. This may lead to a less comprehensive review.
Walkthrough

Adds a crawl4ai typing facade, hardens markdown extraction and sanitization, migrates RAG deduplication to subchunk granularity with richer metadata and orphan cleanup, makes extraction_strategy optional, introduces high-performance browser/file pools, tightens typing and test infra, adjusts dev/docker configs, and adds many docs and tests.
Changes
Cohort / File(s) 	Summary of changes
Type facade for crawl4ai
crawler_mcp/types/crawl4ai_types.py 	New protocol-based typed contract layer and runtime-safe aliases for crawl4ai components (results, markdown, filters, extraction strategies, run configs, enums); guarded runtime imports and exported implementation aliases.
Crawling content handling & APIs
crawler_mcp/crawlers/web.py, crawler_mcp/core/orchestrator.py, crawler_mcp/crawlers/base.py, crawler_mcp/models/crawl.py, crawler_mcp/tools/crawling.py 	Replace markdown/filter imports with typed impls; add robust sanitization and _safe_get_markdown, best_content selection and length/object checks, set PageContent.markdown to string, compute metadata.word_count from sanitized content; change extraction_strategy to `str
RAG processing, deduplication & cache
crawler_mcp/core/rag.py 	Move deduplication to per-subchunk granularity with deterministic subchunk IDs and content_hashes, legacy-upgrade logic, orphaned-chunk deletion, richer chunk metadata (sub_chunk_index, start_pos, end_pos), improved error handling, QueryCache typed to RagResult, and new parallel _process_embeddings_pipeline; return shape includes chunks_stored.
Vector search params
crawler_mcp/core/vectors.py 	Accept typed SearchParams (hnsw_ef) instead of raw dict for dynamic EF configuration in search_similar.
Embedding throughput logging
crawler_mcp/core/embeddings.py 	Add batch timing instrumentation and embeddings-per-second throughput logging for true-batch and batched generators.
Browser & file pools (high-performance)
crawler_mcp/core/browser_pool.py, crawler_mcp/crawlers/directory.py 	New HighPerformanceBrowserPool for pooled AsyncWebCrawler instances; DirectoryCrawlStrategy switched to thread-pooled high-concurrency file processing (sync per-file workers via ThreadPoolExecutor) with post-processing memory checks.
Middleware typing
crawler_mcp/middleware/progress.py 	Refine cleanup task typing to `asyncio.Task[None]
Server warning suppression
crawler_mcp/server.py 	Add warnings.filterwarnings to ignore BeautifulSoup 'text' argument DeprecationWarning patterns.
Dev environment / compose
dev.sh (deleted), docker-compose.yml 	Remove dev.sh; change qdrant host ports to 7000:6333, 7001:6334; adjust text-embeddings-inference startup args and environment (batch sizes, workers, PYTORCH_CUDA_ALLOC_CONF).
Project config & gitignore
pyproject.toml, .gitignore 	Add coverage dependency, stricter mypy and pytest configs (with overrides for crawl4ai); update test tooling groups; add .serena/ ignore rule.
Configuration additions
crawler_mcp/config.py, .env.example 	Add many new CrawlerrSettings fields (qdrant pool/batch/prefetch, embedding_workers, file/thread/pool/concurrency/gpu/cache knobs, arun_many toggle, lxml strategy toggle); add EMBEDDING_WORKERS to .env.example.
Docs
docs/middleware.md, docs/testing.md 	Add middleware guide (hooks, lifecycle, examples) and testing guide (in-memory and deployed testing patterns).
Tests — additions & fixtures
tests/* (many new files: conftest.py, test_*.py, README.md, __init__.py) 	Add extensive unit/integration tests, fixtures, in-memory FastMCP test server/client, and test guidance covering config, crawlers, core services (embeddings/vectors/rag), RAG tools, server, middleware, models, sources, and end-to-end flows.
Tests — removals
tests/test_backwards_compatibility.py, tests/test_deduplication.py, tests/test_deduplication_performance.py, tests/test_migration.py 	Remove several prior deduplication/backwards-compatibility/migration/performance test modules.
Sequence Diagram(s)
Estimated code review effort

🎯 5 (Critical) | ⏱️ ~120 minutes
Possibly related PRs

feat: implement fit markdown extraction for clean content #2 — Overlaps crawling/markdown extraction and RAG deduplication changes (best_content/pruning/content-filter imports and usage).

    🚀 Qwen3 Reranker Integration & Token-Aware Chunking Optimization #1 — Related to RAG/chunking, reranking/tokenization, and deduplication helpers; touches similar regions in RagService.

Poem

    A rabbit nibbled through the docs and code,
    Tuned chunk IDs down a tidy road.
    It cleaned the markdown, split each chunk,
    Pooled fast browsers and threaded each trunk.
    Ports rebound and tests come spring—hop, deploy, behold! 🐇✨

Tip
🔌 Remote MCP (Model Context Protocol) integration is now available!

✨ Finishing Touches

Thanks for using CodeRabbit! It's free for OSS, and your support helps us grow. If you like it, consider giving us a shout-out.
❤️ Share

🪧 Tips

Copilot
Copilot AI reviewed Aug 20, 2025
Copilot AI left a comment
Pull Request Overview

This PR introduces comprehensive test coverage for the crawler MCP server to achieve high code coverage and ensure robust functionality. The tests follow FastMCP best practices with in-memory testing, direct service integration, and behavior-focused testing patterns.

Key changes:

    Adds comprehensive test suites for all major components (RAG, vector services, crawling, sources)
    Implements edge case testing with import error handling and failover mechanisms
    Creates integration tests for complete end-to-end workflows

Reviewed Changes

Copilot reviewed 39 out of 41 changed files in this pull request and generated 4 comments.
Show a summary per file













Comments suppressed due to low confidence (1)

Tip: Customize your code reviews with copilot-instructions.md. Create the file or learn how to get started.
tests/test_vectors_comprehensive.py
                        source_url="https://example.com",
                        source_title="Test Page",
                        content="Test content",
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
Copilot AI
Aug 20, 2025

Consider using a constant for the embedding dimension (384) instead of magic numbers. This makes the test more maintainable and clearer about the expected vector size.
Suggested change
                        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
                        embedding=[0.1, 0.2, 0.3] * (EMBEDDING_DIM // 3),  # 384 dimensions

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_rag_edge_cases.py
                raise ImportError("tiktoken not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
Copilot AI
Aug 20, 2025

The import mocking pattern is repeated multiple times. Consider extracting this into a reusable test fixture or helper function to reduce code duplication.
Suggested change
        with patch("builtins.__import__", side_effect=mock_import):
        # Patch import to raise ImportError for tiktoken
        with import_blocker("tiktoken"):

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_rag_comprehensive.py
        assert result == mock_result

        # Wait for expiration and test miss
        time.sleep(1)
Copilot AI
Aug 20, 2025

Using time.sleep() in async tests can make tests slower and less reliable. Consider using asyncio.sleep() or mocking time-dependent operations instead.
Suggested change
        time.sleep(1)
        await asyncio.sleep(1)

Copilot uses AI. Check for mistakes.
@jmagar
tests/test_integration_flow.py
        # Step 2: Verify content was indexed by checking stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data

Copilot AI
Aug 20, 2025

The hard-coded sleep delays in integration tests make them slower and potentially flaky. Consider implementing proper wait conditions or using event-driven synchronization instead.
Suggested change
        # Wait for indexing to complete by polling stats
        async def wait_for_indexing(client, min_sources=1, min_chunks=1, timeout=15, poll_interval=0.5):
            start = asyncio.get_event_loop().time()
            while True:
                stats_result = await client.call_tool("get_rag_stats", {})
                stats = stats_result.data
                if stats["total_sources"] >= min_sources and stats["total_chunks"] >= min_chunks:
                    return stats
                if asyncio.get_event_loop().time() - start > timeout:
                    raise TimeoutError("Indexing did not complete in time")
                await asyncio.sleep(poll_interval)
        stats = await wait_for_indexing(mcp_client, min_sources=1, min_chunks=1)

Copilot uses AI. Check for mistakes.
@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Actionable comments posted: 101

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (31)











.gitignore
@@ -142,3 +142,4 @@ ipython_config.py
*.key
*.crt
*.p12
.serena/
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Add context and mirror ignore where needed (.dockerignore, packaging).

The new rule looks fine. Two quick checks:

    If you intend to ignore only the repo-root directory, prefer "/.serena/". If any nested ".serena/" folders should be ignored anywhere, current form is correct.
    If ".serena/" may contain large artifacts or secrets, mirror this in .dockerignore (to keep it out of Docker build context) and in packaging config (e.g., MANIFEST.in) if you publish wheels/sdists.

Optional tweak if root-only is desired:

-.serena/
+/.serena/

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +786 to +795
                # Split page content into optimal chunks for Qwen3-Embedding-0.6B
                text_chunks = self._chunk_text_character_based(
                    page.content,
                    metadata={
                        "source_url": page.url,
                        "source_title": page.title,
                        "page_index": i,
                        **page.metadata.get("chunk_metadata", {}),
                    },
                )
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Prefer token-based chunking when tokenizer is available

The pipeline unconditionally uses character-based chunking even when a tokenizer is initialized. This loses boundary quality and can increase fragmentation vs. token-aware chunking.

Switch to token-based when self.tokenizer is set, with character-based as fallback.

Apply:

-                # Split page content into optimal chunks for Qwen3-Embedding-0.6B
-                text_chunks = self._chunk_text_character_based(
+                # Split page content into optimal chunks
+                # Prefer token-based when tokenizer is available; fallback to character-based.
+                chunk_fn = (
+                    self._chunk_text_token_based if self.tokenizer else self._chunk_text_character_based
+                )
+                text_chunks = chunk_fn(
                     page.content,
                     metadata={
                         "source_url": page.url,
                         "source_title": page.title,
                         "page_index": i,
                         **page.metadata.get("chunk_metadata", {}),
                     },
                 )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +895 to +901
                        metadata={
                            **page.metadata,
                            "sub_chunk_index": sub_chunk_idx,
                            "page_index": i,
                            "start_pos": chunk_data["start_pos"],
                            "end_pos": chunk_data["end_pos"],
                        },
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Preserve user-provided per-chunk metadata

You accept page.metadata.get("chunk_metadata", {}) during chunking but don’t include it in the stored chunk metadata. This can drop important, caller-supplied annotations at write time.

Fold the original chunk_metadata back into the payload.

Apply:

                         metadata={
                             **page.metadata,
+                            "chunk_metadata": page.metadata.get("chunk_metadata", {}),
                             "sub_chunk_index": sub_chunk_idx,
                             "page_index": i,
                             "start_pos": chunk_data["start_pos"],
                             "end_pos": chunk_data["end_pos"],
                         },

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/vectors.py
Comment on lines +349 to 352
                search_params=SearchParams(
                    hnsw_ef=ef_value
                ),  # Dynamic ef for optimal speed/accuracy
            )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Per-query hnsw_ef is correct; consider allowing caller override.

Current heuristic ef = min(256, max(64, limit*4)) is sensible. To improve control, allow an optional ef_override param (or part of a typed options object) to override the heuristic when needed.

Example signature tweak:

-    async def search_similar(
+    async def search_similar(
         self,
         query_vector: list[float],
         limit: int = 10,
         score_threshold: float = 0.0,
         source_filter: list[str] | None = None,
         date_range: tuple[datetime, datetime] | None = None,
+        ef_override: int | None = None,
     ) -> list[SearchMatch]:
@@
-            ef_value = min(256, max(64, limit * 4))  # 4x limit, capped at 256
+            ef_value = ef_override if ef_override is not None else min(256, max(64, limit * 4))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/server.py
Comment on lines +19 to +23
warnings.filterwarnings(
    "ignore",
    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
    category=DeprecationWarning,
)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Scope the deprecation warning filter to BeautifulSoup module

The message matches bs4’s deprecation on text=. Constrain by module to avoid accidentally suppressing unrelated libraries’ warnings.

Apply:

 warnings.filterwarnings(
     "ignore",
-    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
-    category=DeprecationWarning,
+    message="The 'text' argument to find\\(\\)-type methods is deprecated.*",
+    category=DeprecationWarning,
+    module="bs4\\..*",
 )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/tools/crawling.py
Comment on lines +242 to 243
        extraction_strategy: str | None = None,
        wait_for: str | None = None,
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Refactor Required: Honor extraction_strategy and wait_for in the crawler implementation

While both the FastMCP tool (scrape) and CrawlerService.scrape_single_page signatures correctly accept extraction_strategy: str | None = None and wait_for: str | None = None, the current implementation never actually uses those values. To complete this API improvement, you should:

• In crawler_mcp/core/orchestrator.py (method scrape_single_page, ~lines 212–260):
– Before calling browser.arun(…), inject logic to
• pass wait_for as a selector or JS condition (e.g. via await browser.wait_for(wait_for) or a wait_for argument to arun).
• apply extraction_strategy (e.g. map it to an extraction_method parameter on arun, or merge into custom_config).

• In crawler_mcp/tools/crawling.py (the FastMCP tool scrape, ~lines 242–260):
– Ensure the docstring reflects the behavior when extraction_strategy=None or wait_for=None.
– Confirm that what you pass into crawler_service.scrape_single_page(...) is actually consumed downstream.

Until these parameters are surfaced to the underlying AsyncWebCrawler, the default strategy will always apply and any supplied string will be ignored. Please implement the missing pass-through and handling logic so that None truly invokes the default and non-None values are respected.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/types/crawl4ai_types.py
Comment on lines +216 to +231
# Updated CrawlerRunConfig with better typing
class CrawlerRunConfigAdvanced(Protocol):
    """Protocol for crawl4ai CrawlerRunConfig."""

    def __init__(
        self,
        deep_crawl_strategy: DeepCrawlStrategy | None = None,
        stream: bool = False,
        cache_mode: CacheMode | str | None = None,
        page_timeout: float | None = None,
        markdown_generator: MarkdownGenerationStrategy | None = None,
        content_filter: ContentFilterStrategy | None = None,
        extraction_strategy: ExtractionStrategy | None = None,
        **kwargs: Any,
    ) -> None: ...

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

You introduced CrawlerRunConfigAdvanced but didn’t export it

If downstream modules import from this types layer, they won’t see CrawlerRunConfigAdvanced via __all__. Export it for consistency with other protocol exports.

Apply:

 __all__ = [
@@
-    "CrawlerRunConfig",
+    "CrawlerRunConfig",
+    "CrawlerRunConfigAdvanced",

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/types/crawl4ai_types.py
Comment on lines +236 to +259
# For actual runtime usage, we just cast the imports to our protocols
# This maintains type safety while allowing runtime flexibility
try:
    from crawl4ai.content_filter_strategy import (
        PruningContentFilter as _PruningContentFilter,  # type: ignore
    )
    from crawl4ai.markdown_generation_strategy import (
        DefaultMarkdownGenerator as _DefaultMarkdownGenerator,  # type: ignore
    )
    from crawl4ai.models import (
        MarkdownGenerationResult as _MarkdownGenerationResult,  # type: ignore
    )

    # Type-safe aliases that satisfy our protocols
    DefaultMarkdownGeneratorImpl = _DefaultMarkdownGenerator
    PruningContentFilterImpl = _PruningContentFilter
    MarkdownGenerationResultImpl = _MarkdownGenerationResult

except ImportError:
    # Fallback for when crawl4ai is not available
    DefaultMarkdownGeneratorImpl = None  # type: ignore
    PruningContentFilterImpl = None  # type: ignore
    MarkdownGenerationResultImpl = None  # type: ignore

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Handle missing crawl4ai implementations at runtime

The aliases DefaultMarkdownGeneratorImpl, PruningContentFilterImpl, and MarkdownGenerationResultImpl may be set to None when the crawl4ai package isn’t installed. Instantiating them unguarded will raise
TypeError: 'NoneType' object is not callable.
Please add explicit checks or fallbacks at each call site.

• In crawler_mcp/crawlers/web.py:688

markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

– guard before calling, e.g.:

if DefaultMarkdownGeneratorImpl is None:
    raise ImportError("crawl4ai package is required for markdown generation")
markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

• Any other direct uses (e.g. orchestrator imports) should similarly verify the alias isn’t None before instantiation or method calls.

For stricter safety, consider exporting stub classes in crawl4ai_types.py that raise a clear ImportError on construction instead of defaulting to None.
🤖 Prompt for AI Agents

@jmagar
docker-compose.yml
Comment on lines +6 to +7
      - "7000:6333"
      - "7001:6334"  # gRPC port
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Mandatory: Parameterize and bind Qdrant host ports consistently across the codebase

Changing the host ports from 6333/6334 to 7000/7001 without updating every reference will break existing clients, tests, CI workflows, and documentation. To contain exposure, bind only to the loopback interface and make the host ports configurable via environment variables.

Please update the following locations:

    docker-compose.yml (root)
        Lines 6–7: replace hard-coded mappings
        Lines 12–13: keep container ports at 6333/6334 but expose via ${QDRANT_HTTP_PORT_HOST}/${QDRANT_GRPC_PORT_HOST}
    .env (root)
        Update QDRANT_URL default from port 6333 to use ${QDRANT_HTTP_PORT_HOST:-7000}
    Configuration model (e.g., src/config.py)
        Field alias QDRANT_URL default should reflect the new host port
    Tests (e.g., tests/test_config.py, around the assertion on line 34)
        Change assert config.qdrant_url == "http://localhost:6333" to use the parameterized port
    CI/workflow scripts (e.g., .github/workflows/*.yml, around line 160)
        Update health-check URLs in curl -f http://localhost:6333/health to use ${QDRANT_HTTP_PORT_HOST}
    README.md and any documentation snippets
        Replace examples pointing at port 6333/6334 with the new defaults or environment-driven values
    Additional Docker‐Compose profiles (lines 235, 311 in the same file)
        Align any other Qdrant service sections to use the same env-driven port mappings and 127.0.0.1 binding

Suggested refactor diff in docker-compose.yml:

-      - "7000:6333"
-      - "7001:6334"  # gRPC port
+      - "127.0.0.1:${QDRANT_HTTP_PORT_HOST:-7000}:6333"
+      - "127.0.0.1:${QDRANT_GRPC_PORT_HOST:-7001}:6334"  # gRPC port

And in .env:

- QDRANT_URL=http://localhost:6333
+ QDRANT_URL=http://localhost:${QDRANT_HTTP_PORT_HOST:-7000}

🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +1 to +16
# MCP Middleware

> Add cross-cutting functionality to your MCP server with middleware that can inspect, modify, and respond to all MCP requests and responses.
export const VersionBadge = ({version}) => {
  return <code className="version-badge-container">
            <p className="version-badge">
                <span className="version-badge-label">New in version:</span>
                <code className="version-badge-version">{version}</code>
            </p>
        </code>;
};

<VersionBadge version="2.9.0" />

MCP middleware is a powerful concept that allows you to add cross-cutting functionality to your FastMCP server. Unlike traditional web middleware, MCP middleware is designed specifically for the Model Context Protocol, providing hooks for different types of MCP operations like tool calls, resource reads, and prompt requests.
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

MDX components in .md: convert to .mdx or use admonitions to avoid lint/build issues

This file uses React/MDX components (VersionBadge, Tip, Warning) and inline HTML in a .md file. This will trigger MD033 and may fail non-MDX doc builds.

Options:

    Rename to docs/middleware.mdx and ensure the docs pipeline supports MDX.
    Or replace components with admonitions:

-<Tip>
-  ...
-</Tip>
+:::tip
+...
+:::

-<Warning>
-  ...
-</Warning>
+:::warning
+...
+:::

As a quick stopgap, add a markdownlint suppression at the top:

+# markdownlint-disable MD033

🧰 Tools
















🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +220 to +234
```python
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class AuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name

        # Deny access to restricted tools
        if tool_name.lower() in ["delete", "admin_config"]:
            raise ToolError("Access denied: tool requires admin privileges")

        # Allow other tools to proceed
        return await call_next(context)
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Unify error type usage: prefer ToolError for tool denial

Earlier sections correctly recommend ToolError for access denial. This section introduces McpError/ErrorData. For consistency and to avoid leaking protocol-layer types, prefer ToolError here as well.

-from mcp import McpError
-from mcp.types import ErrorData
+from fastmcp.exceptions import ToolError
@@
-        if len(self.client_requests[client_id]) >= self.requests_per_minute:
-            raise McpError(ErrorData(code=-32000, message="Rate limit exceeded"))
+        if len(self.client_requests[client_id]) >= self.requests_per_minute:
+            raise ToolError("Rate limit exceeded")

If you intend to show both patterns, clarify when protocol errors are appropriate.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
docs/middleware.md
Comment on lines +315 to +323
<VersionBadge version="2.11.0" />

In addition to modifying the request and response, you can also store state data that your tools can (optionally) access later. To do so, use the FastMCP Context to either `set_state` or `get_state` as appropriate. For more information, see the [Context State Management](/servers/context#state-management) docs.

## Creating Middleware

FastMCP middleware is implemented by subclassing the `Middleware` base class and overriding the hooks you need. You only need to implement the hooks that are relevant to your use case.

```python
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

VersionBadge render note

If you keep VersionBadge, document that it is a site component and include its implementation in the MDX runtime, or replace with plain text to avoid broken renders.

Replace with inline text:

-<VersionBadge version="2.11.0" />
+New in version 2.11.0

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
docs/testing.md
Comment on lines +13 to +40
```python
from fastmcp import FastMCP, Client

# Create your server
server = FastMCP("WeatherServer")

@server.tool
def get_temperature(city: str) -> dict:
    """Get current temperature for a city"""
    temps = {"NYC": 72, "LA": 85, "Chicago": 68}
    return {"city": city, "temp": temps.get(city, 70)}

@server.resource("weather://forecast")
def get_forecast() -> dict:
    """Get 5-day forecast"""
    return {"days": 5, "conditions": "sunny"}

async def test_weather_operations():
    # Pass server directly - no deployment needed
    async with Client(server) as client:
        # Test tool execution
        result = await client.call_tool("get_temperature", {"city": "NYC"})
        assert result.data == {"city": "NYC", "temp": 72}

        # Test resource retrieval
        forecast = await client.read_resource("weather://forecast")
        assert forecast.contents[0].data == {"days": 5, "conditions": "sunny"}
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Example correctness: Client(server) pattern is good; add timeout to avoid hanging in examples

The in-memory example is solid and aligns with our guideline to pass the server instance directly. To keep samples robust for readers who will copy/paste, consider wrapping networked awaits with a timeout (esp. in the HTTP section) to avoid hanging tests.

Apply this minimal change to the HTTP example:

 from fastmcp import Client
+import asyncio

 async def test_deployed_server():
     # Connect to a running server
-    async with Client("http://localhost:8000/mcp/") as client:
-        await client.ping()
+    async with Client("http://localhost:8000/mcp/") as client:
+        await asyncio.wait_for(client.ping(), timeout=5)

🤖 Prompt for AI Agents

@jmagar
docs/testing.md
Comment on lines +112 to +146
```python
from fastmcp import Client

async def test_deployed_server():
    # Connect to a running server
    async with Client("http://localhost:8000/mcp/") as client:
        await client.ping()

        # Test with real network transport
        tools = await client.list_tools()
        assert len(tools) > 0

        result = await client.call_tool("greet", {"name": "World"})
        assert "Hello" in result.data
```

### Testing Authentication

The FastMCP Client handles authentication transparently, making it easy to test secured servers:

```python
async def test_authenticated_server():
    # Bearer token authentication
    async with Client(
        "https://api.example.com/mcp",
        headers={"Authorization": "Bearer test-token"}
    ) as client:
        await client.ping()
        tools = await client.list_tools()

    # OAuth flow (opens browser for authorization)
    async with Client("https://api.example.com/mcp", auth="oauth") as client:
        result = await client.call_tool("protected_tool", {})
        assert result.data is not None
```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Prefer pytest markers for deployed-server tests in docs

These examples hit a running server. Suggest showing pytest markers so readers can categorize/skip them in CI.

Add a marker in the snippet header:

-from fastmcp import Client
+import pytest
+from fastmcp import Client
@@
-async def test_deployed_server():
+@pytest.mark.integration
+@pytest.mark.requires_services
+async def test_deployed_server():

🧰 Tools



🤖 Prompt for AI Agents

@jmagar
pyproject.toml
Comment on lines +50 to 52
    "torch>=2.8.0",
    "coverage>=7.10.3",
]
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Move coverage to test dependencies, not runtime

coverage should not be a runtime dependency of the library/server.

-    "torch>=2.8.0",
-    "coverage>=7.10.3",
+    "torch>=2.8.0",

Then add it to the test group below (see next comment).
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +24 to +36
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

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Add a short note about uv (or alternative) in Prerequisites to avoid confusion.

All commands use “uv run …”. Clarify how to install/use uv or how to substitute with Python/pip so new contributors aren’t blocked.

Proposed addition:

 ## Running Tests

 ### Prerequisites

+0. **Tooling**
+   - Ensure Python 3.11+ is available.
+   - Ensure `uv` is installed (or replace `uv run pytest` with `python -m pytest` in the commands below).
+
 1. **Start Required Services**:

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +26 to +35
1. **Start Required Services**:
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services
   ```

2. **Environment Variables**:
   Copy `tests/.env.test` to `.env` and adjust as needed:
   ```bash
   cp tests/.env.test .env
   ```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix markdownlint MD031: add blank lines around fenced code blocks in Prerequisites.

Current fences begin immediately after list items, triggering MD031. Insert blank lines before the opening and after the closing fences for both steps.

Apply this diff:

 1. **Start Required Services**:
+
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services

    Environment Variables:

    Copy tests/.env.test to .env and adjust as needed:

    Copy tests/.env.test to .env and adjust as needed:

    cp tests/.env.test .env


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~26-~26: Use correct spacing
Context: ...uisites  1. **Start Required Services**:    ```bash    docker-compose up -d  # Start Qdrant and TEI services    ```  2. **Environment Variables**:    Copy `tests/...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~31-~31: There might be a mistake here.
Context: ...es    ```  2. **Environment Variables**:    Copy `tests/.env.test` to `.env` and adj...

(QB_NEW_EN)

---

[grammar] ~32-~32: Use correct spacing
Context: ...nv.test` to `.env` and adjust as needed:    ```bash    cp tests/.env.test .env    ```  ### Test Commands  ```bash # Run all tests u...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

27-27: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

---

33-33: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 26 to 35, the fenced code blocks for the "Start
Required Services" and "Environment Variables" list items lack blank lines
before the opening bash and after the closing , causing markdownlint
MD031; fix by inserting a blank line immediately before each opening fenced code
block and a blank line immediately after each closing fence so each fenced block
is separated from the list item text.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +26 to +29
1. **Start Required Services**:
   ```bash
   docker-compose up -d  # Start Qdrant and TEI services
   ```
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Prefer “docker compose” over “docker-compose” in docs.

Modern Docker recommends the space-separated subcommand. Keep if your team standardizes on the hyphenated plugin, but consider updating for consistency with current docs.
🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +37 to +61
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

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Register custom pytest markers to avoid PytestUnknownMarkWarning.

You use custom markers (unit, integration, slow, requires_services). Add a pytest.ini snippet to document/avoid warnings.

Proposed snippet (documentation-only):

# pytest.ini (at repo root)
[pytest]
markers =
  unit: fast tests without external services
  integration: tests using real services
  slow: long-running tests
  requires_services: requires Qdrant and TEI

🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +39 to +60
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
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Minor: document combined marker usage and coverage targets.

Add examples that reflect typical usage patterns:

    Run integration tests that also require services: uv run pytest -m "integration and requires_services".
    Produce HTML/XML coverage for CI: uv run pytest --cov=crawler_mcp --cov-report=term-missing --cov-report=xml --cov-report=html.

🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +71 to +88
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

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Async test requires an async test runner; mark with anyio/asyncio or make the example sync.

As written, async def tests will require pytest-anyio (preferred with FastMCP) or pytest-asyncio. Add a marker to the example to prevent copy-paste failures.

Apply this diff:

 @pytest.mark.integration
 @pytest.mark.requires_services
+@pytest.mark.anyio
 async def test_example(mcp_client: Client):

Optionally document in README: “Install pytest-anyio and enable it via @pytest.mark.anyio or configure asyncio_mode in pytest.ini.”
📝 Committable suggestion


🧰 Tools


🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +100 to +106
### Service Management

- Tests use a separate Qdrant collection (`test_crawler_mcp`)
- Collection is cleaned before each test requiring it
- Services are checked for availability before running service-dependent tests
- Tests are skipped if required services are not available

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Document isolation strategy for parallel tests with a shared Qdrant instance.

Parallel runs (xdist) can collide on a single collection. Either use a per-test/worker suffix or a unique temporary collection per test in fixtures.

Suggested doc note:

    Use a unique collection name per test/worker (e.g., f"test_crawler_mcp_{request.node.nodeid_hash}" or append PYTEST_XDIST_WORKER) and drop it in teardown to avoid cross-test interference.

🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +120 to +126
### Logging

Adjust log levels in `.env` for more detailed output:
```env
LOG_LEVEL=DEBUG  # For detailed logging
LOG_TO_FILE=true  # To capture logs in files
```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix MD031 and minor wording in Logging section.

Add blank lines around the env fenced block to satisfy MD031 and improve readability.

Apply this diff:

 ### Logging

 Adjust log levels in `.env` for more detailed output:
+
 ```env
 LOG_LEVEL=DEBUG  # For detailed logging
 LOG_TO_FILE=true  # To capture logs in files


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~120-~120: Use correct spacing
Context: ...reakpoint()  # And here ```  ### Logging  Adjust log levels in `.env` for more det...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~122-~122: Use correct spacing
Context: ...vels in `.env` for more detailed output: ```env LOG_LEVEL=DEBUG  # For detailed logging LOG_TO_FILE=true  # To capture logs in files ```  ### Test Data Inspection  Access services di...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

123-123: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 120 to 126, the fenced env code block needs
blank lines before the opening env and after the closing to satisfy MD031
and improve readability; edit the Logging section to insert one blank line
immediately above the env line and one blank line immediately below the closing and keep the env content unchanged (LOG_LEVEL=DEBUG and
LOG_TO_FILE=true), also tidy the surrounding sentence wording if necessary for
clarity.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +128 to +135
### Test Data Inspection

Access services directly in tests:
```python
async def test_inspect_data(vector_service: VectorService):
    info = await vector_service.get_collection_info()
    print(f"Collection has {info['points_count']} points")
```
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Fix MD031 around the “Test Data Inspection” example.

Add blank lines around the code fence.

Apply this diff:

 Access services directly in tests:
+
 ```python
 async def test_inspect_data(vector_service: VectorService):
     info = await vector_service.get_collection_info()
     print(f"Collection has {info['points_count']} points")


<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 LanguageTool</summary>

[grammar] ~128-~128: Use correct spacing
Context: ...s in files ```  ### Test Data Inspection  Access services directly in tests: ```py...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

---

[grammar] ~130-~130: Use correct spacing
Context: ...tion  Access services directly in tests: ```python async def test_inspect_data(vector_service: VectorService):     info = await vector_service.get_collection_info()     print(f"Collection has {info['points_count']} points") ```  ## Performance Considerations  - **Fast Exe...

(QB_NEW_EN_OTHER_ERROR_IDS_5)

</details>
<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

131-131: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 128 to 135, the "Test Data Inspection" example
violates MD031 by lacking blank lines before and after the fenced code block;
fix it by adding a blank line immediately before the opening and a blank line immediately after the closing so the code fence is separated from
surrounding text, preserving the existing code content and spacing inside the
fence.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/README.md
Comment on lines +139 to +143
- **Fast Execution**: In-memory tests run in milliseconds
- **Service Startup**: Allow time for Qdrant/TEI to be fully ready
- **Resource Cleanup**: Tests clean up after themselves
- **Parallel Execution**: Tests can run in parallel (use `pytest-xdist`)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Set expectations for service readiness to reduce flakes.

Add recommended readiness checks/timeouts for Qdrant/TEI in local runs, not only CI. Mention typical warm-up for TEI model load.

You already show health checks in CI; mirror those in a short “Before running integration tests locally” snippet.
🧰 Tools





🤖 Prompt for AI Agents

@jmagar
tests/README.md
Comment on lines +154 to +165
```yaml
- name: Start services
  run: docker-compose up -d

- name: Wait for services
  run: |
    timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'
    timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'
- name: Run tests
  run: uv run pytest -m "not slow"
```
@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Update README example: surround YAML block with blank lines and correct Qdrant port

Please update the tests/README.md example to both satisfy MD031 and align the Qdrant health-check port with your docker-compose.yml mapping:

    Add a blank line before and after the fenced yaml block.
    Change the Qdrant health-check from port 6333 (container port) to 7000 (host port mapped in docker-compose.yml), so the example actually reaches the service.

Suggested diff:

 Example GitHub Actions:

+
 ```yaml
 - name: Start services
   run: docker-compose up -d

 - name: Wait for services
   run: |
-    timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'
+    timeout 60 bash -c 'until curl -f http://localhost:7000/health; do sleep 2; done'
     timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'

 - name: Run tests
   run: uv run pytest -m "not slow"


Tagging this as a critical fix to avoid broken health checks in CI.

<details>
<summary>🧰 Tools</summary>

<details>
<summary>🪛 markdownlint-cli2 (0.17.2)</summary>

154-154: Fenced code blocks should be surrounded by blank lines

(MD031, blanks-around-fences)

</details>

</details>

<details>
<summary>🤖 Prompt for AI Agents</summary>

In tests/README.md around lines 154 to 165, the YAML example block needs a blank
line before and after the fenced yaml to satisfy MD031, and the Qdrant
health-check URL should use host port 7000 (mapped in docker-compose.yml)
instead of 6333 so CI contacts the service correctly; update the example by
adding an empty line above the opening yaml fence and below the closing
fence, and change the curl URL from http://localhost:6333/health to
http://localhost:7000/health in the "Wait for services" step.


</details>

<!-- fingerprinting:phantom:triton:chinchilla -->

<!-- This is an auto-generated comment by CodeRabbit -->

@jmagar
tests/test_config.py
Comment on lines +22 to +53
        # Test that values are loaded from .env file (not hardcoded defaults)
        assert config.server_host == "0.0.0.0"  # From .env SERVER_HOST
        assert config.server_port == 8010  # From .env SERVER_PORT
        assert config.debug is True  # From .env DEBUG=true
        assert config.production is False  # From .env PRODUCTION=false

        # Logging configuration from .env
        assert config.log_level == "INFO"  # From .env LOG_LEVEL
        assert config.log_format == "console"  # From .env LOG_FORMAT
        assert config.log_to_file is True  # From .env LOG_TO_FILE=true

        # Service endpoints from .env
        assert config.qdrant_url == "http://localhost:6333"  # From .env QDRANT_URL
        assert config.qdrant_collection == "crawlerr_documents"  # From .env
        assert config.qdrant_vector_size == 1024  # From .env
        assert config.qdrant_distance == "cosine"  # From .env

        # TEI configuration from .env
        assert config.tei_url == "http://localhost:8080"  # From .env TEI_URL
        assert config.tei_model == "Qwen/Qwen3-Embedding-0.6B"  # From .env TEI_MODEL
        assert config.tei_batch_size == 64  # From .env TEI_BATCH_SIZE

        # Crawling settings from .env
        assert config.chunk_size == 1024  # From .env CHUNK_SIZE
        assert config.chunk_overlap == 200  # From .env CHUNK_OVERLAP
        assert (
            config.reranker_model == "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
        )  # From .env
        assert (
            config.deduplication_strategy == "content_hash"
        )  # From .env DEDUPLICATION_STRATEGY

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Tests rely on a .env in the repo; make them hermetic with monkeypatch

Hardcoding expectations from .env makes tests environment-dependent and flaky in CI. Use monkeypatch to set env vars or pass values directly to CrawlerrSettings in each test.

Example change:

-        config = CrawlerrSettings()
+        monkeypatch = pytest.MonkeyPatch()
+        monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
+        monkeypatch.setenv("SERVER_PORT", "8010")
+        monkeypatch.setenv("DEBUG", "true")
+        monkeypatch.setenv("PRODUCTION", "false")
+        try:
+            config = CrawlerrSettings()
+        finally:
+            monkeypatch.undo()

Repeat pattern where specific .env-driven assertions are made. Alternatively, assert defaults without attributing them to .env.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_config.py
Comment on lines +146 to +157
    @pytest.mark.unit
    def test_log_file_directory_creation(self):
        """Test that log file directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "test.log"

            config = CrawlerrSettings(log_file=str(log_path), log_to_file=True)

            # Directory should be created
            assert log_path.parent.exists()
            assert config.log_file == str(log_path)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Log directory creation: also assert file path normalization

Good coverage for directory creation. Consider also asserting that absolute paths are preserved and that expanduser (~) is handled.

-            assert config.log_file == str(log_path)
+            assert Path(config.log_file) == log_path

🤖 Prompt for AI Agents

@jmagar
tests/test_config.py
Comment on lines +297 to +313
    def test_field_constraints(self):
        """Test field constraints work correctly."""
        # Test max_pages constraints
        config = CrawlerrSettings(crawl_max_pages=1)  # Minimum value
        assert config.crawl_max_pages == 1

        config = CrawlerrSettings(crawl_max_pages=1000)  # Maximum value
        assert config.crawl_max_pages == 1000

        # Invalid: below minimum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=1001)

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Enforce min/max constraints for crawl_max_pages
The CrawlerrSettings model currently defines crawl_max_pages without any ge/le bounds, but the tests in tests/test_config.py (lines 297–313) expect values below 1 and above 1000 to raise a ValueError. Add the missing constraints so that the implementation matches the tests:

• File: crawler_mcp/config.py (around line 92)

-    crawl_max_pages: int = Field(default=1000, alias="CRAWL_MAX_PAGES")
+    crawl_max_pages: int = Field(
+        default=1000,
+        alias="CRAWL_MAX_PAGES",
+        ge=1,
+        le=1000,
+    )

This change enforces the minimum of 1 and maximum of 1000, ensuring the existing tests pass without modification.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_edge_cases.py
Comment on lines +291 to +294
            # The implementation creates adjacent chunks (no gaps, but also no overlap)
            # when the calculated overlap would cause the next chunk to start before current end
            assert next_start <= current_end  # No gaps between chunks

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Tighten overlap assertion to match docstring intent

Comment says “no gaps, but also no overlap,” yet the assertion allows overlap (<=). Use equality to assert adjacency.

Apply:

-            assert next_start <= current_end  # No gaps between chunks
+            assert next_start == current_end  # Adjacent chunks: no gaps, no overlap

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_edge_cases.py
Comment on lines +686 to +691
        # Process crawl result (should detect and delete orphans)
        result = await service.process_crawl_result(
            crawl_result, progress_callback=progress_callback
        )

        # Should have called delete_chunks_by_ids
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused local variable result (ruff F841)

The test doesn’t assert on the returned value. Drop the assignment to avoid the lint violation.

Apply:

-        result = await service.process_crawl_result(
-            crawl_result, progress_callback=progress_callback
-        )
+        await service.process_crawl_result(
+            crawl_result, progress_callback=progress_callback
+        )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +23 to +36
@pytest.fixture
async def rag_service():
    """Create a real RagService instance with live services."""
    service = RagService()
    async with service:
        yield service


@pytest.fixture
async def test_client():
    """Create FastMCP test client with live services."""
    async with Client(mcp) as client:
        yield client

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Fixture duplication with conftest; prefer shared fixtures or unique names

This fixture shadows a likely identical rag_service fixture in tests/conftest.py. Prefer reusing the shared fixture to avoid divergent configuration or rename to make intent explicit (e.g., rag_service_live).

-@pytest.fixture
-async def rag_service():
+@pytest.fixture(name="rag_service_live")
+async def rag_service_live():
     """Create a real RagService instance with live services."""
     service = RagService()
     async with service:
         yield service

Then adjust test signatures accordingly. Alternatively, delete this fixture and import the one from conftest.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +41 to +47
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self, rag_service):
        """Test RAG service initializes properly with live services."""
        assert rag_service.vector_service is not None
        assert rag_service.embedding_service is not None
        assert rag_service._context_count > 0

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Avoid asserting on private attributes

Asserting on _context_count couples tests to internals and may break with harmless refactors.

Remove Line 46 and rely on public behaviors (health_check, context manager entry/exit tests already cover this).

-        assert rag_service._context_count > 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +112 to +118
        # Query for the content
        query = RagQuery(
            query="FastMCP testing capabilities",
            sources=["https://example.com/query-test"],
            limit=5,
            score_threshold=0.1,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix RagQuery field names: use source_filters and min_score

The RagQuery model defines source_filters and min_score, not sources and score_threshold.

Apply:

-        query = RagQuery(
-            query="FastMCP testing capabilities",
-            sources=["https://example.com/query-test"],
-            limit=5,
-            score_threshold=0.1,
-        )
+        query = RagQuery(
+            query="FastMCP testing capabilities",
+            source_filters=["https://example.com/query-test"],
+            limit=5,
+            min_score=0.1,
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +154 to +161
        deleted_count = await rag_service.delete_source(
            "https://example.com/delete-test"
        )

        # Verify deletion
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

delete_source returns bool, not int

According to RagService.delete_source, the return type is bool (True if any documents were deleted). The current assertions expect int and will misinterpret the result.

Use boolean semantics:

-        deleted_count = await rag_service.delete_source(
+        deleted = await rag_service.delete_source(
             "https://example.com/delete-test"
         )

-        # Verify deletion
-        assert isinstance(deleted_count, int)
-        assert deleted_count >= 0
+        # Verify deletion
+        assert isinstance(deleted, bool)

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +257 to +314
    @pytest.mark.asyncio
    async def test_rag_query_tool_integration(self, test_client):
        """Test rag_query tool with live services."""
        # First add some content using scrape tool
        scrape_result = await test_client.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Verify scrape worked
        assert isinstance(scrape_result.data, dict)

        # Query the scraped content
        query_result = await test_client.call_tool(
            "rag_query", {"query": "example content", "limit": 5, "min_score": 0.0}
        )

        # Verify query results
        assert isinstance(query_result.data, dict)
        assert "query" in query_result.data
        assert "matches" in query_result.data
        assert "total_matches" in query_result.data

    @pytest.mark.asyncio
    async def test_list_sources_integration(self, test_client):
        """Test list_sources tool with live services."""
        result = await test_client.call_tool("list_sources", {})

        assert isinstance(result.data, dict)
        assert "sources" in result.data
        assert "pagination" in result.data
        assert isinstance(result.data["sources"], list)

    @pytest.mark.asyncio
    async def test_get_rag_stats_integration(self, test_client):
        """Test get_rag_stats tool with live services."""
        result = await test_client.call_tool("get_rag_stats", {})

        assert isinstance(result.data, dict)
        assert "rag_system" in result.data or "health_summary" in result.data

    @pytest.mark.asyncio
    async def test_delete_source_tool_integration(self, test_client):
        """Test delete_source tool with live services."""
        # First add content to delete
        await test_client.call_tool(
            "scrape",
            {"url": "https://example.com/delete-integration", "process_with_rag": True},
        )

        # Delete the source
        result = await test_client.call_tool(
            "delete_source",
            {"source_url": "https://example.com/delete-integration", "confirm": True},
        )

        assert isinstance(result.data, dict)
        assert "success" in result.data or "documents_deleted" in result.data

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Mark MCP/live-service tests to be skippable in CI without services

These tests depend on running services and network; mark them to allow selective execution.

At module top add:

+pytestmark = [pytest.mark.integration, pytest.mark.requires_services]

And optionally skip when health checks fail:

if not (await rag_service.health_check()).get("vector_service", {}).get("ok"):
    pytest.skip("Vector service unavailable", allow_module_level=False)

@jmagar
tests/test_rag_integration.py
Comment on lines +401 to +404
        query = RagQuery(
            query="FastMCP testing practices", limit=3, score_threshold=0.0
        )

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix RagQuery param name in reranking test

Use min_score instead of score_threshold.

-        query = RagQuery(
-            query="FastMCP testing practices", limit=3, score_threshold=0.0
-        )
+        query = RagQuery(
+            query="FastMCP testing practices", limit=3, min_score=0.0
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_integration.py
Comment on lines +455 to +459
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for result in results:
            assert isinstance(result, dict)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Harden concurrent gather to surface exceptions explicitly

Return exceptions aids debugging and prevents cancellations masking an error.

-        results = await asyncio.gather(*tasks)
+        results = await asyncio.gather(*tasks, return_exceptions=True)
@@
-        for result in results:
-            assert isinstance(result, dict)
-            assert "chunks_created" in result
+        for result in results:
+            if isinstance(result, Exception):
+                raise result
+            assert isinstance(result, dict)
+            assert "chunks_created" in result

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +10 to +16
import pytest

from crawler_mcp.models.sources import (
    SourceInfo,
    SourceMetadata,
    SourceType,
)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Target Pydantic's ValidationError instead of broad exception tuples

Catching (ValueError, TypeError) is imprecise for Pydantic v2; use ValidationError.

@@
-import pytest
+import pytest
+from pydantic import ValidationError
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo()  # Missing required fields
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo(id="test")  # Missing url and source_type
@@
-        with pytest.raises((ValueError, TypeError)):
+        with pytest.raises(ValidationError):
             SourceInfo(id="test", url="https://test.com")  # Missing source_type

Also applies to: 359-367, 362-367, 365-367
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +301 to +304
            total_content_length=15000,
            average_chunk_size=1000.0,
            metadata=complex_metadata,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" here

Avoid passing unsupported fields.

-            average_chunk_size=1000.0,

🤖 Prompt for AI Agents

@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Review continued from previous batch...
crawler_mcp/core/orchestrator.py
Comment on lines +296 to +342
            # Extract content using proper crawl4ai patterns
            # result.markdown is a MarkdownGenerationResult object with raw_markdown and fit_markdown attributes
            best_content = ""
            if result.markdown:
                try:
                    # Check if result.markdown is an integer (hash ID issue)
                    if isinstance(result.markdown, int):
                        self.logger.warning(
                            f"Detected integer markdown ({result.markdown}) for {url}, using empty content"
                        )
                        best_content = ""
                    else:
                        # First try fit_markdown (filtered content) if available
                        if (
                            hasattr(result.markdown, "fit_markdown")
                            and result.markdown.fit_markdown
                        ):
                            content = result.markdown.fit_markdown.strip()
                            if len(content) > 16:  # Avoid hash placeholders
                                best_content = content
                        # Fall back to raw_markdown (full content)
                        elif (
                            hasattr(result.markdown, "raw_markdown")
                            and result.markdown.raw_markdown
                        ):
                            content = result.markdown.raw_markdown.strip()
                            if len(content) > 16:  # Avoid hash placeholders
                                best_content = content
                        else:
                            best_content = ""
                except (AttributeError, TypeError) as e:
                    if "'int' object has no attribute" in str(e):
                        self.logger.warning(
                            f"Caught integer markdown issue for {url}, using empty content"
                        )
                        best_content = ""
                    else:
                        self.logger.warning(
                            f"Failed to extract markdown content for {url}: {e}"
                        )
                        best_content = ""
                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract markdown content for {url}: {e}"
                    )
                    best_content = ""

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Content extraction heuristics: guardrails are helpful but some thresholds are brittle

    Hard-coding a 16-character threshold can discard valid short pages/snippets.
    The “integer markdown” branch is good defensive coding; keep it.
    Consider preferring word-count thresholds aligned with settings.crawl_min_words and only then falling back to character count.

-                            if len(content) > 16:  # Avoid hash placeholders
-                                best_content = content
+                            words = content.split()
+                            if len(words) >= max(1, settings.crawl_min_words):
+                                best_content = content
+                            elif len(content) > 16:
+                                # last-resort guard to avoid hash placeholders
+                                best_content = content

Also consider logging when raw content is dropped for being “too short” to aid debugging with real sites.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/orchestrator.py
Comment on lines +345 to +364
                object_indicators = [
                    "CrawlResultContainer(",
                    "CrawlResult(",
                    "<property object at",
                    "__dict__",
                    "object at 0x",
                    "MarkdownGenerationResult(",
                ]

                if any(indicator in best_content for indicator in object_indicators):
                    self.logger.warning(
                        f"Detected object representation in content for {url}, clearing"
                    )
                    best_content = ""

                # Validate content quality - avoid storing trivial content
                if best_content and len(best_content.strip()) < 10:
                    self.logger.debug(f"Content too short for {url}, clearing")
                    best_content = ""

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Object-representation filters are good, but consider a single utility with tests

You’re checking multiple “object-like” indicators to sanitize content. Encapsulate this into a reusable helper and add unit tests so we don’t accidentally regress or miss cases.

-                object_indicators = [
+                object_indicators = [
                     "CrawlResultContainer(",
                     "CrawlResult(",
                     "<property object at",
                     "__dict__",
                     "object at 0x",
                     "MarkdownGenerationResult(",
                 ]

Follow-up: I can extract this list and the detection into utils/sanitization.py and include tests covering true positives/negatives. Want me to draft it?

    Committable suggestion skipped: line range outside the PR's diff.

@jmagar
tests/conftest.py
Comment on lines +37 to +38
from crawler_mcp.config import CrawlerrSettings, settings
from crawler_mcp.core import EmbeddingService, RagService, VectorService
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Imports not at the top (E402); consider moving to the main import block

While harmless at runtime, moving these imports to the top makes linters happy and avoids mixed top-level logic before imports.

-# (move these up near other imports)
-from crawler_mcp.config import CrawlerrSettings, settings
-from crawler_mcp.core import EmbeddingService, RagService, VectorService
+from crawler_mcp.config import CrawlerrSettings, settings
+from crawler_mcp.core import EmbeddingService, RagService, VectorService

    Committable suggestion skipped: line range outside the PR's diff.

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +101 to +108
    # Cleanup: Delete test collection after tests
    try:
        async with VectorService() as vector_service:
            await vector_service._client.delete_collection(
                test_settings.qdrant_collection
            )
    except Exception:
        pass  # Cleanup is best-effort
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Tests reach into VectorService internals via _client; avoid private attribute access

Both the session teardown and per-test cleanup call vector_service._client.delete_collection(...). That’s brittle and couples tests to implementation details.

Option A (preferred): add a public drop_collection(collection_name: str) method on VectorService and call that in tests.

Option B: in tests, resolve a public client if available, then fall back:

-        async with VectorService() as vector_service:
-            await vector_service._client.delete_collection(
-                test_settings.qdrant_collection
-            )
+        async with VectorService() as vector_service:
+            client = getattr(vector_service, "client", None) or getattr(vector_service, "_client", None)
+            if client is not None:
+                await client.delete_collection(test_settings.qdrant_collection)

Also apply the same change in clean_test_collection().

Also applies to: 121-132
@jmagar
tests/test_core_services.py
Comment on lines +148 to +150
        success = await vector_service.upsert_documents(documents)
        assert isinstance(success, (bool, int))  # May return count or boolean

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Ruff UP038: tuple-of-types in isinstance. Consider suppressing or targeting union if supported.

Ruff suggests int | bool, but isinstance(x, int | bool) is only valid in newer Python versions. To keep compatibility, either suppress UP038 or split the check.

Option A (suppress):

-        assert isinstance(success, (bool, int))  # May return count or boolean
+        assert isinstance(success, (bool, int))  # May return count or boolean  # noqa: UP038

Option B (portable split):

-        assert isinstance(success, (bool, int))  # May return count or boolean
+        assert isinstance(success, bool) or isinstance(success, int)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_core_services.py
Comment on lines +171 to +178
            # Results should be SearchMatch objects with document and score properties
            assert hasattr(result, "document") or hasattr(result, "score")
            if hasattr(result, "document"):
                assert hasattr(result.document, "content")
            if hasattr(result, "score"):
                assert isinstance(result.score, (int, float))
                assert 0 <= result.score <= 1

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same UP038 for score type check.

-                assert isinstance(result.score, (int, float))
+                assert isinstance(result.score, (int, float))  # noqa: UP038

Or split into two isinstance checks as above.
📝 Committable suggestion


🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +35 to +46
        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_pages=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_pages=3000)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_depth=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            CrawlRequest(url="https://example.com", max_depth=11)

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Catch Pydantic ValidationError instead of blanket Exception

Be precise about the expected failure type; aligns with Ruff B017 hint.

-import pytest
+import pytest
+from pydantic import ValidationError
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_pages=0)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_pages=3000)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_depth=0)
@@
-        with pytest.raises(Exception):  # Pydantic ValidationError
+        with pytest.raises(ValidationError):
             CrawlRequest(url="https://example.com", max_depth=11)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +299 to +305
            # Mock pathlib.Path.iterdir to raise an exception
            with patch(
                "pathlib.Path.iterdir", side_effect=PermissionError("Access denied")
            ):
                result = await crawler.execute(request)

                # Should handle error gracefully - directory crawler may just log warnings
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Patch the correct API to simulate permission errors

execute() uses rglob/glob in _discover_files; patching Path.iterdir won't affect it. Patch Path.rglob instead.

-            # Mock pathlib.Path.iterdir to raise an exception
-            with patch(
-                "pathlib.Path.iterdir", side_effect=PermissionError("Access denied")
-            ):
+            # Mock recursive glob to raise an exception during discovery
+            with patch(
+                "pathlib.Path.rglob", side_effect=PermissionError("Access denied")
+            ):

🤖 Prompt for AI Agents

@jmagar
tests/test_crawler_comprehensive.py
Comment on lines +369 to +381
            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=mock_create_subprocess_exec,
                ),
                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
            ):
                # Mock the clone directory to return our test structure
                with patch.object(crawler, "_clone_repository", return_value=repo_dir):
                    result = await crawler.execute(request)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Combine nested with blocks (SIM117) for clarity

Flatten the nested context managers to a single with.

-            with (
-                patch(
-                    "asyncio.create_subprocess_exec",
-                    side_effect=mock_create_subprocess_exec,
-                ),
-                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
-                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
-                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
-            ):
-                # Mock the clone directory to return our test structure
-                with patch.object(crawler, "_clone_repository", return_value=repo_dir):
-                    result = await crawler.execute(request)
+            with (
+                patch(
+                    "asyncio.create_subprocess_exec",
+                    side_effect=mock_create_subprocess_exec,
+                ),
+                patch.object(crawler, "pre_execute_setup", new_callable=AsyncMock),
+                patch.object(crawler, "post_execute_cleanup", new_callable=AsyncMock),
+                patch.object(crawler, "_initialize_managers", new_callable=AsyncMock),
+                patch.object(crawler, "_clone_repository", return_value=repo_dir),
+            ):
+                result = await crawler.execute(request)

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +32 to +96
    async def test_validate_request_valid_urls(self):
        """Test validation with valid web crawl requests."""
        crawler = WebCrawlStrategy()

        # Valid single URL
        request = CrawlRequest(url="https://example.com", max_pages=50, max_depth=2)
        assert await crawler.validate_request(request) is True

        # Valid multiple URLs
        request_multi = CrawlRequest(
            url=["https://example.com", "https://test.com"], max_pages=100, max_depth=3
        )
        assert await crawler.validate_request(request_multi) is True

    @pytest.mark.unit
    async def test_validate_request_invalid_urls(self):
        """Test validation with invalid web crawl requests."""
        crawler = WebCrawlStrategy()

        # No URL provided (empty list)
        request_no_url = CrawlRequest(url=[])
        assert await crawler.validate_request(request_no_url) is False

        # Test with values that exceed crawler's custom validation limits
        # (WebCrawlStrategy checks max_pages <= 2000, but Pydantic allows <= 1000)
        request_high_pages = CrawlRequest(
            url="https://example.com",
            max_pages=1000,  # At Pydantic limit but valid for Pydantic
        )
        # This should pass since 1000 is within both Pydantic and crawler limits
        assert await crawler.validate_request(request_high_pages) is True

        # Test max_depth boundary (crawler allows <= 5, Pydantic allows <= 10)
        request_high_depth = CrawlRequest(
            url="https://example.com",
            max_depth=6,  # Exceeds crawler limit but within Pydantic limit
        )
        assert await crawler.validate_request(request_high_depth) is False

    @pytest.mark.unit
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()

            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

    @pytest.mark.unit
    async def test_validate_request_none_values(self):
        """Test validation with None values."""
        crawler = WebCrawlStrategy()

        # None values should be allowed (defaults will be used)
        request = CrawlRequest(
            url="https://example.com", max_pages=None, max_depth=None
        )
        assert await crawler.validate_request(request) is True


@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Mark async tests with pytest-asyncio (or enable asyncio_mode=auto).

Several tests are async but lack @pytest.mark.asyncio. Unless pytest-asyncio’s auto mode is enabled in configuration, these will be collected but not awaited, leading to false positives/failures. Prefer a module-level marker to keep noise low.

Apply this minimal change near the imports:

 import pytest
+pytestmark = pytest.mark.asyncio

Also applies to: 109-144, 158-204, 227-264
🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +55 to +70
        # Test with values that exceed crawler's custom validation limits
        # (WebCrawlStrategy checks max_pages <= 2000, but Pydantic allows <= 1000)
        request_high_pages = CrawlRequest(
            url="https://example.com",
            max_pages=1000,  # At Pydantic limit but valid for Pydantic
        )
        # This should pass since 1000 is within both Pydantic and crawler limits
        assert await crawler.validate_request(request_high_pages) is True

        # Test max_depth boundary (crawler allows <= 5, Pydantic allows <= 10)
        request_high_depth = CrawlRequest(
            url="https://example.com",
            max_depth=6,  # Exceeds crawler limit but within Pydantic limit
        )
        assert await crawler.validate_request(request_high_depth) is False

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Boundary assertions are good; add a positive case for max_depth=5.

You test that 6 fails; add an assertion that 5 passes to pin the upper bound.

         request_high_depth = CrawlRequest(
             url="https://example.com",
             max_depth=6,  # Exceeds crawler limit but within Pydantic limit
         )
         assert await crawler.validate_request(request_high_depth) is False
+
+        # Upper boundary should pass
+        request_depth_edge = CrawlRequest(url="https://example.com", max_depth=5)
+        assert await crawler.validate_request(request_depth_edge) is True

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +72 to +84
    async def test_initialize_managers(self):
        """Test memory manager initialization."""
        crawler = WebCrawlStrategy()

        with patch("crawler_mcp.core.memory.get_memory_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            await crawler._initialize_managers()

            assert crawler.memory_manager is mock_manager
            mock_get_manager.assert_called_once()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Private method test is acceptable; assert idempotency and no re-init.

Since _initialize_managers is private, keep the test but also assert it doesn’t reinitialize on subsequent calls.

             await crawler._initialize_managers()

             assert crawler.memory_manager is mock_manager
             mock_get_manager.assert_called_once()
+
+            # Calling again should not fetch a new manager
+            await crawler._initialize_managers()
+            mock_get_manager.assert_called_once()

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +206 to +225
class TestCrawlerEdgeCases:
    """Test edge cases and error conditions across crawlers."""

    @pytest.mark.unit
    def test_crawler_inheritance(self):
        """Test that all crawlers inherit from base properly."""
        web_crawler = WebCrawlStrategy()
        dir_crawler = DirectoryCrawlStrategy()
        repo_crawler = RepositoryCrawlStrategy()

        # All should have logger from base class
        assert hasattr(web_crawler, "logger")
        assert hasattr(dir_crawler, "logger")
        assert hasattr(repo_crawler, "logger")

        # All should be callable instances
        assert callable(web_crawler.validate_request)
        assert callable(dir_crawler.validate_request)
        assert callable(repo_crawler.validate_request)

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Tighten edge‐case validation tests to avoid catch-all exceptions

The test_empty_request_handling currently wraps every validate_request call in a broad try / except Exception, which risks silently hiding regressions. Instead, for truly minimal requests you should:

    Remove the try { … } catch Exception: pass around each call to await …validate_request.

    Assert that the result is always a boolean. For example, in tests/test_crawlers.py::TestCrawlerEdgeCases::test_empty_request_handling replace:

  try:

      result = await web_crawler.validate_request(web_request)

      assert result in [True, False]

  except Exception:

      # Some crawlers might throw exceptions for invalid URLs

      # This is acceptable behavior

      pass


with

```diff

  result = await web_crawler.validate_request(web_request)

  assert isinstance(result, bool)


And do the same for the directory and repository sections.

If a specific strategy is expected to raise a known exception for truly invalid inputs (for example, a parsing error or a ValueError when the repo URL is grossly malformed), catch only that exception type:

try:
    result = await repo_crawler.validate_request(bad_repo_request)
    assert isinstance(result, bool)
except ValueError:
    # Explicit, documented failure mode for malformed URLs
    pass

This change makes sure:

    You never hide unexpected errors.
    You verify that every “minimal” request path yields a boolean.
    You still allow for documented, intentional exceptions on truly invalid inputs.

🤖 Prompt for AI Agents

@jmagar
tests/test_crawlers.py
Comment on lines +270 to +282
    def test_web_crawler_uses_settings(self):
        """Test that web crawler accesses configuration settings."""
        crawler = WebCrawlStrategy()

        # This will exercise settings access during validation
        with patch("crawler_mcp.crawlers.web.settings") as mock_settings:
            mock_settings.crawl_headless = True
            mock_settings.crawl_browser = "chromium"

            # Just instantiating should exercise some settings access
            assert crawler is not None

    @pytest.mark.unit
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

This test doesn’t exercise settings; convert to async and call validate_request.

The test name and comment imply we exercise settings access “during validation,” but no validation is performed. Convert to async and invoke validate_request to actually touch the settings path.

-    def test_web_crawler_uses_settings(self):
+    async def test_web_crawler_uses_settings(self):
         """Test that web crawler accesses configuration settings."""
         crawler = WebCrawlStrategy()

         # This will exercise settings access during validation
         with patch("crawler_mcp.crawlers.web.settings") as mock_settings:
             mock_settings.crawl_headless = True
             mock_settings.crawl_browser = "chromium"

-            # Just instantiating should exercise some settings access
-            assert crawler is not None
+            # Call a method that consults settings
+            req = CrawlRequest(url="https://example.com")
+            result = await crawler.validate_request(req)
+            assert isinstance(result, bool)

🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +40 to +50
        await asyncio.sleep(2)

        # Step 2: Verify content was indexed by checking stats
        stats_result = await mcp_client.call_tool("get_rag_stats", {})
        stats = stats_result.data

        assert stats["total_sources"] > 0
        assert stats["total_chunks"] > 0
        assert "directory" in stats["source_types"]

        # Step 3: Query the indexed content
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Fixed sleeps make tests flaky; prefer polling for eventual consistency.

Replace static asyncio.sleep(2) with a short polling loop that checks get_rag_stats for the expected increase or times out (e.g., within ~10s). This reduces flakiness on slow CI.

Example helper (inlined for brevity):

-        await asyncio.sleep(2)
+        async def wait_for_nonzero_sources(client: Client, timeout_s: float = 10.0):
+            import time
+            end = time.time() + timeout_s
+            while time.time() < end:
+                stats = (await client.call_tool("get_rag_stats", {})).data
+                if stats.get("total_sources", 0) > 0:
+                    return stats
+                await asyncio.sleep(0.25)
+            return stats
+
+        stats_result = await wait_for_nonzero_sources(mcp_client)
+        stats = stats_result

Apply analogous polling in other spots waiting for indexing.

Also applies to: 120-129, 299-311
🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +140 to +143
        rag_data = query_result.data
        # Should find some results (may be 0 if content is minimal)
        assert rag_data["total_results"] >= 0

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Trivial assertion: total_results >= 0 is tautological.

Consider asserting types and shape instead:

-        assert rag_data["total_results"] >= 0
+        assert isinstance(rag_data.get("total_results"), int)
+        assert isinstance(rag_data.get("results"), list)

🤖 Prompt for AI Agents

@jmagar
tests/test_integration_flow.py
Comment on lines +239 to +241
        directory_path = sample_text_files[0].parent

        # Create multiple concurrent operations
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused variable (Ruff F841).

directory_path is assigned but never used in this test.

-        directory_path = sample_text_files[0].parent

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_models.py
Comment on lines +199 to +206
        source = SourceInfo(
            source_id="test_source_123",
            source_type=SourceType.WEBPAGE,
            url="https://test.com/page",
            title="Test Page",
            metadata=metadata,
        )

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix SourceInfo field name: use id (not source_id)

The model defines id: str, not source_id. Adjust constructor and assertions.

-        source = SourceInfo(
-            source_id="test_source_123",
-            source_type=SourceType.WEBPAGE,
-            url="https://test.com/page",
-            title="Test Page",
-            metadata=metadata,
-        )
+        source = SourceInfo(
+            id="test_source_123",
+            source_type=SourceType.WEBPAGE,
+            url="https://test.com/page",
+            title="Test Page",
+            metadata=metadata,
+        )
@@
-        assert source.source_id == "test_source_123"
+        assert source.id == "test_source_123"

Also applies to: 207-212
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +31 to +36
@pytest.fixture
async def test_server():
    """Create a test server instance for in-memory testing."""
    async with Client(mcp) as client:
        yield client

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Optional: mark MCP integration tests as integration to avoid flakiness when external services are unavailable

If CI lacks live vector/embedding backends, consider pytest markers to allow selective runs.

I can add pytest markers and CI matrix guards if you want.

Also applies to: 623-676
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +86 to +99
        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("test query", 10, 0.7, ["source1"], True, mock_result)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Instantiate RagResult with only declared fields

average_score, best_match_score, has_high_confidence_matches appear to be computed properties, not constructor fields. Keep to the model surface to avoid silent drops.

         mock_result = RagResult(
             query="test query",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +109 to +121
        mock_result = RagResult(
            query="test query",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for TTL test

Mirror the change above.

         mock_result = RagResult(
             query="test query",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +140 to +178
        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result3 = RagResult(
            query="query3",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for size-limit test (all three instances)

Keep RagResult init minimal and consistent.

         result1 = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
@@
         result2 = RagResult(
             query="query2",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
@@
         result3 = RagResult(
             query="query3",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +202 to +227
        result1 = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        result2 = RagResult(
            query="query2",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for clear() test

Remove non-declared fields.

         result1 = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )
         result2 = RagResult(
             query="query2",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +253 to +266
        result = RagResult(
            query="query1",
            matches=[],
            total_matches=0,
            processing_time=0.1,
            embedding_time=0.05,
            search_time=0.03,
            rerank_time=0.02,
            average_score=0.0,
            best_match_score=0.0,
            has_high_confidence_matches=False,
            timestamp=datetime.utcnow(),
        )
        cache.put("query1", 10, 0.7, ["source1"], True, result)
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same constructor cleanup for stats() test

Align with declared fields only.

         result = RagResult(
             query="query1",
             matches=[],
             total_matches=0,
             processing_time=0.1,
             embedding_time=0.05,
             search_time=0.03,
-            rerank_time=0.02,
-            average_score=0.0,
-            best_match_score=0.0,
-            has_high_confidence_matches=False,
+            rerank_time=0.02,
             timestamp=datetime.utcnow(),
         )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +530 to +535
        query = RagQuery(
            query="FastMCP testing capabilities",
            sources=["https://example.com/test-query"],
            limit=5,
            score_threshold=0.1,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Use RagQuery’s actual field names: source_filters, min_score

The model exposes source_filters and min_score (not sources/score_threshold).

-        query = RagQuery(
-            query="FastMCP testing capabilities",
-            sources=["https://example.com/test-query"],
-            limit=5,
-            score_threshold=0.1,
-        )
+        query = RagQuery(
+            query="FastMCP testing capabilities",
+            source_filters=["https://example.com/test-query"],
+            limit=5,
+            min_score=0.1,
+        )

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +609 to +612
        # delete_source returns an integer count, not a dict
        assert isinstance(result, int)
        assert result >= 0  # Should return number of deleted documents

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Align test_delete_source with actual return type of delete_source

The RagService.delete_source method in crawler_mcp/core/rag.py currently returns a boolean indicating success, not an integer count. The existing test in tests/test_rag_comprehensive.py (around lines 609–612) assumes an integer and will fail. Update the test to accept either a bool or an int, ensuring backward compatibility if the implementation changes in the future.

Required changes in tests/test_rag_comprehensive.py:

• File: tests/test_rag_comprehensive.py
Lines: around 609–612

-        # delete_source returns an integer count, not a dict
-        assert isinstance(result, int)
-        assert result >= 0  # Should return number of deleted documents
+        # delete_source may return a bool (success) or an integer count
+        assert isinstance(result, (bool, int))
+        if isinstance(result, int):
+            assert result >= 0  # Number of deleted documents must be non-negative

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_comprehensive.py
Comment on lines +630 to +637
        scrape_result = await test_server.call_tool(
            "scrape", {"url": "https://example.com", "process_with_rag": True}
        )

        # Now test rag_query
        query_result = await test_server.call_tool(
            "rag_query", {"query": "test content", "limit": 5}
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unused variable assignment (Ruff F841)

No need to bind the scrape result.

-        scrape_result = await test_server.call_tool(
-            "scrape", {"url": "https://example.com", "process_with_rag": True}
-        )
+        await test_server.call_tool(
+            "scrape", {"url": "https://example.com", "process_with_rag": True}
+        )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_server_health.py
Comment on lines +55 to +59
        server = info_data["server"]
        assert server["name"] == "Crawlerr"
        assert server["framework"] == "FastMCP 2.0+"
        assert server["version"] == "0.1.0"

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: avoid hard-coding server name/framework/version unless guaranteed.

If these values are configurable, prefer presence/type checks or fetch from settings.

-        assert server["name"] == "Crawlerr"
-        assert server["framework"] == "FastMCP 2.0+"
-        assert server["version"] == "0.1.0"
+        assert isinstance(server.get("name"), str) and server["name"]
+        assert isinstance(server.get("framework"), str) and server["framework"]
+        assert isinstance(server.get("version"), str) and server["version"]

🤖 Prompt for AI Agents

@jmagar
tests/test_server.py
Comment on lines +124 to +150
            patch("crawler_mcp.server.EmbeddingService") as mock_embedding,
            patch("crawler_mcp.server.VectorService") as mock_vector,
            patch("crawler_mcp.server.RagService") as mock_rag,
        ):
            # Mock embedding service
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.health_check.return_value = True
            mock_embedding_instance.get_model_info.return_value = {
                "model": "test-model"
            }
            mock_embedding().__aenter__.return_value = mock_embedding_instance

            # Mock vector service
            mock_vector_instance = AsyncMock()
            mock_vector_instance.health_check.return_value = True
            mock_vector_instance.get_collection_info.return_value = {"points": 100}
            mock_vector().__aenter__.return_value = mock_vector_instance

            # Mock RAG service
            mock_rag_instance = AsyncMock()
            mock_rag_instance.health_check.return_value = {
                "embedding": True,
                "vector": True,
            }
            mock_rag_instance.get_stats.return_value = {"documents": 100}
            mock_rag().__aenter__.return_value = mock_rag_instance

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Patching server services is redundant in this test fixture context

Because you define a local health_check tool within the test_server fixture, the patches to crawler_mcp.server.* services are not exercised in this test. This isn’t harmful but adds noise.

    Remove the patch("crawler_mcp.server.*") context from test_health_check_tool_mocked, or alternatively, add a separate test that targets the real server module’s health_check.

🤖 Prompt for AI Agents

@jmagar
tests/test_server.py
Comment on lines +711 to +757
class TestServerConfiguration:
    """Test server configuration and startup functionality."""

    def test_server_settings_loaded(self):
        """Test that server settings are properly loaded."""
        # Check key settings are available
        assert hasattr(settings, "server_host")
        assert hasattr(settings, "server_port")
        assert hasattr(settings, "debug")
        assert hasattr(settings, "tei_url")
        assert hasattr(settings, "qdrant_url")

        # Check types
        assert isinstance(settings.server_host, str)
        assert isinstance(settings.server_port, int)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.tei_url, str)
        assert isinstance(settings.qdrant_url, str)

    @pytest.mark.asyncio
    async def test_mcp_instance_tools(self, test_server):
        """Test that MCP server instance has correct tools registered."""
        # Check test server is created
        assert test_server is not None
        assert hasattr(test_server, "get_tools")
        assert callable(test_server.get_tools)

        # Check that tools are registered
        tools = await test_server.get_tools()
        assert len(tools) > 0

        # Check for expected tools
        tool_names = list(tools.keys())
        expected_tools = [
            "health_check",
            "get_server_info",
            "scrape",
            "crawl",
            "rag_query",
            "list_sources",
            "get_rag_stats",
            "delete_source",
        ]

        for expected in expected_tools:
            assert expected in tool_names

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Tool registration assertions are appropriate and future-proof

Asserting get_tools() and expected tool names helps catch accidental deregistration. Consider also asserting that help/params for one tool are well-formed to detect schema regressions.
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +122 to +128
            average_chunk_size=1000.0,
            created_at=created_time,
            updated_at=updated_time,
            last_crawled=last_crawled,
            is_stale=False,
            metadata=metadata,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Fix unsupported fields and wrong attribute access (average_chunk_size/is_stale)

    Remove unsupported "average_chunk_size" and "is_stale" kwargs from SourceInfo().
    Assert via the computed property avg_chunk_size, not a non-existent "average_chunk_size" attribute.

         source = SourceInfo(
             id="source-123",
             url="https://example.com/page",
             title="Test Page",
             source_type=SourceType.WEBPAGE,
             status="active",
             chunk_count=5,
             total_content_length=5000,
-            average_chunk_size=1000.0,
             created_at=created_time,
             updated_at=updated_time,
             last_crawled=last_crawled,
-            is_stale=False,
             metadata=metadata,
         )
@@
-        assert source.average_chunk_size == 1000.0
+        assert source.avg_chunk_size == 1000.0
-        assert source.is_stale is False
+        assert source.is_stale is False

Also applies to: 137-143
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +150 to +160
        assert source.id == "min-source"
        assert source.url == "https://minimal.com"
        assert source.source_type == SourceType.WEBPAGE
        assert source.title is None
        assert source.status == "active"  # Default
        assert source.chunk_count == 0
        assert source.total_content_length == 0
        assert source.average_chunk_size == 0.0
        assert source.is_stale is False
        assert isinstance(source.metadata, SourceMetadata)

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Defaults test: use avg_chunk_size property

The model exposes avg_chunk_size; "average_chunk_size" isn't a field.

-        assert source.average_chunk_size == 0.0
+        assert source.avg_chunk_size == 0.0

Also applies to: 155-159, 157-157
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_comprehensive.py
Comment on lines +161 to +173
    def test_source_info_time_fields(self):
        """Test automatic time field population."""
        before_creation = datetime.utcnow()
        source = SourceInfo(
            id="time-test", url="https://time.com", source_type=SourceType.WEBPAGE
        )
        after_creation = datetime.utcnow()

        # Should have created_at and updated_at set automatically
        assert before_creation <= source.created_at <= after_creation
        assert before_creation <= source.updated_at <= after_creation
        assert source.created_at == source.updated_at  # Should be same initially

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Relax equality of created_at/updated_at

They are set via separate default factories; strict equality is brittle.

-        assert source.created_at == source.updated_at  # Should be same initially
+        # Should be very close initially
+        assert abs((source.created_at - source.updated_at).total_seconds()) < 0.5

Also applies to: 172-172
🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
            source_type=SourceType.WEBPAGE,
            chunk_count=5,
            total_content_length=5000,
            average_chunk_size=1000.0,
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Remove unsupported parameter "average_chunk_size" from SourceInfo construction

SourceInfo doesn't define an "average_chunk_size" field (it exposes avg_chunk_size as a computed property). Passing unknown kwargs is ignored by Pydantic but is misleading. Rely on chunk_count and total_content_length; assert via avg_chunk_size.

-            average_chunk_size=1000.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +200 to +201
        time_diff = abs((source.created_at - source.updated_at).total_seconds())
        assert time_diff < 0.01  # Should be within 10ms
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Relax timing tolerance to reduce flakiness on slower CI runners

Strict 10ms equality is brittle; factory defaults may differ by tens/hundreds of ms.

-        time_diff = abs((source.created_at - source.updated_at).total_seconds())
-        assert time_diff < 0.01  # Should be within 10ms
+        time_diff = abs((source.created_at - source.updated_at).total_seconds())
+        assert time_diff < 0.5  # Allow up to 500ms

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +323 to +324
            average_chunk_size=0.0,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" in zero-source case

Not a model field; the property will already compute 0.0 when chunk_count == 0.

-            average_chunk_size=0.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_sources_simple.py
Comment on lines +336 to +337
            average_chunk_size=5000.0,
        )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Same: drop "average_chunk_size" in large-source case

Keep only chunk_count and total_content_length.

-            average_chunk_size=5000.0,

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +129 to +132
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Use UpdateStatus enum for upsert status; strings won’t match code.

VectorService compares result.status to UpdateStatus.COMPLETED. Returning "completed" (str) will not match and will make the test fail spuriously.

-                mock_qdrant_client.upsert.return_value = MagicMock(
-                    operation_id=0, status="completed"
-                )
+                from qdrant_client.models import UpdateStatus
+                mock_qdrant_client.upsert.return_value = MagicMock(
+                    operation_id=0, status=UpdateStatus.COMPLETED
+                )

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +160 to +163
                mock_qdrant_client.upsert.return_value = MagicMock(
                    operation_id=0, status="completed"
                )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Same issue in batch test—return UpdateStatus.COMPLETED.

-                mock_qdrant_client.upsert.return_value = MagicMock(
-                    operation_id=0, status="completed"
-                )
+                from qdrant_client.models import UpdateStatus
+                mock_qdrant_client.upsert.return_value = MagicMock(
+                    operation_id=0, status=UpdateStatus.COMPLETED
+                )

🤖 Prompt for AI Agents

@jmagar
tests/test_vectors_comprehensive.py
Comment on lines +213 to +222
                query_vector = [0.1, 0.2, 0.3] * 128
                results = await service.search_similar(
                    query_vector, limit=10, score_threshold=0.7
                )

                assert len(results) == 1
                assert results[0].score == 0.95
                assert results[0].document.content == "Test content"
                mock_qdrant_client.query_points.assert_called_once()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Optionally assert dynamic search_params (ef) wiring.

To lock in the “typed SearchParams” change, assert that query_points received a SearchParams with expected hnsw_ef.

                 results = await service.search_similar(
                     query_vector, limit=10, score_threshold=0.7
                 )

                 assert len(results) == 1
                 assert results[0].score == 0.95
                 assert results[0].document.content == "Test content"
                 mock_qdrant_client.query_points.assert_called_once()
+                args, kwargs = mock_qdrant_client.query_points.call_args
+                assert "search_params" in kwargs
+                assert getattr(kwargs["search_params"], "hnsw_ef", None) == 40  # min(256, max(64, 10*4)) = 64 → wait, code caps at min 256 and max 64; for limit=10, ef=64
+                assert kwargs["with_vectors"] is False

Note: For limit=10, ef_value = max(64, 40) → 64.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Review continued from previous batch...
crawler_mcp/crawlers/web.py
Comment on lines 66 to 76
    def __init__(self) -> None:
        super().__init__()
        self.memory_manager: MemoryManager | None = None
        self.memory_manager = None

    async def _initialize_managers(self) -> None:
        """Initialize required managers."""
        if not self.memory_manager:
            from ..core.memory import get_memory_manager

            self.memory_manager = get_memory_manager()

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Minor: keep memory_manager typing consistent with BaseCrawlStrategy

You set self.memory_manager = None without an annotation. Consider keeping the same Optional[...] type annotation as the base to help static checkers.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Outdated
Comment on lines 499 to 537
        try:
            # Check if the private _markdown field contains an integer hash
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
            if hasattr(result, "markdown"):
                try:
                    # Try to access it to see if it would error
                    _ = result.markdown
                except AttributeError as e:
                    if "'int' object has no attribute" in str(e):
                        print(
                            f"CRAWL DEBUG - Markdown property access failed, force setting safe value for {result.url}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Force set a safe markdown value
                        result._markdown = MarkdownGenerationResult(
                            raw_markdown="",
                            markdown_with_citations="",
                            references_markdown="",
                            fit_markdown=None,
                            fit_html=None,
                        )
        except Exception as e:
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Avoid mutating third‑party private attribute _markdown; rely on safe access instead

Directly setting result._markdown is brittle and may break across crawl4ai versions. Since _safe_get_markdown already handles int/hash sentinel values and exceptions, prefer not to mutate the result object. If you must sanitize, store a local fallback rather than overwriting private fields.

Option A (preferred): Remove the mutation and let _safe_get_markdown handle all cases.

-                # Replace the integer hash with an empty MarkdownGenerationResult
-                result._markdown = MarkdownGenerationResult(
-                    raw_markdown="",
-                    markdown_with_citations="",
-                    references_markdown="",
-                    fit_markdown=None,
-                    fit_html=None,
-                )
+                # Defer to _safe_get_markdown which handles integer sentinel cases
+                pass
@@
-                        # Force set a safe markdown value
-                        result._markdown = MarkdownGenerationResult(
-                            raw_markdown="",
-                            markdown_with_citations="",
-                            references_markdown="",
-                            fit_markdown=None,
-                            fit_html=None,
-                        )
+                        # Defer to _safe_get_markdown to provide an empty-safe value
+                        pass

Option B: If you keep the mutation for legacy versions, gate it behind a feature flag (settings.crawl_force_markdown_sanitize) and log a one-time warning.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
crawler_mcp/crawlers/web.py
Outdated
Comment on lines 681 to 689
        content_filter = PruningContentFilter(
            threshold=0.45,  # Prune nodes below 45% relevance score
            threshold=0.48,  # Prune nodes below 48% relevance score for better quality
            threshold_type="dynamic",  # Dynamic scoring
            min_word_threshold=5,  # Ignore very short text blocks
            min_word_threshold=10,  # Higher threshold for quality content blocks
        )

        # Create markdown generator with content filter
        markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Be defensive around PruningContentFilter signature across versions

Some crawl4ai versions may not accept threshold_type="dynamic". Add a fallback.

-        content_filter = PruningContentFilter(
-            threshold=0.48,  # Prune nodes below 48% relevance score for better quality
-            threshold_type="dynamic",  # Dynamic scoring
-            min_word_threshold=10,  # Higher threshold for quality content blocks
-        )
+        try:
+            content_filter = PruningContentFilter(
+                threshold=0.48,
+                threshold_type="dynamic",
+                min_word_threshold=10,
+            )
+        except Exception:
+            content_filter = PruningContentFilter(
+                threshold=0.48,
+                min_word_threshold=10,
+            )

If you want me to confirm the current crawl4ai signature, I can run a quick check against the installed version.
📝 Committable suggestion


@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +728 to +736
        extraction_strategy = getattr(request, "extraction_strategy", None)
        if extraction_strategy == "llm":
            with contextlib.suppress(Exception):
                run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
                    provider="openai",
                    api_token="",
                    instruction="Extract main content and key information from the page",
                )
        elif extraction_strategy == "cosine":
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

LLM extraction: avoid empty API token; gate on configured credentials

Creating an LLMExtractionStrategy with api_token="" is likely to fail at runtime or pull from implicit env. Respect settings and warn if missing.

-        if extraction_strategy == "llm":
-            with contextlib.suppress(Exception):
-                run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
-                    provider="openai",
-                    api_token="",
-                    instruction="Extract main content and key information from the page",
-                )
+        if extraction_strategy == "llm":
+            token = getattr(settings, "openai_api_key", None)
+            if token:
+                with contextlib.suppress(Exception):
+                    run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
+                        provider="openai",
+                        api_token=token,
+                        instruction="Extract main content and key information from the page",
+                    )
+            else:
+                self.logger.warning("LLM extraction requested but no API token configured; skipping LLM strategy.")

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +18 to +29
# Fix NumPy 2.x + SciPy 1.16.1 compatibility issue for coverage measurement
# Force NumPy and SciPy imports before coverage instrumentation to prevent conflicts
try:
    import numpy as np
    import scipy.stats

    # Force initialization to happen before pytest-cov instruments the code
    np.__version__
    scipy.stats.__version__
except Exception:
    # If there are import issues, proceed without the imports
    pass
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Useless expressions for version access; remove or use a harmless reference

The np.__version__ and scipy.stats.__version__ lines are flagged as useless expressions (Ruff B018). If the goal is import priming, the import itself suffices.

-    # Force initialization to happen before pytest-cov instruments the code
-    np.__version__
-    scipy.stats.__version__
+    # Imports above are enough to prime coverage/instrumentation
+    pass

Alternatively, log versions if you want a side effect.
📝 Committable suggestion


🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/conftest.py
Comment on lines +123 to +129
            try:
                await vector_service._client.delete_collection(
                    test_settings.qdrant_collection
                )
            except Exception:
                pass  # Collection might not exist

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Use contextlib.suppress instead of try/except/pass

Cleaner and linter-friendly.

-            try:
-                await vector_service._client.delete_collection(
-                    test_settings.qdrant_collection
-                )
-            except Exception:
-                pass  # Collection might not exist
+            from contextlib import suppress
+            with suppress(Exception):
+                client = getattr(vector_service, "client", None) or getattr(vector_service, "_client", None)
+                if client is not None:
+                    await client.delete_collection(test_settings.qdrant_collection)

    Committable suggestion skipped: line range outside the PR's diff.

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +7 to +11
import pytest
from fastmcp import Client


class TestCrawlingTools:
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Mark async tests to avoid event-loop detection flakiness

These tests are async and rely on an async fixture. Add a file-level marker so they run under pytest-asyncio/anyio regardless of global config.

 from fastmcp import Client


+pytestmark = pytest.mark.anyio
+
 class TestCrawlingTools:

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +50 to +78
    @pytest.mark.integration
    @pytest.mark.requires_services
    async def test_scrape_with_auto_rag(self, mcp_client: Client):
        """Test scraping with automatic RAG indexing."""
        test_url = "https://httpbin.org/html"

        result = await mcp_client.call_tool(
            "scrape",
            {
                "url": test_url,
                "auto_index": True,  # Enable automatic RAG indexing
            },
        )

        assert result.data is not None
        scrape_data = result.data

        assert scrape_data["success"] is True

        # Should have indexing information
        assert "indexed" in scrape_data
        assert "source_id" in scrape_data

        # If indexing succeeded
        if scrape_data["indexed"]:
            assert scrape_data["source_id"] is not None
            assert "chunks_created" in scrape_data
            assert isinstance(scrape_data["chunks_created"], int)

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Assert minimal indexing contract in auto_index path

Good coverage. Consider asserting that source_id is a non-empty string (when indexed) to catch accidental None/empty IDs.

         if scrape_data["indexed"]:
             assert scrape_data["source_id"] is not None
             assert "chunks_created" in scrape_data
             assert isinstance(scrape_data["chunks_created"], int)
+            assert isinstance(scrape_data["source_id"], str) and scrape_data["source_id"].strip()

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +147 to +175
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_crawl_small_website(self, mcp_client: Client):
        """Test crawling a small website (marked as slow)."""
        # Use a simple, reliable test site
        test_url = "https://httpbin.org/"

        result = await mcp_client.call_tool(
            "crawl",
            {
                "target": test_url,
                "max_pages": 3,  # Limit to avoid long test times
                "max_depth": 1,
                "auto_index": False,  # Don't index for this test
            },
        )

        assert result.data is not None
        crawl_data = result.data

        # Should detect as website crawl
        assert crawl_data["crawl_type"] == "website"
        assert crawl_data["success"] is True

        # Should have crawled some pages
        results = crawl_data["results"]
        assert isinstance(results, list)
        assert len(results) >= 1  # At least the main page

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Network-bound integration test: add a defensive skip for offline CI

httpbin is usually reliable, but CI may run without egress. Consider skipping when network is disabled.

 @pytest.mark.slow
 @pytest.mark.integration
 async def test_crawl_small_website(self, mcp_client: Client):
@@
-        result = await mcp_client.call_tool(
+        result = await mcp_client.call_tool(
             "crawl",
             {
                 "target": test_url,
                 "max_pages": 3,  # Limit to avoid long test times
                 "max_depth": 1,
                 "auto_index": False,  # Don't index for this test
             },
         )

And add a small helper at top of file if you want an automatic skip:

+def _has_network() -> bool:
+    import socket
+    try:
+        socket.gethostbyname("httpbin.org")
+        return True
+    except Exception:
+        return False
+
@@
-    async def test_crawl_small_website(self, mcp_client: Client):
+    async def test_crawl_small_website(self, mcp_client: Client):
+        if not _has_network():
+            pytest.skip("No network available in CI")

🤖 Prompt for AI Agents

@jmagar
tests/test_crawling_tools.py
Comment on lines +179 to +186
        with pytest.raises(Exception):  # Should raise ToolError
            await mcp_client.call_tool(
                "scrape",
                {
                    "url": "not-a-valid-url",
                },
            )

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Use specific ToolError (and ValueError) in crawling tests

Catching the base Exception silences real errors and violates Ruff B017. In tests/test_crawling_tools.py, replace each with pytest.raises(Exception): in your crawling tests with an explicit tuple of the exceptions your code actually raises—namely ToolError (from fastmcp.exceptions) and ValueError for invalid parameters.

Key locations to update:

    tests/test_crawling_tools.py, test_invalid_url_handling (lines 179–186)
    tests/test_crawling_tools.py, test_nonexistent_directory (lines 190–196)
    The remaining invalid-parameter tests at lines 202–204, 206–214, 219–221, and 223–230

Example diff (apply analogous changes throughout):

--- a/tests/test_crawling_tools.py
+++ b/tests/test_crawling_tools.py
@@
 from fastmcp import Client
+from fastmcp.exceptions import ToolError

@@ -179,7 +180,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):  # Should raise ToolError
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "scrape",
                 {"url": "not-a-valid-url"},
             )
@@ -190,7 +191,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):  # Should raise ToolError
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "crawl",
                 {"target": "/nonexistent/directory/path"},
             )
@@ -202,7 +203,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool("crawl", {})

@@ -206,7 +207,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "crawl",
                 {"target": "https://httpbin.org", "max_pages": -1},
             )
@@ -219,7 +220,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool("scrape", {})

@@ -223,7 +224,7 @@ class TestCrawlingTools:
-        with pytest.raises(Exception):
+        with pytest.raises((ToolError, ValueError)):
             await mcp_client.call_tool(
                 "scrape",
                 {"url": "https://httpbin.org", "extraction_strategy": "invalid_strategy"},
             )

Be sure to add at the top alongside your other imports:

from fastmcp.exceptions import ToolError

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +29 to +33
    @pytest.fixture
    def middleware(self):
        """Create ErrorHandlingMiddleware instance."""
        return ErrorHandlingMiddleware()

@coderabbitai coderabbitai bot Aug 20, 2025

💡 Verification agent
🧩 Analysis chain

Tests for middleware constructors and methods need fixing

The ErrorHandlingMiddleware, LoggingMiddleware, and ProgressMiddleware classes all require an app: Callable[[Request], Awaitable[Response]] in their constructors and expose a __call__(request: Request) -> Response method. There is no .process(ctx, handler) helper. As written, the tests will fail with:

    TypeError on instantiation (missing required app argument)
    AttributeError when calling a nonexistent .process method

Please update the tests (in tests/test_middleware_comprehensive.py at lines 29–33, 135–138, and 210–214) to either:

    Pass a minimal dummy app into each middleware’s constructor and invoke the middleware via its await middleware(request) interface, or
    Add a .process(ctx, handler) shim in each middleware class that wraps __call__, if you intend to keep the existing test APIs

🤖 Prompt for AI Agents

@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +222 to +233
    def test_progress_tracker_creation(self):
        """Test ProgressTracker creation and properties."""
        tracker = ProgressTracker("test-op-123", "Test Operation")

        assert tracker.operation_id == "test-op-123"
        assert tracker.name == "Test Operation"
        assert tracker.current_step == 0
        assert tracker.total_steps == 0
        assert tracker.status == "pending"
        assert tracker.start_time is not None
        assert tracker.end_time is None

@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

🛠️ Refactor suggestion

ProgressTracker/ProgressMiddleware expectations diverge from implementation

The tests assume:

    ProgressTracker has a name, end_time, .update_progress(), .complete(), .error()
    Initial status == "pending"
    ProgressMiddleware exposes .active_operations (public), .list_operations(), and .cleanup_completed()

The provided implementation shows different attributes/method names (update(...), status starts as "starting", internal _active_operations, .list_active_operations()), so these tests will fail.

Either:

    Update tests to align with the existing API (prefer public helpers and avoid touching internals), or
    Adjust progress.py to add the missing convenience methods and public views used in tests.

Example test-side adjustments (if aligning to existing API):

- tracker = ProgressTracker("test-op-123", "Test Operation")
+ tracker = ProgressTracker("test-op-123")
- assert tracker.name == "Test Operation"
- assert tracker.status == "pending"
+ assert tracker.status in {"starting", "running", "completed", "error"}

- tracker.update_progress(5, 10, "Processing...")
+ tracker.update(5, 10, status="running", message="Processing...")

- tracker.complete("All done!")
+ tracker.update(tracker.total_steps, tracker.total_steps, status="completed", message="All done!")

- operations = middleware.list_operations()
+ operations = list(middleware.list_active_operations().values())

If you prefer the test-facing API, I can add a thin adapter layer to progress.py that preserves current behavior while exposing the test-friendly façade. Want me to draft it?

Also applies to: 235-250, 252-259, 261-274, 276-305, 309-345, 347-373, 387-407, 409-437
@jmagar
tests/test_middleware_comprehensive.py
Comment on lines +297 to +304
        """Test removing trackers."""
        # Create a tracker
        tracker = middleware.create_tracker("removable", "Removable Op")
        assert "removable" in middleware.active_operations

        # Remove it
        middleware.remove_tracker("removable")
        assert "removable" not in middleware.active_operations
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Unused local variable (tracker) — remove assignment

tracker isn’t used after creation.

-        tracker = middleware.create_tracker("removable", "Removable Op")
+        middleware.create_tracker("removable", "Removable Op")

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +26 to +39
        assert result.data is not None
        rag_data = result.data

        # Check basic structure
        assert "query" in rag_data
        assert "results" in rag_data
        assert "total_results" in rag_data
        assert "search_metadata" in rag_data

        assert rag_data["query"] == "test query with no results"
        assert rag_data["results"] == []
        assert rag_data["total_results"] == 0

    @pytest.mark.integration
@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Result shape assertions don’t match the server tool contract

rag_query returns {"query","total_matches","matches","performance","quality_metrics","timestamp"} per current tools, not results/total_results/search_metadata.

Apply this update:

-        # Check basic structure
-        assert "query" in rag_data
-        assert "results" in rag_data
-        assert "total_results" in rag_data
-        assert "search_metadata" in rag_data
+        # Check basic structure
+        assert "query" in rag_data
+        assert "matches" in rag_data
+        assert "total_matches" in rag_data
+        assert "performance" in rag_data
...
-        assert rag_data["results"] == []
-        assert rag_data["total_results"] == 0
+        assert rag_data["matches"] == []
+        assert rag_data["total_matches"] == 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +47 to +59
        # Step 1: Index some content
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        assert crawl_result.data is not None
        assert crawl_result.data["success"] is True

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Crawl tool parameters don’t exist; use process_with_rag instead and align follow-up query checks

crawl doesn’t accept auto_index or max_files. Use process_with_rag=True to index results.

-        crawl_result = await mcp_client.call_tool(
+        crawl_result = await mcp_client.call_tool(
             "crawl",
             {
                 "target": str(directory_path),
-                "auto_index": True,
-                "max_files": 3,
+                "process_with_rag": True,
             },
         )
...
-        # Should find some results now
-        assert rag_data["total_results"] > 0
-        assert len(rag_data["results"]) > 0
+        # Should find some results now
+        assert rag_data["total_matches"] > 0
+        assert len(rag_data["matches"]) > 0
...
-        for result in rag_data["results"]:
-            assert "content" in result
-            assert "metadata" in result
-            assert "score" in result
-            assert "source_id" in result
+        for match in rag_data["matches"]:
+            assert "score" in match
+            assert "relevance" in match
+            assert "document" in match
+            doc = match["document"]
+            assert "content" in doc
+            assert "metadata" in doc
+            assert "source_url" in doc

Also applies to: 65-93
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +102 to +117
        # Index content first
        await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 3,
            },
        )

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # Query with directory filter
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Filter params don’t match; use source_filters and updated keys

rag_query takes source_filters, not source_types, and results are under matches with per-match document.metadata.

-        result = await mcp_client.call_tool(
+        result = await mcp_client.call_tool(
             "rag_query",
             {
                 "query": "document",
-                "source_types": ["directory"],
+                "source_filters": ["directory"],
                 "limit": 10,
             },
         )
...
-        if rag_data["total_results"] > 0:
-            for result in rag_data["results"]:
-                metadata = result["metadata"]
+        if rag_data["total_matches"] > 0:
+            for match in rag_data["matches"]:
+                metadata = match["document"]["metadata"]
                 # Should be from directory source
                 assert metadata.get("source_type") == "directory"

Also applies to: 127-136
🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +139 to +161
    async def test_get_rag_stats(self, mcp_client: Client):
        """Test getting RAG system statistics."""
        result = await mcp_client.call_tool("get_rag_stats", {})

        assert result.data is not None
        stats_data = result.data

        # Check basic structure
        assert "total_sources" in stats_data
        assert "total_chunks" in stats_data
        assert "source_types" in stats_data
        assert "collection_info" in stats_data

        # Values should be non-negative integers
        assert isinstance(stats_data["total_sources"], int)
        assert isinstance(stats_data["total_chunks"], int)
        assert stats_data["total_sources"] >= 0
        assert stats_data["total_chunks"] >= 0

        # Source types should be a dict
        assert isinstance(stats_data["source_types"], dict)

    @pytest.mark.integration
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

get_rag_stats result shape mismatches

Current tool returns rag_system, source_management, health_summary, quick_facts, timestamp. Update assertions accordingly.

-        # Check basic structure
-        assert "total_sources" in stats_data
-        assert "total_chunks" in stats_data
-        assert "source_types" in stats_data
-        assert "collection_info" in stats_data
+        # Check basic structure
+        assert "rag_system" in stats_data
+        assert "source_management" in stats_data
+        assert "health_summary" in stats_data
+        assert "quick_facts" in stats_data
+        assert "timestamp" in stats_data

Then adapt value checks to quick_facts and rag_system fields (e.g., vector_dimension lives under quick_facts).
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +163 to +188
    async def test_list_sources(self, mcp_client: Client):
        """Test listing sources in the database."""
        result = await mcp_client.call_tool(
            "list_sources",
            {
                "limit": 10,
                "offset": 0,
            },
        )

        assert result.data is not None
        sources_data = result.data

        # Check structure
        assert "sources" in sources_data
        assert "total_count" in sources_data
        assert "limit" in sources_data
        assert "offset" in sources_data

        # Should be valid values
        assert isinstance(sources_data["sources"], list)
        assert isinstance(sources_data["total_count"], int)
        assert sources_data["total_count"] >= 0
        assert sources_data["limit"] == 10
        assert sources_data["offset"] == 0

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

list_sources result shape mismatches

list_sources returns sources, pagination, statistics, filters_applied.

-        # Check structure
-        assert "sources" in sources_data
-        assert "total_count" in sources_data
-        assert "limit" in sources_data
-        assert "offset" in sources_data
+        # Check structure
+        assert "sources" in sources_data
+        assert "pagination" in sources_data
+        assert "statistics" in sources_data
+        assert "filters_applied" in sources_data
...
-        assert isinstance(sources_data["total_count"], int)
-        assert sources_data["total_count"] >= 0
-        assert sources_data["limit"] == 10
-        assert sources_data["offset"] == 0
+        p = sources_data["pagination"]
+        assert isinstance(p["total"], int)
+        assert p["total"] >= 0
+        assert p["limit"] == 10
+        assert p["offset"] == 0

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +191 to +233
    async def test_delete_source(
        self, mcp_client: Client, sample_text_files: list[Path]
    ):
        """Test deleting a source from the database."""
        directory_path = sample_text_files[0].parent

        # Index content first to have something to delete
        crawl_result = await mcp_client.call_tool(
            "crawl",
            {
                "target": str(directory_path),
                "auto_index": True,
                "max_files": 1,
            },
        )

        assert crawl_result.data["success"] is True

        # Wait for indexing
        import asyncio

        await asyncio.sleep(1)

        # List sources to get a source_id
        list_result = await mcp_client.call_tool("list_sources", {"limit": 1})
        sources = list_result.data["sources"]

        if len(sources) > 0:
            source_id = sources[0]["source_id"]

            # Delete the source
            delete_result = await mcp_client.call_tool(
                "delete_source", {"source_id": source_id}
            )

            assert delete_result.data is not None
            delete_data = delete_result.data

            assert "success" in delete_data
            assert delete_data["success"] is True
            assert "source_id" in delete_data
            assert delete_data["source_id"] == source_id

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

delete_source tool param/result mismatches

The tool expects source_url and returns {"source_url","documents_deleted","success","timestamp"}.

-            source_id = sources[0]["source_id"]
+            source_url = sources[0]["url"]
...
-            delete_result = await mcp_client.call_tool(
-                "delete_source", {"source_id": source_id}
-            )
+            delete_result = await mcp_client.call_tool(
+                "delete_source", {"source_url": source_url, "confirm": True}
+            )
...
-            assert "source_id" in delete_data
-            assert delete_data["source_id"] == source_id
+            assert "source_url" in delete_data
+            assert delete_data["source_url"] == source_url

🤖 Prompt for AI Agents

@jmagar
tests/test_rag_tools.py
Comment on lines +235 to +260
    async def test_rag_query_parameter_validation(self, mcp_client: Client):
        """Test RAG query parameter validation."""
        # Test missing query
        with pytest.raises(Exception):
            await mcp_client.call_tool("rag_query", {})

        # Test invalid limit
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "limit": 0,
                },
            )

        # Test negative offset
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "rag_query",
                {
                    "query": "test",
                    "offset": -1,
                },
            )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Parameter validation checks: assert specific ToolError and match current validation rules

    rag_query doesn’t take offset; use min_score bounds and high limit instead.
    Avoid catching broad Exception (Ruff B017).

-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool("rag_query", {})
...
-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool(
                 "rag_query",
                 {
                     "query": "test",
-                    "limit": 0,
+                    "limit": 1000,  # Exceeds max 100
                 },
             )
...
-        with pytest.raises(Exception):
+        with pytest.raises(ToolError):
             await mcp_client.call_tool(
                 "rag_query",
                 {
-                    "query": "test",
-                    "offset": -1,
+                    "query": "test",
+                    "min_score": 2.0,  # Out of range
                 },
             )

🧰 Tools

🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
feat: implement parallel pipeline architecture for extreme embedding …
c17894d
coderabbitai[bot]
coderabbitai bot reviewed Aug 20, 2025
coderabbitai bot left a comment

Actionable comments posted: 9

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (4)

♻️ Duplicate comments (3)

📜 Review details



crawler_mcp/core/embeddings.py
Comment on lines 237 to 239
            logger.info(
                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch)"
                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {len(results) / processing_time:.1f} embeddings/sec"
            )
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

🛠️ Refactor suggestion

Avoid divide-by-zero in throughput log; consider a monotonic clock.

If the request returns “instantaneously” (cached/local) processing_time can be 0, causing a ZeroDivisionError in the f-string. Also, prefer time.perf_counter() for timing.

Apply this diff to harden the log:

-            logger.info(
-                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {len(results) / processing_time:.1f} embeddings/sec"
-            )
+            logger.info(
+                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {(len(results) / (processing_time or 1e-6)):.1f} embeddings/sec"
+            )

Optional: switch this to debug if it’s noisy in prod.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/embeddings.py
Comment on lines +289 to +291
        # Start timing the entire batch process
        batch_start_time = time.time()

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Use a monotonic timer for batch measurements.

time.perf_counter() is the recommended monotonic high‑resolution clock for durations.

-        # Start timing the entire batch process
-        batch_start_time = time.time()
+        # Start timing the entire batch process
+        batch_start_time = time.perf_counter()

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/embeddings.py
Comment on lines +339 to +345
        # Log total batch processing time
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time
        logger.info(
            f"Completed embedding generation for {len(valid_texts)} texts in {total_batch_time:.2f}s - {len(valid_texts) / total_batch_time:.1f} embeddings/sec"
        )

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Make batch throughput computation robust and reflect actual outputs.

    Guard against zero elapsed time.
    Use len(results) (produced items) rather than len(valid_texts) to avoid mismatch if something fails mid-batch.

-        # Log total batch processing time
-        batch_end_time = time.time()
-        total_batch_time = batch_end_time - batch_start_time
-        logger.info(
-            f"Completed embedding generation for {len(valid_texts)} texts in {total_batch_time:.2f}s - {len(valid_texts) / total_batch_time:.1f} embeddings/sec"
-        )
+        # Log total batch processing time
+        batch_end_time = time.perf_counter()
+        total_batch_time = batch_end_time - batch_start_time
+        elapsed = total_batch_time or 1e-6
+        logger.info(
+            f"Completed embedding generation for {len(results)} texts in {total_batch_time:.2f}s - {(len(results) / elapsed):.1f} embeddings/sec"
+        )

If you prefer to report “attempted texts per second,” keep len(valid_texts) but rename the metric accordingly to avoid confusion.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +803 to +809
                    # Generate deterministic ID if deduplication is enabled
                    if deduplication:
                        chunk_id = self._generate_deterministic_id(
                            page.url, composite_chunk_index
                        )
                        if legacy_chunk:
                            legacy_chunk_hash = legacy_chunk.get("content_hash", "")
                            if legacy_chunk_hash == content_hash:
                                # Same content, different ID format - skip but note the legacy chunk for replacement
                                chunks_skipped += 1
                                legacy_chunk_to_replace = legacy_chunk
                                legacy_chunks_to_delete.append(legacy_chunk["id"])
                                logger.debug(
                                    f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
                                )
                                should_skip = True
                        content_hash = self._calculate_content_hash(chunk_data["text"])
                    else:
@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Type mismatch: deterministic ID generator expects int but receives str

composite_chunk_index is a string (e.g., "pageIdx_subIdx") but _generate_deterministic_id is typed to accept int for chunk_index. Update the signature to accept str | int and cast to str for hashing.

Apply:

-def _generate_deterministic_id(self, url: str, chunk_index: int) -> str:
+def _generate_deterministic_id(self, url: str, chunk_index: str | int) -> str:
@@
-        normalized_url = self._normalize_url(url)
-        id_string = f"{normalized_url}:{chunk_index}"
+        normalized_url = self._normalize_url(url)
+        id_string = f"{normalized_url}:{str(chunk_index)}"

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines +830 to +846
                        # Backwards compatibility: check for legacy chunks with same content
                        elif use_backwards_compatibility and existing_chunks_list:
                            legacy_chunk = self._find_legacy_chunk_by_content(
                                existing_chunks_list, chunk_data["text"]
                            )
                            if legacy_chunk:
                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
                                if legacy_chunk_hash == content_hash:
                                    # Same content, different ID format - skip but note the legacy chunk for replacement
                                    chunks_skipped += 1
                                    legacy_chunk_to_replace = legacy_chunk
                                    legacy_chunks_to_delete.append(legacy_chunk["id"])
                                    logger.debug(
                                        f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
                                    )
                                    should_skip = True

@coderabbitai coderabbitai bot Aug 20, 2025

⚠️ Potential issue

Bug: “skip unchanged content” deletes legacy chunks without creating replacements

When a legacy chunk with identical content is found, you append the legacy ID to legacy_chunks_to_delete and set should_skip = True, then continue. This deletes the legacy chunk but never creates/upserts a new chunk with the deterministic ID, causing data loss.

Fix by treating this as an upgrade (update), not a skip. Don’t delete immediately and don’t set should_skip. Let the later “update/upgrade” branch handle deletion after successfully creating the deterministic chunk.

-                            if legacy_chunk:
-                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
-                                if legacy_chunk_hash == content_hash:
-                                    # Same content, different ID format - skip but note the legacy chunk for replacement
-                                    chunks_skipped += 1
-                                    legacy_chunk_to_replace = legacy_chunk
-                                    legacy_chunks_to_delete.append(legacy_chunk["id"])
-                                    logger.debug(
-                                        f"Skipping unchanged content, will replace legacy chunk {legacy_chunk['id']} with deterministic ID {chunk_id}"
-                                    )
-                                    should_skip = True
+                            if legacy_chunk:
+                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
+                                if legacy_chunk_hash == content_hash:
+                                    # Same content, different ID format - perform upgrade (create new deterministic chunk)
+                                    legacy_chunk_to_replace = legacy_chunk
+                                    chunks_updated += 1
+                                    logger.debug(
+                                        f"Upgrading legacy chunk {legacy_chunk['id']} to deterministic ID {chunk_id} (content unchanged)"
+                                    )
+                                    # Do not skip; proceed to create doc_chunk and mark legacy for deletion later

@jmagar
crawler_mcp/core/rag.py
Comment on lines +1467 to +1476
                    async with storage_semaphore:
                        batch_id, chunk_batch = await storage_queue.get()

                        # Filter chunks with valid embeddings
                        valid_chunks = [
                            chunk
                            for chunk in chunk_batch
                            if chunk.embedding is not None
                        ]

@coderabbitai coderabbitai bot Aug 20, 2025

🛠️ Refactor suggestion

Semaphore held during awaited queue.get() reduces concurrency

You acquire storage_semaphore before awaiting storage_queue.get(). This holds a permit while idling, throttling concurrency. Get the batch from the queue first, then acquire the semaphore only around the upsert call.

Apply:

-                try:
-                    async with storage_semaphore:
-                        batch_id, chunk_batch = await storage_queue.get()
+                try:
+                    batch_id, chunk_batch = await storage_queue.get()
+                    async with storage_semaphore:
                         # Filter chunks with valid embeddings

@jmagar
crawler_mcp/core/rag.py
Outdated
docker-compose.yml
Outdated
Comment on lines 39 to 47
      - "131072"
      - --max-batch-requests
      - "512"
      - "128"
      - --max-client-batch-size
      - "512"
      - "128"
      - --pooling
      - "last-token"
      - --tokenization-workers
      - "12"
      - "8"
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Parameterize aggressive TEI batching knobs; leave room to tune per environment.

The current values (e.g., max-batch-tokens=131072, max-*-size=128) can amplify peak memory and tail latencies under load. Make them env-driven with sane defaults so CI/dev/prod can tune safely.

-      - "131072"
+      - "${TEI_MAX_BATCH_TOKENS:-131072}"
@@
-      - "128"
+      - "${TEI_MAX_BATCH_REQUESTS:-128}"
@@
-      - "128"
+      - "${TEI_MAX_CLIENT_BATCH_SIZE:-128}"
@@
-      - "8"
+      - "${TEI_TOKENIZATION_WORKERS:-8}"

Action to verify: confirm these defaults fit your GPU memory and target tail latency under expected sequence lengths and traffic.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
docker-compose.yml
Comment on lines +52 to +53
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
      - OMP_NUM_THREADS=8
@coderabbitai coderabbitai bot Aug 20, 2025

🧹 Nitpick (assertive)

Make CUDA/OMP knobs configurable to avoid CPU/GPU over/under-subscription.

Hard-coding PYTORCH_CUDA_ALLOC_CONF and OMP_NUM_THREADS may be suboptimal across machines. Parameterize with env defaults.

-      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
-      - OMP_NUM_THREADS=8
+      - PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:1024}
+      - OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

Consider validating that OMP_NUM_THREADS aligns with available vCPUs and that allocator settings don’t increase fragmentation for your model’s peak batch.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
feat: add browser pool management and optimize crawler performance
9543105
coderabbitai[bot]
coderabbitai bot reviewed Aug 21, 2025
coderabbitai bot left a comment

Actionable comments posted: 27

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (8)

♻️ Duplicate comments (14)

📜 Review details

🔇 Additional comments (4)

.env.example
@@ -34,6 +34,7 @@ EMBEDDING_MAX_LENGTH=32000
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_RETRIES=2
EMBEDDING_WORKERS=4
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Good addition; document bounds and interplay with TEI/client concurrency.

EMBEDDING_WORKERS=4 mirrors the new settings field (ge=1, le=16). Add a short comment in this file noting the valid range [1–16] and that raising this above the TEI server’s concurrency and/or CPU cores can increase tail latencies.

Apply this minimal diff to self-document the knob:

 EMBEDDING_MAX_RETRIES=2
-EMBEDDING_WORKERS=4
+EMBEDDING_WORKERS=4  # [1-16] Parallel embedding workers; keep <= CPU cores and TEI concurrency

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
Comment on lines +45 to +53
    qdrant_connection_pool_size: int = Field(
        default=16, alias="QDRANT_CONNECTION_POOL_SIZE", ge=1, le=32
    )
    qdrant_batch_size: int = Field(
        default=256, alias="QDRANT_BATCH_SIZE", ge=64, le=512
    )
    qdrant_prefetch_size: int = Field(
        default=1024, alias="QDRANT_PREFETCH_SIZE", ge=256, le=2048
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Qdrant client tunables: validate combinations and document memory impact.

Pool size, batch size, and prefetch size interplay directly with client memory and open connections. Add a validator that warns when qdrant_batch_size * qdrant_prefetch_size is unusually high (e.g., > 512k vectors per page) to avoid unexpected spikes.

Example:

@@
     qdrant_prefetch_size: int = Field(
         default=1024, alias="QDRANT_PREFETCH_SIZE", ge=256, le=2048
     )
+
+    @model_validator(mode="after")
+    def _validate_qdrant_tunables(self) -> "CrawlerrSettings":
+        max_inflight = self.qdrant_batch_size * self.qdrant_prefetch_size
+        if max_inflight > 512 * 1024:
+            import logging
+            logging.getLogger(__name__).warning(
+                "Qdrant inflight capacity (%d) is very high; expect elevated memory/network usage",
+                max_inflight,
+            )
+        return self

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
@@ -58,6 +67,7 @@ class CrawlerrSettings(BaseSettings):
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(default=2, alias="EMBEDDING_MAX_RETRIES")
    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Cap embedding_workers by available CPU cores; prevent oversubscription.

At startup, clamp to os.cpu_count() and optionally to TEI_MAX_CONCURRENT_REQUESTS to avoid excessive context-switching and backpressure.

@@
-    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
+    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)
+
+    @model_validator(mode="after")
+    def _validate_embedding_workers(self) -> "CrawlerrSettings":
+        import os, logging
+        cores = max(1, os.cpu_count() or 1)
+        if self.embedding_workers > cores:
+            logging.getLogger(__name__).warning(
+                "EMBEDDING_WORKERS=%d exceeds CPU cores=%d; clamping.",
+                self.embedding_workers, cores,
+            )
+            self.embedding_workers = cores
+        return self

@jmagar
crawler_mcp/config.py
Comment on lines +134 to +137
    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
    crawl_virtual_scroll_batch_size: int = Field(
        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE"
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Scroll tuning defaults: ensure consumers actually use these fields.

These are newly exposed but I don’t see validations or bounds. Consider ge/le constraints (e.g., delay >= 0, batch_size >= 1) to prevent negative values via env overrides.

-    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
-    crawl_virtual_scroll_batch_size: int = Field(
-        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE"
-    )
+    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY", ge=0, le=1000)
+    crawl_virtual_scroll_batch_size: int = Field(
+        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE", ge=1, le=100
+    )

@jmagar
crawler_mcp/config.py
Comment on lines +144 to 156
    # High-Performance Configuration (i7-13700k + RTX 4070)
    browser_pool_size: int = Field(default=8, alias="BROWSER_POOL_SIZE", ge=1, le=16)
    file_processing_threads: int = Field(
        default=16, alias="FILE_PROCESSING_THREADS", ge=1, le=24
    )
    crawl_concurrency: int = Field(default=12, alias="CRAWL_CONCURRENCY", ge=1, le=50)
    content_cache_size_gb: int = Field(
        default=8, alias="CONTENT_CACHE_SIZE_GB", ge=1, le=16
    )
    gpu_memory_fraction: float = Field(
        default=0.95, alias="GPU_MEMORY_FRACTION", ge=0.1, le=1.0
    )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Hardware-specific defaults: avoid locking config to a single SKU.

The “i7-13700k + RTX 4070” comment plus high defaults (e.g., 16 file threads) can be aggressive on smaller machines. Either (a) lower defaults to safe values and document scaling, or (b) auto-scale by cores and memory.

Minimal auto-scaling:

-    browser_pool_size: int = Field(default=8, alias="BROWSER_POOL_SIZE", ge=1, le=16)
-    file_processing_threads: int = Field(
-        default=16, alias="FILE_PROCESSING_THREADS", ge=1, le=24
-    )
+    browser_pool_size: int = Field(default= max(1, (os.cpu_count() or 8)//2 ),
+                                   alias="BROWSER_POOL_SIZE", ge=1, le=16)
+    file_processing_threads: int = Field(default= max(4, (os.cpu_count() or 8) ),
+                                         alias="FILE_PROCESSING_THREADS", ge=1, le=24)

Note: import os at module top if adopting this.

    Committable suggestion skipped: line range outside the PR's diff.

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/config.py
Comment on lines +171 to +183
    # Alternative crawling approach settings
    use_arun_many_for_sitemaps: bool = Field(
        default=False,
        alias="USE_ARUN_MANY_FOR_SITEMAPS",
        description="Use arun_many() with sitemap URLs instead of BFSDeepCrawlStrategy",
    )
    max_concurrent_sessions: int = Field(
        default=20,
        alias="CRAWL_MAX_CONCURRENT_SESSIONS",
        ge=1,
        le=50,
        description="Maximum concurrent sessions for arun_many() approach",
    )
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Mutual exclusivity and defaults for sitemap crawling path.

When USE_ARUN_MANY_FOR_SITEMAPS=True, clarify precedence vs the default BFS strategy. Consider a validator to warn if max_concurrent_sessions is set but the feature is disabled, or if both strategies are toggled on by mistake upstream.
@jmagar
crawler_mcp/config.py
Comment on lines +185 to +201
    # Crawl4AI Performance Optimizations
    crawl_text_mode: bool = Field(
        default=False,
        alias="CRAWL_TEXT_MODE",
        description="Enable text-only mode for 3-4x faster crawling (disables images)",
    )
    crawl_light_mode: bool = Field(
        default=True,
        alias="CRAWL_LIGHT_MODE",
        description="Enable light mode to optimize browser performance",
    )
    use_lxml_strategy: bool = Field(
        default=True,
        alias="USE_LXML_STRATEGY",
        description="Use LXMLWebScrapingStrategy for 20x faster parsing",
    )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Light/text/LXML feature toggles: add a compatibility validator.

Enabling crawl_text_mode=True with crawl_extract_media=True would be contradictory. Add a post-model validator to warn or auto-disable the conflicting option. Similarly, LXML requires specific parsers installed—validate availability at startup and downgrade gracefully.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +24 to +33
    async def initialize(self) -> None:
        """Initialize browser pool with hardware-optimized config."""
        if self.is_initialized:
            return

        try:
            # Import here to avoid circular imports
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            # Hardware-optimized browser configuration
@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Guard initialization with a lock; fix race when two coroutines call acquire() concurrently.

Without synchronization, two tasks can run initialize() simultaneously and create duplicate browser sets.

Apply:

 class HighPerformanceBrowserPool:
@@
     def __init__(self, pool_size: int = 8):
         self.pool_size = pool_size
         self.browsers: list[Any] = []  # Will be AsyncWebCrawler instances
         self.available_browsers: asyncio.Queue[Any] = asyncio.Queue(maxsize=pool_size)
         self.is_initialized = False
+        self._init_lock = asyncio.Lock()
         self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
@@
     async def initialize(self) -> None:
         """Initialize browser pool with hardware-optimized config."""
-        if self.is_initialized:
-            return
+        if self.is_initialized:
+            return
+        async with self._init_lock:
+            if self.is_initialized:
+                return

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +31 to +33
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            # Hardware-optimized browser configuration
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Safer imports and clearer error if crawl4ai is missing.

Importing from crawl4ai inside the method is good to avoid cycles, but improve diagnostics on ImportError and prefer local Protocols for type hints under TYPE_CHECKING.

-            from crawl4ai import AsyncWebCrawler, BrowserConfig
+            try:
+                from crawl4ai import AsyncWebCrawler, BrowserConfig
+            except Exception as e:
+                self.logger.error("crawl4ai is not installed or incompatible: %s", e, exc_info=True)
+                raise

For better typing without runtime dependency:

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..types.crawl4ai_types import AsyncWebCrawler as _AsyncWebCrawler, BrowserConfig as _BrowserConfig

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +34 to +63
            browser_config = BrowserConfig(
                headless=True,
                browser_type="chromium",
                verbose=False,
                # RTX 4070 + i7-13700k optimized Chrome flags
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu-sandbox",
                    "--max_old_space_size=4096",  # 4GB per browser instance
                    "--js-flags=--max-old-space-size=4096",
                    "--renderer-process-limit=4",  # Limit renderer processes
                    "--process-per-site",
                    "--aggressive-cache-discard",
                    "--memory-pressure-off",
                    "--enable-gpu-rasterization",
                    "--enable-zero-copy",
                    "--enable-oop-rasterization",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--no-zygote",  # Better for concurrent instances
                    # Performance optimizations for high-end hardware
                    "--max-renderer-processes=4",
                    "--renderer-process-limit=4",
                    "--enable-accelerated-2d-canvas",
                    "--enable-gpu-compositing",
                ],
            )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Chrome flags are overly aggressive/hardware-specific; make them settings-driven and drop risky ones.

Flags like --no-sandbox and duplicate renderer limits can hurt stability; memory V8 flags may be ineffective. Honor settings (light/text mode) and keep a minimal, portable set.

-            browser_config = BrowserConfig(
+            # TODO: derive from settings (light_mode, block images/media) if available
+            browser_config = BrowserConfig(
                 headless=True,
                 browser_type="chromium",
                 verbose=False,
                 # RTX 4070 + i7-13700k optimized Chrome flags
                 extra_args=[
-                    "--no-sandbox",
-                    "--disable-dev-shm-usage",
-                    "--disable-gpu-sandbox",
-                    "--max_old_space_size=4096",  # 4GB per browser instance
-                    "--js-flags=--max-old-space-size=4096",
-                    "--renderer-process-limit=4",  # Limit renderer processes
-                    "--process-per-site",
-                    "--aggressive-cache-discard",
-                    "--memory-pressure-off",
-                    "--enable-gpu-rasterization",
-                    "--enable-zero-copy",
-                    "--enable-oop-rasterization",
-                    "--disable-background-timer-throttling",
-                    "--disable-backgrounding-occluded-windows",
-                    "--disable-renderer-backgrounding",
-                    "--disable-features=TranslateUI",
-                    "--no-zygote",  # Better for concurrent instances
-                    # Performance optimizations for high-end hardware
-                    "--max-renderer-processes=4",
-                    "--renderer-process-limit=4",
-                    "--enable-accelerated-2d-canvas",
-                    "--enable-gpu-compositing",
+                    "--disable-dev-shm-usage",
+                    "--disable-background-timer-throttling",
+                    "--disable-renderer-backgrounding",
+                    "--disable-backgrounding-occluded-windows",
+                    "--enable-zero-copy",
                 ],
             )

If sandbox must be disabled for containers, gate --no-sandbox behind an env/settings flag and document the risk.
📝 Committable suggestion


🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +69 to +81
            # Create browser instances
            for i in range(self.pool_size):
                try:
                    browser = AsyncWebCrawler(config=browser_config)
                    await browser.__aenter__()
                    self.browsers.append(browser)
                    await self.available_browsers.put(browser)
                    self.logger.debug(f"Initialized browser {i + 1}/{self.pool_size}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize browser {i + 1}: {e}")
                    # Continue with fewer browsers rather than failing completely

            if not self.browsers:
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Initialize browsers with partial failure handling and metrics; skip queue put if aenter fails.

You already catch and continue; add exc_info=True and count successes/failures.

-                except Exception as e:
-                    self.logger.error(f"Failed to initialize browser {i + 1}: {e}")
+                except Exception as e:
+                    self.logger.error("Failed to initialize browser %d: %s", i + 1, e, exc_info=True)

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +94 to +107
    async def acquire(self) -> Any:
        """Get browser from pool."""
        if not self.is_initialized:
            await self.initialize()

        browser = await self.available_browsers.get()
        self.logger.debug("Browser acquired from pool")
        return browser

    async def release(self, browser: Any) -> None:
        """Return browser to pool."""
        try:
            await self.available_browsers.put(browser)
            self.logger.debug("Browser returned to pool")
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Ergonomics: provide an async lease contextmanager to ensure release on exceptions.

Avoids borrowers forgetting to call release().

Add:

@@
     async def acquire(self) -> Any:
@@
         return browser
@@
     async def release(self, browser: Any) -> None:
@@
         except Exception as e:
             self.logger.error(f"Failed to return browser to pool: {e}")
+
+    async def lease(self):
+        """Async context manager yielding a browser and auto-releasing it."""
+        browser = await self.acquire()
+        try:
+            yield browser
+        finally:
+            await self.release(browser)

Usage:

from contextlib import asynccontextmanager
HighPerformanceBrowserPool.lease = asynccontextmanager(HighPerformanceBrowserPool.lease)

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +111 to +133
    async def cleanup(self) -> None:
        """Cleanup all browsers."""
        self.logger.info("Cleaning up browser pool")

        # Close all browsers
        for browser in self.browsers:
            try:
                await browser.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}")

        self.browsers.clear()

        # Clear the queue
        while not self.available_browsers.empty():
            try:
                self.available_browsers.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.is_initialized = False
        self.logger.info("Browser pool cleanup completed")

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Close browsers concurrently and drain queue robustly.

Speed up shutdown and avoid leaving stale references in the queue.

-        for browser in self.browsers:
-            try:
-                await browser.__aexit__(None, None, None)
-            except Exception as e:
-                self.logger.error(f"Error closing browser: {e}")
+        await asyncio.gather(
+            *[
+                browser.__aexit__(None, None, None)
+                for browser in list(self.browsers)
+            ],
+            return_exceptions=True,
+        )
+        # Log any close errors
+        # (gather already returned exceptions; consider iterating for logging if needed)
@@
-        while not self.available_browsers.empty():
-            try:
-                self.available_browsers.get_nowait()
-            except asyncio.QueueEmpty:
-                break
+        try:
+            while True:
+                self.available_browsers.get_nowait()
+        except asyncio.QueueEmpty:
+            pass

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/browser_pool.py
Comment on lines +158 to +165
async def get_browser_pool(pool_size: int = 8) -> HighPerformanceBrowserPool:
    """Get or create the global browser pool."""
    global _browser_pool

    if _browser_pool is None:
        _browser_pool = HighPerformanceBrowserPool(pool_size=pool_size)

    return _browser_pool
@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Pool size argument is ignored after first creation; warn on mismatch.

If a caller requests a different size later, log it to avoid confusion.

     if _browser_pool is None:
         _browser_pool = HighPerformanceBrowserPool(pool_size=pool_size)
+    elif _browser_pool.pool_size != pool_size:
+        logger.warning(
+            "Browser pool already initialized with size=%d (requested %d); using existing pool.",
+            _browser_pool.pool_size, pool_size
+        )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/core/rag.py
Comment on lines 1016 to +1023
            stored_count = await self.vector_service.upsert_documents(document_chunks)
            logger.info(f"Stored {stored_count} document chunks in vector database")

            # Log vector storage time
            storage_end_time = time.time()
            storage_duration = storage_end_time - storage_start_time
            logger.info(
                f"Stored {stored_count} document chunks in vector database in {storage_duration:.2f}s"
            )
@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Duplicate storage operation after pipeline already stored chunks

After the parallel pipeline completes (which already stores chunks via upsert_documents in the storage worker), this code calls upsert_documents again on the same chunks, causing duplicate storage operations.

Remove the duplicate storage operation since the pipeline already handles it:

-        # Store in vector database
-        if progress_callback:
-            progress_callback(
-                total_pages + 1,
-                total_pages + 2,
-                f"Storing {len(document_chunks)} embeddings in vector database",
-            )
-
-        try:
-            # Start timing vector storage
-            storage_start_time = time.time()
-
-            stored_count = await self.vector_service.upsert_documents(document_chunks)
-
-            # Log vector storage time
-            storage_end_time = time.time()
-            storage_duration = storage_end_time - storage_start_time
-            logger.info(
-                f"Stored {stored_count} document chunks in vector database in {storage_duration:.2f}s"
-            )
-
-        except Exception as e:
-            logger.error(f"Error storing embeddings: {e}")
-            raise ToolError(f"Failed to store embeddings: {e!s}") from e
+        # Storage is handled by the pipeline's storage worker
+        stored_count = len(document_chunks)

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +119 to 124
            # Use ThreadPoolExecutor for maximum CPU utilization
            if progress_callback:
                progress_callback(
                    0, len(files), "Starting high-performance file processing..."
                )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Progress reporting: consider emitting progress as files complete.

You set an initial message but never update the numerator. If the callback supports it, increment on each processed file to reflect true progress.
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +125 to +135
            batch_results = await self._process_files_highly_concurrent(
                files, directory
            )

                # Small delay to prevent overwhelming the system
                if i + batch_size < len(files):
                    await asyncio.sleep(0.01)
            for batch_result in batch_results:
                if isinstance(batch_result, PageContent):
                    pages.append(batch_result)
                    total_bytes += len(batch_result.content)
                elif isinstance(batch_result, Exception):
                    errors.append(str(batch_result))

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Count bytes, not characters, for total_bytes.

len(string) counts code points; use UTF-8 byte length for network/storage stats.

-                    total_bytes += len(batch_result.content)
+                    total_bytes += len(batch_result.content.encode("utf-8", errors="ignore"))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +136 to 144
            # Memory pressure check after processing
            if (
                self.memory_manager
                and await self.memory_manager.check_memory_pressure()
            ):
                self.logger.warning(
                    "Memory pressure detected during directory processing"
                )

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Memory pressure check is post hoc only; consider occasional mid-flight checks for very large directories.

For very large file sets, a periodic check (e.g., every N files) can prevent overshooting memory thresholds. Optional, depending on expected workloads.
@jmagar
crawler_mcp/crawlers/directory.py
Comment on lines +410 to 449
    async def _process_files_highly_concurrent(
        self, file_paths: list[Path], base_directory: Path
    ) -> list[PageContent | Exception]:
        """Process a batch of files concurrently."""
        semaphore = asyncio.Semaphore(5)  # Limit concurrent file operations
        """Process files with full CPU utilization using ThreadPoolExecutor."""
        from ..config import settings

        async def process_single_file(file_path: Path) -> PageContent | Exception:
            async with semaphore:
                try:
                    return await self._process_single_file(file_path, base_directory)
                except Exception as e:
                    return Exception(f"Error processing {file_path}: {e}")
        # Use configured thread count (default 16 for i7-13700k)
        max_workers = getattr(settings, "file_processing_threads", 16)

        self.logger.info(
            f"Processing {len(file_paths)} files with {max_workers} threads"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()

            # Process files in parallel using all available threads
            tasks = [
                loop.run_in_executor(
                    executor, self._process_single_file_sync, file_path, base_directory
                )
                for file_path in file_paths
            ]

            # Use asyncio.gather for true parallelism
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter and return results
            processed_results = []
            for result in results:
                if isinstance(result, PageContent | Exception):
                    processed_results.append(result)
                else:
                    # Handle any unexpected return types
                    processed_results.append(
                        Exception(f"Unexpected result type: {type(result)}")
                    )

            return processed_results

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Tests: add coverage for error returns and mixed results from the thread pool.

Ensure that Exceptions returned by _process_single_file_sync are collected into errors and that stats are computed correctly when some files fail.

Would you like me to add unit tests targeting:

    isinstance fix behavior (no TypeError)
    byte-accurate total_bytes
    mixed success/failure aggregation?

🤖 Prompt for AI Agents

⚠️ Potential issue

Bug: isinstance with PEP 604 union is invalid; use a tuple. Also use get_running_loop().

isinstance(result, PageContent | Exception) raises TypeError at runtime. And inside an async context, prefer asyncio.get_running_loop().

Apply:

-        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
-            loop = asyncio.get_event_loop()
+        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
+            loop = asyncio.get_running_loop()
@@
-            processed_results = []
-            for result in results:
-                if isinstance(result, PageContent | Exception):
-                    processed_results.append(result)
-                else:
-                    # Handle any unexpected return types
-                    processed_results.append(
-                        Exception(f"Unexpected result type: {type(result)}")
-                    )
+            processed_results: list[PageContent | Exception] = []
+            for result in results:
+                if isinstance(result, (PageContent, Exception)):
+                    processed_results.append(result)
+                else:
+                    processed_results.append(Exception(f"Unexpected result type: {type(result)}"))

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +133 to 168
            # High-performance browser config optimized for i7-13700k + RTX 4070
            browser_config = BrowserConfig(
                headless=settings.crawl_headless,
                browser_type=settings.crawl_browser,
                light_mode=True,  # Let Crawl4AI optimize performance
                text_mode=getattr(settings, "crawl_block_images", False),
                light_mode=getattr(
                    settings, "crawl_light_mode", True
                ),  # Optimized performance mode
                text_mode=getattr(
                    settings, "crawl_text_mode", False
                ),  # 3-4x faster when enabled
                verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
                # NO extra_args - avoid flag conflicts
                # Aggressive performance settings
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    # GPU Acceleration for RTX 4070
                    "--enable-gpu",  # Enable GPU acceleration
                    "--enable-accelerated-2d-canvas",  # GPU for 2D canvas
                    "--enable-gpu-compositing",  # GPU for compositing
                    "--enable-gpu-rasterization",  # GPU for rasterization
                    "--ignore-gpu-blocklist",  # Use GPU even if blocklisted
                    "--disable-gpu-sandbox",  # Remove GPU sandbox restrictions
                    "--enable-zero-copy",  # Zero-copy GPU textures
                    "--use-gl=egl",  # Use EGL for headless GPU
                    "--max_old_space_size=4096",  # 4GB memory per browser
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--aggressive-cache-discard",
                    "--memory-pressure-off",
                    # Network optimizations for faster page loads
                    "--max-connections-per-host=30",  # Match semaphore_count
                    "--enable-quic",
                    "--enable-tcp-fast-open",
                ],
            )
@coderabbitai coderabbitai bot Aug 21, 2025

💡 Verification agent
❓ Verification inconclusive

Harden BrowserConfig GPU flags and remove non-Chromium switch

We verified that headless Chromium always disables GPU by default—forcing real-GPU flags on a GPU-less CI or container will cause the GPU process to crash (e.g. “Exiting GPU process due to errors during initialization”) (github.com, reddit.com).
We also confirmed that --max_old_space_size is a V8/Node.js heap flag (passed via --js-flags) and is not a recognized Chromium switch; using it directly will be ignored or emit warnings (peter.sh, bigbinary.com).

Please update crawler_mcp/crawlers/web.py (around lines 133–168) as follows:

• Gate all GPU-specific switches behind a new crawl_enable_gpu setting
• Remove the standalone --max_old_space_size=4096 argument

             browser_config = BrowserConfig(
                 headless=settings.crawl_headless,
                 browser_type=settings.crawl_browser,
                 light_mode=getattr(
                     settings, "crawl_light_mode", True
                 ),  # Optimized performance mode
                 text_mode=getattr(
                     settings, "crawl_text_mode", False
                 ),  # 3-4x faster when enabled
                 verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
-                # Aggressive performance settings
-                extra_args=[
+                # Aggressive performance settings
+                extra_args=[
                     "--no-sandbox",
                     "--disable-dev-shm-usage",
-                    # GPU Acceleration for RTX 4070
-                    "--enable-gpu",  # Enable GPU acceleration
-                    "--enable-accelerated-2d-canvas",  # GPU for 2D canvas
-                    "--enable-gpu-compositing",  # GPU for compositing
-                    "--enable-gpu-rasterization",  # GPU for rasterization
-                    "--ignore-gpu-blocklist",  # Use GPU even if blocklisted
-                    "--disable-gpu-sandbox",  # Remove GPU sandbox restrictions
-                    "--enable-zero-copy",  # Zero-copy GPU textures
-                    "--use-gl=egl",  # Use EGL for headless GPU
-                    "--max_old_space_size=4096",  # 4GB memory per browser
+                    # Conditionally enable GPU flags only when a GPU is available
+                    *(
+                        [
+                            "--enable-gpu",
+                            "--enable-accelerated-2d-canvas",
+                            "--enable-gpu-compositing",
+                            "--enable-gpu-rasterization",
+                            "--ignore-gpu-blocklist",
+                            "--disable-gpu-sandbox",
+                            "--enable-zero-copy",
+                            "--use-gl=egl",
+                        ]
+                        if getattr(settings, "crawl_enable_gpu", False)
+                        else []
+                    ),
                     "--disable-background-timer-throttling",
                     "--disable-backgrounding-occluded-windows",
                     "--disable-renderer-backgrounding",
                     "--aggressive-cache-discard",
                     "--memory-pressure-off",
                     # Network optimizations for faster page loads
                     "--max-connections-per-host=30",  # Match semaphore_count
                     "--enable-quic",
                     "--enable-tcp-fast-open",
                 ],
             )

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +209 to 229
                                        raise
                            else:
                                errors.append(
                                    f"Failed to crawl {result.url}: {result.error_message}"
                                )
                            if len(successful_results) >= max_pages:
                                break
                    elif isinstance(crawl_result, Crawl4aiResult):
                        # Single result case (when stream=False and no deep crawl)
                        crawl_count = 1
                        self.logger.info(
                            "Processing single CrawlResult (stream=False, no deep crawl)"
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed single result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                            )
                    elif hasattr(crawl_result, "success") and hasattr(
                        crawl_result, "url"
                    ):
                        # Handle CrawlResultContainer and other container types
                        crawl_count = 1
                        self.logger.info(
                            "Processing container result type: %s",
                            type(crawl_result).__name__,
                        )
                        if crawl_result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(
                                    crawl_result
                                )
                                successful_results.append(sanitized_result)
                                self.logger.info(
                                    "Successfully processed container result for %s",
                                    crawl_result.url,
                                )
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        crawl_result.url,
                                    )
                                else:
                                    raise
                        else:
                            errors.append(
                                "Failed to crawl {}: {}".format(
                                    crawl_result.url,
                                    getattr(
                                        crawl_result, "error_message", "Unknown error"
                                    ),
                                )
                            )
                    else:
                        raise Exception(
                            f"Unexpected crawl result type: {type(crawl_result)} (deep_crawl={run_config.deep_crawl_strategy is not None})"
                        )
            if (
                getattr(settings, "use_arun_many_for_sitemaps", False)
                and sitemap_seeds
                and len(sitemap_seeds) > 1
            ):
                self.logger.info(
                    f"Using arun_many() approach with {len(sitemap_seeds)} sitemap URLs (max_concurrent_sessions={getattr(settings, 'max_concurrent_sessions', 20)})"
                )
                successful_results = await self._crawl_using_arun_many(
                    browser, sitemap_seeds, run_config, request, progress_callback
                )
            else:
                self.logger.info(
                    "Using BFSDeepCrawlStrategy approach with async iteration..."
                )
                successful_results = await self._crawl_using_deep_strategy(
                    browser, first_url, run_config, max_pages
                )

@coderabbitai coderabbitai bot Aug 21, 2025

⚠️ Potential issue

Fix crawl_count and logging; remove unused counter

crawl_count is initialized (Line 210) but never incremented, so the final log is misleading.

Apply:

-            crawl_count = 0
+            # Defer computing crawl_count until after we have results

-                successful_results = await self._crawl_using_arun_many(
+                successful_results, errors = await self._crawl_using_arun_many(
                     browser, sitemap_seeds, run_config, request, progress_callback
                 )

-                successful_results = await self._crawl_using_deep_strategy(
+                successful_results, errors = await self._crawl_using_deep_strategy(
                     browser, first_url, run_config, max_pages
                 )

-            self.logger.info(
-                "Crawl loop completed: %s results processed, %s successful pages",
-                crawl_count,
-                len(pages),
-            )
+            self.logger.info(
+                "Crawl loop completed: %s results processed, %s successful pages",
+                len(successful_results),
+                len(pages),
+            )

Also applies to: 263-267
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +219 to +221
                successful_results = await self._crawl_using_arun_many(
                    browser, sitemap_seeds, run_config, request, progress_callback
                )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Wire returned errors into execute() statistics

After propagating errors from the two crawl paths, use them to compute totals.

-                successful_results = await self._crawl_using_arun_many(
+                successful_results, errors = await self._crawl_using_arun_many(
                     browser, sitemap_seeds, run_config, request, progress_callback
                 )
@@
-                successful_results = await self._crawl_using_deep_strategy(
+                successful_results, errors = await self._crawl_using_deep_strategy(
                     browser, first_url, run_config, max_pages
                 )

No code diff needed in 234-247; you already populate pages from successful_results. Ensure statistics uses len(errors) below.

Also applies to: 226-228, 234-247, 273-283
@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +231 to 233
            pages = []
            errors = []

@coderabbitai coderabbitai bot Aug 21, 2025

🧹 Nitpick (assertive)

Remove redundant reinitialization of pages/errors

pages and errors were already initialized before the crawl; re-initializing here serves no purpose and risks losing error state if you ever populate it earlier.

-            # Process crawling results
-            pages = []
-            errors = []
+            # Process crawling results

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +760 to 768
            # Minimal BFS strategy configuration for maximum crawling
            # Omit filter_chain entirely (don't set to None) as per crawl4ai docs
            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                max_depth=max_depth,
                include_external=False,
                max_pages=max_pages,
                # filter_chain=filter_chain,  # Disabled - even minimal filters break it
                # Omit filter_chain - it defaults to empty FilterChain() which allows all URLs
                # Omit score_threshold - it defaults to -infinity which allows all URLs
            )
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Include user include/exclude patterns; filter_chain currently ignored

You build filter_chain but don’t pass it to BFSDeepCrawlStrategy, so include/exclude patterns are ignored. Wire it in with a safe fallback.

-            # Minimal BFS strategy configuration for maximum crawling
-            # Omit filter_chain entirely (don't set to None) as per crawl4ai docs
-            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
-                max_depth=max_depth,
-                include_external=False,
-                max_pages=max_pages,
-                # Omit filter_chain - it defaults to empty FilterChain() which allows all URLs
-                # Omit score_threshold - it defaults to -infinity which allows all URLs
-            )
+            # Prefer passing filter_chain when available (with defensive fallback)
+            try:
+                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                    filter_chain=filter_chain if filter_chain else None,
+                )
+            except TypeError:
+                # Older versions may not accept filter_chain kwarg
+                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                )

Also applies to: 773-777
🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +906 to +1008
        """Crawl using BFSDeepCrawlStrategy with async generator."""
        successful_results = []
        errors = []

        with suppress_stdout():
            try:
                # Get result from arun - type depends on config.stream setting
                self.logger.info(
                    "About to call browser.arun with stream=%s", run_config.stream
                )
                crawl_result = await browser.arun(url=first_url, config=run_config)
                self.logger.info("browser.arun completed successfully")

                # Debug: Log the actual type we received
                self.logger.info(
                    "CRAWL DEBUG: crawl_result type = %s, stream=%s, deep_crawl=%s",
                    type(crawl_result).__name__,
                    run_config.stream,
                    run_config.deep_crawl_strategy is not None,
                )

                # Handle different return types based on deep crawl strategy and stream setting
                if hasattr(crawl_result, "__aiter__"):
                    # AsyncGenerator case (when stream=True)
                    self.logger.info(
                        "Processing AsyncGenerator results (stream=True mode) - starting iteration"
                    )
                    generator_count = 0
                    async for result in crawl_result:
                        generator_count += 1

                        self.logger.info(
                            f"AsyncGenerator yielded result #{generator_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                        )

                        # Pre-check for unexpected types (defensive programming)
                        if isinstance(result, int):
                            self.logger.warning(
                                "Received integer %d instead of CrawlResult in streaming mode, skipping",
                                result,
                            )
                            continue

                        # Ensure result is a CrawlResult object
                        if not hasattr(result, "success"):
                            self.logger.warning(
                                "Received unexpected type %s in streaming mode, skipping",
                                type(result).__name__,
                            )
                            continue

                        if result.success:
                            try:
                                sanitized_result = self._sanitize_crawl_result(result)
                                successful_results.append(sanitized_result)
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        result.url,
                                    )
                                    continue
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {result.url}: {result.error_message}"
                            )
                        if len(successful_results) >= max_pages:
                            self.logger.info(
                                f"Breaking from AsyncGenerator loop: reached max_pages ({max_pages})"
                            )
                            break

                    self.logger.info(
                        f"AsyncGenerator iteration completed: yielded {generator_count} results, {len(successful_results)} successful"
                    )
                else:
                    # Handle single result or list cases
                    self.logger.info(
                        f"Received non-async result: {type(crawl_result).__name__}"
                    )
                    if hasattr(crawl_result, "success"):
                        if crawl_result.success:
                            sanitized_result = self._sanitize_crawl_result(crawl_result)
                            successful_results.append(sanitized_result)
                        else:
                            errors.append(
                                f"Failed to crawl {crawl_result.url}: {crawl_result.error_message}"
                            )

            except Exception as e:
                self.logger.error(f"Deep crawl strategy failed: {e}", exc_info=True)
                errors.append(str(e))

        return successful_results

@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Propagate errors up to execute(); currently dropped

_errors collected here are never surfaced; execute() always reports 0 failures. Return both successes and errors so statistics.total_pages_failed is accurate.

-    async def _crawl_using_deep_strategy(
-        self, browser: Any, first_url: str, run_config: Any, max_pages: int
-    ) -> list[Any]:
+    async def _crawl_using_deep_strategy(
+        self, browser: Any, first_url: str, run_config: Any, max_pages: int
+    ) -> tuple[list[Any], list[str]]:
@@
-        return successful_results
+        return successful_results, errors

🤖 Prompt for AI Agents

@jmagar
crawler_mcp/crawlers/web.py
Comment on lines +1010 to +1099
        self,
        browser: Any,
        sitemap_urls: list[str],
        run_config: Any,
        request: Any,
        progress_callback: Any,
    ) -> list[Any]:
        """Crawl using arun_many() with discovered sitemap URLs."""
        from crawl4ai import MemoryAdaptiveDispatcher  # type: ignore

        successful_results = []
        max_pages = request.max_pages or len(sitemap_urls)
        max_concurrent = getattr(settings, "max_concurrent_sessions", 20)

        # Limit sitemap URLs to max_pages
        urls_to_crawl = sitemap_urls[:max_pages]

        self.logger.info(
            f"Creating MemoryAdaptiveDispatcher with max_session_permit={max_concurrent}"
        )

        # Create dispatcher for memory-adaptive concurrency
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=getattr(settings, "crawl_memory_threshold", 80.0),
            check_interval=0.5,
            max_session_permit=max_concurrent,
        )

        # Remove deep_crawl_strategy to avoid recursion and set streaming
        batch_config = (
            run_config.clone() if hasattr(run_config, "clone") else run_config
        )
        if hasattr(batch_config, "deep_crawl_strategy"):
            batch_config.deep_crawl_strategy = None
        batch_config.stream = True

        self.logger.info(f"Starting arun_many with {len(urls_to_crawl)} URLs")

        with suppress_stdout():
            try:
                # Use arun_many for concurrent crawling
                results_generator = await browser.arun_many(
                    urls=urls_to_crawl, config=batch_config, dispatcher=dispatcher
                )

                processed_count = 0
                async for result in results_generator:
                    processed_count += 1
                    self.logger.info(
                        f"arun_many result #{processed_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                    )

                    if hasattr(result, "success") and result.success:
                        try:
                            sanitized_result = self._sanitize_crawl_result(result)
                            successful_results.append(sanitized_result)

                            if progress_callback:
                                progress_callback(
                                    len(successful_results),
                                    max_pages,
                                    f"Crawled {result.url}",
                                )

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to process result for {result.url}: {e}"
                            )

                    if len(successful_results) >= max_pages:
                        self.logger.info(f"Reached max_pages limit ({max_pages})")
                        break

                self.logger.info(
                    f"arun_many completed: {processed_count} processed, {len(successful_results)} successful"
                )

            except Exception as e:
                self.logger.error(f"arun_many approach failed: {e}", exc_info=True)
                # Fallback to single URL if arun_many fails
                if urls_to_crawl:
                    self.logger.info("Falling back to single URL crawl")
                    single_result = await browser.arun(
                        url=urls_to_crawl[0], config=batch_config
                    )
                    if hasattr(single_result, "success") and single_result.success:
                        sanitized_result = self._sanitize_crawl_result(single_result)
                        successful_results.append(sanitized_result)

        return successful_results
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

⚠️ Potential issue

Propagate errors for arun_many path too; update execute callers

Same issue as above: errors are collected but never returned to execute().

-    ) -> list[Any]:
+    ) -> tuple[list[Any], list[str]]:
@@
-        successful_results = []
+        successful_results: list[Any] = []
+        errors: list[str] = []
@@
-                            self.logger.warning(
-                                f"Failed to process result for {result.url}: {e}"
-                            )
+                            self.logger.warning(
+                                "Failed to process result for %s: %s", getattr(result, "url", "unknown"), e
+                            )
+                            errors.append(str(e))
@@
-                self.logger.error(f"arun_many approach failed: {e}", exc_info=True)
+                self.logger.error("arun_many approach failed: %s", e, exc_info=True)
                 # Fallback to single URL if arun_many fails
                 if urls_to_crawl:
                     self.logger.info("Falling back to single URL crawl")
                     single_result = await browser.arun(
                         url=urls_to_crawl[0], config=batch_config
                     )
                     if hasattr(single_result, "success") and single_result.success:
                         sanitized_result = self._sanitize_crawl_result(single_result)
                         successful_results.append(sanitized_result)
+                    else:
+                        errors.append(f"Failed to crawl {getattr(single_result, 'url', urls_to_crawl[0])}")
@@
-        return successful_results
+        return successful_results, errors

🤖 Prompt for AI Agents

@jmagar
@jmagar
@claude
fix: suppress BeautifulSoup 'text' argument deprecation warnings
0272afc
coderabbitai[bot]
coderabbitai bot reviewed Aug 21, 2025
coderabbitai bot left a comment

Actionable comments posted: 3

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.
⚠️ Outside diff range comments (3)

♻️ Duplicate comments (3)

📜 Review details


🔇 Additional comments (3)

















pyproject.toml
    "types-psutil>=7.0.0.20250801",
    "torch>=2.8.0",
@coderabbitai coderabbitai bot Aug 21, 2025

💡 Verification agent
🧩 Analysis chain

Make torch an optional dependency group

The torch import in crawler_mcp/core/rag.py is gated by settings.reranker_enabled—but since reranker_enabled defaults to True (see crawler_mcp/config.py:81), every installation pulls in torch (and sentence-transformers) by default. To avoid forcing heavy, platform-specific installs for users who don’t need local embeddings at runtime, move these packages into an optional “ml” extras group.

Locations requiring attention:

    crawler_mcp/config.py:81 — reranker_enabled: bool = Field(default=True, ...)
    crawler_mcp/core/rag.py:224–225 — import torch and from sentence_transformers import CrossEncoder under the if settings.reranker_enabled guard

Suggested changes:

 pyproject.toml
 [project]
-dependencies = [
-    "torch>=2.8.0",
-    …
-]
+dependencies = [
+    # other mandatory deps, but no torch or sentence-transformers
+    …
+]

+[project.optional-dependencies]
+ml = [
+    "torch>=2.8.0",
+    "transformers>=4.55.0",
+    "sentence-transformers>=5.1.0",
+]

Please also update documentation to instruct users to install pip install crawler-mcp[ml] if they need reranking with local models.
📝 Committable suggestion


@jmagar
pyproject.toml
Comment on lines +112 to +129
strict_optional = true
show_error_codes = true
disallow_any_generics = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true

# Allow protocols to be mixed with runtime implementations
allow_subclassing_any = true
allow_untyped_calls = false

# Disable specific checks for dynamic imports
disable_error_code = ["no-any-return"]

# Handle missing imports gracefully for optional dependencies
ignore_missing_imports = false
follow_imports = "silent"

@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Resolve mypy config contradiction: you enable warn_return_any but disable no-any-return

warn_return_any is rendered toothless if you disable the “no-any-return” error. Either keep the warning or drop the disable. Recommend enforcing the check.

Apply this diff to re-enable the check and remove the stale comment:

-# Disable specific checks for dynamic imports
-disable_error_code = ["no-any-return"]

🤖 Prompt for AI Agents

@jmagar
pyproject.toml
Comment on lines +140 to +147
minversion = "7.4"
addopts = [
    "--strict-config",
    "--strict-markers",
    "--disable-warnings",
    "--tb=short",
    "-ra",
]
@coderabbitai coderabbitai bot Aug 21, 2025

🛠️ Refactor suggestion

Align pytest minversion with your dev tooling and drop redundant warning suppression

    dev dependency group pins pytest>=8.4.1; set minversion accordingly.
    --disable-warnings conflicts with filterwarnings=error/ignore rules and obscures useful info. Remove it.

Apply this diff:

-minversion = "7.4"
+minversion = "8.4"
@@
 addopts = [
     "--strict-config",
     "--strict-markers",
-    "--disable-warnings",
     "--tb=short",
     "-ra",
 ]

🤖 Prompt for AI Agents

@jmagar
Merge info
All checks have passed

2 successful checks
successful checks

CodeRabbit
CodeRabbit — Review completed
GitGuardian Security Checks
GitGuardian Security ChecksSuccessful in 2s — No secrets detected ✅

No conflicts with base branch

Merging can be performed automatically.
You can also merge this with the command line.
@jmagar
Add a comment
Comment

Add your comment here...
Remember, contributions to this repository should follow our GitHub Community Guidelines.
ProTip! Add comments to specific lines under Files changed.
Reviewers

@coderabbitai
coderabbitai[bot]

Copilot code review
Copilot

Still in progress?
Assignees
No one—
Labels
None yet
Projects
None yet
Milestone
No milestone
Development

Successfully merging this pull request may close these issues.

None yet

You’re receiving notifications because you authored the thread.
1 participant
@jmagar
Footer
© 2025 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Docs
    Contact

coderabbitai bot reviewed View reviewed changes coderabbitai bot left a comment Actionable comments posted: 3 Caution Some comments are outside the diff and can’t be posted inline due to platform limitations. ⚠️ Outside diff range comments (3) ♻️ Duplicate comments (3) 📜 Review details 🔇 Additional comments (3) pyproject.toml "types-psutil>=7.0.0.20250801", "torch>=2.8.0", coderabbitai bot 💡 Verification agent 🧩 Analysis chain Make torch an optional dependency group The torch import in crawler_mcp/core/rag.py is gated by settings.reranker_enabled—but since reranker_enabled defaults to True (see crawler_mcp/config.py:81), every installation pulls in torch (and sentence-transformers) by default. To avoid forcing heavy, platform-specific installs for users who don’t need local embeddings at runtime, move these packages into an optional “ml” extras group. Locations requiring attention: crawler_mcp/config.py:81 — reranker_enabled: bool = Field(default=True, ...) crawler_mcp/core/rag.py:224–225 — import torch and from sentence_transformers import CrossEncoder under the if settings.reranker_enabled guard Suggested changes: pyproject.toml [project] -dependencies = [ - "torch>=2.8.0", - … -] +dependencies = [ + # other mandatory deps, but no torch or sentence-transformers + … +] +[project.optional-dependencies] +ml = [ + "torch>=2.8.0", + "transformers>=4.55.0", + "sentence-transformers>=5.1.0", +] Please also update documentation to instruct users to install pip install crawler-mcp[ml] if they need reranking with local models. 📝 Committable suggestion Reply... Resolve conversation pyproject.toml Comment on lines +112 to +129 strict_optional = true show_error_codes = true disallow_any_generics = true disallow_incomplete_defs = true check_untyped_defs = true disallow_untyped_decorators = true # Allow protocols to be mixed with runtime implementations allow_subclassing_any = true allow_untyped_calls = false # Disable specific checks for dynamic imports disable_error_code = ["no-any-return"] # Handle missing imports gracefully for optional dependencies ignore_missing_imports = false follow_imports = "silent" coderabbitai bot 🛠️ Refactor suggestion Resolve mypy config contradiction: you enable warn_return_any but disable no-any-return warn_return_any is rendered toothless if you disable the “no-any-return” error. Either keep the warning or drop the disable. Recommend enforcing the check. Apply this diff to re-enable the check and remove the stale comment: -# Disable specific checks for dynamic imports -disable_error_code = ["no-any-return"] 📝 Committable suggestion 🤖 Prompt for AI Agents Reply... Resolve conversation pyproject.toml Comment on lines +140 to +147 minversion = "7.4" addopts = [ "--strict-config", "--strict-markers", "--disable-warnings", "--tb=short", "-ra", ] coderabbitai bot 🛠️ Refactor suggestion Align pytest minversion with your dev tooling and drop redundant warning suppression dev dependency group pins pytest>=8.4.1; set minversion accordingly. --disable-warnings conflicts with filterwarnings=error/ignore rules and obscures useful info. Remove it. Apply this diff: -minversion = "7.4" +minversion = "8.4" @@ addopts = [ "--strict-config", "--strict-markers", - "--disable-warnings", "--tb=short", "-ra", ] 📝 Committable suggestion 🤖 Prompt for AI Agents Reply... Resolve conversation
