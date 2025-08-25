# AI Review Content from PR #9

[COPILOT REVIEW - copilot-pull-request-reviewer[bot]]
## Pull Request Overview

This PR implements comprehensive content filtering capabilities for web crawling, introducing configuration options for cleaner markdown extraction and improved content quality through CSS selector filtering and pruning strategies.

Key changes:
- Adds content filtering configuration fields to CrawlRequest model including excluded tags/selectors, content selectors, and pruning thresholds
- Implements configurable markdown preference (fit_markdown vs raw_markdown) with fallback logic
- Adds thread-safe tokenizer initialization with class-level caching to prevent repeated downloads

### Reviewed Changes

Copilot reviewed 4 out of 4 changed files in this pull request and generated 4 comments.

| File | Description |
|---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -            chunk_size_words = int(self.chunk_size / approx_tokens_per_word)
> -            overlap_words = int(self.overlap / approx_tokens_per_word)
> +            # Ensure at least one word per chunk and valid overlap relation
> +            chunk_size_words = max(1, int(round(self.chunk_size / approx_tokens_per_word)))
> +            overlap_words = max(
> +                0,
> +                min(int(round(self.overlap / approx_tokens_per_word)), chunk_size_words - 1),
> +            )
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -                print(
> -                    f"CRAWL DEBUG - result.markdown is integer {result.markdown}, returning empty",
> -                    file=sys.stderr,
> -                    flush=True,
> -                )
> +                self.logger.debug(
> +                    "CRAWL DEBUG - result.markdown is integer %s, returning empty",
> +                    result.markdown,
> +                )
> @@
> -            print(
> -                f"CRAWL DEBUG - Exception accessing markdown attributes: {e}",
> -                file=sys.stderr,
> -                flush=True,
> -            )
> +            self.logger.debug(
> +                "CRAWL DEBUG - Exception accessing markdown attributes: %s", e
> +            )
> @@
> -        print(debug_msg, file=sys.stderr, flush=True)
> +        self.logger.debug(debug_msg)
> +        if hasattr(self, "ctx") and getattr(self, "ctx", None):
> +            with contextlib.suppress(Exception):
> +                self.ctx.info(f"Processed content for {getattr(result, 'url', 'unknown')}")
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -            title=result.metadata.get("title", ""),
> +            title=(getattr(result, "metadata", {}) or {}).get("title", ""),
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -                total_bytes += len(page_content.content)
> +                total_bytes += len(page_content.content.encode("utf-8", "ignore"))
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -            crawl_count = 0
> +            # Number of results received from crawl4ai before PageContent conversion
> +            crawl_count = 0
> @@
> -            self.logger.info(
> -                "Crawl loop completed: %s results processed, %s successful pages",
> -                crawl_count,
> -                len(pages),
> -            )
> +            # If we used deep strategy or arun_many, successful_results exists
> +            with contextlib.suppress(Exception):
> +                crawl_count = len(successful_results)
> +            self.logger.info(
> +                "Crawl loop completed: %s results processed, %s successful pages",
> +                crawl_count,
> +                len(pages),
> +            )
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
> -        except ImportError as e:
> -            logger.error(
> +        except ImportError as e:
> +            self.logger.error(
>                  "MemoryAdaptiveDispatcher not available from crawl4ai: %s. "
>                  "Falling back to sequential crawling.",
>                  e,
>              )
> @@
> -        except Exception as e:
> -            logger.error(
> +        except Exception as e:
> +            self.logger.error(
>                  "Unexpected error importing MemoryAdaptiveDispatcher: %s. "
>                  "Falling back to sequential crawling.",
>                  e,
>              )
>---

[PYTHON BLOCK - coderabbitai[bot] - crawler_mcp/config.py:445-447]
if 'css_selector' in inspect.signature(CrawlerRunConfig.__init__).parameters:
      run_config = CrawlerRunConfig(..., css_selector=content_selector, ...)
  else:
      run_config = CrawlerRunConfig(...)
      with suppress(AttributeError):
          run_config.css_selector = content_selector---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
> -        if extraction_strategy == "llm":
> +        if extraction_strategy == "llm":
>              with contextlib.suppress(Exception):
> -                run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
> -                    provider="openai",
> -                    api_token="",
> +                token = os.getenv("LLM_PROVIDER_API_TOKEN", "")
> +                if not token:
> +                    self.logger.warning("LLM extraction requested but no API token set; skipping.")
> +                else:
> +                    run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
> +                    provider=os.getenv("LLM_PROVIDER", "openai"),
> +                    api_token=token,
>                      instruction="Extract main content and key information from the page",
>                  )
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
> -            KeywordRelevanceScorer(keywords=keywords, weight=0.7)  # type: ignore[attr-defined]
> +            try:
> +                # If supported by your crawl4ai version:
> +                deep_kwargs = {"scorer": KeywordRelevanceScorer(keywords=keywords, weight=0.7)}
> +            except Exception:
> +                deep_kwargs = {}
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
> -            crawl_count = 0
> +            crawl_count = 0  # kept if you plan to count yielded items upstream
> ...
> -            self.logger.info(
> -                "Crawl loop completed: %s results processed, %s successful pages",
> -                crawl_count,
> -                len(pages),
> -            )
> +            self.logger.info(
> +                "Crawl loop completed: %s results processed, %s successful pages",
> +                len(pages),
> +                len(pages),
> +            )
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
> -            import re
>---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
".banner",
-            ".alert",
-            ".notification",
             ".edit-page",
             ".improve-page",---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
+    strict_ui_filtering: bool = Field(
+        default=False,
+        alias="STRICT_UI_FILTERING",
+        description="When true, also exclude alerts/notifications and other aggressive UI elements",
+    )---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-        self.logger.info(
-            "WebCrawlStrategy.execute() started for URL: %s", request.url[0]
-        )
+        self.logger.info("WebCrawlStrategy.execute() started for URLs: %s", request.url)

-        self.logger.info(
-            "Starting web crawl: %s (max_pages: %s, max_depth: %s)",
-            request.url[0],
+        self.logger.info(
+            "Starting web crawl: %s (max_pages: %s, max_depth: %s)",
+            request.url,
             request.max_pages,
             request.max_depth,
         )

-            sitemap_seeds = await self._discover_sitemap_seeds(
-                request.url[0], request.max_pages or 100
-            )
+            # Discover seeds from all provided URLs (same-domain)
+            sitemap_seeds: list[str] = []
+            for u in request.url:
+                sitemap_seeds.extend(
+                    await self._discover_sitemap_seeds(u, request.max_pages or 100)
+                )
+            # Deduplicate while preserving order
+            seen = set()
+            sitemap_seeds = [s for s in sitemap_seeds if not (s in seen or seen.add(s))]

-                successful_results, errors = await self._crawl_using_deep_strategy(
-                    browser, request.url[0], run_config, max_pages
-                )
+                # Use the first URL as the starting point but keep others for discovery via seeds
+                start_url = request.url[0]
+                successful_results, errors = await self._crawl_using_deep_strategy(
+                    browser, start_url, run_config, max_pages
+                )---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-            prefer_fit_markdown = getattr(request, "prefer_fit_markdown", True)
+            prefer_fit_markdown = (
+                request.prefer_fit_markdown
+                if getattr(request, "prefer_fit_markdown", None) is not None
+                else getattr(settings, "crawl_prefer_fit_markdown", True)
+            )---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-        if content_selector:
-            config_params["css_selector"] = content_selector
-            self.logger.info(f"Using content selector: {content_selector}")
-
-        if excluded_selector_string:
-            config_params["excluded_selector"] = excluded_selector_string
-            self.logger.info(
-                f"Using excluded selectors: {excluded_selector_string[:100]}..."
-            )
+        optional_params = {
+            "css_selector": content_selector,
+            "excluded_selector": excluded_selector_string,
+        }
+        filtered = {k: v for k, v in optional_params.items() if v}
+        config_params.update(filtered)
+        for k, v in filtered.items():
+            msg = "content selector" if k == "css_selector" else "excluded selectors"
+            self.logger.info("Using %s: %s", msg, str(v)[:100])---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-            # BFS strategy with minimal filtering for maximum crawling capability
-            # Omit filter_chain to allow crawl4ai to discover all possible URLs
-            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
-                max_depth=max_depth,
-                include_external=False,
-                max_pages=max_pages,
-                # Intentionally omit filter_chain - it defaults to empty FilterChain() which allows all URLs
-                # This ensures maximum crawling capability for documentation sites
-            )
+            # Apply user-provided include/exclude patterns when present
+            try:
+                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                    filter_chain=filter_chain if filter_chain is not None else None,
+                    **(deep_kwargs if "deep_kwargs" in locals() else {}),
+                )
+            except TypeError:
+                strat = BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
+                    max_depth=max_depth,
+                    include_external=False,
+                    max_pages=max_pages,
+                )
+                with contextlib.suppress(Exception):
+                    if filter_chain is not None:
+                        setattr(strat, "filter_chain", filter_chain)
+                return strat---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-        description="CSS selector to focus on main content area (None = auto-detect using semantic HTML5)",
+        description="CSS selector to focus on main content area (None = no CSS filtering; crawler should not auto-inject)",---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/config.py:756-771]
-        content_filter = PruningContentFilter(
-            threshold=pruning_threshold,  # Use configurable threshold for relevance scoring
-            threshold_type="dynamic",  # Dynamic scoring for adaptive filtering
-            min_word_threshold=min_word_threshold,  # Configurable word threshold for content blocks
-        )
+        try:
+            content_filter = PruningContentFilter(
+                threshold=pruning_threshold,
+                threshold_type="dynamic",
+                min_word_threshold=min_word_threshold,
+            )
+        except TypeError:
+            content_filter = PruningContentFilter(
+                threshold=pruning_threshold,
+                min_word_threshold=min_word_threshold,
+            )---

[COPILOT SUGGESTION - Copilot - crawler_mcp/core/rag/chunking.py:73]
_tokenizer: Any | None = None
    _tokenizer_type: str = "word-estimate"
    _tokenizer_lock: threading.Lock = threading.Lock()
    _tokenizer_initialized: bool = False---

[COPILOT SUGGESTION - Copilot - crawler_mcp/crawlers/web.py:639]
pruning_threshold = get_config_value(
            request.pruning_threshold, "crawl_pruning_threshold", 0.5
        )
        min_word_threshold = get_config_value(
            request.min_word_threshold, "crawl_min_word_threshold", 20---

[COPILOT SUGGESTION - Copilot - crawler_mcp/crawlers/web.py:742]
# Add CSS selector parameters if available, using dictionary comprehension
        optional_params = {
            "css_selector": content_selector,
            "excluded_selector": excluded_selector_string,
        }
        filtered_params = {k: v for k, v in optional_params.items() if v is not None}
        config_params.update(filtered_params)
        for k, v in filtered_params.items():
            if k == "css_selector":
                self.logger.info(f"Using content selector: {v}")
            elif k == "excluded_selector":
                self.logger.info(f"Using excluded selectors: {str(v)[:100]}...")---

[AI PROMPT - crawler_mcp/config.py:421]
In crawler_mcp/config.py around lines 351-421 (also note similar entries at
~401-403), the default crawl_excluded_selectors list is too aggressive: remove
".alert" and ".notification" from the default list OR make their exclusion
conditional behind a new boolean config (e.g., strict_ui_filtering) so
admonitions/notes remain by default; update the Field description/alias to
mention the new toggle if added and adjust any tests or docs that assume those
selectors are excluded.---

[AI PROMPT - crawler_mcp/config.py]
In crawler_mcp/config.py around lines 424 to 428, the Field default for
crawl_content_selector currently sets a non-empty CSS selector string which can
cause pages that don’t match to yield empty content; change the Field default to
None (i.e., default=None) so the selector is opt-in, keep the alias and
description, and update any type annotation/docs if needed to reflect that None
is the default and callers should provide a selector when they want content
filtering.---

[AI PROMPT - crawler_mcp/core/rag/chunking.py:76]
In crawler_mcp/core/rag/chunking.py around line 76, the word_to_token_ratio is
hard-coded to QWEN3_WORD_TO_TOKEN_RATIO; change it to use the configurable
setting (settings.word_to_token_ratio or equivalent from your config module)
with a sane fallback to QWEN3_WORD_TO_TOKEN_RATIO. Update the assignment to pull
from the config, validate that the value is a positive number (fallback if
missing/invalid), and ensure any necessary import for the settings/config module
is added at the top of the file.---

[AI PROMPT - crawler_mcp/core/rag/chunking.py:117]
In crawler_mcp/core/rag/chunking.py around lines 85 to 117, tighten tokenizer
initialization to avoid remote/network calls and surface model revision: import
AutoTokenizer at top (or ensure import inside try), call from_pretrained with
local_files_only=True and trust_remote_code=False (use an optional revision
argument pulled from settings, e.g. settings.tei_model_revision if present), log
the model name and revision when initialization succeeds, and keep the existing
fallback behavior and thread-safe double-checking; ensure exceptions are handled
the same way and tokenizer_initialized is set in finally.---

[AI PROMPT - crawler_mcp/crawlers/web.py:113]
In crawler_mcp/crawlers/web.py around lines 110-112 (and similarly at 181-183
and 231-233), the code uses only request.url[0] as the crawl seed; change
logging to include the full list of provided URLs and update sitemap discovery /
deep-crawl seeding to consider all same-domain inputs rather than a single
element. Specifically, log the entire request.url list for visibility, compute a
filtered list of same-origin URLs (or use a simple heuristic: prefer
request.url[0] but fall back to other same-domain entries if sitemap discovery
fails), and pass that list into sitemap discovery and the deep-crawl starter so
multiple entry points on the same domain are used for discovery.---

[AI PROMPT - crawler_mcp/crawlers/web.py:242]
In crawler_mcp/crawlers/web.py around lines 239-240, the code unconditionally
sets prefer_fit_markdown from request with a default True, which overrides
settings.crawl_prefer_fit_markdown; change it to obtain the request attribute
without a default (e.g., getattr(request, "prefer_fit_markdown", None)) and if
the result is None (or the attribute is missing) fall back to
settings.crawl_prefer_fit_markdown so the effective preference honors settings
when the request does not explicitly set it.---

[AI PROMPT - crawler_mcp/crawlers/web.py:908]
In crawler_mcp/crawlers/web.py around lines 844 to 852, the code constructs a
filter_chain from include/exclude patterns but intentionally omits passing it to
BFSDeepCrawlStrategy, causing user patterns to be ignored; update the return to
pass filter_chain (if present) into BFSDeepCrawlStrategy so include/exclude are
applied, defaulting to an empty permissive FilterChain only when no patterns
were provided; also address the unused KeywordRelevanceScorer by either passing
it into the strategy constructor (if the strategy supports a scorer parameter)
or removing the scorer creation to eliminate dead code.---

[AI PROMPT - crawler_mcp/models/crawl.py:99]
In crawler_mcp/models/crawl.py around lines 74-99, the prefer_fit_markdown field
is currently a required bool with default True which prevents the global setting
crawl_prefer_fit_markdown from being respected; change the field to be optional
(bool | None) with default=None and update its Field description to indicate
when None the global setting will be used; then update
crawler_mcp/crawlers/web.py to treat None as "defer to settings" and compute the
effective boolean at use-site by falling back to the crawl_prefer_fit_markdown
setting.---

[AI PROMPT - crawler_mcp/config.py:77]
In crawler_mcp/config.py around lines 74–77, the fixed tei_batch_size can exceed
TEI_MAX_BATCH_TOKENS and cause silent truncation or OOM; change the config to
either auto-derive tei_batch_size from TEI_MAX_BATCH_TOKENS using an estimated
tokens_per_item (configurable default, e.g., 20–50) by computing derived =
max(1, TEI_MAX_BATCH_TOKENS // tokens_per_item) and set tei_batch_size =
min(user_provided, derived), or keep the user-provided value but add a startup
guardrail that checks if tei_batch_size * tokens_per_item > TEI_MAX_BATCH_TOKENS
and logs a warning and caps the effective batch size to derived; ensure
tokens_per_item is configurable and that logs include both values for debugging.---

[AI PROMPT - crawler_mcp/config.py:106]
In crawler_mcp/config.py around lines 86 to 106, the exponential backoff
parameters lack jitter which can cause thundering herd; modify the backoff
calculation to apply a small random jitter (e.g., multiply the computed delay by
a factor sampled from uniform(0.8, 1.2)) and clamp the result between
retry_initial_delay and retry_max_delay; add or reference a small helper
function (e.g., compute_backoff(attempts) or get_retry_delay) that computes
base_delay = min(retry_max_delay, retry_initial_delay * (retry_exponential_base
** attempts)) then returns int/float(base_delay * random.uniform(0.8, 1.2))
ensuring imports for random are present and type hints/validation remain intact.---

[AI PROMPT - crawler_mcp/config.py:331]
In crawler_mcp/config.py around lines 325 to 331, the crawl_exclude_url_patterns
default is an empty list which expands crawl scope and cost; replace it with a
conservative set of default exclusion patterns (e.g. common admin/auth endpoints
like /admin, /login, /logout, /signup, /auth and sensitive paths like /wp-admin,
/dashboard, plus common large/binary file extensions like .zip, .exe, .bin,
.pdf, .jpg, .png, .mp4) so the app avoids login/admin areas and heavy binary
downloads by default, keep the Field alias="CRAWL_EXCLUDE_URL_PATTERNS" so
environment overrides still work, and ensure the patterns are strings in the
list and update any nearby docstring or config comment to mention these
conservative defaults and that users can opt-in to broader crawling via the env
var.---

[COMMITTABLE SUGGESTION - coderabbitai[bot] - crawler_mcp/config.py:350]
# Content Filtering Configuration - Clean Markdown Generation
    crawl_excluded_tags: list[str] = Field(
        default=[
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
        ],
        alias="CRAWL_EXCLUDED_TAGS",
        description="HTML tags to exclude during content extraction for cleaner markdown",
    )---

[AI PROMPT - crawler_mcp/crawlers/web.py:413]
In crawler_mcp/crawlers/web.py around lines 411-413 (and similarly at 485-494
and 555-562), the method _safe_get_markdown (and other spots) currently prints
directly to stderr; replace all direct print/error prints with structured
logging or context methods: use the module logger (e.g.,
logger.debug/info/warn/error) for internal/diagnostic messages and use the
provided ctx.info or ctx.warn for client-facing messages when a ctx is
available; remove any bare prints to stderr, propagate the original
message/exception text into the logger/ctx call, and ensure log level matches
intent (debug for verbose internals, info for user-facing notifications, error
for failures).---

[AI PROMPT - crawler_mcp/crawlers/web.py:535]
In crawler_mcp/crawlers/web.py around lines 495-535, the post-processing regexes
are too aggressive (they remove any inline "Copy" and package manager mentions
anywhere); update them to only target UI artifacts by matching whole lines or
content inside code-fence / known UI element contexts: change the "Copy" regex
to anchor to full-line/button patterns (e.g., match ^\s*Copy\s*$ or common
button wrappers), change the package-manager regex to only remove lines that
consist solely of package-manager tokens/flags (anchored ^...$ and allow leading
prompt chars) or UI tab labels, and restrict the repeated-navigation removal to
full-line repeated nav breadcrumbs; additionally, consider gating this cleanup
behind a boolean flag (e.g., clean_ui_artifacts) from settings so it can be
disabled for real docs.---

[AI PROMPT - crawler_mcp/crawlers/web.py:683]
In crawler_mcp/crawlers/web.py around lines 666 to 683, the code unconditionally
injects semantic HTML5 content selectors when request.content_selector is None;
change this to be opt-in via a new config flag. Add
crawl_use_semantic_default_selector (alias CRAWL_USE_SEMANTIC_DEFAULT_SELECTOR,
default False) to config.py as described, then wrap the semantic default
assignment in a conditional that checks getattr(settings,
"crawl_use_semantic_default_selector", False) before assigning the semantic
selector and logging; leave behavior unchanged when the flag is False so None
remains allowed.---

[AI PROMPT - crawler_mcp/crawlers/web.py:683]
In crawler_mcp/crawlers/web.py around lines 666-683 (and similarly adjust lines
~732-742), the code currently logs progress only to the server logger; update it
to emit user-facing progress via ctx.info when a context is available while
retaining server diagnostics in self.logger. Specifically, where you log
start/approach choices and per-URL progress, call ctx.info(...) with concise
progress messages in addition to self.logger.info(...), and also invoke the
existing progress_callback if provided; guard ctx.info calls with "if ctx and
hasattr(ctx, 'info')" to avoid attribute errors. Ensure messages to ctx.info are
user-friendly and not verbose diagnostics.---

[DIFF BLOCK - coderabbitai[bot] - crawler_mcp/crawlers/web.py:731]
-        run_config = CrawlerRunConfig(**config_params)
+        try:
+            run_config = CrawlerRunConfig(**config_params)
+        except TypeError as e:
+            self.logger.warning("Retrying run config without optional CSS params: %s", e)
+            for key in ("css_selector", "excluded_selector", "scraping_strategy"):
+                config_params.pop(key, None)
+            run_config = CrawlerRunConfig(**config_params)---

[PYTHON BLOCK - coderabbitai[bot] - crawler_mcp/config.py:421]
# Content Filtering Configuration - Clean Markdown Generation
crawl_excluded_tags: list[str] = Field(
    default=[
        "nav",
        "header", 
        "footer",
        "aside",
        "sidebar",
        "form",
        "button",
        "input",
        "select",
        "textarea",
    ],
    alias="CRAWL_EXCLUDED_TAGS",
    description="HTML tags to exclude during content extraction for cleaner markdown",
)

crawl_strict_ui_filtering: bool = Field(
    default=False,
    alias="CRAWL_STRICT_UI_FILTERING", 
    description="Enable strict UI filtering that may remove documentation alerts/notifications",
)

@property
def crawl_excluded_selectors_list(self) -> list[str]:
    """Get excluded selectors list based on strict filtering setting."""
    base_selectors = [
        # Copy buttons - comprehensive patterns
        ".copy-button",
        ".copy-code-button", 
        ".copy-btn",
        ".btn-copy",
        ".btn-clipboard",
        "button[title*='Copy']",
        "button[aria-label*='Copy']",
        "button[class*='copy']", 
        "button[data-copy]",
        "[data-copy-button]",
        ".clipboard-button",
        # Tab navigation - all variants
        ".tab-nav",
        ".tab-nav-item",
        ".tab-switcher",
        ".tabs",
        ".tab-buttons", 
        ".tab-container",
        ".package-manager-tabs",
        ".code-tabs",
        "[role='tablist']",
        ".tab-list",
        "[data-tabs]",
        # Navigation elements
        ".breadcrumb",
        ".breadcrumbs",
        ".nav-breadcrumb",
        ".breadcrumb-nav",
        ".sidebar",
        ".navigation",
        ".nav-menu",
        ".menu-nav",
        ".site-nav", 
        ".toc-sidebar",
        ".doc-nav",
        ".header-nav",
        ".footer-nav",
        ".pagination-nav",
        ".mobile-nav",
        ".nav-toggle",
        ".hamburger-menu",
        # Documentation UI artifacts (safe to always exclude)
        ".social-share",
        ".share-buttons",
        ".ad-banner", 
        ".promo",
        ".banner",
        ".edit-page",
        ".improve-page",
        ".feedback",
        ".edit-link",
        ".improve-doc",
        ".report-issue",
        ".last-updated",
        ".contributors",
        ".page-metadata",
        ".version-selector",
        ".language-selector",
        # Search and interactive elements
        ".search-box",
        ".filter-bar",
        ".sort-options", 
        ".search-input",
    ]
    
    # Add aggressive selectors only if strict filtering is enabled
    if self.crawl_strict_ui_filtering:
        base_selectors.extend([
            ".alert",
            ".notification", 
        ])
    
    return base_selectors

crawl_excluded_selectors: list[str] = Field(
    default_factory=list,  # Will be populated by property
    alias="CRAWL_EXCLUDED_SELECTORS",
    description="CSS selectors for UI elements to exclude from content extraction (use crawl_strict_ui_filtering for alerts/notifications)",
)