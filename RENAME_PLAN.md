# Rename Plan: crawlerr ➜ crawler_mcp (plus folder/file simplifications)

This document specifies the full refactor plan to:
- Rename the top-level package `crawlerr` ➜ `crawler_mcp`
- Rename folder `services` ➜ `core`
- Rename folder `services/strategies` ➜ `crawlers`
- Simplify redundant filenames (drop `_service`, `_manager`, `_strategy`, `_tools` suffixes)
- Rename tools, middleware, and models files accordingly
- Update imports, exports, and `pyproject.toml` entrypoint

Follow the steps in order. Commit in small increments; run checks after each major step.

---

## 1) Top-level package rename

- Directory: `crawlerr/` ➜ `crawler_mcp/`
- Update all imports from `crawlerr.*` ➜ `crawler_mcp.*`
- Update `pyproject.toml` entrypoint:
  - `[project.scripts] crawlerr = "crawlerr.server:main"` ➜ `[project.scripts] crawler-mcp = "crawler_mcp.server:main"`
  - Optionally keep old script alias temporarily for transition, or provide a console script alias.
- Update `crawler_mcp/__init__.py` to mirror the old `crawlerr/__init__.py` with adjusted imports and `__all__`.

Notes:
- Ensure `get_mcp()`/`get_main()` still lazily import from `crawler_mcp.server`.

---

## 2) services ➜ core (and file simplifications)

- Folder rename: `crawlerr/services/` ➜ `crawler_mcp/core/`
- File renames within `core/`:
  - `crawl_orchestrator.py` ➜ `orchestrator.py`
  - `embedding_service.py` ➜ `embeddings.py`
  - `memory_manager.py` ➜ `memory.py`
  - `rag_service.py` ➜ `rag.py`
  - `source_service.py` ➜ `sources.py`
  - `vector_service.py` ➜ `vectors.py`

- Exports (new `crawler_mcp/core/__init__.py`):
  - Export canonical names used externally:
    - `CrawlerService` from `.orchestrator`
    - `EmbeddingService` from `.embeddings`
    - `RagService` from `.rag`
    - `SourceService` from `.sources`
    - `VectorService` from `.vectors`
  - `__all__ = ["CrawlerService", "EmbeddingService", "RagService", "SourceService", "VectorService"]`

- Update all imports of old paths:
  - `from crawlerr.services import ...` ➜ `from crawler_mcp.core import ...`
  - `from crawlerr.services.rag_service import RagService` ➜ `from crawler_mcp.core.rag import RagService`
  - `from crawlerr.services.source_service import SourceService` ➜ `from crawler_mcp.core.sources import SourceService`
  - `from crawlerr.services.embedding_service import EmbeddingService` ➜ `from crawler_mcp.core.embeddings import EmbeddingService`
  - `from crawlerr.services.vector_service import VectorService` ➜ `from crawler_mcp.core.vectors import VectorService`
  - `from crawlerr.services.memory_manager import ...` ➜ `from crawler_mcp.core.memory import ...`

---

## 3) strategies ➜ crawlers (and file simplifications)

- Folder rename: `crawlerr/services/strategies/` ➜ `crawler_mcp/crawlers/`
- File renames within `crawlers/`:
  - `base_strategy.py` ➜ `base.py`
  - `web_strategy.py` ➜ `web.py`
  - `directory_strategy.py` ➜ `directory.py`
  - `repository_strategy.py` ➜ `repository.py`

- Update imports in orchestrator (moved to `crawler_mcp/core/orchestrator.py`):
  - `from .strategies.web_strategy import WebCrawlStrategy` ➜ `from ..crawlers.web import WebCrawlStrategy`
  - `from .strategies.directory_strategy import DirectoryCrawlStrategy, DirectoryRequest` ➜ `from ..crawlers.directory import DirectoryCrawlStrategy, DirectoryRequest`
  - `from .strategies.repository_strategy import RepositoryCrawlStrategy, RepositoryRequest` ➜ `from ..crawlers.repository import RepositoryCrawlStrategy, RepositoryRequest`
  - `from .memory_manager import MemoryManager, cleanup_memory_manager, get_memory_manager` ➜ `from .memory import MemoryManager, cleanup_memory_manager, get_memory_manager`

- Update any other references to `services.strategies.*` ➜ `crawlers.*`

---

## 4) tools filenames (simplify and rehome imports)

- Files in `tools/` remain under the top-level package, but imports change due to package/folder renames.
- File renames:
  - `crawlerr/tools/crawling_tools.py` ➜ `crawler_mcp/tools/crawling.py`
  - `crawlerr/tools/rag_tools.py` ➜ `crawler_mcp/tools/rag.py`

- Update imports within these files:
  - `from ..services import ...` ➜ `from ..core import ...`
  - Any direct imports from `services.*` ➜ `core.*`
  - Any `services.strategies.*` ➜ `crawlers.*`

- Ensure decorator-based tool registration and middleware imports still resolve.

---

## 5) middleware filenames (simplify)

- Folder path will change due to package rename: `crawlerr/middleware/` ➜ `crawler_mcp/middleware/`
- File renames:
  - `error_middleware.py` ➜ `error.py`
  - `logging_middleware.py` ➜ `logging.py`
  - `progress_middleware.py` ➜ `progress.py`

- Update imports:
  - `from ..middleware.progress_middleware import progress_middleware` ➜ `from ..middleware.progress import progress_middleware`
  - If server wires middlewares explicitly, update those references accordingly.

- Update `crawler_mcp/middleware/__init__.py` exports to:
  - `__all__ = ["ErrorHandlingMiddleware", "LoggingMiddleware", "ProgressMiddleware", "progress_middleware"]`
  - From `.error`, `.logging`, `.progress`

---

## 6) models filenames (simplify)

- Folder path will change due to package rename: `crawlerr/models/` ➜ `crawler_mcp/models/`
- File renames (3 primary model modules):
  - `crawl_models.py` ➜ `crawl.py`
  - `rag_models.py` ➜ `rag.py`
  - `source_models.py` ➜ `sources.py`

- Update imports across tools and services:
  - `from ..models.crawl_models import CrawlRequest, ...` ➜ `from ..models.crawl import CrawlRequest, ...`
  - `from ..models.rag_models import RagQuery, ...` ➜ `from ..models.rag import RagQuery, ...`
  - `from ..models.source_models import SourceFilter, SourceType` ➜ `from ..models.sources import SourceFilter, SourceType`

- Keep `crawler_mcp/models/__init__.py` minimal (optional) or export key schemas if used publicly.

---

## 7) server module and package `__init__`

- Move `crawlerr/server.py` ➜ `crawler_mcp/server.py` (same filename).
- Update `crawler_mcp/__init__.py` lazy import helpers to import from `.server`.
- Search-replace imports: `from crawlerr.server` ➜ `from crawler_mcp.server` (if any).

---

## 8) Config and settings

- Move `crawlerr/config.py` (if present) ➜ `crawler_mcp/config.py` and update imports to `from ..config import settings` ➜ path-adjusted based on new locations.

---

## 9) Update references across the codebase

Perform targeted replacements (exact, case-sensitive where appropriate):

- Packages/namespaces:
  - `crawlerr.` ➜ `crawler_mcp.`
  - `from crawlerr import` ➜ `from crawler_mcp import`

- Services ➜ core module paths:
  - `from ..services` ➜ `from ..core`
  - `from crawlerr.services` ➜ `from crawler_mcp.core`

- Strategies ➜ crawlers module paths:
  - `.strategies.` ➜ `..crawlers.` (relative)
  - `services.strategies.` ➜ `crawlers.` (absolute from package root)

- Middleware filenames:
  - `.progress_middleware` ➜ `.progress`
  - `.logging_middleware` ➜ `.logging`
  - `.error_middleware` ➜ `.error`

- Models filenames:
  - `.crawl_models` ➜ `.crawl`
  - `.rag_models` ➜ `.rag`
  - `.source_models` ➜ `.sources`

---

## 10) pyproject.toml adjustments

- Update console script:
  - Old: `crawlerr = "crawlerr.server:main"`
  - New: `crawler-mcp = "crawler_mcp.server:main"`
- Consider keeping a temporary alias:
  - `crawlerr = "crawler_mcp.server:main"` (optional during migration)

---

## 11) Verification steps

Run after each major step and at the end:

- Ruff (lint, unused imports/vars, unreachable):
  - `ruff check .`
- MyPy (type-check; catches broken imports):
  - `mypy crawler_mcp`
- Unit tests with coverage (identify paths not executing):
  - `pytest -q --cov=crawler_mcp --cov-branch`
- Import graph sanity (optional):
  - `pip install snakeviz grimp` or use `pyan` to ensure no orphan modules remain

---

## 12) Rollback plan

- Perform renames in small commits:
  1. Package rename
  2. services ➜ core + import updates
  3. strategies ➜ crawlers + import updates
  4. tools/middleware/models file renames + import updates
  5. pyproject update
- If any step breaks, revert the last commit and adjust paths/imports.

---

## 13) Post-rename checklist

- [ ] All module imports resolve (`mypy` passes)
- [ ] Lint passes (`ruff check .`)
- [ ] Tools still register (check `crawler_mcp/tools/crawling.py`, `crawler_mcp/tools/rag.py`)
- [ ] Server runs with new console script (`crawler-mcp`)
- [ ] No references to `crawlerr` or `services/strategies` remain
- [ ] Middleware imports point to simplified filenames
- [ ] Models imports updated to simplified filenames

---

## 14) Concrete mapping summary

- Package: `crawlerr/` ➜ `crawler_mcp/`
- Folders:
  - `services/` ➜ `core/`
  - `services/strategies/` ➜ `crawlers/`
  - `middleware/` ➜ `middleware/` (package prefix changes only)
  - `models/` ➜ `models/` (package prefix changes only)
  - `tools/` ➜ `tools/` (package prefix changes only)

- Files:
  - Tools: `tools/crawling_tools.py` ➜ `tools/crawling.py`
  - Tools: `tools/rag_tools.py` ➜ `tools/rag.py`
  - Middleware: `middleware/error_middleware.py` ➜ `middleware/error.py`
  - Middleware: `middleware/logging_middleware.py` ➜ `middleware/logging.py`
  - Middleware: `middleware/progress_middleware.py` ➜ `middleware/progress.py`
  - Models: `models/crawl_models.py` ➜ `models/crawl.py`
  - Models: `models/rag_models.py` ➜ `models/rag.py`
  - Models: `models/source_models.py` ➜ `models/sources.py`
  - Services: `services/crawl_orchestrator.py` ➜ `core/orchestrator.py`
  - Services: `services/embedding_service.py` ➜ `core/embeddings.py`
  - Services: `services/memory_manager.py` ➜ `core/memory.py`
  - Services: `services/rag_service.py` ➜ `core/rag.py`
  - Services: `services/source_service.py` ➜ `core/sources.py`
  - Services: `services/vector_service.py` ➜ `core/vectors.py`
  - Server: `server.py` stays `server.py` under the new package: `crawler_mcp/server.py`

---

## 15) Notes

- This plan assumes three primary models modules (`crawl_models.py`, `rag_models.py`, `source_models.py`). Adjust if there are more.
- `middleware/error.py` and `middleware/logging.py` are currently not wired in the server; either wire them or consider removal in a later cleanup.
- Keep PRs small and focused; run CI at each stage.
