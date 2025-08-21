"""
Directory crawling strategy with intelligent file processing.
"""

import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..models.crawl import (
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .base import BaseCrawlStrategy

logger = logging.getLogger(__name__)


class DirectoryRequest:
    """Request object for directory crawling."""

    def __init__(
        self,
        directory_path: str,
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        max_files: int = 1000,
    ) -> None:
        self.directory_path = directory_path
        self.file_patterns = file_patterns or ["*"]
        self.recursive = recursive
        self.max_files = max_files


class DirectoryCrawlStrategy(BaseCrawlStrategy):
    """
    Intelligent directory processing with relevance scoring and batch optimization.
    """

    async def validate_request(self, request: DirectoryRequest) -> bool:
        """Validate directory crawl request."""
        if not request.directory_path:
            self.logger.error("Directory path is required")
            return False

        directory = Path(request.directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {request.directory_path}")
            return False

        if not directory.is_dir():
            self.logger.error(f"Path is not a directory: {request.directory_path}")
            return False

        return True

    async def execute(
        self,
        request: DirectoryRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """Execute directory crawling with intelligent processing."""
        await self._initialize_managers()

        start_time = time.time()
        self.logger.info(f"Starting directory crawl: {request.directory_path}")

        await self.pre_execute_setup()

        try:
            directory = Path(request.directory_path)

            # Discover files
            if progress_callback:
                progress_callback(0, 100, "Discovering files...")

            files = await self._discover_files(directory, request)

            if not files:
                self.logger.warning(
                    f"No matching files found in {request.directory_path}"
                )
                return CrawlResult(
                    request_id=f"dir_crawl_{int(time.time())}",
                    status=CrawlStatus.COMPLETED,
                    urls=[request.directory_path],
                    pages=[],
                    errors=[],
                    statistics=CrawlStatistics(),
                )

            # Limit files if necessary
            if len(files) > request.max_files:
                self.logger.info(
                    f"Limiting to {request.max_files} files (found {len(files)})"
                )
                files = files[: request.max_files]

            # Memory check
            if self.memory_manager is None:
                raise RuntimeError("Memory manager not initialized")
            if not await self.memory_manager.can_handle_crawl(
                len(files), avg_page_size_kb=50
            ):
                self.logger.warning(
                    "System may have insufficient memory for directory processing"
                )

            # Process files with high-performance concurrent processing
            pages = []
            errors = []
            total_bytes = 0

            # Use ThreadPoolExecutor for maximum CPU utilization
            if progress_callback:
                progress_callback(
                    0, len(files), "Starting high-performance file processing..."
                )

            batch_results = await self._process_files_highly_concurrent(
                files, directory
            )

            for batch_result in batch_results:
                if isinstance(batch_result, PageContent):
                    pages.append(batch_result)
                    total_bytes += len(batch_result.content)
                elif isinstance(batch_result, Exception):
                    errors.append(str(batch_result))

            # Memory pressure check after processing
            if (
                self.memory_manager
                and await self.memory_manager.check_memory_pressure()
            ):
                self.logger.warning(
                    "Memory pressure detected during directory processing"
                )

            # Calculate statistics
            end_time = time.time()
            processing_duration = end_time - start_time
            files_per_second = (
                len(pages) / processing_duration if processing_duration > 0 else 0
            )
            # success_rate calculated as property on CrawlResult

            statistics = CrawlStatistics(
                total_pages_requested=len(files),
                total_pages_crawled=len(pages),
                total_pages_failed=len(errors),
                unique_domains=1,  # Single directory
                total_links_discovered=0,
                total_bytes_downloaded=total_bytes,
                crawl_duration_seconds=processing_duration,
                pages_per_second=files_per_second,
                average_page_size=total_bytes / len(pages) if pages else 0,
            )

            result = CrawlResult(
                request_id=f"dir_crawl_{int(time.time())}",
                status=CrawlStatus.COMPLETED,
                urls=[request.directory_path],
                pages=pages,
                errors=errors,
                statistics=statistics,
            )

            self.logger.info(
                f"Directory crawl completed: {len(pages)} files processed, "
                f"{processing_duration:.1f}s, {files_per_second:.1f} files/sec"
            )

            return result

        except Exception as e:
            self.logger.error(f"Directory crawl failed: {e}")

            return CrawlResult(
                request_id=f"dir_crawl_failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[request.directory_path],
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

        finally:
            await self.post_execute_cleanup()

    async def _discover_files(
        self, directory: Path, request: DirectoryRequest
    ) -> list[Path]:
        """Discover files matching patterns with relevance scoring."""
        files: list[Path] = []

        try:
            if request.recursive:
                # Recursive search with pattern matching
                for pattern in request.file_patterns:
                    files.extend(directory.rglob(pattern))
            else:
                # Non-recursive search
                for pattern in request.file_patterns:
                    files.extend(directory.glob(pattern))

            # Remove duplicates and filter
            unique_files = list(set(files))

            # Filter out non-files, empty files, and unreadable files
            valid_files = []
            for file_path in unique_files:
                if self._is_valid_file(file_path):
                    valid_files.append(file_path)

            # Sort by relevance (file extension, size, modification time)
            scored_files = await self._score_files(valid_files, directory)

            self.logger.info(
                f"Discovered {len(scored_files)} valid files from {len(unique_files)} candidates"
            )

            return [file_info[0] for file_info in scored_files]

        except Exception as e:
            self.logger.error(f"File discovery failed: {e}")
            return []

    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for processing."""
        try:
            if not file_path.is_file():
                return False

            # Skip empty files
            if file_path.stat().st_size == 0:
                return False

            # Skip very large files (>10MB) to prevent memory issues
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return False

            # Skip binary files by extension
            binary_extensions = {
                ".exe",
                ".dll",
                ".so",
                ".dylib",
                ".bin",
                ".obj",
                ".o",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".ico",
                ".tiff",
                ".mp3",
                ".mp4",
                ".avi",
                ".mov",
                ".wmv",
                ".flv",
                ".wav",
                ".zip",
                ".tar",
                ".gz",
                ".bz2",
                ".7z",
                ".rar",
                ".pdf",
                ".doc",
                ".docx",
                ".xls",
                ".xlsx",
                ".ppt",
                ".pptx",
            }

            if file_path.suffix.lower() in binary_extensions:
                return False

            # Basic readability test
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                    f.read(100)  # Try to read first 100 chars
                return True
            except (OSError, UnicodeDecodeError):
                return False

        except (OSError, PermissionError):
            return False

    async def _score_files(
        self, files: list[Path], base_directory: Path
    ) -> list[tuple[Path, float]]:
        """Score files by relevance for intelligent prioritization."""
        scored_files = []

        # File extension priorities (higher = more important)
        extension_scores = {
            ".py": 10,
            ".js": 9,
            ".ts": 9,
            ".jsx": 8,
            ".tsx": 8,
            ".java": 8,
            ".cpp": 8,
            ".c": 8,
            ".h": 7,
            ".hpp": 7,
            ".go": 8,
            ".rs": 8,
            ".php": 7,
            ".rb": 7,
            ".cs": 7,
            ".html": 6,
            ".css": 5,
            ".scss": 5,
            ".sass": 5,
            ".md": 6,
            ".rst": 6,
            ".txt": 4,
            ".yaml": 5,
            ".yml": 5,
            ".json": 5,
            ".xml": 4,
            ".csv": 3,
            ".log": 2,
            ".sh": 6,
            ".bat": 6,
            ".ps1": 6,
            ".sql": 5,
            ".dockerfile": 6,
            ".gitignore": 3,
        }

        for file_path in files:
            try:
                score = 0.0

                # Extension score
                ext_score = extension_scores.get(file_path.suffix.lower(), 1)
                score += ext_score * 10

                # Size score (prefer moderate-sized files)
                size_kb = file_path.stat().st_size / 1024
                if 1 <= size_kb <= 100:  # 1KB to 100KB is ideal
                    score += 20
                elif 100 < size_kb <= 500:  # 100KB to 500KB is good
                    score += 10
                elif size_kb > 1000:  # Over 1MB is less preferred
                    score -= 10

                # Depth score (prefer files closer to root)
                try:
                    relative_path = file_path.relative_to(base_directory)
                    depth = len(relative_path.parts) - 1
                    score += max(0, 10 - depth * 2)
                except ValueError:
                    pass  # File not under base directory

                # Name patterns (prefer important files)
                filename_lower = file_path.name.lower()
                if any(
                    pattern in filename_lower
                    for pattern in ["readme", "main", "index", "app"]
                ):
                    score += 15
                elif any(
                    pattern in filename_lower
                    for pattern in ["config", "setting", "env"]
                ):
                    score += 10
                elif any(
                    pattern in filename_lower for pattern in ["test", "spec", "example"]
                ):
                    score += 5
                elif any(
                    pattern in filename_lower
                    for pattern in ["temp", "tmp", "cache", "backup"]
                ):
                    score -= 10

                # Modification time (prefer recently modified)
                mtime = file_path.stat().st_mtime
                age_days = (time.time() - mtime) / (24 * 3600)
                if age_days <= 7:  # Modified within a week
                    score += 5
                elif age_days <= 30:  # Modified within a month
                    score += 2

                scored_files.append((file_path, score))

            except (OSError, PermissionError):
                # If we can't stat the file, give it a low score
                scored_files.append((file_path, 1.0))

        # Sort by score (highest first)
        scored_files.sort(key=lambda x: x[1], reverse=True)

        return scored_files

    async def _process_files_highly_concurrent(
        self, file_paths: list[Path], base_directory: Path
    ) -> list[PageContent | Exception]:
        """Process files with full CPU utilization using ThreadPoolExecutor."""
        from ..config import settings

        # Use configured thread count (default 16 for i7-13700k)
        max_workers = getattr(settings, "file_processing_threads", 16)

        self.logger.info(
            f"Processing {len(file_paths)} files with {max_workers} threads"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_running_loop()

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
            processed_results: list[PageContent | Exception] = []
            for result in results:
                if isinstance(result, (PageContent, Exception)):
                    processed_results.append(result)
                else:
                    # Handle any unexpected return types
                    processed_results.append(
                        Exception(f"Unexpected result type: {type(result)}")
                    )

            return processed_results

    def _process_single_file_sync(
        self, file_path: Path, base_directory: Path
    ) -> PageContent | Exception:
        """Synchronous file processing for thread pool."""
        try:
            return self._process_single_file_sync_impl(file_path, base_directory)
        except Exception as e:
            return Exception(f"Error processing {file_path}: {e}")

    def _process_single_file_sync_impl(
        self, file_path: Path, base_directory: Path
    ) -> PageContent:
        """Synchronous implementation of file processing."""
        try:
            # Read file content
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Get relative path for display
            try:
                relative_path = file_path.relative_to(base_directory)
                display_path = str(relative_path)
            except ValueError:
                display_path = str(file_path)

            # Get file stats
            stat = file_path.stat()

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "relative_path": display_path,
                "file_extension": file_path.suffix,
                "file_size": stat.st_size,
                "modification_time": stat.st_mtime,
                "word_count": len(content.split()),
                "line_count": content.count("\n") + 1,
            }

            # Create PageContent
            page_content = PageContent(
                url=f"file://{file_path}",
                title=file_path.name,
                content=content,
                html=None,  # Files don't have HTML
                markdown=content if file_path.suffix.lower() == ".md" else None,
                links=[],  # Could extract file references in future
                images=[],
                metadata=metadata,
                timestamp=datetime.fromtimestamp(time.time()),
                word_count=int(str(metadata.get("word_count", 0))),
            )

            return page_content

        except Exception as e:
            raise Exception(f"Failed to process file {file_path}: {e}") from e
