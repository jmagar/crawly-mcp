"""
Memory management with adaptive dispatching and optimized thresholds.
"""

import asyncio
import gc
import logging
import time
from collections.abc import Callable
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "psutil not available - memory monitoring disabled"
    )

from ..config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Optimized memory manager with 70% threshold to prevent crawling pauses.
    Provides predictive memory management and aggressive cleanup strategies.
    """

    def __init__(self) -> None:
        self.threshold_percent = settings.crawl_memory_threshold  # 70.0 instead of 90.0
        self.available = PSUTIL_AVAILABLE

        # Memory tracking
        self._last_check = 0.0
        self._cached_memory_info: dict[str, Any] = {}
        self._memory_history: list[float] = []
        self._cleanup_callbacks: list[Callable[[], None]] = []

        # Performance metrics
        self._total_checks = 0
        self._cleanup_events = 0
        self._bytes_cleaned = 0

    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called during memory cleanup."""
        self._cleanup_callbacks.append(callback)

    async def check_memory_pressure(self, force_check: bool = False) -> bool:
        """
        Check if system is under memory pressure.

        Args:
            force_check: Force fresh check instead of using cache

        Returns:
            True if memory pressure detected and cleanup was performed
        """
        if not self.available:
            return False

        current_time = time.time()

        # Use cached result if recent (within 5 seconds) and not forcing
        if not force_check and current_time - self._last_check < 5.0:
            cached_percent = self._cached_memory_info.get("memory_percent", 0)
            return bool(cached_percent > self.threshold_percent)

        self._last_check = current_time
        self._total_checks += 1

        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Update cache
            self._cached_memory_info = {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory_percent,
                "memory_pressure": memory_percent > self.threshold_percent,
                "last_updated": current_time,
            }

            # Track memory history for prediction
            self._memory_history.append(memory_percent)
            if len(self._memory_history) > 20:  # Keep last 20 readings
                self._memory_history.pop(0)

            # Check if cleanup needed
            if memory_percent > self.threshold_percent:
                logger.warning(
                    f"Memory pressure detected: {memory_percent:.1f}% > {self.threshold_percent}%"
                )
                await self._cleanup_memory()
                return True

            # Predictive cleanup at 60% if memory is rising rapidly
            elif memory_percent > 60.0 and self._is_memory_rising():
                logger.info(
                    f"Predictive memory cleanup at {memory_percent:.1f}% (rising trend detected)"
                )
                await self._light_cleanup()

            return False

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False

    def _is_memory_rising(self) -> bool:
        """Check if memory usage is rising rapidly."""
        if len(self._memory_history) < 5:
            return False

        # Check if memory increased by more than 10% in last 5 readings
        recent = self._memory_history[-5:]
        return recent[-1] - recent[0] > 10.0

    async def _cleanup_memory(self) -> None:
        """Perform aggressive memory cleanup."""
        logger.info("Performing memory cleanup...")
        start_time = time.time()
        self._cleanup_events += 1

        # Get memory before cleanup
        if self.available:
            initial_memory = psutil.virtual_memory().used

        # Call registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

        # Get memory after cleanup
        if self.available:
            final_memory = psutil.virtual_memory().used
            bytes_freed = initial_memory - final_memory
            self._bytes_cleaned += bytes_freed
            mb_freed = bytes_freed / (1024**2)

            cleanup_time = time.time() - start_time
            logger.info(
                f"Memory cleanup completed: {mb_freed:.1f}MB freed in {cleanup_time:.2f}s"
            )

    async def _light_cleanup(self) -> None:
        """Perform light memory cleanup without aggressive measures."""
        logger.debug("Performing light memory cleanup...")

        # Just run garbage collection
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"Light cleanup: garbage collection freed {collected} objects")

    def get_memory_info(self) -> dict[str, Any]:
        """Get current memory information."""
        if not self.available:
            return {"available": False, "error": "psutil not available"}

        # Force fresh check (don't await - let it run in background)
        asyncio.create_task(self.check_memory_pressure(force_check=True))  # noqa: RUF006

        return {
            **self._cached_memory_info,
            "threshold_percent": self.threshold_percent,
            "memory_history": self._memory_history.copy(),
            "is_rising": self._is_memory_rising(),
        }

    def estimate_crawl_memory(
        self, page_count: int, avg_page_size_kb: int = 100
    ) -> float:
        """
        Estimate memory requirements for a crawl operation.

        Args:
            page_count: Number of pages to crawl
            avg_page_size_kb: Average page size in KB

        Returns:
            Estimated memory usage in MB
        """
        # Base crawler overhead
        base_overhead_mb = 50  # Browser + crawler overhead

        # Page content memory (with processing overhead factor of 3x)
        content_mb = (page_count * avg_page_size_kb * 3) / 1024

        # Browser memory (single AsyncWebCrawler instance)
        browser_mb = 100  # ~100MB per browser instance

        total_mb = base_overhead_mb + content_mb + browser_mb

        logger.debug(f"Memory estimate: {page_count} pages = {total_mb:.1f}MB")
        return total_mb

    async def can_handle_crawl(
        self, page_count: int, avg_page_size_kb: int = 100
    ) -> bool:
        """
        Check if system can handle a crawl without memory pressure.

        Args:
            page_count: Number of pages to crawl
            avg_page_size_kb: Average page size in KB

        Returns:
            True if crawl can proceed without memory issues
        """
        if not self.available:
            return True  # Assume OK if we can't check

        estimated_mb = self.estimate_crawl_memory(page_count, avg_page_size_kb)

        # Get current memory state
        await self.check_memory_pressure(force_check=True)
        available_gb = float(self._cached_memory_info.get("memory_available_gb", 0))
        available_mb = available_gb * 1024

        # Keep 25% buffer for system and other processes
        usable_mb = available_mb * 0.75

        can_handle = estimated_mb <= usable_mb

        if not can_handle:
            logger.warning(
                f"Crawl may cause memory pressure: need {estimated_mb:.1f}MB, "
                f"available {usable_mb:.1f}MB (with buffer)"
            )
        else:
            logger.debug(
                f"Memory check passed: need {estimated_mb:.1f}MB, available {usable_mb:.1f}MB"
            )

        return can_handle

    def get_stats(self) -> dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "available": self.available,
            "threshold_percent": self.threshold_percent,
            "total_checks": self._total_checks,
            "cleanup_events": self._cleanup_events,
            "bytes_cleaned_mb": self._bytes_cleaned / (1024**2),
            "cleanup_callbacks_registered": len(self._cleanup_callbacks),
            **self._cached_memory_info,
        }

    async def optimize_for_crawl(self) -> None:
        """Optimize memory settings for crawling performance."""
        logger.info("Optimizing memory for crawling...")

        # Pre-cleanup before starting
        if await self.check_memory_pressure(force_check=True):
            logger.info("Initial cleanup completed")

        # Set more aggressive garbage collection
        import gc

        gc.set_threshold(400, 5, 5)  # More frequent GC during crawling
        logger.debug("Set aggressive garbage collection thresholds")


# Global memory manager instance
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def cleanup_memory_manager() -> None:
    """Cleanup the global memory manager."""
    global _memory_manager
    _memory_manager = None
