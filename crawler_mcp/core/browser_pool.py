"""
High-performance browser pool for concurrent operations.

Optimized for i7-13700k + RTX 4070 hardware configuration.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HighPerformanceBrowserPool:
    """Browser pool optimized for i7-13700k + RTX 4070."""

    def __init__(self, pool_size: int = 8):
        self.pool_size = pool_size
        self.browsers: list[Any] = []  # Will be AsyncWebCrawler instances
        self.available_browsers: asyncio.Queue[Any] = asyncio.Queue(maxsize=pool_size)
        self.is_initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize browser pool with hardware-optimized config."""
        if self.is_initialized:
            return

        try:
            # Import here to avoid circular imports
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            # Hardware-optimized browser configuration
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

            self.logger.info(
                f"Initializing browser pool with {self.pool_size} browsers"
            )

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
                raise RuntimeError("Failed to initialize any browsers")

            self.is_initialized = True
            self.logger.info(
                f"Browser pool initialized successfully with {len(self.browsers)} browsers"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize browser pool: {e}")
            await self.cleanup()
            raise

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
        except Exception as e:
            self.logger.error(f"Failed to return browser to pool: {e}")

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

    async def __aenter__(self) -> "HighPerformanceBrowserPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    @property
    def available_count(self) -> int:
        """Get number of available browsers in pool."""
        return self.available_browsers.qsize()

    @property
    def total_browsers(self) -> int:
        """Get total number of browsers in pool."""
        return len(self.browsers)


# Global browser pool instance
_browser_pool: HighPerformanceBrowserPool | None = None


async def get_browser_pool(pool_size: int = 8) -> HighPerformanceBrowserPool:
    """Get or create the global browser pool."""
    global _browser_pool

    if _browser_pool is None:
        _browser_pool = HighPerformanceBrowserPool(pool_size=pool_size)

    return _browser_pool


async def cleanup_browser_pool() -> None:
    """Cleanup the global browser pool."""
    global _browser_pool

    if _browser_pool is not None:
        await _browser_pool.cleanup()
        _browser_pool = None
