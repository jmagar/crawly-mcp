"""
Browser session pool management with connection reuse and caching optimizations.
"""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode

from ..config import settings

logger = logging.getLogger(__name__)


class BrowserPool:
    """
    High-performance browser pool with warm initialization and session reuse.
    Optimized for RTX 4070 + i7-13700K hardware configuration.
    """

    def __init__(
        self,
        pool_size: int | None = None,
        warm_pool: bool | None = None,
        recycle_after: int | None = None,
    ) -> None:
        self.pool_size = pool_size or settings.browser_pool_size
        self.warm_pool_enabled = (
            warm_pool if warm_pool is not None else settings.browser_warm_pool
        )
        self.recycle_after = recycle_after or settings.browser_recycle_after

        # Browser pool queue
        self.available_browsers: asyncio.Queue[AsyncWebCrawler] = asyncio.Queue(
            maxsize=self.pool_size
        )
        self.active_browsers: set[AsyncWebCrawler] = set()
        self.total_browsers = 0

        # Session caching for connection reuse
        self.domain_sessions: dict[str, AsyncWebCrawler] = {}
        self.session_usage: dict[str, int] = {}
        self.session_created: dict[str, float] = {}

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the browser pool with warm browsers if enabled."""
        if self._initialized:
            return

        logger.info(
            f"Initializing browser pool (size: {self.pool_size}, warm: {self.warm_pool_enabled})"
        )

        if self.warm_pool_enabled:
            await self._warm_pool()

        self._initialized = True

    async def _warm_pool(self) -> None:
        """Pre-initialize browsers for immediate availability."""
        logger.info("Warming browser pool...")

        for i in range(self.pool_size):
            try:
                browser = await self._create_browser()
                await self.available_browsers.put(browser)
                logger.debug(f"Warmed browser {i + 1}/{self.pool_size}")
            except Exception as e:
                logger.warning(f"Failed to warm browser {i + 1}: {e}")

    async def _create_browser(self) -> AsyncWebCrawler:
        """Create a new optimized browser instance."""
        config = BrowserConfig(
            headless=settings.crawl_headless,
            browser_type=settings.crawl_browser,
            viewport={"width": 1920, "height": 1080},
            # Performance optimizations
            cache_mode=CacheMode.ENABLED
            if settings.crawl_enable_caching
            else CacheMode.BYPASS,
            persistent_context=True,  # Enable session persistence
            # GPU acceleration flags for RTX 4070
            extra_args=self._get_optimized_chrome_flags(),
            # Resource blocking for performance
            block_images=settings.crawl_block_images,
            block_media=settings.crawl_block_media,
            block_stylesheets=settings.crawl_block_stylesheets,
            block_fonts=settings.crawl_block_fonts,
        )

        crawler = AsyncWebCrawler(config=config)
        await crawler.start()

        self.total_browsers += 1
        logger.debug(f"Created new browser (total: {self.total_browsers})")
        return crawler

    def _get_optimized_chrome_flags(self) -> list[str]:
        """Get optimized Chrome flags for RTX 4070."""
        base_flags = settings.chrome_gpu_flags.split()
        advanced_flags = settings.chrome_advanced_gpu_flags.split()

        # Additional performance flags
        performance_flags = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI,VizDisplayCompositor",
            "--enable-features=NetworkService,NetworkServiceInProcess",
            "--memory-pressure-off",
            "--max_old_space_size=4096",
        ]

        return base_flags + advanced_flags + performance_flags

    async def get_session(self, url: str, force_new: bool = False) -> AsyncWebCrawler:
        """
        Get a browser session, preferring cached sessions for the same domain.

        Args:
            url: Target URL for domain-based caching
            force_new: Force creation of new session instead of reusing

        Returns:
            AsyncWebCrawler instance ready for use
        """
        domain = urlparse(url).netloc

        # Check for existing domain session that hasn't exceeded recycle limit
        if not force_new and domain in self.domain_sessions:
            usage_count = self.session_usage.get(domain, 0)
            session_age = time.time() - self.session_created.get(domain, 0)

            if (
                usage_count < self.recycle_after
                and session_age < settings.session_cache_ttl
            ):
                self.session_usage[domain] = usage_count + 1
                logger.debug(
                    f"Reusing session for {domain} (usage: {usage_count + 1}/{self.recycle_after})"
                )
                return self.domain_sessions[domain]
            else:
                # Session needs recycling
                await self._recycle_domain_session(domain)

        # Get browser from pool or create new one
        browser = await self._acquire_browser()

        # Cache session by domain
        self.domain_sessions[domain] = browser
        self.session_usage[domain] = 1
        self.session_created[domain] = time.time()

        logger.debug(f"Created new session for {domain}")
        return browser

    async def _acquire_browser(self) -> AsyncWebCrawler:
        """Acquire a browser from the pool or create a new one."""
        try:
            # Try to get from pool first (non-blocking)
            browser = self.available_browsers.get_nowait()
            self.active_browsers.add(browser)
            return browser
        except asyncio.QueueEmpty:
            # Pool empty, create new browser
            if self.total_browsers >= self.pool_size * 2:  # Soft limit
                logger.warning(
                    "Browser pool exhausted, waiting for available browser..."
                )
                browser = await self.available_browsers.get()
                self.active_browsers.add(browser)
                return browser
            else:
                browser = await self._create_browser()
                self.active_browsers.add(browser)
                return browser

    async def release_browser(self, browser: AsyncWebCrawler) -> None:
        """Release a browser back to the pool."""
        if browser in self.active_browsers:
            self.active_browsers.remove(browser)

        try:
            # Return to pool if there's space
            self.available_browsers.put_nowait(browser)
        except asyncio.QueueFull:
            # Pool full, close this browser
            await browser.close()
            self.total_browsers -= 1
            logger.debug(f"Closed excess browser (total: {self.total_browsers})")

    async def _recycle_domain_session(self, domain: str) -> None:
        """Recycle a domain session that has exceeded limits."""
        if domain in self.domain_sessions:
            old_browser = self.domain_sessions[domain]
            await self.release_browser(old_browser)

            del self.domain_sessions[domain]
            del self.session_usage[domain]
            del self.session_created[domain]

            logger.debug(f"Recycled session for {domain}")

    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions based on TTL."""
        current_time = time.time()
        expired_domains = []

        for domain, created_time in self.session_created.items():
            if current_time - created_time > settings.session_cache_ttl:
                expired_domains.append(domain)

        for domain in expired_domains:
            await self._recycle_domain_session(domain)

        if expired_domains:
            logger.debug(f"Cleaned up {len(expired_domains)} expired sessions")

    async def get_stats(self) -> dict[str, Any]:
        """Get browser pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available_browsers": self.available_browsers.qsize(),
            "active_browsers": len(self.active_browsers),
            "total_browsers": self.total_browsers,
            "cached_sessions": len(self.domain_sessions),
            "session_stats": {
                domain: {
                    "usage_count": self.session_usage.get(domain, 0),
                    "age_seconds": time.time() - self.session_created.get(domain, 0),
                }
                for domain in self.domain_sessions
            },
        }

    async def shutdown(self) -> None:
        """Shutdown all browsers and cleanup resources."""
        logger.info("Shutting down browser pool...")

        # Close all active browsers
        for browser in self.active_browsers.copy():
            try:
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing active browser: {e}")

        # Close all available browsers
        while not self.available_browsers.empty():
            try:
                browser = self.available_browsers.get_nowait()
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing pooled browser: {e}")

        # Close cached sessions
        for browser in self.domain_sessions.values():
            try:
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing cached session: {e}")

        # Clear all tracking
        self.active_browsers.clear()
        self.domain_sessions.clear()
        self.session_usage.clear()
        self.session_created.clear()
        self.total_browsers = 0

        logger.info("Browser pool shutdown complete")


# Global browser pool instance
_browser_pool: BrowserPool | None = None


async def get_browser_pool() -> BrowserPool:
    """Get or create the global browser pool instance."""
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool()
        await _browser_pool.initialize()
    return _browser_pool


async def cleanup_browser_pool() -> None:
    """Cleanup the global browser pool."""
    global _browser_pool
    if _browser_pool:
        await _browser_pool.shutdown()
        _browser_pool = None
