"""
Advanced caching implementations with TTL and LRU eviction policies.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache(Generic[T]):
    """
    Time-To-Live cache with automatic expiration.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 900,  # 15 minutes default
        cleanup_interval: int = 60,  # Cleanup every minute
    ):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live for cache entries in seconds
            cleanup_interval: Interval for cleanup task in seconds
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cleanup_interval = cleanup_interval

        # Use OrderedDict for LRU behavior
        self.cache: OrderedDict[str, tuple[T, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    async def start(self) -> None:
        """Start the cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug(
                f"Started TTL cache cleanup task (interval: {self.cleanup_interval}s)"
            )

    async def stop(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.debug("Stopped TTL cache cleanup task")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        async with self._lock:
            now = datetime.utcnow()
            expired_keys = []

            for key, (_value, timestamp) in self.cache.items():
                if now - timestamp > self.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                self.expirations += 1

            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired cache entries")

    async def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]

                # Check if expired
                if datetime.utcnow() - timestamp > self.ttl:
                    del self.cache[key]
                    self.expirations += 1
                    self.misses += 1
                    return None

                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return value

            self.misses += 1
            return None

    async def set(self, key: str, value: T) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Remove oldest if at capacity (LRU)
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1

            # Add or update entry
            self.cache[key] = (value, datetime.utcnow())
            self.cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl.total_seconds(),
        }


class LRUCache(Generic[T]):
    """
    Least Recently Used cache with efficient eviction.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, T] = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    async def set(self, key: str, value: T) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1

            # Add or update entry
            self.cache[key] = value
            self.cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        }


class EmbeddingCache:
    """
    Specialized cache for embedding results with content-based keys.
    """

    def __init__(self, max_size: int = 5000, ttl_minutes: int = 30):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_minutes: Time-to-live in minutes
        """
        self.cache = TTLCache[list[float]](
            max_size=max_size,
            ttl_seconds=ttl_minutes * 60,
        )
        self._started = False

    async def start(self) -> None:
        """Start the cache cleanup task."""
        if not self._started:
            await self.cache.start()
            self._started = True

    async def stop(self) -> None:
        """Stop the cache cleanup task."""
        if self._started:
            await self.cache.stop()
            self._started = False

    def _generate_key(self, text: str, model: str | None = None) -> str:
        """Generate cache key from text content."""
        content = f"{model or 'default'}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, text: str, model: str | None = None) -> list[float] | None:
        """
        Get embedding from cache.

        Args:
            text: Text that was embedded
            model: Model name (optional)

        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text, model)
        return await self.cache.get(key)

    async def set(
        self, text: str, embedding: list[float], model: str | None = None
    ) -> None:
        """
        Cache an embedding.

        Args:
            text: Text that was embedded
            embedding: The embedding vector
            model: Model name (optional)
        """
        key = self._generate_key(text, model)
        await self.cache.set(key, embedding)

    async def get_batch(
        self, texts: list[str], model: str | None = None
    ) -> tuple[dict[int, list[float]], list[int]]:
        """
        Get multiple embeddings from cache.

        Args:
            texts: List of texts
            model: Model name (optional)

        Returns:
            Tuple of (cached embeddings by index, indices of cache misses)
        """
        cached = {}
        misses = []

        for i, text in enumerate(texts):
            embedding = await self.get(text, model)
            if embedding is not None:
                cached[i] = embedding
            else:
                misses.append(i)

        return cached, misses

    async def set_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str | None = None,
    ) -> None:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            model: Model name (optional)
        """
        for text, embedding in zip(texts, embeddings, strict=False):
            await self.set(text, embedding, model)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class QueryResultCache:
    """
    Cache for RAG query results with intelligent key generation.
    """

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 15):
        """
        Initialize query result cache.

        Args:
            max_size: Maximum number of results to cache
            ttl_minutes: Time-to-live in minutes
        """
        self.cache = TTLCache[Any](
            max_size=max_size,
            ttl_seconds=ttl_minutes * 60,
        )
        self._started = False

    async def start(self) -> None:
        """Start the cache cleanup task."""
        if not self._started:
            await self.cache.start()
            self._started = True

    async def stop(self) -> None:
        """Stop the cache cleanup task."""
        if self._started:
            await self.cache.stop()
            self._started = False

    def _generate_key(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        source_filters: list[str] | None = None,
        rerank: bool = False,
    ) -> str:
        """Generate cache key from query parameters."""
        # Sort source filters for consistent keys
        sorted_filters = sorted(source_filters) if source_filters else []

        # Normalize numeric and boolean inputs for stable keys
        normalized_min_score = f"{min_score:.8g}"  # up to 8 significant digits
        normalized_rerank = "1" if rerank else "0"
        key_parts = [
            query,
            str(int(limit)),
            normalized_min_score,
            ",".join(sorted_filters),
            normalized_rerank,
        ]

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        source_filters: list[str] | None = None,
        rerank: bool = False,
    ) -> Any | None:
        """Get cached query result."""
        key = self._generate_key(query, limit, min_score, source_filters, rerank)
        return await self.cache.get(key)

    async def set(
        self,
        result: Any,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        source_filters: list[str] | None = None,
        rerank: bool = False,
    ) -> None:
        """Cache a query result."""
        key = self._generate_key(query, limit, min_score, source_filters, rerank)
        await self.cache.set(key, result)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global cache instances
_embedding_cache: EmbeddingCache | None = None
_query_cache: QueryResultCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_query_cache() -> QueryResultCache:
    """Get or create the global query result cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResultCache()
    return _query_cache


async def initialize_caches(ctx: Any | None = None) -> None:
    """Initialize all global caches."""
    embedding_cache = get_embedding_cache()
    query_cache = get_query_cache()

    await embedding_cache.start()
    await query_cache.start()
    if ctx:
        # client-facing progress
        ctx.info("Caches initialized (embedding, query)")

    logger.info("Caches initialized with TTL and LRU eviction")


async def close_caches(ctx: Any | None = None) -> None:
    """Close all global caches."""
    global _embedding_cache, _query_cache

    if _embedding_cache:
        await _embedding_cache.stop()
        _embedding_cache = None

    if _query_cache:
        await _query_cache.stop()
        _query_cache = None

    if ctx:
        ctx.info("Caches closed")
    logger.info("Caches closed")
