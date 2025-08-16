"""
Service for RAG (Retrieval-Augmented Generation) operations.
"""

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse

from fastmcp.exceptions import ToolError

from ..config import settings
from ..models.crawl import CrawlResult
from ..models.rag import DocumentChunk, RagQuery, RagResult, SearchMatch
from .embeddings import EmbeddingService
from .vectors import VectorService

logger = logging.getLogger(__name__)

# Approximate word-to-token ratio for different tokenizers
WORD_TO_TOKEN_RATIO = 1.3  # General estimate for English text
QWEN3_WORD_TO_TOKEN_RATIO = (
    1.4  # More accurate for Qwen3 tokenizer based on empirical testing
)


class QueryCache:
    """
    Simple in-memory cache for RAG query results with TTL support.
    """

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 15):
        self.cache: dict[str, tuple[RagResult, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)

    def _generate_cache_key(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
    ) -> str:
        """Generate a deterministic cache key from query parameters."""
        key_components = [
            query.strip().lower(),
            str(limit),
            str(min_score),
            str(sorted(source_filters) if source_filters else ""),
            str(rerank),
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
    ) -> RagResult | None:
        """Get cached result if it exists and hasn't expired."""
        cache_key = self._generate_cache_key(
            query, limit, min_score, source_filters, rerank
        )

        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.utcnow() - timestamp < self.ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                logger.debug(f"Cache expired for query: {query[:50]}...")

        return None

    def put(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
        result: RagResult,
    ) -> None:
        """Cache a result with current timestamp."""
        cache_key = self._generate_cache_key(
            query, limit, min_score, source_filters, rerank
        )

        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug("Cache full, removed oldest entry")

        self.cache[cache_key] = (result, datetime.utcnow())
        logger.debug(f"Cached result for query: {query[:50]}...")

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = sum(
            1 for _, timestamp in self.cache.values() if now - timestamp < self.ttl
        )

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60,
        }


def find_paragraph_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find paragraph break boundary."""
    paragraph_breaks = [
        i for i, char in enumerate(search_text) if search_text[i : i + 2] == "\n\n"
    ]
    suitable_breaks = [
        b for b in paragraph_breaks if ideal_end - 100 <= b <= ideal_end + 100
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_sentence_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find sentence ending boundary."""
    sentence_patterns = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    sentence_breaks = []
    for pattern in sentence_patterns:
        sentence_breaks.extend(
            [
                i + len(pattern)
                for i in range(len(search_text) - len(pattern))
                if search_text[i : i + len(pattern)] == pattern
            ]
        )
    suitable_breaks = [
        b for b in sentence_breaks if ideal_end - 50 <= b <= ideal_end + 50
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_line_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find line break boundary."""
    line_breaks = [i + 1 for i, char in enumerate(search_text) if char == "\n"]
    suitable_breaks = [b for b in line_breaks if ideal_end - 30 <= b <= ideal_end + 30]
    return max(suitable_breaks) if suitable_breaks else None


def find_word_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find word boundary."""
    word_breaks = [i + 1 for i, char in enumerate(search_text) if char == " "]
    suitable_breaks = [b for b in word_breaks if ideal_end - 20 <= b <= ideal_end + 20]
    return max(suitable_breaks) if suitable_breaks else None


class RagService:
    """
    Service for RAG operations combining embedding generation and vector search.
    Uses singleton pattern to keep models loaded in memory for optimal performance.
    """

    _instance = None
    _initialized = False
    _context_count = 0
    _lock = None

    def __new__(cls) -> "RagService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Prevent re-initialization of models
        if self._initialized:
            return

        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

        # Initialize query cache for performance optimization
        self.query_cache = QueryCache(max_size=1000, ttl_minutes=15)

        # Add missing attributes from settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.tokenizer = None
        self.tokenizer_type = "tiktoken"  # Default tokenizer type

        # Initialize tokenizer
        try:
            import tiktoken

            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizer_type = "tiktoken"
        except ImportError:
            logger.warning("tiktoken not available, using character-based chunking")
            self.tokenizer_type = "character"

        # Initialize Qwen3 reranker with GPU optimization
        self.reranker = None
        self.reranker_type = "none"

        if settings.reranker_enabled:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                logger.warning(
                    "sentence-transformers not installed. Reranking disabled: %s", e
                )
            else:
                try:
                    # Force GPU usage for reranker if available
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.reranker = CrossEncoder(settings.reranker_model, device=device)
                    self.reranker_type = "qwen3"
                    logger.info(
                        "Using Qwen3 reranker: %s on device: %s",
                        settings.reranker_model,
                        device,
                    )
                except (OSError, ValueError) as e:
                    logger.warning(
                        "Failed to load Qwen3 reranker %s: %s",
                        settings.reranker_model,
                        e,
                    )
                    if settings.reranker_fallback_to_custom:
                        self.reranker_type = "custom"
                        logger.info("Using custom reranking algorithm as fallback")
                    else:
                        self.reranker_type = "none"
                        logger.info("Reranking disabled due to model loading failure")

        # Initialize lock for context manager reference counting
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()

        # Mark as initialized to prevent reloading models
        RagService._initialized = True

    async def __aenter__(self) -> "RagService":
        """Async context manager entry with reference counting."""
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()
        async with RagService._lock:
            self._context_count += 1
            # Only initialize underlying services on first context entry
            if self._context_count == 1:
                logger.debug("Initializing underlying services (first context)")
                await self.embedding_service.__aenter__()
                await self.vector_service.__aenter__()
            else:
                logger.debug(f"Reusing services (context count: {self._context_count})")
        return self

    async def __aexit__(
        self,
        exc_type: type,
        exc_val: Exception,
        exc_tb: object,
    ) -> None:
        """Async context manager exit with reference counting."""
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()
        async with RagService._lock:
            self._context_count -= 1
            # Only close underlying services when last context exits
            if self._context_count == 0:
                logger.debug("Closing underlying services (last context)")
                await self.embedding_service.__aexit__(exc_type, exc_val, exc_tb)
                await self.vector_service.__aexit__(exc_type, exc_val, exc_tb)
            else:
                logger.debug(
                    f"Keeping services alive (context count: {self._context_count})"
                )

    async def close(self) -> None:
        """Close all services."""
        await self.embedding_service.close()
        await self.vector_service.close()

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all dependent services.

        Returns:
            Dictionary with service health status
        """
        return {
            "embedding_service": await self.embedding_service.health_check(),
            "vector_service": await self.vector_service.health_check(),
        }

    def _find_paragraph_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find paragraph break boundary."""
        return find_paragraph_boundary(search_text, ideal_end)

    def _find_sentence_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find sentence ending boundary."""
        return find_sentence_boundary(search_text, ideal_end)

    # Deduplication helper methods
    def _generate_deterministic_id(self, url: str, chunk_index: int) -> str:
        """
        Generate deterministic ID from URL and chunk index.

        Args:
            url: Source URL
            chunk_index: Index of the chunk within the document

        Returns:
            Deterministic UUID string
        """
        import uuid

        normalized_url = self._normalize_url(url)
        id_string = f"{normalized_url}:{chunk_index}"
        # Generate a deterministic UUID from the hash
        hash_bytes = hashlib.sha256(id_string.encode()).digest()[:16]
        # Create UUID from the first 16 bytes of the hash
        deterministic_uuid = uuid.UUID(bytes=hash_bytes)
        return str(deterministic_uuid)

    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate SHA256 hash of content for change detection.

        Args:
            content: Text content to hash

        Returns:
            SHA256 hash hexdigest string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistent hashing.

        Normalizes:
        - Protocol (http -> https)
        - Removes trailing slashes
        - Sorts query parameters
        - Removes fragments

        Args:
            url: URL to normalize

        Returns:
            Normalized URL string
        """
        parsed = urlparse(url)

        # Normalize protocol to https
        scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme

        # Remove trailing slash from path
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters for consistency
        if parsed.query:
            from urllib.parse import parse_qs, urlencode

            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query = urlencode(sorted_params, doseq=True)
        else:
            query = ""

        # Reconstruct normalized URL (without fragment)
        normalized = urlunparse(
            (
                scheme,
                parsed.netloc.lower(),  # Lowercase domain
                path,
                parsed.params,
                query,
                "",  # Remove fragment
            )
        )

        return normalized

    # Backwards compatibility helper methods
    def _is_random_uuid(self, chunk_id: str) -> bool:
        """
        Check if a chunk ID appears to be a random UUID.

        This helps identify chunks that were created before deterministic IDs
        were implemented.

        Args:
            chunk_id: Chunk ID to check

        Returns:
            True if the ID appears to be a random UUID
        """
        # Pattern for random UUIDs (32+ hex characters, possibly with dashes)
        uuid_pattern = re.compile(r"^[0-9a-f\-]{32,}$", re.IGNORECASE)
        clean_id = chunk_id.replace("-", "")
        return bool(uuid_pattern.match(clean_id))

    def _find_legacy_chunk_by_content(
        self, existing_chunks: list[dict[str, Any]], content: str
    ) -> dict[str, Any] | None:
        """
        Find a legacy chunk by content hash for backwards compatibility.

        In mixed environments, we need to match content even if the chunk
        has a random UUID instead of a deterministic ID.

        Args:
            existing_chunks: List of existing chunks
            content: Content to match against

        Returns:
            Matching chunk or None
        """
        content_hash = self._calculate_content_hash(content)

        for chunk in existing_chunks:
            # Direct content hash comparison
            if chunk.get("content_hash") == content_hash:
                return chunk

            # Fallback: direct content comparison for chunks without hashes
            if chunk.get("content") == content:
                return chunk

        return None

    def _should_use_backwards_compatibility(
        self, existing_chunks_map: dict[str, str]
    ) -> bool:
        """
        Determine if backwards compatibility mode should be used.

        This checks if the existing chunks contain random UUIDs, indicating
        a mixed environment that needs backwards compatibility handling.

        Args:
            existing_chunks_map: Map of chunk_id -> content_hash

        Returns:
            True if backwards compatibility is needed
        """
        if not existing_chunks_map:
            return False

        # Sample a few chunk IDs to determine if they're random UUIDs
        sample_size = min(5, len(existing_chunks_map))
        sample_ids = list(existing_chunks_map.keys())[:sample_size]

        random_uuid_count = sum(
            1 for chunk_id in sample_ids if self._is_random_uuid(chunk_id)
        )

        # If more than half are random UUIDs, use backwards compatibility
        return random_uuid_count > (sample_size // 2)

    def _find_line_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find line break boundary."""
        return find_line_boundary(search_text, ideal_end)

    def _find_word_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find word boundary."""
        return find_word_boundary(search_text, ideal_end)

    def _chunk_text_character_based(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Optimized character-based chunking with semantic boundary detection.
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        while start < text_length:
            end = start + self.chunk_size

            # Find a good break point (end of sentence or paragraph)
            if end < text_length:
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_ends = []

                for i in range(search_start, min(end + 50, text_length)):
                    if (
                        text[i] in ".!?"
                        and i + 1 < text_length
                        and text[i + 1] in " \n\t"
                    ):
                        sentence_ends.append(i + 1)

                # Use the last sentence end if found, otherwise stick to character limit
                if sentence_ends:
                    end = sentence_ends[-1]

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "token_count_estimate": int(
                        len(chunk_text.split()) * settings.word_to_token_ratio
                    ),
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)

            # Prevent infinite loop
            if start <= end - self.chunk_size:
                start = end

        return chunks

    def _chunk_text_token_based(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Token-based chunking using Qwen3 tokenizer for optimal semantic boundaries.
        """
        chunks = []

        if self.tokenizer:
            # Use actual tokenizer
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
            start_token = 0
            chunk_index = 0

            while start_token < total_tokens:
                end_token = min(start_token + self.chunk_size, total_tokens)

                # Extract token chunk
                chunk_tokens = tokens[start_token:end_token]

                # Ensure chunk_tokens is a flat list of integers
                if (
                    isinstance(chunk_tokens, list)
                    and len(chunk_tokens) > 0
                    and isinstance(chunk_tokens[0], list)
                ):
                    # Flatten nested list
                    chunk_tokens = [
                        token for sublist in chunk_tokens for token in sublist
                    ]

                chunk_text = self.tokenizer.decode(chunk_tokens)

                if chunk_text.strip():
                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": start_token,  # Use start_pos for consistency with character-based chunking
                        "end_pos": end_token,  # Use end_pos for consistency with character-based chunking
                        "start_token": start_token,
                        "end_token": end_token,
                        "token_count": len(chunk_tokens),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start position with overlap
                start_token = max(
                    start_token + self.chunk_size - self.chunk_overlap, end_token
                )

                # Prevent infinite loop
                if start_token <= end_token - self.chunk_size:
                    start_token = end_token
        else:
            # Fallback to approximate token-based chunking using word estimation
            words = text.split()
            total_words = len(words)
            # Use configurable word-to-token ratio for accuracy
            approx_tokens_per_word = settings.word_to_token_ratio
            chunk_size_words = int(self.chunk_size / approx_tokens_per_word)
            overlap_words = int(self.chunk_overlap / approx_tokens_per_word)

            start_word = 0
            chunk_index = 0

            while start_word < total_words:
                end_word = min(start_word + chunk_size_words, total_words)
                chunk_words = words[start_word:end_word]
                chunk_text = " ".join(chunk_words)

                if chunk_text.strip():
                    estimated_tokens = int(len(chunk_words) * approx_tokens_per_word)
                    # Calculate character positions for consistency
                    text_start_pos = len(" ".join(words[:start_word])) + (
                        1 if start_word > 0 else 0
                    )
                    text_end_pos = text_start_pos + len(chunk_text)

                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": text_start_pos,  # Character position for consistency
                        "end_pos": text_end_pos,  # Character position for consistency
                        "start_word": start_word,
                        "end_word": end_word,
                        "token_count_estimate": estimated_tokens,
                        "word_count": len(chunk_words),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start position with overlap
                start_word = max(
                    start_word + chunk_size_words - overlap_words, end_word
                )

                # Prevent infinite loop
                if start_word <= end_word - chunk_size_words:
                    start_word = end_word

        return chunks

    async def process_crawl_result(
        self,
        crawl_result: CrawlResult,
        progress_callback: Callable[..., None] | None = None,
        deduplication: bool | None = None,
        force_update: bool = False,
    ) -> dict[str, int]:
        """
        Process a crawl result by chunking content and generating embeddings.

        With deduplication enabled, this method:
        1. Queries existing chunks for the source URL
        2. Generates deterministic IDs based on URL and chunk position
        3. Compares content hashes to detect changes
        4. Only upserts new or modified chunks
        5. Deletes orphaned chunks that no longer exist

        Args:
            crawl_result: Result from crawler service
            progress_callback: Optional progress callback
            deduplication: Enable deduplication (defaults to settings.deduplication_enabled)
            force_update: Force update all chunks even if content unchanged

        Returns:
            Dictionary with processing statistics including deduplication metrics
        """
        # Use settings default if not specified
        if deduplication is None:
            deduplication = settings.deduplication_enabled
        if not crawl_result.pages:
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": 0,
                "chunks_updated": 0,
                "chunks_deleted": 0,
            }

        total_pages = len(crawl_result.pages)
        total_chunks = 0
        total_embeddings = 0
        document_chunks = []

        # Deduplication tracking
        chunks_skipped = 0
        chunks_updated = 0
        chunks_deleted = 0
        existing_chunks_map = {}
        legacy_chunks_to_delete = []  # Track legacy chunks that need to be replaced/deleted

        logger.info(
            f"Processing {total_pages} pages for RAG indexing (dedup={deduplication})"
        )

        # Initialize backwards compatibility variables
        use_backwards_compatibility = False
        existing_chunks_list = None

        # Step 1: Get existing chunks if deduplication is enabled
        if deduplication and total_pages > 0:
            # Use the first page's URL as the source URL (they should all be from the same source)
            source_url = crawl_result.pages[0].url
            if progress_callback:
                progress_callback(
                    0, total_pages + 2, "Retrieving existing chunks for deduplication"
                )

            try:
                existing_chunks = await self.vector_service.get_chunks_by_source(
                    source_url
                )
                # Build map of chunk_id -> content_hash for comparison
                existing_chunks_map = {
                    chunk["id"]: chunk.get("content_hash", "")
                    for chunk in existing_chunks
                }
                logger.info(
                    f"Found {len(existing_chunks_map)} existing chunks for {source_url}"
                )

                # Check if backwards compatibility is needed
                use_backwards_compatibility = self._should_use_backwards_compatibility(
                    existing_chunks_map
                )
                if use_backwards_compatibility:
                    logger.info(
                        f"Detected {len(existing_chunks_map)} chunks with random UUIDs, enabling backwards compatibility mode"
                    )

                # Store full chunks list for backwards compatibility
                existing_chunks_list = (
                    existing_chunks if use_backwards_compatibility else None
                )

                # Fast path optimization: if no existing chunks, skip deduplication logic
                if not existing_chunks_map and not force_update:
                    logger.info(
                        "Fast path: No existing chunks found, disabling deduplication for this crawl"
                    )
                    deduplication = False

            except Exception as e:
                logger.warning(
                    f"Could not retrieve existing chunks for deduplication: {e}"
                )
                # Continue without deduplication on error
                deduplication = False

        # Initialize variables for cases where deduplication is disabled
        if not deduplication:
            use_backwards_compatibility = False
            existing_chunks_list = None

        # Process each page
        for i, page in enumerate(crawl_result.pages):
            try:
                if progress_callback:
                    progress_callback(
                        i + 1,
                        total_pages + 2,
                        f"Processing page {i + 1}/{total_pages}: {page.url}",
                    )

                chunk_metadata = page.metadata.get("chunk_metadata", {})
                chunk_index = chunk_metadata.get("chunk_index", i)

                # Generate deterministic ID if deduplication is enabled
                if deduplication:
                    chunk_id = self._generate_deterministic_id(page.url, chunk_index)
                    content_hash = self._calculate_content_hash(page.content)
                else:
                    # Fallback to random UUID for backwards compatibility
                    chunk_id = f"{uuid.uuid4()}"
                    content_hash = None

                # Check if we should skip this chunk (unchanged content)
                should_skip = False
                legacy_chunk_to_replace = None

                if deduplication and not force_update:
                    # Check deterministic ID first (normal case)
                    if (
                        chunk_id in existing_chunks_map
                        and existing_chunks_map[chunk_id] == content_hash
                    ):
                        chunks_skipped += 1
                        logger.debug(
                            f"Skipping unchanged chunk {chunk_id} for {page.url}"
                        )
                        should_skip = True

                    # Backwards compatibility: check for legacy chunks with same content
                    elif use_backwards_compatibility and existing_chunks_list:
                        legacy_chunk = self._find_legacy_chunk_by_content(
                            existing_chunks_list, page.content
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

                if should_skip:
                    continue

                # Determine if this is an update
                if deduplication:
                    if chunk_id in existing_chunks_map:
                        chunks_updated += 1
                        logger.debug(
                            f"Updating changed chunk {chunk_id} for {page.url}"
                        )
                    elif (
                        use_backwards_compatibility
                        and existing_chunks_list
                        and legacy_chunk_to_replace
                    ):
                        # This will replace a legacy chunk
                        chunks_updated += 1
                        legacy_chunks_to_delete.append(legacy_chunk_to_replace["id"])
                        logger.debug(
                            f"Upgrading legacy chunk {legacy_chunk_to_replace['id']} to deterministic ID {chunk_id}"
                        )
                    elif use_backwards_compatibility and existing_chunks_list:
                        # Check if there's a legacy chunk with different content that should be updated
                        legacy_chunk = self._find_legacy_chunk_by_content(
                            existing_chunks_list, page.content
                        )
                        if (
                            legacy_chunk
                            and legacy_chunk.get("content_hash") != content_hash
                        ):
                            chunks_updated += 1
                            legacy_chunk_to_replace = legacy_chunk
                            legacy_chunks_to_delete.append(legacy_chunk["id"])
                            logger.debug(
                                f"Updating and upgrading legacy chunk {legacy_chunk['id']} to {chunk_id}"
                            )

                # Create document chunk with deduplication fields
                now = datetime.utcnow()
                doc_chunk = DocumentChunk(
                    id=chunk_id,
                    content=page.content,
                    source_url=page.url,
                    source_title=page.title,
                    chunk_index=chunk_index,
                    word_count=page.word_count,
                    char_count=len(page.content),
                    metadata=page.metadata,
                    content_hash=content_hash,
                    # For new chunks, first_seen will be set by default_factory
                    # For existing chunks, we should preserve the original first_seen,
                    # but since we don't have that info in the test, let default_factory handle it
                    last_modified=now,  # Always set last_modified to current time
                )
                document_chunks.append(doc_chunk)
                total_chunks += 1

            except Exception as e:
                logger.error(f"Error processing page {page.url}: {e}")
                continue

        if not document_chunks and not deduplication:
            logger.warning("No document chunks created from crawl result")
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": 0,
                "chunks_updated": 0,
                "chunks_deleted": 0,
            }

        # Step 2: Handle orphaned chunks (chunks that exist but are not in new crawl)
        if deduplication and settings.delete_orphaned_chunks and total_pages > 0:
            # Find orphaned chunk IDs
            new_chunk_ids = set()
            for i, page in enumerate(crawl_result.pages):
                chunk_metadata = page.metadata.get("chunk_metadata", {})
                chunk_index = chunk_metadata.get("chunk_index", i)
                chunk_id = self._generate_deterministic_id(page.url, chunk_index)
                new_chunk_ids.add(chunk_id)

            # Identify orphaned chunks (exist in DB but not in new crawl)
            orphaned_ids = set(existing_chunks_map.keys()) - new_chunk_ids

            # Add legacy chunks that need to be replaced/deleted
            all_ids_to_delete = orphaned_ids.union(set(legacy_chunks_to_delete))

            if all_ids_to_delete:
                if progress_callback:
                    progress_callback(
                        total_pages + 1,
                        total_pages + 3,
                        f"Deleting {len(orphaned_ids)} orphaned chunks and {len(legacy_chunks_to_delete)} legacy chunks",
                    )

                try:
                    chunks_deleted = await self.vector_service.delete_chunks_by_ids(
                        list(all_ids_to_delete)
                    )
                    logger.info(
                        f"Deleted {chunks_deleted} chunks ({len(orphaned_ids)} orphaned, {len(legacy_chunks_to_delete)} legacy)"
                    )
                except Exception as e:
                    logger.error(f"Error deleting chunks: {e}")
                    # Continue even if deletion fails

        # If all chunks were skipped (no changes), return early
        if not document_chunks:
            logger.info(
                f"No chunks to process (skipped={chunks_skipped}, deleted={chunks_deleted})"
            )
            return {
                "documents_processed": total_pages,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": chunks_skipped,
                "chunks_updated": chunks_updated,
                "chunks_deleted": chunks_deleted,
            }

        # Generate embeddings in batches
        if progress_callback:
            progress_callback(
                total_pages,
                total_pages + 1,
                f"Generating embeddings for {len(document_chunks)} chunks",
            )

        try:
            # Extract texts for embedding
            texts = [chunk.content for chunk in document_chunks]

            # Generate embeddings using true batch processing for speed
            if len(texts) <= settings.tei_batch_size:
                # Small batch - use true batch for maximum speed
                embedding_results = (
                    await self.embedding_service.generate_embeddings_true_batch(texts)
                )
            else:
                # Large batch - fallback to chunked processing
                embedding_results = (
                    await self.embedding_service.generate_embeddings_batch(
                        texts, batch_size=settings.tei_batch_size
                    )
                )

            # Attach embeddings to document chunks
            for chunk, embedding_result in zip(
                document_chunks, embedding_results, strict=False
            ):
                chunk.embedding = embedding_result.embedding

            total_embeddings = len(embedding_results)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ToolError(f"Failed to generate embeddings: {e!s}") from e

        # Store in vector database
        if progress_callback:
            progress_callback(
                total_pages + 1,
                total_pages + 2,
                f"Storing {len(document_chunks)} embeddings in vector database",
            )

        try:
            stored_count = await self.vector_service.upsert_documents(document_chunks)
            logger.info(f"Stored {stored_count} document chunks in vector database")

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise ToolError(f"Failed to store embeddings: {e!s}") from e

        return {
            "documents_processed": total_pages,
            "chunks_created": total_chunks,
            "embeddings_generated": total_embeddings,
            "chunks_stored": stored_count,
            "chunks_skipped": chunks_skipped,
            "chunks_updated": chunks_updated,
            "chunks_deleted": chunks_deleted,
        }

    async def query(self, query: RagQuery, rerank: bool = True) -> RagResult:
        """
        Perform a RAG query to find relevant documents.

        Args:
            query: The RAG query parameters
            rerank: Whether to apply re-ranking to results

        Returns:
            RagResult with matched documents
        """
        start_time = time.time()

        # Check cache first for exact query match
        cached_result = self.query_cache.get(
            query.query, query.limit, query.min_score, query.source_filters, rerank
        )
        if cached_result is not None:
            logger.info(f"Cache hit for query: '{query.query[:50]}...'")
            return cached_result

        try:
            # Generate query embedding
            embedding_start = time.time()
            embedding_result = await self.embedding_service.generate_embedding(
                query.query
            )
            embedding_time = time.time() - embedding_start

            # Search vector database
            search_start = time.time()
            search_matches = await self.vector_service.search_similar(
                query_vector=embedding_result.embedding,
                limit=query.limit,
                score_threshold=query.min_score,
                source_filter=query.source_filters,
                date_range=query.date_range,
            )
            search_time = time.time() - search_start

            # Apply re-ranking if requested
            rerank_time = None
            if rerank and len(search_matches) > 1 and self.reranker_type != "none":
                rerank_start = time.time()
                search_matches = await self._rerank_results(query.query, search_matches)
                rerank_time = time.time() - rerank_start

            # Filter matches based on query parameters
            filtered_matches = []
            for match in search_matches:
                # Apply content/metadata filters if specified
                if not query.include_content:
                    match.document.content = ""  # Remove content to save bandwidth

                if not query.include_metadata:
                    match.document.metadata = {}

                filtered_matches.append(match)

            processing_time = time.time() - start_time

            result = RagResult(
                query=query.query,
                matches=filtered_matches,
                total_matches=len(filtered_matches),
                processing_time=processing_time,
                embedding_time=embedding_time,
                search_time=search_time,
                rerank_time=rerank_time,
            )

            logger.info(
                f"RAG query completed: {len(filtered_matches)} matches in {processing_time:.3f}s "
                f"(embed: {embedding_time:.3f}s, search: {search_time:.3f}s"
                f"{f', rerank: {rerank_time:.3f}s' if rerank_time else ''})"
            )

            # Cache the result for future queries
            self.query_cache.put(
                query.query,
                query.limit,
                query.min_score,
                query.source_filters,
                rerank,
                result,
            )

            return result

        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise ToolError(f"RAG query failed: {e!s}") from e

    async def _rerank_results(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank search results using Qwen3 reranker or custom algorithm.

        Args:
            query: Original query text
            matches: Initial search matches
            top_k: Number of top results to return (optional)

        Returns:
            Re-ranked list of search matches
        """
        if not matches:
            return matches

        try:
            if self.reranker_type == "qwen3":
                return await self._rerank_with_qwen3(query, matches, top_k)

            if self.reranker_type == "custom":
                return await self._rerank_with_custom_algorithm(query, matches, top_k)

            return matches  # No reranking

        except Exception as e:
            logger.warning("Re-ranking failed, returning original results: %s", e)
            return matches

    async def _rerank_with_qwen3(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank using Qwen3 CrossEncoder reranker.
        """
        if not self.reranker or not matches:
            return matches

        try:
            # Prepare query-document pairs for the reranker
            pairs = []
            for match in matches:
                # Truncate content to reranker max length
                content = match.document.content[: settings.reranker_max_length]
                pairs.append([query, content])

            # Get reranking scores from Qwen3 model using proper batching
            import asyncio

            # Process all pairs in a single batch for efficiency
            scores = await asyncio.to_thread(self.reranker.predict, pairs)

            # Update match scores with reranker predictions
            for match, score in zip(matches, scores, strict=True):
                # Convert reranker logits to normalized probability scores
                reranker_score = float(score)

                # Apply sigmoid normalization for better score distribution
                import math

                normalized_reranker_score = 1.0 / (1.0 + math.exp(-reranker_score))

                # Combine scores with reranker taking priority (it's query-specific)
                # Keep original vector similarity but boost with reranker confidence
                original_score = match.score
                match.score = 0.4 * match.score + 0.6 * normalized_reranker_score

                logger.debug(
                    "Reranking: raw_logit=%.4f, sigmoid_score=%.4f, original_score=%.4f, final_score=%.4f, reranker_model=%s",
                    reranker_score,
                    normalized_reranker_score,
                    original_score,
                    match.score,
                    settings.reranker_model,
                )

            # Sort by updated scores
            matches.sort(key=lambda m: m.score, reverse=True)

            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]

            logger.debug(f"Qwen3 reranked {len(matches)} results")
            return matches

        except Exception:
            logger.exception("Qwen3 reranking failed")
            # Fallback to custom algorithm if available
            if settings.reranker_fallback_to_custom:
                logger.info("Falling back to custom reranking algorithm")
                return await self._rerank_with_custom_algorithm(query, matches, top_k)
            return matches

    async def _rerank_with_custom_algorithm(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank using custom hybrid scoring algorithm (original implementation).
        """
        try:
            # Simple re-ranking based on keyword overlap and length
            query_words = set(query.lower().split())

            for match in matches:
                content_words = set(match.document.content.lower().split())

                # Keyword overlap score
                keyword_overlap = len(query_words.intersection(content_words)) / len(
                    query_words
                )

                # Length penalty (prefer moderate length chunks)
                optimal_length = 500  # characters
                length_penalty = (
                    1.0
                    - abs(len(match.document.content) - optimal_length) / optimal_length
                )
                length_penalty = max(0.1, min(1.0, length_penalty))

                # Title relevance bonus
                title_bonus = 0.0
                if match.document.source_title:
                    title_words = set(match.document.source_title.lower().split())
                    title_overlap = len(query_words.intersection(title_words)) / len(
                        query_words
                    )
                    title_bonus = title_overlap * 0.1

                # Combine scores (weighted average)
                combined_score = (
                    match.score * 0.7  # Vector similarity (primary)
                    + keyword_overlap * 0.2  # Keyword overlap
                    + length_penalty * 0.05  # Length preference
                    + title_bonus * 0.05  # Title relevance
                )

                match.score = min(1.0, combined_score)

            # Sort by combined score
            matches.sort(key=lambda m: m.score, reverse=True)

            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]

            logger.debug("Custom algorithm reranked %d results", len(matches))
            return matches

        except Exception as e:
            logger.warning("Custom reranking failed: %s", e)
            return matches

    async def get_stats(self) -> dict[str, Any]:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with system statistics
        """
        try:
            health = await self.health_check()
            collection_info = await self.vector_service.get_collection_info()
            source_stats = await self.vector_service.get_sources_stats()

            return {
                "health": health,
                "collection": collection_info,
                "sources": source_stats,
                "cache": self.query_cache.get_stats(),
                "config": {
                    "chunk_size_tokens": self.chunk_size
                    if self.tokenizer_type == "token"
                    else None,
                    "chunk_overlap_tokens": self.chunk_overlap
                    if self.tokenizer_type == "token"
                    else None,
                    "chunk_size_chars": self.chunk_size
                    if self.tokenizer_type == "character"
                    else None,
                    "chunk_overlap_chars": self.chunk_overlap
                    if self.tokenizer_type == "character"
                    else None,
                    "tokenizer_type": getattr(
                        self, "tokenizer_type", "character_based"
                    ),
                    "reranker_type": getattr(self, "reranker_type", "none"),
                    "reranker_model": settings.reranker_model
                    if self.reranker_type == "qwen3"
                    else None,
                    "reranker_enabled": settings.reranker_enabled,
                    "embedding_model": settings.tei_model,
                    "vector_dimension": settings.qdrant_vector_size,
                    "distance_metric": settings.qdrant_distance,
                },
            }

        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return {"error": str(e)}

    async def delete_source(self, source_url: str) -> bool:
        """
        Delete all documents from a specific source.

        Args:
            source_url: URL of the source to delete

        Returns:
            True if deletion was successful
        """
        try:
            deleted_count = await self.vector_service.delete_documents_by_source(
                source_url
            )
            logger.info(f"Deleted documents from source: {source_url}")
            # Ensure deleted_count is an integer for comparison
            return int(deleted_count) > 0

        except Exception as e:
            logger.error(f"Error deleting source {source_url}: {e}")
            return False
