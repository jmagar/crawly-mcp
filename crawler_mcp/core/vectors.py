"""
Service for vector database operations using Qdrant.
"""

import logging
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    UpdateStatus,
    VectorParams,
)

from ..config import settings
from ..models.rag import DocumentChunk, SearchMatch

logger = logging.getLogger(__name__)


def _parse_timestamp(timestamp_value: Any) -> datetime:
    """
    Parse timestamp from various formats to datetime.
    Handles ISO strings, datetime objects, and empty values.
    """
    if isinstance(timestamp_value, datetime):
        return timestamp_value
    if isinstance(timestamp_value, str) and timestamp_value:
        try:
            return date_parser.parse(timestamp_value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse timestamp: {timestamp_value}")
    # Default to current time for invalid/empty timestamps
    return datetime.utcnow()


class VectorService:
    """
    Service for vector database operations using Qdrant.
    """

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout),
        )
        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.qdrant_vector_size

        # Map distance strings to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(
            settings.qdrant_distance.lower(), Distance.COSINE
        )

    async def __aenter__(self) -> "VectorService":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type, exc_val: Exception, exc_tb: object
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()

    async def _recreate_client(self) -> None:
        """Recreate the Qdrant client."""
        logger.debug("Recreating Qdrant client...")
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout),
        )

    async def health_check(self) -> bool:
        """
        Check if Qdrant service is healthy and responsive.
        """
        try:
            # Try to get collections as health check
            collections = await self.client.get_collections()
            return collections is not None
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def ensure_collection(self) -> bool:
        """
        Ensure the collection exists, create if it doesn't.

        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists with client recreation on error
            try:
                collections = await self.client.get_collections()
            except Exception as e:
                if "client has been closed" in str(e):
                    logger.debug(
                        "Client closed error during collection check, recreating client..."
                    )
                    await self._recreate_client()
                    collections = await self.client.get_collections()
                else:
                    raise

            existing_names = {col.name for col in collections.collections}

            if self.collection_name in existing_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True

            # Create collection with optimized HNSW configuration
            logger.info(
                f"Creating collection '{self.collection_name}' with optimized HNSW config"
            )
            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=self.distance
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,  # Production value for accuracy/memory balance
                        ef_construct=128,  # Build-time accuracy
                        max_indexing_threads=0,  # Use all available threads
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,  # Batch indexing for performance
                        memmap_threshold=50000,  # Memory management threshold
                        max_segment_size=1_000_000,  # Optimize segment size
                    ),
                )
            except Exception as e:
                if "client has been closed" in str(e):
                    logger.debug(
                        "Client closed error during collection creation, recreating client..."
                    )
                    await self._recreate_client()
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size, distance=self.distance
                        ),
                        hnsw_config=HnswConfigDiff(
                            m=16,  # Production value for accuracy/memory balance
                            ef_construct=128,  # Build-time accuracy
                            max_indexing_threads=0,  # Use all available threads
                        ),
                        optimizers_config=OptimizersConfigDiff(
                            indexing_threshold=20000,  # Batch indexing for performance
                            memmap_threshold=50000,  # Memory management threshold
                            max_segment_size=1_000_000,  # Optimize segment size
                        ),
                    )
                else:
                    raise

            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise ToolError(f"Failed to initialize vector database: {e!s}") from e

    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.
        """
        try:
            info = await self.client.get_collection(self.collection_name)
            # Handle vector config which might be dict or VectorParams
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                # If it's a dict, get the default vector config
                default_config = vectors_config.get("", vectors_config.get("default"))
                distance_info = getattr(default_config, "distance", "unknown")
                size_info = getattr(default_config, "size", 0)
            else:
                # If it's VectorParams, access directly
                distance_info = getattr(vectors_config, "distance", "unknown")
                size_info = getattr(vectors_config, "size", 0)

            return {
                "name": getattr(info, "name", self.collection_name),
                "status": info.status,
                "vectors_count": info.vectors_count if info.vectors_count else 0,
                "indexed_vectors_count": info.indexed_vectors_count
                if info.indexed_vectors_count
                else 0,
                "points_count": info.points_count if info.points_count else 0,
                "segments_count": info.segments_count if info.segments_count else 0,
                "config": {
                    "distance": getattr(distance_info, "name", str(distance_info)),
                    "vector_size": size_info,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def upsert_documents(
        self, documents: list[DocumentChunk], batch_size: int = 100
    ) -> int:
        """
        Upsert document chunks into the vector database.

        Args:
            documents: List of document chunks to upsert
            batch_size: Size of batches for bulk operations

        Returns:
            Number of documents successfully upserted
        """
        if not documents:
            return 0

        await self.ensure_collection()

        total_batches = (len(documents) + batch_size - 1) // batch_size
        total_upserted = 0

        logger.info(f"Upserting {len(documents)} documents in {total_batches} batches")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            logger.debug(
                f"Upserting batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)"
            )

            try:
                points = []
                for doc in batch:
                    if not doc.embedding:
                        logger.warning(f"Skipping document {doc.id} - no embedding")
                        continue

                    # Prepare point data
                    point = PointStruct(
                        id=doc.id,
                        vector=doc.embedding,
                        payload={
                            "content": doc.content,
                            "source_url": doc.source_url,
                            "source_title": doc.source_title,
                            "chunk_index": doc.chunk_index,
                            "word_count": doc.word_count,
                            "char_count": doc.char_count,
                            "timestamp": doc.timestamp.isoformat(),
                            **doc.metadata,
                        },
                    )
                    points.append(point)

                if points:
                    # Upsert the batch with client recreation on error
                    try:
                        result = await self.client.upsert(
                            collection_name=self.collection_name, points=points
                        )
                    except Exception as e:
                        if "client has been closed" in str(e):
                            logger.debug("Client closed error, recreating client...")
                            await self._recreate_client()
                            result = await self.client.upsert(
                                collection_name=self.collection_name, points=points
                            )
                        else:
                            raise

                    if result.status == UpdateStatus.COMPLETED:
                        total_upserted += len(points)
                        logger.debug(f"Successfully upserted batch {batch_idx + 1}")
                    else:
                        logger.error(
                            f"Failed to upsert batch {batch_idx + 1}: {result.status}"
                        )

            except Exception as e:
                logger.error(f"Error upserting batch {batch_idx + 1}: {e}")
                # Continue with next batch rather than failing completely

        logger.info(f"Total documents upserted: {total_upserted}")
        return total_upserted

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[SearchMatch]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            source_filter: Filter by source URLs (optional)
            date_range: Filter by date range as ISO strings (optional)

        Returns:
            List of search matches with similarity scores
        """
        await self.ensure_collection()

        try:
            # Build filters
            filter_conditions: list[FieldCondition] = []

            if source_filter:
                filter_conditions.append(
                    FieldCondition(key="source_url", match=MatchAny(any=source_filter))
                )

            # Add date range filtering using Qdrant's range filter
            if date_range:
                from qdrant_client.models import Range

                start, end = date_range
                if start is not None:
                    start_timestamp = _parse_timestamp(start).isoformat()
                    filter_conditions.append(
                        FieldCondition(
                            key="timestamp", range=Range(gte=start_timestamp)
                        )
                    )

                if end is not None:
                    end_timestamp = _parse_timestamp(end).isoformat()
                    filter_conditions.append(
                        FieldCondition(key="timestamp", range=Range(lte=end_timestamp))
                    )

            # Prepare search request
            search_filter = None
            if filter_conditions:
                # Qdrant expects a list, not a Sequence
                search_filter = Filter(must=list(filter_conditions))

            # Dynamic ef parameter based on query complexity
            # Higher ef for better accuracy when needed
            ef_value = min(256, max(64, limit * 4))  # 4x limit, capped at 256

            # Perform search using modern query_points API with optimized search params
            query_response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
                search_params=SearchParams(
                    hnsw_ef=ef_value,
                    exact=settings.qdrant_search_exact,
                ),  # Dynamic ef with configurable exact-search toggle
            )

            # Extract results - handle different return types from query_points
            results: list[Any] = []
            if hasattr(query_response, "points"):
                results = query_response.points
            elif isinstance(query_response, (tuple, list)) and len(query_response) > 0:  # noqa: UP038
                results = query_response[0]
            else:
                # Cast to list for consistent handling
                if hasattr(query_response, "__iter__"):
                    results = list(query_response)
                else:
                    results = [query_response]

            # Convert results to SearchMatch objects
            matches = []
            for result in results:
                # Safely access payload with null checks
                payload = getattr(result, "payload", None)
                if payload is None:
                    continue

                # Safely access result attributes
                result_id = getattr(result, "id", "unknown")
                result_score = getattr(result, "score", 0.0)

                # Reconstruct document chunk
                document = DocumentChunk(
                    id=str(result_id),
                    content=payload.get("content", ""),
                    source_url=payload.get("source_url", ""),
                    source_title=payload.get("source_title"),
                    chunk_index=payload.get("chunk_index", 0),
                    word_count=payload.get("word_count", 0),
                    char_count=payload.get("char_count", 0),
                    timestamp=_parse_timestamp(payload.get("timestamp", "")),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in [
                            "content",
                            "source_url",
                            "source_title",
                            "chunk_index",
                            "word_count",
                            "char_count",
                            "timestamp",
                        ]
                    },
                )

                # Create search match
                match = SearchMatch(document=document, score=float(result_score))
                matches.append(match)

            logger.debug(f"Found {len(matches)} similar documents")
            return matches

        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            raise ToolError(f"Vector search failed: {e!s}") from e

    async def get_document_by_id(self, document_id: str) -> DocumentChunk | None:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: The document ID to retrieve

        Returns:
            DocumentChunk if found, None otherwise
        """
        await self.ensure_collection()

        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=False,
            )

            if not result:
                return None

            point = result[0]
            payload = getattr(point, "payload", None)
            if payload is None:
                return None

            return DocumentChunk(
                id=str(getattr(point, "id", "unknown")),
                content=payload.get("content", ""),
                source_url=payload.get("source_url", ""),
                source_title=payload.get("source_title"),
                chunk_index=payload.get("chunk_index", 0),
                word_count=payload.get("word_count", 0),
                char_count=payload.get("char_count", 0),
                timestamp=_parse_timestamp(payload.get("timestamp", "")),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in [
                        "content",
                        "source_url",
                        "source_title",
                        "chunk_index",
                        "word_count",
                        "char_count",
                        "timestamp",
                    ]
                },
            )

        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None

    async def delete_documents_by_source(self, source_url: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_url: URL of the source to delete documents from

        Returns:
            Number of documents deleted
        """
        await self.ensure_collection()

        try:
            # Delete points with matching source_url
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_url", match=MatchValue(value=source_url)
                        )
                    ]
                ),
            )

            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Deleted documents from source: {source_url}")
                return 1  # Qdrant doesn't return exact count for filter deletes
            else:
                logger.error(
                    f"Failed to delete documents from source {source_url}: {result.status}"
                )
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents from source {source_url}: {e}")
            return 0

    async def get_chunks_by_source(self, source_url: str) -> list[dict[str, Any]]:
        """
        Get all existing chunks for a source URL.

        Args:
            source_url: URL of the source to retrieve chunks from

        Returns:
            List of chunk documents with their IDs and content hashes
        """
        await self.ensure_collection()

        try:
            # Query all points with matching source_url
            response = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_url", match=MatchValue(value=source_url)
                        )
                    ]
                ),
                limit=10000,  # Large limit to get all chunks
                with_payload=True,
                with_vectors=False,  # Don't need embeddings for dedup
            )

            chunks = []
            for point in response[0]:  # response is (points, next_offset)
                payload = point.payload or {}
                chunk_data = {
                    "id": point.id,
                    "content": payload.get("content", ""),
                    "content_hash": payload.get("content_hash"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "metadata": payload.get("metadata", {}),
                }
                chunks.append(chunk_data)

            logger.info(f"Retrieved {len(chunks)} chunks for source: {source_url}")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks for source {source_url}: {e}")
            return []

    async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """
        Delete specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        await self.ensure_collection()

        try:
            # Delete points by IDs
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=chunk_ids,  # Direct ID list
            )

            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Deleted {len(chunk_ids)} chunks by ID")
                return len(chunk_ids)
            else:
                logger.error(f"Failed to delete chunks by ID: {result.status}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting chunks by ID: {e}")
            return 0

    async def get_sources_stats(self) -> dict[str, Any]:
        """
        Get statistics about sources in the vector database.

        Returns:
            Dictionary with source statistics
        """
        await self.ensure_collection()

        try:
            # Scroll through all points to collect statistics
            stats: dict[str, Any] = {
                "total_documents": 0,
                "unique_sources": set(),
                "source_counts": {},
                "total_content_length": 0,
                "average_chunk_size": 0.0,
            }

            offset = None
            limit = 1000

            while True:
                result = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = result[0]
                next_offset = result[1]

                if not points:
                    break

                for point in points:
                    payload = getattr(point, "payload", None)
                    if payload is None:
                        continue
                    source_url = payload.get("source_url", "unknown")
                    char_count = payload.get("char_count", 0)

                    stats["total_documents"] += 1
                    stats["unique_sources"].add(source_url)
                    stats["source_counts"][source_url] = (
                        stats["source_counts"].get(source_url, 0) + 1
                    )
                    stats["total_content_length"] += char_count

                if next_offset is None:
                    break
                offset = next_offset

            # Calculate average chunk size
            if stats["total_documents"] > 0:
                stats["average_chunk_size"] = (
                    stats["total_content_length"] / stats["total_documents"]
                )

            # Convert set to count
            stats["unique_sources"] = len(stats["unique_sources"])

            return stats

        except Exception as e:
            logger.error(f"Error getting source statistics: {e}")
            return {}

    async def get_unique_sources(
        self,
        domains: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get unique sources from the vector database with filtering and pagination.

        Args:
            domains: Filter by domains (e.g., ["github.com", "example.com"])
            search_term: Search term to filter titles and URLs
            limit: Maximum number of sources to return
            offset: Offset for pagination

        Returns:
            Dictionary with sources and metadata
        """
        await self.ensure_collection()

        try:
            # Collect source information
            sources_data: dict[str, dict[str, Any]] = {}
            total_sources = 0

            # Scroll through all points to collect source information
            scroll_offset = None
            scroll_limit = 1000

            while True:
                result = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=scroll_limit,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = result[0]
                next_offset = result[1]

                if not points:
                    break

                for point in points:
                    payload = getattr(point, "payload", None)
                    if payload is None:
                        continue

                    source_url = payload.get("source_url", "")
                    source_title = payload.get("source_title", "")
                    char_count = payload.get("char_count", 0)
                    word_count = payload.get("word_count", 0)
                    timestamp = payload.get("timestamp", "")

                    if not source_url:
                        continue

                    # Apply domain filtering
                    if domains:
                        from urllib.parse import urlparse

                        parsed_url = urlparse(source_url)
                        host = parsed_url.netloc.lower()
                        domains_lc = {d.lower() for d in domains}
                        if not any(
                            host == d or host.endswith(f".{d}") for d in domains_lc
                        ):
                            continue

                    # Apply search term filtering
                    if search_term:
                        search_lower = search_term.lower()
                        if (
                            search_lower not in source_url.lower()
                            and search_lower not in (source_title or "").lower()
                        ):
                            continue

                    # Aggregate source data
                    if source_url not in sources_data:
                        sources_data[source_url] = {
                            "url": source_url,
                            "title": source_title,
                            "chunk_count": 0,
                            "total_content_length": 0,
                            "total_word_count": 0,
                            "last_crawled": timestamp,
                            "source_type": "webpage",  # Default, could be enhanced
                            "status": "active",
                        }

                    # Update aggregated stats
                    source_data = sources_data[source_url]
                    source_data["chunk_count"] += 1
                    source_data["total_content_length"] += char_count
                    source_data["total_word_count"] += word_count

                    # Keep the most recent timestamp
                    if timestamp and (
                        not source_data["last_crawled"]
                        or timestamp > source_data["last_crawled"]
                    ):
                        source_data["last_crawled"] = timestamp

                if next_offset is None:
                    break
                scroll_offset = next_offset

            # Convert to list and apply pagination
            all_sources = list(sources_data.values())
            total_sources = len(all_sources)

            # Sort by URL for consistent ordering
            all_sources.sort(key=lambda x: x["url"])

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_sources = all_sources[start_idx:end_idx]

            return {
                "sources": paginated_sources,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_sources,
                    "returned": len(paginated_sources),
                },
                "filters_applied": {
                    "domains": domains,
                    "search_term": search_term,
                },
            }

        except Exception as e:
            logger.error(f"Error getting unique sources: {e}")
            return {
                "sources": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0,
                    "returned": 0,
                },
                "filters_applied": {"domains": domains, "search_term": search_term},
                "error": str(e),
            }


# Feature flag support for modular vector implementation
def create_vector_service():
    """
    Create a VectorService instance based on configuration.

    Returns:
        VectorService instance (either original or modular based on feature flag)
    """
    if settings.use_modular_vectors:
        # Import and use modular implementation
        from .vectors import VectorService as ModularVectorService

        return ModularVectorService()
    else:
        # Use original implementation
        return VectorService()


# Backward compatibility - allows existing code to continue working
# while enabling gradual migration to modular implementation
if settings.use_modular_vectors:
    try:
        from .vectors import VectorService as ModularVectorService

        # Replace the original VectorService with the modular one when flag is enabled
        VectorService = ModularVectorService
    except ImportError:
        logger.warning(
            "Modular vector service requested but not available, using original implementation"
        )
