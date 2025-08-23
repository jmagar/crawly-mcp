"""
Collection management for Qdrant vector database operations.
"""

import logging
from typing import Any

from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from .base import BaseVectorService

logger = logging.getLogger(__name__)


class CollectionManager(BaseVectorService):
    """
    Manages Qdrant collection lifecycle and configuration.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the collection manager.

        Args:
            client: Optional shared Qdrant client instance
        """
        super().__init__(client)

    async def health_check(self) -> bool:
        """
        Check if Qdrant service is healthy and responsive.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to get collections as health check
            collections = await self.client.get_collections()
            return collections is not None
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def ensure_collection_exists(self) -> bool:
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
                if await self._handle_client_error(e):
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
            await self._create_collection_with_config()

            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise ToolError(f"Failed to initialize vector database: {e!s}") from e

    async def _create_collection_with_config(self) -> None:
        """
        Create collection with optimized configuration.
        Handles client recreation on connection errors.
        """
        collection_config = {
            "collection_name": self.collection_name,
            "vectors_config": VectorParams(
                size=self.vector_size, distance=self.distance
            ),
            "hnsw_config": HnswConfigDiff(
                m=16,  # Production value for accuracy/memory balance
                ef_construct=128,  # Build-time accuracy
                max_indexing_threads=0,  # Use all available threads
            ),
            "optimizers_config": OptimizersConfigDiff(
                indexing_threshold=20000,  # Batch indexing for performance
                memmap_threshold=50000,  # Memory management threshold
                max_segment_size=1_000_000,  # Optimize segment size
            ),
        }

        try:
            await self.client.create_collection(**collection_config)
        except Exception as e:
            if await self._handle_client_error(e):
                await self.client.create_collection(**collection_config)
            else:
                raise

    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
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

    async def delete_collection(self) -> bool:
        """
        Delete the collection.

        Returns:
            True if collection was deleted successfully
        """
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def recreate_client(self) -> None:
        """
        Public method to recreate the Qdrant client.
        """
        await self._recreate_client()
