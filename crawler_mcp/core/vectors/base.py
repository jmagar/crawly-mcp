"""
Base class for vector service modules providing shared functionality.
"""

import logging
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance

from ...config import settings

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


class BaseVectorService:
    """
    Base class for vector service modules providing shared client management
    and common functionality.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the base vector service.
        
        Args:
            client: Optional shared Qdrant client instance
        """
        if client is not None:
            self.client = client
        else:
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

    async def _recreate_client(self) -> None:
        """Recreate the Qdrant client."""
        logger.debug("Recreating Qdrant client...")
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout),
        )

    async def _handle_client_error(self, e: Exception) -> bool:
        """
        Handle client connection errors with automatic recreation.
        
        Args:
            e: The exception that occurred
            
        Returns:
            True if client was recreated, False if error should be re-raised
        """
        if "client has been closed" in str(e):
            logger.debug("Client closed error detected, recreating client...")
            await self._recreate_client()
            return True
        return False

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()