"""
Service for managing crawled sources and their metadata.
"""

import logging
from datetime import datetime
from typing import Any

from fastmcp.exceptions import ToolError

from ..models.crawl import CrawlResult, PageContent
from ..models.sources import (
    SourceFilter,
    SourceInfo,
    SourceMetadata,
    SourceType,
)
from .vectors import VectorService

logger = logging.getLogger(__name__)


class SourceService:
    """
    Service for managing source information and metadata.
    """

    def __init__(self) -> None:
        self.vector_service = VectorService()
        # In-memory source registry (in production, this would be persisted)
        self._sources: dict[str, SourceInfo] = {}

    async def __aenter__(self) -> "SourceService":
        """Async context manager entry."""
        await self.vector_service.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type, exc_val: Exception, exc_tb: object
    ) -> None:
        """Async context manager exit."""
        await self.vector_service.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        """Close the vector service."""
        await self.vector_service.close()

    async def register_crawl_result(
        self, crawl_result: CrawlResult, source_type: SourceType = SourceType.WEBPAGE
    ) -> list[SourceInfo]:
        """
        Register sources from a crawl result.

        Args:
            crawl_result: Result from crawler service
            source_type: Type of source being registered

        Returns:
            List of registered source info objects
        """
        registered_sources = []

        try:
            for page in crawl_result.pages:
                source_info = await self._create_source_info(
                    page, source_type, crawl_result
                )
                self._sources[source_info.id] = source_info
                registered_sources.append(source_info)

            logger.info(
                f"Registered {len(registered_sources)} sources from crawl result"
            )
            return registered_sources

        except Exception as e:
            logger.error(f"Error registering crawl result: {e}")
            raise ToolError(f"Failed to register sources: {e!s}") from e

    async def _create_source_info(
        self,
        page_content: PageContent,
        source_type: SourceType,
        crawl_result: CrawlResult,
    ) -> SourceInfo:
        """
        Create source info from page content.

        Args:
            page_content: PageContent object
            source_type: Type of source
            crawl_result: Parent crawl result

        Returns:
            SourceInfo object
        """
        from urllib.parse import urlparse

        # Generate source ID
        source_id = f"src_{hash(page_content.url) & 0x7FFFFFFF:08x}"

        # Parse URL for domain
        parsed_url = urlparse(page_content.url)
        domain = parsed_url.netloc

        # Extract metadata
        metadata = SourceMetadata(
            domain=domain,
            content_type=page_content.metadata.get("content_type", "text/html"),
            word_count=page_content.word_count,
            character_count=len(page_content.content),
            link_count=len(page_content.links),
            image_count=len(page_content.images),
            http_status=page_content.metadata.get("http_status"),
            file_size=page_content.metadata.get(
                "content_length", len(page_content.content)
            ),
            custom_fields=page_content.metadata,
        )

        # Estimate chunk count (will be updated later)
        estimated_chunks = max(1, len(page_content.content) // 1000)

        source_info = SourceInfo(
            id=source_id,
            url=page_content.url,
            title=page_content.title,
            source_type=source_type,
            status="active",
            chunk_count=estimated_chunks,
            total_content_length=len(page_content.content),
            metadata=metadata,
            last_crawled=datetime.utcnow(),
        )

        return source_info

    async def list_sources(
        self, filter_criteria: SourceFilter | None = None
    ) -> list[SourceInfo]:
        """
        List sources with optional filtering.

        Args:
            filter_criteria: Optional filter criteria

        Returns:
            List of matching sources
        """
        try:
            # Get all sources
            all_sources = list(self._sources.values())

            # Apply filtering if provided
            if filter_criteria:
                filtered_sources = [
                    source
                    for source in all_sources
                    if filter_criteria.matches_source(source)
                ]
            else:
                filtered_sources = all_sources

            # Apply pagination
            if filter_criteria:
                start_idx = filter_criteria.offset
                end_idx = start_idx + filter_criteria.limit
                filtered_sources = filtered_sources[start_idx:end_idx]

            return filtered_sources

        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            raise ToolError(f"Failed to list sources: {e!s}") from e

    async def get_source(self, source_id: str) -> SourceInfo | None:
        """
        Get a specific source by ID.

        Args:
            source_id: ID of the source to retrieve

        Returns:
            SourceInfo if found, None otherwise
        """
        return self._sources.get(source_id)

    async def update_source_stats(self, source_id: str) -> bool:
        """
        Update source statistics from vector database.

        Args:
            source_id: ID of the source to update

        Returns:
            True if update was successful
        """
        try:
            source_info = self._sources.get(source_id)
            if not source_info:
                return False

            # Get actual statistics from vector database
            stats = await self.vector_service.get_sources_stats()

            # Update source info with actual chunk count
            source_counts = stats.get("source_counts", {})
            actual_chunk_count = source_counts.get(source_info.url, 0)

            if actual_chunk_count > 0:
                source_info.chunk_count = actual_chunk_count
                source_info.updated_at = datetime.utcnow()

                logger.debug(
                    f"Updated source {source_id} with {actual_chunk_count} chunks"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating source stats for {source_id}: {e}")
            return False

    async def delete_source(self, source_id: str) -> bool:
        """
        Delete a source and all its associated data.

        Args:
            source_id: ID of the source to delete

        Returns:
            True if deletion was successful
        """
        try:
            source_info = self._sources.get(source_id)
            if not source_info:
                logger.warning(f"Source {source_id} not found for deletion")
                return False

            # Delete from vector database
            _deleted_count = await self.vector_service.delete_documents_by_source(
                source_info.url
            )

            # Remove from in-memory registry
            if source_id in self._sources:
                del self._sources[source_id]

            logger.info(f"Deleted source {source_id} ({source_info.url})")
            return True

        except Exception as e:
            logger.error(f"Error deleting source {source_id}: {e}")
            return False

    async def get_source_statistics(self) -> dict[str, Any]:
        """
        Get overall source statistics.

        Returns:
            Dictionary with source statistics
        """
        try:
            # Get vector database stats
            vector_stats = await self.vector_service.get_sources_stats()

            # Get source registry stats
            registry_stats: dict[str, Any] = {
                "registered_sources": len(self._sources),
                "sources_by_type": {},
                "sources_by_status": {},
                "stale_sources": 0,
            }

            for source in self._sources.values():
                # Count by type
                source_type = source.source_type.value
                registry_stats["sources_by_type"][source_type] = (
                    registry_stats["sources_by_type"].get(source_type, 0) + 1
                )

                # Count by status
                registry_stats["sources_by_status"][source.status] = (
                    registry_stats["sources_by_status"].get(source.status, 0) + 1
                )

                # Count stale sources
                if source.is_stale:
                    registry_stats["stale_sources"] += 1

            return {"vector_database": vector_stats, "source_registry": registry_stats}

        except Exception as e:
            logger.error(f"Error getting source statistics: {e}")
            return {"error": str(e)}

    async def refresh_all_sources(self) -> dict[str, int | str]:
        """
        Refresh statistics for all registered sources.

        Returns:
            Dictionary with refresh statistics
        """
        try:
            updated_count = 0
            failed_count = 0

            for source_id in list(self._sources.keys()):
                try:
                    if await self.update_source_stats(source_id):
                        updated_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    logger.error(f"Error refreshing source {source_id}: {e}")
                    failed_count += 1

            logger.info(f"Refreshed {updated_count} sources, {failed_count} failed")

            return {
                "updated": updated_count,
                "failed": failed_count,
                "total": len(self._sources),
            }

        except Exception as e:
            logger.error(f"Error refreshing sources: {e}")
            return {"error": str(e)}

    def clear_registry(self) -> None:
        """Clear the in-memory source registry."""
        self._sources.clear()
        logger.info("Cleared source registry")

    async def export_sources(self) -> list[dict[str, Any]]:
        """
        Export all sources as a list of dictionaries.

        Returns:
            List of source dictionaries
        """
        try:
            return [source.dict() for source in self._sources.values()]
        except Exception as e:
            logger.error(f"Error exporting sources: {e}")
            return []

    async def import_sources(self, sources_data: list[dict[str, Any]]) -> int:
        """
        Import sources from a list of dictionaries.

        Args:
            sources_data: List of source dictionaries

        Returns:
            Number of sources successfully imported
        """
        try:
            imported_count = 0

            for source_data in sources_data:
                try:
                    source_info = SourceInfo(**source_data)
                    self._sources[source_info.id] = source_info
                    imported_count += 1
                except Exception as e:
                    logger.error(f"Error importing source: {e}")

            logger.info(f"Imported {imported_count} sources")
            return imported_count

        except Exception as e:
            logger.error(f"Error importing sources: {e}")
            return 0
