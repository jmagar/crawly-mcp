"""
FastMCP tools for RAG operations.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from ..services import RagService, SourceService
from ..models.rag_models import RagQuery
from ..models.source_models import SourceFilter, SourceType
from ..middleware.progress_middleware import progress_middleware


logger = logging.getLogger(__name__)


def register_rag_tools(mcp: FastMCP):
    """Register all RAG tools with the FastMCP server."""
    
    @mcp.tool
    async def rag_query(
        ctx: Context,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        source_filters: Optional[List[str]] = None,
        rerank: bool = True,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """
        Perform semantic search using RAG to find relevant documents.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return (1-100)
            min_score: Minimum similarity score threshold (0.0-1.0)
            source_filters: List of source URLs to filter by (optional)
            rerank: Whether to apply re-ranking to improve results
            include_content: Whether to include full content in results
            
        Returns:
            Dictionary with search results and metadata
        """
        await ctx.info(f"Performing RAG query: '{query}' (limit: {limit})")
        
        # Validate parameters
        if limit > 100:
            raise ToolError("limit cannot exceed 100")
        if not 0.0 <= min_score <= 1.0:
            raise ToolError("min_score must be between 0.0 and 1.0")
        if not query.strip():
            raise ToolError("query cannot be empty")
        
        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"rag_{hash(query)}")
        
        try:
            await ctx.report_progress(progress=1, total=4)
            
            # Create RAG query object
            rag_query_obj = RagQuery(
                query=query.strip(),
                limit=limit,
                min_score=min_score,
                source_filters=source_filters,
                include_content=include_content,
                rerank=rerank
            )
            
            await ctx.report_progress(progress=2, total=4)
            
            # Perform RAG query
            async with RagService() as rag_service:
                await ctx.report_progress(progress=3, total=4)
                
                rag_result = await rag_service.query(rag_query_obj, rerank=rerank)
            
            await ctx.report_progress(progress=4, total=4)
            
            # Prepare response
            result = {
                "query": rag_result.query,
                "total_matches": rag_result.total_matches,
                "matches": [],
                "performance": {
                    "total_time": rag_result.processing_time,
                    "embedding_time": rag_result.embedding_time,
                    "search_time": rag_result.search_time,
                    "rerank_time": rag_result.rerank_time
                },
                "quality_metrics": {
                    "average_score": rag_result.average_score,
                    "best_match_score": rag_result.best_match_score,
                    "high_confidence_matches": rag_result.has_high_confidence_matches
                },
                "timestamp": rag_result.timestamp.isoformat()
            }
            
            # Process matches
            for match in rag_result.matches:
                match_data = {
                    "score": match.score,
                    "relevance": match.relevance,
                    "document": {
                        "id": match.document.id,
                        "source_url": match.document.source_url,
                        "source_title": match.document.source_title,
                        "chunk_index": match.document.chunk_index,
                        "word_count": match.document.word_count,
                        "timestamp": match.document.timestamp.isoformat()
                    }
                }
                
                # Include content if requested
                if include_content:
                    match_data["document"]["content"] = match.document.content
                    if match.highlighted_content:
                        match_data["highlighted_content"] = match.highlighted_content
                
                # Include metadata if available
                if match.document.metadata:
                    match_data["document"]["metadata"] = match.document.metadata
                
                result["matches"].append(match_data)
            
            await ctx.info(
                f"RAG query completed: {rag_result.total_matches} matches found in {rag_result.processing_time:.3f}s "
                f"(avg score: {rag_result.average_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"RAG query failed: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
        
        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)
    
    @mcp.tool
    async def list_sources(
        ctx: Context,
        source_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        search_term: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List and filter crawled sources with their metadata.
        
        Args:
            source_types: Filter by source types ("webpage", "repository", "directory")
            domains: Filter by domains (e.g., ["github.com", "example.com"])
            statuses: Filter by status ("active", "inactive", "error")
            search_term: Search term to filter titles and URLs
            limit: Maximum number of sources to return (1-200)
            offset: Offset for pagination
            
        Returns:
            Dictionary with filtered sources and metadata
        """
        await ctx.info(f"Listing sources (limit: {limit}, offset: {offset})")
        
        # Validate parameters
        if limit > 200:
            raise ToolError("limit cannot exceed 200")
        if offset < 0:
            raise ToolError("offset cannot be negative")
        
        try:
            # Convert source types to enum values
            source_type_enums = None
            if source_types:
                try:
                    source_type_enums = [SourceType(st.lower()) for st in source_types]
                except ValueError as e:
                    raise ToolError(f"Invalid source type: {str(e)}")
            
            # Create filter
            filter_criteria = SourceFilter(
                source_types=source_type_enums,
                domains=domains,
                statuses=statuses,
                search_term=search_term,
                limit=limit,
                offset=offset
            )
            
            # Get sources
            async with SourceService() as source_service:
                sources = await source_service.list_sources(filter_criteria)
                stats = await source_service.get_source_statistics()
            
            # Prepare response
            result = {
                "sources": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "returned": len(sources)
                },
                "statistics": stats,
                "filters_applied": {
                    "source_types": source_types,
                    "domains": domains,
                    "statuses": statuses,
                    "search_term": search_term
                }
            }
            
            # Process sources
            for source in sources:
                source_data = {
                    "id": source.id,
                    "url": source.url,
                    "title": source.title,
                    "source_type": source.source_type.value,
                    "status": source.status,
                    "chunk_count": source.chunk_count,
                    "total_content_length": source.total_content_length,
                    "average_chunk_size": source.avg_chunk_size,
                    "created_at": source.created_at.isoformat(),
                    "updated_at": source.updated_at.isoformat(),
                    "last_crawled": source.last_crawled.isoformat() if source.last_crawled else None,
                    "is_stale": source.is_stale,
                    "metadata": {
                        "domain": source.metadata.domain,
                        "word_count": source.metadata.word_count,
                        "character_count": source.metadata.character_count,
                        "link_count": source.metadata.link_count,
                        "image_count": source.metadata.image_count,
                        "language": source.metadata.language,
                        "content_type": source.metadata.content_type,
                        "tags": source.metadata.tags,
                        "categories": source.metadata.categories
                    }
                }
                
                result["sources"].append(source_data)
            
            await ctx.info(f"Found {len(sources)} sources matching criteria")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to list sources: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
    
    @mcp.tool
    async def get_rag_stats(ctx: Context) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the RAG system.
        
        Returns:
            Dictionary with RAG system statistics and health information
        """
        await ctx.info("Retrieving RAG system statistics")
        
        try:
            async with RagService() as rag_service:
                stats = await rag_service.get_stats()
            
            async with SourceService() as source_service:
                source_stats = await source_service.get_source_statistics()
            
            # Combine statistics
            result = {
                "rag_system": stats,
                "source_management": source_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add health summary
            health = stats.get("health", {})
            all_healthy = all(health.values()) if health else False
            
            result["health_summary"] = {
                "all_services_healthy": all_healthy,
                "services": health
            }
            
            # Add quick facts
            collection_info = stats.get("collection", {})
            result["quick_facts"] = {
                "total_documents": collection_info.get("points_count", 0),
                "total_vectors": collection_info.get("vectors_count", 0),
                "total_sources": source_stats.get("source_registry", {}).get("registered_sources", 0),
                "vector_dimension": collection_info.get("config", {}).get("vector_size", 0),
                "collection_status": collection_info.get("status", "unknown")
            }
            
            await ctx.info(f"RAG stats retrieved: {result['quick_facts']['total_documents']} documents, {result['quick_facts']['total_sources']} sources")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to get RAG statistics: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)
    
    @mcp.tool
    async def delete_source(
        ctx: Context,
        source_url: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a source and all its associated documents from the RAG system.
        
        Args:
            source_url: URL of the source to delete
            confirm: Confirmation flag (must be True to proceed)
            
        Returns:
            Dictionary with deletion results
        """
        await ctx.info(f"Request to delete source: {source_url}")
        
        if not confirm:
            raise ToolError("Deletion requires confirmation. Set confirm=True to proceed.")
        
        try:
            # Delete from RAG service
            async with RagService() as rag_service:
                rag_deleted = await rag_service.delete_source(source_url)
            
            # Find and delete from source service
            async with SourceService() as source_service:
                # Find source by URL
                all_sources = await source_service.list_sources()
                matching_sources = [s for s in all_sources if s.url == source_url]
                
                source_deleted = False
                for source in matching_sources:
                    if await source_service.delete_source(source.id):
                        source_deleted = True
            
            result = {
                "source_url": source_url,
                "rag_documents_deleted": rag_deleted,
                "source_registry_deleted": source_deleted,
                "success": rag_deleted or source_deleted,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if result["success"]:
                await ctx.info(f"Successfully deleted source: {source_url}")
            else:
                await ctx.info(f"No data found to delete for source: {source_url}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to delete source: {str(e)}"
            await ctx.info(error_msg)
            raise ToolError(error_msg)