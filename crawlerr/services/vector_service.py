"""
Service for vector database operations using Qdrant.
"""
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct, 
    Filter, FieldCondition, Range, SearchRequest, ScrollRequest,
    UpdateStatus, CollectionInfo
)
from ..config import settings
from ..models.rag_models import DocumentChunk, SearchMatch
from fastmcp.exceptions import ToolError


logger = logging.getLogger(__name__)


class VectorService:
    """
    Service for vector database operations using Qdrant.
    """
    
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout
        )
        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.qdrant_vector_size
        
        # Map distance strings to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(settings.qdrant_distance.lower(), Distance.COSINE)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the Qdrant client."""
        await self.client.close()
    
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
            # Check if collection exists
            collections = await self.client.get_collections()
            existing_names = {col.name for col in collections.collections}
            
            if self.collection_name in existing_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            logger.info(f"Creating collection '{self.collection_name}'")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise ToolError(f"Failed to initialize vector database: {str(e)}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        """
        try:
            info = await self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "status": info.status,
                "vectors_count": info.vectors_count if info.vectors_count else 0,
                "indexed_vectors_count": info.indexed_vectors_count if info.indexed_vectors_count else 0,
                "points_count": info.points_count if info.points_count else 0,
                "segments_count": info.segments_count if info.segments_count else 0,
                "config": {
                    "distance": info.config.params.vectors.distance.name,
                    "vector_size": info.config.params.vectors.size,
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    async def upsert_documents(
        self, 
        documents: List[DocumentChunk],
        batch_size: int = 100
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
            
            logger.debug(f"Upserting batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)")
            
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
                            **doc.metadata
                        }
                    )
                    points.append(point)
                
                if points:
                    # Upsert the batch
                    result = await self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    if result.status == UpdateStatus.COMPLETED:
                        total_upserted += len(points)
                        logger.debug(f"Successfully upserted batch {batch_idx + 1}")
                    else:
                        logger.error(f"Failed to upsert batch {batch_idx + 1}: {result.status}")
                
            except Exception as e:
                logger.error(f"Error upserting batch {batch_idx + 1}: {e}")
                # Continue with next batch rather than failing completely
        
        logger.info(f"Total documents upserted: {total_upserted}")
        return total_upserted
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[SearchMatch]:
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
            filter_conditions = []
            
            if source_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="source_url",
                        match={"any": source_filter}
                    )
                )
            
            if date_range:
                start_date, end_date = date_range
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(
                            gte=start_date,
                            lte=end_date
                        )
                    )
                )
            
            # Prepare search request
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search using modern query_points API
            query_response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Extract results from tuple (results, next_offset)
            results = query_response.points if hasattr(query_response, 'points') else query_response[0] if isinstance(query_response, (tuple, list)) else query_response
            
            # Convert results to SearchMatch objects
            matches = []
            for result in results:
                payload = result.payload
                
                # Reconstruct document chunk
                document = DocumentChunk(
                    id=str(result.id),
                    content=payload.get("content", ""),
                    source_url=payload.get("source_url", ""),
                    source_title=payload.get("source_title"),
                    chunk_index=payload.get("chunk_index", 0),
                    word_count=payload.get("word_count", 0),
                    char_count=payload.get("char_count", 0),
                    timestamp=payload.get("timestamp", ""),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["content", "source_url", "source_title", 
                                   "chunk_index", "word_count", "char_count", "timestamp"]
                    }
                )
                
                # Create search match
                match = SearchMatch(
                    document=document,
                    score=float(result.score)
                )
                matches.append(match)
            
            logger.debug(f"Found {len(matches)} similar documents")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            raise ToolError(f"Vector search failed: {str(e)}")
    
    async def get_document_by_id(self, document_id: str) -> Optional[DocumentChunk]:
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
                with_vectors=False
            )
            
            if not result:
                return None
            
            point = result[0]
            payload = point.payload
            
            return DocumentChunk(
                id=str(point.id),
                content=payload.get("content", ""),
                source_url=payload.get("source_url", ""),
                source_title=payload.get("source_title"),
                chunk_index=payload.get("chunk_index", 0),
                word_count=payload.get("word_count", 0),
                char_count=payload.get("char_count", 0),
                timestamp=payload.get("timestamp", ""),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["content", "source_url", "source_title", 
                               "chunk_index", "word_count", "char_count", "timestamp"]
                }
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
                            key="source_url",
                            match={"value": source_url}
                        )
                    ]
                )
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Deleted documents from source: {source_url}")
                return 1  # Qdrant doesn't return exact count for filter deletes
            else:
                logger.error(f"Failed to delete documents from source {source_url}: {result.status}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents from source {source_url}: {e}")
            return 0
    
    async def get_sources_stats(self) -> Dict[str, Any]:
        """
        Get statistics about sources in the vector database.
        
        Returns:
            Dictionary with source statistics
        """
        await self.ensure_collection()
        
        try:
            # Scroll through all points to collect statistics
            stats = {
                "total_documents": 0,
                "unique_sources": set(),
                "source_counts": {},
                "total_content_length": 0,
                "average_chunk_size": 0.0
            }
            
            offset = None
            limit = 1000
            
            while True:
                result = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = result[0]
                next_offset = result[1]
                
                if not points:
                    break
                
                for point in points:
                    payload = point.payload
                    source_url = payload.get("source_url", "unknown")
                    char_count = payload.get("char_count", 0)
                    
                    stats["total_documents"] += 1
                    stats["unique_sources"].add(source_url)
                    stats["source_counts"][source_url] = stats["source_counts"].get(source_url, 0) + 1
                    stats["total_content_length"] += char_count
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Calculate average chunk size
            if stats["total_documents"] > 0:
                stats["average_chunk_size"] = stats["total_content_length"] / stats["total_documents"]
            
            # Convert set to count
            stats["unique_sources"] = len(stats["unique_sources"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting source statistics: {e}")
            return {}