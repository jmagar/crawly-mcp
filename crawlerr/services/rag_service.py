"""
Service for RAG (Retrieval-Augmented Generation) operations.
"""
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import tiktoken
from ..config import settings
from ..models.rag_models import (
    RagQuery, RagResult, SearchMatch, DocumentChunk, EmbeddingResult
)
from ..models.crawl_models import PageContent, CrawlResult
from .embedding_service import EmbeddingService
from .vector_service import VectorService
from fastmcp.exceptions import ToolError


logger = logging.getLogger(__name__)


class RagService:
    """
    Service for RAG operations combining embedding generation and vector search.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        
        # Token-based chunking configuration
        self.chunk_size_tokens = 1024  # Tokens per chunk (optimal for embeddings)
        self.chunk_overlap_tokens = 100  # Token overlap between chunks (10% of chunk size)
        
        # Initialize tokenizer for the embedding model
        try:
            # Try to use Qwen's tokenizer first (matches our embedding model)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(settings.tei_model)
            self.tokenizer_type = "qwen"
            logger.info(f"Using Qwen tokenizer for {settings.tei_model}")
        except Exception as e:
            logger.warning(f"Failed to load Qwen tokenizer: {e}. Trying tiktoken fallback.")
            try:
                # Fallback to cl100k_base encoding for general compatibility
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenizer_type = "tiktoken"
                logger.info("Using tiktoken cl100k_base encoding as fallback")
            except Exception as e2:
                logger.warning(f"Failed to load tiktoken: {e2}. Using character-based chunking.")
                self.tokenizer = None
                self.tokenizer_type = "character"
                # Fallback to character-based for backward compatibility
                self.chunk_size = 1000  # Characters per chunk
                self.chunk_overlap = 200  # Overlap between chunks
        
        # Initialize Qwen3 reranker
        self.reranker = None
        self.reranker_type = "none"
        
        if settings.reranker_enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(settings.reranker_model)
                self.reranker_type = "qwen3"
                logger.info(f"Using Qwen3 reranker: {settings.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load Qwen3 reranker: {e}. Using custom reranking fallback.")
                if settings.reranker_fallback_to_custom:
                    self.reranker_type = "custom"
                    logger.info("Using custom reranking algorithm as fallback")
                else:
                    self.reranker_type = "none"
                    logger.info("Reranking disabled due to model loading failure")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.embedding_service.__aenter__()
        await self.vector_service.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.embedding_service.__aexit__(exc_type, exc_val, exc_tb)
        await self.vector_service.__aexit__(exc_type, exc_val, exc_tb)
    
    async def close(self):
        """Close all services."""
        await self.embedding_service.close()
        await self.vector_service.close()
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all dependent services.
        
        Returns:
            Dictionary with service health status
        """
        return {
            "embedding_service": await self.embedding_service.health_check(),
            "vector_service": await self.vector_service.health_check(),
        }
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding using token-based chunking with semantic awareness.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Fallback to character-based if tokenizer unavailable
        if self.tokenizer is None:
            return self._chunk_text_character_based(text, metadata)
        
        return self._chunk_text_token_based(text, metadata)
    
    def _chunk_text_token_based(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Token-based chunking with semantic awareness.
        """
        chunks = []
        chunk_index = 0
        
        # Tokenize the entire text based on tokenizer type
        if self.tokenizer_type == "qwen":
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:  # tiktoken
            tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size_tokens:
            # Text fits in single chunk
            chunk = {
                'text': text.strip(),
                'chunk_index': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'token_count': total_tokens,
                'char_count': len(text),
                'word_count': len(text.split()),
                **(metadata or {})
            }
            return [chunk]
        
        start_token_idx = 0
        
        while start_token_idx < total_tokens:
            end_token_idx = min(start_token_idx + self.chunk_size_tokens, total_tokens)
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_token_idx:end_token_idx]
            if self.tokenizer_type == "qwen":
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            else:  # tiktoken
                chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Apply semantic-aware boundary detection
            if end_token_idx < total_tokens:  # Not the last chunk
                chunk_text, adjusted_end_idx = self._find_semantic_boundary(
                    text, chunk_text, start_token_idx, end_token_idx, tokens
                )
                end_token_idx = adjusted_end_idx
            
            if chunk_text.strip():
                # Calculate positions in original text
                if start_token_idx > 0:
                    if self.tokenizer_type == "qwen":
                        start_char_pos = len(self.tokenizer.decode(tokens[:start_token_idx], skip_special_tokens=True))
                    else:  # tiktoken
                        start_char_pos = len(self.tokenizer.decode(tokens[:start_token_idx]))
                else:
                    start_char_pos = 0
                    
                if self.tokenizer_type == "qwen":
                    end_char_pos = len(self.tokenizer.decode(tokens[:end_token_idx], skip_special_tokens=True))
                else:  # tiktoken
                    end_char_pos = len(self.tokenizer.decode(tokens[:end_token_idx]))
                
                chunk = {
                    'text': chunk_text.strip(),
                    'chunk_index': chunk_index,
                    'start_pos': start_char_pos,
                    'end_pos': end_char_pos,
                    'token_count': end_token_idx - start_token_idx,
                    'char_count': len(chunk_text.strip()),
                    'word_count': len(chunk_text.strip().split()),
                    **(metadata or {})
                }
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            # Only apply overlap if we have more content to process
            if end_token_idx < total_tokens:
                start_token_idx = end_token_idx - self.chunk_overlap_tokens
                # Ensure we always make progress (avoid infinite loops)
                if start_token_idx <= chunk_index * 10:  # Safety check
                    start_token_idx = end_token_idx
            else:
                break  # We've processed all tokens
        
        return chunks
    
    def _find_semantic_boundary(
        self, 
        full_text: str, 
        chunk_text: str, 
        start_token_idx: int, 
        end_token_idx: int, 
        tokens: List[int]
    ) -> Tuple[str, int]:
        """
        Find optimal semantic boundary for chunk splitting.
        
        Returns:
            Tuple of (adjusted_chunk_text, adjusted_end_token_idx)
        """
        # Convert back to character positions to work with text
        if start_token_idx > 0:
            if self.tokenizer_type == "qwen":
                start_char = len(self.tokenizer.decode(tokens[:start_token_idx], skip_special_tokens=True))
            else:  # tiktoken
                start_char = len(self.tokenizer.decode(tokens[:start_token_idx]))
        else:
            start_char = 0
            
        if self.tokenizer_type == "qwen":
            end_char = len(self.tokenizer.decode(tokens[:end_token_idx], skip_special_tokens=True))
        else:  # tiktoken
            end_char = len(self.tokenizer.decode(tokens[:end_token_idx]))
        
        # Look for good break points in order of preference
        search_text = full_text[start_char:min(end_char + 200, len(full_text))]
        ideal_end = end_char - start_char
        
        # 1. Paragraph breaks (double newlines)
        paragraph_breaks = [i for i, char in enumerate(search_text) if search_text[i:i+2] == '\n\n']
        suitable_paragraph_breaks = [b for b in paragraph_breaks if ideal_end - 100 <= b <= ideal_end + 100]
        
        if suitable_paragraph_breaks:
            best_break = max(suitable_paragraph_breaks)
            adjusted_text = search_text[:best_break].strip()
            # Convert back to token index
            if self.tokenizer_type == "qwen":
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break], add_special_tokens=False)
            else:  # tiktoken
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break])
            return adjusted_text, start_token_idx + len(adjusted_tokens)
        
        # 2. Sentence endings
        sentence_patterns = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        sentence_breaks = []
        for pattern in sentence_patterns:
            sentence_breaks.extend([
                i + len(pattern) for i in range(len(search_text) - len(pattern)) 
                if search_text[i:i+len(pattern)] == pattern
            ])
        
        suitable_sentence_breaks = [b for b in sentence_breaks if ideal_end - 50 <= b <= ideal_end + 50]
        if suitable_sentence_breaks:
            best_break = max(suitable_sentence_breaks)
            adjusted_text = search_text[:best_break].strip()
            if self.tokenizer_type == "qwen":
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break], add_special_tokens=False)
            else:  # tiktoken
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break])
            return adjusted_text, start_token_idx + len(adjusted_tokens)
        
        # 3. Line breaks
        line_breaks = [i + 1 for i, char in enumerate(search_text) if char == '\n']
        suitable_line_breaks = [b for b in line_breaks if ideal_end - 30 <= b <= ideal_end + 30]
        
        if suitable_line_breaks:
            best_break = max(suitable_line_breaks)
            adjusted_text = search_text[:best_break].strip()
            if self.tokenizer_type == "qwen":
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break], add_special_tokens=False)
            else:  # tiktoken
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break])
            return adjusted_text, start_token_idx + len(adjusted_tokens)
        
        # 4. Word boundaries (spaces)
        word_breaks = [i + 1 for i, char in enumerate(search_text) if char == ' ']
        suitable_word_breaks = [b for b in word_breaks if ideal_end - 20 <= b <= ideal_end + 20]
        
        if suitable_word_breaks:
            best_break = max(suitable_word_breaks)
            adjusted_text = search_text[:best_break].strip()
            if self.tokenizer_type == "qwen":
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break], add_special_tokens=False)
            else:  # tiktoken
                adjusted_tokens = self.tokenizer.encode(full_text[start_char:start_char + best_break])
            return adjusted_text, start_token_idx + len(adjusted_tokens)
        
        # Fallback: use original chunk
        return chunk_text, end_token_idx
    
    def _chunk_text_character_based(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Fallback character-based chunking for backward compatibility.
        """
        if not hasattr(self, 'chunk_size'):
            self.chunk_size = 1000
            self.chunk_overlap = 200
        
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
                    if text[i] in '.!?':
                        # Check if it's likely end of sentence (followed by space/newline)
                        if i + 1 < text_length and text[i + 1] in ' \n\t':
                            sentence_ends.append(i + 1)
                
                # Use the last sentence end if found, otherwise stick to character limit
                if sentence_ends:
                    end = sentence_ends[-1]
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'start_pos': start,
                    'end_pos': end,
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text),
                    'token_count': len(chunk_text.split()) * 1.3 if self.tokenizer is None else (
                        len(self.tokenizer.encode(chunk_text, add_special_tokens=False)) 
                        if self.tokenizer_type == "qwen" 
                        else len(self.tokenizer.encode(chunk_text))
                    ),  # Accurate token count
                    **(metadata or {})
                }
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
            # Prevent infinite loop
            if start <= end - self.chunk_size:
                start = end
        
        return chunks
    
    async def process_crawl_result(
        self,
        crawl_result: CrawlResult,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, int]:
        """
        Process a crawl result by chunking content and generating embeddings.
        
        Args:
            crawl_result: Result from crawler service
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with processing statistics
        """
        if not crawl_result.pages:
            return {"documents_processed": 0, "chunks_created": 0, "embeddings_generated": 0}
        
        total_pages = len(crawl_result.pages)
        total_chunks = 0
        total_embeddings = 0
        document_chunks = []
        
        logger.info(f"Processing {total_pages} pages for RAG indexing")
        
        # Process each page
        for i, page in enumerate(crawl_result.pages):
            try:
                if progress_callback:
                    await progress_callback(
                        i, total_pages, 
                        f"Processing page {i+1}/{total_pages}: {page.url}"
                    )
                
                # Chunk the page content
                chunks = self.chunk_text(
                    page.content,
                    metadata={
                        'source_url': page.url,
                        'source_title': page.title,
                        'page_metadata': page.metadata
                    }
                )
                
                total_chunks += len(chunks)
                
                # Create DocumentChunk objects for each chunk
                for chunk_data in chunks:
                    chunk_id = f"{uuid.uuid4()}"
                    
                    doc_chunk = DocumentChunk(
                        id=chunk_id,
                        content=chunk_data['text'],
                        source_url=chunk_data['source_url'],
                        source_title=chunk_data.get('source_title'),
                        chunk_index=chunk_data['chunk_index'],
                        word_count=chunk_data['word_count'],
                        char_count=chunk_data['char_count'],
                        metadata={
                            'start_pos': chunk_data['start_pos'],
                            'end_pos': chunk_data['end_pos'],
                            **chunk_data.get('page_metadata', {})
                        }
                    )
                    
                    document_chunks.append(doc_chunk)
                
            except Exception as e:
                logger.error(f"Error processing page {page.url}: {e}")
                continue
        
        if not document_chunks:
            logger.warning("No document chunks created from crawl result")
            return {"documents_processed": 0, "chunks_created": 0, "embeddings_generated": 0}
        
        # Generate embeddings in batches
        if progress_callback:
            await progress_callback(
                total_pages, total_pages + 1,
                f"Generating embeddings for {len(document_chunks)} chunks"
            )
        
        try:
            # Extract texts for embedding
            texts = [chunk.content for chunk in document_chunks]
            
            # Generate embeddings
            embedding_results = await self.embedding_service.generate_embeddings_batch(
                texts, batch_size=settings.tei_batch_size
            )
            
            # Attach embeddings to document chunks
            for chunk, embedding_result in zip(document_chunks, embedding_results):
                chunk.embedding = embedding_result.embedding
            
            total_embeddings = len(embedding_results)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ToolError(f"Failed to generate embeddings: {str(e)}")
        
        # Store in vector database
        if progress_callback:
            await progress_callback(
                total_pages + 1, total_pages + 2,
                f"Storing {len(document_chunks)} embeddings in vector database"
            )
        
        try:
            stored_count = await self.vector_service.upsert_documents(document_chunks)
            logger.info(f"Stored {stored_count} document chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise ToolError(f"Failed to store embeddings: {str(e)}")
        
        return {
            "documents_processed": total_pages,
            "chunks_created": total_chunks,
            "embeddings_generated": total_embeddings,
            "chunks_stored": stored_count
        }
    
    async def query(
        self,
        query: RagQuery,
        rerank: bool = True
    ) -> RagResult:
        """
        Perform a RAG query to find relevant documents.
        
        Args:
            query: The RAG query parameters
            rerank: Whether to apply re-ranking to results
            
        Returns:
            RagResult with matched documents
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            embedding_start = time.time()
            embedding_result = await self.embedding_service.generate_embedding(query.query)
            embedding_time = time.time() - embedding_start
            
            # Search vector database
            search_start = time.time()
            search_matches = await self.vector_service.search_similar(
                query_vector=embedding_result.embedding,
                limit=query.limit,
                score_threshold=query.min_score,
                source_filter=query.source_filters,
                date_range=query.date_range
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
                rerank_time=rerank_time
            )
            
            logger.info(
                f"RAG query completed: {len(filtered_matches)} matches in {processing_time:.3f}s "
                f"(embed: {embedding_time:.3f}s, search: {search_time:.3f}s"
                f"{f', rerank: {rerank_time:.3f}s' if rerank_time else ''})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise ToolError(f"RAG query failed: {str(e)}")
    
    async def _rerank_results(
        self,
        query: str,
        matches: List[SearchMatch],
        top_k: Optional[int] = None
    ) -> List[SearchMatch]:
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
            elif self.reranker_type == "custom":
                return await self._rerank_with_custom_algorithm(query, matches, top_k)
            else:
                return matches  # No reranking
                
        except Exception as e:
            logger.warning(f"Re-ranking failed, returning original results: {e}")
            return matches
    
    async def _rerank_with_qwen3(
        self,
        query: str,
        matches: List[SearchMatch],
        top_k: Optional[int] = None
    ) -> List[SearchMatch]:
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
                content = match.document.content[:settings.reranker_max_length]
                pairs.append([query, content])
            
            # Get reranking scores from Qwen3 model
            # Process pairs individually to avoid padding issues
            import asyncio
            loop = asyncio.get_event_loop()
            
            scores = []
            for pair in pairs:
                score = await loop.run_in_executor(
                    None, 
                    lambda p=pair: self.reranker.predict([p])[0]
                )
                scores.append(score)
            
            # Update match scores with reranker predictions
            for match, score in zip(matches, scores):
                # Convert reranker logits to probability-like scores
                reranker_score = float(score)
                # Combine original embedding score with reranker score
                # Weight reranker more heavily as it's more sophisticated
                match.score = 0.3 * match.score + 0.7 * max(0.0, min(1.0, (reranker_score + 5) / 10))
            
            # Sort by updated scores
            matches.sort(key=lambda m: m.score, reverse=True)
            
            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]
            
            logger.debug(f"Qwen3 reranked {len(matches)} results")
            return matches
            
        except Exception as e:
            logger.error(f"Qwen3 reranking failed: {e}")
            # Fallback to custom algorithm if available
            if settings.reranker_fallback_to_custom:
                logger.info("Falling back to custom reranking algorithm")
                return await self._rerank_with_custom_algorithm(query, matches, top_k)
            return matches
    
    async def _rerank_with_custom_algorithm(
        self,
        query: str,
        matches: List[SearchMatch],
        top_k: Optional[int] = None
    ) -> List[SearchMatch]:
        """
        Re-rank using custom hybrid scoring algorithm (original implementation).
        """
        try:
            # Simple re-ranking based on keyword overlap and length
            query_words = set(query.lower().split())
            
            for match in matches:
                content_words = set(match.document.content.lower().split())
                
                # Keyword overlap score
                keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
                
                # Length penalty (prefer moderate length chunks)
                optimal_length = 500  # characters
                length_penalty = 1.0 - abs(len(match.document.content) - optimal_length) / optimal_length
                length_penalty = max(0.1, min(1.0, length_penalty))
                
                # Title relevance bonus
                title_bonus = 0.0
                if match.document.source_title:
                    title_words = set(match.document.source_title.lower().split())
                    title_overlap = len(query_words.intersection(title_words)) / len(query_words)
                    title_bonus = title_overlap * 0.1
                
                # Combine scores (weighted average)
                combined_score = (
                    match.score * 0.7 +  # Vector similarity (primary)
                    keyword_overlap * 0.2 +  # Keyword overlap
                    length_penalty * 0.05 +  # Length preference
                    title_bonus * 0.05  # Title relevance
                )
                
                match.score = min(1.0, combined_score)
            
            # Sort by combined score
            matches.sort(key=lambda m: m.score, reverse=True)
            
            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]
            
            logger.debug(f"Custom algorithm reranked {len(matches)} results")
            return matches
            
        except Exception as e:
            logger.warning(f"Custom reranking failed: {e}")
            return matches
    
    async def get_stats(self) -> Dict[str, Any]:
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
                "config": {
                    "chunking_method": getattr(self, 'tokenizer_type', 'character_based'),
                    "chunk_size_tokens": self.chunk_size_tokens if self.tokenizer else None,
                    "chunk_overlap_tokens": self.chunk_overlap_tokens if self.tokenizer else None,
                    "chunk_size_chars": getattr(self, 'chunk_size', None) if not self.tokenizer else None,
                    "chunk_overlap_chars": getattr(self, 'chunk_overlap', None) if not self.tokenizer else None,
                    "tokenizer_type": getattr(self, 'tokenizer_type', 'character_based'),
                    "reranker_type": getattr(self, 'reranker_type', 'none'),
                    "reranker_model": settings.reranker_model if self.reranker_type == "qwen3" else None,
                    "reranker_enabled": settings.reranker_enabled,
                    "embedding_model": settings.tei_model,
                    "vector_dimension": settings.qdrant_vector_size,
                    "distance_metric": settings.qdrant_distance
                }
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
            deleted_count = await self.vector_service.delete_documents_by_source(source_url)
            logger.info(f"Deleted documents from source: {source_url}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting source {source_url}: {e}")
            return False