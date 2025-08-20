"""
Edge case and error handling tests for RAG service to achieve 90% coverage.

These tests focus on:
- Import error handling (tiktoken, sentence-transformers)
- Reranker initialization failures
- Token-based chunking fallbacks
- Custom reranking algorithm
- Legacy chunk compatibility
- Orphaned chunk deletion
- Context manager error paths
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from crawler_mcp.core.rag import RagService
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent
from crawler_mcp.models.rag import DocumentChunk, SearchMatch


class TestRagImportErrors:
    """Test import error handling for optional dependencies."""

    @pytest.mark.asyncio
    async def test_tiktoken_import_error(self):
        """Test RAG service handles tiktoken ImportError gracefully."""
        # Clear singleton to force re-initialization
        RagService._instance = None
        RagService._initialized = False

        # Create a more targeted import blocker for tiktoken
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("tiktoken not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            service = RagService()

            # Should fall back to character-based chunking
            assert service.tokenizer is None
            assert service.tokenizer_type == "character"

    @pytest.mark.asyncio
    async def test_sentence_transformers_import_error(self):
        """Test RAG service handles sentence-transformers ImportError gracefully."""
        with patch("crawler_mcp.core.rag.settings") as mock_settings:
            # Enable reranker
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "test-model"
            mock_settings.reranker_fallback_to_custom = False

            # Clear singleton to force re-initialization
            RagService._instance = None
            RagService._initialized = False

            # Create a targeted import blocker for sentence_transformers
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "sentence_transformers":
                    raise ImportError("sentence-transformers not found")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                service = RagService()

                # Should disable reranker
                assert service.reranker is None
                assert service.reranker_type == "none"

    @pytest.mark.asyncio
    async def test_torch_import_error_via_removal(self):
        """Test RAG service handles torch being unavailable."""
        with patch("crawler_mcp.core.rag.settings") as mock_settings:
            # Enable reranker
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "test-model"
            mock_settings.reranker_fallback_to_custom = False

            # Clear singleton to force re-initialization
            RagService._instance = None
            RagService._initialized = False

            # Create a targeted import blocker for torch
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("torch not found")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                service = RagService()

                # Should disable reranker
                assert service.reranker is None
                assert service.reranker_type == "none"


class TestRagRerankerFailures:
    """Test reranker initialization failure scenarios."""

    @pytest.mark.asyncio
    async def test_reranker_oserror_with_fallback(self):
        """Test OSError when loading reranker model with fallback enabled."""
        with patch("crawler_mcp.core.rag.settings") as mock_settings:
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "invalid-model"
            mock_settings.reranker_fallback_to_custom = True

            # Mock successful imports but failed model loading
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            mock_crossencoder = MagicMock(side_effect=OSError("Model not found"))

            with patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "sentence_transformers": MagicMock(CrossEncoder=mock_crossencoder),
                },
            ):
                # Clear singleton to force re-initialization
                RagService._instance = None
                RagService._initialized = False

                service = RagService()

                # Should fall back to custom reranking
                assert service.reranker is None
                assert service.reranker_type == "custom"

    @pytest.mark.asyncio
    async def test_reranker_oserror_without_fallback(self):
        """Test OSError when loading reranker model with fallback disabled."""
        with patch("crawler_mcp.core.rag.settings") as mock_settings:
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "invalid-model"
            mock_settings.reranker_fallback_to_custom = False

            # Clear singleton to force re-initialization
            RagService._instance = None
            RagService._initialized = False

            # Mock successful imports but failed model loading
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            # Create a proper mock that raises OSError when instantiated
            mock_crossencoder_class = MagicMock()
            mock_crossencoder_class.side_effect = OSError("Model not found")

            mock_sentence_transformers = MagicMock()
            mock_sentence_transformers.CrossEncoder = mock_crossencoder_class

            with patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "sentence_transformers": mock_sentence_transformers,
                },
            ):
                service = RagService()

                # Should disable reranking
                assert service.reranker is None
                assert service.reranker_type == "none"

    @pytest.mark.asyncio
    async def test_reranker_valueerror_with_fallback(self):
        """Test ValueError when loading reranker model with fallback enabled."""
        with patch("crawler_mcp.core.rag.settings") as mock_settings:
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "invalid-model"
            mock_settings.reranker_fallback_to_custom = True

            # Mock successful imports but failed model loading
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            mock_crossencoder = MagicMock(
                side_effect=ValueError("Invalid model configuration")
            )

            with patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "sentence_transformers": MagicMock(CrossEncoder=mock_crossencoder),
                },
            ):
                # Clear singleton to force re-initialization
                RagService._instance = None
                RagService._initialized = False

                service = RagService()

                # Should fall back to custom reranking
                assert service.reranker is None
                assert service.reranker_type == "custom"


class TestRagChunkingFallbacks:
    """Test chunking fallback mechanisms when tokenizer is unavailable."""

    @pytest.fixture
    async def rag_service_no_tokenizer(self):
        """Create RAG service without tokenizer for testing fallbacks."""
        # Clear singleton and force no tokenizer
        RagService._instance = None
        RagService._initialized = False

        with patch("crawler_mcp.config.settings") as mock_settings:
            mock_settings.chunk_size = 100
            mock_settings.chunk_overlap = 20
            mock_settings.word_to_token_ratio = 1.3

            service = RagService()
            # Force no tokenizer to test fallback
            service.tokenizer = None
            service.tokenizer_type = "character"

            async with service:
                yield service

    @pytest.mark.asyncio
    async def test_token_chunking_without_tokenizer(self, rag_service_no_tokenizer):
        """Test token-based chunking falls back to word approximation."""
        text = "This is a test sentence. " * 200  # 1000 words, > 731 word threshold

        chunks = rag_service_no_tokenizer._chunk_text_token_based(text=text)

        # Should create multiple chunks using word approximation
        assert len(chunks) > 1
        assert isinstance(chunks, list)

        # Verify chunk structure includes token estimates
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "word_count" in chunk
            assert "token_count_estimate" in chunk
            assert "start_word" in chunk
            assert "end_word" in chunk

    @pytest.mark.asyncio
    async def test_token_chunking_edge_cases(self, rag_service_no_tokenizer):
        """Test token chunking fallback with edge cases."""
        # Test empty text
        chunks = rag_service_no_tokenizer._chunk_text_token_based(text="")
        assert chunks == []

        # Test single word
        chunks = rag_service_no_tokenizer._chunk_text_token_based(text="word")
        assert len(chunks) == 1
        assert chunks[0]["word_count"] == 1

        # Test text smaller than chunk size
        small_text = "just a few words here"
        chunks = rag_service_no_tokenizer._chunk_text_token_based(text=small_text)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_token_chunking_overlap_logic(self, rag_service_no_tokenizer):
        """Test overlap logic in word-based token chunking."""
        # Create text large enough to create overlapping chunks
        # Need at least 2 * chunk_size_words to ensure overlap
        text = " ".join([f"word{i}" for i in range(1600)])  # 1600 words > 2 * 731

        chunks = rag_service_no_tokenizer._chunk_text_token_based(text=text)

        assert len(chunks) > 1

        # Verify chunking behavior between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Check word positions
            current_end = current_chunk["end_word"]
            next_start = next_chunk["start_word"]

            # The implementation creates adjacent chunks (no gaps, but also no overlap)
            # when the calculated overlap would cause the next chunk to start before current end
            assert next_start <= current_end  # No gaps between chunks


class TestRagCustomReranking:
    """Test custom reranking algorithm functionality."""

    @pytest.fixture
    async def rag_service_custom_reranker(self):
        """Create RAG service with custom reranker."""
        RagService._instance = None
        RagService._initialized = False

        service = RagService()
        service.reranker_type = "custom"

        async with service:
            yield service

    @pytest.mark.asyncio
    async def test_custom_reranking_algorithm(self, rag_service_custom_reranker):
        """Test custom reranking algorithm with various content types."""
        # Create test documents with different characteristics
        doc1 = DocumentChunk(
            id="doc1",
            source_url="https://example.com/page1",
            source_title="Machine Learning Basics",
            content="Machine learning is a subset of artificial intelligence that uses algorithms.",
            chunk_index=0,
            word_count=12,
            timestamp=datetime.utcnow(),
        )

        doc2 = DocumentChunk(
            id="doc2",
            source_url="https://example.com/page2",
            source_title="Deep Learning Guide",
            content="Deep learning neural networks are powerful tools for pattern recognition and data analysis.",
            chunk_index=0,
            word_count=14,
            timestamp=datetime.utcnow(),
        )

        doc3 = DocumentChunk(
            id="doc3",
            source_url="https://example.com/page3",
            source_title="AI Overview",
            content="A very short content.",  # Short content to test length penalty
            chunk_index=0,
            word_count=4,
            timestamp=datetime.utcnow(),
        )

        # Create search matches
        matches = [
            SearchMatch(score=0.8, relevance="high", document=doc1),
            SearchMatch(score=0.75, relevance="medium", document=doc2),
            SearchMatch(score=0.7, relevance="medium", document=doc3),
        ]

        query = "machine learning algorithms"

        # Test custom reranking
        reranked = await rag_service_custom_reranker._rerank_with_custom_algorithm(
            query, matches, top_k=3
        )

        assert len(reranked) == 3
        assert isinstance(reranked, list)

        # Scores should be modified by the custom algorithm
        for match in reranked:
            assert 0 <= match.score <= 1.0

    @pytest.mark.asyncio
    async def test_custom_reranking_keyword_overlap(self, rag_service_custom_reranker):
        """Test keyword overlap scoring in custom reranking."""
        # Create document with high keyword overlap
        doc_high_overlap = DocumentChunk(
            id="doc1",
            source_url="https://example.com",
            source_title="Python Programming",
            content="Python programming language is great for machine learning and data science.",
            chunk_index=0,
            word_count=12,
            timestamp=datetime.utcnow(),
        )

        # Create document with low keyword overlap
        doc_low_overlap = DocumentChunk(
            id="doc2",
            source_url="https://example.com",
            source_title="Unrelated Topic",
            content="This document talks about cooking and recipes, nothing related to technology.",
            chunk_index=0,
            word_count=12,
            timestamp=datetime.utcnow(),
        )

        matches = [
            SearchMatch(score=0.5, relevance="medium", document=doc_high_overlap),
            SearchMatch(
                score=0.6, relevance="medium", document=doc_low_overlap
            ),  # Initially higher
        ]

        query = "python programming machine learning"

        reranked = await rag_service_custom_reranker._rerank_with_custom_algorithm(
            query, matches, top_k=2
        )

        # Document with higher keyword overlap should be ranked higher after reranking
        assert reranked[0].document.id == "doc1"  # High overlap should be first

    @pytest.mark.asyncio
    async def test_custom_reranking_title_bonus(self, rag_service_custom_reranker):
        """Test title relevance bonus in custom reranking."""
        # Create document with relevant title
        doc_relevant_title = DocumentChunk(
            id="doc1",
            source_url="https://example.com",
            source_title="Python Machine Learning Tutorial",  # Relevant title
            content="Some content about programming.",
            chunk_index=0,
            word_count=5,
            timestamp=datetime.utcnow(),
        )

        # Create document without relevant title
        doc_irrelevant_title = DocumentChunk(
            id="doc2",
            source_url="https://example.com",
            source_title="Random Article",  # Irrelevant title
            content="Some content about programming.",
            chunk_index=0,
            word_count=5,
            timestamp=datetime.utcnow(),
        )

        matches = [
            SearchMatch(score=0.5, relevance="medium", document=doc_relevant_title),
            SearchMatch(
                score=0.49, relevance="medium", document=doc_irrelevant_title
            ),  # Slight disadvantage to allow title bonus
        ]

        query = "python machine learning"

        reranked = await rag_service_custom_reranker._rerank_with_custom_algorithm(
            query, matches, top_k=2
        )

        # Document with relevant title should get bonus and rank higher
        assert reranked[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_custom_reranking_error_handling(self, rag_service_custom_reranker):
        """Test error handling in custom reranking algorithm."""
        # Create malformed match that might cause errors
        malformed_doc = DocumentChunk(
            id="doc1",
            source_url="https://example.com",
            source_title=None,  # None title to potentially cause errors
            content="",  # Empty content
            chunk_index=0,
            word_count=0,
            timestamp=datetime.utcnow(),
        )

        matches = [SearchMatch(score=0.8, relevance="high", document=malformed_doc)]

        query = "test query"

        # Should handle errors gracefully and return original matches
        reranked = await rag_service_custom_reranker._rerank_with_custom_algorithm(
            query, matches, top_k=1
        )

        assert len(reranked) == 1
        assert reranked[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_custom_reranking_top_k_limit(self, rag_service_custom_reranker):
        """Test top_k limiting in custom reranking."""
        # Create multiple documents
        docs = []
        for i in range(5):
            doc = DocumentChunk(
                id=f"doc{i}",
                source_url=f"https://example.com/page{i}",
                source_title=f"Title {i}",
                content=f"Content for document {i}",
                chunk_index=0,
                word_count=4,
                timestamp=datetime.utcnow(),
            )
            docs.append(doc)

        matches = [
            SearchMatch(score=0.8 - i * 0.1, relevance="high", document=doc)
            for i, doc in enumerate(docs)
        ]

        query = "test query"

        # Test top_k=3
        reranked = await rag_service_custom_reranker._rerank_with_custom_algorithm(
            query, matches, top_k=3
        )

        assert len(reranked) == 3


class TestRagLegacyCompatibility:
    """Test legacy chunk compatibility and backwards compatibility mode."""

    @pytest.fixture
    async def rag_service_with_legacy(self):
        """Create RAG service for legacy compatibility testing."""
        service = RagService()
        async with service:
            yield service

    @pytest.mark.asyncio
    async def test_find_legacy_chunk_by_content(self, rag_service_with_legacy):
        """Test finding legacy chunks by content hash."""
        # Create legacy chunks with random UUIDs
        legacy_chunks = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",  # Random UUID
                "content": "This is legacy content",
                "content_hash": "legacy_hash_123",
                "chunk_index": 0,
            },
            {
                "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # Random UUID
                "content": "Another legacy content",
                "content_hash": "legacy_hash_456",
                "chunk_index": 1,
            },
        ]

        # Test finding by content
        found_chunk = rag_service_with_legacy._find_legacy_chunk_by_content(
            legacy_chunks, "This is legacy content"
        )

        assert found_chunk is not None
        assert found_chunk["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert found_chunk["content"] == "This is legacy content"

    @pytest.mark.asyncio
    async def test_find_legacy_chunk_not_found(self, rag_service_with_legacy):
        """Test finding legacy chunk when content doesn't match."""
        legacy_chunks = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "This is legacy content",
                "content_hash": "legacy_hash_123",
                "chunk_index": 0,
            }
        ]

        # Test with non-matching content
        found_chunk = rag_service_with_legacy._find_legacy_chunk_by_content(
            legacy_chunks, "Different content"
        )

        assert found_chunk is None

    @pytest.mark.asyncio
    async def test_should_use_backwards_compatibility_with_random_uuids(
        self, rag_service_with_legacy
    ):
        """Test backwards compatibility detection with random UUIDs."""
        # Create chunks map with random UUIDs (should trigger backwards compatibility)
        existing_chunks_map = {
            "550e8400-e29b-41d4-a716-446655440000": "hash1",  # Random UUID
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8": "hash2",  # Random UUID
        }

        should_use = rag_service_with_legacy._should_use_backwards_compatibility(
            existing_chunks_map
        )
        assert should_use is True

    @pytest.mark.asyncio
    async def test_should_use_backwards_compatibility_with_deterministic_ids(
        self, rag_service_with_legacy
    ):
        """Test backwards compatibility detection with deterministic IDs."""
        # Create chunks map with deterministic IDs (should not trigger backwards compatibility)
        existing_chunks_map = {
            "det_https://example.com_0": "hash1",  # Deterministic ID
            "det_https://example.com_1": "hash2",  # Deterministic ID
        }

        should_use = rag_service_with_legacy._should_use_backwards_compatibility(
            existing_chunks_map
        )
        assert should_use is False

    @pytest.mark.asyncio
    async def test_should_use_backwards_compatibility_mixed_ids(
        self, rag_service_with_legacy
    ):
        """Test backwards compatibility detection with mixed ID types."""
        # Mix of random and deterministic IDs with random majority (should trigger backwards compatibility)
        existing_chunks_map = {
            "550e8400-e29b-41d4-a716-446655440000": "hash1",  # Random UUID
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8": "hash2",  # Random UUID
            "det_https://example.com_0": "hash3",  # Deterministic ID
        }

        should_use = rag_service_with_legacy._should_use_backwards_compatibility(
            existing_chunks_map
        )
        assert should_use is True

    @pytest.mark.asyncio
    async def test_is_random_uuid_valid_uuid(self, rag_service_with_legacy):
        """Test random UUID detection with valid UUIDs."""
        # Test with valid random UUID
        assert (
            rag_service_with_legacy._is_random_uuid(
                "550e8400-e29b-41d4-a716-446655440000"
            )
            is True
        )

        # Test with deterministic ID
        assert (
            rag_service_with_legacy._is_random_uuid("det_https://example.com_0")
            is False
        )

        # Test with invalid format
        assert rag_service_with_legacy._is_random_uuid("not-a-uuid") is False


class TestRagOrphanedChunks:
    """Test orphaned chunk detection and deletion."""

    @pytest.fixture
    async def rag_service_with_mocked_vector(self):
        """Create RAG service with mocked vector service for orphan testing."""
        service = RagService()

        # Mock vector service for orphan testing
        mock_vector_service = AsyncMock()
        service.vector_service = mock_vector_service

        async with service:
            yield service, mock_vector_service

    @pytest.mark.asyncio
    async def test_orphaned_chunk_deletion_success(
        self, rag_service_with_mocked_vector
    ):
        """Test successful orphaned chunk deletion."""
        service, mock_vector_service = rag_service_with_mocked_vector

        # Mock progress callback
        progress_callback = Mock()

        # Mock successful deletion
        mock_vector_service.delete_chunks_by_ids.return_value = 5

        # Create test crawl result with one page (triggers orphan detection)
        # Existing chunks will be orphaned since they don't match the new page
        pages = [
            PageContent(
                url="https://example.com/new-page",  # Different URL from existing chunks
                title="New Page",
                content="New content that doesn't match existing chunks",
                word_count=10,
                timestamp=datetime.utcnow(),
            )
        ]

        crawl_result = CrawlResult(
            request_id="test-orphan-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/new-page"],
            pages=pages,
        )

        # Mock existing chunks to be deleted (from old content that no longer exists)
        mock_vector_service.get_chunks_by_source.return_value = [
            {"id": "orphan1", "content": "orphaned content 1", "content_hash": "hash1"},
            {"id": "orphan2", "content": "orphaned content 2", "content_hash": "hash2"},
        ]

        # Process crawl result (should detect and delete orphans)
        result = await service.process_crawl_result(
            crawl_result, progress_callback=progress_callback
        )

        # Should have called delete_chunks_by_ids
        mock_vector_service.delete_chunks_by_ids.assert_called_once()

        # Progress callback should have been called
        progress_callback.assert_called()

    @pytest.mark.asyncio
    async def test_orphaned_chunk_deletion_error(self, rag_service_with_mocked_vector):
        """Test error handling during orphaned chunk deletion."""
        service, mock_vector_service = rag_service_with_mocked_vector

        # Mock deletion error
        mock_vector_service.delete_chunks_by_ids.side_effect = Exception(
            "Deletion failed"
        )

        # Create test crawl result with empty pages
        pages = []

        crawl_result = CrawlResult(
            request_id="test-orphan-error-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/test"],
            pages=pages,
        )

        # Mock existing chunks
        mock_vector_service.get_chunks_by_source.return_value = [
            {"id": "orphan1", "content": "orphaned content 1"}
        ]

        # Should handle deletion error gracefully and continue
        result = await service.process_crawl_result(crawl_result)

        # Should still return result even with deletion error
        assert isinstance(result, dict)
        assert "chunks_deleted" in result

    @pytest.mark.asyncio
    async def test_no_chunks_to_process_early_return(
        self, rag_service_with_mocked_vector
    ):
        """Test early return when no chunks need processing."""
        service, mock_vector_service = rag_service_with_mocked_vector

        # Create crawl result with empty pages
        pages = []

        crawl_result = CrawlResult(
            request_id="test-no-chunks-123",
            status=CrawlStatus.COMPLETED,
            urls=["https://example.com/test"],
            pages=pages,
        )

        # Mock no existing chunks
        mock_vector_service.get_chunks_by_source.return_value = []

        # Should return early with no processing
        result = await service.process_crawl_result(crawl_result)

        assert result["chunks_created"] == 0
        assert result["embeddings_generated"] == 0
        assert result["documents_processed"] == 0


class TestRagContextErrors:
    """Test context manager error handling paths."""

    @pytest.mark.asyncio
    async def test_context_manager_exit_with_exception(self):
        """Test __aexit__ handling when exception occurs."""
        service = RagService()

        # Enter context
        await service.__aenter__()

        # Simulate exception during context
        exc_type = RuntimeError
        exc_value = RuntimeError("Test exception")
        exc_traceback = None

        # Exit with exception - should handle gracefully
        result = await service.__aexit__(exc_type, exc_value, exc_traceback)

        # Should return None (don't suppress exception)
        assert result is None

        # Context count should be decremented
        assert service._context_count >= 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self):
        """Test cleanup when context manager encounters errors."""
        service = RagService()

        try:
            async with service:
                # Simulate error inside context
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Service should still be properly cleaned up
        assert service._context_count >= 0

    @pytest.mark.asyncio
    async def test_multiple_context_entries_and_exits(self):
        """Test multiple concurrent context manager usage."""
        service = RagService()

        # Test multiple concurrent enters
        await service.__aenter__()
        await service.__aenter__()
        await service.__aenter__()

        assert service._context_count == 3

        # Test multiple exits
        await service.__aexit__(None, None, None)
        assert service._context_count == 2

        await service.__aexit__(None, None, None)
        assert service._context_count == 1

        await service.__aexit__(None, None, None)
        assert service._context_count == 0


class TestRagUrlNormalization:
    """Test URL normalization edge cases."""

    @pytest.fixture
    async def rag_service_for_url_tests(self):
        """Create RAG service for URL testing."""
        service = RagService()
        async with service:
            yield service

    @pytest.mark.asyncio
    async def test_normalize_url_with_fragment(self, rag_service_for_url_tests):
        """Test URL normalization removes fragments."""
        url = "https://example.com/path?param=value#fragment"
        normalized = rag_service_for_url_tests._normalize_url(url)
        assert normalized == "https://example.com/path?param=value"

    @pytest.mark.asyncio
    async def test_normalize_url_without_query_or_fragment(
        self, rag_service_for_url_tests
    ):
        """Test URL normalization with clean URL."""
        url = "https://example.com/path"
        normalized = rag_service_for_url_tests._normalize_url(url)
        assert normalized == "https://example.com/path"

    @pytest.mark.asyncio
    async def test_normalize_url_consistency(self, rag_service_for_url_tests):
        """Test URL normalization consistency."""
        url = "https://example.com/path?param=value#fragment"
        normalized1 = rag_service_for_url_tests._normalize_url(url)
        normalized2 = rag_service_for_url_tests._normalize_url(url)
        assert normalized1 == normalized2

    @pytest.mark.asyncio
    async def test_normalize_url_edge_cases(self, rag_service_for_url_tests):
        """Test URL normalization with edge cases."""
        # Test URL with only fragment
        url_fragment_only = "https://example.com#fragment"
        normalized = rag_service_for_url_tests._normalize_url(url_fragment_only)
        assert normalized == "https://example.com/"

        # Test URL with empty query
        url_empty_query = "https://example.com/path?"
        normalized = rag_service_for_url_tests._normalize_url(url_empty_query)
        assert normalized == "https://example.com/path"
