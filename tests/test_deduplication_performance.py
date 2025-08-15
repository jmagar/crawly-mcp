"""
Performance tests for deduplication functionality.
"""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from crawler_mcp.core.rag import RagService
from crawler_mcp.models.crawl import CrawlResult, CrawlStatus, PageContent


class TestDeduplicationPerformance:
    """Performance tests for deduplication functionality."""

    def _create_large_crawl_result(
        self, url_base: str, num_pages: int, content_base: str = "Test content"
    ) -> CrawlResult:
        """Create a large crawl result for performance testing.

        Note: All pages use the same base URL to simulate chunks from a single source,
        which is how deduplication works (per source URL).
        """
        pages = []
        for i in range(num_pages):
            pages.append(
                PageContent(
                    url=url_base,  # Use same URL for all pages to simulate single source
                    title=f"Test Page {i}",
                    content=f"{content_base} for page {i}"
                    + " " * (100 + i),  # Varying content length
                    word_count=20 + i,
                    metadata={
                        "page_number": i,
                        "chunk_metadata": {"chunk_index": i},
                    },  # Proper chunk_index location
                )
            )

        return CrawlResult(
            request_id=f"perf-test-{num_pages}",
            status=CrawlStatus.COMPLETED,
            urls=[url_base],  # Single source URL
            pages=pages,
        )

    def _setup_mock_services(
        self,
        rag_service: RagService,
        num_existing_chunks: int = 0,
        url_base: str = "https://example.com",
    ) -> None:
        """Setup mock services for performance testing."""
        rag_service.vector_service = AsyncMock()
        rag_service.embedding_service = AsyncMock()

        # Mock existing chunks if specified
        existing_chunks = []
        for i in range(num_existing_chunks):
            # Generate content that matches what _create_large_crawl_result creates
            content = f"Test content for page {i}" + " " * (100 + i)
            chunk_id = rag_service._generate_deterministic_id(
                url_base, i
            )  # Use chunk index from metadata
            content_hash = rag_service._calculate_content_hash(content)
            existing_chunks.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "content_hash": content_hash,
                }
            )

        rag_service.vector_service.get_chunks_by_source.return_value = existing_chunks

        # Mock embedding generation (simulate processing time)
        def mock_embedding_batch(texts):
            """Mock embedding generation with simulated processing time."""
            # Simulate some processing time (0.1ms per text)
            time.sleep(len(texts) * 0.0001)
            return [Mock(embedding=[0.1, 0.2, 0.3]) for _ in texts]

        rag_service.embedding_service.generate_embeddings_true_batch.side_effect = (
            mock_embedding_batch
        )
        rag_service.vector_service.upsert_documents.return_value = lambda chunks: len(
            chunks
        )
        rag_service.vector_service.delete_chunks_by_ids.return_value = 0

    @pytest.mark.asyncio
    async def test_performance_first_crawl_large(self):
        """Test performance of first crawl with large dataset."""
        rag_service = RagService()
        self._setup_mock_services(rag_service, num_existing_chunks=0)

        # Create large crawl result (100 pages)
        crawl_result = self._create_large_crawl_result("https://example.com", 100)

        # Test with deduplication enabled
        start_time = time.time()
        result_with_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )
        time_with_dedup = time.time() - start_time

        # Test without deduplication
        start_time = time.time()
        result_without_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=False
        )
        time_without_dedup = time.time() - start_time

        # Verify results
        assert result_with_dedup["chunks_created"] == 100
        assert result_without_dedup["chunks_created"] == 100

        # For first crawl (no existing chunks), deduplication should use fast path
        # and be similar in performance
        print("First crawl performance:")
        print(f"  With deduplication: {time_with_dedup:.3f}s")
        print(f"  Without deduplication: {time_without_dedup:.3f}s")
        print(f"  Overhead ratio: {time_with_dedup / time_without_dedup:.2f}x")

        # Deduplication should not add significant overhead for first crawl (fast path)
        assert time_with_dedup < time_without_dedup * 2.0, (
            "Deduplication should use fast path for first crawl"
        )

    @pytest.mark.asyncio
    async def test_performance_recrawl_no_changes(self):
        """Test performance of re-crawl with no changes (best case for deduplication)."""
        rag_service = RagService()

        # Create crawl result
        crawl_result = self._create_large_crawl_result("https://example.com", 50)

        # Setup with existing chunks that match the content
        self._setup_mock_services(rag_service, num_existing_chunks=50)

        # Test with deduplication (should skip all chunks)
        start_time = time.time()
        result_with_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )
        time_with_dedup = time.time() - start_time

        # Reset mocks and test without deduplication
        self._setup_mock_services(rag_service, num_existing_chunks=0)
        start_time = time.time()
        result_without_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=False
        )
        time_without_dedup = time.time() - start_time

        # Verify results
        assert result_with_dedup["chunks_created"] == 0
        assert result_with_dedup["chunks_skipped"] == 50
        assert result_without_dedup["chunks_created"] == 50

        print("Re-crawl (no changes) performance:")
        print(
            f"  With deduplication: {time_with_dedup:.3f}s ({result_with_dedup['chunks_skipped']} skipped)"
        )
        print(
            f"  Without deduplication: {time_without_dedup:.3f}s ({result_without_dedup['chunks_created']} created)"
        )
        print(f"  Speed improvement: {time_without_dedup / time_with_dedup:.2f}x")

        # Deduplication should be significantly faster when no changes detected
        assert time_with_dedup < time_without_dedup * 0.8, (
            "Deduplication should be faster for unchanged content"
        )

    @pytest.mark.asyncio
    async def test_performance_recrawl_partial_changes(self):
        """Test performance of re-crawl with partial changes."""
        rag_service = RagService()

        # Create crawl result with 30 pages
        crawl_result = self._create_large_crawl_result("https://example.com", 30)

        # Setup with existing chunks for first 20 pages (unchanged)
        # Last 10 pages will be new/changed
        self._setup_mock_services(rag_service, num_existing_chunks=20)

        # Test with deduplication
        start_time = time.time()
        result_with_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )
        time_with_dedup = time.time() - start_time

        # Reset and test without deduplication
        self._setup_mock_services(rag_service, num_existing_chunks=0)
        start_time = time.time()
        result_without_dedup = await rag_service.process_crawl_result(
            crawl_result, deduplication=False
        )
        time_without_dedup = time.time() - start_time

        # Verify results
        assert result_with_dedup["chunks_created"] == 10  # Only new/changed chunks
        assert result_with_dedup["chunks_skipped"] == 20  # Unchanged chunks
        assert result_without_dedup["chunks_created"] == 30  # All chunks

        print("Re-crawl (partial changes) performance:")
        print(
            f"  With deduplication: {time_with_dedup:.3f}s ({result_with_dedup['chunks_skipped']} skipped, {result_with_dedup['chunks_created']} created)"
        )
        print(
            f"  Without deduplication: {time_without_dedup:.3f}s ({result_without_dedup['chunks_created']} created)"
        )
        print(f"  Speed improvement: {time_without_dedup / time_with_dedup:.2f}x")

        # Deduplication should still be faster due to skipped processing
        assert time_with_dedup < time_without_dedup, (
            "Deduplication should be faster even with partial changes"
        )

    @pytest.mark.asyncio
    async def test_performance_orphan_cleanup(self):
        """Test performance of orphaned chunk cleanup."""
        rag_service = RagService()

        # Create crawl result with fewer pages than existing chunks
        crawl_result = self._create_large_crawl_result("https://example.com", 10)

        # Setup with more existing chunks (some will become orphans)
        self._setup_mock_services(rag_service, num_existing_chunks=20)
        rag_service.vector_service.delete_chunks_by_ids.return_value = (
            10  # 10 orphans deleted
        )

        # Test orphan cleanup performance
        start_time = time.time()
        result = await rag_service.process_crawl_result(
            crawl_result, deduplication=True
        )
        cleanup_time = time.time() - start_time

        # Verify orphan cleanup occurred
        assert result["chunks_deleted"] == 10
        assert result["chunks_skipped"] == 10  # Existing chunks that match

        print("Orphan cleanup performance:")
        print(f"  Time with cleanup: {cleanup_time:.3f}s")
        print(f"  Chunks deleted: {result['chunks_deleted']}")
        print(f"  Chunks skipped: {result['chunks_skipped']}")

        # Cleanup should complete in reasonable time
        assert cleanup_time < 1.0, "Orphan cleanup should complete quickly"

    def test_memory_usage_deterministic_ids(self):
        """Test memory usage of deterministic ID generation."""
        rag_service = RagService()

        # Generate many deterministic IDs and measure memory efficiency
        num_urls = 1000
        urls = [f"https://example.com/page-{i}" for i in range(num_urls)]

        # Test ID generation performance
        start_time = time.time()
        ids = []
        for i, url in enumerate(urls):
            chunk_id = rag_service._generate_deterministic_id(url, 0)
            ids.append(chunk_id)
        id_generation_time = time.time() - start_time

        # Test content hash generation performance
        start_time = time.time()
        hashes = []
        for i in range(num_urls):
            content = f"Content for page {i}"
            content_hash = rag_service._calculate_content_hash(content)
            hashes.append(content_hash)
        hash_generation_time = time.time() - start_time

        # Verify uniqueness and consistency
        assert len(set(ids)) == num_urls, "All IDs should be unique"
        assert len(set(hashes)) == num_urls, "All hashes should be unique"

        # Test ID consistency (same input = same output)
        duplicate_id = rag_service._generate_deterministic_id(urls[100], 0)
        assert duplicate_id == ids[100], "Same input should produce same ID"

        print("Memory usage performance:")
        print(
            f"  ID generation time: {id_generation_time:.3f}s for {num_urls} URLs ({id_generation_time / num_urls * 1000:.2f}ms per ID)"
        )
        print(
            f"  Hash generation time: {hash_generation_time:.3f}s for {num_urls} contents ({hash_generation_time / num_urls * 1000:.2f}ms per hash)"
        )

        # Performance should be reasonable
        assert id_generation_time < 0.5, "ID generation should be fast"
        assert hash_generation_time < 0.5, "Hash generation should be fast"

    def test_url_normalization_performance(self):
        """Test URL normalization performance with various URL types."""
        rag_service = RagService()

        # Test with various URL patterns
        test_urls = [
            "https://example.com/page",
            "http://EXAMPLE.COM/PAGE",
            "https://example.com/page/",
            "https://example.com/page?b=2&a=1&c=3",
            "https://example.com/page#section",
            "https://example.com/page?param=value&other=123#anchor",
        ] * 100  # Repeat for performance testing

        # Test normalization performance
        start_time = time.time()
        normalized_urls = []
        for url in test_urls:
            normalized = rag_service._normalize_url(url)
            normalized_urls.append(normalized)
        normalization_time = time.time() - start_time

        print("URL normalization performance:")
        print(
            f"  Time: {normalization_time:.3f}s for {len(test_urls)} URLs ({normalization_time / len(test_urls) * 1000:.2f}ms per URL)"
        )

        # Verify consistency (same logical URL should normalize to same result)
        url1 = rag_service._normalize_url("https://example.com/page?a=1&b=2")
        url2 = rag_service._normalize_url("https://example.com/page?b=2&a=1")
        assert url1 == url2, (
            "URLs with reordered params should normalize to same result"
        )

        # Performance should be reasonable
        assert normalization_time < 0.1, "URL normalization should be fast"

    @pytest.mark.asyncio
    async def test_performance_scaling(self):
        """Test how deduplication performance scales with dataset size."""
        rag_service = RagService()

        dataset_sizes = [10, 50, 100, 200]
        results = {}

        for size in dataset_sizes:
            # Test first crawl (no existing chunks)
            self._setup_mock_services(rag_service, num_existing_chunks=0)
            crawl_result = self._create_large_crawl_result("https://example.com", size)

            start_time = time.time()
            await rag_service.process_crawl_result(crawl_result, deduplication=True)
            processing_time = time.time() - start_time

            results[size] = {
                "time": processing_time,
                "time_per_chunk": processing_time / size,
            }

        print("Scaling performance:")
        for size, metrics in results.items():
            print(
                f"  {size} chunks: {metrics['time']:.3f}s total, {metrics['time_per_chunk'] * 1000:.1f}ms per chunk"
            )

        # Verify linear scaling (time per chunk should remain relatively constant)
        times_per_chunk = [metrics["time_per_chunk"] for metrics in results.values()]
        max_time_per_chunk = max(times_per_chunk)
        min_time_per_chunk = min(times_per_chunk)
        scaling_ratio = max_time_per_chunk / min_time_per_chunk

        print(f"  Scaling ratio: {scaling_ratio:.2f}x (lower is better)")

        # For small datasets, timing can be variable due to overhead
        # Scaling should be reasonable (ratio < 10x for acceptable performance)
        assert scaling_ratio < 10.0, "Performance scaling should be reasonable"
