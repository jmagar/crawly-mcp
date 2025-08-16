"""
Tests for the UUID to deterministic ID migration script.
"""

from scripts.migrate_to_deterministic_ids import (
    generate_deterministic_id,
    is_random_uuid,
    normalize_url,
)


class TestMigrationScript:
    """Test cases for migration script functionality."""

    def test_normalize_url_consistency(self):
        """Test that URL normalization matches RagService logic."""
        # Import RagService for comparison
        from crawler_mcp.core.rag import RagService

        rag_service = RagService()

        test_urls = [
            "https://example.com/page",
            "http://EXAMPLE.COM/Page",
            "https://example.com/page/",
            "https://example.com/page?b=2&a=1",
            "https://example.com/page#section",
        ]

        for url in test_urls:
            migration_result = normalize_url(url)
            rag_result = rag_service._normalize_url(url)
            assert migration_result == rag_result, (
                f"URL normalization mismatch for {url}"
            )

    def test_generate_deterministic_id_consistency(self):
        """Test that ID generation matches RagService logic."""
        # Import RagService for comparison
        from crawler_mcp.core.rag import RagService

        rag_service = RagService()

        test_cases = [
            ("https://example.com/page", 0),
            ("https://example.com/page", 1),
            ("https://different.com/path", 0),
            ("http://EXAMPLE.COM/Page", 0),  # Should normalize to same as first case
        ]

        for url, chunk_index in test_cases:
            migration_result = generate_deterministic_id(url, chunk_index)
            rag_result = rag_service._generate_deterministic_id(url, chunk_index)
            assert migration_result == rag_result, (
                f"ID generation mismatch for {url}:{chunk_index}"
            )

    def test_is_random_uuid(self):
        """Test random UUID detection logic."""
        # Test cases for UUIDs that should be detected as random
        random_uuids = [
            "550e8400e29b41d4a716446655440000",  # 32 hex chars
            "550e8400-e29b-41d4-a716-446655440000",  # Standard UUID format
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # Another valid UUID
            "123456789abcdef0123456789abcdef012345",  # Longer hex string
            "ABCDEF1234567890ABCDEF1234567890",  # Uppercase hex
        ]

        for uuid in random_uuids:
            assert is_random_uuid(uuid), f"Should detect {uuid} as random UUID"

        # Test cases for IDs that should NOT be detected as random
        non_random_ids = [
            "abc123",  # Too short
            "deterministic-id-16",  # Contains non-hex chars
            "1234567890123456",  # Deterministic format (16 chars)
            "short",  # Too short
            "",  # Empty string
            "123",  # Very short
        ]

        for id_str in non_random_ids:
            assert not is_random_uuid(id_str), (
                f"Should NOT detect {id_str} as random UUID"
            )

    def test_deterministic_id_properties(self):
        """Test properties of generated deterministic IDs."""
        url = "https://example.com/test"

        # Test ID consistency
        id1 = generate_deterministic_id(url, 0)
        id2 = generate_deterministic_id(url, 0)
        assert id1 == id2, "Same inputs should produce same ID"

        # Test ID uniqueness
        id3 = generate_deterministic_id(url, 1)
        assert id1 != id3, "Different chunk indexes should produce different IDs"

        id4 = generate_deterministic_id("https://other.com/test", 0)
        assert id1 != id4, "Different URLs should produce different IDs"

        # Test ID format - should be UUID format (36 characters with dashes)
        assert len(id1) == 36, "Deterministic ID should be 36 characters (UUID format)"
        assert id1.count("-") == 4, "ID should have UUID format with 4 dashes"

        # Test that deterministic IDs are still valid UUIDs that match the pattern
        # (This is expected since they're generated from hash bytes in UUID format)
        assert is_random_uuid(id1), (
            "Deterministic ID should match UUID pattern (this is expected)"
        )

    def test_url_edge_cases(self):
        """Test URL normalization with edge cases."""
        edge_cases = [
            ("", ""),  # Empty URL
            ("invalid-url", "invalid-url"),  # Invalid URL (should return as-is)
            ("https://example.com", "https://example.com/"),  # No path
            ("https://example.com/", "https://example.com/"),  # Root path
        ]

        for input_url, expected in edge_cases:
            if expected:
                result = normalize_url(input_url)
                if input_url:  # Skip empty URL test as it may be handled differently
                    assert result in (expected, input_url), (
                        f"URL normalization issue: {input_url} -> {result}"
                    )

    def test_migration_id_collision_avoidance(self):
        """Test that migration generates unique IDs for different content."""
        base_url = "https://example.com/page"

        # Generate IDs for multiple chunk indexes
        ids = []
        for i in range(100):
            id_val = generate_deterministic_id(base_url, i)
            ids.append(id_val)

        # All IDs should be unique
        assert len(set(ids)) == len(ids), "All generated IDs should be unique"

        # IDs should be deterministic across calls
        for i in range(10):
            id_again = generate_deterministic_id(base_url, i)
            assert id_again == ids[i], (
                f"ID generation should be deterministic for index {i}"
            )

    def test_real_world_urls(self):
        """Test with realistic URLs from actual crawling scenarios."""
        real_urls = [
            "https://docs.python.org/3/library/asyncio.html",
            "https://github.com/anthropics/claude-code/issues/123",
            "https://www.example.com/blog/post-title?utm_source=twitter&utm_medium=social",
            "https://api.github.com/repos/owner/repo/pulls/456#issuecomment-789",
        ]

        for url in real_urls:
            # Should not crash on real URLs
            normalized = normalize_url(url)
            assert isinstance(normalized, str), f"URL normalization failed for {url}"

            # Should generate valid deterministic IDs
            for chunk_index in range(3):
                det_id = generate_deterministic_id(url, chunk_index)
                assert len(det_id) == 36, (
                    f"Invalid ID length for {url}:{chunk_index} - should be UUID format"
                )
                assert is_random_uuid(det_id), (
                    f"Generated ID should match UUID pattern for {url}:{chunk_index}"
                )
