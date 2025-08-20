"""
Test configuration management with Pydantic settings.
"""

import os
import tempfile
from pathlib import Path

import pytest

from crawler_mcp.config import CrawlerrSettings, get_settings, settings


class TestCrawlerrSettings:
    """Test the main configuration class."""

    @pytest.mark.unit
    def test_environment_loading(self):
        """Test that configuration correctly loads from environment (.env file)."""
        config = CrawlerrSettings()

        # Test that values are loaded from .env file (not hardcoded defaults)
        assert config.server_host == "0.0.0.0"  # From .env SERVER_HOST
        assert config.server_port == 8010  # From .env SERVER_PORT
        assert config.debug is True  # From .env DEBUG=true
        assert config.production is False  # From .env PRODUCTION=false

        # Logging configuration from .env
        assert config.log_level == "INFO"  # From .env LOG_LEVEL
        assert config.log_format == "console"  # From .env LOG_FORMAT
        assert config.log_to_file is True  # From .env LOG_TO_FILE=true

        # Service endpoints from .env
        assert config.qdrant_url == "http://localhost:6333"  # From .env QDRANT_URL
        assert config.qdrant_collection == "crawlerr_documents"  # From .env
        assert config.qdrant_vector_size == 1024  # From .env
        assert config.qdrant_distance == "cosine"  # From .env

        # TEI configuration from .env
        assert config.tei_url == "http://localhost:8080"  # From .env TEI_URL
        assert config.tei_model == "Qwen/Qwen3-Embedding-0.6B"  # From .env TEI_MODEL
        assert config.tei_batch_size == 64  # From .env TEI_BATCH_SIZE

        # Crawling settings from .env
        assert config.chunk_size == 1024  # From .env CHUNK_SIZE
        assert config.chunk_overlap == 200  # From .env CHUNK_OVERLAP
        assert (
            config.reranker_model == "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
        )  # From .env
        assert (
            config.deduplication_strategy == "content_hash"
        )  # From .env DEDUPLICATION_STRATEGY

    @pytest.mark.unit
    def test_field_validation(self):
        """Test that field validation works (environment takes precedence)."""
        # Test that current .env values are valid and loaded correctly
        config = CrawlerrSettings()

        # These should be the actual values from .env (environment takes precedence)
        assert config.server_port == 8010  # From SERVER_PORT=8010 in .env
        assert config.crawl_max_pages == 1000  # From CRAWL_MAX_PAGES=1000 in .env
        assert config.chunk_size == 1024  # From CHUNK_SIZE=1024 in .env
        assert config.chunk_overlap == 200  # From CHUNK_OVERLAP=200 in .env

    @pytest.mark.unit
    def test_chunk_validation(self):
        """Test chunk size validation."""
        # Valid chunking configuration
        config = CrawlerrSettings(
            chunk_size=1000, chunk_overlap=200, embedding_max_length=2000
        )
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

        # Invalid: overlap >= chunk_size
        with pytest.raises(
            ValueError, match="CHUNK_OVERLAP must be less than CHUNK_SIZE"
        ):
            CrawlerrSettings(
                chunk_size=1000,
                chunk_overlap=1000,  # Equal to chunk_size
            )

        # Invalid: chunk_size > embedding_max_length
        with pytest.raises(
            ValueError, match="CHUNK_SIZE must be <= EMBEDDING_MAX_LENGTH"
        ):
            CrawlerrSettings(
                chunk_size=2000,
                embedding_max_length=1000,  # Less than chunk_size
            )

    @pytest.mark.unit
    def test_reranker_model_validation(self):
        """Test reranker model validation."""
        # Valid reranker model override
        config = CrawlerrSettings(reranker_model="test/model")
        assert config.reranker_model == "test/model"

        # Invalid empty reranker model
        with pytest.raises(
            ValueError, match="RERANKER_MODEL must be a non-empty string"
        ):
            CrawlerrSettings(reranker_model="")

        # Invalid whitespace-only reranker model
        with pytest.raises(
            ValueError, match="RERANKER_MODEL must be a non-empty string"
        ):
            CrawlerrSettings(reranker_model="   ")

    @pytest.mark.unit
    def test_deduplication_validation(self):
        """Test deduplication configuration validation."""
        # Test that we can override the .env default with valid values
        for strategy in ["content_hash", "timestamp", "none"]:
            config = CrawlerrSettings(deduplication_strategy=strategy)
            assert config.deduplication_strategy == strategy

        # Invalid deduplication strategy
        with pytest.raises(ValueError):
            CrawlerrSettings(deduplication_strategy="invalid_strategy")

    @pytest.mark.unit
    def test_cors_origins_list_property(self):
        """Test cors_origins_list property."""
        # Test default from .env (should be "*")
        config = CrawlerrSettings()
        assert config.cors_origins_list == ["*"]

        # Test override with multiple origins
        config = CrawlerrSettings(
            cors_origins="http://localhost:3000,https://example.com"
        )
        assert len(config.cors_origins_list) == 2
        assert "http://localhost:3000" in config.cors_origins_list
        assert "https://example.com" in config.cors_origins_list

        # Test parsing with spaces
        config = CrawlerrSettings(
            cors_origins=" http://localhost:3000 , https://example.com "
        )
        assert len(config.cors_origins_list) == 2

    @pytest.mark.unit
    def test_log_file_directory_creation(self):
        """Test that log file directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "test.log"

            config = CrawlerrSettings(log_file=str(log_path), log_to_file=True)

            # Directory should be created
            assert log_path.parent.exists()
            assert config.log_file == str(log_path)

    @pytest.mark.unit
    def test_pid_file_directory_creation(self):
        """Test that PID file directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "run" / "crawler.pid"

            config = CrawlerrSettings(pid_file=str(pid_path))

            # Directory should be created
            assert pid_path.parent.exists()
            assert config.pid_file == str(pid_path)

    @pytest.mark.unit
    def test_crawling_configuration(self):
        """Test crawling-specific configuration."""
        config = CrawlerrSettings(
            crawl_headless=False,
            crawl_browser="firefox",
            crawl_max_pages=2000,
            crawl_max_depth=5,
            max_concurrent_crawls=50,
            crawler_delay=1.0,
            crawler_timeout=60.0,
        )

        assert config.crawl_headless is False
        assert config.crawl_browser == "firefox"
        assert config.crawl_max_pages == 2000
        assert config.crawl_max_depth == 5
        assert config.max_concurrent_crawls == 50
        assert config.crawler_delay == 1.0
        assert config.crawler_timeout == 60.0

    @pytest.mark.unit
    def test_performance_configuration(self):
        """Test performance-related configuration."""
        config = CrawlerrSettings(
            crawl_block_images=True,
            crawl_block_media=True,
            crawl_block_stylesheets=True,
            crawl_block_fonts=True,
            crawl_enable_streaming=False,
            crawl_enable_caching=False,
        )

        assert config.crawl_block_images is True
        assert config.crawl_block_media is True
        assert config.crawl_block_stylesheets is True
        assert config.crawl_block_fonts is True
        assert config.crawl_enable_streaming is False
        assert config.crawl_enable_caching is False

    @pytest.mark.unit
    def test_gpu_configuration(self):
        """Test GPU-related configuration."""
        config = CrawlerrSettings(gpu_acceleration=True, crawl_gpu_enabled=True)

        assert config.gpu_acceleration is True
        assert config.crawl_gpu_enabled is True

    @pytest.mark.unit
    def test_advanced_crawl_features(self):
        """Test Crawl4AI advanced features configuration."""
        config = CrawlerrSettings(
            crawl_adaptive_mode=False,
            crawl_confidence_threshold=0.9,
            crawl_top_k_links=10,
            crawl_url_seeding=False,
            crawl_score_threshold=0.5,
            crawl_virtual_scroll=False,
            crawl_scroll_count=30,
        )

        assert config.crawl_adaptive_mode is False
        assert config.crawl_confidence_threshold == 0.9
        assert config.crawl_top_k_links == 10
        assert config.crawl_url_seeding is False
        assert config.crawl_score_threshold == 0.5
        assert config.crawl_virtual_scroll is False
        assert config.crawl_scroll_count == 30


class TestSettingsSingleton:
    """Test the settings singleton functionality."""

    @pytest.mark.unit
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    @pytest.mark.unit
    def test_settings_module_variable(self):
        """Test that the module-level settings variable works."""
        assert settings is not None
        assert isinstance(settings, CrawlerrSettings)

        # Should be the same as get_settings()
        assert settings is get_settings()

    @pytest.mark.unit
    def test_settings_environment_loading(self):
        """Test that settings can load from environment variables."""
        # Test with a temporary environment variable
        test_port = 9999
        original_port = os.environ.get("SERVER_PORT")

        try:
            os.environ["SERVER_PORT"] = str(test_port)

            # Create new settings instance (not singleton)
            config = CrawlerrSettings()
            assert config.server_port == test_port

        finally:
            # Clean up
            if original_port is not None:
                os.environ["SERVER_PORT"] = original_port
            else:
                os.environ.pop("SERVER_PORT", None)

    @pytest.mark.unit
    def test_model_dump_serialization(self):
        """Test that settings can be serialized."""
        config = CrawlerrSettings()

        # Should be able to dump to dict
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "server_host" in data
        assert "server_port" in data

        # Should preserve values
        assert data["server_host"] == config.server_host
        assert data["server_port"] == config.server_port

    @pytest.mark.unit
    def test_field_constraints(self):
        """Test field constraints work correctly."""
        # Test max_pages constraints
        config = CrawlerrSettings(crawl_max_pages=1)  # Minimum value
        assert config.crawl_max_pages == 1

        config = CrawlerrSettings(crawl_max_pages=1000)  # Maximum value
        assert config.crawl_max_pages == 1000

        # Invalid: below minimum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            CrawlerrSettings(crawl_max_pages=1001)

    @pytest.mark.unit
    def test_boolean_defaults(self):
        """Test boolean field defaults."""
        config = CrawlerrSettings()

        # Should have correct boolean defaults
        assert config.debug is False
        assert config.production is False
        assert config.crawl_headless is True
        assert config.remove_overlay_elements is True
        assert config.extract_media is False
        assert config.deduplication_enabled is True
        assert config.cors_credentials is True
