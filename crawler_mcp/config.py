"""
Configuration management for Crawlerr using Pydantic Settings.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants for error messages
RERANKER_MODEL_ERROR = "RERANKER_MODEL must be a non-empty string"


class CrawlerrSettings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    """

    # Server Configuration
    server_host: str = Field(default="127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    production: bool = Field(default=False, alias="PRODUCTION")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")
    log_file: str | None = Field(default=None, alias="LOG_FILE")
    log_to_file: bool = Field(default=False, alias="LOG_TO_FILE")
    pid_file: str = Field(default="logs/crawlerr.pid", alias="PID_FILE")

    # Qdrant Vector Database
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="crawlerr_documents", alias="QDRANT_COLLECTION"
    )
    qdrant_vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    qdrant_distance: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine", alias="QDRANT_DISTANCE"
    )
    qdrant_timeout: float = Field(default=10.0, alias="QDRANT_TIMEOUT")
    qdrant_retry_count: int = Field(default=3, alias="QDRANT_RETRY_COUNT")
    qdrant_connection_pool_size: int = Field(
        default=16, alias="QDRANT_CONNECTION_POOL_SIZE", ge=1, le=32
    )
    # Unified batch configuration for optimal performance
    default_batch_size: int = Field(
        default=256,
        alias="DEFAULT_BATCH_SIZE",
        ge=64,
        le=512,
        description="Default batch size for all operations",
    )
    qdrant_batch_size: int = Field(
        default=256, alias="QDRANT_BATCH_SIZE", ge=64, le=512
    )
    qdrant_prefetch_size: int = Field(
        default=1024, alias="QDRANT_PREFETCH_SIZE", ge=256, le=2048
    )
    qdrant_search_exact: bool = Field(default=False, alias="QDRANT_SEARCH_EXACT")

    # Vector Service Configuration - using modular implementation

    # HF Text Embeddings Inference (TEI)
    tei_url: str = Field(default="http://localhost:8080", alias="TEI_URL")
    tei_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="TEI_MODEL")
    tei_max_concurrent_requests: int = Field(
        default=128, alias="TEI_MAX_CONCURRENT_REQUESTS"
    )
    tei_max_batch_tokens: int = Field(default=32768, alias="TEI_MAX_BATCH_TOKENS")
    tei_batch_size: int = Field(
        default=256, alias="TEI_BATCH_SIZE"
    )  # Increased for better throughput
    tei_timeout: float = Field(default=30.0, alias="TEI_TIMEOUT")

    # Embedding Configuration
    embedding_max_length: int = Field(default=32000, alias="EMBEDDING_MAX_LENGTH")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(default=3, alias="EMBEDDING_MAX_RETRIES")

    # Retry configuration with exponential backoff
    retry_initial_delay: float = Field(
        default=1.0,
        alias="RETRY_INITIAL_DELAY",
        ge=0.1,
        le=10.0,
        description="Initial delay in seconds for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=60.0,
        alias="RETRY_MAX_DELAY",
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds for exponential backoff",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        alias="RETRY_EXPONENTIAL_BASE",
        ge=1.1,
        le=5.0,
        description="Base for exponential backoff calculation",
    )
    embedding_workers: int = Field(default=4, alias="EMBEDDING_WORKERS", ge=1, le=16)

    # Chunking Configuration
    chunk_size: int = Field(default=1024, alias="CHUNK_SIZE", gt=0, le=32768)
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP", ge=0)
    word_to_token_ratio: float = Field(default=1.4, alias="WORD_TO_TOKEN_RATIO", gt=0)

    # Reranker Configuration
    reranker_model: str = Field(
        default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_top_k: int = Field(default=10, alias="RERANKER_TOP_K", gt=0, le=100)
    reranker_max_length: int = Field(
        default=512, alias="RERANKER_MAX_LENGTH", gt=0, le=4096
    )
    reranker_fallback_to_custom: bool = Field(
        default=True, alias="RERANKER_FALLBACK_TO_CUSTOM"
    )

    @field_validator("reranker_model", mode="before")
    @classmethod
    def _validate_reranker_model(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError(RERANKER_MODEL_ERROR)
        return v

    @field_validator("embedding_workers")
    @classmethod
    def validate_embedding_workers(cls, v: int) -> int:
        if not 1 <= v <= 16:
            raise ValueError("embedding_workers must be between 1 and 16")
        return v

    @field_validator("browser_pool_size")
    @classmethod
    def validate_browser_pool_size(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("browser_pool_size must be between 1 and 20")
        return v

    @field_validator("browser_type")
    @classmethod
    def validate_browser_type(cls, v: str) -> str:
        allowed_types = {"chromium", "firefox", "webkit"}
        if v.lower() not in allowed_types:
            raise ValueError(f"browser_type must be one of {allowed_types}")
        return v.lower()

    @field_validator("browser_extra_args")
    @classmethod
    def validate_browser_extra_args(cls, v: list[str]) -> list[str]:
        # Whitelist of safe flags that won't break headless mode
        safe_flags = {
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features",
            "--process-per-site",
            "--aggressive-cache-discard",
            "--memory-pressure-off",
            "--no-zygote",
            "--max_old_space_size",
            "--js-flags",
            "--renderer-process-limit",
            "--max-renderer-processes",
            "--enable-accelerated-2d-canvas",
            "--enable-gpu-compositing",
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--enable-oop-rasterization",
        }

        for arg in v:
            if not arg.startswith("--"):
                raise ValueError(f"Browser arg must start with '--': {arg}")
            flag_name = arg.split("=")[0]
            if not any(flag_name.startswith(safe) for safe in safe_flags):
                logger = logging.getLogger(__name__)
                logger.warning(f"Potentially unsafe browser flag: {arg}")
        return v

    @field_validator("browser_hardware_profile")
    @classmethod
    def validate_browser_hardware_profile(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed_profiles = {"rtx4070_i7", "high_memory", "basic"}
        if v not in allowed_profiles:
            raise ValueError(
                f"browser_hardware_profile must be one of {allowed_profiles} or None"
            )
        return v

    # GPU flag validation removed - no longer using custom Chrome flags

    # Crawling Configuration
    crawl_headless: bool = Field(default=True, alias="CRAWL_HEADLESS")
    crawl_browser: str = Field(default="chromium", alias="CRAWL_BROWSER")
    crawl_max_pages: int = Field(default=1000, alias="CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=3, alias="CRAWL_MAX_DEPTH")
    max_concurrent_crawls: int = Field(default=25, alias="MAX_CONCURRENT_CRAWLS")
    crawler_delay: float = Field(default=0.1, alias="CRAWLER_DELAY")
    crawler_timeout: float = Field(default=30.0, alias="CRAWLER_TIMEOUT")
    crawl_min_words: int = Field(default=50, alias="CRAWL_MIN_WORDS")
    crawl_remove_overlays: bool = Field(default=True, alias="CRAWL_REMOVE_OVERLAYS")
    crawl_extract_media: bool = Field(default=False, alias="CRAWL_EXTRACT_MEDIA")
    crawl_knowledge_graph: bool = Field(default=False, alias="CRAWL_KNOWLEDGE_GRAPH")
    crawl_user_agent: str = Field(
        default="Crawlerr/0.1.0 (+https://github.com/user/crawlerr)",
        alias="CRAWL_USER_AGENT",
    )

    # Resource blocking (performance)
    crawl_block_images: bool = Field(default=False, alias="CRAWL_BLOCK_IMAGES")
    crawl_block_media: bool = Field(default=False, alias="CRAWL_BLOCK_MEDIA")
    crawl_block_stylesheets: bool = Field(
        default=False, alias="CRAWL_BLOCK_STYLESHEETS"
    )
    crawl_block_fonts: bool = Field(default=False, alias="CRAWL_BLOCK_FONTS")

    # Crawl4AI 0.7.0 Advanced Features
    crawl_adaptive_mode: bool = Field(default=True, alias="CRAWL_ADAPTIVE_MODE")
    crawl_confidence_threshold: float = Field(
        default=0.8, alias="CRAWL_CONFIDENCE_THRESHOLD"
    )
    crawl_top_k_links: int = Field(default=5, alias="CRAWL_TOP_K_LINKS")
    crawl_url_seeding: bool = Field(default=True, alias="CRAWL_URL_SEEDING")
    crawl_score_threshold: float = Field(default=0.4, alias="CRAWL_SCORE_THRESHOLD")
    crawl_virtual_scroll: bool = Field(default=True, alias="CRAWL_VIRTUAL_SCROLL")
    crawl_scroll_count: int = Field(default=20, alias="CRAWL_SCROLL_COUNT")
    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
    crawl_virtual_scroll_batch_size: int = Field(
        default=10, alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE"
    )
    crawl_memory_threshold: float = Field(default=70.0, alias="CRAWL_MEMORY_THRESHOLD")

    # Performance Optimization Configuration
    crawl_enable_streaming: bool = Field(default=True, alias="CRAWL_ENABLE_STREAMING")
    crawl_enable_caching: bool = Field(default=True, alias="CRAWL_ENABLE_CACHING")

    # Browser Configuration
    browser_headless: bool = Field(default=True, alias="BROWSER_HEADLESS")
    browser_type: str = Field(default="chromium", alias="BROWSER_TYPE")
    browser_verbose: bool = Field(default=False, alias="BROWSER_VERBOSE")
    browser_extra_args: list[str] = Field(
        default_factory=list, alias="BROWSER_EXTRA_ARGS"
    )
    browser_hardware_profile: str | None = Field(
        default=None, alias="BROWSER_HARDWARE_PROFILE"
    )

    # High-Performance Configuration (i7-13700k + RTX 4070)
    browser_pool_size: int = Field(default=8, alias="BROWSER_POOL_SIZE", ge=1, le=16)
    file_processing_threads: int = Field(
        default=16, alias="FILE_PROCESSING_THREADS", ge=1, le=24
    )
    crawl_concurrency: int = Field(default=12, alias="CRAWL_CONCURRENCY", ge=1, le=50)
    content_cache_size_gb: int = Field(
        default=8, alias="CONTENT_CACHE_SIZE_GB", ge=1, le=16
    )
    gpu_memory_fraction: float = Field(
        default=0.95, alias="GPU_MEMORY_FRACTION", ge=0.1, le=1.0
    )

    # RTX 4070 GPU Acceleration Configuration
    gpu_acceleration: bool = Field(
        default=True,
        alias="GPU_ACCELERATION",
        description="Enable GPU acceleration features",
    )
    crawl_gpu_enabled: bool = Field(
        default=True,
        alias="CRAWL_GPU_ENABLED",
        description="Enable GPU acceleration for web crawling",
    )
    # Chrome flags removed - Crawl4AI light_mode handles optimization
    # GPU management removed - let Crawl4AI handle GPU acceleration

    # Alternative crawling approach settings
    use_arun_many_for_sitemaps: bool = Field(
        default=False,
        alias="USE_ARUN_MANY_FOR_SITEMAPS",
        description="Use arun_many() with sitemap URLs instead of BFSDeepCrawlStrategy",
    )
    max_concurrent_sessions: int = Field(
        default=20,
        alias="CRAWL_MAX_CONCURRENT_SESSIONS",
        ge=1,
        le=50,
        description="Maximum concurrent sessions for arun_many() approach",
    )

    # Crawl4AI Performance Optimizations
    crawl_text_mode: bool = Field(
        default=False,
        alias="CRAWL_TEXT_MODE",
        description="Enable text-only mode for 3-4x faster crawling (disables images)",
    )
    crawl_light_mode: bool = Field(
        default=True,
        alias="CRAWL_LIGHT_MODE",
        description="Enable light mode to optimize browser performance",
    )
    crawl_enable_gpu: bool = Field(
        default=False,
        alias="CRAWL_ENABLE_GPU",
        description="Enable GPU acceleration for browsers (requires GPU support)",
    )
    use_lxml_strategy: bool = Field(
        default=True,
        alias="USE_LXML_STRATEGY",
        description="Use LXMLWebScrapingStrategy for 20x faster parsing",
    )

    # URL Pattern Exclusions - Comprehensive filtering to avoid non-HTML content
    crawl_exclude_url_patterns: list[str] = Field(
        default=[
            # Authentication/Admin pages
            "*login.php*",
            "*admin.php*",
            "*login*",
            "*admin*",
            # Archives & Downloads
            "*.zip",
            "*.tar",
            "*.gz",
            "*.rar",
            "*.7z",
            # Documents
            "*.pdf",
            "*.doc",
            "*.docx",
            "*.xls",
            "*.xlsx",
            "*.ppt",
            "*.pptx",
            # Images (comprehensive)
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.svg",
            "*.webp",
            "*.ico",
            "*.bmp",
            "*.tiff",
            # Media files
            "*.mp3",
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.wmv",
            "*.flv",
            "*.wav",
            "*.m4v",
            # Executables & Installers
            "*.exe",
            "*.msi",
            "*.app",
            "*.deb",
            "*.rpm",
            "*.dmg",
            # Web fonts
            "*.woff",
            "*.woff2",
            "*.ttf",
            "*.eot",
            "*.otf",
            # Data/Config files
            "*.json",
            "*.xml",
            "*.csv",
            "*.sql",
            "*.yaml",
            "*.yml",
            # Stylesheets and scripts (usually not content)
            "*.css",
            "*.js",
            "*.min.js",
            "*.min.css",
            # Version control
            "*/.git/*",
            "*/.svn/*",
            "*/.hg/*",
            # Common non-content patterns
            "*/api/*",
            "*/static/*",
            "*/assets/*",
            "*/_next/static/*",  # Next.js static assets
            # External domains that should be avoided
            "*github.com/*",  # GitHub profiles/repos
            "*twitter.com/*",  # Social media
            "*x.com/*",
            "*linkedin.com/*",
            "*facebook.com/*",
            "*youtube.com/*",  # Video platforms
            "*vimeo.com/*",
        ],
        alias="CRAWL_EXCLUDE_URL_PATTERNS",
    )

    # Deduplication Configuration
    deduplication_enabled: bool = Field(
        default=True,
        alias="DEDUPLICATION_ENABLED",
        description="Enable content-based deduplication for crawled pages",
    )
    deduplication_strategy: str = Field(
        default="content_hash",
        alias="DEDUPLICATION_STRATEGY",
        description="Deduplication strategy: 'content_hash', 'timestamp', or 'none'",
    )
    delete_orphaned_chunks: bool = Field(
        default=True,
        alias="DELETE_ORPHANED_CHUNKS",
        description="Delete chunks that no longer exist in re-crawled content",
    )
    preserve_unchanged_metadata: bool = Field(
        default=True,
        alias="PRESERVE_UNCHANGED_METADATA",
        description="Preserve metadata for unchanged chunks during re-crawls",
    )

    # Directory Crawling Configuration
    directory_excluded_extensions: list[str] = Field(
        default=[
            # Binary executables and libraries
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".obj",
            ".o",
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".tiff",
            ".webp",
            ".svg",
            # Audio/Video
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".wav",
            ".mkv",
            ".webm",
            # Archives
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".rar",
            ".xz",
            # Documents (can be made configurable for future PDF/Office extraction)
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            # Other binary formats
            ".iso",
            ".dmg",
            ".pkg",
            ".deb",
            ".rpm",
        ],
        alias="DIRECTORY_EXCLUDED_EXTENSIONS",
        description="File extensions to exclude when crawling directories",
    )
    directory_max_file_size_mb: int = Field(
        default=10,
        alias="DIRECTORY_MAX_FILE_SIZE_MB",
        ge=1,
        le=100,
        description="Maximum file size in MB to process when crawling directories",
    )

    # CORS & Security
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, alias="CORS_CREDENTIALS")
    max_request_size: int = Field(default=10485760, alias="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")

    @property
    def cors_origins_list(self) -> list[str]:
        """Convert cors_origins string to list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @field_validator("log_file", mode="before")
    @classmethod
    def create_log_directory(cls, v: str | None) -> str | None:
        if v:
            log_path = Path(str(v)).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("pid_file", mode="before")
    @classmethod
    def create_pid_directory(cls, v: str) -> str:
        pid_path = Path(str(v)).expanduser()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _validate_chunking(self) -> "CrawlerrSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.chunk_size > self.embedding_max_length:
            raise ValueError("CHUNK_SIZE must be <= EMBEDDING_MAX_LENGTH")
        return self

    @model_validator(mode="after")
    def _validate_deduplication(self) -> "CrawlerrSettings":
        """Validate deduplication configuration."""
        valid_strategies = {"content_hash", "timestamp", "none"}
        if self.deduplication_strategy not in valid_strategies:
            raise ValueError(
                f"DEDUPLICATION_STRATEGY must be one of: {', '.join(valid_strategies)}"
            )

        # Warn if deduplication is disabled but orphan deletion is enabled
        if not self.deduplication_enabled and self.delete_orphaned_chunks:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "DELETE_ORPHANED_CHUNKS is enabled but DEDUPLICATION_ENABLED is False. "
                "Orphan detection requires deduplication to be enabled."
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore deprecated settings for backward compatibility
    )


# Lazy settings accessor (avoids import-time side effects)
_settings: CrawlerrSettings | None = None


def get_settings() -> CrawlerrSettings:
    global _settings
    if _settings is None:
        _settings = CrawlerrSettings()
    return _settings


# For backward compatibility
settings = get_settings()
