"""
Configuration management for Crawlerr using Pydantic Settings.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class CrawlerrSettings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    """
    
    # Server Configuration
    server_host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    production: bool = Field(default=False, env="PRODUCTION")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="console", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_to_file: bool = Field(default=False, env="LOG_TO_FILE")
    
    # Qdrant Vector Database
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="crawlerr_documents", env="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=1024, env="QDRANT_VECTOR_SIZE")
    qdrant_distance: str = Field(default="cosine", env="QDRANT_DISTANCE")
    qdrant_timeout: float = Field(default=10.0, env="QDRANT_TIMEOUT")
    qdrant_retry_count: int = Field(default=3, env="QDRANT_RETRY_COUNT")
    
    # HF Text Embeddings Inference (TEI)
    tei_url: str = Field(default="http://localhost:8080", env="TEI_URL")
    tei_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", env="TEI_MODEL")
    tei_max_concurrent_requests: int = Field(default=128, env="TEI_MAX_CONCURRENT_REQUESTS")
    tei_max_batch_tokens: int = Field(default=32768, env="TEI_MAX_BATCH_TOKENS")
    tei_batch_size: int = Field(default=64, env="TEI_BATCH_SIZE")
    tei_timeout: float = Field(default=30.0, env="TEI_TIMEOUT")
    
    # Embedding Configuration
    embedding_max_length: int = Field(default=32000, env="EMBEDDING_MAX_LENGTH")
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")
    embedding_normalize: bool = Field(default=True, env="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(default=2, env="EMBEDDING_MAX_RETRIES")
    
    # Reranker Configuration
    reranker_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B", env="RERANKER_MODEL")
    reranker_enabled: bool = Field(default=True, env="RERANKER_ENABLED")
    reranker_top_k: int = Field(default=10, env="RERANKER_TOP_K")
    reranker_max_length: int = Field(default=512, env="RERANKER_MAX_LENGTH")
    reranker_fallback_to_custom: bool = Field(default=True, env="RERANKER_FALLBACK_TO_CUSTOM")
    
    # Crawling Configuration
    crawl_headless: bool = Field(default=True, env="CRAWL_HEADLESS")
    crawl_browser: str = Field(default="chromium", env="CRAWL_BROWSER")
    crawl_max_pages: int = Field(default=100, env="CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=3, env="CRAWL_MAX_DEPTH")
    max_concurrent_crawls: int = Field(default=25, env="MAX_CONCURRENT_CRAWLS")
    crawler_delay: float = Field(default=0.1, env="CRAWLER_DELAY")
    crawler_timeout: float = Field(default=30.0, env="CRAWLER_TIMEOUT")
    crawl_min_words: int = Field(default=100, env="CRAWL_MIN_WORDS")
    crawl_remove_overlays: bool = Field(default=True, env="CRAWL_REMOVE_OVERLAYS")
    crawl_extract_media: bool = Field(default=False, env="CRAWL_EXTRACT_MEDIA")
    crawl_knowledge_graph: bool = Field(default=False, env="CRAWL_KNOWLEDGE_GRAPH")
    crawl_user_agent: str = Field(
        default="Crawlerr/0.1.0 (+https://github.com/user/crawlerr)",
        env="CRAWL_USER_AGENT"
    )
    
    # Crawl4AI 0.7.0 Advanced Features
    crawl_adaptive_mode: bool = Field(default=True, env="CRAWL_ADAPTIVE_MODE")
    crawl_confidence_threshold: float = Field(default=0.8, env="CRAWL_CONFIDENCE_THRESHOLD")
    crawl_top_k_links: int = Field(default=5, env="CRAWL_TOP_K_LINKS")
    crawl_url_seeding: bool = Field(default=True, env="CRAWL_URL_SEEDING")
    crawl_score_threshold: float = Field(default=0.4, env="CRAWL_SCORE_THRESHOLD")
    crawl_virtual_scroll: bool = Field(default=True, env="CRAWL_VIRTUAL_SCROLL")
    crawl_scroll_count: int = Field(default=20, env="CRAWL_SCROLL_COUNT")
    crawl_memory_threshold: float = Field(default=90.0, env="CRAWL_MEMORY_THRESHOLD")
    
    # CORS & Security  
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")
    
    @property 
    def cors_origins_list(self) -> List[str]:
        """Convert cors_origins string to list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @validator("log_file", pre=True)
    def create_log_directory(cls, v):
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = CrawlerrSettings()