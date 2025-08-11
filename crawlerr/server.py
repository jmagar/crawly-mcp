"""
Crawlerr FastMCP Server - RAG-enabled web crawling with Crawl4AI and Qdrant.

This server provides comprehensive web crawling capabilities with automatic RAG indexing
using Crawl4AI 0.7.0, Qdrant vector database, and HF Text Embeddings Inference.
"""
import warnings

# Suppress specific deprecation warnings from external dependencies FIRST
warnings.filterwarnings("ignore", message=".*@validator.*is deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Support for class-based.*config.*is deprecated.*", category=DeprecationWarning)

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Configure logging before importing other modules
try:
    from .config import settings
except ImportError:
    # Handle case when run as standalone script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from crawlerr.config import settings

# Set up colorized logging configuration
def setup_logging() -> None:
    """Configure rich colorized logging for the application."""
    # Install rich tracebacks for better error display
    install(show_locals=settings.debug)
    
    # Create rich console
    console = Console(force_terminal=True, width=120)
    
    # Configure rich handler with colors
    rich_handler = RichHandler(
        console=console,
        show_path=settings.debug,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )
    
    # Create file handler if specified
    if settings.log_to_file and settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler without colors
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("chromium").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Import FastMCP and other dependencies
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

# Import middleware
try:
    from .middleware import ErrorHandlingMiddleware, LoggingMiddleware, ProgressMiddleware
    # Import tools
    from .tools.crawling_tools import register_crawling_tools
    from .tools.rag_tools import register_rag_tools
    # Import services for health checks
    from .services import EmbeddingService, VectorService, RagService
except ImportError:
    # Handle case when run as standalone script
    from crawlerr.middleware import ErrorHandlingMiddleware, LoggingMiddleware, ProgressMiddleware
    # Import tools
    from crawlerr.tools.crawling_tools import register_crawling_tools
    from crawlerr.tools.rag_tools import register_rag_tools
    # Import services for health checks
    from crawlerr.services import EmbeddingService, VectorService, RagService


# Create FastMCP instance
mcp = FastMCP("crawlerr-server")

# Middleware will be handled by the HTTP transport automatically
# For now, we'll focus on the core functionality

# Register all tools
register_crawling_tools(mcp)
register_rag_tools(mcp)

logger.info("Registered all FastMCP tools")


@mcp.tool
async def health_check(ctx: Context) -> dict[str, Any]:
    """
    Perform a comprehensive health check of all services.
    
    Returns:
        Dictionary with health status of all components
    """
    await ctx.info("Performing health check of all services")
    
    try:
        health_results = {
            "server": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-01T00:00:00Z"
            },
            "services": {},
            "configuration": {
                "embedding_model": settings.tei_model,
                "vector_database": settings.qdrant_url,
                "embedding_service": settings.tei_url,
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "chunk_size": 1000,  # From RAG service
                "vector_dimension": settings.qdrant_vector_size
            }
        }
        
        # Check embedding service
        try:
            async with EmbeddingService() as embedding_service:
                embedding_healthy = await embedding_service.health_check()
                model_info = await embedding_service.get_model_info()
                
                health_results["services"]["embedding"] = {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "url": settings.tei_url,
                    "model": settings.tei_model,
                    "model_info": model_info
                }
        except Exception as e:
            health_results["services"]["embedding"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check vector service
        try:
            async with VectorService() as vector_service:
                vector_healthy = await vector_service.health_check()
                collection_info = await vector_service.get_collection_info()
                
                health_results["services"]["vector"] = {
                    "status": "healthy" if vector_healthy else "unhealthy",
                    "url": settings.qdrant_url,
                    "collection": settings.qdrant_collection,
                    "collection_info": collection_info
                }
        except Exception as e:
            health_results["services"]["vector"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check RAG service (combines both)
        try:
            async with RagService() as rag_service:
                rag_health = await rag_service.health_check()
                rag_stats = await rag_service.get_stats()
                
                health_results["services"]["rag"] = {
                    "status": "healthy" if all(rag_health.values()) else "unhealthy",
                    "component_health": rag_health,
                    "stats": rag_stats
                }
        except Exception as e:
            health_results["services"]["rag"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall health status
        service_statuses = [
            service.get("status") == "healthy" 
            for service in health_results["services"].values()
        ]
        
        all_healthy = all(service_statuses)
        health_results["overall_status"] = "healthy" if all_healthy else "degraded"
        
        await ctx.info("Health check completed - Overall status: %s", health_results['overall_status'])
        
        return health_results
        
    except Exception as e:
        error_msg = "Health check failed: %s" % str(e)
        await ctx.info(error_msg)
        raise ToolError(error_msg)


@mcp.tool
async def get_server_info(ctx: Context) -> dict[str, Any]:
    """
    Get detailed information about the server configuration and capabilities.
    
    Returns:
        Dictionary with server information
    """
    await ctx.info("Retrieving server information")
    
    return {
        "server": {
            "name": "Crawlerr",
            "description": "RAG-enabled web crawling MCP server",
            "version": "0.1.0",
            "framework": "FastMCP 2.0+",
            "host": settings.server_host,
            "port": settings.server_port,
            "debug_mode": settings.debug,
            "production_mode": settings.production
        },
        "capabilities": {
            "crawling": {
                "single_page_scraping": True,
                "website_crawling": True,
                "sitemap_support": True,
                "repository_cloning": True,
                "directory_processing": True,
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "supported_browsers": ["chromium"],
                "extraction_strategies": ["css", "llm", "cosine"]
            },
            "rag": {
                "semantic_search": True,
                "automatic_indexing": True,
                "chunk_processing": True,
                "reranking": True,
                "source_filtering": True,
                "embedding_model": settings.tei_model,
                "vector_dimension": settings.qdrant_vector_size,
                "distance_metric": settings.qdrant_distance
            },
            "sources": {
                "source_management": True,
                "metadata_tracking": True,
                "filtering": True,
                "statistics": True,
                "supported_types": ["webpage", "repository", "directory"]
            }
        },
        "configuration": {
            "crawling": {
                "headless": settings.crawl_headless,
                "browser": settings.crawl_browser,
                "max_pages_default": settings.crawl_max_pages,
                "max_depth_default": settings.crawl_max_depth,
                "min_words": settings.crawl_min_words,
                "user_agent": settings.crawl_user_agent,
                "timeout": settings.crawler_timeout,
                "delay": settings.crawler_delay
            },
            "embedding": {
                "service_url": settings.tei_url,
                "model": settings.tei_model,
                "batch_size": settings.tei_batch_size,
                "max_length": settings.embedding_max_length,
                "normalize": settings.embedding_normalize,
                "timeout": settings.tei_timeout
            },
            "vector_database": {
                "service_url": settings.qdrant_url,
                "collection": settings.qdrant_collection,
                "vector_size": settings.qdrant_vector_size,
                "distance": settings.qdrant_distance,
                "timeout": settings.qdrant_timeout
            }
        },
        "available_tools": [
            "scrape - Single page web scraping",
            "crawl - Multi-page website crawling", 
            "crawl_repo - Git repository analysis",
            "crawl_dir - Local directory processing",
            "rag_query - Semantic search queries",
            "list_sources - Source management",
            "health_check - System health monitoring",
            "get_server_info - Server information",
            "get_rag_stats - RAG system statistics",
            "delete_source - Source deletion"
        ]
    }


# Server lifecycle management will be handled by FastMCP automatically
async def startup_checks():
    """Initialize services and perform startup checks."""
    logger.info("[bold]📋 Starting Crawlerr server v0.1.0[/bold]")
    logger.info("[dim]🐛 Debug mode: %s | 🏭 Production: %s[/dim]", settings.debug, settings.production)
    
    # Log service endpoints with emojis
    logger.info("[blue]🗂️  Qdrant endpoint: %s[/blue]", settings.qdrant_url)
    logger.info("[purple]🤖 TEI endpoint: %s[/purple]", settings.tei_url)
    logger.info("[magenta]🧠 TEI model: %s[/magenta]", settings.tei_model)
    logger.info("[yellow]🔄 Reranker model: %s[/yellow]", settings.reranker_model)
    logger.info("[red]⏱️  Crawl timeout: %ss[/red]", settings.crawler_timeout)
    
    # Perform basic health check
    try:
        # Check if services are reachable
        async with EmbeddingService() as embedding_service:
            embedding_healthy = await embedding_service.health_check()
            status = "[green]✅[/green]" if embedding_healthy else "[red]❌[/red]"
            logger.info("🚀 Embedding service health: %s", status)
        
        async with VectorService() as vector_service:
            vector_healthy = await vector_service.health_check()
            # Ensure collection exists
            collection_created = await vector_service.ensure_collection()
            v_status = "[green]✅[/green]" if vector_healthy else "[red]❌[/red]"
            c_status = "[green]✅[/green]" if collection_created else "[red]❌[/red]"
            logger.info("🗄️  Vector service health: %s", v_status)
            logger.info("📚 Vector collection ready: %s", c_status)
        
        logger.info("[bold green]🎉 Crawlerr server started successfully![/bold green]")
        
    except ToolError as e:
        logger.exception("[red]❌ Critical startup error[/red]")
        logger.info("[dim]🤷 Server started but some services may be unavailable[/dim]")
    except (ConnectionError, TimeoutError) as e:
        logger.warning("[yellow]⚠️  Service connection failed: %s[/yellow]", e)
        logger.info("[dim]🤷 Server started but some services may be unavailable[/dim]")
    except Exception:
        logger.exception("[red]💥 Unexpected startup error[/red]")
        logger.info("[dim]🤷 Server started but some services may be unavailable[/dim]")


# CLI entry point
def main() -> None:
    """Main entry point for the CLI."""
    try:
        # Rich startup banner
        console = Console()
        console.print("\n[bold blue]🕷️  Crawly MCP Server[/bold blue]", style="bold blue")
        console.print("[dim]RAG-Enabled Web Crawling with Qwen3 Intelligence[/dim]\n")
        
        # Run startup checks
        logger.info("[green]🚀 Starting Crawlerr FastMCP server...[/green]")
        asyncio.run(startup_checks())
        
        # Start the FastMCP server with HTTP transport
        logger.info("[cyan]🌐 Starting FastMCP server on %s:%s[/cyan]", settings.server_host, settings.server_port)
        
        # Import uvicorn for server
        import uvicorn
        
        # Start the server
        uvicorn.run(
            mcp.http_app(),
            host=settings.server_host,
            port=settings.server_port,
            log_level="info" if settings.debug else "warning"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Server failed to start: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()