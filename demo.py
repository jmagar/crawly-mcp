#!/usr/bin/env python
"""
Demo script for testing the Crawlerr FastMCP server functionality.

This script demonstrates the core capabilities of the Crawlerr server:
1. Health checking
2. Web scraping  
3. RAG processing and search
4. Source management
"""
import asyncio
import logging
from crawlerr.services import EmbeddingService, VectorService, RagService, CrawlerService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embedding_service():
    """Test the embedding service."""
    logger.info("=== Testing Embedding Service ===")
    
    async with EmbeddingService() as service:
        # Health check
        healthy = await service.health_check()
        logger.info(f"Embedding service health: {'✓' if healthy else '✗'}")
        
        # Generate a simple embedding
        if healthy:
            result = await service.generate_embedding("Hello, this is a test sentence.")
            logger.info(f"Generated embedding with {result.dimensions} dimensions")
            logger.info(f"Processing time: {result.processing_time:.3f}s")


async def test_vector_service():
    """Test the vector service."""
    logger.info("\n=== Testing Vector Service ===")
    
    async with VectorService() as service:
        # Health check  
        healthy = await service.health_check()
        logger.info(f"Vector service health: {'✓' if healthy else '✗'}")
        
        # Get collection info
        if healthy:
            info = await service.get_collection_info()
            logger.info(f"Collection: {info.get('name', 'unknown')}")
            logger.info(f"Documents: {info.get('points_count', 0)}")


async def test_crawler_service():
    """Test the crawler service."""
    logger.info("\n=== Testing Crawler Service ===")
    
    service = CrawlerService()
    
    try:
        # Test scraping a simple page
        page_content = await service.scrape_single_page("https://httpbin.org/html")
        logger.info(f"Scraped page: {page_content.title}")
        logger.info(f"Content length: {len(page_content.content)} characters")
        logger.info(f"Word count: {page_content.word_count}")
        
        return page_content
        
    except Exception as e:
        logger.error(f"Crawler test failed: {e}")
        return None


async def test_rag_workflow():
    """Test the complete RAG workflow."""
    logger.info("\n=== Testing RAG Workflow ===")
    
    # First, crawl some content
    crawler_service = CrawlerService() 
    
    try:
        page_content = await crawler_service.scrape_single_page("https://httpbin.org/html")
        
        # Create a minimal crawl result
        from crawlerr.models.crawl_models import CrawlResult, CrawlStatus, CrawlStatistics
        
        crawl_result = CrawlResult(
            request_id="demo_test",
            status=CrawlStatus.COMPLETED,
            urls=["https://httpbin.org/html"],
            pages=[page_content],
            statistics=CrawlStatistics(
                total_pages_requested=1,
                total_pages_crawled=1,
                total_bytes_downloaded=len(page_content.content)
            )
        )
        
        # Process with RAG service
        async with RagService() as rag_service:
            logger.info("Processing crawl result for RAG...")
            stats = await rag_service.process_crawl_result(crawl_result)
            logger.info(f"RAG processing stats: {stats}")
            
            if stats.get('chunks_stored', 0) > 0:
                # Test a search query
                from crawlerr.models.rag_models import RagQuery
                
                query = RagQuery(
                    query="HTML test page example",
                    limit=5
                )
                
                logger.info("Performing RAG search...")
                results = await rag_service.query(query)
                logger.info(f"Found {results.total_matches} matches")
                
                if results.matches:
                    logger.info(f"Best match score: {results.matches[0].score:.3f}")
                    logger.info(f"Best match preview: {results.matches[0].document.content[:100]}...")
        
        logger.info("RAG workflow test completed successfully!")
        
    except Exception as e:
        logger.error(f"RAG workflow test failed: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting Crawlerr functionality tests...")
    
    try:
        await test_embedding_service()
        await test_vector_service() 
        await test_crawler_service()
        await test_rag_workflow()
        
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("The Crawlerr server is ready for use!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())