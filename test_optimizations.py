#!/usr/bin/env python3
"""
Test script to verify all optimizations are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crawler_mcp.core import VectorService, RagService, EmbeddingService
from crawler_mcp.core.connection_pool import get_pool
from crawler_mcp.core.caching import get_embedding_cache, get_query_cache
from crawler_mcp.core.resilience import get_circuit_breaker
from crawler_mcp.config import settings


async def test_imports():
    """Test that all imports work correctly."""
    print("✓ Testing imports...")
    
    # Test vector service (modular)
    try:
        from crawler_mcp.core.vectors import VectorService as ModularVectorService
        print("  ✓ Modular VectorService imported")
    except ImportError as e:
        print(f"  ✗ Failed to import modular VectorService: {e}")
        return False
    
    # Test RAG service (modular)
    try:
        from crawler_mcp.core.rag import RagService as ModularRagService
        print("  ✓ Modular RagService imported")
    except ImportError as e:
        print(f"  ✗ Failed to import modular RagService: {e}")
        return False
    
    # Test new utilities
    try:
        from crawler_mcp.core.connection_pool import QdrantConnectionPool
        from crawler_mcp.core.resilience import CircuitBreaker, exponential_backoff
        from crawler_mcp.core.caching import TTLCache, LRUCache, EmbeddingCache
        from crawler_mcp.core.streaming import ChunkStream, stream_process_chunks
        print("  ✓ All utility modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import utilities: {e}")
        return False
    
    return True


async def test_connection_pool():
    """Test Qdrant connection pooling."""
    print("\n✓ Testing connection pool...")
    
    try:
        pool = get_pool()
        print(f"  ✓ Pool created with size: {pool.size}")
        
        # Initialize pool
        await pool.initialize()
        print(f"  ✓ Pool initialized with {len(pool.connections)} connections")
        
        # Get stats
        stats = pool.get_stats()
        print(f"  ✓ Pool stats: {stats}")
        
        # Clean up
        await pool.close()
        print("  ✓ Pool closed successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Connection pool test failed: {e}")
        return False


async def test_caching():
    """Test caching implementations."""
    print("\n✓ Testing caching...")
    
    try:
        # Test embedding cache
        embedding_cache = get_embedding_cache()
        await embedding_cache.start()
        
        # Cache a test embedding
        test_text = "This is a test text for caching"
        test_embedding = [0.1] * 1024  # Mock embedding
        
        await embedding_cache.set(test_text, test_embedding)
        cached = await embedding_cache.get(test_text)
        
        if cached == test_embedding:
            print("  ✓ Embedding cache working")
        else:
            print("  ✗ Embedding cache not working correctly")
            return False
        
        # Get stats
        stats = embedding_cache.get_stats()
        print(f"  ✓ Cache stats: hits={stats['hits']}, misses={stats['misses']}, size={stats['size']}")
        
        await embedding_cache.stop()
        
        # Test query cache
        query_cache = get_query_cache()
        await query_cache.start()
        
        # Cache a test result
        test_query = "test query"
        test_result = {"results": ["result1", "result2"]}
        
        await query_cache.set(test_result, test_query)
        cached_result = await query_cache.get(test_query)
        
        if cached_result == test_result:
            print("  ✓ Query cache working")
        else:
            print("  ✗ Query cache not working correctly")
            return False
        
        await query_cache.stop()
        
        return True
    except Exception as e:
        print(f"  ✗ Caching test failed: {e}")
        return False


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n✓ Testing circuit breaker...")
    
    try:
        cb = get_circuit_breaker("test", failure_threshold=2, recovery_timeout=1)
        
        # Test normal operation
        async def success_func():
            return "success"
        
        result = await cb.async_call(success_func)
        if result == "success":
            print("  ✓ Circuit breaker allows successful calls")
        
        # Test failure handling
        failure_count = 0
        async def failure_func():
            nonlocal failure_count
            failure_count += 1
            raise RuntimeError("Test failure")
        
        # Should fail twice and open circuit
        for i in range(2):
            try:
                await cb.async_call(failure_func)
            except RuntimeError:
                pass
        
        # Circuit should be open now
        try:
            await cb.async_call(success_func)
            print("  ✗ Circuit breaker didn't open after failures")
            return False
        except RuntimeError as e:
            if "OPEN" in str(e):
                print("  ✓ Circuit breaker opened after failures")
        
        # Get stats
        stats = cb.get_stats()
        print(f"  ✓ Circuit breaker stats: state={stats['state']}, failures={stats['total_failures']}")
        
        cb.reset()
        
        return True
    except Exception as e:
        print(f"  ✗ Circuit breaker test failed: {e}")
        return False


async def test_config_changes():
    """Test configuration changes."""
    print("\n✓ Testing configuration...")
    
    # Check that feature flag is removed
    if not hasattr(settings, 'use_modular_vectors'):
        print("  ✓ Feature flag 'use_modular_vectors' removed")
    else:
        print("  ✗ Feature flag still exists")
        return False
    
    # Check new configuration fields
    if hasattr(settings, 'default_batch_size'):
        print(f"  ✓ Default batch size configured: {settings.default_batch_size}")
    else:
        print("  ✗ Default batch size not configured")
        return False
    
    if hasattr(settings, 'retry_initial_delay'):
        print(f"  ✓ Retry configuration added: initial_delay={settings.retry_initial_delay}s")
    else:
        print("  ✗ Retry configuration not added")
        return False
    
    # Check batch sizes are consistent
    if settings.tei_batch_size == 256:
        print(f"  ✓ TEI batch size increased to optimal: {settings.tei_batch_size}")
    else:
        print(f"  ⚠ TEI batch size not optimal: {settings.tei_batch_size}")
    
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("CRAWLER MCP OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Connection Pool", test_connection_pool),
        ("Caching", test_caching),
        ("Circuit Breaker", test_circuit_breaker),
        ("Configuration", test_config_changes),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL OPTIMIZATIONS WORKING CORRECTLY!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)