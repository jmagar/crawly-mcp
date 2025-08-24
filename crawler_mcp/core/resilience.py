"""
Resilience patterns for handling failures with exponential backoff and circuit breakers.
"""

import asyncio
import functools
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for logging
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch (None = all)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception or Exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.success_count = 0
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_opens = 0

    def call(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection."""
        return asyncio.run(self.async_call(func))

    async def async_call(self, func: Callable) -> Any:
        """Execute async function with circuit breaker protection."""
        self.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Execute the function
            result = await func() if asyncio.iscoroutinefunction(func) else func()
            
            # Success - update state
            self.success_count += 1
            self.total_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                # Recovery successful
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED state")
            
            return result
            
        except self.expected_exception as e:
            # Failure - update state
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                # Recovery failed, reopen circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after recovery failure")
                
            elif self.failure_count >= self.failure_threshold:
                # Too many failures, open circuit
                self.state = CircuitState.OPEN
                self.circuit_opens += 1
                logger.error(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
                )
            
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "circuit_opens": self.circuit_opens,
            "failure_rate": (
                self.total_failures / self.total_calls 
                if self.total_calls > 0 else 0
            ),
        }


def exponential_backoff(
    max_retries: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
    exponential_base: float | None = None,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] | None = None,
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add randomization to prevent thundering herd
        exceptions: Tuple of exceptions to catch (None = all)
    """
    # Use settings or defaults
    max_retries = max_retries or settings.embedding_max_retries
    initial_delay = initial_delay or settings.retry_initial_delay
    max_delay = max_delay or settings.retry_max_delay
    exponential_base = exponential_base or settings.retry_exponential_base
    exceptions = exceptions or (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_retries} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter if enabled
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Function '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry logic error in {func.__name__}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_retries} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter if enabled
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Function '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    time.sleep(delay)
            
            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry logic error in {func.__name__}")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RateLimiter:
    """
    Token bucket rate limiter for controlling request rates.
    """

    def __init__(
        self,
        rate: int,
        per: float = 1.0,
        burst: int | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Number of requests allowed
            per: Time period in seconds
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.per = per
        self.burst = burst or rate
        
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while tokens > self.tokens:
                # Calculate tokens to add based on time passed
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst,
                    self.tokens + elapsed * (self.rate / self.per)
                )
                self.last_update = now
                
                if tokens > self.tokens:
                    # Still not enough tokens, wait
                    sleep_time = (tokens - self.tokens) * (self.per / self.rate)
                    await asyncio.sleep(sleep_time)
            
            # Consume tokens
            self.tokens -= tokens


# Global circuit breakers for different services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _circuit_breakers[name]


# Pre-configured circuit breakers for services
qdrant_circuit = get_circuit_breaker("qdrant", failure_threshold=5, recovery_timeout=30)
tei_circuit = get_circuit_breaker("tei", failure_threshold=3, recovery_timeout=60)
crawl4ai_circuit = get_circuit_breaker("crawl4ai", failure_threshold=5, recovery_timeout=45)