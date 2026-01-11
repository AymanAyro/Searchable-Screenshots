"""Retry utilities with exponential backoff and circuit breaker pattern."""

import time
import functools
from typing import Callable, TypeVar, Optional, Any
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for failing services."""
    
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout: float = 60.0  # Time to wait before trying again (seconds)
    
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0
    
    def call(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If circuit is open
        """
        # Check if we should try to close the circuit
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
            else:
                raise RuntimeError("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    logger: Optional[Any] = None,
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
        logger: Optional logger instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if logger:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                f"Retrying in {delay:.2f}s..."
                            )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        if logger:
                            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator
