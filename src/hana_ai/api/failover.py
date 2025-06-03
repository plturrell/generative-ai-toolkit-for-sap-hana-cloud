"""
Failover handling and resilience module for SAP HANA AI Toolkit.

This module provides advanced failover mechanisms and resilience patterns
to ensure high availability of the SAP HANA AI Toolkit in production environments.
"""

import time
import logging
import threading
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

from hana_ai.api.env_constants import (
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
    DEFAULT_RETRY_COUNT,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_BULKHEAD_MAX_CONCURRENT,
    DEFAULT_BULKHEAD_MAX_WAITING,
)

# Type variable for generic functions
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for preventing cascading failures.
    
    This pattern monitors for failures and trips the circuit when a threshold is reached,
    preventing further calls to the failing service and allowing it time to recover.
    """
    
    def __init__(
        self,
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        reset_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
    ):
        """
        Initialize a new CircuitBreaker.
        
        Args:
            failure_threshold: Number of failures before tripping the circuit
            reset_timeout: Time in seconds before attempting to reset the circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to apply circuit breaker pattern to a function.
        
        Args:
            func: The function to wrap with circuit breaker logic
            
        Returns:
            The wrapped function with circuit breaker logic
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with self._lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time >= self.reset_timeout:
                        # Transition to half-open state
                        logger.info(f"Circuit breaker for {func.__name__} transitioning to HALF-OPEN state")
                        self.state = "HALF-OPEN"
                    else:
                        # Circuit is open, fast-fail
                        logger.warning(f"Circuit breaker for {func.__name__} is OPEN - fast failing")
                        raise CircuitBreakerOpenError(f"Circuit breaker for {func.__name__} is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success, potentially reset the circuit
                with self._lock:
                    if self.state == "HALF-OPEN":
                        logger.info(f"Circuit breaker for {func.__name__} resetting to CLOSED state")
                        self.failure_count = 0
                        self.state = "CLOSED"
                
                return result
                
            except Exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if (self.state == "CLOSED" and self.failure_count >= self.failure_threshold) or \
                       self.state == "HALF-OPEN":
                        # Trip the circuit or keep it open after a half-open failure
                        logger.warning(f"Circuit breaker for {func.__name__} tripping to OPEN state")
                        self.state = "OPEN"
                
                # Re-raise the original exception
                raise
        
        return wrapper


class Retry:
    """
    Retry pattern implementation with exponential backoff.
    
    This pattern automatically retries failed operations with configurable
    backoff to handle transient failures.
    """
    
    def __init__(
        self,
        max_retries: int = DEFAULT_RETRY_COUNT,
        delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
        exceptions: Tuple[Exception, ...] = (Exception,),
    ):
        """
        Initialize a new Retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff_factor: Factor by which the delay increases with each retry
            exceptions: Tuple of exceptions that trigger a retry
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to apply retry logic to a function.
        
        Args:
            func: The function to wrap with retry logic
            
        Returns:
            The wrapped function with retry logic
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = self.delay
            
            for attempt in range(self.max_retries + 1):  # +1 for the initial attempt
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{self.max_retries} for {func.__name__} after {current_delay:.2f}s")
                        time.sleep(current_delay)
                        # Increase delay for next attempt
                        current_delay *= self.backoff_factor
                    
                    return func(*args, **kwargs)
                    
                except self.exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries+1} failed for {func.__name__}: {str(e)}")
                    
                    # Don't sleep after the last attempt
                    if attempt == self.max_retries:
                        break
            
            # If we got here, all retries failed
            logger.error(f"All {self.max_retries+1} attempts failed for {func.__name__}")
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"All {self.max_retries+1} attempts failed for {func.__name__}")
        
        return wrapper


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation.
    
    This pattern limits the number of concurrent executions to prevent
    resource exhaustion and isolate failures.
    """
    
    def __init__(
        self,
        max_concurrent: int = DEFAULT_BULKHEAD_MAX_CONCURRENT,
        max_waiting: int = DEFAULT_BULKHEAD_MAX_WAITING,
    ):
        """
        Initialize a new Bulkhead.
        
        Args:
            max_concurrent: Maximum number of concurrent executions
            max_waiting: Maximum number of waiting executions
        """
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.semaphore = threading.Semaphore(max_concurrent)
        self.waiting_count = 0
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to apply bulkhead pattern to a function.
        
        Args:
            func: The function to wrap with bulkhead logic
            
        Returns:
            The wrapped function with bulkhead logic
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if we can add to waiting queue
            with self._lock:
                if self.waiting_count >= self.max_waiting:
                    raise BulkheadFullError(f"Bulkhead for {func.__name__} has reached maximum waiting capacity")
                self.waiting_count += 1
            
            acquired = False
            try:
                # Try to acquire semaphore
                acquired = self.semaphore.acquire(blocking=True, timeout=10)  # 10 second timeout
                if not acquired:
                    raise BulkheadTimeoutError(f"Bulkhead for {func.__name__} timed out waiting for execution slot")
                
                # Decrement waiting count as we're now executing
                with self._lock:
                    self.waiting_count -= 1
                
                # Execute the function
                return func(*args, **kwargs)
                
            finally:
                # Release semaphore if acquired
                if acquired:
                    self.semaphore.release()
                else:
                    # If we never acquired, still need to decrement waiting count
                    with self._lock:
                        self.waiting_count -= 1
        
        return wrapper


class Timeout:
    """
    Timeout pattern implementation to prevent blocking operations.
    
    This pattern ensures that operations complete within a specified time limit.
    """
    
    def __init__(self, seconds: float):
        """
        Initialize a new Timeout.
        
        Args:
            seconds: Maximum execution time in seconds
        """
        self.seconds = seconds
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to apply timeout pattern to a function.
        
        Args:
            func: The function to wrap with timeout logic
            
        Returns:
            The wrapped function with timeout logic
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(self.seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} execution timed out after {self.seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper


class Fallback:
    """
    Fallback pattern implementation for graceful degradation.
    
    This pattern provides alternative behavior when the primary operation fails.
    """
    
    def __init__(self, fallback_func: Callable[..., T]):
        """
        Initialize a new Fallback.
        
        Args:
            fallback_func: Function to call when the primary function fails
        """
        self.fallback_func = fallback_func
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to apply fallback pattern to a function.
        
        Args:
            func: The primary function to wrap with fallback logic
            
        Returns:
            The wrapped function with fallback logic
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {str(e)}")
                return self.fallback_func(*args, **kwargs)
        
        return wrapper


class FailoverManager:
    """
    Comprehensive failover management for high availability.
    
    This class provides a centralized way to manage all resilience patterns
    and handle failover scenarios in the application.
    """
    
    def __init__(self):
        """Initialize a new FailoverManager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_health: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register_service(
        self,
        service_name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        health_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a service with the failover manager.
        
        Args:
            service_name: Unique name for the service
            circuit_breaker: Optional circuit breaker for the service
            health_check: Optional health check function
        """
        with self._lock:
            if circuit_breaker:
                self.circuit_breakers[service_name] = circuit_breaker
            
            self.service_health[service_name] = {
                "health_check": health_check,
                "last_check_time": 0,
                "is_healthy": True,
                "failure_count": 0,
            }
    
    def check_service_health(self, service_name: str) -> bool:
        """
        Check if a service is healthy.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if the service is healthy, False otherwise
        """
        with self._lock:
            if service_name not in self.service_health:
                logger.warning(f"Service {service_name} not registered with failover manager")
                return False
            
            service = self.service_health[service_name]
            
            # If no health check function, assume healthy
            if not service["health_check"]:
                return True
            
            # Check if we should refresh health status
            current_time = time.time()
            if current_time - service["last_check_time"] > 60:  # Check every 60 seconds
                try:
                    is_healthy = service["health_check"]()
                    service["is_healthy"] = is_healthy
                    service["last_check_time"] = current_time
                    
                    if is_healthy:
                        service["failure_count"] = 0
                    else:
                        service["failure_count"] += 1
                        
                except Exception as e:
                    logger.error(f"Health check for {service_name} failed: {str(e)}")
                    service["is_healthy"] = False
                    service["failure_count"] += 1
                    service["last_check_time"] = current_time
            
            return service["is_healthy"]
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all registered services.
        
        Returns:
            Dictionary mapping service names to their current status
        """
        result = {}
        with self._lock:
            for service_name, service in self.service_health.items():
                circuit_state = "UNKNOWN"
                if service_name in self.circuit_breakers:
                    circuit_state = self.circuit_breakers[service_name].state
                
                result[service_name] = {
                    "is_healthy": service["is_healthy"],
                    "failure_count": service["failure_count"],
                    "last_check_time": service["last_check_time"],
                    "circuit_state": circuit_state,
                }
        
        return result


# Custom exception classes
class CircuitBreakerOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass


class BulkheadFullError(Exception):
    """Exception raised when a bulkhead is full."""
    pass


class BulkheadTimeoutError(Exception):
    """Exception raised when a bulkhead times out."""
    pass


# Global failover manager instance
failover_manager = FailoverManager()


# Convenience functions for resilience patterns
def with_retry(
    max_retries: int = DEFAULT_RETRY_COUNT,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    exceptions: Tuple[Exception, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying retry pattern with custom parameters.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        exceptions: Tuple of exceptions that trigger a retry
        
    Returns:
        Decorator function that applies retry pattern
    """
    return Retry(max_retries, delay, backoff_factor, exceptions)


def with_circuit_breaker(
    failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    reset_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying circuit breaker pattern with custom parameters.
    
    Args:
        failure_threshold: Number of failures before tripping the circuit
        reset_timeout: Time in seconds before attempting to reset the circuit
        
    Returns:
        Decorator function that applies circuit breaker pattern
    """
    return CircuitBreaker(failure_threshold, reset_timeout)


def with_bulkhead(
    max_concurrent: int = DEFAULT_BULKHEAD_MAX_CONCURRENT,
    max_waiting: int = DEFAULT_BULKHEAD_MAX_WAITING,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying bulkhead pattern with custom parameters.
    
    Args:
        max_concurrent: Maximum number of concurrent executions
        max_waiting: Maximum number of waiting executions
        
    Returns:
        Decorator function that applies bulkhead pattern
    """
    return Bulkhead(max_concurrent, max_waiting)


def with_timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying timeout pattern with custom parameters.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Returns:
        Decorator function that applies timeout pattern
    """
    return Timeout(seconds)


def with_fallback(fallback_func: Callable[..., T]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying fallback pattern with custom function.
    
    Args:
        fallback_func: Function to call when the primary function fails
        
    Returns:
        Decorator function that applies fallback pattern
    """
    return Fallback(fallback_func)