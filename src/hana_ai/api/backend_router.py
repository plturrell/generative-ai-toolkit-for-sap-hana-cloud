"""
Backend router for SAP HANA AI Toolkit.

This module provides the routing and selection logic for multiple backend
options including NVIDIA LaunchPad, Together.ai, and CPU-only processing.
"""

import os
import logging
import time
import random
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from functools import wraps

from .backend_config import backend_config, BackendType

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic result type
T = TypeVar('T')

class BackendStatus(str, Enum):
    """Enumeration of backend status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

# Cache for backend status
backend_status_cache: Dict[BackendType, BackendStatus] = {
    BackendType.NVIDIA: BackendStatus.UNKNOWN,
    BackendType.TOGETHER_AI: BackendStatus.UNKNOWN,
    BackendType.CPU: BackendStatus.AVAILABLE,  # CPU is always available
}

# Cache for last backend error
backend_error_cache: Dict[BackendType, Optional[Exception]] = {
    BackendType.NVIDIA: None,
    BackendType.TOGETHER_AI: None,
    BackendType.CPU: None,
}

# Lock for thread safety
backend_status_lock = threading.Lock()

class BackendRequest(Generic[T]):
    """Class for handling backend requests with failover."""
    
    def __init__(
        self,
        nvidia_func: Optional[Callable[..., T]] = None,
        together_ai_func: Optional[Callable[..., T]] = None,
        cpu_func: Optional[Callable[..., T]] = None,
        failover_attempts: Optional[int] = None,
        failover_timeout: Optional[float] = None,
        log_performance: bool = True,
    ):
        """
        Initialize the backend request.
        
        Args:
            nvidia_func: Function to call for NVIDIA backend.
            together_ai_func: Function to call for Together.ai backend.
            cpu_func: Function to call for CPU backend.
            failover_attempts: Number of attempts before failing over.
            failover_timeout: Timeout in seconds before failing over.
            log_performance: Whether to log performance metrics.
        """
        self.backend_funcs = {
            BackendType.NVIDIA: nvidia_func,
            BackendType.TOGETHER_AI: together_ai_func,
            BackendType.CPU: cpu_func,
        }
        
        self.failover_attempts = failover_attempts or backend_config.priority.failover_attempts
        self.failover_timeout = failover_timeout or backend_config.priority.failover_timeout
        self.log_performance = log_performance
    
    def execute(self, *args, **kwargs) -> T:
        """
        Execute the request with the appropriate backend.
        
        Args:
            *args: Positional arguments to pass to the backend function.
            **kwargs: Keyword arguments to pass to the backend function.
            
        Returns:
            T: The result of the backend function.
            
        Raises:
            Exception: If all backends fail.
        """
        # Determine primary and secondary backends
        primary_backend = backend_config.get_primary_backend()
        secondary_backend = backend_config.get_secondary_backend()
        
        # Check if load balancing is enabled
        if backend_config.priority.load_balancing and secondary_backend:
            # Randomly select primary or secondary based on load ratio
            if random.random() > backend_config.priority.load_ratio:
                primary_backend, secondary_backend = secondary_backend, primary_backend
        
        # Try primary backend
        result = self._try_backend(primary_backend, *args, **kwargs)
        if result is not None:
            return result
        
        # If auto failover is enabled and we have a secondary backend, try it
        if backend_config.priority.auto_failover and secondary_backend:
            logger.warning(f"Failing over from {primary_backend} to {secondary_backend}")
            result = self._try_backend(secondary_backend, *args, **kwargs)
            if result is not None:
                return result
        
        # If we get here, all backends failed
        primary_error = backend_error_cache.get(primary_backend)
        secondary_error = backend_error_cache.get(secondary_backend) if secondary_backend else None
        
        error_msg = f"All backends failed. Primary ({primary_backend}): {primary_error}"
        if secondary_backend:
            error_msg += f", Secondary ({secondary_backend}): {secondary_error}"
        
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _try_backend(self, backend_type: BackendType, *args, **kwargs) -> Optional[T]:
        """
        Try to execute the request with the specified backend.
        
        Args:
            backend_type: The backend type to use.
            *args: Positional arguments to pass to the backend function.
            **kwargs: Keyword arguments to pass to the backend function.
            
        Returns:
            Optional[T]: The result of the backend function, or None if it fails.
        """
        # Check if the backend function is available
        backend_func = self.backend_funcs.get(backend_type)
        if backend_func is None:
            logger.warning(f"No function provided for backend {backend_type}")
            return None
        
        # Get current backend status
        with backend_status_lock:
            current_status = backend_status_cache.get(backend_type, BackendStatus.UNKNOWN)
        
        # If backend is known to be unavailable, skip it
        if current_status == BackendStatus.UNAVAILABLE:
            logger.warning(f"Backend {backend_type} is marked as unavailable, skipping")
            return None
        
        # Try the backend with the specified number of attempts
        for attempt in range(self.failover_attempts):
            try:
                # Measure performance if enabled
                start_time = time.time()
                
                # Execute the backend function
                result = backend_func(*args, **kwargs)
                
                # Update backend status to available
                with backend_status_lock:
                    backend_status_cache[backend_type] = BackendStatus.AVAILABLE
                    backend_error_cache[backend_type] = None
                
                # Log performance if enabled
                if self.log_performance:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Backend {backend_type} executed in {elapsed_time:.4f} seconds")
                
                return result
                
            except Exception as e:
                # Log the error
                logger.warning(f"Backend {backend_type} failed on attempt {attempt + 1}/{self.failover_attempts}: {str(e)}")
                
                # Update backend status and error cache
                with backend_status_lock:
                    backend_error_cache[backend_type] = e
                
                # If this is the last attempt, mark the backend as unavailable
                if attempt == self.failover_attempts - 1:
                    with backend_status_lock:
                        backend_status_cache[backend_type] = BackendStatus.UNAVAILABLE
                    logger.error(f"Backend {backend_type} marked as unavailable after {self.failover_attempts} failed attempts")
                
                # Wait before retrying (except on the last attempt)
                if attempt < self.failover_attempts - 1:
                    time.sleep(min(2 ** attempt, self.failover_timeout))  # Exponential backoff
        
        return None

def with_backend_router(
    nvidia_func: Optional[Callable] = None,
    together_ai_func: Optional[Callable] = None,
    cpu_func: Optional[Callable] = None,
    failover_attempts: Optional[int] = None,
    failover_timeout: Optional[float] = None,
    log_performance: bool = True,
):
    """
    Decorator for routing a function call to the appropriate backend.
    
    Args:
        nvidia_func: Function to call for NVIDIA backend.
        together_ai_func: Function to call for Together.ai backend.
        cpu_func: Function to call for CPU backend.
        failover_attempts: Number of attempts before failing over.
        failover_timeout: Timeout in seconds before failing over.
        log_performance: Whether to log performance metrics.
    
    Returns:
        Callable: The decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create backend request
            request = BackendRequest(
                nvidia_func=nvidia_func or (lambda *a, **kw: func(*a, backend=BackendType.NVIDIA, **kw)) if nvidia_func is not None else None,
                together_ai_func=together_ai_func or (lambda *a, **kw: func(*a, backend=BackendType.TOGETHER_AI, **kw)) if together_ai_func is not None else None,
                cpu_func=cpu_func or (lambda *a, **kw: func(*a, backend=BackendType.CPU, **kw)) if cpu_func is not None else None,
                failover_attempts=failover_attempts,
                failover_timeout=failover_timeout,
                log_performance=log_performance,
            )
            
            # Execute the request
            return request.execute(*args, **kwargs)
        
        return wrapper
    
    return decorator

def get_backend_status() -> Dict[str, Any]:
    """
    Get the status of all backends.
    
    Returns:
        Dict[str, Any]: The status of all backends.
    """
    with backend_status_lock:
        return {
            "primary_backend": backend_config.get_primary_backend(),
            "secondary_backend": backend_config.get_secondary_backend(),
            "backends": {
                backend_type.value: {
                    "status": backend_status_cache.get(backend_type, BackendStatus.UNKNOWN).value,
                    "error": str(backend_error_cache.get(backend_type)) if backend_error_cache.get(backend_type) else None,
                }
                for backend_type in BackendType
                if backend_type != BackendType.AUTO
            },
            "load_balancing": backend_config.priority.load_balancing,
            "auto_failover": backend_config.priority.auto_failover,
        }

def mark_backend_available(backend_type: BackendType) -> None:
    """
    Mark a backend as available.
    
    Args:
        backend_type: The backend type to mark as available.
    """
    with backend_status_lock:
        backend_status_cache[backend_type] = BackendStatus.AVAILABLE
        backend_error_cache[backend_type] = None
    
    logger.info(f"Backend {backend_type} marked as available")

def mark_backend_unavailable(backend_type: BackendType, error: Optional[Exception] = None) -> None:
    """
    Mark a backend as unavailable.
    
    Args:
        backend_type: The backend type to mark as unavailable.
        error: The error that caused the backend to be unavailable.
    """
    with backend_status_lock:
        backend_status_cache[backend_type] = BackendStatus.UNAVAILABLE
        backend_error_cache[backend_type] = error
    
    logger.warning(f"Backend {backend_type} marked as unavailable: {error}")

# Export public items
__all__ = [
    "with_backend_router",
    "BackendRequest",
    "get_backend_status",
    "mark_backend_available",
    "mark_backend_unavailable",
]