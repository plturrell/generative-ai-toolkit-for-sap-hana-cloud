"""
Metrics collection for the API server.

Supports Prometheus metrics for monitoring API performance, usage patterns,
and error rates in production environments.
"""
import time
from typing import Dict, Any, Optional
import logging

# Only import prometheus_client if available
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Metrics objects
REQUEST_LATENCY = None
REQUEST_COUNT = None
DB_QUERY_LATENCY = None
LLM_LATENCY = None
ACTIVE_CONNECTIONS = None
ERROR_COUNT = None

def setup_metrics():
    """
    Initialize Prometheus metrics collectors.
    
    This creates metrics for tracking API performance and usage.
    Metrics are limited to internal BTP networks only.
    """
    global REQUEST_LATENCY, REQUEST_COUNT, DB_QUERY_LATENCY, LLM_LATENCY, ACTIVE_CONNECTIONS, ERROR_COUNT
    
    if not PROMETHEUS_AVAILABLE:
        logging.warning("Prometheus client not available. Metrics will not be collected.")
        return
    
    # Request metrics
    REQUEST_COUNT = prom.Counter(
        'api_requests_total',
        'Total number of API requests',
        ['method', 'endpoint', 'status']
    )
    
    REQUEST_LATENCY = prom.Histogram(
        'api_request_latency_seconds',
        'API request latency in seconds',
        ['method', 'endpoint'],
        buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
    )
    
    # Database metrics
    DB_QUERY_LATENCY = prom.Histogram(
        'db_query_latency_seconds',
        'Database query latency in seconds',
        ['query_type'],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
    )
    
    ACTIVE_CONNECTIONS = prom.Gauge(
        'db_active_connections',
        'Number of active database connections'
    )
    
    # LLM metrics
    LLM_LATENCY = prom.Histogram(
        'llm_request_latency_seconds',
        'LLM request latency in seconds',
        ['model'],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, float('inf'))
    )
    
    # Error metrics
    ERROR_COUNT = prom.Counter(
        'api_errors_total',
        'Total number of API errors',
        ['method', 'endpoint', 'error_type']
    )
    
    # Start HTTP server for Prometheus scraping, but only on localhost
    # to ensure metrics are only available within the BTP network
    try:
        from .config import settings
        prom_port = getattr(settings, 'PROMETHEUS_PORT', 9090)
        
        # Restrict metrics server to localhost (internal network only)
        # This ensures metrics aren't exposed externally
        prom.start_http_server(prom_port, addr='127.0.0.1')
        logging.info(f"Prometheus metrics server started on localhost:{prom_port} (internal BTP network only)")
    except Exception as e:
        logging.error(f"Failed to start Prometheus HTTP server: {str(e)}")

def record_request_metric(method: str, path: str, status_code: int, duration: float):
    """
    Record metrics for an API request.
    
    Parameters
    ----------
    method : str
        HTTP method (GET, POST, etc.)
    path : str
        Request path
    status_code : int
        HTTP status code
    duration : float
        Request duration in seconds
    """
    if not PROMETHEUS_AVAILABLE or REQUEST_COUNT is None:
        return
    
    # Normalize path by removing IDs to prevent high cardinality
    normalized_path = path
    
    # Extract endpoint from path (remove version and params)
    parts = path.split('/')
    if len(parts) >= 4 and parts[1] == 'api' and parts[2].startswith('v'):
        # For paths like /api/v1/resource/action
        endpoint = f"/{parts[3]}" + (f"/{parts[4]}" if len(parts) > 4 else "")
    else:
        endpoint = path
    
    # Record request count
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()
    
    # Record request latency
    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)
    
    # Record errors
    if status_code >= 400:
        error_type = 'client_error' if status_code < 500 else 'server_error'
        ERROR_COUNT.labels(
            method=method,
            endpoint=endpoint,
            error_type=error_type
        ).inc()

def record_db_query(query_type: str, duration: float):
    """
    Record metrics for a database query.
    
    Parameters
    ----------
    query_type : str
        Type of query (select, insert, update, delete)
    duration : float
        Query duration in seconds
    """
    if not PROMETHEUS_AVAILABLE or DB_QUERY_LATENCY is None:
        return
    
    DB_QUERY_LATENCY.labels(query_type=query_type).observe(duration)

def update_connection_count(count: int):
    """
    Update the active connections gauge.
    
    Parameters
    ----------
    count : int
        Number of active connections
    """
    if not PROMETHEUS_AVAILABLE or ACTIVE_CONNECTIONS is None:
        return
    
    ACTIVE_CONNECTIONS.set(count)

def record_llm_request(model: str, duration: float):
    """
    Record metrics for an LLM request.
    
    Parameters
    ----------
    model : str
        Name of the LLM model used
    duration : float
        Request duration in seconds
    """
    if not PROMETHEUS_AVAILABLE or LLM_LATENCY is None:
        return
    
    LLM_LATENCY.labels(model=model).observe(duration)

def get_metrics():
    """
    Get current metrics as text.
    
    Returns
    -------
    str
        Prometheus metrics in text format
    """
    if not PROMETHEUS_AVAILABLE:
        return "Prometheus client not available"
    
    return prom.generate_latest().decode('utf-8')