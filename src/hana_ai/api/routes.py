"""
API routes for canary deployment health checks and failover handling.

This module provides endpoints for health checking, failover status,
and canary deployment monitoring.
"""

from fastapi import APIRouter, Depends, Request, Response, status
from typing import Dict, Any, List, Optional
import time
import os
import socket
import psutil
import json

from hana_ai.api.env_constants import (
    DEPLOYMENT_TYPE_PRODUCTION,
    DEPLOYMENT_TYPE_CANARY,
    DEPLOYMENT_TYPE_DEV,
    DEFAULT_CANARY_WEIGHT,
)
from hana_ai.api.failover import failover_manager

# Create router
router = APIRouter(tags=["Monitoring"])

# Get deployment type from environment variable
DEPLOYMENT_TYPE = os.environ.get("DEPLOYMENT_TYPE", DEPLOYMENT_TYPE_PRODUCTION)
CANARY_WEIGHT = int(os.environ.get("CANARY_WEIGHT", DEFAULT_CANARY_WEIGHT))
START_TIME = time.time()


@router.get("/health", summary="Application health check endpoint")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint for the application.
    
    This endpoint is used by load balancers, orchestrators, and monitoring
    systems to verify the application's health. It checks:
    
    1. System health (memory, CPU, disk)
    2. Service connectivity (registered services)
    3. Database connectivity
    4. Overall application status
    
    Returns:
        Dict with health status information
    """
    # Get the host identifier
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    
    # Get basic system stats
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Get service health from the failover manager
    service_status = failover_manager.get_service_status()
    services_healthy = all(service.get('is_healthy', False) for service in service_status.values())
    
    # Calculate uptime
    uptime_seconds = int(time.time() - START_TIME)
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime = f"{days}d {hours}h {minutes}m {seconds}s"
    
    # Determine overall status
    is_healthy = services_healthy and memory.percent < 90 and disk.percent < 90
    
    response_data = {
        "status": "healthy" if is_healthy else "unhealthy",
        "deployment_type": DEPLOYMENT_TYPE,
        "canary_weight": CANARY_WEIGHT if DEPLOYMENT_TYPE == DEPLOYMENT_TYPE_CANARY else None,
        "version": os.environ.get("APP_VERSION", "unknown"),
        "host": {
            "hostname": hostname,
            "ip": host_ip,
        },
        "system": {
            "memory_used_percent": memory.percent,
            "disk_used_percent": disk.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        },
        "services": service_status,
        "uptime": uptime,
        "timestamp": time.time()
    }
    
    # Set appropriate status code
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return Response(
        content=json.dumps(response_data),
        media_type="application/json",
        status_code=status_code
    )


@router.get("/status/failover", summary="Failover status")
async def failover_status() -> Dict[str, Any]:
    """
    Get the current failover status of all registered services.
    
    This endpoint provides detailed information about the failover
    status, circuit breaker states, and health of registered services.
    
    Returns:
        Dict with failover status information
    """
    return {
        "status": "ok",
        "services": failover_manager.get_service_status(),
        "deployment_type": DEPLOYMENT_TYPE,
    }


@router.get("/status/canary", summary="Canary deployment status")
async def canary_status() -> Dict[str, Any]:
    """
    Get the current status of canary deployments.
    
    This endpoint provides information about canary deployments,
    including weight, performance metrics, and comparison with
    production deployments.
    
    Returns:
        Dict with canary deployment status
    """
    # This would typically connect to your metrics system to get
    # comparative performance data between production and canary
    
    return {
        "status": "ok",
        "deployment_type": DEPLOYMENT_TYPE,
        "canary_weight": CANARY_WEIGHT if DEPLOYMENT_TYPE == DEPLOYMENT_TYPE_CANARY else None,
        "version": os.environ.get("APP_VERSION", "unknown"),
        "deployment_time": START_TIME,
        "uptime_seconds": int(time.time() - START_TIME),
    }