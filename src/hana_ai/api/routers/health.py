"""
Health check router for the SAP HANA AI Toolkit API.

This module provides comprehensive health check endpoints for monitoring
the application and all backends, including NVIDIA, Together.ai, and CPU.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
import psutil
import json
from fastapi import APIRouter, Depends, Request, HTTPException, Response
from pydantic import BaseModel

from ..config import settings
from ..backend_config import backend_config, BackendType
from ..backend_manager import backend_manager
from ..environment import get_platform_info, detect_deployment_platform
from ..auth import get_admin_api_key, get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

class BackendHealthStatus(BaseModel):
    """Health status of a backend."""
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Complete health check response."""
    status: str
    service: str = "SAP HANA AI Toolkit API"
    version: str = "1.0.0"
    deployment_mode: str
    deployment_platform: str
    environment: str
    uptime_seconds: float
    backends: Dict[str, BackendHealthStatus]
    system: Dict[str, Any]
    database: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# Service start time
START_TIME = time.time()

@router.get(
    "/backend-status",
    summary="Backend status",
    description="Check the status of all backends (NVIDIA, Together.ai, CPU)"
)
async def backend_status(request: Request, api_key: str = Depends(get_api_key)):
    """
    Get the status of all backends.
    
    This endpoint checks the health of all configured backends
    (NVIDIA, Together.ai, CPU) and returns their status.
    
    Returns:
        Dict: Status of all backends
    """
    result = {}
    
    # Get backend status
    status = backend_manager.get_backend_status()
    
    # Format response
    for backend_type in BackendType:
        if backend_type == BackendType.AUTO:
            continue
            
        backend_name = backend_type.value
        backend_info = status.get("backends", {}).get(backend_name, {})
        
        result[backend_name] = {
            "status": backend_info.get("status", "unknown"),
            "error": backend_info.get("error"),
            "is_primary": backend_type == status.get("primary_backend"),
            "is_secondary": backend_type == status.get("secondary_backend"),
        }
    
    # Add status of failover and load balancing
    result["failover_enabled"] = status.get("auto_failover", False)
    result["load_balancing_enabled"] = status.get("load_balancing", False)
    
    return result

@router.get(
    "/backend-check/{backend_type}",
    summary="Backend health check",
    description="Check the health of a specific backend by running a simple operation"
)
async def backend_check(
    backend_type: str,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """
    Check the health of a specific backend by running a simple operation.
    
    Args:
        backend_type: The backend to check (nvidia, together_ai, cpu)
    
    Returns:
        Dict: Health status of the backend
    """
    # Validate backend type
    try:
        backend = BackendType(backend_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend type: {backend_type}. Must be one of: nvidia, together_ai, cpu"
        )
    
    # Skip auto backend type
    if backend == BackendType.AUTO:
        raise HTTPException(
            status_code=400,
            detail="Cannot check AUTO backend type"
        )
    
    # Initialize backend if not already initialized
    if backend not in backend_manager.initialized_backends:
        try:
            backend_manager.initialize_backend(backend)
        except Exception as e:
            return {
                "status": "unavailable",
                "latency_ms": None,
                "error": str(e),
                "details": None
            }
    
    # Run a simple operation to check health
    try:
        start_time = time.time()
        
        # Generate a simple text
        response = backend_manager.generate_text(
            prompt="Hello, world!",
            max_tokens=5,
            backend=backend
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "error": None,
            "details": {
                "model": response.get("model"),
                "response": response.get("text")
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": str(e),
            "details": None
        }

@router.get(
    "/ping",
    summary="Simple ping check",
    description="Simple ping health check for load balancers and monitoring systems"
)
async def ping():
    """
    Simple ping health check.
    
    This endpoint returns a simple OK response for basic health checking
    by load balancers and monitoring systems.
    
    Returns:
        Dict: Simple health status
    """
    return {"status": "ok"}

@router.get(
    "/health",
    summary="Complete health check",
    description="Comprehensive health check of the entire system"
)
async def health_check(request: Request):
    """
    Comprehensive health check of the entire system.
    
    This endpoint performs a complete health check of the application,
    including all backends, system resources, and database connections.
    
    Returns:
        HealthResponse: Complete health status
    """
    # Prepare response
    response = {
        "status": "healthy",
        "service": "SAP HANA AI Toolkit API",
        "version": "1.0.0",
        "deployment_mode": settings.DEPLOYMENT_MODE,
        "deployment_platform": detect_deployment_platform().value,
        "environment": "production" if not settings.DEVELOPMENT_MODE else "development",
        "uptime_seconds": time.time() - START_TIME,
        "backends": {},
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
    
    # Check backends
    status = backend_manager.get_backend_status()
    overall_healthy = True
    
    for backend_type in BackendType:
        if backend_type == BackendType.AUTO:
            continue
            
        backend_name = backend_type.value
        backend_info = status.get("backends", {}).get(backend_name, {})
        backend_status = backend_info.get("status", "unknown")
        
        # Only check active backends
        if backend_type in backend_manager.initialized_backends:
            # Run a quick health check
            check_result = await backend_check(backend_name, request, None)
            
            response["backends"][backend_name] = {
                "status": check_result["status"],
                "latency_ms": check_result["latency_ms"],
                "error": check_result["error"],
                "details": check_result["details"]
            }
            
            # Update overall health status
            if check_result["status"] != "healthy" and backend_type in [
                backend_config.get_primary_backend(),
                backend_config.get_secondary_backend()
            ]:
                overall_healthy = False
        else:
            response["backends"][backend_name] = {
                "status": "inactive",
                "latency_ms": None,
                "error": None,
                "details": None
            }
    
    # Check database connection if we have one
    if hasattr(request.app.state, "connection_pool") and request.app.state.connection_pool:
        try:
            # Try to get a connection
            conn = next(iter(request.app.state.connection_pool.values()))
            
            # Test query
            result = conn.sql("SELECT 1 FROM DUMMY").collect()
            
            response["database"] = {
                "status": "connected",
                "error": None
            }
        except Exception as e:
            response["database"] = {
                "status": "error",
                "error": str(e)
            }
            overall_healthy = False
    
    # Update overall status
    if not overall_healthy:
        response["status"] = "degraded"
        response["message"] = "One or more critical components are unhealthy"
    
    return response

@router.get(
    "/platform-info",
    summary="Platform information",
    description="Get detailed information about the deployment platform",
    dependencies=[Depends(get_admin_api_key)]
)
async def platform_info():
    """
    Get detailed information about the deployment platform.
    
    This endpoint returns detailed information about the deployment
    platform, including system information, environment variables,
    and backend configuration.
    
    Returns:
        Dict: Platform information
    """
    return get_platform_info()

@router.get(
    "/metrics",
    summary="Health metrics",
    description="Get health metrics in Prometheus format",
    response_class=Response
)
async def metrics():
    """
    Get health metrics in Prometheus format.
    
    This endpoint returns health metrics in Prometheus format
    for monitoring systems.
    
    Returns:
        Response: Prometheus-formatted metrics
    """
    # Prepare metrics
    metrics = []
    
    # System metrics
    metrics.append(f'system_cpu_percent {psutil.cpu_percent()}')
    metrics.append(f'system_memory_percent {psutil.virtual_memory().percent}')
    metrics.append(f'system_disk_percent {psutil.disk_usage("/").percent}')
    
    # Uptime metric
    metrics.append(f'app_uptime_seconds {time.time() - START_TIME}')
    
    # Backend status metrics
    status = backend_manager.get_backend_status()
    
    for backend_type in BackendType:
        if backend_type == BackendType.AUTO:
            continue
            
        backend_name = backend_type.value
        backend_info = status.get("backends", {}).get(backend_name, {})
        backend_status = backend_info.get("status", "unknown")
        
        # Convert status to numeric value (1 = available, 0 = unavailable)
        status_value = 1 if backend_status == "available" else 0
        metrics.append(f'backend_{backend_name}_available {status_value}')
    
    # Return metrics as plain text
    return Response(content="\n".join(metrics), media_type="text/plain")