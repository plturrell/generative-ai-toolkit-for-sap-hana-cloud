"""
Main FastAPI application for the HANA AI Toolkit API.
"""
import logging
import socket
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .config import settings
from .routers import agents, dataframes, tools, vectorstore, config, developer, backend, health
from .middleware import (
    RequestLoggerMiddleware, 
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    MetricsMiddleware
)
from .logging import setup_logging
from .metrics import setup_metrics, get_metrics
from .security import validate_external_request
from .validation import validate_environment

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize metrics
setup_metrics()

# Create FastAPI application
app = FastAPI(
    title="SAP HANA AI Toolkit API",
    description="REST API for the Generative AI Toolkit for SAP HANA Cloud",
    version="1.0.0",
    docs_url="/api/docs" if settings.DEVELOPMENT_MODE else None,
    redoc_url="/api/redoc" if settings.DEVELOPMENT_MODE else None,
)

# Custom middleware to prevent external calls
class ExternalCallPreventionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check if the request is trying to access external resources
        # This is a simplified implementation focused on the Host header
        if settings.RESTRICT_EXTERNAL_CALLS:
            host = request.headers.get("Host", "")
            if host:
                is_valid, reason = validate_external_request(f"https://{host}")
                if not is_valid:
                    logger.warning(f"Blocked potential external request: {reason}")
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "External calls are not allowed by this application",
                            "reason": reason
                        }
                    )
        
        # Process the request
        return await call_next(request)

# Add middleware
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ExternalCallPreventionMiddleware)  # Add our new middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)

# Add HTTPS redirect in production
if not settings.DEVELOPMENT_MODE and settings.ENFORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# Add CORS middleware with dynamic configuration based on frontend URL
# If FRONTEND_URL is specified, add it to allowed origins
cors_origins = settings.CORS_ORIGINS.copy()
if settings.FRONTEND_URL and settings.FRONTEND_URL not in cors_origins:
    cors_origins.append(settings.FRONTEND_URL)
    logger.info(f"Added frontend URL to CORS origins: {settings.FRONTEND_URL}")

# If we're in API_ONLY mode and no specific origins are set, allow all origins by default
if settings.DEPLOYMENT_MODE == "api_only" and not cors_origins:
    cors_origins = ["*"]
    logger.info("API-only mode with no CORS origins specified, allowing all origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Backend-Used"],
)

# Mount static files and UI components only if not in API-only mode
import os
from .env_constants import DEPLOYMENT_MODE_API_ONLY, DEPLOYMENT_MODE_UI_ONLY

# Setup UI proxy route if we're in UI-only mode and API_BASE_URL is specified
if settings.DEPLOYMENT_MODE == DEPLOYMENT_MODE_UI_ONLY and settings.API_BASE_URL:
    from fastapi.responses import RedirectResponse
    
    @app.get("/api/v1/{path:path}", include_in_schema=False)
    async def proxy_api_request(path: str, request: Request):
        """Proxy API requests to the specified API_BASE_URL."""
        target_url = f"{settings.API_BASE_URL}/api/v1/{path}"
        # For GET requests, redirect to the target URL
        if request.method == "GET":
            return RedirectResponse(url=target_url)
        # For other methods, return information about where to send the request
        return {
            "message": "API endpoint is located on a separate backend server",
            "api_url": target_url,
            "documentation": f"{settings.API_BASE_URL}/api/docs"
        }
    
    logger.info(f"UI-only mode configured with API proxy to: {settings.API_BASE_URL}")

# Only mount static files if we're not in API-only mode
if settings.DEPLOYMENT_MODE != DEPLOYMENT_MODE_API_ONLY:
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Register routers
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(dataframes.router, prefix="/api/v1/dataframes", tags=["DataFrames"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["Tools"])
app.include_router(vectorstore.router, prefix="/api/v1/vectorstore", tags=["VectorStore"])
app.include_router(config.router, prefix="/api/v1/config", tags=["Configuration"])
app.include_router(backend.router, prefix="/api/v1/backend", tags=["Backend"])
app.include_router(developer.router, prefix="/api/v1/developer", tags=["Developer"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])

# Only include UI endpoints if we're not in API-only mode
if settings.DEPLOYMENT_MODE != DEPLOYMENT_MODE_API_ONLY:
    # Admin panel
    @app.get("/admin", response_class=HTMLResponse, tags=["Admin"])
    async def admin_panel():
        """Serve the admin panel UI."""
        admin_html_path = os.path.join(static_dir, "admin", "index.html")
        with open(admin_html_path, "r") as f:
            return f.read()

    # Developer Studio
    @app.get("/developer", response_class=HTMLResponse, tags=["Developer"])
    async def developer_studio():
        """Serve the developer studio UI."""
        developer_html_path = os.path.join(static_dir, "developer", "index.html")
        with open(developer_html_path, "r") as f:
            return f.read()

    # Root endpoint - redirect to developer studio
    @app.get("/", tags=["Root"])
    async def root():
        """Redirect to developer studio."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/developer")
else:
    # API-only mode root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API information endpoint."""
        return {
            "name": "SAP HANA AI Toolkit API",
            "version": "1.0.0",
            "mode": "api_only",
            "description": "REST API for the Generative AI Toolkit for SAP HANA Cloud",
            "docs_url": "/api/docs" if settings.DEVELOPMENT_MODE else None
        }

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    """Initialize resources on application startup."""
    from .environment import detect_deployment_platform, apply_platform_defaults
    
    # Detect deployment platform and apply platform-specific defaults
    platform = detect_deployment_platform()
    logger.info(f"Starting HANA AI Toolkit API on platform: {platform}")
    
    # Apply platform-specific configuration defaults
    apply_platform_defaults(settings)
    
    # Set NVIDIA environment variables for optimal GPU performance
    if settings.ENABLE_GPU_ACCELERATION:
        import os
        # Set CUDA device order
        os.environ["CUDA_DEVICE_ORDER"] = settings.NVIDIA_CUDA_DEVICE_ORDER
        # Set visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.NVIDIA_CUDA_VISIBLE_DEVICES
        # Set TF32 precision (Ampere+ GPUs)
        os.environ["NVIDIA_TF32_OVERRIDE"] = str(settings.NVIDIA_TF32_OVERRIDE)
        # Set CUDA cache settings
        os.environ["CUDA_CACHE_MAXSIZE"] = str(settings.NVIDIA_CUDA_CACHE_MAXSIZE)
        os.environ["CUDA_CACHE_PATH"] = settings.NVIDIA_CUDA_CACHE_PATH
        
        # Initialize PyTorch settings if available
        try:
            import torch
            # Enable TF32 precision if available
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                
            # Set memory fraction
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(settings.CUDA_MEMORY_FRACTION, i)
                    
            logger.info(f"GPU acceleration enabled with {torch.cuda.device_count()} devices")
        except ImportError:
            logger.info("PyTorch not available, some GPU optimizations disabled")
        except Exception as e:
            logger.warning(f"Error configuring GPU settings: {str(e)}")
    
    # Initialize GPU detection and optimization
    try:
        from .gpu_detection import gpu_optimizer, get_gpu_info
        
        # Get GPU information
        gpu_info = get_gpu_info()
        app.state.gpu_info = gpu_info
        
        if gpu_info["has_gpu"]:
            logger.info(f"Detected GPU: {gpu_info['architecture']} architecture, {gpu_info['count']} device(s)")
            for i, model in enumerate(gpu_info["models"]):
                memory = gpu_info["memory_gb"][i] if i < len(gpu_info["memory_gb"]) else "unknown"
                logger.info(f"  GPU {i}: {model} with {memory} GB memory")
            
            # Store GPU optimizer
            app.state.gpu_optimizer = gpu_optimizer
            
            # For T4 GPUs, initialize specific T4 optimizer
            if "turing" in gpu_info["architecture"].lower() or any("t4" in model.lower() for model in gpu_info["models"]):
                from .gpu_utils_t4 import optimize_for_t4
                app.state.t4_optimizer = optimize_for_t4()
                logger.info("Initialized T4-specific GPU optimizations")
        else:
            logger.info("No GPU detected, running in CPU-only mode")
    except Exception as e:
        logger.warning(f"Error during GPU detection and optimization: {str(e)}")
    
    # Initialize Multi-GPU Manager if GPU acceleration is enabled
    if settings.ENABLE_GPU_ACCELERATION:
        from .gpu_utils import MultiGPUManager
        try:
            app.state.gpu_manager = MultiGPUManager(
                strategy=settings.MULTI_GPU_STRATEGY,
                memory_fraction=settings.CUDA_MEMORY_FRACTION,
                enable_mixed_precision=True
            )
            logger.info(f"Multi-GPU Manager initialized with strategy: {settings.MULTI_GPU_STRATEGY}")
        except Exception as e:
            logger.warning(f"Failed to initialize Multi-GPU Manager: {str(e)}")
    
    # Run environment validation
    validation_results = validate_environment()
    app.state.validation_results = validation_results
    
    # Log validation results
    if validation_results["overall"]["status"] == "error":
        logger.error(
            f"Environment validation failed: {validation_results['overall']['message']}", 
            extra={"validation": validation_results}
        )
    elif validation_results["overall"]["status"] == "warning":
        logger.warning(
            f"Environment validation warnings: {validation_results['overall']['message']}",
            extra={"validation": validation_results}
        )
    else:
        logger.info(
            "Environment validation passed successfully",
            extra={"validation": validation_results}
        )
    
    # Initialize connection pool
    app.state.connection_pool = {}
    
    # Initialize agent sessions
    app.state.agent_sessions = {}

@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down HANA AI Toolkit API")
    # Close all database connections
    for conn in app.state.connection_pool.values():
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")

# Health check endpoint
@app.get(
    "/", 
    tags=["Health"],
    summary="Health check",
    description="Verify that the API is operational and check the status of dependencies"
)
async def health_check(request: Request):
    """
    Health check endpoint to verify the API is running.
    
    Also checks connection to database if available and reports GPU status.
    """
    from hana_ml.dataframe import ConnectionContext
    from .dependencies import get_connection_context
    
    # Use validation results if available
    if hasattr(app.state, "validation_results"):
        status = app.state.validation_results["overall"]["status"]
        health_status = {
            "status": "healthy" if status == "ok" else ("degraded" if status == "warning" else "unhealthy"),
            "service": "HANA AI Toolkit API",
            "version": "1.0.0",
            "message": app.state.validation_results["overall"]["message"]
        }
    else:
        health_status = {
            "status": "healthy", 
            "service": "HANA AI Toolkit API",
            "version": "1.0.0"
        }
    
    # Check database if we have connections
    if hasattr(app.state, "connection_pool") and app.state.connection_pool:
        try:
            # Try to get a connection
            conn = next(iter(app.state.connection_pool.values()))
            
            # Test query
            conn.sql("SELECT 1 FROM DUMMY").collect()
            health_status["database"] = "connected"
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["database"] = "disconnected"
            health_status["database_error"] = str(e)
    
    # Add GPU information if available
    if hasattr(app.state, "gpu_info"):
        gpu_info = app.state.gpu_info
        health_status["gpu"] = {
            "available": gpu_info["has_gpu"],
            "count": gpu_info["count"],
            "architecture": gpu_info["architecture"]
        }
        
        if gpu_info["has_gpu"]:
            health_status["gpu"]["models"] = gpu_info["models"]
            
            # Add T4-specific information if applicable
            if hasattr(app.state, "t4_optimizer"):
                health_status["gpu"]["t4_optimized"] = True
    
    return health_status

# Additional /health endpoint for compatibility with standard health check patterns
@app.get(
    "/health", 
    tags=["Health"],
    summary="Health check",
    description="Verify that the API is operational - simplified endpoint for monitoring systems"
)
async def health_endpoint(request: Request):
    """
    Simplified health check endpoint for monitoring systems.
    Returns 200 OK if the service is running.
    """
    # Use validation results if available
    if hasattr(app.state, "validation_results"):
        status = app.state.validation_results["overall"]["status"]
        if status == "error":
            return JSONResponse(
                status_code=503,  # Service Unavailable
                content={"status": "unhealthy"}
            )
    
    # Return simple healthy response
    return {"status": "healthy"}

# Detailed validation endpoint
@app.get(
    "/validate",
    tags=["Health"],
    summary="Environment validation",
    description="Run comprehensive environment validation checks"
)
async def validation_check(request: Request):
    """
    Run validation checks on the environment.
    
    This endpoint runs comprehensive checks on the environment, including:
    - System configuration
    - NVIDIA GPU availability and configuration
    - HANA database connectivity
    - SAP AI Core SDK integration
    - Required environment variables
    """
    # Trigger a new validation
    from .validation import validate_environment
    validation_results = validate_environment()
    
    # Update stored results
    app.state.validation_results = validation_results
    
    return validation_results

# Metrics endpoint - Restricted to BTP internal networks only
@app.get(
    "/metrics", 
    tags=["Monitoring"],
    summary="Prometheus metrics (internal BTP networks only)",
    description="Export Prometheus-compatible metrics (restricted to internal BTP networks)"
)
async def metrics(request: Request):
    """
    Endpoint for scraping Prometheus metrics.
    Restricted to BTP internal networks only.
    """
    # Get client IP address
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else client_host
    
    # Check if IP is internal
    is_internal = False
    
    # Local development IPs are always allowed
    if ip in ["127.0.0.1", "localhost", "::1"]:
        is_internal = True
    else:
        # Use the security module to check if IP is within BTP ranges
        from .security import is_ip_in_range, BTP_IP_RANGES
        for ip_range in BTP_IP_RANGES:
            if is_ip_in_range(ip, ip_range):
                is_internal = True
                break
    
    if not is_internal:
        logger.warning(f"Blocked metrics access from external IP: {ip}")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Metrics endpoint is only available from internal BTP networks"
            }
        )
    
    return Response(content=get_metrics(), media_type="text/plain")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    
    Logs the exception and returns a consistent error response.
    """
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}",
        exc_info=exc
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"An unexpected error occurred: {str(exc)}",
            "type": type(exc).__name__
        }
    )