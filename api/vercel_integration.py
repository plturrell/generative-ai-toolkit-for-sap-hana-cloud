"""
Vercel Frontend Integration for SAP HANA Cloud Generative AI Toolkit on T4 GPU

This module provides the integration layer between the Vercel frontend and the 
T4 GPU backend running on Brev Cloud. It handles authentication, request routing,
and performance optimization for API calls.
"""

import os
import time
import json
import logging
import sys
import traceback
import requests
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel, Field
import jwt
from jwt.exceptions import InvalidTokenError

# Import debug handler
try:
    from api.debug_handler import debug_exception_handler, log_request, log_response
    DEBUG_HANDLER_AVAILABLE = True
except ImportError:
    DEBUG_HANDLER_AVAILABLE = False

# Import timeout manager
try:
    from api.timeout_manager import get_timeout, get_all_timeouts
    TIMEOUT_MANAGER_AVAILABLE = True
except ImportError:
    TIMEOUT_MANAGER_AVAILABLE = False

# Import T4 GPU optimizer if available
try:
    from api.t4_gpu_optimizer import get_t4_gpu_info
    T4_OPTIMIZER_AVAILABLE = True
except ImportError:
    T4_OPTIMIZER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
T4_GPU_BACKEND_URL = os.getenv("T4_GPU_BACKEND_URL", "https://jupyter0-513syzm60.brevlab.com")
JWT_SECRET = os.getenv("JWT_SECRET", "sap-hana-generative-ai-t4-integration-secret-key-2025")  # Should be set in environment variables
VERCEL_URL = os.getenv("VERCEL_URL", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))

# Function to handle backend request exceptions
def handle_backend_exception(e, endpoint_path=None):
    """
    Handles backend request exceptions with more user-friendly error messages
    
    Args:
        e: The exception to handle
        endpoint_path: Optional path of the endpoint being called
        
    Returns:
        HTTPException with appropriate status code and detailed message
    """
    logger.error(f"Backend request error: {str(e)}")
    
    # Provide a more helpful error message
    error_detail = str(e)
    status_code = 503
    
    # Get timeout information if available
    timeout_info = ""
    if TIMEOUT_MANAGER_AVAILABLE and endpoint_path:
        timeout_value = get_timeout(endpoint_path)
        timeout_info = f" The request timed out after {timeout_value} seconds."
    
    # Check if the error is a timeout
    if "timeout" in error_detail.lower() or isinstance(e, requests.exceptions.Timeout):
        error_detail = f"Backend service timed out.{timeout_info} This may be because the T4 GPU is initializing or under heavy load. Please try again in a few moments."
        status_code = 504  # Gateway Timeout
    
    # Check if the error is a connection error
    elif ("connection" in error_detail.lower() or 
          "connectederror" in error_detail.lower() or 
          isinstance(e, requests.exceptions.ConnectionError)):
        error_detail = f"Unable to connect to the T4 GPU backend. The backend service may be down or unreachable. Please verify the backend URL: {T4_GPU_BACKEND_URL}"
    
    # Check if the error is a DNS resolution error
    elif "gaierror" in error_detail.lower() or "name or service not known" in error_detail.lower():
        error_detail = f"DNS resolution failed for the T4 GPU backend. The hostname could not be resolved. Please check if the backend URL is correct: {T4_GPU_BACKEND_URL}"
    
    # Check for SSL/TLS errors
    elif "ssl" in error_detail.lower() or "certificate" in error_detail.lower():
        error_detail = f"SSL/TLS error when connecting to the T4 GPU backend. This could be due to certificate validation issues."
    
    # Check for proxy issues
    elif "proxy" in error_detail.lower():
        error_detail = f"Proxy error when connecting to the T4 GPU backend. This could be due to incorrect proxy configuration or network restrictions."
    
    # Add debug information for development environments
    if ENVIRONMENT != "production":
        error_detail += f"\n\nDebug information: {type(e).__name__}: {str(e)}"
    
    return HTTPException(
        status_code=status_code,
        detail=error_detail
    )

# Models for API requests and responses
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    use_tensorrt: Optional[bool] = True
    precision: Optional[str] = "int8"
    batch_size: Optional[int] = None  # Use dynamic batch sizing if None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    processing_time_ms: float
    gpu_used: bool
    batch_size_used: Optional[int] = None

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 4
    filter: Optional[Dict[str, Any]] = None
    table_name: str

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    processing_time_ms: float
    query: str

class MMRSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 4
    lambda_mult: Optional[float] = 0.5
    filter: Optional[Dict[str, Any]] = None
    table_name: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    request_id: str

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="SAP HANA Cloud Generative AI Toolkit",
    description="API for SAP HANA Cloud Generative AI Toolkit with T4 GPU Acceleration",
    version="1.0.0",
    debug=ENVIRONMENT != "production",  # Enable debug mode in non-production environments
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"https://{VERCEL_URL}",
        "http://localhost:3000",
        "https://localhost:3000",
    ] if ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add debug exception handler if available
if DEBUG_HANDLER_AVAILABLE:
    from fastapi.exceptions import RequestValidationError
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return await debug_exception_handler(request, exc)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return await debug_exception_handler(request, exc)
    
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        # Log the incoming request
        await log_request(request)
        
        # Process the request
        response = await call_next(request)
        
        # Clone the response to log it
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Log the response
        await log_response(response, response_body)
        
        # Return a new response with the same body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

# Authentication functions
def create_access_token(data: dict) -> str:
    """Create a JWT token"""
    to_encode = data.copy()
    expire = time.time() + 3600  # 1 hour expiration
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

async def get_current_user(authorization: str = Header(None)) -> Optional[str]:
    """Validate JWT token and return username"""
    if not authorization:
        return None
        
    try:
        # Extract token from "Bearer {token}"
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username = payload.get("sub")
        return username
    except (InvalidTokenError, IndexError):
        return None

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to response headers and to the request state"""
    request_id = f"req_{int(time.time() * 1000)}"
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Error handling middleware
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    """Global exception handler"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": str(e) if ENVIRONMENT != "production" else None,
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )

# Authentication credential validation
def validate_credentials(username: str, password: str) -> bool:
    """
    Validate user credentials.
    
    In a production environment, this should connect to a secure credential store.
    For this demonstration, we use a simplified approach with a predefined
    set of valid credentials.
    """
    # Valid credentials for demonstration
    valid_credentials = {
        "admin": "sap-hana-t4-admin",
        "demo": "sap-hana-t4-demo",
        "user": "sap-hana-t4-user"
    }
    
    # Check if username exists and password matches
    return username in valid_credentials and password == valid_credentials[username]

# Login request model
class LoginRequest(BaseModel):
    username: str
    password: str

# Authentication endpoint
@app.post("/api/auth/token", response_model=Token)
async def login_for_access_token(request: LoginRequest):
    """Get JWT token for authentication"""
    # Validate credentials
    if validate_credentials(request.username, request.password):
        # Create token with user information
        token_data = {
            "sub": request.username,
            "role": "admin" if request.username == "admin" else "user"
        }
        access_token = create_access_token(token_data)
        return {"access_token": access_token, "token_type": "bearer"}
    
    # Log failed login attempts (for security monitoring)
    logger.warning(f"Failed login attempt for user: {request.username}")
    
    # Return generic error message (don't reveal if username exists)
    raise HTTPException(
        status_code=401,
        detail="Invalid username or password"
    )

# API endpoints that proxy to the T4 GPU backend
@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Generate embeddings using T4 GPU backend"""
    try:
        # Optional authentication check
        if ENVIRONMENT == "production" and not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get appropriate timeout for embeddings endpoint
        timeout_value = get_timeout("embeddings") if TIMEOUT_MANAGER_AVAILABLE else DEFAULT_TIMEOUT
        logger.debug(f"Using timeout of {timeout_value}s for embeddings endpoint")
        
        # Forward request to backend
        # Use model_dump instead of dict() for newer Pydantic compatibility
        response = requests.post(
            f"{T4_GPU_BACKEND_URL}/api/embeddings",
            json=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
            timeout=timeout_value
        )
        
        # Handle errors
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        # Return response
        return response.json()
    except requests.RequestException as e:
        raise handle_backend_exception(e, "embeddings")

@app.post("/api/vectorstore/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Perform similarity search using T4 GPU backend"""
    try:
        # Optional authentication check
        if ENVIRONMENT == "production" and not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get appropriate timeout for search endpoint
        timeout_value = get_timeout("search") if TIMEOUT_MANAGER_AVAILABLE else DEFAULT_TIMEOUT
        logger.debug(f"Using timeout of {timeout_value}s for vectorstore/search endpoint")
        
        # Forward request to backend
        # Use model_dump instead of dict() for newer Pydantic compatibility
        response = requests.post(
            f"{T4_GPU_BACKEND_URL}/api/vectorstore/search",
            json=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
            timeout=timeout_value
        )
        
        # Handle errors
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        # Return response
        return response.json()
    except requests.RequestException as e:
        raise handle_backend_exception(e, "search")

@app.post("/api/vectorstore/mmr_search", response_model=SearchResponse)
async def mmr_search(
    request: MMRSearchRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Perform MMR search using T4 GPU backend"""
    try:
        # Optional authentication check
        if ENVIRONMENT == "production" and not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get appropriate timeout for MMR search endpoint
        timeout_value = get_timeout("mmr_search") if TIMEOUT_MANAGER_AVAILABLE else DEFAULT_TIMEOUT
        logger.debug(f"Using timeout of {timeout_value}s for vectorstore/mmr_search endpoint")
        
        # Forward request to backend
        # Use model_dump instead of dict() for newer Pydantic compatibility
        response = requests.post(
            f"{T4_GPU_BACKEND_URL}/api/vectorstore/mmr_search",
            json=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
            timeout=timeout_value
        )
        
        # Handle errors
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        # Return response
        return response.json()
    except requests.RequestException as e:
        raise handle_backend_exception(e, "mmr_search")

@app.get("/api/health")
async def health_check():
    """Check health of the API and backend"""
    try:
        # Get appropriate timeout for health check endpoint
        timeout_value = get_timeout("health") if TIMEOUT_MANAGER_AVAILABLE else 10
        logger.debug(f"Using timeout of {timeout_value}s for health check endpoint")
        
        # Check backend health
        backend_response = requests.get(
            f"{T4_GPU_BACKEND_URL}/api/health",
            timeout=timeout_value
        )
        
        backend_status = "healthy" if backend_response.status_code == 200 else "unhealthy"
        backend_details = backend_response.json() if backend_response.status_code == 200 else None
        
        # Get T4 GPU info if available
        gpu_info = get_t4_gpu_info() if T4_OPTIMIZER_AVAILABLE else {"status": "unavailable"}
        
        return {
            "status": "healthy" if backend_status == "healthy" else "degraded",
            "api_version": "1.0.0",
            "backend": {
                "status": backend_status,
                "details": backend_details
            },
            "gpu": gpu_info,
            "timeouts": get_all_timeouts() if TIMEOUT_MANAGER_AVAILABLE else {"default": DEFAULT_TIMEOUT}
        }
    except requests.RequestException as e:
        logger.error(f"Backend health check error: {str(e)}")
        
        # Get T4 GPU info even if backend is unreachable
        gpu_info = get_t4_gpu_info() if T4_OPTIMIZER_AVAILABLE else {"status": "unavailable"}
        
        return {
            "status": "degraded",
            "api_version": "1.0.0",
            "backend": {
                "status": "unreachable",
                "error": str(e)
            },
            "gpu": gpu_info,
            "timeouts": get_all_timeouts() if TIMEOUT_MANAGER_AVAILABLE else {"default": DEFAULT_TIMEOUT}
        }

# Performance metrics endpoint
@app.get("/api/metrics")
async def get_metrics(current_user: Optional[str] = Depends(get_current_user)):
    """Get performance metrics from the T4 GPU backend"""
    try:
        # Optional authentication check
        if ENVIRONMENT == "production" and not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get appropriate timeout for metrics endpoint
        timeout_value = get_timeout("metrics") if TIMEOUT_MANAGER_AVAILABLE else DEFAULT_TIMEOUT
        logger.debug(f"Using timeout of {timeout_value}s for metrics endpoint")
        
        # Forward request to backend
        response = requests.get(
            f"{T4_GPU_BACKEND_URL}/api/metrics",
            timeout=timeout_value
        )
        
        # Handle errors
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        # Return response
        return response.json()
    except requests.RequestException as e:
        raise handle_backend_exception(e, "metrics")

# GPU information endpoint
@app.get("/api/gpu_info")
async def gpu_info(current_user: Optional[str] = Depends(get_current_user)):
    """Get T4 GPU information"""
    # Optional authentication check in production
    if ENVIRONMENT == "production" and not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if T4_OPTIMIZER_AVAILABLE:
        return get_t4_gpu_info()
    else:
        return {"error": "T4 GPU optimizer not available"}

# Root route for API information
@app.get("/api")
async def api_info():
    """Get API information"""
    return {
        "name": "SAP HANA Cloud Generative AI Toolkit API",
        "version": "1.0.0",
        "description": "API for SAP HANA Cloud Generative AI Toolkit with T4 GPU Acceleration",
        "endpoints": {
            "auth": "/api/auth/token",
            "embeddings": "/api/embeddings",
            "search": "/api/vectorstore/search",
            "mmr_search": "/api/vectorstore/mmr_search",
            "health": "/api/health",
            "metrics": "/api/metrics",
            "gpu_info": "/api/gpu_info"
        },
        "status": "active"
    }

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint providing basic information about the API"""
    return {
        "message": "SAP HANA Cloud Generative AI Toolkit API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "api_info": "/api",
        "backend_url": T4_GPU_BACKEND_URL,
        "environment": ENVIRONMENT
    }

# Function for development/testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vercel_integration:app", host="0.0.0.0", port=8000, reload=True)