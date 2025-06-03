"""
Middleware for the HANA AI Toolkit API.

This module provides middleware components for:
- Request/response logging
- Security headers
- Rate limiting
- Metrics collection
- Error handling
"""
import time
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from .logging import log_request_details, get_request_id
from .config import settings
from .metrics import record_request_metric

class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging request details and timing.
    
    This middleware logs the start and end of each request,
    including duration, status code, and other metadata.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or get request ID
        request_id = get_request_id(request)
        
        # Record start time
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Store status code for logging
            request.state.status_code = response.status_code
            
            # Log request details
            log_request_details(request, start_time)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            # Log unhandled exceptions
            request.state.status_code = 500
            log_request_details(request, start_time)
            raise

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    
    This adds headers recommended by OWASP to enhance
    security of the API.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.
    
    Implements a token bucket algorithm to limit requests 
    per client based on API key.
    """
    def __init__(self, app, rate_limit_per_minute=100):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.tokens = {}  # Maps API keys to available tokens
        self.last_refill = {}  # Maps API keys to last token refill time
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks and in development mode
        if settings.DEVELOPMENT_MODE or request.url.path == "/":
            return await call_next(request)
        
        # Get API key from headers or use IP as fallback
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            client_host = request.client.host if request.client else "unknown"
            forwarded_for = request.headers.get("X-Forwarded-For", "")
            api_key = forwarded_for.split(",")[0] if forwarded_for else client_host
        
        # Initialize tokens if not exist
        if api_key not in self.tokens:
            self.tokens[api_key] = self.rate_limit_per_minute
            self.last_refill[api_key] = time.time()
        
        # Refill tokens if needed (token bucket algorithm)
        now = time.time()
        time_passed = now - self.last_refill[api_key]
        tokens_to_add = time_passed * (self.rate_limit_per_minute / 60.0)
        self.tokens[api_key] = min(self.rate_limit_per_minute, self.tokens[api_key] + tokens_to_add)
        self.last_refill[api_key] = now
        
        # Check if we have a token available
        if self.tokens[api_key] >= 1:
            # Consume a token
            self.tokens[api_key] -= 1
            
            # Process the request
            return await call_next(request)
        else:
            # Rate limit exceeded
            return Response(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content='{"detail":"Rate limit exceeded. Please try again later."}',
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    
    Records timing, status codes, and endpoint usage for monitoring.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record start time
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request_metric(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response