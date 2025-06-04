"""
Debug and Logging Handler for SAP HANA Cloud Generative AI Toolkit

This module provides comprehensive debugging and logging utilities for API requests
and responses, error handling, and detailed diagnostics for development environments.
"""

import time
import json
import logging
import traceback
from typing import Any, Dict, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def log_request(request: Request) -> None:
    """
    Log details about the incoming request
    
    Args:
        request: The FastAPI request object
    """
    body = await request.body()
    try:
        # Try to parse as JSON
        if body:
            body_str = body.decode('utf-8')
            try:
                body_json = json.loads(body_str)
                # Format for better readability
                body_log = json.dumps(body_json, indent=2)
            except json.JSONDecodeError:
                body_log = body_str
        else:
            body_log = "(empty body)"
    except Exception as e:
        body_log = f"(Error parsing body: {str(e)})"
    
    # Get request ID if available
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log request details
    logger.info(f"REQUEST [{request_id}] {request.method} {request.url.path}")
    logger.info(f"REQUEST HEADERS [{request_id}]: {dict(request.headers)}")
    logger.info(f"REQUEST BODY [{request_id}]: {body_log}")

async def log_response(response: Response, body: bytes) -> None:
    """
    Log details about the outgoing response
    
    Args:
        response: The FastAPI response object
        body: The response body content
    """
    try:
        # Try to parse as JSON
        if body:
            body_str = body.decode('utf-8')
            try:
                body_json = json.loads(body_str)
                # Format for better readability but omit very large fields
                if isinstance(body_json, dict):
                    # Remove embeddings from logging as they're large
                    if 'embeddings' in body_json:
                        body_json['embeddings'] = f"[... {len(body_json['embeddings'])} embeddings omitted ...]"
                    body_log = json.dumps(body_json, indent=2)
                else:
                    body_log = json.dumps(body_json, indent=2)
            except json.JSONDecodeError:
                body_log = body_str
        else:
            body_log = "(empty body)"
    except Exception as e:
        body_log = f"(Error parsing body: {str(e)})"
    
    # Get request ID if available from response headers
    request_id = response.headers.get("X-Request-ID", "unknown")
    
    # Log response details
    logger.info(f"RESPONSE [{request_id}] Status: {response.status_code}")
    logger.info(f"RESPONSE HEADERS [{request_id}]: {dict(response.headers)}")
    logger.info(f"RESPONSE BODY [{request_id}]: {body_log}")

async def debug_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle exceptions with detailed error responses in development environments
    
    Args:
        request: The FastAPI request object
        exc: The exception that was raised
        
    Returns:
        A JSONResponse with detailed error information
    """
    # Get request ID if available
    request_id = getattr(request.state, "request_id", f"err_{int(time.time() * 1000)}")
    
    # Log the exception with traceback
    logger.error(f"Exception handling request [{request_id}]: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Get status code from exception if it's an HTTPException
    status_code = getattr(exc, "status_code", 500)
    
    # Prepare error response
    error_response = {
        "error": str(exc),
        "request_id": request_id,
        "timestamp": time.time(),
        "path": str(request.url),
        "method": request.method,
    }
    
    # Include traceback in development environments
    import os
    if os.getenv("ENVIRONMENT", "development") != "production":
        error_response["traceback"] = traceback.format_exc().split("\n")
        
        # Include request details for easier debugging
        try:
            body = await request.body()
            if body:
                error_response["request_body"] = json.loads(body)
        except:
            # If the body can't be parsed as JSON, include it as a string
            try:
                error_response["request_body"] = body.decode('utf-8')
            except:
                error_response["request_body"] = "(unparseable)"
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )