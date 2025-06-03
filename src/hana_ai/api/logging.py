"""
Enhanced logging configuration for production environments.

This module provides structured logging with correlation IDs,
support for various log formats (JSON/text), and integration 
with common enterprise logging systems.
"""
import json
import logging
import sys
import time
import uuid
from typing import Dict, Any, Optional

from contextvars import ContextVar
from fastapi import Request

from .config import settings

# Context variable to track request IDs across asynchronous code
request_id_contextvar: ContextVar[str] = ContextVar('request_id', default='')

class RequestContextFilter(logging.Filter):
    """
    Filter that adds request context information to log records.
    """
    def filter(self, record):
        # Add request_id to all log records
        record.request_id = request_id_contextvar.get()
        return True

class JsonFormatter(logging.Formatter):
    """
    Enhanced JSON formatter for structured logging with better enterprise integration.
    
    Features:
    - ISO 8601 timestamps with UTC timezone
    - Consistent field naming for log aggregation systems
    - Standardized exception formatting
    - Support for nested exception chains
    - Automatic inclusion of hostname and process info
    - SAP BTP specific fields
    """
    def __init__(self, *args, **kwargs):
        self.timestamp_format = kwargs.pop('timestamp_format', '%Y-%m-%dT%H:%M:%S.%fZ')
        self.hostname = socket.gethostname()
        self.app_name = os.environ.get('APP_NAME', 'hana-ai-toolkit')
        self.app_space = os.environ.get('APP_SPACE', '')
        self.app_org = os.environ.get('APP_ORG', '')
        self.environment = os.environ.get('ENVIRONMENT', 'production' if not settings.DEVELOPMENT_MODE else 'development')
        super().__init__(*args, **kwargs)

    def format(self, record):
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Basic log structure following ELK common schema
        log_data = {
            '@timestamp': timestamp,
            'log': {
                'level': record.levelname,
                'logger': record.name
            },
            'message': record.getMessage(),
            'service': {
                'name': self.app_name,
                'environment': self.environment,
                'type': 'api'
            },
            'host': {
                'hostname': self.hostname
            },
            'process': {
                'pid': record.process,
                'thread_name': record.threadName
            },
            'trace': {
                'id': getattr(record, 'request_id', '')
            },
            'http': {
                'request': {
                    'method': getattr(record, 'method', ''),
                    'url': {
                        'path': getattr(record, 'path', '')
                    }
                },
                'response': {
                    'status_code': getattr(record, 'status_code', 0)
                },
                'client': {
                    'ip': getattr(record, 'remote_ip', '')
                }
            },
            'event': {
                'duration': getattr(record, 'duration_ms', None),
                'dataset': 'hana_ai.api',
                'module': record.module,
                'created': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            },
            'file': {
                'path': record.pathname,
                'line': record.lineno
            },
            'sap': {
                'space': self.app_space,
                'org': self.app_org,
                'component': 'hana-ai-toolkit'
            }
        }

        # Include full exception info if available
        if record.exc_info:
            exception_data = self._format_exception(record.exc_info)
            log_data['error'] = exception_data

        # Include any custom fields
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName',
                          'request_id', 'path', 'method', 'remote_ip', 'duration_ms', 'status_code'):
                if not key.startswith('_'):
                    # Add custom fields to a dedicated section
                    log_data.setdefault('labels', {})[key] = value

        return json.dumps(log_data, default=str)
    
    def _format_exception(self, exc_info):
        """Format exception info in a structured way"""
        exc_type, exc_value, exc_traceback = exc_info
        frames = traceback.extract_tb(exc_traceback)
        
        # Create structured exception data
        return {
            'type': exc_type.__name__,
            'message': str(exc_value),
            'stack_trace': ''.join(traceback.format_exception(*exc_info)),
            'causes': self._get_exception_chain(exc_value) if hasattr(exc_value, '__cause__') and exc_value.__cause__ else [],
            'frames': [
                {
                    'filename': frame.filename,
                    'lineno': frame.lineno,
                    'function': frame.name,
                    'code': frame.line
                } for frame in frames
            ]
        }
    
    def _get_exception_chain(self, exception):
        """Get the chain of exception causes"""
        causes = []
        current = exception.__cause__
        
        while current:
            causes.append({
                'type': type(current).__name__,
                'message': str(current)
            })
            current = current.__cause__
            
        return causes

def setup_logging():
    """
    Configure application-wide logging based on settings.
    """
    log_handlers = []
    
    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter in production, human-readable in development
    if settings.DEVELOPMENT_MODE:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
        )
    else:
        formatter = JsonFormatter()
    
    stdout_handler.setFormatter(formatter)
    log_handlers.append(stdout_handler)
    
    # File logging if configured
    if hasattr(settings, 'LOG_FILE') and settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setFormatter(formatter)
        log_handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Add our handlers
    for handler in log_handlers:
        root_logger.addHandler(handler)
    
    # Add request context filter to all handlers
    context_filter = RequestContextFilter()
    for handler in root_logger.handlers:
        handler.addFilter(context_filter)
    
    # Log startup
    logging.info(f"Logging initialized at level {settings.LOG_LEVEL}")

def get_request_id(request: Optional[Request] = None) -> str:
    """
    Get the current request ID or generate a new one.
    
    Parameters
    ----------
    request : Request, optional
        The FastAPI request object
        
    Returns
    -------
    str
        The request ID
    """
    # Try to get from context first
    current_id = request_id_contextvar.get()
    if current_id:
        return current_id
    
    # Try to get from request headers
    if request and 'X-Request-ID' in request.headers:
        new_id = request.headers['X-Request-ID']
    else:
        # Generate new ID
        new_id = str(uuid.uuid4())
    
    # Store in context
    request_id_contextvar.set(new_id)
    return new_id

def log_request_details(request: Request, start_time: float):
    """
    Log detailed information about a request.
    
    Parameters
    ----------
    request : Request
        The FastAPI request object
    start_time : float
        Request start time (from time.time())
    """
    # Calculate request duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Get client IP, handling proxy forwarding
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    remote_ip = forwarded_for.split(",")[0] if forwarded_for else client_host
    
    # Prepare log record with request details
    logger = logging.getLogger("api.request")
    extra = {
        'request_id': get_request_id(request),
        'method': request.method,
        'path': request.url.path,
        'remote_ip': remote_ip,
        'duration_ms': round(duration_ms, 2)
    }
    
    # Log at appropriate level based on response status
    status_code = getattr(request.state, 'status_code', 200)
    if status_code >= 500:
        logger.error(f"{request.method} {request.url.path} - {status_code} - {duration_ms:.2f}ms", extra=extra)
    elif status_code >= 400:
        logger.warning(f"{request.method} {request.url.path} - {status_code} - {duration_ms:.2f}ms", extra=extra)
    else:
        logger.info(f"{request.method} {request.url.path} - {status_code} - {duration_ms:.2f}ms", extra=extra)