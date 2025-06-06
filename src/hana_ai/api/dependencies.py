"""
Dependencies for the FastAPI application.

This module provides dependency functions for the FastAPI application,
including database connections, authentication, and GPU acceleration.
"""

import os
import logging
from typing import Optional, Dict, Any, Generator
from hana_ml.dataframe import ConnectionContext
from .config import settings
from .security import api_key_auth

# Configure logging
logger = logging.getLogger(__name__)

# Dictionary to cache connection contexts
CONNECTION_CACHE = {}

def get_connection_context(
    connection_string: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    userkey: Optional[str] = None,
    schema: Optional[str] = None,
) -> ConnectionContext:
    """
    Get a ConnectionContext for SAP HANA.
    
    Args:
        connection_string: Full connection string (overrides other parameters if provided)
        host: HANA host address
        port: HANA port
        user: HANA username
        password: HANA password
        userkey: HANA secure user store key
        schema: HANA schema
        
    Returns:
        ConnectionContext: Connection to HANA
        
    Raises:
        ValueError: If connection parameters are invalid
    """
    # Generate a cache key from the connection parameters
    if connection_string:
        cache_key = connection_string
    elif userkey:
        cache_key = f"userkey:{userkey}"
    elif host and port and (user and password):
        cache_key = f"{host}:{port}:{user}:{schema or ''}"
    else:
        # Use default connection parameters from settings
        host = host or settings.HANA_HOST
        port = port or settings.HANA_PORT
        user = user or settings.HANA_USER
        password = password or settings.HANA_PASSWORD
        userkey = userkey or settings.HANA_USERKEY
        schema = schema or settings.HANA_SCHEMA
        
        if userkey:
            cache_key = f"userkey:{userkey}"
        elif host and port and (user and password):
            cache_key = f"{host}:{port}:{user}:{schema or ''}"
        else:
            raise ValueError(
                "Invalid connection parameters. Please provide either a connection_string, "
                "userkey, or host/port/user/password combination."
            )
    
    # Check if the connection context is already in the cache
    if cache_key in CONNECTION_CACHE:
        try:
            # Test the connection by running a simple query
            CONNECTION_CACHE[cache_key].sql("SELECT 1 FROM DUMMY").collect()
            return CONNECTION_CACHE[cache_key]
        except Exception as e:
            # Connection is stale, remove it from the cache
            logger.warning(f"Stale connection detected, recreating: {str(e)}")
            del CONNECTION_CACHE[cache_key]
    
    # Create a new connection context
    try:
        if connection_string:
            conn = ConnectionContext(connection_string)
        elif userkey:
            conn = ConnectionContext(userkey=userkey)
        else:
            conn = ConnectionContext(
                address=host,
                port=int(port) if port else None,
                user=user,
                password=password,
                schema=schema,
                encrypt="true",
                sslValidateCertificate="false"
            )
        
        # Test the connection
        conn.sql("SELECT 1 FROM DUMMY").collect()
        
        # Add to cache
        CONNECTION_CACHE[cache_key] = conn
        
        return conn
    except Exception as e:
        logger.error(f"Failed to create HANA connection: {str(e)}")
        raise ValueError(f"Failed to connect to HANA: {str(e)}")

def get_gpu_provider() -> str:
    """
    Get the configured GPU provider for the application.
    
    Returns:
        str: The GPU provider name ('local', 'together_ai', 'none')
    """
    # Check if GPU acceleration is enabled
    if not settings.ENABLE_GPU_ACCELERATION:
        return "none"
    
    # Check if Together.ai integration is enabled
    if settings.ENABLE_TOGETHER_AI and settings.TOGETHER_API_KEY:
        return "together_ai"
    
    # Check if local GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            return "local"
    except ImportError:
        pass
    
    return "none"

def get_together_ai_client():
    """
    Get the Together.ai client as a FastAPI dependency.
    
    Returns:
        TogetherAIClient: The Together.ai client
    """
    # Import here to avoid circular imports
    from .together_ai import get_together_ai_client as get_client
    
    return get_client()

def get_llm_client():
    """
    Get an LLM client for generating text.
    
    This function provides a simplified abstraction over different LLM backends.
    For demo purposes, it will work without requiring any API keys.
    
    Returns:
        LLMClient: A client for interacting with language models
    """
    # Determine which LLM backend to use
    # First try to use environment variables
    api_key = os.environ.get("OPENAI_API_KEY", None)
    
    # If no API key, create a mock client for demo purposes
    if not api_key:
        logger.info("No API key found, creating mock LLM client for demos")
        # Simple mock client that doesn't require external APIs
        class MockLLMClient:
            def generate(self, prompt):
                return f"Generated response for: {prompt}"
                
            def embed(self, text):
                # Return mock embeddings
                import numpy as np
                return np.random.rand(1536)
        
        return MockLLMClient()
    
    # Otherwise, try to initialize a real client
    try:
        # Try OpenAI client first
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # Wrap in a standardized interface
        class OpenAIClient:
            def __init__(self, client):
                self.client = client
                
            def generate(self, prompt):
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
                
            def embed(self, text):
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
        
        return OpenAIClient(client)
        
    except (ImportError, Exception) as e:
        # Fall back to mock client if something goes wrong
        logger.warning(f"Failed to initialize LLM client: {str(e)}. Using mock client.")
        class MockLLMClient:
            def generate(self, prompt):
                return f"Generated response for: {prompt}"
                
            def embed(self, text):
                # Return mock embeddings
                import numpy as np
                return np.random.rand(1536)
                
        return MockLLMClient()

# Function to reset connections (useful for testing)
def reset_connections():
    """Reset all cached connections."""
    global CONNECTION_CACHE
    
    # Close all connections
    for conn in CONNECTION_CACHE.values():
        try:
            conn.close()
        except:
            pass
    
    # Clear the cache
    CONNECTION_CACHE = {}