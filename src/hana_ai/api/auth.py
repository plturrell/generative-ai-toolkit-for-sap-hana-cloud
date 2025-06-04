"""
Authentication and authorization handling for the API.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from .config import settings

# Define API key security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Validate the API key from the request header.
    
    Parameters
    ----------
    api_key : str
        The API key provided in the X-API-Key header
        
    Returns
    -------
    str
        The validated API key
        
    Raises
    ------
    HTTPException
        If the API key is missing or invalid
    """
    if not settings.AUTH_REQUIRED:
        return "auth-disabled"
        
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    if api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return api_key

async def get_admin_api_key(api_key: str = Depends(admin_api_key_header)):
    """
    Validate the admin API key from the request header.
    
    Parameters
    ----------
    api_key : str
        The admin API key provided in the X-Admin-API-Key header
        
    Returns
    -------
    str
        The validated admin API key
        
    Raises
    ------
    HTTPException
        If the admin API key is missing or invalid
    """
    if not settings.AUTH_REQUIRED:
        return "auth-disabled"
        
    if api_key is None:
        # Try to use the regular API key header as fallback
        regular_api_key = await api_key_header()
        if regular_api_key and regular_api_key in settings.ADMIN_API_KEYS:
            return regular_api_key
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key is missing",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    if api_key not in settings.ADMIN_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return api_key