"""
Authentication and authorization handling for the API.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from .config import settings

# Define API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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