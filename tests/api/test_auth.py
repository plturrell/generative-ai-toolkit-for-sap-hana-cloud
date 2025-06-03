"""
Tests for the API authentication.
"""
import pytest
from fastapi.testclient import TestClient

def test_api_key_auth_required(test_client):
    """Test API key authentication when required."""
    # Override auth required setting for this test
    from hana_ai.api.config import settings
    original_setting = settings.AUTH_REQUIRED
    settings.AUTH_REQUIRED = True
    settings.API_KEYS = ["test-api-key"]
    
    try:
        # Request without API key should fail
        response = test_client.get("/")
        assert response.status_code == 401
        assert "API key is missing" in response.json()["detail"]
        
        # Request with invalid API key should fail
        response = test_client.get(
            "/", 
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
        
        # Request with valid API key should succeed
        response = test_client.get(
            "/", 
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 200
    finally:
        # Restore original setting
        settings.AUTH_REQUIRED = original_setting

def test_api_key_auth_disabled(test_client):
    """Test behavior when authentication is disabled."""
    # Override auth required setting for this test
    from hana_ai.api.config import settings
    original_setting = settings.AUTH_REQUIRED
    settings.AUTH_REQUIRED = False
    
    try:
        # Request without API key should succeed
        response = test_client.get("/")
        assert response.status_code == 200
    finally:
        # Restore original setting
        settings.AUTH_REQUIRED = original_setting

def test_api_key_dependency(test_client):
    """Test the get_api_key dependency directly."""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from hana_ai.api.auth import get_api_key
    from hana_ai.api.config import settings
    
    # Create a temporary app for testing the dependency
    app = FastAPI()
    
    @app.get("/test-auth")
    async def test_auth(api_key: str = Depends(get_api_key)):
        return {"api_key": api_key}
    
    # Create client for this app
    client = TestClient(app)
    
    # Override auth settings
    original_setting = settings.AUTH_REQUIRED
    original_keys = settings.API_KEYS
    settings.AUTH_REQUIRED = True
    settings.API_KEYS = ["valid-key"]
    
    try:
        # Test with valid key
        response = client.get(
            "/test-auth", 
            headers={"X-API-Key": "valid-key"}
        )
        assert response.status_code == 200
        assert response.json()["api_key"] == "valid-key"
        
        # Test with invalid key
        response = client.get(
            "/test-auth", 
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
    finally:
        # Restore original settings
        settings.AUTH_REQUIRED = original_setting
        settings.API_KEYS = original_keys