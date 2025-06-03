"""
Tests for the main FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

def test_app_initialization():
    """Test that the app initializes correctly."""
    from hana_ai.api.app import app
    
    # Check that the app has expected attributes
    assert hasattr(app, "title")
    assert app.title == "SAP HANA AI Toolkit API"
    
    # Check routers are registered
    route_paths = [route.path for route in app.routes]
    assert "/api/v1/agents/conversation" in route_paths
    assert "/api/v1/dataframes/query" in route_paths
    assert "/api/v1/tools/list" in route_paths
    assert "/api/v1/vectorstore/query" in route_paths

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()

def test_invalid_endpoint(test_client):
    """Test accessing an invalid endpoint."""
    response = test_client.get("/non_existent_path")
    assert response.status_code == 404

def test_method_not_allowed(test_client):
    """Test using wrong HTTP method."""
    # POST to health check endpoint which only accepts GET
    response = test_client.post("/")
    assert response.status_code == 405  # Method Not Allowed

def test_cors_headers(test_client):
    """Test CORS headers in response."""
    # Options request to check CORS preflight
    response = test_client.options(
        "/", 
        headers={
            "Origin": "http://testorigin.com",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers