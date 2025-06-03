"""
Tests for the tools API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

def test_list_tools(test_client, mock_tools, mock_connection_context):
    """Test the list_tools endpoint."""
    # Call the endpoint
    response = test_client.get("/api/v1/tools/list")
    
    # Check the response
    assert response.status_code == 200
    assert "tools" in response.json()
    assert "count" in response.json()
    assert response.json()["count"] == 1
    assert response.json()["tools"][0]["name"] == "test_tool"

def test_execute_tool(test_client, mock_tools, mock_connection_context):
    """Test the execute_tool endpoint."""
    # Prepare test data
    request_data = {
        "tool_name": "test_tool",
        "parameters": {
            "param1": "value1",
            "param2": 42
        }
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/tools/execute", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] == {"result": "success"}
    assert "execution_time" in response.json()
    assert "tool" in response.json()
    assert response.json()["tool"] == "test_tool"
    
    # Verify tool was called with parameters
    mock_tools[0].run.assert_called_once_with(request_data["parameters"])

def test_execute_tool_not_found(test_client, mock_tools, mock_connection_context):
    """Test the execute_tool endpoint with nonexistent tool."""
    # Prepare test data
    request_data = {
        "tool_name": "nonexistent_tool",
        "parameters": {}
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/tools/execute", json=request_data)
    
    # Check the error response
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "not found" in response.json()["detail"]

def test_forecast_timeseries(test_client, mock_connection_context):
    """Test the forecast_timeseries endpoint."""
    from unittest.mock import patch, MagicMock
    
    # Create specific mocks for this test
    mock_ts_check = MagicMock()
    mock_ts_check.run.return_value = "Time series check results"
    
    mock_fit_tool = MagicMock()
    mock_fit_tool.run.return_value = {
        "model_storage_name": "test_model",
        "model_storage_version": 1
    }
    
    # Mock the toolkit to return our mock tools
    with patch("hana_ai.tools.toolkit.HANAMLToolkit") as mock_toolkit_class:
        mock_toolkit = mock_toolkit_class.return_value
        mock_toolkit.get_tools.return_value = [mock_ts_check, mock_fit_tool]
        
        # Since we're using 'next(t for t in tools...)' in the endpoint,
        # we need to make sure the first mock is the ts_check tool and the second is the fit tool
        mock_ts_check.name = "ts_check"
        mock_fit_tool.name = "automatic_timeseries_fit_and_save"
        
        # Call the endpoint
        response = test_client.post(
            "/api/v1/tools/forecast",
            json={
                "table_name": "SALES_DATA",
                "key_column": "DATE",
                "value_column": "SALES",
                "horizon": 12,
                "model_name": "sales_forecast"
            }
        )
        
        # Check the response
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "success"
        assert "model_details" in response.json()
        assert "time_series_properties" in response.json()
        assert "execution_time" in response.json()
        
        # Verify tools were called with expected parameters
        mock_ts_check.run.assert_called_once_with({
            "table_name": "SALES_DATA",
            "key": "DATE",
            "endog": "SALES"
        })
        
        mock_fit_tool.run.assert_called_once_with({
            "fit_table": "SALES_DATA",
            "name": "sales_forecast",
            "key": "DATE",
            "endog": "SALES"
        })

def test_error_handling(test_client, mock_connection_context):
    """Test error handling in tools endpoints."""
    from unittest.mock import patch
    
    with patch("hana_ai.tools.toolkit.HANAMLToolkit") as mock_toolkit_class:
        # Configure the mock to raise an exception
        mock_toolkit_class.side_effect = Exception("Toolkit error")
        
        # Call the endpoint
        response = test_client.get("/api/v1/tools/list")
        
        # Check the error response
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Toolkit error" in response.json()["detail"]