"""
Tests for the dataframes API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

def test_query_database(test_client, mock_connection_context):
    """Test the query_database endpoint."""
    # Prepare test data
    request_data = {
        "query": "SELECT * FROM TEST_TABLE",
        "limit": 100,
        "offset": 0
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/dataframes/query", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "columns" in response.json()
    assert "data" in response.json()
    assert "row_count" in response.json()
    assert "query_time" in response.json()
    
    # Verify SQL was executed
    mock_connection_context.sql.assert_called_once()

def test_smart_dataframe_ask(test_client, mock_connection_context, mock_llm, mock_smart_dataframe):
    """Test the smart_dataframe_ask endpoint."""
    # Prepare test data
    request_data = {
        "table_name": "TEST_TABLE",
        "question": "What is the average value?",
        "is_sql_query": False,
        "transform": False
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/dataframes/smart/ask", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "type" in response.json()
    assert response.json()["type"] == "ask"
    assert "result" in response.json()
    assert "query_time" in response.json()
    
    # Verify SmartDataFrame was called
    mock_smart_dataframe.ask.assert_called_once_with("What is the average value?")

def test_smart_dataframe_transform(test_client, mock_connection_context, mock_llm, mock_smart_dataframe):
    """Test the smart_dataframe_ask endpoint with transform=True."""
    # Prepare test data
    request_data = {
        "table_name": "SELECT * FROM TEST_TABLE",
        "question": "Filter where value > 100",
        "is_sql_query": True,
        "transform": True
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/dataframes/smart/ask", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "type" in response.json()
    assert response.json()["type"] == "transform"
    assert "sql" in response.json()
    assert "columns" in response.json()
    assert "data" in response.json()
    
    # Verify SmartDataFrame.transform was called
    mock_smart_dataframe.transform.assert_called_once_with("Filter where value > 100")

def test_list_tables(test_client, mock_connection_context):
    """Test the list_tables endpoint."""
    # Mock specific behavior for this test
    mock_connection_context.sql.return_value.collect.return_value = [
        {"TABLE_NAME": "TABLE1"}, 
        {"TABLE_NAME": "TABLE2"}
    ]
    
    # Call the endpoint without schema
    response = test_client.get("/api/v1/dataframes/tables")
    
    # Check the response
    assert response.status_code == 200
    assert "tables" in response.json()
    assert len(response.json()["tables"]) == 2
    assert "count" in response.json()
    assert response.json()["count"] == 2
    
    # Call with schema parameter
    response = test_client.get("/api/v1/dataframes/tables?schema=TEST")
    
    # Check that SQL query includes schema filter
    assert "TEST" in str(mock_connection_context.sql.call_args)
    
    # Verify connection context methods were called
    mock_connection_context.sql.assert_called()
    mock_connection_context.get_current_schema.assert_called()

def test_error_handling(test_client, mock_connection_context):
    """Test error handling in dataframe endpoints."""
    # Mock connection to raise an exception
    mock_connection_context.sql.side_effect = Exception("Database error")
    
    # Prepare test data
    request_data = {
        "query": "SELECT * FROM TEST_TABLE",
        "limit": 100,
        "offset": 0
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/dataframes/query", json=request_data)
    
    # Check the error response
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Database error" in response.json()["detail"]