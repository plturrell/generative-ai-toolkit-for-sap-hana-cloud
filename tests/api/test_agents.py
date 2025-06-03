"""
Tests for the agents API endpoints.
"""
import pytest
import json
from fastapi.testclient import TestClient

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_conversation(test_client, mock_agent, mock_tools, mock_connection_context):
    """Test the process_conversation endpoint."""
    # Prepare test data
    request_data = {
        "message": "What tables are available?",
        "session_id": "test_session",
        "return_intermediate_steps": False,
        "verbose": False
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/agents/conversation", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["conversation_id"] == "test_session"
    
    # Verify the agent was called
    mock_agent.run.assert_called_once_with("What tables are available?")

def test_process_conversation_with_steps(test_client, mock_llm, mock_tools, mock_connection_context):
    """Test the process_conversation endpoint with intermediate steps."""
    # Mock the stateless_call function
    from unittest.mock import patch
    
    with patch("hana_ai.agents.hanaml_agent_with_memory.stateless_call") as mock_stateless_call:
        # Configure the mock
        mock_stateless_call.return_value = {
            "output": "Test response",
            "intermediate_steps": json.dumps([["thought", "action"]])
        }
        
        # Prepare test data
        request_data = {
            "message": "What tables are available?",
            "session_id": "new_session",
            "return_intermediate_steps": True,
            "verbose": False
        }
        
        # Call the endpoint
        response = test_client.post("/api/v1/agents/conversation", json=request_data)
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["response"] == "Test response"
        assert "intermediate_steps" in response.json()
        
        # Verify stateless_call was called
        mock_stateless_call.assert_called_once()

def test_sql_agent(test_client, mock_llm, mock_connection_context):
    """Test the sql_agent endpoint."""
    from unittest.mock import patch
    
    with patch("hana_ai.agents.hana_sql_agent.create_hana_sql_agent") as mock_create_agent:
        # Configure the mock
        mock_agent = mock_create_agent.return_value
        mock_agent.invoke.return_value = "SQL query result"
        
        # Call the endpoint
        response = test_client.post(
            "/api/v1/agents/sql",
            params={"query": "Show me all sales data"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert "result" in response.json()
        assert response.json()["result"] == "SQL query result"
        assert "execution_time" in response.json()
        
        # Verify agent was created and called
        mock_create_agent.assert_called_once()
        mock_agent.invoke.assert_called_once_with("Show me all sales data")

def test_error_handling(test_client, mock_connection_context):
    """Test error handling in the API."""
    from unittest.mock import patch
    
    with patch("hana_ai.agents.hanaml_agent_with_memory.HANAMLAgentWithMemory") as mock_agent_class:
        # Configure the mock to raise an exception
        mock_agent = mock_agent_class.return_value
        mock_agent.run.side_effect = ValueError("Test error")
        
        # Prepare test data
        request_data = {
            "message": "What tables are available?",
            "session_id": "error_session",
            "return_intermediate_steps": False,
            "verbose": False
        }
        
        # Call the endpoint
        response = test_client.post("/api/v1/agents/conversation", json=request_data)
        
        # Check the response
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Test error" in response.json()["detail"]