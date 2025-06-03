"""
Integration tests for the API endpoints.

These tests verify that the different API components work together correctly.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

def test_end_to_end_forecast_workflow(test_client, mock_connection_context, mock_tools):
    """Test a complete time series forecasting workflow across multiple endpoints."""
    # Step 1: List tables
    response = test_client.get("/api/v1/dataframes/tables")
    assert response.status_code == 200
    
    # Step 2: Examine data with smart dataframe
    with patch("hana_ai.smart_dataframe.SmartDataFrame") as mock_sdf_class:
        # Configure the mock
        mock_sdf = mock_sdf_class.return_value
        mock_sdf.ask.return_value = "The data shows sales by date"
        
        # Call the endpoint
        response = test_client.post(
            "/api/v1/dataframes/smart/ask",
            json={
                "table_name": "SALES_TABLE",
                "question": "What does this data show?",
                "is_sql_query": False,
                "transform": False
            }
        )
        assert response.status_code == 200
        assert "The data shows sales by date" in response.json()["result"]
    
    # Step 3: Check time series properties with tools
    with patch("hana_ai.tools.hana_ml_tools.ts_check_tools.TimeSeriesCheck") as mock_tool_class:
        # Configure the mock
        mock_tool = MagicMock()
        mock_tool.name = "ts_check"
        mock_tool.run.return_value = "Time series has seasonality"
        mock_tool_class.return_value = mock_tool
        
        # Create toolkit mock
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        
        with patch("hana_ai.tools.toolkit.HANAMLToolkit", return_value=mock_toolkit):
            # Call the endpoint
            response = test_client.post(
                "/api/v1/tools/execute",
                json={
                    "tool_name": "ts_check",
                    "parameters": {
                        "table_name": "SALES_TABLE",
                        "key": "DATE",
                        "endog": "SALES"
                    }
                }
            )
            assert response.status_code == 200
            assert response.json()["result"] == "Time series has seasonality"
    
    # Step 4: Train forecast model
    with patch("hana_ai.tools.hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesFitAndSave") as mock_tool_class:
        # Configure the mock
        mock_tool = MagicMock()
        mock_tool.name = "automatic_timeseries_fit_and_save"
        mock_tool.run.return_value = {
            "trained_table": "SALES_TABLE",
            "model_storage_name": "sales_forecast",
            "model_storage_version": 1
        }
        mock_tool_class.return_value = mock_tool
        
        # Create toolkit mock
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        
        with patch("hana_ai.tools.toolkit.HANAMLToolkit", return_value=mock_toolkit):
            # Call the forecast endpoint
            response = test_client.post(
                "/api/v1/tools/forecast",
                json={
                    "table_name": "SALES_TABLE",
                    "key_column": "DATE",
                    "value_column": "SALES",
                    "horizon": 12,
                    "model_name": "sales_forecast",
                    "model_type": "automatic_timeseries"
                }
            )
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            assert "model_details" in response.json()

def test_agent_with_tools_integration(test_client, mock_connection_context):
    """Test integration between agents and tools."""
    # Create mocks for the agent and tools
    with patch("hana_ai.agents.hanaml_agent_with_memory.HANAMLAgentWithMemory") as mock_agent_class, \
         patch("hana_ai.tools.toolkit.HANAMLToolkit") as mock_toolkit_class:
        # Configure the agent mock
        mock_agent = mock_agent_class.return_value
        mock_agent.run.return_value = "Used forecast tool to analyze data"
        
        # Configure the toolkit mock
        mock_toolkit = mock_toolkit_class.return_value
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.run.return_value = "Tool result"
        mock_toolkit.get_tools.return_value = [mock_tool]
        
        # Add the agent to the sessions
        with patch("hana_ai.api.routers.agents.AGENT_SESSIONS", {"test_session": mock_agent}):
            # Call the agent endpoint
            response = test_client.post(
                "/api/v1/agents/conversation",
                json={
                    "message": "Run a forecast on sales data",
                    "session_id": "test_session",
                    "return_intermediate_steps": False,
                    "verbose": False
                }
            )
            
            # Check response
            assert response.status_code == 200
            assert "Used forecast tool" in response.json()["response"]
            
            # Verify agent was called
            mock_agent.run.assert_called_once()
            
            # Verify toolkit was initialized properly
            mock_toolkit_class.assert_called_once_with(
                connection_context=mock_connection_context, 
                used_tools="all"
            )

def test_vector_store_with_embedding_integration(test_client, mock_connection_context):
    """Test integration between vector store and embedding generation."""
    # Create mocks for vector store and embeddings
    with patch("hana_ai.vectorstore.hana_vector_engine.HANAMLinVectorEngine") as mock_vs_class, \
         patch("hana_ai.vectorstore.embedding_service.HANAVectorEmbeddings") as mock_emb_class:
        # Configure the vector store mock
        mock_vs = mock_vs_class.return_value
        mock_vs.query.return_value = "Vector store result"
        mock_vs.current_query_distance = 0.85
        
        # Configure the embedding model mock
        mock_emb = mock_emb_class.return_value
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(3)]
        
        # First generate embeddings
        response = test_client.post(
            "/api/v1/vectorstore/embed",
            json=["Test document"],
            params={"model_type": "hana"}
        )
        assert response.status_code == 200
        assert "embeddings" in response.json()
        
        # Then query the vector store
        response = test_client.post(
            "/api/v1/vectorstore/query",
            json={
                "query": "Test query",
                "top_k": 1,
                "collection_name": "test_collection"
            }
        )
        assert response.status_code == 200
        assert "Vector store result" in str(response.json()["results"])