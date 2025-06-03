"""
Pytest fixtures for API tests.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from hana_ai.api.app import app
from hana_ai.api.config import settings

@pytest.fixture
def test_client():
    """
    Create a FastAPI TestClient instance.
    """
    # Override settings for testing
    settings.AUTH_REQUIRED = False
    
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_connection_context():
    """
    Create a mock HANA connection context.
    """
    mock_conn = MagicMock()
    
    # Mock SQL execution
    mock_result = MagicMock()
    mock_result.collect.return_value = [{"COL1": "value1"}, {"COL1": "value2"}]
    mock_result.columns = ["COL1"]
    mock_conn.sql.return_value = mock_result
    
    # Mock table access
    mock_conn.table.return_value = mock_result
    mock_conn.get_current_schema.return_value = "TEST_SCHEMA"
    mock_conn.has_table.return_value = True
    
    # Add to app state for dependency injection
    app.state.connection_pool = {1: mock_conn}
    
    with patch("hana_ai.api.dependencies.get_connection_context", return_value=mock_conn):
        yield mock_conn

@pytest.fixture
def mock_llm():
    """
    Create a mock language model.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "This is a test response from the mock LLM."
    
    with patch("hana_ai.api.dependencies.get_llm", return_value=mock_llm):
        yield mock_llm

@pytest.fixture
def mock_tools():
    """
    Create mock tools for testing.
    """
    # Create a mock tool
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.run.return_value = {"result": "success"}
    
    # Create a mock toolkit that returns our mock tool
    mock_toolkit = MagicMock()
    mock_toolkit.get_tools.return_value = [mock_tool]
    
    with patch("hana_ai.tools.toolkit.HANAMLToolkit", return_value=mock_toolkit):
        yield [mock_tool]

@pytest.fixture
def mock_agent():
    """
    Create a mock agent for testing.
    """
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Mock agent response"
    
    with patch("hana_ai.agents.hanaml_agent_with_memory.HANAMLAgentWithMemory", return_value=mock_agent), \
         patch("hana_ai.api.routers.agents.AGENT_SESSIONS", {"test_session": mock_agent}):
        yield mock_agent

@pytest.fixture
def mock_smart_dataframe():
    """
    Create a mock SmartDataFrame for testing.
    """
    mock_sdf = MagicMock()
    mock_sdf.ask.return_value = "Smart DataFrame response"
    mock_sdf.transform.return_value = MagicMock()
    mock_sdf.transform.return_value.head.return_value.collect.return_value = [{"COL1": "value1"}]
    mock_sdf.transform.return_value.columns = ["COL1"]
    mock_sdf.transform.return_value.select_statement = "SELECT * FROM TEST"
    
    with patch("hana_ai.smart_dataframe.SmartDataFrame", return_value=mock_sdf):
        yield mock_sdf

@pytest.fixture
def mock_vector_store():
    """
    Create a mock vector store for testing.
    """
    mock_vs = MagicMock()
    mock_vs.query.return_value = "Vector store result"
    mock_vs.current_query_distance = 0.95
    mock_vs.upsert_knowledge.return_value = None
    
    with patch("hana_ai.vectorstore.hana_vector_engine.HANAMLinVectorEngine", return_value=mock_vs):
        yield mock_vs

@pytest.fixture
def mock_embedding_model():
    """
    Create a mock embedding model for testing.
    """
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(5)]
    
    with patch("hana_ai.vectorstore.embedding_service.HANAVectorEmbeddings", return_value=mock_emb), \
         patch("hana_ai.vectorstore.embedding_service.PALModelEmbeddings", return_value=mock_emb):
        yield mock_emb