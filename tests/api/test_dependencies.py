"""
Tests for API dependencies.
"""
import pytest
from fastapi import HTTPException, Request
from unittest.mock import MagicMock, patch

def test_get_connection_context():
    """Test the get_connection_context dependency."""
    from hana_ai.api.dependencies import get_connection_context
    from hana_ai.api.config import settings
    
    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.connection_pool = {}
    
    # Mock ConnectionContext
    with patch("hana_ml.dataframe.ConnectionContext") as mock_conn_class:
        # Configure the mock
        mock_conn = mock_conn_class.return_value
        
        # Call the dependency
        result = get_connection_context(mock_request)
        
        # Check result
        assert result == mock_conn
        
        # Verify connection was created correctly
        if settings.HANA_USERKEY:
            mock_conn_class.assert_called_once_with(userkey=settings.HANA_USERKEY)
        else:
            mock_conn_class.assert_called_once()

def test_connection_pooling():
    """Test connection pooling in get_connection_context."""
    from hana_ai.api.dependencies import get_connection_context
    
    # Create mock request and connection
    mock_request = MagicMock(spec=Request)
    mock_conn = MagicMock()
    mock_conn.sql.return_value.collect.return_value = [{"DUMMY": 1}]
    
    # Add the connection to the pool
    mock_request.app.state.connection_pool = {id(mock_conn): mock_conn}
    
    # Call the dependency
    result = get_connection_context(mock_request)
    
    # Check result
    assert result == mock_conn
    
    # Verify connection test was performed
    mock_conn.sql.assert_called_with("SELECT 1 FROM DUMMY")

def test_connection_error_handling():
    """Test error handling in get_connection_context."""
    from hana_ai.api.dependencies import get_connection_context
    
    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.connection_pool = {}
    
    # Mock ConnectionContext to raise an exception
    with patch("hana_ml.dataframe.ConnectionContext") as mock_conn_class:
        mock_conn_class.side_effect = Exception("Connection error")
        
        # Call the dependency and check for exception
        with pytest.raises(HTTPException) as excinfo:
            get_connection_context(mock_request)
        
        # Verify the exception details
        assert excinfo.value.status_code == 500
        assert "Connection error" in excinfo.value.detail

def test_get_llm():
    """Test the get_llm dependency."""
    from hana_ai.api.dependencies import get_llm
    
    # Create mock request
    mock_request = MagicMock(spec=Request)
    
    # Test with GenAI Hub SDK
    with patch("gen_ai_hub.proxy.langchain.init_llm") as mock_init_llm:
        # Configure the mock
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm
        
        # Call the dependency
        result = get_llm(mock_request, "test_api_key")
        
        # Check result
        assert result == mock_llm
        
        # Verify LLM was initialized correctly
        mock_init_llm.assert_called_once()
    
    # Test with ImportError (fallback to FakeListLLM)
    with patch("gen_ai_hub.proxy.langchain.init_llm", side_effect=ImportError), \
         patch("langchain_community.llms.FakeListLLM") as mock_fake_llm:
        # Configure the mock
        mock_llm = MagicMock()
        mock_fake_llm.return_value = mock_llm
        
        # Call the dependency
        result = get_llm(mock_request, "test_api_key")
        
        # Check result
        assert result == mock_llm
        
        # Verify FakeListLLM was initialized correctly
        mock_fake_llm.assert_called_once()

def test_llm_error_handling():
    """Test error handling in get_llm."""
    from hana_ai.api.dependencies import get_llm
    
    # Create mock request
    mock_request = MagicMock(spec=Request)
    
    # Mock both LLM options to raise exceptions
    with patch("gen_ai_hub.proxy.langchain.init_llm", side_effect=ImportError), \
         patch("langchain_community.llms.FakeListLLM", side_effect=Exception("LLM error")):
        
        # Call the dependency and check for exception
        with pytest.raises(HTTPException) as excinfo:
            get_llm(mock_request, "test_api_key")
        
        # Verify the exception details
        assert excinfo.value.status_code == 500
        assert "LLM error" in excinfo.value.detail