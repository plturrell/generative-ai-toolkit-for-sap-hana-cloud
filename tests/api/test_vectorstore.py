"""
Tests for the vectorstore API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

def test_query_vector_store(test_client, mock_vector_store, mock_connection_context):
    """Test the query_vector_store endpoint."""
    # Prepare test data
    request_data = {
        "query": "Test query",
        "top_k": 3,
        "collection_name": "test_collection"
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/vectorstore/query", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "results" in response.json()
    assert "query_time" in response.json()
    assert len(response.json()["results"]) == 1
    assert "content" in response.json()["results"][0]
    assert "score" in response.json()["results"][0]
    
    # Verify vector store was queried
    mock_vector_store.query.assert_called_once_with(
        input="Test query",
        top_n=3,
        distance="cosine_similarity"
    )

def test_add_to_vector_store(test_client, mock_vector_store, mock_connection_context):
    """Test the add_to_vector_store endpoint."""
    # Prepare test data
    request_data = {
        "documents": [
            {
                "id": "doc1",
                "description": "Test document 1",
                "content": "This is a test document"
            },
            {
                "id": "doc2",
                "description": "Test document 2",
                "content": "This is another test document"
            }
        ],
        "store_name": "test_store"
    }
    
    # Call the endpoint
    response = test_client.post("/api/v1/vectorstore/store", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"
    assert "documents_added" in response.json()
    assert response.json()["documents_added"] == 2
    assert "vector_store" in response.json()
    assert response.json()["vector_store"] == "test_store"
    
    # Verify vector store was created and updated
    mock_vector_store.upsert_knowledge.assert_called_once()
    assert len(mock_vector_store.upsert_knowledge.call_args[0][0]) == 2

def test_generate_embeddings_hana(test_client, mock_connection_context, mock_embedding_model):
    """Test the generate_embeddings endpoint with HANA model."""
    # Prepare test data
    texts = ["This is a test", "This is another test"]
    
    # Call the endpoint
    response = test_client.post(
        "/api/v1/vectorstore/embed",
        json=texts,
        params={"model_type": "hana"}
    )
    
    # Check the response
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "dimensions" in response.json()
    assert "count" in response.json()
    assert "model_type" in response.json()
    assert response.json()["model_type"] == "hana"
    assert response.json()["count"] == 5  # From the mock
    
    # Verify embedding model was called
    mock_embedding_model.embed_documents.assert_called_once_with(texts)

def test_generate_embeddings_pal(test_client, mock_connection_context, mock_embedding_model):
    """Test the generate_embeddings endpoint with PAL model."""
    # Prepare test data
    texts = ["This is a test", "This is another test"]
    
    # Call the endpoint
    response = test_client.post(
        "/api/v1/vectorstore/embed",
        json=texts,
        params={"model_type": "pal"}
    )
    
    # Check the response
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "model_type" in response.json()
    assert response.json()["model_type"] == "pal"
    
    # Verify embedding model was called
    mock_embedding_model.embed_documents.assert_called_once_with(texts)

def test_invalid_model_type(test_client, mock_connection_context):
    """Test the generate_embeddings endpoint with invalid model type."""
    # Prepare test data
    texts = ["This is a test"]
    
    # Call the endpoint
    response = test_client.post(
        "/api/v1/vectorstore/embed",
        json=texts,
        params={"model_type": "invalid_model"}
    )
    
    # Check the error response
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Unsupported embedding model type" in response.json()["detail"]

def test_error_handling(test_client, mock_connection_context):
    """Test error handling in vectorstore endpoints."""
    from unittest.mock import patch
    
    with patch("hana_ai.vectorstore.hana_vector_engine.HANAMLinVectorEngine") as mock_vs_class:
        # Configure the mock to raise an exception
        mock_vs_class.side_effect = Exception("Vector store error")
        
        # Prepare test data
        request_data = {
            "query": "Test query",
            "top_k": 3,
            "collection_name": "test_collection"
        }
        
        # Call the endpoint
        response = test_client.post("/api/v1/vectorstore/query", json=request_data)
        
        # Check the error response
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Vector store error" in response.json()["detail"]