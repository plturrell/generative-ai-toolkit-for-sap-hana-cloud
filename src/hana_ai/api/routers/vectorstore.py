"""
API endpoints for working with vector stores and embeddings.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field

from hana_ml.dataframe import ConnectionContext
from langchain.llms.base import BaseLLM

from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from hana_ai.vectorstore.embedding_service import HANAVectorEmbeddings, PALModelEmbeddings

from ..dependencies import get_connection_context, get_llm
from ..auth import get_api_key
from ..models import VectorStoreRequest, VectorStoreResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class DocumentRequest(BaseModel):
    """Request to add documents to a vector store."""
    documents: List[Dict[str, str]] = Field(..., description="Documents to add to vector store")
    store_name: str = Field(..., description="Name of the vector store")
    schema: Optional[str] = Field(None, description="Database schema")

@router.post(
    "/query",
    response_model=VectorStoreResponse,
    summary="Query vector store",
    description="Search for similar documents in a vector store"
)
async def query_vector_store(
    request: VectorStoreRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Query a vector store for similar documents.
    
    Parameters
    ----------
    request : VectorStoreRequest
        The query request
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    VectorStoreResponse
        Query results
    """
    try:
        start_time = time.time()
        
        # Initialize vector store
        vector_store = HANAMLinVectorEngine(
            connection_context=connection_context, 
            table_name=request.collection_name or "hana_vec_default"
        )
        
        # Execute query
        result = vector_store.query(
            input=request.query,
            top_n=request.top_k,
            distance="cosine_similarity"
        )
        
        query_time = time.time() - start_time
        
        return VectorStoreResponse(
            results=[{"content": result, "score": vector_store.current_query_distance}],
            query_time=query_time
        )
    except Exception as e:
        logger.error(f"Vector store query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error querying vector store: {str(e)}"
        )

@router.post(
    "/store",
    summary="Add documents to vector store",
    description="Add documents to a HANA vector store with auto-generated embeddings"
)
async def add_to_vector_store(
    request: DocumentRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Add documents to a vector store.
    
    Parameters
    ----------
    request : DocumentRequest
        The documents to add
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    Dict
        Status of the operation
    """
    try:
        start_time = time.time()
        
        # Format documents for vector store
        knowledge_items = []
        for i, doc in enumerate(request.documents):
            item = {
                "id": doc.get("id", f"doc_{i}"),
                "description": doc.get("description", ""),
                "example": doc.get("content", doc.get("text", ""))
            }
            knowledge_items.append(item)
            
        # Initialize vector store
        vector_store = HANAMLinVectorEngine(
            connection_context=connection_context, 
            table_name=request.store_name,
            schema=request.schema
        )
        
        # Add documents
        vector_store.upsert_knowledge(knowledge_items)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "documents_added": len(knowledge_items),
            "vector_store": request.store_name,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error adding documents to vector store: {str(e)}"
        )

@router.post(
    "/embed",
    summary="Generate embeddings",
    description="Generate embeddings for text using HANA embedding services"
)
async def generate_embeddings(
    texts: List[str] = Body(..., description="Texts to embed"),
    model_type: str = Query("hana", description="Embedding model type (hana or pal)"),
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Generate embeddings for texts.
    
    Parameters
    ----------
    texts : List[str]
        Texts to embed
    model_type : str
        Type of embedding model to use
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    Dict
        Generated embeddings
    """
    try:
        start_time = time.time()
        
        # Initialize embedding model
        if model_type.lower() == "hana":
            embedding_model = HANAVectorEmbeddings(connection_context)
        elif model_type.lower() == "pal":
            embedding_model = PALModelEmbeddings(connection_context)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported embedding model type: {model_type}"
            )
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # For response size considerations, truncate very large embeddings in the display
        display_embeddings = []
        for emb in embeddings:
            # If embedding is very large, truncate for display
            if len(emb) > 20:
                display_emb = emb[:10] + ["..."] + emb[-10:]
            else:
                display_emb = emb
            display_embeddings.append(display_emb)
        
        processing_time = time.time() - start_time
        
        return {
            "embeddings": display_embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings),
            "model_type": model_type,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )