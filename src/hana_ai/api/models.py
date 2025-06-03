"""
Pydantic models for API request and response validation.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class LLMConfig(BaseModel):
    """Configuration for language model."""
    model: str = Field(default="gpt-4", description="The LLM model to use")
    temperature: float = Field(default=0.0, description="Temperature parameter for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for LLM response")

class AgentMessage(BaseModel):
    """A message in a conversation."""
    role: str = Field(..., description="Message role (human or ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Message timestamp")

class ConversationRequest(BaseModel):
    """Request to interact with a conversation agent."""
    message: str = Field(..., description="User message to the agent")
    session_id: str = Field(..., description="Session identifier for conversation continuity")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM configuration")
    return_intermediate_steps: bool = Field(default=False, description="Whether to return intermediate reasoning steps")
    verbose: bool = Field(default=False, description="Enable verbose logging")

class ConversationResponse(BaseModel):
    """Response from a conversation agent."""
    response: str = Field(..., description="Agent's response")
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    created_at: datetime = Field(default_factory=datetime.now)
    intermediate_steps: Optional[List[Any]] = Field(default=None, description="Intermediate reasoning steps")
    
class ForecastHorizon(str, Enum):
    """Forecast horizon options."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    CUSTOM = "custom"

class TimeSeriesRequest(BaseModel):
    """Request to perform time series operations."""
    table_name: str = Field(..., description="Table containing time series data")
    key_column: str = Field(..., description="Column containing date/time values")
    value_column: str = Field(..., description="Column containing the target values")
    horizon: Union[ForecastHorizon, int] = Field(..., description="Forecast horizon")
    model_name: Optional[str] = Field(default=None, description="Name to save the model under")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional model parameters")

class TimeSeriesResponse(BaseModel):
    """Response from a time series operation."""
    model_id: str = Field(..., description="Identifier of the created/used model")
    model_version: int = Field(..., description="Model version")
    result_table: Optional[str] = Field(default=None, description="Table containing results if applicable")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Performance metrics if applicable")
    visualization_url: Optional[str] = Field(default=None, description="URL to visualization if generated")

class SmartDataFrameRequest(BaseModel):
    """Request for SmartDataFrame operations."""
    table_name: str = Field(..., description="Table or SQL query to create dataframe from")
    question: str = Field(..., description="Question to ask about the data")
    is_sql_query: bool = Field(default=False, description="Whether table_name is an SQL query")
    transform: bool = Field(default=False, description="Whether to transform the dataframe or just query it")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM configuration")

class DataResponse(BaseModel):
    """Response containing dataframe data."""
    columns: List[str] = Field(..., description="Column names")
    data: List[Dict[str, Any]] = Field(..., description="Data rows")
    row_count: int = Field(..., description="Total number of rows")
    query_time: float = Field(..., description="Time taken to execute the query in seconds")

class VectorStoreRequest(BaseModel):
    """Request for vector store operations."""
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(default=3, description="Number of results to return")
    collection_name: Optional[str] = Field(default=None, description="Vector store collection")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Filter criteria")
    
class VectorStoreResponse(BaseModel):
    """Response from vector store query."""
    results: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    query_time: float = Field(..., description="Time taken to execute the query in seconds")
    
class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")