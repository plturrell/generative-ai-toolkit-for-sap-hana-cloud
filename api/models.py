"""
Data models for the API.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    type: Optional[str] = Field(None, description="Error type")
    code: Optional[str] = Field(None, description="Error code")


class GPUInfo(BaseModel):
    """GPU information response model."""
    available: bool = Field(..., description="Whether GPU acceleration is available")
    count: int = Field(0, description="Number of available GPUs")
    names: List[str] = Field([], description="Names of available GPUs")
    memory_total: List[int] = Field([], description="Total memory in MB for each GPU")
    memory_used: List[int] = Field([], description="Used memory in MB for each GPU")
    utilization: List[float] = Field([], description="Utilization percentage for each GPU")
    tensorrt_available: bool = Field(False, description="Whether TensorRT is available")
    hopper_features: Dict[str, bool] = Field({}, description="Available Hopper features")


class TensorRTInfo(BaseModel):
    """TensorRT information model."""
    available: bool = Field(..., description="Whether TensorRT is available")
    version: Optional[str] = Field(None, description="TensorRT version")
    optimized_models: List[str] = Field([], description="List of optimized models")
    supported_precisions: List[str] = Field([], description="Supported precision modes")


class PerformanceMetric(BaseModel):
    """Performance metric model."""
    operation: str = Field(..., description="Operation name")
    standard_ms: float = Field(..., description="Standard performance in milliseconds")
    optimized_ms: float = Field(..., description="Optimized performance in milliseconds")
    speedup: float = Field(..., description="Speedup factor")
    batch_size: int = Field(1, description="Batch size used for measurement")
    device: str = Field("CUDA", description="Device used for measurement")


class SystemInfo(BaseModel):
    """System information response model."""
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    gpu_info: GPUInfo = Field(..., description="GPU information")
    tensorrt_info: TensorRTInfo = Field(..., description="TensorRT information")
    performance_metrics: List[PerformanceMetric] = Field([], description="Performance metrics")
    hana_connected: bool = Field(False, description="Whether connected to HANA")


class ConnectionConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(..., description="HANA host")
    port: int = Field(..., description="HANA port")
    user: str = Field(..., description="HANA user")
    password: str = Field(..., description="HANA password")
    encrypt: bool = Field(True, description="Whether to use encryption")
    sslValidateCertificate: bool = Field(False, description="Whether to validate SSL certificate")


class GenerationConfig(BaseModel):
    """Text generation configuration."""
    model: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Prompt for text generation")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    stop_sequences: List[str] = Field([], description="Sequences that stop generation")
    use_tensorrt: bool = Field(True, description="Whether to use TensorRT optimization")


class GenerationResponse(BaseModel):
    """Text generation response."""
    model: str = Field(..., description="Model used")
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_prompt: int = Field(..., description="Number of tokens in prompt")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    accelerated: bool = Field(..., description="Whether acceleration was used")


class EmbeddingRequest(BaseModel):
    """Embedding request model."""
    text: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field("default", description="Embedding model name")
    use_tensorrt: bool = Field(True, description="Whether to use TensorRT optimization")


class EmbeddingResponse(BaseModel):
    """Embedding response model."""
    embeddings: Union[List[float], List[List[float]]] = Field(..., description="Vector embeddings")
    dimensions: int = Field(..., description="Embedding dimensions")
    model: str = Field(..., description="Model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    accelerated: bool = Field(..., description="Whether acceleration was used")


class DataframeQueryRequest(BaseModel):
    """DataFrame query request model."""
    query: str = Field(..., description="Natural language query")
    table_name: str = Field(..., description="HANA table name")
    transform: bool = Field(False, description="Whether to transform the data")
    limit: int = Field(100, description="Maximum number of rows to return")


class DataframeQueryResponse(BaseModel):
    """DataFrame query response model."""
    query: str = Field(..., description="Original query")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    data: List[Dict[str, Any]] = Field(..., description="Query results")
    row_count: int = Field(..., description="Number of rows returned")
    column_count: int = Field(..., description="Number of columns returned")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")


class AgentRequest(BaseModel):
    """Agent request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for stateful conversations")
    tools: List[str] = Field([], description="Tools to enable for the agent")
    verbose: bool = Field(False, description="Whether to return verbose output")


class AgentResponse(BaseModel):
    """Agent response model."""
    message: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    thinking: Optional[str] = Field(None, description="Agent's thinking process")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")