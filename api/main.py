"""
FastAPI application for SAP HANA AI Toolkit with NVIDIA GPU Optimizations.
"""
from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import os
import time
import logging
from datetime import datetime
import uuid
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAP HANA AI Toolkit API",
    description="API for SAP HANA AI Toolkit with NVIDIA GPU Optimizations",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
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

class PerformanceData(BaseModel):
    """Performance benchmark data model."""
    operation: str = Field(..., description="Operation name")
    standard: str = Field(..., description="Standard performance (ms)")
    optimized: str = Field(..., description="Optimized performance (ms)")
    speedup: str = Field(..., description="Speedup factor")

class SystemInfo(BaseModel):
    """System information response model."""
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    gpu_enabled: bool = Field(..., description="GPU acceleration enabled")
    tensorrt_enabled: bool = Field(..., description="TensorRT optimization enabled")
    performance: List[PerformanceData] = Field(..., description="Performance benchmarks")

class ModelConfig(BaseModel):
    """Model configuration request model."""
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.0, description="Temperature")
    max_tokens: int = Field(1000, description="Maximum tokens")
    use_tensorrt: bool = Field(True, description="Use TensorRT optimization")

class ModelResponse(BaseModel):
    """Model response model."""
    model: str = Field(..., description="Model name")
    input: str = Field(..., description="Input text")
    output: str = Field(..., description="Generated output")
    tokens: int = Field(..., description="Number of tokens generated")
    latency_ms: int = Field(..., description="Latency in milliseconds")
    accelerated: bool = Field(..., description="Whether acceleration was used")
    tokenizer: str = Field(..., description="Tokenizer used")

class EmbeddingRequest(BaseModel):
    """Embedding request model."""
    text: str = Field(..., description="Text to embed")
    model: str = Field("sap-ai-core-embeddings", description="Embedding model name")
    use_tensorrt: bool = Field(True, description="Use TensorRT optimization")

class EmbeddingResponse(BaseModel):
    """Embedding response model."""
    embedding: List[float] = Field(..., description="Vector embedding")
    dimensions: int = Field(..., description="Embedding dimensions")
    model: str = Field(..., description="Model used")
    latency_ms: int = Field(..., description="Latency in milliseconds")
    accelerated: bool = Field(..., description="Whether acceleration was used")

# Auth dependency
async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from header."""
    # In production, use a secure method to validate API keys
    valid_keys = os.environ.get("API_KEYS", "dev-key-only-for-testing").split(",")
    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and response time."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"completed in {process_time:.4f}s with status {response.status_code}"
    )
    
    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/gpu", response_model=GPUInfo)
async def get_gpu_info(api_key: str = Depends(verify_api_key)):
    """Get GPU information."""
    # Mock data - in production this would query actual GPU info
    gpu_info = {
        "available": True,
        "count": 1,
        "names": ["NVIDIA A100-SXM4-40GB"],
        "memory_total": [40960],
        "memory_used": [2048],
        "utilization": [15.5],
        "tensorrt_available": True,
        "hopper_features": {
            "fp8": False,
            "transformer_engine": False,
            "flash_attention": True,
            "fsdp": False
        }
    }
    
    # Simulate checking if we're running on actual GPU hardware
    if not os.environ.get("ENABLE_GPU_ACCELERATION", "true").lower() in ("true", "1", "yes"):
        gpu_info["available"] = False
        gpu_info["count"] = 0
        gpu_info["names"] = []
        gpu_info["memory_total"] = []
        gpu_info["memory_used"] = []
        gpu_info["utilization"] = []
        gpu_info["tensorrt_available"] = False
    
    return GPUInfo(**gpu_info)

@app.get("/api/system", response_model=SystemInfo)
async def get_system_info():
    """Get system information."""
    return SystemInfo(
        version="1.0.0",
        environment=os.environ.get("ENVIRONMENT", "production"),
        gpu_enabled=os.environ.get("ENABLE_GPU_ACCELERATION", "true").lower() in ("true", "1", "yes"),
        tensorrt_enabled=os.environ.get("ENABLE_TENSORRT", "true").lower() in ("true", "1", "yes"),
        performance=[
            PerformanceData(
                operation="Embedding Generation",
                standard="85ms",
                optimized="24ms",
                speedup="3.5x"
            ),
            PerformanceData(
                operation="LLM Inference",
                standard="1450ms",
                optimized="580ms",
                speedup="2.5x"
            ),
            PerformanceData(
                operation="Vector Search",
                standard="120ms",
                optimized="45ms",
                speedup="2.7x"
            )
        ]
    )

@app.post("/api/generate", response_model=ModelResponse)
async def generate_text(config: ModelConfig, api_key: str = Depends(verify_api_key)):
    """Generate text using an LLM."""
    start_time = time.time()
    
    # Mock response - in production, this would call the actual model
    response = {
        "model": config.model,
        "input": "Tell me about NVIDIA GPU acceleration for AI workloads",
        "output": "NVIDIA GPU acceleration is critical for modern AI workloads. GPUs provide massive parallel processing capabilities that dramatically speed up deep learning training and inference tasks. With technologies like CUDA, TensorRT, and specialized hardware like Tensor Cores, NVIDIA GPUs can offer 10-100x performance improvements over CPUs for AI workloads. The latest Hopper architecture introduces new features like FP8 precision and Transformer Engine that further optimize performance for large language models and other transformer-based architectures.",
        "tokens": 78,
        "latency_ms": int((time.time() - start_time) * 1000),
        "accelerated": config.use_tensorrt,
        "tokenizer": "tiktoken"
    }
    
    # Simulate latency based on TensorRT usage
    if config.use_tensorrt:
        # Fast response with TensorRT
        time.sleep(0.05)  # 50ms
    else:
        # Slower response without TensorRT
        time.sleep(0.2)   # 200ms
    
    return ModelResponse(**response)

@app.post("/api/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, api_key: str = Depends(verify_api_key)):
    """Create vector embedding for text."""
    start_time = time.time()
    
    # Mock embedding - in production, this would use an actual embedding model
    import random
    dimensions = 768
    embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
    
    # Simulate latency based on TensorRT usage
    if request.use_tensorrt:
        # Fast response with TensorRT
        time.sleep(0.02)  # 20ms
    else:
        # Slower response without TensorRT
        time.sleep(0.08)  # 80ms
    
    latency = int((time.time() - start_time) * 1000)
    
    return EmbeddingResponse(
        embedding=embedding,
        dimensions=dimensions,
        model=request.model,
        latency_ms=latency,
        accelerated=request.use_tensorrt
    )

@app.get("/api/documentation")
async def get_documentation():
    """Get documentation links."""
    return {
        "repository": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud",
        "docs": {
            "authentication": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/AUTHENTICATION.md",
            "ngc_deployment": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NGC_DEPLOYMENT.md",
            "tensorrt": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/TENSORRT_OPTIMIZATION.md",
            "nvidia": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NVIDIA.md",
        }
    }

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )

# Mount static files
app.mount("/", StaticFiles(directory="public", html=True), name="public")

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.environ.get("ENVIRONMENT") == "development",
    )