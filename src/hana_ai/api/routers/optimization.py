"""
FastAPI router for model optimization.

This module provides API endpoints for model optimization,
including sparsity, quantization, and other techniques to
improve performance on memory-constrained devices.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from pydantic import BaseModel, Field, validator

# Import optimization tools
from ...tools.optimization.sparsity_optimizer import (
    get_sparsity_optimizer,
    SparsityOptimizer,
    SparsityStats,
    DEFAULT_SPARSITY_CONFIG
)

# Import authentication
from ..auth import get_api_key, verify_api_key

# Logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Models for API
class SparsityConfig(BaseModel):
    """Sparsity configuration model."""
    
    enabled: bool = Field(default=True, description="Enable sparsity optimization")
    target_sparsity: float = Field(
        default=0.8, 
        description="Target sparsity level (0.0-1.0)",
        ge=0.0,
        le=0.99
    )
    use_block_sparsity: bool = Field(default=False, description="Use block sparsity")
    block_size: List[int] = Field(default=[1, 1], description="Block size for block sparsity")
    use_quantization: bool = Field(default=True, description="Enable quantization")
    quantization_bits: int = Field(
        default=8, 
        description="Quantization bit width",
        ge=1,
        le=16
    )
    quantization_scheme: str = Field(
        default="symmetric", 
        description="Quantization scheme (symmetric or asymmetric)"
    )
    per_channel_quantization: bool = Field(default=False, description="Use per-channel quantization")
    skip_layers: List[str] = Field(
        default=["embedding", "layernorm", "bias"],
        description="Layer types to skip"
    )
    min_params_to_sparsify: int = Field(
        default=1000,
        description="Minimum number of parameters for a tensor to be sparsified",
        ge=0
    )
    
    @validator("quantization_scheme")
    def validate_scheme(cls, v):
        """Validate quantization scheme."""
        allowed_schemes = ["symmetric", "asymmetric"]
        if v not in allowed_schemes:
            raise ValueError(f"Quantization scheme must be one of {allowed_schemes}")
        return v
    
    @validator("block_size")
    def validate_block_size(cls, v):
        """Validate block size."""
        if len(v) != 2:
            raise ValueError("Block size must be a list of 2 integers")
        if v[0] < 1 or v[1] < 1:
            raise ValueError("Block size must be at least 1x1")
        return v


class ModelOptimizationRequest(BaseModel):
    """Model optimization request."""
    
    model_id: str = Field(..., description="Model identifier")
    config: Optional[SparsityConfig] = Field(default=None, description="Optimization configuration")
    use_cache: bool = Field(default=True, description="Use cache for optimized models")


class ModelOptimizationResponse(BaseModel):
    """Model optimization response."""
    
    model_id: str = Field(..., description="Model identifier")
    optimized: bool = Field(..., description="Whether optimization was applied")
    stats: Dict[str, Any] = Field(..., description="Optimization statistics")
    config: Dict[str, Any] = Field(..., description="Configuration used")
    cached: bool = Field(..., description="Whether result was from cache")


class ModelOptimizationStatus(BaseModel):
    """Model optimization status."""
    
    status: str = Field(..., description="Optimization status")
    message: str = Field(..., description="Status message")
    stats: Optional[Dict[str, Any]] = Field(default=None, description="Optimization statistics if available")


# Helper function to convert SparsityStats to dict
def stats_to_dict(stats: SparsityStats) -> Dict[str, Any]:
    """Convert SparsityStats to dictionary."""
    if not stats:
        return {}
    
    return {
        "model_name": stats.model_name,
        "original_size_mb": stats.original_size_mb,
        "sparse_size_mb": stats.sparse_size_mb,
        "compression_ratio": stats.compression_ratio,
        "average_sparsity": stats.average_sparsity,
        "memory_reduction": stats.memory_reduction,
        "applied_techniques": stats.applied_techniques,
        "timestamp": stats.timestamp
    }


@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Get optimization status",
    description="Get the current status of the optimization service",
)
async def get_optimization_status(
    api_key: str = Depends(get_api_key)
):
    """Get optimization status endpoint."""
    # Verify API key
    verify_api_key(api_key)
    
    # Get sparsity optimizer
    optimizer = get_sparsity_optimizer()
    
    # Get current config
    config = optimizer.get_config()
    
    # Get cached models count
    cached_models = len(optimizer.optimized_models)
    
    return {
        "status": "active" if config.get("enabled", True) else "disabled",
        "config": config,
        "cached_models": cached_models,
        "pytorch_available": optimizer.sparse_model_optimizer.HAS_TORCH,
        "supported_techniques": [
            "unstructured_sparsity",
            "block_sparsity",
            "int8_quantization",
            "int4_quantization"
        ]
    }


@router.post(
    "/sparsify",
    response_model=ModelOptimizationResponse,
    summary="Optimize a model with sparsity",
    description=(
        "Apply sparsity optimization to a model to reduce memory usage and improve performance. "
        "This endpoint is for asynchronous optimization and returns immediately with a task ID."
    ),
)
async def optimize_model_endpoint(
    request: ModelOptimizationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Optimize model endpoint."""
    # Verify API key
    verify_api_key(api_key)
    
    # Get model ID
    model_id = request.model_id
    
    # Validate model ID
    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID is required")
    
    # Get config
    config = request.config.dict() if request.config else {}
    
    # Get sparsity optimizer
    optimizer = get_sparsity_optimizer(config)
    
    # Check if model is already optimized and cached
    if request.use_cache:
        stats = optimizer.get_stats(model_id)
        if stats:
            # Return cached result
            return ModelOptimizationResponse(
                model_id=model_id,
                optimized=True,
                stats=stats_to_dict(stats),
                config=optimizer.get_config(),
                cached=True
            )
    
    # Get model from registry
    from ..app import app
    if not hasattr(app.state, "model_registry"):
        raise HTTPException(status_code=404, detail="Model registry not available")
    
    model_registry = app.state.model_registry
    model = model_registry.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Optimize model in background task
    def optimize_model_task():
        try:
            # Get model type
            model_type = "pytorch"  # Default to PyTorch
            
            # Determine model type from class name if possible
            model_class = model.__class__.__name__.lower()
            if "transformers" in model_class or "bert" in model_class or "gpt" in model_class:
                model_type = "pytorch"
            elif "tensorflow" in model_class or "keras" in model_class:
                model_type = "tensorflow"
            
            # Apply optimization
            _, stats = optimizer.optimize_model(
                model,
                model_name=model_id,
                model_type=model_type,
                cache_id=model_id
            )
            
            # Update model in registry
            model_registry.update_model(model_id, model)
            
            logger.info(f"Optimized model {model_id} with sparsity")
        except Exception as e:
            logger.error(f"Error optimizing model {model_id}: {str(e)}")
    
    # Start background task
    background_tasks.add_task(optimize_model_task)
    
    # Return immediate response
    return ModelOptimizationResponse(
        model_id=model_id,
        optimized=True,
        stats={
            "model_name": model_id,
            "status": "optimizing",
            "message": "Optimization started in background"
        },
        config=optimizer.get_config(),
        cached=False
    )


@router.get(
    "/models/{model_id}/stats",
    response_model=Dict[str, Any],
    summary="Get model optimization statistics",
    description="Get optimization statistics for a model",
)
async def get_model_stats(
    model_id: str = Path(..., description="Model ID"),
    api_key: str = Depends(get_api_key)
):
    """Get model optimization statistics endpoint."""
    # Verify API key
    verify_api_key(api_key)
    
    # Get sparsity optimizer
    optimizer = get_sparsity_optimizer()
    
    # Get stats
    stats = optimizer.get_stats(model_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"No optimization stats found for model {model_id}")
    
    return stats_to_dict(stats)


@router.delete(
    "/models/{model_id}/cache",
    response_model=Dict[str, Any],
    summary="Clear model optimization cache",
    description="Clear the optimization cache for a specific model",
)
async def clear_model_cache(
    model_id: str = Path(..., description="Model ID"),
    api_key: str = Depends(get_api_key)
):
    """Clear model optimization cache endpoint."""
    # Verify API key
    verify_api_key(api_key)
    
    # Get sparsity optimizer
    optimizer = get_sparsity_optimizer()
    
    # Clear cache
    optimizer.clear_cache(model_id)
    
    return {"status": "success", "message": f"Cleared optimization cache for model {model_id}"}


@router.put(
    "/config",
    response_model=Dict[str, Any],
    summary="Update optimization configuration",
    description="Update the global optimization configuration",
)
async def update_config(
    config: SparsityConfig = Body(..., description="New configuration"),
    api_key: str = Depends(get_api_key)
):
    """Update optimization configuration endpoint."""
    # Verify API key
    verify_api_key(api_key)
    
    # Get sparsity optimizer
    optimizer = get_sparsity_optimizer()
    
    # Update config
    optimizer.update_config(config.dict())
    
    return {"status": "success", "config": optimizer.get_config()}