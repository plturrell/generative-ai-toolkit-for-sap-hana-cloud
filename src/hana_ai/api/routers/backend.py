"""
Backend configuration router for managing GPU acceleration in the SAP HANA AI Toolkit.

This module provides endpoints for configuring and managing multiple backend options
including NVIDIA LaunchPad, Together.ai, and CPU-only processing.
"""
import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Body, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..auth import get_admin_api_key
from ..config import settings
from ..backend_config import backend_config, BackendType, BackendConfig
from ..backend_manager import backend_manager
from ..backend_router import get_backend_status, mark_backend_available, mark_backend_unavailable

router = APIRouter()
logger = logging.getLogger(__name__)

# Config file path
BACKEND_CONFIG_FILE = os.path.join(settings.CONFIG_DIR, "backend_config.json")

# Create config directory if it doesn't exist
os.makedirs(settings.CONFIG_DIR, exist_ok=True)

# Models for request validation
class BackendPriorityRequest(BaseModel):
    """Configuration for backend priority and failover."""
    primary: str = Field(..., description="Primary backend to use (nvidia, together_ai, cpu, auto)")
    secondary: Optional[str] = Field(None, description="Secondary backend for failover")
    auto_failover: bool = Field(True, description="Whether to automatically failover to secondary backend")
    failover_attempts: int = Field(3, description="Number of attempts before failing over")
    failover_timeout: float = Field(10.0, description="Timeout in seconds before failing over")
    load_balancing: bool = Field(False, description="Whether to load balance between backends")
    load_ratio: float = Field(0.8, description="Ratio of requests to send to primary backend (0.0-1.0)")

class NvidiaBackendRequest(BaseModel):
    """Configuration for NVIDIA backend."""
    enabled: bool = Field(..., description="Whether NVIDIA backend is enabled")
    enable_tensorrt: bool = Field(True, description="Whether to enable TensorRT optimizations")
    enable_flash_attention: bool = Field(True, description="Whether to enable Flash Attention")
    enable_transformer_engine: bool = Field(True, description="Whether to enable Transformer Engine")
    enable_fp8: bool = Field(True, description="Whether to enable FP8 precision")
    enable_gptq: bool = Field(True, description="Whether to enable GPTQ quantization")
    enable_awq: bool = Field(True, description="Whether to enable AWQ quantization")
    default_quant_method: str = Field("gptq", description="Default quantization method")
    quantization_bit_width: int = Field(4, description="Quantization bit width")
    cuda_memory_fraction: float = Field(0.85, description="Fraction of GPU memory to use")
    multi_gpu_strategy: str = Field("auto", description="Multi-GPU parallelism strategy")

class TogetherAIBackendRequest(BaseModel):
    """Configuration for Together.ai backend."""
    enabled: bool = Field(..., description="Whether Together.ai backend is enabled")
    api_key: str = Field(..., description="Together.ai API key")
    default_model: str = Field("meta-llama/Llama-2-70b-chat-hf", description="Default model for completions")
    default_embedding_model: str = Field("togethercomputer/m2-bert-80M-8k-retrieval", description="Default model for embeddings")
    timeout: float = Field(60.0, description="Request timeout in seconds")
    endpoint_url: Optional[str] = Field(None, description="URL for dedicated endpoint if available")
    max_retries: int = Field(3, description="Maximum number of retries for API calls")

class CPUBackendRequest(BaseModel):
    """Configuration for CPU-only backend."""
    enabled: bool = Field(..., description="Whether CPU backend is enabled")
    default_model: str = Field("llama-2-7b-chat.Q4_K_M.gguf", description="Default model for CPU inference")
    default_embedding_model: str = Field("all-MiniLM-L6-v2", description="Default model for embeddings")
    num_threads: int = Field(4, description="Number of threads to use for CPU inference")
    context_size: int = Field(2048, description="Context size for CPU inference")

class BackendConfigRequest(BaseModel):
    """Complete backend configuration for SAP HANA AI Toolkit."""
    priority: BackendPriorityRequest = Field(..., description="Backend priority and failover configuration")
    nvidia: NvidiaBackendRequest = Field(..., description="NVIDIA backend configuration")
    together_ai: TogetherAIBackendRequest = Field(..., description="Together.ai backend configuration")
    cpu: CPUBackendRequest = Field(..., description="CPU backend configuration")

def save_backend_config(config: BackendConfig):
    """Save backend configuration to file."""
    config.save_to_file(BACKEND_CONFIG_FILE)
    logger.info(f"Saved backend configuration to {BACKEND_CONFIG_FILE}")

def load_backend_config() -> BackendConfig:
    """Load backend configuration from file or create default."""
    if os.path.exists(BACKEND_CONFIG_FILE):
        try:
            config = BackendConfig.load_from_file(BACKEND_CONFIG_FILE)
            logger.info(f"Loaded backend configuration from {BACKEND_CONFIG_FILE}")
            return config
        except Exception as e:
            logger.error(f"Error loading backend configuration: {str(e)}")
    
    # Return default configuration
    config = BackendConfig.from_environment()
    logger.info("Created default backend configuration from environment")
    return config

@router.get(
    "/",
    summary="Get backend configuration",
    description="Get the current backend configuration"
)
async def get_backend_config(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Get the current backend configuration.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Backend configuration
    """
    return backend_config.to_dict()

@router.post(
    "/",
    summary="Update backend configuration",
    description="Update the backend configuration"
)
async def update_backend_config(
    config_request: BackendConfigRequest,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Update the backend configuration.
    
    Parameters
    ----------
    config_request : BackendConfigRequest
        Backend configuration request
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Updated backend configuration
    """
    try:
        # Convert request to backend config
        new_config = BackendConfig(
            priority=config_request.priority.dict(),
            nvidia=config_request.nvidia.dict(),
            together_ai=config_request.together_ai.dict(),
            cpu=config_request.cpu.dict()
        )
        
        # Save configuration
        save_backend_config(new_config)
        
        # Update global config
        global backend_config
        backend_config = new_config
        
        # Reinitialize backends
        await reinitialize_backends()
        
        return new_config.to_dict()
        
    except Exception as e:
        logger.error(f"Error updating backend configuration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating backend configuration: {str(e)}"
        )

@router.post(
    "/priority",
    summary="Update backend priority",
    description="Update backend priority and failover settings"
)
async def update_backend_priority(
    priority: BackendPriorityRequest,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Update backend priority and failover settings.
    
    Parameters
    ----------
    priority : BackendPriorityRequest
        Backend priority configuration
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Updated backend configuration
    """
    try:
        # Load current config
        config = backend_config
        
        # Update priority settings
        config.priority.primary = BackendType(priority.primary)
        config.priority.secondary = BackendType(priority.secondary) if priority.secondary else None
        config.priority.auto_failover = priority.auto_failover
        config.priority.failover_attempts = priority.failover_attempts
        config.priority.failover_timeout = priority.failover_timeout
        config.priority.load_balancing = priority.load_balancing
        config.priority.load_ratio = priority.load_ratio
        
        # Save configuration
        save_backend_config(config)
        
        return config.to_dict()
        
    except Exception as e:
        logger.error(f"Error updating backend priority: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating backend priority: {str(e)}"
        )

@router.post(
    "/nvidia",
    summary="Update NVIDIA backend configuration",
    description="Update NVIDIA GPU acceleration settings"
)
async def update_nvidia_backend(
    config: NvidiaBackendRequest,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Update NVIDIA backend configuration.
    
    Parameters
    ----------
    config : NvidiaBackendRequest
        NVIDIA backend configuration
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Updated backend configuration
    """
    try:
        # Load current config
        current_config = backend_config
        
        # Update NVIDIA settings
        current_config.nvidia.enabled = config.enabled
        current_config.nvidia.enable_tensorrt = config.enable_tensorrt
        current_config.nvidia.enable_flash_attention = config.enable_flash_attention
        current_config.nvidia.enable_transformer_engine = config.enable_transformer_engine
        current_config.nvidia.enable_fp8 = config.enable_fp8
        current_config.nvidia.enable_gptq = config.enable_gptq
        current_config.nvidia.enable_awq = config.enable_awq
        current_config.nvidia.default_quant_method = config.default_quant_method
        current_config.nvidia.quantization_bit_width = config.quantization_bit_width
        current_config.nvidia.cuda_memory_fraction = config.cuda_memory_fraction
        current_config.nvidia.multi_gpu_strategy = config.multi_gpu_strategy
        
        # Save configuration
        save_backend_config(current_config)
        
        # Reinitialize NVIDIA backend if it's enabled
        if config.enabled:
            backend_manager.initialize_backend(BackendType.NVIDIA)
        
        return current_config.to_dict()
        
    except Exception as e:
        logger.error(f"Error updating NVIDIA backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating NVIDIA backend: {str(e)}"
        )

@router.post(
    "/together-ai",
    summary="Update Together.ai backend configuration",
    description="Update Together.ai cloud GPU settings"
)
async def update_together_ai_backend(
    config: TogetherAIBackendRequest,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Update Together.ai backend configuration.
    
    Parameters
    ----------
    config : TogetherAIBackendRequest
        Together.ai backend configuration
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Updated backend configuration
    """
    try:
        # Load current config
        current_config = backend_config
        
        # Update Together.ai settings
        current_config.together_ai.enabled = config.enabled
        current_config.together_ai.api_key = config.api_key
        current_config.together_ai.default_model = config.default_model
        current_config.together_ai.default_embedding_model = config.default_embedding_model
        current_config.together_ai.timeout = config.timeout
        current_config.together_ai.endpoint_url = config.endpoint_url
        current_config.together_ai.max_retries = config.max_retries
        
        # Save configuration
        save_backend_config(current_config)
        
        # Reinitialize Together.ai backend if it's enabled
        if config.enabled and config.api_key:
            backend_manager.initialize_backend(BackendType.TOGETHER_AI)
        
        return current_config.to_dict()
        
    except Exception as e:
        logger.error(f"Error updating Together.ai backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating Together.ai backend: {str(e)}"
        )

@router.post(
    "/cpu",
    summary="Update CPU backend configuration",
    description="Update CPU-only processing settings"
)
async def update_cpu_backend(
    config: CPUBackendRequest,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Update CPU backend configuration.
    
    Parameters
    ----------
    config : CPUBackendRequest
        CPU backend configuration
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Updated backend configuration
    """
    try:
        # Load current config
        current_config = backend_config
        
        # Update CPU settings
        current_config.cpu.enabled = config.enabled
        current_config.cpu.default_model = config.default_model
        current_config.cpu.default_embedding_model = config.default_embedding_model
        current_config.cpu.num_threads = config.num_threads
        current_config.cpu.context_size = config.context_size
        
        # Save configuration
        save_backend_config(current_config)
        
        # Reinitialize CPU backend if it's enabled
        if config.enabled:
            backend_manager.initialize_backend(BackendType.CPU)
        
        return current_config.to_dict()
        
    except Exception as e:
        logger.error(f"Error updating CPU backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating CPU backend: {str(e)}"
        )

@router.get(
    "/status",
    summary="Get backend status",
    description="Get the status of all backends"
)
async def get_backend_status_endpoint():
    """
    Get the status of all backends.
    
    Returns
    -------
    Dict
        Status of all backends
    """
    return backend_manager.get_backend_status()

@router.post(
    "/reinitialize",
    summary="Reinitialize backends",
    description="Reinitialize all backends"
)
async def reinitialize_backends(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Reinitialize all backends.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Status of all backends after reinitialization
    """
    try:
        # Get active backends
        active_backends = backend_config.determine_active_backends()
        
        # Reset initialized backends
        backend_manager.initialized_backends = set()
        
        # Initialize active backends
        for backend_type in active_backends:
            backend_manager.initialize_backend(backend_type)
        
        return backend_manager.get_backend_status()
        
    except Exception as e:
        logger.error(f"Error reinitializing backends: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error reinitializing backends: {str(e)}"
        )

@router.post(
    "/test/nvidia",
    summary="Test NVIDIA backend",
    description="Test the NVIDIA GPU backend"
)
async def test_nvidia_backend(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the NVIDIA backend.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Check if NVIDIA is enabled
        if not backend_config.nvidia.enabled:
            return {
                "status": "disabled",
                "message": "NVIDIA backend is disabled in configuration"
            }
        
        # Initialize NVIDIA backend
        result = backend_manager.initialize_backend(BackendType.NVIDIA)
        
        if not result:
            return {
                "status": "error",
                "message": "Failed to initialize NVIDIA backend"
            }
        
        # Try a simple operation
        try:
            # Generate a simple text
            start_time = time.time()
            response = backend_manager.generate_text(
                prompt="Hello, world!",
                max_tokens=10,
                backend=BackendType.NVIDIA
            )
            elapsed_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "NVIDIA backend test successful",
                "elapsed_time": elapsed_time,
                "response": response
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"NVIDIA backend test failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error testing NVIDIA backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing NVIDIA backend: {str(e)}"
        )

@router.post(
    "/test/together-ai",
    summary="Test Together.ai backend",
    description="Test the Together.ai cloud GPU backend"
)
async def test_together_ai_backend(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the Together.ai backend.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Check if Together.ai is enabled
        if not backend_config.together_ai.enabled:
            return {
                "status": "disabled",
                "message": "Together.ai backend is disabled in configuration"
            }
        
        if not backend_config.together_ai.api_key:
            return {
                "status": "error",
                "message": "Together.ai API key is not set"
            }
        
        # Initialize Together.ai backend
        result = backend_manager.initialize_backend(BackendType.TOGETHER_AI)
        
        if not result:
            return {
                "status": "error",
                "message": "Failed to initialize Together.ai backend"
            }
        
        # Try a simple operation
        try:
            # Generate a simple text
            start_time = time.time()
            response = backend_manager.generate_text(
                prompt="Hello, world!",
                max_tokens=10,
                backend=BackendType.TOGETHER_AI
            )
            elapsed_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "Together.ai backend test successful",
                "elapsed_time": elapsed_time,
                "response": response
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Together.ai backend test failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error testing Together.ai backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing Together.ai backend: {str(e)}"
        )

@router.post(
    "/test/cpu",
    summary="Test CPU backend",
    description="Test the CPU-only backend"
)
async def test_cpu_backend(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the CPU backend.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Check if CPU is enabled
        if not backend_config.cpu.enabled:
            return {
                "status": "disabled",
                "message": "CPU backend is disabled in configuration"
            }
        
        # Initialize CPU backend
        result = backend_manager.initialize_backend(BackendType.CPU)
        
        if not result:
            return {
                "status": "error",
                "message": "Failed to initialize CPU backend"
            }
        
        # Try a simple operation
        try:
            # Generate a simple text
            start_time = time.time()
            response = backend_manager.generate_text(
                prompt="Hello, world!",
                max_tokens=10,
                backend=BackendType.CPU
            )
            elapsed_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "CPU backend test successful",
                "elapsed_time": elapsed_time,
                "response": response
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"CPU backend test failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error testing CPU backend: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing CPU backend: {str(e)}"
        )

@router.post(
    "/test/failover",
    summary="Test backend failover",
    description="Test the backend failover mechanism"
)
async def test_failover(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the backend failover mechanism.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Check if auto failover is enabled
        if not backend_config.priority.auto_failover:
            return {
                "status": "disabled",
                "message": "Auto failover is disabled in configuration"
            }
        
        # Get primary and secondary backends
        primary = backend_config.get_primary_backend()
        secondary = backend_config.get_secondary_backend()
        
        if not secondary:
            return {
                "status": "error",
                "message": "No secondary backend available for failover test"
            }
        
        # Temporarily mark primary as unavailable
        mark_backend_unavailable(primary, error=Exception("Test failover"))
        
        # Try a simple operation (should use secondary)
        try:
            # Generate a simple text
            start_time = time.time()
            response = backend_manager.generate_text(
                prompt="Hello, world!",
                max_tokens=10
            )
            elapsed_time = time.time() - start_time
            
            # Reset backend status
            mark_backend_available(primary)
            
            # Verify the backend used
            used_backend = response.get("backend")
            
            if used_backend == secondary.value:
                return {
                    "status": "success",
                    "message": f"Failover test successful: Used {secondary} backend",
                    "elapsed_time": elapsed_time,
                    "response": response
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failover test failed: Used {used_backend} instead of {secondary}"
                }
                
        except Exception as e:
            # Reset backend status
            mark_backend_available(primary)
            
            return {
                "status": "error",
                "message": f"Failover test failed: {str(e)}"
            }
        
    except Exception as e:
        # Make sure to reset backend status
        try:
            mark_backend_available(backend_config.get_primary_backend())
        except:
            pass
            
        logger.error(f"Error testing failover: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing failover: {str(e)}"
        )