"""
Dependencies for the FastAPI application.
"""
import os
import logging
from typing import Any, Dict
from fastapi import Request, Depends, HTTPException
from hana_ml.dataframe import ConnectionContext
from langchain.llms.base import BaseLLM

logger = logging.getLogger(__name__)

from .auth import get_api_key
from .config import settings
from .gpu_utils import MultiGPUManager
from .gpu_utils_hopper import HopperOptimizer, detect_and_optimize_for_hopper

# Check for TensorRT availability
try:
    from .tensorrt_utils import get_tensorrt_optimizer
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT utilities not available. Some optimizations will be disabled.")

def get_connection_context(request: Request) -> ConnectionContext:
    """
    Get or create a database connection context.
    
    Uses connection pooling to reuse existing connections.
    
    Parameters
    ----------
    request : Request
        The FastAPI request object
        
    Returns
    -------
    ConnectionContext
        A connection to the HANA database
    """
    # Check if we already have a connection pool
    if not hasattr(request.app.state, "connection_pool"):
        request.app.state.connection_pool = {}
    
    # Create new connection if needed
    if len(request.app.state.connection_pool) < settings.CONNECTION_POOL_SIZE:
        try:
            # Try to create connection using userkey first if available
            if settings.HANA_USERKEY:
                conn = ConnectionContext(userkey=settings.HANA_USERKEY)
            else:
                # Fall back to direct credentials
                conn = ConnectionContext(
                    address=settings.HANA_HOST,
                    port=settings.HANA_PORT,
                    user=settings.HANA_USER,
                    password=settings.HANA_PASSWORD,
                    encrypt=True
                )
            
            # Add to pool
            conn_id = id(conn)
            request.app.state.connection_pool[conn_id] = conn
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database connection failed: {str(e)}"
            )
    
    # Reuse an existing connection
    try:
        for conn_id, conn in request.app.state.connection_pool.items():
            # Check if connection is still valid
            try:
                # Simple query to test connection
                conn.sql("SELECT 1 FROM DUMMY").collect()
                return conn
            except Exception as e:
                # Connection is invalid, remove it from pool
                logger.warning(f"Removing invalid connection from pool: {str(e)}")
                request.app.state.connection_pool.pop(conn_id, None)
                
        # If we get here, all connections in the pool are invalid
        # Create a new connection
        return get_connection_context(request)
        
    except Exception as e:
        logger.error(f"Failed to get database connection from pool: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )

def get_llm(
    request: Request,
    api_key: str = Depends(get_api_key),
    model_params: Dict[str, Any] = None
) -> BaseLLM:
    """
    Get a language model instance from SAP AI Core only with optimized GPU configuration.
    
    Parameters
    ----------
    request : Request
        The FastAPI request object
    api_key : str
        The validated API key
    model_params : Dict[str, Any], optional
        Override default model parameters
        
    Returns
    -------
    BaseLLM
        A language model instance from SAP AI Core with optimized GPU configuration
    """
    params = model_params or {}
    model = params.get("model", settings.DEFAULT_LLM_MODEL)
    
    # Get a unique task ID for this request
    task_id = f"llm_{request.state.request_id if hasattr(request.state, 'request_id') else id(request)}"
    
    # Ensure model name is a valid SAP AI Core model
    if not model.startswith("sap-ai-core"):
        logger.warning(f"Non-SAP model requested: {model}. Defaulting to {settings.DEFAULT_LLM_MODEL}")
        model = settings.DEFAULT_LLM_MODEL
    
    temperature = params.get("temperature", settings.DEFAULT_LLM_TEMPERATURE)
    max_tokens = params.get("max_tokens", settings.DEFAULT_LLM_MAX_TOKENS)
    
    # Configure advanced GPU settings using MultiGPUManager
    gpu_config = {}
    if settings.ENABLE_GPU_ACCELERATION:
        # Initialize MultiGPUManager if not already present in app state
        if not hasattr(request.app.state, "gpu_manager"):
            try:
                request.app.state.gpu_manager = MultiGPUManager(
                    strategy="auto",
                    memory_fraction=settings.CUDA_MEMORY_FRACTION,
                    enable_mixed_precision=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize GPU manager: {str(e)}")
                request.app.state.gpu_manager = None
        
        # Get optimal device for this task
        if hasattr(request.app.state, "gpu_manager") and request.app.state.gpu_manager:
            try:
                # Estimate memory requirements based on model size and max_tokens
                # This is a rough estimate - in production, use more precise metrics
                memory_requirement = 2000  # Base memory in MB
                memory_requirement += max_tokens * 0.5  # Add per-token estimate
                
                # Get optimal device
                device_id = request.app.state.gpu_manager.get_optimal_device(
                    task_id=task_id,
                    memory_requirement=memory_requirement
                )
                
                if device_id >= 0:  # Valid GPU device
                    gpu_config = {
                        "use_gpu": True,
                        "gpu_memory_fraction": settings.CUDA_MEMORY_FRACTION,
                        "device_id": device_id,
                        "use_mixed_precision": True
                    }
                    
                    # Set environment variables for this request
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                    
                    logger.info(f"Using GPU device {device_id} for LLM task {task_id}")
                else:
                    logger.warning(f"No suitable GPU found for task {task_id}, using CPU")
                    gpu_config = {"use_gpu": False}
            except Exception as e:
                logger.warning(f"Error in GPU device selection: {str(e)}")
                gpu_config = {
                    "use_gpu": True,
                    "gpu_memory_fraction": settings.CUDA_MEMORY_FRACTION
                }
        else:
            # Default GPU config if MultiGPUManager not available
            gpu_config = {
                "use_gpu": True,
                "gpu_memory_fraction": settings.CUDA_MEMORY_FRACTION
            }
    
    try:
        # Only use SAP GenAI Hub SDK - no fallback to external services
        try:
            from gen_ai_hub.proxy.langchain import init_llm
            
            # First, check for Hopper architecture (H100) optimizations
            is_hopper, hopper_specific_args = detect_and_optimize_for_hopper()
            
            # Add performance optimization settings
            optimization_config = {
                # Enable tensor cores if available (A100, H100, etc.)
                "enable_tensor_cores": True,
                # Enable tensor parallelism if model supports it
                "tensor_parallel": True if gpu_config.get("use_gpu", False) else False,
                # Optimize kernel launches
                "optimize_kernels": True,
                # Use flash attention if available
                "use_flash_attention": True,
                # Enable FP8 if available (H100)
                "enable_fp8": is_hopper,
                # For large models, enable checkpoint activation memory optimization
                "checkpoint_activations": max_tokens > 1000,
            }
            
            # Add TensorRT optimization if available
            tensorrt_config = {}
            if TENSORRT_AVAILABLE and settings.ENABLE_TENSORRT:
                try:
                    # Get TensorRT optimizer with customized settings from config
                    tensorrt_optimizer = get_tensorrt_optimizer()
                    
                    # Configure TensorRT cache directory
                    if not os.path.exists(settings.TENSORRT_CACHE_DIR):
                        os.makedirs(settings.TENSORRT_CACHE_DIR, exist_ok=True)
                    tensorrt_optimizer.cache_dir = settings.TENSORRT_CACHE_DIR
                    
                    # Set precision based on settings and hardware capability
                    precision = settings.TENSORRT_PRECISION
                    if precision == "fp16" and not is_hopper:
                        # Check if the GPU supports FP16
                        if not hasattr(tensorrt_optimizer, "engines") or not tensorrt_optimizer.engines:
                            # Default to FP32 if FP16 not supported
                            precision = "fp32"
                            logger.warning("FP16 not supported on this GPU, falling back to FP32")
                    
                    # Define input shapes for the model
                    # This is a simplified version - in production, use model-specific configurations
                    input_shapes = {
                        "input_ids": [max_tokens],
                        "attention_mask": [max_tokens]
                    }
                    
                    # For models with additional inputs, add them here
                    if "gpt" in model.lower() or "llama" in model.lower():
                        input_shapes["position_ids"] = [max_tokens]
                    
                    # Build TensorRT configuration with settings
                    tensorrt_config = {
                        "use_tensorrt": True,
                        "tensorrt_precision": precision,
                        "tensorrt_input_shapes": input_shapes,
                        "tensorrt_max_batch_size": settings.TENSORRT_MAX_BATCH_SIZE,
                        "tensorrt_workspace_size": settings.TENSORRT_WORKSPACE_SIZE_MB * (1024 * 1024),  # Convert to bytes
                        "tensorrt_dynamic_shapes": True,
                        "tensorrt_builder_optimization_level": settings.TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
                        "tensorrt_cache_dir": settings.TENSORRT_CACHE_DIR,
                    }
                    
                    # Add model type hint if possible (helps TensorRT optimizer)
                    if "bert" in model.lower():
                        tensorrt_config["model_type"] = "encoder"
                    elif "gpt" in model.lower() or "llama" in model.lower():
                        tensorrt_config["model_type"] = "decoder"
                    elif "t5" in model.lower() or "bart" in model.lower():
                        tensorrt_config["model_type"] = "encoder-decoder"
                    
                    logger.info(f"TensorRT optimization enabled for LLM with precision {precision}")
                except Exception as e:
                    logger.warning(f"Failed to initialize TensorRT: {str(e)}")
            
            # If we're on Hopper architecture, apply H100-specific optimizations
            if is_hopper:
                # Create a Hopper optimizer for additional tuning
                hopper_optimizer = HopperOptimizer()
                
                # Get optimized dtype configuration for H100
                dtype_config = hopper_optimizer.optimize_dtype_config()
                
                # Add Hopper-specific args to the optimization config
                optimization_config.update(hopper_specific_args)
                
                logger.info("Applied H100 Hopper-specific optimizations")
            
            # Merge all configs
            combined_config = {**gpu_config, **optimization_config, **tensorrt_config}
            
            # Initialize LLM with all optimizations
            llm = init_llm(
                model, 
                temperature=temperature, 
                max_tokens=max_tokens,
                **combined_config
            )
            
            # Apply post-initialization optimizations if on Hopper
            if is_hopper and hasattr(llm, "model"):
                try:
                    # Apply TensorRT optimization for inference if available
                    if TENSORRT_AVAILABLE and settings.ENABLE_TENSORRT:
                        # Get model name for caching
                        model_name = model.replace("/", "_").replace("-", "_")
                        
                        # Apply TensorRT optimizations to the underlying model
                        hopper_optimizer.optimize_inference(
                            model=llm.model,
                            model_name=model_name,
                            input_shapes=input_shapes
                        )
                except Exception as e:
                    logger.warning(f"Post-initialization optimization failed: {str(e)}")
            
            # Log configuration for debugging
            if settings.DEVELOPMENT_MODE:
                logger.debug(f"LLM initialized with config: {combined_config}")
                
            return llm
            
        except ImportError:
            logger.error("SAP GenAI Hub SDK is required but not installed")
            raise HTTPException(
                status_code=500,
                detail="SAP GenAI Hub SDK is required for this application. Please install it using 'pip install generative-ai-hub-sdk[all]'"
            )
            
    except Exception as e:
        logger.error(f"Failed to initialize SAP AI Core language model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"SAP AI Core language model initialization failed: {str(e)}"
        )