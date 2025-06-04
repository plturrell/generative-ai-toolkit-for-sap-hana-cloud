"""
T4-specific validation module for NVIDIA T4 GPUs.

This module provides validation functions to ensure the T4 GPU is properly
configured and optimized.
"""
import os
import logging
from typing import Dict, Any

import torch

from .config import settings
from .gpu_utils_t4 import is_t4_gpu, optimize_for_t4

logger = logging.getLogger(__name__)

def validate_t4_gpu() -> Dict[str, Any]:
    """
    Perform validation specific to NVIDIA T4 GPUs.
    
    This function checks:
    1. T4 GPU detection
    2. CUDA and PyTorch configuration
    3. TensorRT availability
    4. Mixed precision support
    5. Optimal environment settings

    Returns
    -------
    Dict[str, Any]
        Validation results specific to T4 GPUs
    """
    results = {
        "status": "ok",
        "message": "T4 GPU checks completed successfully",
        "details": {}
    }
    
    # Check if GPU acceleration is enabled
    if not settings.ENABLE_GPU_ACCELERATION:
        results["status"] = "warning"
        results["message"] = "GPU acceleration is disabled"
        results["details"]["gpu_acceleration"] = {
            "status": "warning",
            "message": "GPU acceleration is disabled in settings"
        }
        return results
    
    # Check if PyTorch is available
    if not torch.cuda.is_available():
        results["status"] = "error"
        results["message"] = "CUDA not available"
        results["details"]["cuda"] = {
            "status": "error",
            "message": "CUDA not available"
        }
        return results
    
    # Check if T4 GPU is detected
    t4_detected = is_t4_gpu()
    if not t4_detected:
        results["status"] = "error"
        results["message"] = "NVIDIA T4 GPU not detected"
        results["details"]["t4_detection"] = {
            "status": "error",
            "message": "NVIDIA T4 GPU not detected"
        }
        return results
    
    # Get T4 GPU information
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "T4" in device_name:
            props = torch.cuda.get_device_properties(i)
            results["details"]["t4_gpu"] = {
                "status": "ok",
                "device_id": i,
                "name": device_name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multiprocessor_count": props.multi_processor_count
            }
            break
    
    # Check TensorRT availability
    try:
        import tensorrt as trt
        results["details"]["tensorrt"] = {
            "status": "ok",
            "version": trt.__version__,
            "message": "TensorRT available for T4 optimization"
        }
    except ImportError:
        results["status"] = "warning"
        results["message"] = "TensorRT not installed - recommended for T4 GPUs"
        results["details"]["tensorrt"] = {
            "status": "warning",
            "message": "TensorRT not installed - recommended for T4 GPUs"
        }
    
    # Check mixed precision support (critical for T4 performance)
    try:
        mixed_precision_ok = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
        if mixed_precision_ok:
            results["details"]["mixed_precision"] = {
                "status": "ok",
                "message": "Mixed precision (FP16) support available"
            }
        else:
            results["status"] = "warning"
            results["message"] = "Mixed precision not available - performance will be reduced"
            results["details"]["mixed_precision"] = {
                "status": "warning",
                "message": "Mixed precision not available - performance will be reduced"
            }
    except Exception as e:
        results["details"]["mixed_precision"] = {
            "status": "warning",
            "message": f"Error checking mixed precision: {str(e)}"
        }
    
    # Check environment variables for T4 optimization
    env_vars = {
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "Not set"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "Not set"),
        "ENABLE_TENSOR_CORES": os.environ.get("ENABLE_TENSOR_CORES", "Not set"),
        "ENABLE_FLASH_ATTENTION": os.environ.get("ENABLE_FLASH_ATTENTION", "Not set"),
        "ENABLE_TENSORRT": os.environ.get("ENABLE_TENSORRT", "Not set"),
        "TENSORRT_PRECISION": os.environ.get("TENSORRT_PRECISION", "Not set"),
        "CHECKPOINT_ACTIVATIONS": os.environ.get("CHECKPOINT_ACTIVATIONS", "Not set")
    }
    
    # Validate environment variables
    env_ok = True
    missing_vars = []
    for key, value in env_vars.items():
        if value == "Not set":
            missing_vars.append(key)
            env_ok = False
    
    # Set environment status
    if env_ok:
        results["details"]["environment"] = {
            "status": "ok",
            "variables": env_vars,
            "message": "T4 environment variables properly configured"
        }
    else:
        results["status"] = "warning"
        results["message"] = "Some T4 optimization environment variables not set"
        results["details"]["environment"] = {
            "status": "warning",
            "variables": env_vars,
            "missing": missing_vars,
            "message": "Some T4 optimization environment variables not set"
        }
    
    # Verify T4 optimizer is working
    try:
        t4_optimizer = optimize_for_t4()
        t4_config = t4_optimizer.get_t4_config()
        
        results["details"]["t4_optimizer"] = {
            "status": "ok",
            "message": "T4 optimizer properly configured",
            "config": t4_config
        }
    except Exception as e:
        results["status"] = "warning"
        results["message"] = f"T4 optimizer error: {str(e)}"
        results["details"]["t4_optimizer"] = {
            "status": "warning",
            "message": f"Error initializing T4 optimizer: {str(e)}"
        }
    
    # Verify GPU computation
    try:
        # Simple T4 tensor test with tensor cores (FP16)
        with torch.cuda.amp.autocast():
            a = torch.rand(1024, 1024, device="cuda")
            b = torch.rand(1024, 1024, device="cuda")
            c = torch.matmul(a, b)
            result = c.sum().item()
        
        results["details"]["computation_test"] = {
            "status": "ok",
            "message": "GPU computation test passed"
        }
    except Exception as e:
        results["status"] = "error"
        results["message"] = f"GPU computation test failed: {str(e)}"
        results["details"]["computation_test"] = {
            "status": "error",
            "message": f"GPU computation test failed: {str(e)}"
        }
    
    return results