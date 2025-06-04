"""
GPU detection and automatic optimization module for the HANA AI Toolkit.

This module detects the available GPU hardware and automatically configures
the appropriate optimization settings for the detected architecture.
"""
import os
import logging
import importlib
import platform
import subprocess
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Define known GPU architectures and their module mappings
GPU_ARCHITECTURES = {
    "hopper": {
        "module": "gpu_utils_hopper",
        "class": "HopperOptimizer",
        "detector": "is_hopper_gpu",
        "models": ["H100"]
    },
    "turing": {
        "module": "gpu_utils_t4",
        "class": "T4Optimizer",
        "detector": "is_t4_gpu",
        "models": ["T4"]
    },
    "default": {
        "module": "gpu_utils",
        "class": "GPUProfiler",
        "models": ["default"]
    }
}

def detect_gpu_architecture() -> Tuple[str, Optional[str]]:
    """
    Detect the GPU architecture and specific model.
    
    Returns
    -------
    Tuple[str, Optional[str]]
        Architecture name and model name
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logger.info("No CUDA-capable GPU detected")
        return "none", None
    
    # Check for specific GPU architecture
    architecture = "default"
    model = None
    
    # Try each architecture detector
    for arch, config in GPU_ARCHITECTURES.items():
        if arch == "default":
            continue
            
        try:
            # Import the architecture-specific module
            module_name = f"hana_ai.api.{config['module']}"
            module = importlib.import_module(module_name)
            
            # Call the detector function
            if hasattr(module, config["detector"]):
                detector_func = getattr(module, config["detector"])
                if detector_func():
                    architecture = arch
                    
                    # Try to get specific model name
                    for i in range(torch.cuda.device_count()):
                        device_name = torch.cuda.get_device_name(i)
                        if any(model_name in device_name for model_name in config["models"]):
                            model = device_name
                            break
                    
                    break
        except ImportError:
            logger.debug(f"Module {module_name} not available")
        except Exception as e:
            logger.debug(f"Error checking for {arch} architecture: {str(e)}")
    
    # If no specific architecture detected, try to get model name from the first GPU
    if architecture == "default" and torch.cuda.device_count() > 0:
        model = torch.cuda.get_device_name(0)
    
    logger.info(f"Detected GPU architecture: {architecture}, model: {model}")
    return architecture, model

def get_gpu_optimizer():
    """
    Get the appropriate GPU optimizer for the detected hardware.
    
    Returns
    -------
    Any
        GPU optimizer instance
    """
    architecture, _ = detect_gpu_architecture()
    
    if architecture == "none":
        return None
    
    config = GPU_ARCHITECTURES.get(architecture, GPU_ARCHITECTURES["default"])
    module_name = f"hana_ai.api.{config['module']}"
    
    try:
        module = importlib.import_module(module_name)
        
        if architecture == "hopper":
            # For Hopper, use specific function that returns optimizer
            if hasattr(module, "detect_and_optimize_for_hopper"):
                return module.detect_and_optimize_for_hopper()
        elif architecture == "turing":
            # For T4/Turing, use specific function
            if hasattr(module, "optimize_for_t4"):
                return module.optimize_for_t4()
        else:
            # For default/other architectures, use the profiler
            if hasattr(module, config["class"]):
                class_obj = getattr(module, config["class"])
                return class_obj()
    except Exception as e:
        logger.warning(f"Failed to initialize GPU optimizer for {architecture}: {str(e)}")
    
    # Fallback to default GPU profiler
    try:
        from .gpu_utils import GPUProfiler
        return GPUProfiler()
    except Exception as e:
        logger.warning(f"Failed to initialize default GPU profiler: {str(e)}")
        return None

def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about available GPUs.
    
    Returns
    -------
    Dict[str, Any]
        GPU information
    """
    info = {
        "has_gpu": False,
        "count": 0,
        "architecture": "none",
        "models": [],
        "memory_gb": [],
        "cuda_version": None,
        "torch_version": None,
    }
    
    # Get PyTorch and CUDA versions if available
    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        if torch.version.cuda:
            info["cuda_version"] = torch.version.cuda
    
    # Check if CUDA is available
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return info
    
    # Set basic GPU info
    info["has_gpu"] = True
    info["count"] = torch.cuda.device_count()
    
    # Get architecture
    architecture, _ = detect_gpu_architecture()
    info["architecture"] = architecture
    
    # Get details for each GPU
    for i in range(info["count"]):
        device_name = torch.cuda.get_device_name(i)
        info["models"].append(device_name)
        
        # Get memory info
        if hasattr(torch.cuda, "get_device_properties"):
            props = torch.cuda.get_device_properties(i)
            memory_gb = round(props.total_memory / (1024**3), 2)
            info["memory_gb"].append(memory_gb)
    
    return info

# Initialize optimizer on module import
gpu_optimizer = get_gpu_optimizer()