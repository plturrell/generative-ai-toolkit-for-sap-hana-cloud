"""
NVIDIA T4 (Turing architecture) specific optimizations.

This module provides specialized optimizations for NVIDIA T4 GPUs, which use the
Turing architecture. It includes TensorRT configuration, mixed precision settings,
and tuned parameters specifically for T4 GPUs.
"""
import os
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

class T4Optimizer:
    """
    Specialized optimizations for NVIDIA T4 GPUs (Turing architecture).
    
    The T4 has 16GB of GDDR6 memory, 320 Tensor Cores, and 2,560 CUDA cores.
    This class provides tuned parameters and optimization techniques specific
    to this GPU type.
    """
    
    def __init__(self, 
                enable_tensorrt: bool = True,
                mixed_precision: bool = True,
                optimize_memory: bool = True,
                tensorrt_cache_dir: str = "/tmp/tensorrt_engines",
                tensorrt_precision: str = "fp16"):
        """
        Initialize the T4 optimizer.
        
        Parameters
        ----------
        enable_tensorrt : bool
            Whether to enable TensorRT optimization
        mixed_precision : bool
            Whether to enable mixed precision (FP16)
        optimize_memory : bool
            Whether to enable memory optimization techniques
        tensorrt_cache_dir : str
            Directory to cache TensorRT engines
        tensorrt_precision : str
            Precision for TensorRT engines (fp16, fp32, int8)
        """
        self.enable_tensorrt = enable_tensorrt
        self.mixed_precision = mixed_precision
        self.optimize_memory = optimize_memory
        self.tensorrt_cache_dir = tensorrt_cache_dir
        self.tensorrt_precision = tensorrt_precision
        
        # Initialize TensorRT if available
        self.tensorrt_initialized = False
        if TENSORRT_AVAILABLE and self.enable_tensorrt:
            try:
                self._init_tensorrt()
                self.tensorrt_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize TensorRT: {str(e)}")
        
        # Initialize torch settings if available
        if TORCH_AVAILABLE:
            self._init_torch_settings()
            
    def _init_tensorrt(self):
        """Initialize TensorRT with optimized settings for T4."""
        if not TENSORRT_AVAILABLE:
            return
            
        # Create TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.tensorrt_cache_dir, exist_ok=True)
        
        # TensorRT builder configuration optimized for T4
        self.builder_config = {
            "max_workspace_size": 1 << 30,  # 1GB
            "precision": self.tensorrt_precision,
            "max_batch_size": 16,  # T4 optimal batch size
            "builder_optimization_level": 3,
        }
        
        logger.info("TensorRT initialized for T4 GPU")
        
    def _init_torch_settings(self):
        """Initialize PyTorch settings optimized for T4."""
        if not TORCH_AVAILABLE:
            return
            
        # Enable CUDA graph capture for repeated operations
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to avoid OOM
        if self.optimize_memory:
            memory_fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "0.8"))
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(memory_fraction, i)
        
        # Enable mixed precision for T4
        if self.mixed_precision:
            # T4 has good FP16 support with Tensor Cores
            if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
                logger.info("Enabling mixed precision (FP16) for T4 GPU")
            else:
                logger.warning("PyTorch AMP not available, mixed precision disabled")
        
    def get_optimal_batch_size(self, model_size_mb: float) -> int:
        """
        Get the optimal batch size for a model on T4 GPU.
        
        Parameters
        ----------
        model_size_mb : float
            Model size in megabytes
            
        Returns
        -------
        int
            Optimal batch size
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 1
            
        # Get available GPU memory
        with torch.cuda.device(0):
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
            memory_fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "0.8"))
            available_memory = total_memory * memory_fraction
        
        # Calculate max batch size (conservative estimate)
        # For T4, we need to account for tensor cores and memory bandwidth
        max_batch = max(1, int(available_memory / (model_size_mb * 2.5)))
        
        # Adjust to be a power of 2 for better performance on T4
        batch_size = 2 ** int(max(0, min(10, (max_batch - 1).bit_length() - 1)))
        
        # T4-specific adjustment: limit to 16 for most cases
        batch_size = min(batch_size, 16)
        
        logger.info(f"Calculated optimal batch size for T4: {batch_size}")
        return batch_size
    
    def optimize_for_inference(self, model: Any) -> Any:
        """
        Apply T4-specific optimizations to a model for inference.
        
        Parameters
        ----------
        model : Any
            PyTorch model to optimize
            
        Returns
        -------
        Any
            Optimized model
        """
        if not TORCH_AVAILABLE:
            return model
            
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        try:
            # Apply TorchScript optimization (good for T4)
            optimized_model = torch.jit.script(model)
            optimized_model = torch.jit.freeze(optimized_model)
            
            # Apply torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                optimized_model = torch.compile(
                    optimized_model,
                    backend="inductor",
                    mode="reduce-overhead"
                )
                logger.info("Applied torch.compile optimization for T4")
            
            logger.info("Model optimized for T4 inference")
            return optimized_model
        except Exception as e:
            logger.warning(f"Failed to optimize model for T4: {str(e)}")
            return model
    
    def get_t4_config(self) -> Dict[str, Any]:
        """
        Get T4-specific configuration parameters.
        
        Returns
        -------
        Dict[str, Any]
            Configuration parameters
        """
        config = {
            "gpu_type": "t4",
            "architecture": "turing",
            "tensor_cores": True,
            "optimal_precision": "fp16",
            "optimal_batch_size": 16,
            "mixed_precision": self.mixed_precision,
            "tensorrt_enabled": self.tensorrt_initialized,
            "memory_optimization": self.optimize_memory,
            "recommended_settings": {
                "NVIDIA_TF32_OVERRIDE": 0,  # T4 doesn't support TF32
                "ENABLE_CUDA_GRAPHS": True,
                "ENABLE_KERNEL_FUSION": True,
                "ENABLE_FLASH_ATTENTION": True,
                "TENSORRT_PRECISION": "fp16",
                "CHECKPOINT_ACTIVATIONS": True,
            }
        }
        
        # Add current PyTorch settings if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            config["current_settings"] = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            }
            
            # Add device properties for each GPU
            config["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                config["devices"].append({
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 * 1024 * 1024), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
        
        return config

def is_t4_gpu() -> bool:
    """
    Check if the current GPU is a T4.
    
    Returns
    -------
    bool
        True if T4 GPU is detected
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
        
    try:
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            if "T4" in device_name:
                return True
                
        # Try using nvidia-smi as a fallback
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            return "T4" in result.stdout
            
        return False
    except Exception:
        return False

def optimize_for_t4() -> T4Optimizer:
    """
    Create and configure a T4 optimizer based on environment settings.
    
    Returns
    -------
    T4Optimizer
        Configured optimizer for T4
    """
    # Get settings from environment
    enable_tensorrt = os.environ.get("ENABLE_TENSORRT", "true").lower() == "true"
    mixed_precision = os.environ.get("ENABLE_MIXED_PRECISION", "true").lower() == "true"
    optimize_memory = os.environ.get("OPTIMIZE_MEMORY", "true").lower() == "true"
    tensorrt_cache_dir = os.environ.get("TENSORRT_CACHE_DIR", "/tmp/tensorrt_engines")
    tensorrt_precision = os.environ.get("TENSORRT_PRECISION", "fp16")
    
    # Create and return optimizer
    return T4Optimizer(
        enable_tensorrt=enable_tensorrt,
        mixed_precision=mixed_precision,
        optimize_memory=optimize_memory,
        tensorrt_cache_dir=tensorrt_cache_dir,
        tensorrt_precision=tensorrt_precision
    )