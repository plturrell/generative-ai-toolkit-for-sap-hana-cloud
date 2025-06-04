"""
GPU utilities for the SAP HANA AI Toolkit API.

This module provides functions for detecting and managing GPU resources,
as well as applying GPU optimizations for deep learning models.
"""
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

from .tensorrt_utils import (
    TensorRTOptimizer, 
    TRTConfig, 
    is_tensorrt_available, 
    get_tensorrt_version
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUManager:
    """Manage GPU resources and optimizations."""
    
    def __init__(
        self,
        enable_gpu: bool = True,
        enable_tensorrt: bool = True,
        memory_fraction: float = 0.9,
        device_id: Optional[int] = None
    ):
        """
        Initialize GPU manager.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            enable_tensorrt: Whether to enable TensorRT optimization
            memory_fraction: Fraction of GPU memory to use
            device_id: Specific GPU device ID to use
        """
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.enable_tensorrt = enable_tensorrt and is_tensorrt_available()
        self.memory_fraction = max(0.1, min(1.0, memory_fraction))
        
        # Device selection
        if self.enable_gpu:
            if device_id is not None and device_id < torch.cuda.device_count():
                self.device_id = device_id
            else:
                self.device_id = 0
                
            # Set device
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f"cuda:{self.device_id}")
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device_id)
        else:
            self.device_id = -1
            self.device = torch.device("cpu")
            
        # Initialize TensorRT if available
        if self.enable_tensorrt:
            self.tensorrt_config = TRTConfig(
                fp16_mode=True,
                int8_mode=False,
                max_workspace_size=1 << 30,
                cache_dir=os.path.join(os.path.dirname(__file__), "tensorrt_engines")
            )
            self.tensorrt_optimizer = TensorRTOptimizer(self.tensorrt_config)
        else:
            self.tensorrt_optimizer = None
            
        logger.info(f"GPU Manager initialized: GPU={self.enable_gpu}, TensorRT={self.enable_tensorrt}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get information about available GPUs.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            "available": self.enable_gpu,
            "count": 0,
            "names": [],
            "memory_total": [],
            "memory_used": [],
            "utilization": [],
            "tensorrt_available": self.enable_tensorrt,
            "hopper_features": {}
        }
        
        if not self.enable_gpu:
            return info
            
        # Get GPU count and names
        info["count"] = torch.cuda.device_count()
        
        for i in range(info["count"]):
            # Get device name
            info["names"].append(torch.cuda.get_device_name(i))
            
            # Get memory information
            try:
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)  # MB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)  # MB
                
                info["memory_total"].append(int(memory_total))
                info["memory_used"].append(int(memory_allocated))
                
                # Calculate utilization
                if memory_total > 0:
                    utilization = memory_allocated / memory_total * 100
                else:
                    utilization = 0
                    
                info["utilization"].append(float(utilization))
            except Exception as e:
                logger.warning(f"Error getting memory info for GPU {i}: {str(e)}")
                info["memory_total"].append(0)
                info["memory_used"].append(0)
                info["utilization"].append(0.0)
        
        # Check for Hopper architecture features
        for name in info["names"]:
            if "H100" in name or "H800" in name or "H10" in name:
                info["hopper_features"] = {
                    "fp8": True,
                    "transformer_engine": True,
                    "flash_attention": True
                }
                break
        
        return info
    
    def get_tensorrt_info(self) -> Dict[str, Any]:
        """
        Get information about TensorRT.
        
        Returns:
            Dictionary with TensorRT information
        """
        info = {
            "available": self.enable_tensorrt,
            "version": get_tensorrt_version(),
            "optimized_models": [],
            "supported_precisions": []
        }
        
        if not self.enable_tensorrt:
            return info
            
        # Get optimized models
        if self.tensorrt_optimizer:
            info["optimized_models"] = list(self.tensorrt_optimizer.engines.keys())
            
        # Get supported precisions
        info["supported_precisions"] = ["FP32"]
        
        if self.enable_gpu:
            # Check for FP16 support
            if torch.cuda.get_device_capability(self.device_id)[0] >= 6:
                info["supported_precisions"].append("FP16")
                
            # Check for INT8 support
            if torch.cuda.get_device_capability(self.device_id)[0] >= 6:
                info["supported_precisions"].append("INT8")
                
            # Check for FP8 support (Hopper)
            if torch.cuda.get_device_capability(self.device_id)[0] >= 9:
                info["supported_precisions"].append("FP8")
        
        return info
    
    def optimize_model(
        self, 
        model: torch.nn.Module, 
        model_name: str,
        sample_input: Dict[str, torch.Tensor]
    ) -> Tuple[torch.nn.Module, bool]:
        """
        Optimize a PyTorch model for inference.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            sample_input: Sample input tensors
            
        Returns:
            Tuple of (optimized model, whether TensorRT was used)
        """
        if not self.enable_gpu:
            return model, False
            
        # Move model to GPU
        model = model.to(self.device)
        model.eval()
        
        # Try TensorRT optimization
        if self.enable_tensorrt and self.tensorrt_optimizer:
            # Prepare input shapes
            input_shapes = {name: list(tensor.shape) for name, tensor in sample_input.items()}
            
            # Try to optimize with TensorRT
            engine = self.tensorrt_optimizer.optimize_torch_model(model, model_name, input_shapes)
            
            if engine:
                logger.info(f"Model {model_name} optimized with TensorRT")
                return engine, True
                
        # If TensorRT optimization failed or not enabled, use PyTorch with CUDA
        logger.info(f"Using standard PyTorch optimization for {model_name}")
        
        # Apply PyTorch optimizations
        if torch.cuda.is_available():
            try:
                # Try to compile with torch.compile (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    model = torch.compile(model)
                    logger.info(f"Model {model_name} compiled with torch.compile")
                # Otherwise try TorchScript
                else:
                    # Prepare example inputs for tracing
                    example_inputs = tuple(sample_input.values())
                    
                    # Try to create TorchScript model
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, example_inputs)
                        traced_model = torch.jit.freeze(traced_model)
                        model = traced_model
                        
                    logger.info(f"Model {model_name} compiled with TorchScript")
            except Exception as e:
                logger.warning(f"Failed to compile model {model_name}: {str(e)}")
        
        return model, False
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        sample_input: Dict[str, torch.Tensor],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark a model with and without optimizations.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            sample_input: Sample input tensors
            num_iterations: Number of iterations to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.enable_gpu:
            return {
                "model_name": model_name,
                "pytorch_time_ms": 0,
                "tensorrt_time_ms": 0,
                "speedup": 1.0,
                "error": "GPU not available"
            }
            
        if self.enable_tensorrt and self.tensorrt_optimizer:
            return self.tensorrt_optimizer.benchmark(
                model, 
                model_name, 
                sample_input, 
                num_iterations=num_iterations
            )
        else:
            # Move model to GPU
            model = model.to(self.device)
            model.eval()
            
            # Move inputs to GPU
            cuda_inputs = {name: tensor.to(self.device) for name, tensor in sample_input.items()}
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    if len(cuda_inputs) == 1:
                        _ = model(next(iter(cuda_inputs.values())))
                    else:
                        _ = model(**cuda_inputs)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    if len(cuda_inputs) == 1:
                        _ = model(next(iter(cuda_inputs.values())))
                    else:
                        _ = model(**cuda_inputs)
            
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time) / num_iterations
            
            return {
                "model_name": model_name,
                "pytorch_time_ms": elapsed_time,
                "tensorrt_time_ms": None,
                "speedup": 1.0,
                "iterations": num_iterations
            }