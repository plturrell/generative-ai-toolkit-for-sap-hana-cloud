"""
T4 GPU Optimization Utilities for SAP HANA Cloud Generative AI Toolkit

This module provides specialized optimizations for NVIDIA T4 GPUs, including
optimized TensorRT settings, dynamic batch sizing, and memory management
techniques specific to the T4 architecture.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import math

import torch
import numpy as np

# Import TensorRT utilities
from api.tensorrt_utils import (
    TRTConfig, 
    TensorRTEngine, 
    TensorRTOptimizer, 
    is_tensorrt_available
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# T4 GPU specifications
T4_SPECS = {
    "compute_capability": "7.5",
    "cuda_cores": 2560,
    "tensor_cores": 320,
    "memory": 16 * 1024 * 1024 * 1024,  # 16GB in bytes
    "memory_bandwidth": 320,  # GB/s
    "fp32_tflops": 8.1,
    "fp16_tflops": 65,
    "int8_tops": 130,
    "tdp": 70,  # Watts
}

# T4 optimized settings
DEFAULT_T4_CONFIG = {
    "fp16_mode": True,
    "int8_mode": False,  # Can be enabled with calibration
    "max_workspace_size": 4 * (1 << 30),  # 4GB
    "dynamic_batch_size": True,
    "enable_profiling": False,
    "enable_tensor_cores": True,
    "enable_sparse": False,
    "dla_enabled": False,
    "precision": "fp16",  # One of "fp32", "fp16", "int8"
    "memory_fraction": 0.8,  # Fraction of GPU memory to use
    "max_batch_size": 64,
    "optimization_level": 3,  # 0-5, higher is more aggressive
}

class T4GPUConfig:
    """Configuration settings optimized for T4 GPU."""
    
    def __init__(self, **kwargs):
        """
        Initialize T4 GPU configuration.
        
        Args:
            **kwargs: Override default T4 configuration
        """
        self.config = DEFAULT_T4_CONFIG.copy()
        
        # Override with custom settings
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration option: {key}")
                
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Log configuration
        logger.info(f"T4 GPU configuration: {json.dumps(self.config, indent=2)}")
    
    def _apply_env_overrides(self):
        """Apply configuration overrides from environment variables."""
        for key in self.config:
            env_key = f"T4_GPU_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Convert to appropriate type
                if isinstance(self.config[key], bool):
                    self.config[key] = value.lower() in ("true", "1", "yes")
                elif isinstance(self.config[key], int):
                    self.config[key] = int(value)
                elif isinstance(self.config[key], float):
                    self.config[key] = float(value)
                else:
                    self.config[key] = value
                    
                logger.info(f"Applied environment override: {key}={self.config[key]}")
    
    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def to_dict(self):
        """Get configuration as dictionary."""
        return self.config.copy()
    
    def to_trt_config(self) -> TRTConfig:
        """Convert to TensorRT configuration."""
        # Extract TensorRT specific settings
        trt_settings = {
            "fp16_mode": self.config["fp16_mode"],
            "int8_mode": self.config["int8_mode"],
            "max_workspace_size": self.config["max_workspace_size"],
            "max_batch_size": self.config["max_batch_size"],
            "strict_type_constraints": False,
            "enable_sparsity": self.config["enable_sparse"],
        }
        
        return TRTConfig(**trt_settings)


class T4MemoryManager:
    """
    Memory manager optimized for T4 GPU.
    """
    
    def __init__(self, config: Optional[T4GPUConfig] = None):
        """
        Initialize memory manager.
        
        Args:
            config: T4 GPU configuration
        """
        self.config = config or T4GPUConfig()
        self.reserved_memory = 0
        self._setup_memory()
    
    def _setup_memory(self):
        """Setup GPU memory management."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Memory management disabled.")
            return
            
        # Set memory fraction
        memory_fraction = self.config.get("memory_fraction", 0.8)
        
        try:
            # Only import if CUDA is available
            import gc
            import torch.cuda
            
            # Run garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Get total memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate memory to reserve
            self.reserved_memory = int(total_memory * (1 - memory_fraction))
            
            logger.info(f"T4 GPU memory: {total_memory / (1024**3):.2f} GB")
            logger.info(f"Memory fraction: {memory_fraction}, Reserved: {self.reserved_memory / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"Error setting up memory management: {str(e)}")
    
    def get_free_memory(self) -> int:
        """
        Get free GPU memory in bytes.
        
        Returns:
            Free memory in bytes
        """
        if not torch.cuda.is_available():
            return 0
            
        try:
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free_reserved = reserved - allocated
            free_total = torch.cuda.get_device_properties(0).total_memory - allocated - self.reserved_memory
            
            return free_reserved + free_total
        except Exception as e:
            logger.error(f"Error getting free memory: {str(e)}")
            return 0
    
    def calculate_optimal_batch_size(
        self, 
        input_size_per_sample: int, 
        output_size_per_sample: int,
        processing_size_per_sample: Optional[int] = None,
        min_batch: int = 1,
        max_batch: Optional[int] = None
    ) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            input_size_per_sample: Memory required for one input sample (bytes)
            output_size_per_sample: Memory required for one output sample (bytes)
            processing_size_per_sample: Additional memory for processing (bytes)
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return min_batch
        
        # If processing size not provided, estimate as 5x the input size
        if processing_size_per_sample is None:
            processing_size_per_sample = 5 * input_size_per_sample
        
        # Default max batch from config if not specified
        if max_batch is None:
            max_batch = self.config.get("max_batch_size", 64)
        
        # Total memory per sample
        total_per_sample = input_size_per_sample + output_size_per_sample + processing_size_per_sample
        
        # Get available memory (with safety factor)
        available_memory = self.get_free_memory() * 0.9
        
        # Calculate maximum possible batch size
        max_possible_batch = int(available_memory / total_per_sample)
        
        # Ensure within bounds
        batch_size = max(min_batch, min(max_possible_batch, max_batch))
        
        logger.info(f"Calculated optimal batch size: {batch_size} " +
                   f"(Free memory: {available_memory / (1024**3):.2f} GB, " +
                   f"Per sample: {total_per_sample / (1024**2):.2f} MB)")
        
        return batch_size
    
    def optimize_for_embedding_model(
        self, 
        model_name: str,
        max_sequence_length: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.float16
    ) -> int:
        """
        Optimize batch size for embedding model.
        
        Args:
            model_name: Name of the model
            max_sequence_length: Maximum sequence length
            embedding_dim: Embedding dimension
            dtype: Data type for computation
            
        Returns:
            Optimal batch size for the model
        """
        # Calculate memory requirements
        bytes_per_element = 2 if dtype == torch.float16 else 4
        
        # Input size: tokens (sequence length) for each sample
        input_size = max_sequence_length * bytes_per_element
        
        # Output size: embedding vector for each sample
        output_size = embedding_dim * bytes_per_element
        
        # Processing size: depends on model architecture, use model-specific estimates
        if "all-MiniLM-L6" in model_name:
            # Small model, relatively lightweight
            processing_size = input_size * 10
        elif "distilbert" in model_name:
            # Medium sized model
            processing_size = input_size * 20
        elif "mpnet" in model_name or "roberta" in model_name:
            # Larger models
            processing_size = input_size * 30
        else:
            # Default case - conservative estimate
            processing_size = input_size * 25
        
        # Add model overhead (weights, etc.)
        if "base" in model_name:
            model_overhead = 500 * 1024 * 1024  # ~500MB for base models
        elif "large" in model_name:
            model_overhead = 1.5 * 1024 * 1024 * 1024  # ~1.5GB for large models
        else:
            model_overhead = 300 * 1024 * 1024  # ~300MB default
        
        # Temporarily reserve model overhead
        self.reserved_memory += model_overhead
        
        try:
            # Calculate batch size
            batch_size = self.calculate_optimal_batch_size(
                input_size, 
                output_size, 
                processing_size,
                min_batch=1,
                max_batch=self.config.get("max_batch_size", 64)
            )
        finally:
            # Restore reserved memory
            self.reserved_memory -= model_overhead
            
        return batch_size


class T4TensorRTOptimizer(TensorRTOptimizer):
    """
    TensorRT optimizer specifically tuned for T4 GPU.
    """
    
    def __init__(self, t4_config: Optional[T4GPUConfig] = None):
        """
        Initialize T4 TensorRT optimizer.
        
        Args:
            t4_config: T4 GPU configuration
        """
        self.t4_config = t4_config or T4GPUConfig()
        self.memory_manager = T4MemoryManager(self.t4_config)
        
        # Convert T4 config to TensorRT config
        trt_config = self.t4_config.to_trt_config()
        
        # Initialize parent class
        super().__init__(trt_config)
        
        # Initialize optimizations for T4
        self._init_t4_optimizations()
    
    def _init_t4_optimizations(self):
        """Initialize T4-specific optimizations."""
        if not is_tensorrt_available():
            logger.warning("TensorRT not available. T4 optimizations disabled.")
            return
            
        try:
            import tensorrt as trt
            
            # Set precision flags based on configuration
            precision = self.t4_config.get("precision", "fp16").lower()
            
            if precision == "fp16":
                self.config.fp16_mode = True
                self.config.int8_mode = False
            elif precision == "int8":
                self.config.fp16_mode = True  # Keep FP16 enabled for layers that don't support INT8
                self.config.int8_mode = True
            else:  # fp32
                self.config.fp16_mode = False
                self.config.int8_mode = False
                
            logger.info(f"T4 TensorRT precision: {precision}")
            
        except Exception as e:
            logger.error(f"Error initializing T4 optimizations: {str(e)}")
    
    def optimize_embedding_model(
        self,
        model: Any,
        model_name: str,
        max_sequence_length: int,
        embedding_dim: int,
        batch_size: Optional[int] = None,
        dynamic_axes: bool = True
    ) -> Optional[TensorRTEngine]:
        """
        Optimize embedding model specifically for T4 GPU.
        
        Args:
            model: PyTorch model
            model_name: Name of the model for caching
            max_sequence_length: Maximum sequence length
            embedding_dim: Embedding dimension
            batch_size: Batch size or None for dynamic batch sizing
            dynamic_axes: Whether to use dynamic axes for variable batch size
            
        Returns:
            TensorRTEngine: Optimized TensorRT engine or None if optimization failed
        """
        if not is_tensorrt_available():
            logger.warning("TensorRT not available. Running with standard PyTorch.")
            return None
            
        # Determine optimal batch size if not provided
        if batch_size is None:
            batch_size = self.memory_manager.optimize_for_embedding_model(
                model_name,
                max_sequence_length,
                embedding_dim,
                dtype=torch.float16 if self.config.fp16_mode else torch.float32
            )
        
        # Prepare input shapes
        if dynamic_axes:
            # Dynamic batch dimension
            input_shapes = {
                "input_ids": [-1, max_sequence_length],
                "attention_mask": [-1, max_sequence_length],
                "token_type_ids": [-1, max_sequence_length]
            }
        else:
            # Fixed batch dimension
            input_shapes = {
                "input_ids": [batch_size, max_sequence_length],
                "attention_mask": [batch_size, max_sequence_length],
                "token_type_ids": [batch_size, max_sequence_length]
            }
        
        # Prepare output shapes
        output_shapes = {"embeddings": [batch_size, embedding_dim]}
        
        # Optimize model
        return self.optimize_torch_model(model, model_name, input_shapes, output_shapes)
    
    def benchmark_embedding_model(
        self,
        model: Any,
        model_name: str,
        max_sequence_length: int,
        embedding_dim: int,
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        num_iterations: int = 50,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark embedding model with different batch sizes.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            max_sequence_length: Maximum sequence length
            embedding_dim: Embedding dimension
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "model_name": model_name,
            "max_sequence_length": max_sequence_length,
            "embedding_dim": embedding_dim,
            "batch_results": [],
            "optimal_batch_size": None,
            "optimal_throughput": 0,
            "optimal_latency": float('inf'),
            "tensorrt_speedup": 0
        }
        
        # Move model to GPU
        model = model.cuda().eval()
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            # Create sample inputs
            inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, max_sequence_length), device="cuda"),
                "attention_mask": torch.ones(batch_size, max_sequence_length, device="cuda"),
                "token_type_ids": torch.zeros(batch_size, max_sequence_length, device="cuda")
            }
            
            # Benchmark PyTorch
            torch.cuda.synchronize()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = model(**inputs)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(**inputs)
                    
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / num_iterations * 1000  # ms
            
            # Optimize with TensorRT
            engine = self.optimize_embedding_model(
                model, 
                f"{model_name}_batch{batch_size}", 
                max_sequence_length, 
                embedding_dim, 
                batch_size,
                dynamic_axes=False  # Use fixed batch size for better performance
            )
            
            if engine:
                # Warmup
                for _ in range(warmup_iterations):
                    _ = engine.infer(inputs)
                    
                # Benchmark
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(num_iterations):
                    _ = engine.infer(inputs)
                    
                torch.cuda.synchronize()
                tensorrt_time = (time.time() - start_time) / num_iterations * 1000  # ms
                
                speedup = pytorch_time / tensorrt_time if tensorrt_time > 0 else float('inf')
                
                # Calculate throughput (samples/second)
                pytorch_throughput = batch_size * 1000 / pytorch_time
                tensorrt_throughput = batch_size * 1000 / tensorrt_time
                
                batch_result = {
                    "batch_size": batch_size,
                    "pytorch_latency_ms": pytorch_time,
                    "tensorrt_latency_ms": tensorrt_time,
                    "speedup": speedup,
                    "pytorch_throughput": pytorch_throughput,
                    "tensorrt_throughput": tensorrt_throughput
                }
            else:
                # TensorRT optimization failed
                batch_result = {
                    "batch_size": batch_size,
                    "pytorch_latency_ms": pytorch_time,
                    "tensorrt_latency_ms": None,
                    "speedup": 1.0,
                    "pytorch_throughput": batch_size * 1000 / pytorch_time,
                    "tensorrt_throughput": None,
                    "error": "TensorRT optimization failed"
                }
                
            results["batch_results"].append(batch_result)
            
            # Update optimal batch size based on throughput
            if engine and batch_result["tensorrt_throughput"] > results["optimal_throughput"]:
                results["optimal_batch_size"] = batch_size
                results["optimal_throughput"] = batch_result["tensorrt_throughput"]
                results["optimal_latency"] = batch_result["tensorrt_latency_ms"]
                results["tensorrt_speedup"] = batch_result["speedup"]
        
        return results


# Utility functions
def get_t4_gpu_info() -> Dict[str, Any]:
    """
    Get information about the T4 GPU.
    
    Returns:
        Dictionary with T4 GPU information
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        device_properties = torch.cuda.get_device_properties(0)
        
        # Check if it's a T4
        is_t4 = "T4" in device_properties.name
        
        info = {
            "name": device_properties.name,
            "is_t4": is_t4,
            "compute_capability": f"{device_properties.major}.{device_properties.minor}",
            "total_memory_gb": device_properties.total_memory / (1024**3),
            "multi_processor_count": device_properties.multi_processor_count,
            "max_threads_per_block": device_properties.max_threads_per_block,
            "max_shared_memory_per_block": device_properties.max_shared_memory_per_block,
            "tensorrt_available": is_tensorrt_available(),
            "tensorrt_version": get_tensorrt_version() if is_tensorrt_available() else None
        }
        
        # Add currently allocated memory
        info["allocated_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["reserved_memory_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        return info
    except Exception as e:
        return {"error": str(e)}

def get_tensorrt_version() -> Optional[str]:
    """Get TensorRT version if available."""
    if is_tensorrt_available():
        try:
            import tensorrt as trt
            return trt.__version__
        except ImportError:
            return None
    return None