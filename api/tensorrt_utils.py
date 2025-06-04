"""
TensorRT optimization utilities for SAP HANA AI Toolkit.

This module provides integration with NVIDIA TensorRT for optimizing
deep learning models for inference. It supports model conversion,
calibration, and execution with TensorRT for significant performance
improvements on NVIDIA GPUs.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import tempfile

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check TensorRT availability
try:
    import tensorrt as trt
    from tensorrt.tensorrt import ICudaEngine, IExecutionContext
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT is available: %s", trt.__version__)
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Models will run with standard PyTorch.")

class TRTConfig:
    """Configuration for TensorRT optimization settings."""
    
    def __init__(
        self,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 16,
        strict_type_constraints: bool = False,
        enable_sparsity: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize TensorRT configuration.
        
        Args:
            fp16_mode: Enable FP16 precision
            int8_mode: Enable INT8 precision (requires calibration)
            max_workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size
            strict_type_constraints: Whether to strictly follow layer precision
            enable_sparsity: Enable sparse tensor cores if available
            cache_dir: Directory to cache TensorRT engines
            **kwargs: Additional configuration options
        """
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.strict_type_constraints = strict_type_constraints
        self.enable_sparsity = enable_sparsity
        
        # Set cache directory for TensorRT engines
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "tensorrt_engines")
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store additional configuration
        self.additional_config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "fp16_mode": self.fp16_mode,
            "int8_mode": self.int8_mode,
            "max_workspace_size": self.max_workspace_size,
            "max_batch_size": self.max_batch_size,
            "strict_type_constraints": self.strict_type_constraints,
            "enable_sparsity": self.enable_sparsity,
            "cache_dir": self.cache_dir,
            **self.additional_config
        }
        
    def get_engine_path(self, model_name: str) -> str:
        """Get the path for the serialized TensorRT engine."""
        config_hash = hash(json.dumps(self.to_dict(), sort_keys=True))
        filename = f"{model_name}_{config_hash}.engine"
        return os.path.join(self.cache_dir, filename)


class TensorRTEngine:
    """
    TensorRT engine wrapper for optimized inference.
    """
    
    def __init__(self, config: Optional[TRTConfig] = None, model_name: Optional[str] = None):
        """
        Initialize TensorRT engine.
        
        Args:
            config: TensorRT configuration
            model_name: Name of the model for caching
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install TensorRT.")
        
        self.config = config or TRTConfig()
        self.model_name = model_name or "model"
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.binding_shapes = {}
        
    def convert_torch_model(
        self, 
        model: Any, 
        input_shapes: Dict[str, List[int]], 
        output_shapes: Optional[Dict[str, List[int]]] = None
    ) -> bool:
        """
        Convert PyTorch model to TensorRT engine.
        
        Args:
            model: PyTorch model
            input_shapes: Dictionary of input names to shapes
            output_shapes: Dictionary of output names to shapes
            
        Returns:
            bool: Whether conversion was successful
        """
        try:
            # Check if engine exists in cache
            engine_path = self.config.get_engine_path(self.model_name)
            if os.path.exists(engine_path):
                logger.info(f"Loading cached TensorRT engine from {engine_path}")
                with open(engine_path, "rb") as f:
                    engine_bytes = f.read()
                
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
                self._setup_bindings()
                return True
                
            # Model needs to be in eval mode
            model.eval()
            
            # Create inputs with the correct shapes
            inputs = {}
            for name, shape in input_shapes.items():
                inputs[name] = torch.randn(*shape, device="cuda")
            
            # Create TensorRT wrapped model using torch_tensorrt
            logger.info(f"Converting model to TensorRT: {self.model_name}")
            trt_ts_module = torch_tensorrt.compile(
                model,
                inputs=inputs,
                enabled_precisions={torch.float16 if self.config.fp16_mode else torch.float32},
                workspace_size=self.config.max_workspace_size,
            )
            
            # Save the model
            torch.jit.save(trt_ts_module, engine_path)
            logger.info(f"TensorRT engine saved to {engine_path}")
            
            # Load the saved model
            trt_ts_module = torch.jit.load(engine_path)
            self.trt_model = trt_ts_module
            
            # Store input and output information
            self.input_names = list(input_shapes.keys())
            self.output_names = list(output_shapes.keys()) if output_shapes else ["output"]
            
            return True
        
        except Exception as e:
            logger.error(f"Error converting model to TensorRT: {str(e)}", exc_info=True)
            return False
    
    def _setup_bindings(self):
        """Set up input and output bindings for the TensorRT engine."""
        if not self.engine:
            return
            
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                
            shape = self.engine.get_binding_shape(i)
            self.binding_shapes[name] = shape
            
    def infer(
        self, 
        inputs: Dict[str, torch.Tensor], 
        output_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with TensorRT engine.
        
        Args:
            inputs: Dictionary of input name to tensor
            output_names: List of output names to return
            
        Returns:
            Dictionary of output name to tensor
        """
        try:
            # Using torch_tensorrt model
            if hasattr(self, 'trt_model'):
                input_list = [inputs[name] for name in self.input_names]
                if len(input_list) == 1:
                    outputs = self.trt_model(input_list[0])
                else:
                    outputs = self.trt_model(*input_list)
                
                # Convert outputs to dictionary
                if isinstance(outputs, torch.Tensor):
                    outputs = {self.output_names[0]: outputs}
                elif isinstance(outputs, tuple):
                    outputs = {name: output for name, output in zip(self.output_names, outputs)}
                
                return outputs
            
            # Using TensorRT engine directly
            elif self.engine:
                # Set up context if not done yet
                if not self.context:
                    self.context = self.engine.create_execution_context()
                
                # Prepare input and output buffers
                bindings = []
                output_dict = {}
                
                # Process inputs
                for name in self.input_names:
                    if name not in inputs:
                        raise ValueError(f"Input {name} not provided")
                    
                    input_tensor = inputs[name].contiguous().cuda()
                    bindings.append(input_tensor.data_ptr())
                
                # Prepare output buffers
                for name in self.output_names:
                    shape = self.binding_shapes[name]
                    output_tensor = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
                    output_dict[name] = output_tensor
                    bindings.append(output_tensor.data_ptr())
                
                # Run inference
                self.context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=torch.cuda.current_stream().cuda_stream
                )
                
                # Return requested outputs
                if output_names:
                    return {name: output_dict[name] for name in output_names if name in output_dict}
                return output_dict
                
            else:
                raise RuntimeError("No TensorRT engine or model available")
                
        except Exception as e:
            logger.error(f"Error during TensorRT inference: {str(e)}", exc_info=True)
            raise


class TensorRTOptimizer:
    """
    Optimizer for converting and running models with TensorRT.
    """
    
    def __init__(self, config: Optional[TRTConfig] = None):
        """
        Initialize the TensorRT optimizer.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config or TRTConfig()
        self.engines = {}
        
    def optimize_torch_model(
        self, 
        model: Any, 
        model_name: str,
        input_shapes: Dict[str, List[int]],
        output_shapes: Optional[Dict[str, List[int]]] = None
    ) -> Optional[TensorRTEngine]:
        """
        Optimize a PyTorch model with TensorRT.
        
        Args:
            model: PyTorch model
            model_name: Name of the model for caching
            input_shapes: Dictionary of input names to shapes
            output_shapes: Dictionary of output names to shapes
            
        Returns:
            TensorRTEngine: Optimized TensorRT engine or None if optimization failed
        """
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Running with standard PyTorch.")
            return None
            
        # Check if we already have an engine for this model
        if model_name in self.engines:
            return self.engines[model_name]
            
        # Create new engine
        engine = TensorRTEngine(self.config, model_name)
        
        # Convert model to TensorRT
        success = engine.convert_torch_model(model, input_shapes, output_shapes)
        
        if success:
            self.engines[model_name] = engine
            return engine
        else:
            logger.warning(f"Failed to convert {model_name} to TensorRT. Using PyTorch.")
            return None
            
    def get_engine(self, model_name: str) -> Optional[TensorRTEngine]:
        """
        Get a TensorRT engine by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            TensorRTEngine or None if not found
        """
        return self.engines.get(model_name)
        
    def benchmark(
        self, 
        model: Any,
        model_name: str,
        sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark a model with and without TensorRT optimization.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            sample_input: Sample input tensor or dictionary of tensors
            num_iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Benchmarks will be inaccurate.")
            
        # Prepare input shapes
        if isinstance(sample_input, torch.Tensor):
            input_shapes = {"input": list(sample_input.shape)}
            inputs = {"input": sample_input.cuda()}
        else:
            input_shapes = {name: list(tensor.shape) for name, tensor in sample_input.items()}
            inputs = {name: tensor.cuda() for name, tensor in sample_input.items()}
            
        # Move model to GPU and set to eval mode
        model = model.cuda().eval()
        
        # Benchmark PyTorch model
        torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                if isinstance(sample_input, torch.Tensor):
                    _ = model(sample_input.cuda())
                else:
                    _ = model(**{name: tensor.cuda() for name, tensor in sample_input.items()})
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if isinstance(sample_input, torch.Tensor):
                    _ = model(sample_input.cuda())
                else:
                    _ = model(**{name: tensor.cuda() for name, tensor in sample_input.items()})
                    
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / num_iterations * 1000  # ms
        
        # Optimize with TensorRT
        engine = self.optimize_torch_model(model, model_name, input_shapes)
        
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
            
            return {
                "model_name": model_name,
                "pytorch_time_ms": pytorch_time,
                "tensorrt_time_ms": tensorrt_time,
                "speedup": speedup,
                "iterations": num_iterations
            }
        else:
            return {
                "model_name": model_name,
                "pytorch_time_ms": pytorch_time,
                "tensorrt_time_ms": None,
                "speedup": 1.0,
                "iterations": num_iterations,
                "error": "TensorRT optimization failed"
            }


# Utility functions
def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    return TENSORRT_AVAILABLE

def get_tensorrt_version() -> Optional[str]:
    """Get TensorRT version if available."""
    if TENSORRT_AVAILABLE:
        return trt.__version__
    return None