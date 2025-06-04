"""
NVIDIA TensorRT utilities for optimized inference.

This module provides TensorRT acceleration for deep learning models,
with a focus on transformer-based architectures and embeddings generation.
It offers significant performance improvements over standard PyTorch inference.
"""
import os
import json
import time
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import threading
import hashlib

logger = logging.getLogger(__name__)

# Try importing TensorRT with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. TensorRT acceleration disabled.")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Install with: pip install nvidia-tensorrt")

try:
    from cuda import cudart
    CUDART_AVAILABLE = True
except ImportError:
    CUDART_AVAILABLE = False
    logger.warning("CUDA Runtime not available. Some TensorRT features disabled.")


@dataclass
class TRTConfig:
    """Configuration for TensorRT optimization."""
    precision: str = "fp16"  # Options: fp32, fp16, int8
    workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 32
    cache_dir: str = "/tmp/tensorrt_engines"
    dynamic_shapes: bool = True
    use_dla: bool = False
    builder_optimization_level: int = 3
    calibration_data: Optional[str] = None
    serialized_engine_path: Optional[str] = None
    verbose: bool = False


class TensorRTEngine:
    """
    TensorRT engine wrapper for optimized inference.
    
    Provides utilities to convert PyTorch models to TensorRT engines and
    accelerate inference with minimal code changes.
    """
    
    def __init__(self, 
                 config: Optional[TRTConfig] = None,
                 model_name: Optional[str] = None):
        """
        Initialize the TensorRT engine.
        
        Parameters
        ----------
        config : TRTConfig, optional
            TensorRT configuration
        model_name : str, optional
            Name of the model for engine caching
        """
        self.config = config or TRTConfig()
        self.model_name = model_name
        self.engine = None
        self.context = None
        self.engine_path = None
        self.binding_shapes = {}
        self.binding_names = []
        self.binding_idx = {}
        self.initialized = False
        self.lock = threading.Lock()
        
        # Check for TensorRT availability
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Acceleration disabled.")
            return
            
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE if self.config.verbose else trt.Logger.WARNING)
        
        logger.info(f"TensorRT engine initialized with {self.config.precision} precision")
    
    def _get_engine_path(self, model_hash: str) -> str:
        """Get the path to the serialized engine file."""
        precision = self.config.precision
        batch_size = self.config.max_batch_size
        dla = "_dla" if self.config.use_dla else ""
        dynamic = "_dyn" if self.config.dynamic_shapes else ""
        
        if self.model_name:
            engine_name = f"{self.model_name}_{model_hash}_{precision}_b{batch_size}{dla}{dynamic}.engine"
        else:
            engine_name = f"model_{model_hash}_{precision}_b{batch_size}{dla}{dynamic}.engine"
            
        return os.path.join(self.config.cache_dir, engine_name)
    
    def _get_model_hash(self, model: Any) -> str:
        """Generate a hash for the model to use in engine caching."""
        if hasattr(model, "state_dict"):
            # Get the model architecture and weights hash
            model_state = model.state_dict()
            
            # Create a buffer for the model state
            buffer = []
            for name, param in model_state.items():
                if isinstance(param, torch.Tensor):
                    # Only include the first few values to speed up hashing
                    flat_tensor = param.flatten()
                    sample = flat_tensor[:min(100, flat_tensor.numel())].detach().cpu().numpy()
                    buffer.append(f"{name}:{sample.tobytes()}")
            
            # Hash the buffer
            hasher = hashlib.md5()
            for item in sorted(buffer):
                hasher.update(item.encode())
            
            return hasher.hexdigest()[:16]
        else:
            # Fallback for non-PyTorch models
            return str(hash(str(model.__class__)) % 10000)
    
    def convert_torch_model(self, 
                           model: Any, 
                           input_shapes: Dict[str, List[int]],
                           output_shapes: Optional[Dict[str, List[int]]] = None) -> bool:
        """
        Convert a PyTorch model to a TensorRT engine.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to convert
        input_shapes : Dict[str, List[int]]
            Dictionary of input names and shapes (excluding batch dimension)
        output_shapes : Dict[str, List[int]], optional
            Dictionary of output names and shapes (excluding batch dimension)
            
        Returns
        -------
        bool
            True if conversion was successful
        """
        if not TORCH_AVAILABLE or not TENSORRT_AVAILABLE:
            logger.warning("PyTorch or TensorRT not available. Conversion skipped.")
            return False
            
        with self.lock:
            if self.initialized:
                logger.warning("Engine already initialized. Call destroy() first.")
                return False
                
            try:
                # Set model to evaluation mode
                model.eval()
                
                # Get model hash for engine caching
                model_hash = self._get_model_hash(model)
                self.engine_path = self._get_engine_path(model_hash)
                
                # Check if serialized engine already exists
                if os.path.exists(self.engine_path):
                    logger.info(f"Loading cached TensorRT engine: {self.engine_path}")
                    return self._load_engine()
                
                logger.info("Converting PyTorch model to TensorRT...")
                
                # Create ONNX file in a temporary directory
                with tempfile.TemporaryDirectory() as tmpdirname:
                    onnx_path = os.path.join(tmpdirname, "model.onnx")
                    
                    # Create dummy inputs
                    dummy_inputs = {}
                    dynamic_axes = {}
                    
                    for name, shape in input_shapes.items():
                        # Add batch dimension
                        full_shape = [1] + shape
                        dummy_inputs[name] = torch.ones(full_shape, device='cuda')
                        
                        if self.config.dynamic_shapes:
                            # Mark batch dimension as dynamic
                            dynamic_axes[name] = {0: 'batch'}
                    
                    # Add output dynamic axes if needed
                    if output_shapes and self.config.dynamic_shapes:
                        for name in output_shapes:
                            dynamic_axes[name] = {0: 'batch'}
                    
                    # Export to ONNX
                    torch.onnx.export(
                        model,
                        tuple(dummy_inputs.values()) if len(dummy_inputs) > 1 else next(iter(dummy_inputs.values())),
                        onnx_path,
                        input_names=list(input_shapes.keys()),
                        output_names=list(output_shapes.keys()) if output_shapes else None,
                        dynamic_axes=dynamic_axes if self.config.dynamic_shapes else None,
                        opset_version=17,
                        do_constant_folding=True,
                        verbose=self.config.verbose
                    )
                    
                    # Convert ONNX to TensorRT
                    return self._convert_onnx_to_trt(onnx_path, input_shapes)
                    
            except Exception as e:
                logger.error(f"Failed to convert PyTorch model to TensorRT: {str(e)}")
                return False
    
    def _convert_onnx_to_trt(self, onnx_path: str, input_shapes: Dict[str, List[int]]) -> bool:
        """Convert ONNX model to TensorRT engine."""
        try:
            logger.info("Building TensorRT engine from ONNX model...")
            
            # Create builder and network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            
            # Set workspace size
            config.max_workspace_size = self.config.workspace_size
            
            # Set optimization level
            config.builder_optimization_level = self.config.builder_optimization_level
            
            # Set precision
            if self.config.precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif self.config.precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Enabled INT8 precision")
                
                # Set up INT8 calibrator if provided
                if self.config.calibration_data:
                    class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
                        def __init__(self, calibration_data, batch_size, input_name, input_shape):
                            super().__init__()
                            self.calibration_data = calibration_data
                            self.batch_size = batch_size
                            self.input_name = input_name
                            self.input_shape = input_shape
                            self.current_index = 0
                            self.num_calibration_batches = min(128, len(calibration_data) // batch_size)
                            
                            # Allocate GPU memory for calibration data
                            self.device_input = cuda.cuMemAlloc(
                                trt.volume(input_shape) * trt.float32.itemsize * batch_size
                            )
                            
                            # Cache path for calibration cache
                            self.cache_file = os.path.join(
                                os.path.dirname(self.config.cache_dir), 
                                f"calibration_{os.path.basename(self.config.serialized_engine_path or 'model')}.cache"
                            )
                            
                        def get_batch_size(self):
                            return self.batch_size
                            
                        def get_batch(self, names):
                            if self.current_index >= self.num_calibration_batches:
                                return None
                                
                            # Prepare batch data
                            batch_data = np.zeros((self.batch_size,) + tuple(self.input_shape[1:]), dtype=np.float32)
                            for i in range(self.batch_size):
                                data_index = self.current_index * self.batch_size + i
                                if data_index < len(self.calibration_data):
                                    # Convert data to appropriate format
                                    # This would need to be customized based on data format
                                    batch_data[i] = self.calibration_data[data_index]
                            
                            # Copy data to GPU
                            cuda.cuMemcpyHtoD(
                                self.device_input, 
                                np.ascontiguousarray(batch_data), 
                                batch_data.nbytes
                            )
                            
                            # Increment counter
                            self.current_index += 1
                            
                            return [int(self.device_input)]
                            
                        def read_calibration_cache(self):
                            if os.path.exists(self.cache_file):
                                with open(self.cache_file, "rb") as f:
                                    return f.read()
                            return None
                            
                        def write_calibration_cache(self, cache):
                            with open(self.cache_file, "wb") as f:
                                f.write(cache)
                    
                    # Get first input name
                    input_name = list(input_shapes.keys())[0]
                    
                    # Create calibrator
                    calibrator = Int8EntropyCalibrator2(
                        calibration_data=self.config.calibration_data,
                        batch_size=self.config.max_batch_size,
                        input_name=input_name,
                        input_shape=[self.config.max_batch_size] + input_shapes[input_name]
                    )
                    
                    # Set calibrator
                    config.int8_calibrator = calibrator
                    logger.info("INT8 calibration enabled")
            
            # Parse ONNX file
            parser = trt.OnnxParser(network, self.trt_logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    return False
            
            # Configure dynamic shapes if enabled
            if self.config.dynamic_shapes:
                profile = builder.create_optimization_profile()
                
                for name, shape in input_shapes.items():
                    # Find input tensor by name
                    tensor_idx = network.get_input(0).name.find(name)
                    if tensor_idx == -1:
                        # Try to find by iterating through inputs
                        for i in range(network.num_inputs):
                            if network.get_input(i).name == name:
                                tensor_idx = i
                                break
                    
                    if tensor_idx != -1:
                        # Set min/opt/max dimensions
                        min_shape = [1] + shape
                        opt_shape = [self.config.max_batch_size // 2] + shape
                        max_shape = [self.config.max_batch_size] + shape
                        
                        profile.set_shape(
                            name, 
                            min_shape,
                            opt_shape,
                            max_shape
                        )
                
                config.add_optimization_profile(profile)
            
            # Use DLA if requested
            if self.config.use_dla:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = 0
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if not serialized_engine:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine to file
            with open(self.engine_path, "wb") as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT engine saved to: {self.engine_path}")
            
            # Load the engine
            return self._load_engine()
            
        except Exception as e:
            logger.error(f"Failed to convert ONNX to TensorRT: {str(e)}")
            return False
    
    def _load_engine(self) -> bool:
        """Load a serialized TensorRT engine."""
        try:
            # Create runtime
            runtime = trt.Runtime(self.trt_logger)
            
            # Load engine from file
            with open(self.engine_path, "rb") as f:
                serialized_engine = f.read()
            
            # Deserialize engine
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if not self.engine:
                logger.error("Failed to deserialize TensorRT engine")
                return False
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Get binding information
            self.binding_names = []
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.binding_names.append(name)
                self.binding_idx[name] = i
            
            self.initialized = True
            logger.info("TensorRT engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {str(e)}")
            return False
    
    def infer(self, 
             inputs: Dict[str, torch.Tensor],
             output_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Run inference with the TensorRT engine.
        
        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Dictionary of input names and tensors
        output_names : List[str], optional
            List of output tensor names to return
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output names and tensors
        """
        if not self.initialized:
            logger.error("TensorRT engine not initialized")
            raise RuntimeError("TensorRT engine not initialized")
        
        # Get batch size from first input
        batch_size = next(iter(inputs.values())).shape[0]
        
        # Prepare bindings
        bindings = []
        outputs = {}
        
        # Allocate output tensors
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            
            if self.engine.binding_is_input(i):
                # Handle input tensor
                if name not in inputs:
                    logger.error(f"Missing input tensor: {name}")
                    raise ValueError(f"Missing input tensor: {name}")
                
                input_tensor = inputs[name]
                bindings.append(input_tensor.data_ptr())
                
                # Set binding shape for dynamic shapes
                if self.config.dynamic_shapes:
                    self.context.set_binding_shape(i, input_tensor.shape)
            else:
                # Handle output tensor
                if output_names is None or name in output_names:
                    # Get shape from context for dynamic shapes
                    if self.config.dynamic_shapes:
                        shape = tuple(self.context.get_binding_shape(i))
                        if shape[0] == -1:  # If batch dimension is dynamic
                            shape = (batch_size,) + shape[1:]
                    else:
                        shape = tuple(self.engine.get_binding_shape(i))
                    
                    # Allocate output tensor
                    output = torch.empty(shape, dtype=torch.float32, device='cuda')
                    outputs[name] = output
                    bindings.append(output.data_ptr())
                else:
                    # Allocate dummy output if not needed
                    shape = tuple(self.context.get_binding_shape(i))
                    if shape[0] == -1:  # If batch dimension is dynamic
                        shape = (batch_size,) + shape[1:]
                    
                    dummy = torch.empty(shape, dtype=torch.float32, device='cuda')
                    bindings.append(dummy.data_ptr())
        
        # Run inference
        if not self.context.execute_v2(bindings):
            logger.error("Failed to execute TensorRT inference")
            raise RuntimeError("Failed to execute TensorRT inference")
        
        return outputs
    
    def destroy(self):
        """Free TensorRT resources."""
        with self.lock:
            if self.context:
                self.context = None
            
            if self.engine:
                self.engine = None
            
            self.initialized = False
            logger.info("TensorRT engine destroyed")


class TensorRTOptimizer:
    """
    Optimizer for converting and accelerating deep learning models with TensorRT.
    
    Provides utilities to manage multiple TensorRT engines and acceleration
    pipelines for different types of models.
    """
    
    def __init__(self, 
                 cache_dir: str = "/tmp/tensorrt_engines",
                 default_precision: str = "fp16"):
        """
        Initialize the TensorRT optimizer.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache TensorRT engines
        default_precision : str
            Default precision to use (fp32, fp16, int8)
        """
        self.cache_dir = cache_dir
        self.default_precision = default_precision
        self.engines = {}
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check for TensorRT availability
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Install with: pip install nvidia-tensorrt")
        else:
            logger.info(f"TensorRT version: {trt.__version__}")
            
            # Check CUDA compatibility
            if CUDART_AVAILABLE:
                _, version = cudart.cudaRuntimeGetVersion()
                logger.info(f"CUDA Runtime version: {version}")
            
            # Log available precision modes
            trt_builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            logger.info(f"TensorRT FP16 support: {trt_builder.platform_has_fast_fp16}")
            logger.info(f"TensorRT INT8 support: {trt_builder.platform_has_fast_int8}")
    
    def optimize_torch_model(self, 
                            model: Any, 
                            model_name: str,
                            input_shapes: Dict[str, List[int]],
                            output_shapes: Optional[Dict[str, List[int]]] = None,
                            precision: Optional[str] = None,
                            max_batch_size: int = 32,
                            dynamic_shapes: bool = True) -> TensorRTEngine:
        """
        Optimize a PyTorch model with TensorRT.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to optimize
        model_name : str
            Name of the model for caching
        input_shapes : Dict[str, List[int]]
            Dictionary of input names and shapes (excluding batch dimension)
        output_shapes : Dict[str, List[int]], optional
            Dictionary of output names and shapes (excluding batch dimension)
        precision : str, optional
            Precision to use (fp32, fp16, int8)
        max_batch_size : int
            Maximum batch size for the engine
        dynamic_shapes : bool
            Whether to use dynamic shapes
            
        Returns
        -------
        TensorRTEngine
            Optimized TensorRT engine
        """
        if not TORCH_AVAILABLE or not TENSORRT_AVAILABLE:
            logger.warning("PyTorch or TensorRT not available. Returning unoptimized model.")
            return None
            
        # Use default precision if not specified
        precision = precision or self.default_precision
        
        # Create engine config
        config = TRTConfig(
            precision=precision,
            max_batch_size=max_batch_size,
            cache_dir=self.cache_dir,
            dynamic_shapes=dynamic_shapes,
            verbose=False
        )
        
        # Create and initialize engine
        engine = TensorRTEngine(config, model_name)
        success = engine.convert_torch_model(model, input_shapes, output_shapes)
        
        if success:
            # Cache the engine
            self.engines[model_name] = engine
            return engine
        else:
            logger.warning(f"Failed to optimize model {model_name} with TensorRT")
            return None
    
    def get_engine(self, model_name: str) -> Optional[TensorRTEngine]:
        """Get a cached TensorRT engine by name."""
        return self.engines.get(model_name)
    
    def optimize_embedding_model(self, 
                                model: Any, 
                                model_name: str,
                                embedding_dim: int,
                                max_sequence_length: int,
                                precision: Optional[str] = None) -> TensorRTEngine:
        """
        Optimize an embedding model with TensorRT.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch embedding model to optimize
        model_name : str
            Name of the model for caching
        embedding_dim : int
            Dimension of the embeddings
        max_sequence_length : int
            Maximum sequence length
        precision : str, optional
            Precision to use (fp32, fp16, int8)
            
        Returns
        -------
        TensorRTEngine
            Optimized TensorRT engine
        """
        # Define input and output shapes for embedding models
        input_shapes = {
            "input_ids": [max_sequence_length],
            "attention_mask": [max_sequence_length]
        }
        
        output_shapes = {
            "embeddings": [embedding_dim]
        }
        
        return self.optimize_torch_model(
            model=model,
            model_name=model_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            precision=precision,
            max_batch_size=64,  # Embedding models can handle larger batches
            dynamic_shapes=True  # Variable sequence lengths
        )
    
    def optimize_transformer_model(self, 
                                 model: Any, 
                                 model_name: str,
                                 max_sequence_length: int,
                                 vocab_size: int,
                                 precision: Optional[str] = None) -> TensorRTEngine:
        """
        Optimize a transformer model with TensorRT.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch transformer model to optimize
        model_name : str
            Name of the model for caching
        max_sequence_length : int
            Maximum sequence length
        vocab_size : int
            Size of the vocabulary
        precision : str, optional
            Precision to use (fp32, fp16, int8)
            
        Returns
        -------
        TensorRTEngine
            Optimized TensorRT engine
        """
        # Define input and output shapes for transformer models
        input_shapes = {
            "input_ids": [max_sequence_length],
            "attention_mask": [max_sequence_length]
        }
        
        output_shapes = {
            "logits": [max_sequence_length, vocab_size]
        }
        
        return self.optimize_torch_model(
            model=model,
            model_name=model_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            precision=precision,
            max_batch_size=16,  # Transformer models usually need smaller batches
            dynamic_shapes=True  # Variable sequence lengths
        )
    
    def cleanup(self):
        """Clean up all TensorRT engines."""
        for engine in self.engines.values():
            engine.destroy()
        
        self.engines = {}
        logger.info("All TensorRT engines cleaned up")


# Global TensorRT optimizer instance
def get_tensorrt_optimizer() -> TensorRTOptimizer:
    """Get the global TensorRT optimizer instance."""
    global _tensorrt_optimizer
    
    if '_tensorrt_optimizer' not in globals():
        _tensorrt_optimizer = TensorRTOptimizer()
    
    return _tensorrt_optimizer