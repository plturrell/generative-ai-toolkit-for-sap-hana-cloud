"""
Sparsity Optimizer for model compression and acceleration.

This module implements neural network sparsification techniques based on Google Research's
state_of_sparsity project to optimize models for memory-constrained environments like T4 GPUs.
The implementation focuses on transparent integration with existing model pipelines without
changing API interfaces.
"""

import os
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import threading
from dataclasses import dataclass, field
import inspect

# Conditionally import PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPARSITY_CONFIG = {
    "enabled": True,
    "target_sparsity": 0.8,  # 80% sparsity (keep 20% of weights)
    "pruning_schedule": "polynomial_decay",
    "pruning_frequency": 1000,
    "begin_step": 0,
    "end_step": 10000,
    "weight_decay": 0.0,
    "initial_sparsity": 0.0,
    "final_sparsity": 0.8,
    "block_size": (1, 1),  # (m, n) for block sparsity
    "use_block_sparsity": False,
    "mask_init_method": "random",
    "use_quantization": True,
    "quantization_bits": 8,
    "quantization_scheme": "symmetric",
    "skip_layers": ["embedding", "layernorm", "bias"],
    "layer_override": {},
}

@dataclass
class SparsityStats:
    """Statistics for model sparsity."""
    model_name: str = ""
    original_size_mb: float = 0.0
    sparse_size_mb: float = 0.0
    compression_ratio: float = 1.0
    average_sparsity: float = 0.0
    layer_sparsity: Dict[str, float] = field(default_factory=dict)
    inference_speedup: float = 1.0
    memory_reduction: float = 0.0
    applied_techniques: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MaskGenerator:
    """
    Generates sparse masks for neural network tensors.
    
    This class implements various sparsity mask generation techniques 
    based on the state_of_sparsity methodology.
    """
    
    @staticmethod
    def generate_random_mask(tensor: 'torch.Tensor', sparsity: float) -> 'torch.Tensor':
        """
        Generate a random binary mask with the specified sparsity level.
        
        Args:
            tensor: Input tensor to generate mask for
            sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Binary mask tensor (1s for weights to keep, 0s for weights to prune)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for mask generation")
            
        if sparsity <= 0.0:
            # No sparsity, keep all weights
            return torch.ones_like(tensor)
        elif sparsity >= 1.0:
            # Full sparsity, remove all weights (not practical)
            return torch.zeros_like(tensor)
        
        # Generate random values
        random_values = torch.rand_like(tensor)
        
        # Create mask based on threshold
        threshold = torch.quantile(random_values.view(-1), sparsity)
        mask = (random_values >= threshold).float()
        
        return mask
    
    @staticmethod
    def generate_magnitude_mask(tensor: 'torch.Tensor', sparsity: float) -> 'torch.Tensor':
        """
        Generate a mask based on weight magnitudes.
        
        Args:
            tensor: Input tensor to generate mask for
            sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Binary mask tensor (1s for weights to keep, 0s for weights to prune)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for mask generation")
            
        if sparsity <= 0.0:
            # No sparsity, keep all weights
            return torch.ones_like(tensor)
        elif sparsity >= 1.0:
            # Full sparsity, remove all weights (not practical)
            return torch.zeros_like(tensor)
        
        # Calculate absolute values
        abs_values = torch.abs(tensor)
        
        # Create mask based on threshold
        threshold = torch.quantile(abs_values.view(-1), sparsity)
        mask = (abs_values >= threshold).float()
        
        return mask
    
    @staticmethod
    def generate_structured_mask(
        tensor: 'torch.Tensor', 
        sparsity: float, 
        block_size: Tuple[int, int] = (1, 1)
    ) -> 'torch.Tensor':
        """
        Generate a structured sparsity mask with block pattern.
        
        Args:
            tensor: Input tensor to generate mask for
            sparsity: Target sparsity level (0.0 to 1.0)
            block_size: Block dimensions (m, n)
            
        Returns:
            Binary mask tensor with structured block pattern
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for mask generation")
            
        if sparsity <= 0.0:
            # No sparsity, keep all weights
            return torch.ones_like(tensor)
        elif sparsity >= 1.0:
            # Full sparsity, remove all weights (not practical)
            return torch.zeros_like(tensor)
        
        # Get tensor shape
        shape = tensor.shape
        
        # For non-2D tensors, reshape to 2D
        if len(shape) != 2:
            tensor_2d = tensor.reshape(shape[0], -1)
        else:
            tensor_2d = tensor
        
        # Get dimensions
        rows, cols = tensor_2d.shape
        
        # Calculate block parameters
        m, n = block_size
        num_blocks_m = (rows + m - 1) // m
        num_blocks_n = (cols + n - 1) // n
        
        # Create block-level mask
        num_blocks = num_blocks_m * num_blocks_n
        num_blocks_to_keep = int(num_blocks * (1 - sparsity))
        
        # Generate random values for blocks
        block_values = torch.rand(num_blocks_m, num_blocks_n)
        
        # Create block-level mask based on threshold
        threshold = torch.quantile(block_values.view(-1), sparsity)
        block_mask = (block_values >= threshold).float()
        
        # Expand block mask to full tensor size
        mask = torch.zeros_like(tensor_2d)
        
        for i in range(num_blocks_m):
            for j in range(num_blocks_n):
                if block_mask[i, j] > 0.5:
                    # Keep this block
                    row_start = i * m
                    row_end = min(row_start + m, rows)
                    col_start = j * n
                    col_end = min(col_start + n, cols)
                    
                    mask[row_start:row_end, col_start:col_end] = 1.0
        
        # Reshape mask back to original tensor shape if needed
        if len(shape) != 2:
            mask = mask.reshape(shape)
        
        return mask


class SparsityScheduler:
    """
    Implements sparsity scheduling for gradual pruning.
    
    This class provides various sparsity schedules based on the
    state_of_sparsity methodology for gradual pruning during training.
    """
    
    @staticmethod
    def constant_sparsity(
        step: int,
        target_sparsity: float,
        begin_step: int = 0,
        end_step: int = 0,
        **kwargs
    ) -> float:
        """
        Constant sparsity schedule.
        
        Args:
            step: Current training step
            target_sparsity: Target sparsity level
            begin_step: Step to begin pruning (not used in constant schedule)
            end_step: Step to end pruning (not used in constant schedule)
            
        Returns:
            Current sparsity level
        """
        return target_sparsity
    
    @staticmethod
    def polynomial_decay(
        step: int,
        initial_sparsity: float,
        final_sparsity: float,
        begin_step: int,
        end_step: int,
        power: float = 3.0,
        **kwargs
    ) -> float:
        """
        Polynomial decay sparsity schedule.
        
        Args:
            step: Current training step
            initial_sparsity: Initial sparsity level
            final_sparsity: Final sparsity level
            begin_step: Step to begin pruning
            end_step: Step to end pruning
            power: Power of polynomial function
            
        Returns:
            Current sparsity level
        """
        if step <= begin_step:
            return initial_sparsity
        elif step >= end_step:
            return final_sparsity
        
        # Calculate normalized step
        normalized_step = (step - begin_step) / (end_step - begin_step)
        
        # Calculate current sparsity using polynomial function
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (normalized_step ** power)
        
        return current_sparsity
    
    @staticmethod
    def exponential_decay(
        step: int,
        initial_sparsity: float,
        final_sparsity: float,
        begin_step: int,
        end_step: int,
        decay_rate: float = 0.9,
        **kwargs
    ) -> float:
        """
        Exponential decay sparsity schedule.
        
        Args:
            step: Current training step
            initial_sparsity: Initial sparsity level
            final_sparsity: Final sparsity level
            begin_step: Step to begin pruning
            end_step: Step to end pruning
            decay_rate: Exponential decay rate
            
        Returns:
            Current sparsity level
        """
        if step <= begin_step:
            return initial_sparsity
        elif step >= end_step:
            return final_sparsity
        
        # Calculate number of decay steps
        num_steps = end_step - begin_step
        current_step = step - begin_step
        
        # Calculate current sparsity using exponential function
        sparsity_range = final_sparsity - initial_sparsity
        decay_factor = decay_rate ** (current_step / num_steps)
        current_sparsity = final_sparsity - sparsity_range * decay_factor
        
        return current_sparsity


class QuantizationHelper:
    """
    Helper class for tensor quantization.
    
    Implements various quantization schemes for reducing
    model memory footprint based on state_of_sparsity approach.
    """
    
    @staticmethod
    def quantize_tensor(
        tensor: 'torch.Tensor',
        bits: int = 8,
        scheme: str = "symmetric",
        per_channel: bool = False,
        channel_dim: int = 0
    ) -> Tuple['torch.Tensor', Dict[str, Any]]:
        """
        Quantize a tensor to specified bit precision.
        
        Args:
            tensor: Input tensor to quantize
            bits: Bit precision (e.g., 8 for int8)
            scheme: Quantization scheme ("symmetric" or "asymmetric")
            per_channel: Whether to quantize per channel
            channel_dim: Channel dimension for per-channel quantization
            
        Returns:
            Tuple of (quantized tensor, quantization parameters)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for quantization")
            
        # Calculate quantization range
        qmin = 0
        qmax = 2 ** bits - 1
        
        if per_channel:
            # Per-channel quantization
            dim_size = tensor.size(channel_dim)
            min_values = []
            max_values = []
            scales = []
            zero_points = []
            
            # Get min/max for each channel
            for c in range(dim_size):
                # Select channel
                if channel_dim == 0:
                    channel_tensor = tensor[c]
                elif channel_dim == 1:
                    channel_tensor = tensor[:, c]
                else:
                    # Slice along specified dimension
                    indices = [slice(None)] * tensor.dim()
                    indices[channel_dim] = c
                    channel_tensor = tensor[tuple(indices)]
                
                # Calculate min/max
                if scheme == "symmetric":
                    # Symmetric quantization
                    abs_max = torch.max(torch.abs(channel_tensor))
                    min_val = -abs_max
                    max_val = abs_max
                else:
                    # Asymmetric quantization
                    min_val = torch.min(channel_tensor)
                    max_val = torch.max(channel_tensor)
                
                # Calculate scale and zero point
                scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else torch.tensor(1.0)
                zero_point = qmin - (min_val / scale).round() if scale != 0 else torch.tensor(0)
                
                # Store values
                min_values.append(min_val)
                max_values.append(max_val)
                scales.append(scale)
                zero_points.append(zero_point)
            
            # Convert to tensors
            min_values = torch.stack(min_values)
            max_values = torch.stack(max_values)
            scales = torch.stack(scales)
            zero_points = torch.stack(zero_points)
            
            # Quantize
            indices = [slice(None)] * tensor.dim()
            q_tensor = torch.zeros_like(tensor)
            
            for c in range(dim_size):
                # Select channel
                indices[channel_dim] = c
                
                # Quantize channel
                channel_tensor = tensor[tuple(indices)]
                q_channel = torch.clamp(
                    (channel_tensor / scales[c]) + zero_points[c],
                    qmin, qmax
                ).round()
                
                # Store quantized channel
                q_tensor[tuple(indices)] = q_channel
            
            # Create quantization params
            quant_params = {
                "scheme": scheme,
                "bits": bits,
                "per_channel": True,
                "channel_dim": channel_dim,
                "min_values": min_values,
                "max_values": max_values,
                "scales": scales,
                "zero_points": zero_points
            }
        else:
            # Per-tensor quantization
            if scheme == "symmetric":
                # Symmetric quantization
                abs_max = torch.max(torch.abs(tensor))
                min_val = -abs_max
                max_val = abs_max
            else:
                # Asymmetric quantization
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
            
            # Calculate scale and zero point
            scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else torch.tensor(1.0)
            zero_point = qmin - (min_val / scale).round() if scale != 0 else torch.tensor(0)
            
            # Quantize tensor
            q_tensor = torch.clamp(
                (tensor / scale) + zero_point,
                qmin, qmax
            ).round()
            
            # Create quantization params
            quant_params = {
                "scheme": scheme,
                "bits": bits,
                "per_channel": False,
                "min_val": min_val.item(),
                "max_val": max_val.item(),
                "scale": scale.item(),
                "zero_point": zero_point.item()
            }
        
        return q_tensor, quant_params
    
    @staticmethod
    def dequantize_tensor(
        q_tensor: 'torch.Tensor',
        quant_params: Dict[str, Any]
    ) -> 'torch.Tensor':
        """
        Dequantize a tensor using quantization parameters.
        
        Args:
            q_tensor: Quantized tensor
            quant_params: Quantization parameters
            
        Returns:
            Dequantized tensor
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for dequantization")
            
        if quant_params["per_channel"]:
            # Per-channel dequantization
            channel_dim = quant_params["channel_dim"]
            scales = quant_params["scales"]
            zero_points = quant_params["zero_points"]
            
            # Dequantize
            dim_size = q_tensor.size(channel_dim)
            indices = [slice(None)] * q_tensor.dim()
            dq_tensor = torch.zeros_like(q_tensor, dtype=torch.float32)
            
            for c in range(dim_size):
                # Select channel
                indices[channel_dim] = c
                
                # Dequantize channel
                q_channel = q_tensor[tuple(indices)]
                dq_channel = (q_channel - zero_points[c]) * scales[c]
                
                # Store dequantized channel
                dq_tensor[tuple(indices)] = dq_channel
        else:
            # Per-tensor dequantization
            scale = quant_params["scale"]
            zero_point = quant_params["zero_point"]
            
            # Dequantize tensor
            dq_tensor = (q_tensor - zero_point) * scale
        
        return dq_tensor


class SparseModelOptimizer:
    """
    Optimizer for applying sparsity to neural network models.
    
    This class implements the core functionality for sparse tensor operations
    based on the state_of_sparsity approach, focusing on optimizing models
    for inference on memory-constrained devices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sparse model optimizer.
        
        Args:
            config: Configuration dictionary for sparsity settings
        """
        # Use default config if not provided
        self.config = config or DEFAULT_SPARSITY_CONFIG.copy()
        
        # Initialize mask generator
        self.mask_generator = MaskGenerator()
        
        # Initialize quantization helper
        self.quantizer = QuantizationHelper()
        
        # Initialize scheduler
        self.scheduler = SparsityScheduler()
        
        # Get scheduler function
        schedule_name = self.config.get("pruning_schedule", "polynomial_decay")
        if hasattr(self.scheduler, schedule_name):
            self.schedule_fn = getattr(self.scheduler, schedule_name)
        else:
            logger.warning(f"Unknown pruning schedule: {schedule_name}, using polynomial_decay")
            self.schedule_fn = self.scheduler.polynomial_decay
        
        # Initialize step counter
        self.step = self.config.get("begin_step", 0)
        
        # Initialize statistics
        self.stats = {}
        
        # Initialize cache for sparse tensors
        self.sparse_tensor_cache = {}
        
        # Check PyTorch availability
        if not HAS_TORCH:
            logger.warning("PyTorch not available, some sparsity features will be disabled")
    
    def _should_sparsify_layer(self, name: str, tensor: 'torch.Tensor') -> bool:
        """
        Check if a layer should be sparsified.
        
        Args:
            name: Layer name
            tensor: Layer tensor
            
        Returns:
            True if layer should be sparsified, False otherwise
        """
        # Check layer override
        layer_override = self.config.get("layer_override", {})
        if name in layer_override:
            return layer_override[name].get("sparsify", True)
        
        # Check skip layers
        skip_layers = self.config.get("skip_layers", [])
        for skip_pattern in skip_layers:
            if skip_pattern in name.lower():
                return False
        
        # Skip small tensors
        min_params = self.config.get("min_params_to_sparsify", 1000)
        if tensor.numel() < min_params:
            return False
        
        return True
    
    def _get_layer_sparsity(self, name: str) -> float:
        """
        Get target sparsity for a specific layer.
        
        Args:
            name: Layer name
            
        Returns:
            Target sparsity for the layer
        """
        # Check layer override
        layer_override = self.config.get("layer_override", {})
        if name in layer_override:
            return layer_override[name].get("sparsity", self.config.get("target_sparsity", 0.8))
        
        # Use global target sparsity
        return self.config.get("target_sparsity", 0.8)
    
    def _get_layer_block_size(self, name: str) -> Tuple[int, int]:
        """
        Get block size for a specific layer.
        
        Args:
            name: Layer name
            
        Returns:
            Block size for the layer
        """
        # Check layer override
        layer_override = self.config.get("layer_override", {})
        if name in layer_override:
            return layer_override[name].get("block_size", self.config.get("block_size", (1, 1)))
        
        # Use global block size
        return self.config.get("block_size", (1, 1))
    
    def _should_quantize_layer(self, name: str) -> bool:
        """
        Check if a layer should be quantized.
        
        Args:
            name: Layer name
            
        Returns:
            True if layer should be quantized, False otherwise
        """
        # Check if quantization is enabled
        if not self.config.get("use_quantization", True):
            return False
        
        # Check layer override
        layer_override = self.config.get("layer_override", {})
        if name in layer_override:
            return layer_override[name].get("quantize", True)
        
        # Check skip layers
        skip_quantization = self.config.get("skip_quantization", [])
        for skip_pattern in skip_quantization:
            if skip_pattern in name.lower():
                return False
        
        return True
    
    def _get_quantization_bits(self, name: str) -> int:
        """
        Get quantization bits for a specific layer.
        
        Args:
            name: Layer name
            
        Returns:
            Quantization bits for the layer
        """
        # Check layer override
        layer_override = self.config.get("layer_override", {})
        if name in layer_override:
            return layer_override[name].get("quantization_bits", self.config.get("quantization_bits", 8))
        
        # Use global quantization bits
        return self.config.get("quantization_bits", 8)
    
    def sparsify_tensor(
        self, 
        name: str, 
        tensor: 'torch.Tensor', 
        mask: Optional['torch.Tensor'] = None
    ) -> Tuple['torch.Tensor', 'torch.Tensor', Dict[str, Any]]:
        """
        Apply sparsity to a tensor.
        
        Args:
            name: Tensor name
            tensor: Input tensor
            mask: Optional pre-computed mask
            
        Returns:
            Tuple of (sparse tensor, mask, sparsity info)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for tensor sparsification")
            
        # Check if we should sparsify this tensor
        if not self._should_sparsify_layer(name, tensor):
            # Return original tensor with full mask and empty info
            return tensor, torch.ones_like(tensor), {"sparsity": 0.0}
        
        # Get target sparsity for this tensor
        target_sparsity = self._get_layer_sparsity(name)
        
        # Apply sparsity scheduling if training
        if self.config.get("pruning_schedule", "constant") != "constant":
            # Get current sparsity level based on step
            current_sparsity = self.schedule_fn(
                step=self.step,
                initial_sparsity=self.config.get("initial_sparsity", 0.0),
                final_sparsity=target_sparsity,
                begin_step=self.config.get("begin_step", 0),
                end_step=self.config.get("end_step", 10000),
                **self.config
            )
        else:
            current_sparsity = target_sparsity
        
        # Generate mask if not provided
        if mask is None:
            if self.config.get("use_block_sparsity", False):
                # Use structured sparsity with blocks
                block_size = self._get_layer_block_size(name)
                mask = self.mask_generator.generate_structured_mask(
                    tensor, current_sparsity, block_size
                )
            else:
                # Use method specified in config
                mask_method = self.config.get("mask_init_method", "magnitude")
                if mask_method == "random":
                    mask = self.mask_generator.generate_random_mask(tensor, current_sparsity)
                elif mask_method == "magnitude":
                    mask = self.mask_generator.generate_magnitude_mask(tensor, current_sparsity)
                else:
                    logger.warning(f"Unknown mask method: {mask_method}, using magnitude")
                    mask = self.mask_generator.generate_magnitude_mask(tensor, current_sparsity)
        
        # Apply mask to tensor
        sparse_tensor = tensor * mask
        
        # Calculate actual sparsity
        nonzero = torch.sum(mask > 0).item()
        total = mask.numel()
        actual_sparsity = 1.0 - (nonzero / total)
        
        # Create sparsity info
        sparsity_info = {
            "sparsity": actual_sparsity,
            "nonzero": nonzero,
            "total": total,
            "target_sparsity": target_sparsity,
            "current_sparsity": current_sparsity,
            "block_sparsity": self.config.get("use_block_sparsity", False),
            "block_size": self._get_layer_block_size(name) if self.config.get("use_block_sparsity", False) else (1, 1)
        }
        
        return sparse_tensor, mask, sparsity_info
    
    def quantize_tensor(
        self, 
        name: str, 
        tensor: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', Dict[str, Any]]:
        """
        Apply quantization to a tensor.
        
        Args:
            name: Tensor name
            tensor: Input tensor
            
        Returns:
            Tuple of (quantized tensor, quantization info)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for tensor quantization")
            
        # Check if we should quantize this tensor
        if not self._should_quantize_layer(name):
            # Return original tensor with empty info
            return tensor, {"quantized": False}
        
        # Get quantization bits
        bits = self._get_quantization_bits(name)
        
        # Get quantization scheme
        scheme = self.config.get("quantization_scheme", "symmetric")
        
        # Check if we should use per-channel quantization
        per_channel = self.config.get("per_channel_quantization", False)
        channel_dim = self.config.get("channel_dim", 0)
        
        # Apply quantization
        q_tensor, quant_params = self.quantizer.quantize_tensor(
            tensor, bits, scheme, per_channel, channel_dim
        )
        
        # Add extra info
        quant_params["quantized"] = True
        quant_params["name"] = name
        
        return q_tensor, quant_params
    
    def sparsify_model(
        self, 
        model: 'torch.nn.Module',
        model_name: str = "",
        cache_id: Optional[str] = None
    ) -> Tuple['torch.nn.Module', SparsityStats]:
        """
        Apply sparsity to a PyTorch model.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            cache_id: Optional cache identifier
            
        Returns:
            Tuple of (sparsified model, sparsity statistics)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for model sparsification")
            
        # Check if sparsity is enabled
        if not self.config.get("enabled", True):
            # Return original model with empty stats
            return model, SparsityStats(model_name=model_name)
        
        # Use model_name as cache_id if not provided
        cache_id = cache_id or model_name
        
        # Check if model is already cached
        if cache_id and cache_id in self.sparse_tensor_cache:
            logger.info(f"Using cached sparse model: {cache_id}")
            
            # Get cached model and stats
            cached_data = self.sparse_tensor_cache[cache_id]
            sparse_model = cached_data.get("model")
            stats = cached_data.get("stats")
            
            if sparse_model is not None and stats is not None:
                return sparse_model, stats
        
        # Clone model to avoid modifying original
        sparse_model = model
        
        # Create statistics object
        stats = SparsityStats(model_name=model_name)
        
        # Calculate original model size
        original_size_bytes = 0
        for name, param in model.named_parameters():
            original_size_bytes += param.numel() * param.element_size()
        
        stats.original_size_mb = original_size_bytes / (1024 * 1024)
        
        # Track overall sparsity information
        total_params = 0
        total_nonzero = 0
        layer_masks = {}
        layer_quant_params = {}
        
        # Apply sparsity and quantization to each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Apply sparsity
                sparse_tensor, mask, sparsity_info = self.sparsify_tensor(name, param.data)
                
                # Store mask for later use
                layer_masks[name] = mask
                
                # Update stats
                total_params += sparsity_info["total"]
                total_nonzero += sparsity_info["nonzero"]
                stats.layer_sparsity[name] = sparsity_info["sparsity"]
                
                # Apply quantization if enabled
                if self.config.get("use_quantization", True):
                    q_tensor, quant_params = self.quantize_tensor(name, sparse_tensor)
                    
                    # Store quantization params
                    layer_quant_params[name] = quant_params
                    
                    # Update tensor in model
                    if quant_params["quantized"]:
                        # For inference, we need to dequantize back to float
                        dq_tensor = self.quantizer.dequantize_tensor(q_tensor, quant_params)
                        param.data.copy_(dq_tensor)
                        
                        # Add quantization to applied techniques
                        technique = f"int{quant_params['bits']}_quantization"
                        if technique not in stats.applied_techniques:
                            stats.applied_techniques.append(technique)
                    else:
                        # Just update with sparse tensor
                        param.data.copy_(sparse_tensor)
                else:
                    # Update with sparse tensor
                    param.data.copy_(sparse_tensor)
        
        # Calculate average sparsity
        if total_params > 0:
            stats.average_sparsity = 1.0 - (total_nonzero / total_params)
        
        # Calculate sparse model size (approximation)
        sparse_size_bytes = original_size_bytes * (1.0 - stats.average_sparsity)
        if self.config.get("use_quantization", True):
            # Account for quantization
            bits_reduction = 32 / self.config.get("quantization_bits", 8)
            sparse_size_bytes /= bits_reduction
        
        stats.sparse_size_mb = sparse_size_bytes / (1024 * 1024)
        
        # Calculate compression ratio
        if stats.sparse_size_mb > 0:
            stats.compression_ratio = stats.original_size_mb / stats.sparse_size_mb
        
        # Estimate memory reduction
        stats.memory_reduction = 1.0 - (stats.sparse_size_mb / stats.original_size_mb)
        
        # Add applied techniques
        if stats.average_sparsity > 0.01:
            if self.config.get("use_block_sparsity", False):
                stats.applied_techniques.append("block_sparsity")
            else:
                stats.applied_techniques.append("unstructured_sparsity")
        
        # Cache sparse model and stats
        if cache_id:
            self.sparse_tensor_cache[cache_id] = {
                "model": sparse_model,
                "stats": stats,
                "masks": layer_masks,
                "quant_params": layer_quant_params
            }
        
        return sparse_model, stats
    
    def get_stats(self, cache_id: Optional[str] = None) -> Optional[SparsityStats]:
        """
        Get sparsity statistics for a cached model.
        
        Args:
            cache_id: Cache identifier
            
        Returns:
            SparsityStats object or None if not found
        """
        if cache_id and cache_id in self.sparse_tensor_cache:
            return self.sparse_tensor_cache[cache_id].get("stats")
        
        return None
    
    def clear_cache(self, cache_id: Optional[str] = None):
        """
        Clear sparse tensor cache.
        
        Args:
            cache_id: Optional cache identifier to clear specific entry
        """
        if cache_id:
            if cache_id in self.sparse_tensor_cache:
                del self.sparse_tensor_cache[cache_id]
        else:
            self.sparse_tensor_cache.clear()
    
    def update_step(self, step: int):
        """
        Update training step for sparsity scheduling.
        
        Args:
            step: Current training step
        """
        self.step = step


class SparsityOptimizer:
    """
    Main interface for applying sparsity optimizations to models.
    
    This class provides a high-level API for using sparsity techniques
    to optimize models for memory-constrained environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sparsity optimizer.
        
        Args:
            config: Configuration dictionary for sparsity settings
        """
        # Apply environment variable overrides to config
        env_config = self._get_env_config()
        
        # Start with default config
        self.config = DEFAULT_SPARSITY_CONFIG.copy()
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        # Override with environment variables
        self.config.update(env_config)
        
        # Create sparse model optimizer
        self.sparse_model_optimizer = SparseModelOptimizer(self.config)
        
        # Initialize cache for optimized models
        self.optimized_models = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Log configuration
        self.logger.info(f"Initialized SparsityOptimizer with config: {json.dumps(self.config, indent=2)}")
    
    def _get_env_config(self) -> Dict[str, Any]:
        """
        Get configuration from environment variables.
        
        Returns:
            Dictionary with configuration from environment variables
        """
        env_config = {}
        
        # Check if sparsity is enabled
        if "SPARSITY_ENABLED" in os.environ:
            env_config["enabled"] = os.environ["SPARSITY_ENABLED"].lower() in ("true", "1", "yes")
        
        # Get target sparsity
        if "SPARSITY_TARGET" in os.environ:
            try:
                env_config["target_sparsity"] = float(os.environ["SPARSITY_TARGET"])
            except ValueError:
                pass
        
        # Check if quantization is enabled
        if "SPARSITY_QUANTIZATION_ENABLED" in os.environ:
            env_config["use_quantization"] = os.environ["SPARSITY_QUANTIZATION_ENABLED"].lower() in ("true", "1", "yes")
        
        # Get quantization bits
        if "SPARSITY_QUANTIZATION_BITS" in os.environ:
            try:
                env_config["quantization_bits"] = int(os.environ["SPARSITY_QUANTIZATION_BITS"])
            except ValueError:
                pass
        
        # Check if block sparsity is enabled
        if "SPARSITY_BLOCK_ENABLED" in os.environ:
            env_config["use_block_sparsity"] = os.environ["SPARSITY_BLOCK_ENABLED"].lower() in ("true", "1", "yes")
        
        # Get block size
        if "SPARSITY_BLOCK_SIZE" in os.environ:
            try:
                block_size = os.environ["SPARSITY_BLOCK_SIZE"].split(",")
                if len(block_size) == 2:
                    env_config["block_size"] = (int(block_size[0]), int(block_size[1]))
            except ValueError:
                pass
        
        return env_config
    
    def optimize_model(
        self, 
        model: Any,
        model_name: str = "",
        model_type: str = "pytorch",
        cache_id: Optional[str] = None
    ) -> Tuple[Any, SparsityStats]:
        """
        Optimize a model using sparsity techniques.
        
        Args:
            model: Model to optimize
            model_name: Name of the model
            model_type: Type of model (currently only "pytorch" supported)
            cache_id: Optional cache identifier
            
        Returns:
            Tuple of (optimized model, sparsity statistics)
        """
        # Check if sparsity is enabled
        if not self.config.get("enabled", True):
            # Return original model with empty stats
            return model, SparsityStats(model_name=model_name)
        
        # Use model_name as cache_id if not provided
        cache_id = cache_id or model_name
        
        # Check if model is already cached
        if cache_id and cache_id in self.optimized_models:
            self.logger.info(f"Using cached optimized model: {cache_id}")
            return self.optimized_models[cache_id]["model"], self.optimized_models[cache_id]["stats"]
        
        # Check model type
        if model_type.lower() == "pytorch":
            if not HAS_TORCH:
                self.logger.warning("PyTorch not available, cannot optimize model")
                return model, SparsityStats(model_name=model_name)
            
            # Ensure model is in eval mode for inference
            if hasattr(model, "eval"):
                model.eval()
            
            # Apply sparsity to PyTorch model
            optimized_model, stats = self.sparse_model_optimizer.sparsify_model(
                model, model_name, cache_id
            )
            
            # Cache optimized model
            if cache_id:
                self.optimized_models[cache_id] = {
                    "model": optimized_model,
                    "stats": stats
                }
            
            return optimized_model, stats
        else:
            self.logger.warning(f"Unsupported model type: {model_type}")
            return model, SparsityStats(model_name=model_name)
    
    def get_stats(self, model_id: str) -> Optional[SparsityStats]:
        """
        Get sparsity statistics for a cached model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            SparsityStats object or None if not found
        """
        if model_id in self.optimized_models:
            return self.optimized_models[model_id].get("stats")
        
        return self.sparse_model_optimizer.get_stats(model_id)
    
    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear optimized model cache.
        
        Args:
            model_id: Optional model identifier to clear specific entry
        """
        if model_id:
            if model_id in self.optimized_models:
                del self.optimized_models[model_id]
            
            self.sparse_model_optimizer.clear_cache(model_id)
        else:
            self.optimized_models.clear()
            self.sparse_model_optimizer.clear_cache()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update configuration.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        self.sparse_model_optimizer = SparseModelOptimizer(self.config)
        
        # Log configuration update
        self.logger.info(f"Updated SparsityOptimizer config: {json.dumps(self.config, indent=2)}")


# Singleton instance
_sparsity_optimizer = None

def get_sparsity_optimizer(config: Optional[Dict[str, Any]] = None) -> SparsityOptimizer:
    """
    Get singleton sparsity optimizer instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SparsityOptimizer instance
    """
    global _sparsity_optimizer
    
    if _sparsity_optimizer is None:
        _sparsity_optimizer = SparsityOptimizer(config)
    elif config:
        # Update config if provided
        _sparsity_optimizer.update_config(config)
    
    return _sparsity_optimizer