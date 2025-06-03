"""
NVIDIA Hopper-specific optimizations for H100 GPUs.

This module provides specialized optimizations for the latest NVIDIA Hopper
architecture, including Transformer Engine integration, FP8 training/inference,
and hardware-specific kernel optimizations.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check for PyTorch and CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for Transformer Engine availability
try:
    import transformer_engine as te
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

# Check for Apex availability (NVIDIA optimized PyTorch extensions)
try:
    import apex
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class HopperOptimizer:
    """
    Advanced optimizations for NVIDIA H100 (Hopper architecture) GPUs.
    
    Provides specialized optimizations including:
    - FP8 precision for 2-4x speedup
    - Transformer Engine integration
    - Distributed/scaling engine acceleration
    - Hopper-specific kernels and memory optimization
    """
    
    def __init__(self, 
                 enable_fp8: bool = True,
                 enable_distributed_fused_adam: bool = True,
                 enable_flash_attention_2: bool = True,
                 enable_fused_layernorm: bool = True,
                 enable_nvfuser: bool = True):
        """
        Initialize the Hopper optimizer.
        
        Parameters
        ----------
        enable_fp8 : bool
            Whether to enable FP8 precision
        enable_distributed_fused_adam : bool
            Whether to enable distributed fused Adam optimizer
        enable_flash_attention_2 : bool
            Whether to enable Flash Attention 2
        enable_fused_layernorm : bool
            Whether to enable fused LayerNorm
        enable_nvfuser : bool
            Whether to enable nvFuser
        """
        self.enable_fp8 = enable_fp8
        self.enable_distributed_fused_adam = enable_distributed_fused_adam
        self.enable_flash_attention_2 = enable_flash_attention_2
        self.enable_fused_layernorm = enable_fused_layernorm
        self.enable_nvfuser = enable_nvfuser
        
        # Check if running on Hopper architecture
        self.is_hopper = self._detect_hopper()
        if not self.is_hopper:
            logger.warning("NVIDIA Hopper (H100) GPU not detected. Some optimizations will be disabled.")
        
        # Initialize Transformer Engine if available and on Hopper
        self.transformer_engine_initialized = False
        if self.is_hopper and TE_AVAILABLE and self.enable_fp8:
            try:
                # Configure Transformer Engine
                self._init_transformer_engine()
                self.transformer_engine_initialized = True
                logger.info("Transformer Engine initialized for FP8 computation")
            except Exception as e:
                logger.warning(f"Failed to initialize Transformer Engine: {str(e)}")
        
    def _detect_hopper(self) -> bool:
        """
        Detect if running on Hopper architecture (H100).
        
        Returns
        -------
        bool
            True if Hopper GPU detected
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        
        try:
            # Check device capability - Hopper is 9.0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.major >= 9:  # Hopper or newer
                    return True
            return False
        except:
            return False
    
    def _init_transformer_engine(self):
        """Initialize Transformer Engine for FP8 computation."""
        if not TE_AVAILABLE or not self.is_hopper:
            return
        
        # Set default FP8 recipe
        # This is a recipe for FP8 training that balances accuracy and performance
        self.fp8_recipe = te.common.recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=te.common.recipe.Format.E4M3,
            amax_history_len=16,
            amax_compute_algo="max",
        )
    
    def optimize_model(self, model: Any) -> Any:
        """
        Apply Hopper-specific optimizations to a PyTorch model.
        
        Parameters
        ----------
        model : Any
            PyTorch model to optimize
            
        Returns
        -------
        Any
            Optimized model
        """
        if not self.is_hopper or not TORCH_AVAILABLE:
            return model
        
        # Apply FSDP for large models if available
        if hasattr(torch.distributed, "fsdp") and torch.cuda.device_count() > 1:
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                    ShardingStrategy,
                    BackwardPrefetch,
                    CPUOffload,
                )
                
                # Configure mixed precision policy
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
                
                # Apply FSDP wrapping
                model = FSDP(
                    model,
                    mixed_precision=mixed_precision_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    cpu_offload=CPUOffload(offload_params=False),
                )
                
                logger.info("Applied FSDP optimization to model")
            except Exception as e:
                logger.warning(f"Failed to apply FSDP optimization: {str(e)}")
        
        # Apply nvFuser
        if self.enable_nvfuser and hasattr(torch, "_C"):
            try:
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._jit_override_can_fuse_on_gpu(True)
                logger.info("Enabled nvFuser for kernel fusion")
            except:
                pass
        
        # Apply fused LayerNorm if available
        if self.enable_fused_layernorm and APEX_AVAILABLE:
            try:
                # Replace standard LayerNorms with fused version
                from apex.normalization import FusedLayerNorm
                
                def replace_layernorm(module):
                    for name, child in module.named_children():
                        if isinstance(child, torch.nn.LayerNorm):
                            setattr(module, name, FusedLayerNorm(child.normalized_shape, 
                                                               eps=child.eps,
                                                               elementwise_affine=child.elementwise_affine))
                        else:
                            replace_layernorm(child)
                
                replace_layernorm(model)
                logger.info("Replaced standard LayerNorm with FusedLayerNorm")
            except Exception as e:
                logger.warning(f"Failed to apply FusedLayerNorm: {str(e)}")
        
        # Apply Flash Attention 2 if available
        if self.enable_flash_attention_2:
            try:
                import flash_attn
                from flash_attn.flash_attention import FlashAttention
                
                # This implementation depends on the specific model architecture
                # As a placeholder, log that it's available
                logger.info("Flash Attention 2 is available for use with compatible models")
            except ImportError:
                pass
        
        return model
    
    def get_fp8_context(self):
        """
        Get FP8 autocast context for Transformer Engine.
        
        Returns
        -------
        context manager
            FP8 autocast context manager
        """
        if not self.is_hopper or not TE_AVAILABLE or not self.transformer_engine_initialized:
            # Return a dummy context manager if TE not available
            from contextlib import nullcontext
            return nullcontext()
        
        return fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
    
    def optimize_inference(self, model: Any) -> Any:
        """
        Apply Hopper-specific inference optimizations.
        
        Parameters
        ----------
        model : Any
            Model to optimize
            
        Returns
        -------
        Any
            Optimized model
        """
        if not self.is_hopper or not TORCH_AVAILABLE:
            return model
        
        # Convert model to TorchScript if possible
        try:
            # First set model to eval mode
            model.eval()
            
            # Try to TorchScript the model
            scripted_model = torch.jit.script(model)
            
            # Apply optimizations
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            logger.info("Model optimized with TorchScript for inference")
            return scripted_model
        except Exception as e:
            logger.warning(f"Failed to apply TorchScript optimization: {str(e)}")
            return model
    
    def optimize_dtype_config(self) -> Dict[str, Any]:
        """
        Get optimized dtype configuration for Hopper.
        
        Returns
        -------
        Dict[str, Any]
            Optimized dtype configuration
        """
        if not self.is_hopper:
            # For non-Hopper GPUs, return standard mixed precision config
            return {
                "compute_dtype": torch.float16 if TORCH_AVAILABLE else "float16",
                "param_dtype": torch.float16 if TORCH_AVAILABLE else "float16",
                "use_amp": True
            }
        
        # Hopper-specific configuration with FP8
        config = {
            "compute_dtype": torch.bfloat16 if TORCH_AVAILABLE else "bfloat16",  # BF16 for Hopper
            "param_dtype": torch.bfloat16 if TORCH_AVAILABLE else "bfloat16",
            "use_amp": True,
            "use_fp8": self.enable_fp8 and TE_AVAILABLE,
            "use_te": self.transformer_engine_initialized,
            "use_nvfuser": self.enable_nvfuser
        }
        
        return config
    
    def get_hopper_specific_args(self) -> Dict[str, Any]:
        """
        Get Hopper-specific arguments for model initialization.
        
        Returns
        -------
        Dict[str, Any]
            Hopper-specific arguments
        """
        if not self.is_hopper:
            return {}
        
        args = {
            "use_fp8": self.enable_fp8 and TE_AVAILABLE,
            "transformer_engine": self.transformer_engine_initialized,
            "enable_fused_kernels": True,
            "enable_flash_attention": self.enable_flash_attention_2,
            "enable_fused_layernorm": self.enable_fused_layernorm,
            "enable_nvfuser": self.enable_nvfuser,
            "enable_distributed_fused_adam": self.enable_distributed_fused_adam,
            "use_hmma_fp8_accumulation": True  # Special H100 hardware feature
        }
        
        return args


def detect_and_optimize_for_hopper() -> Tuple[bool, Dict[str, Any]]:
    """
    Detect Hopper architecture and return optimized configuration.
    
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_hopper, optimized_config)
    """
    optimizer = HopperOptimizer()
    return optimizer.is_hopper, optimizer.get_hopper_specific_args()