"""
NVIDIA Hopper-specific optimizations for H100 GPUs.

This module provides specialized optimizations for the latest NVIDIA Hopper
architecture, including Transformer Engine integration, FP8 training/inference,
TensorRT acceleration, advanced quantization methods (GPTQ, AWQ), and 
hardware-specific kernel optimizations.
"""
import os
import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

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

# Check for TensorRT availability
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Import TensorRT utilities if available
try:
    from .tensorrt_utils import TensorRTOptimizer, get_tensorrt_optimizer
    TENSORRT_UTILS_AVAILABLE = True
except ImportError:
    TENSORRT_UTILS_AVAILABLE = False
    logger.warning("TensorRT utilities not available. Some optimizations will be disabled.")

# Check for GPTQ availability
try:
    import auto_gptq
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False
    logger.debug("AutoGPTQ not available. GPTQ quantization will be disabled.")

# Check for AWQ availability
try:
    import awq
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    logger.debug("AWQ not available. AWQ quantization will be disabled.")


class HopperOptimizer:
    """
    Advanced optimizations for NVIDIA H100 (Hopper architecture) GPUs.
    
    Provides specialized optimizations including:
    - FP8 precision for 2-4x speedup
    - Transformer Engine integration
    - Distributed/scaling engine acceleration
    - Hopper-specific kernels and memory optimization
    - Advanced quantization (GPTQ, AWQ) for inference efficiency
    """
    
    def __init__(self, 
                 enable_fp8: bool = True,
                 enable_distributed_fused_adam: bool = True,
                 enable_flash_attention_2: bool = True,
                 enable_fused_layernorm: bool = True,
                 enable_nvfuser: bool = True,
                 enable_tensorrt: bool = True,
                 enable_gptq: bool = True,
                 enable_awq: bool = True,
                 quantization_bit_width: int = 4,
                 quantization_dataset_path: Optional[str] = None,
                 quantization_cache_dir: Optional[str] = "/tmp/quantization_cache"):
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
        enable_tensorrt : bool
            Whether to enable TensorRT optimization
        enable_gptq : bool
            Whether to enable GPTQ quantization for inference
        enable_awq : bool
            Whether to enable AWQ quantization for inference
        quantization_bit_width : int
            Bit width for quantization (4 or 8)
        quantization_dataset_path : str, optional
            Path to calibration dataset for quantization
        quantization_cache_dir : str, optional
            Directory to cache quantized models
        """
        self.enable_fp8 = enable_fp8
        self.enable_distributed_fused_adam = enable_distributed_fused_adam
        self.enable_flash_attention_2 = enable_flash_attention_2
        self.enable_fused_layernorm = enable_fused_layernorm
        self.enable_nvfuser = enable_nvfuser
        self.enable_tensorrt = enable_tensorrt
        self.enable_gptq = enable_gptq and GPTQ_AVAILABLE
        self.enable_awq = enable_awq and AWQ_AVAILABLE
        
        # Quantization parameters
        self.quantization_bit_width = quantization_bit_width
        self.quantization_dataset_path = quantization_dataset_path
        self.quantization_cache_dir = quantization_cache_dir
        
        # Create cache directory if it doesn't exist
        if self.quantization_cache_dir:
            os.makedirs(self.quantization_cache_dir, exist_ok=True)
        
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
        
        # Initialize TensorRT optimizer if available
        self.tensorrt_optimizer = None
        if TENSORRT_UTILS_AVAILABLE and self.enable_tensorrt:
            try:
                self.tensorrt_optimizer = get_tensorrt_optimizer()
                logger.info("TensorRT optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorRT optimizer: {str(e)}")
        
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
            # Method 1: Check device capability - Hopper is 9.0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Check compute capability (Hopper is 9.x)
                if props.major >= 9:
                    logger.info(f"Detected Hopper GPU with compute capability {props.major}.{props.minor}")
                    return True
                
                # Check device name for explicit "H100" mention
                if "H100" in props.name:
                    logger.info(f"Detected H100 GPU: {props.name}")
                    return True
            
            # Method 2: Try using nvidia-smi for more detailed information
            if NVIDIA_SMI_AVAILABLE:
                try:
                    nvidia_smi.nvmlInit()
                    device_count = nvidia_smi.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                        device_name = nvidia_smi.nvmlDeviceGetName(handle)
                        
                        # Check for H100 in device name
                        if "H100" in device_name:
                            logger.info(f"Detected H100 GPU via NVML: {device_name}")
                            nvidia_smi.nvmlShutdown()
                            return True
                        
                    nvidia_smi.nvmlShutdown()
                except:
                    pass
            
            # Method 3: Parse nvidia-smi output directly
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True
                )
                
                gpu_names = result.stdout.strip().split('\n')
                for name in gpu_names:
                    if "H100" in name:
                        logger.info(f"Detected H100 GPU via nvidia-smi: {name}")
                        return True
            except:
                pass
            
            # No Hopper GPU detected by any method
            logger.info("No Hopper/H100 GPU detected")
            return False
        except Exception as e:
            logger.warning(f"Error during Hopper GPU detection: {str(e)}")
            return False
    
    def _init_transformer_engine(self):
        """Initialize Transformer Engine for FP8 computation."""
        if not TE_AVAILABLE or not self.is_hopper:
            logger.warning("Transformer Engine not available or not running on Hopper architecture")
            return
        
        try:
            # Configure TE parameters based on available hardware
            logger.info("Initializing Transformer Engine for H100 GPU...")
            
            # Detect available TE components
            available_components = dir(te)
            logger.info(f"Available TE components: {', '.join(c for c in available_components if not c.startswith('_'))}")
            
            # Set default FP8 recipe with appropriate settings for Hopper
            fp8_formats = {
                "E4M3": te.common.recipe.Format.E4M3,  # Higher precision for activations
                "E5M2": te.common.recipe.Format.E5M2   # Higher range for weights
            }
            
            # Choose FP8 format based on workload characteristics
            # E4M3 is generally better for activations (more precision)
            # E5M2 is generally better for weights (more range)
            selected_format = fp8_formats["E4M3"]  # Default to E4M3 for most use cases
            
            # Create the FP8 recipe
            self.fp8_recipe = te.common.recipe.DelayedScaling(
                margin=0,                      # Conservative margin for stability
                interval=1,                    # Update scaling factors every step
                fp8_format=selected_format,    # FP8 format (E4M3 or E5M2)
                amax_history_len=16,           # Keep history of amax values for stability
                amax_compute_algo="max",       # Use max to be conservative
                scaling_factor_compute_algo="max_forward_max_backward_range",  # Consider both forward and backward passes
                compute_fp8_ldm=False,         # Disabled for multi-GPU stability
                override_linear_precision=(True, False, False)  # Enable FP8 for matmul, keep input and output in FP16/BF16
            )
            
            # Configure HMMA FP8 TF32 compatible mode for H100
            if hasattr(te.common, "set_fp8_hmma_tf32_compatible_mode"):
                te.common.set_fp8_hmma_tf32_compatible_mode(True)
                logger.info("Enabled HMMA FP8 TF32 compatible mode")
            
            # Enable cublas workspace for better performance
            if hasattr(te.common, "set_cublas_workspace_config"):
                te.common.set_cublas_workspace_config(":4096:8")
                logger.info("Configured cuBLAS workspace for optimal performance")
            
            # Register custom FP8 operators if available
            if hasattr(te, "register_fp8_operators"):
                te.register_fp8_operators()
                logger.info("Registered custom FP8 operators")
                
            logger.info(f"Transformer Engine initialized with FP8 format: {selected_format}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformer Engine: {str(e)}")
            return False
    
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
                
                # Implement Flash Attention 2 for transformer models
                def replace_attention_with_flash(module):
                    if hasattr(module, "self_attn") or hasattr(module, "attention"):
                        try:
                            # Get the attention module
                            attn_module = getattr(module, "self_attn") if hasattr(module, "self_attn") else getattr(module, "attention")
                            
                            # Check if it's already using FlashAttention
                            if isinstance(attn_module, FlashAttention) or hasattr(attn_module, "_flash_attn_func"):
                                return
                                
                            # Store original forward method
                            orig_forward = attn_module.forward
                            
                            # Create flash attention instance
                            flash_attn_instance = FlashAttention(
                                softmax_scale=attn_module.scale if hasattr(attn_module, "scale") else None,
                                attention_dropout=attn_module.dropout if hasattr(attn_module, "dropout") else 0.0
                            )
                            
                            # Create new forward method with flash attention
                            def forward_with_flash_attn(q, k, v, *args, **kwargs):
                                # Apply flash attention
                                return flash_attn_instance(q, k, v, causal=True)
                            
                            # Add flash attention reference to the module
                            attn_module._flash_attn_func = flash_attn_instance
                            
                            # Replace forward method if it matches expected signature
                            if hasattr(attn_module, "forward") and callable(attn_module.forward):
                                if "q" in inspect.signature(attn_module.forward).parameters and \
                                   "k" in inspect.signature(attn_module.forward).parameters and \
                                   "v" in inspect.signature(attn_module.forward).parameters:
                                    attn_module.forward = forward_with_flash_attn
                                    logger.info(f"Replaced attention in {attn_module.__class__.__name__} with Flash Attention 2")
                        except Exception as e:
                            logger.warning(f"Failed to apply Flash Attention 2 to module: {str(e)}")
                    
                    # Recursively process child modules
                    for name, child in module.named_children():
                        replace_attention_with_flash(child)
                
                # Apply Flash Attention to the model
                import inspect
                replace_attention_with_flash(model)
                logger.info("Applied Flash Attention 2 to compatible attention layers")
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
            # Create a proper FP8 emulation context when TE is not available
            try:
                # Try to use PyTorch's AMP instead
                from torch.cuda.amp import autocast
                logger.info("Using PyTorch AMP autocast as fallback for FP8")
                return autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            except:
                # If that fails, provide a proper no-op context manager
                from contextlib import nullcontext
                logger.warning("FP8/AMP not available, using no-op context manager")
                return nullcontext()
        
        return fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
    
    def optimize_inference(self, 
                         model: Any, 
                         model_name: Optional[str] = None,
                         input_shapes: Optional[Dict[str, List[int]]] = None,
                         quantization_method: Optional[str] = "auto") -> Any:
        """
        Apply Hopper-specific inference optimizations.
        
        Parameters
        ----------
        model : Any
            Model to optimize
        model_name : str, optional
            Name of the model for caching
        input_shapes : Dict[str, List[int]], optional
            Dictionary of input names and shapes for TensorRT
        quantization_method : str, optional
            Quantization method to use: 'auto', 'gptq', 'awq', 'tensorrt', or 'none'
            
        Returns
        -------
        Any
            Optimized model
        """
        if not TORCH_AVAILABLE:
            return model
            
        if model_name is None and hasattr(model, "_name_or_path"):
            model_name = model._name_or_path
            
        if model_name is None:
            model_name = "unknown_model"
            
        # Set model to eval mode first
        if hasattr(model, 'eval'):
            model.eval()
            
        # Decide on the quantization method if 'auto'
        if quantization_method == "auto":
            # Choose the best quantization method based on availability and GPU capabilities
            if self.is_hopper:
                # For Hopper, prefer AWQ then GPTQ then TensorRT
                if self.enable_awq and AWQ_AVAILABLE:
                    quantization_method = "awq"
                elif self.enable_gptq and GPTQ_AVAILABLE:
                    quantization_method = "gptq"
                elif self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE:
                    quantization_method = "tensorrt"
                else:
                    quantization_method = "none"
            else:
                # For non-Hopper, prefer GPTQ then AWQ then TensorRT
                if self.enable_gptq and GPTQ_AVAILABLE:
                    quantization_method = "gptq"
                elif self.enable_awq and AWQ_AVAILABLE:
                    quantization_method = "awq"
                elif self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE:
                    quantization_method = "tensorrt"
                else:
                    quantization_method = "none"
                    
        logger.info(f"Selected quantization method: {quantization_method}")
        
        # Apply the selected quantization method
        if quantization_method == "gptq" and self.enable_gptq and GPTQ_AVAILABLE:
            try:
                return self.quantize_with_gptq(model, model_name)
            except Exception as e:
                logger.warning(f"GPTQ quantization failed: {str(e)}, falling back to other methods")
                
        if quantization_method == "awq" and self.enable_awq and AWQ_AVAILABLE:
            try:
                return self.quantize_with_awq(model, model_name)
            except Exception as e:
                logger.warning(f"AWQ quantization failed: {str(e)}, falling back to other methods")
            
        # Try TensorRT optimization if enabled and inputs are provided
        if (quantization_method == "tensorrt" or quantization_method == "none") and \
           self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE and self.tensorrt_optimizer and input_shapes:
            try:
                logger.info(f"Attempting TensorRT optimization for model {model_name}")
                trt_engine = self.tensorrt_optimizer.optimize_torch_model(
                    model=model,
                    model_name=model_name,
                    input_shapes=input_shapes
                )
                
                if trt_engine:
                    logger.info(f"Model {model_name} successfully optimized with TensorRT")
                    return trt_engine
                else:
                    logger.warning(f"TensorRT optimization failed for {model_name}, falling back to TorchScript")
            except Exception as e:
                logger.warning(f"TensorRT optimization error: {str(e)}, falling back to TorchScript")
        
        # Fall back to TorchScript if other methods fail or are not enabled
        if quantization_method == "none" or not self.is_hopper:
            try:
                # Try to TorchScript the model
                scripted_model = torch.jit.script(model)
                
                # Apply optimizations
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                
                logger.info("Model optimized with TorchScript for inference")
                return scripted_model
            except Exception as e:
                logger.warning(f"Failed to apply TorchScript optimization: {str(e)}")
                
        # Return the original model if all optimization attempts fail
        return model
    
    def _get_calibration_data(self, num_samples: int = 128) -> List[str]:
        """
        Get calibration data for quantization.
        
        Parameters
        ----------
        num_samples : int
            Number of calibration samples to return
            
        Returns
        -------
        List[str]
            List of text samples for calibration
        """
        # Use provided dataset if available
        if self.quantization_dataset_path and os.path.exists(self.quantization_dataset_path):
            try:
                # Try to load calibration data from file
                if self.quantization_dataset_path.endswith('.json'):
                    with open(self.quantization_dataset_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle different JSON formats
                    if isinstance(data, list):
                        if all(isinstance(item, str) for item in data):
                            # Direct list of strings
                            samples = data[:num_samples]
                        elif all(isinstance(item, dict) for item in data):
                            # List of dictionaries, try to extract text field
                            text_fields = ['text', 'content', 'sentence', 'input', 'prompt']
                            for field in text_fields:
                                if field in data[0]:
                                    samples = [item[field] for item in data[:num_samples] if field in item]
                                    break
                            else:
                                # No recognized text field found
                                logger.warning(f"No recognized text field found in JSON dataset: {self.quantization_dataset_path}")
                                samples = self._get_default_calibration_data(num_samples)
                        else:
                            samples = self._get_default_calibration_data(num_samples)
                    elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                        # Handle nested data structure
                        nested_data = data['data']
                        text_fields = ['text', 'content', 'sentence', 'input', 'prompt']
                        for field in text_fields:
                            if field in nested_data[0]:
                                samples = [item[field] for item in nested_data[:num_samples] if field in item]
                                break
                        else:
                            samples = self._get_default_calibration_data(num_samples)
                    else:
                        samples = self._get_default_calibration_data(num_samples)
                elif self.quantization_dataset_path.endswith('.txt'):
                    with open(self.quantization_dataset_path, 'r') as f:
                        lines = f.readlines()
                    samples = [line.strip() for line in lines[:num_samples] if line.strip()]
                else:
                    logger.warning(f"Unsupported calibration dataset format: {self.quantization_dataset_path}")
                    samples = self._get_default_calibration_data(num_samples)
                    
                # Ensure we have enough samples
                if len(samples) < num_samples:
                    logger.warning(f"Insufficient calibration samples ({len(samples)}), using default samples to supplement")
                    samples.extend(self._get_default_calibration_data(num_samples - len(samples)))
                    
                return samples[:num_samples]
                
            except Exception as e:
                logger.warning(f"Failed to load calibration dataset: {str(e)}")
                return self._get_default_calibration_data(num_samples)
        else:
            return self._get_default_calibration_data(num_samples)
    
    def _get_default_calibration_data(self, num_samples: int = 128) -> List[str]:
        """
        Get default calibration data for quantization when no dataset is provided.
        
        Parameters
        ----------
        num_samples : int
            Number of calibration samples to return
            
        Returns
        -------
        List[str]
            List of text samples for calibration
        """
        # Determine domain-specific calibration data based on current environment
        # Try to detect domain from system or available models
        domain = self._detect_domain()
        
        # Generate domain-specific calibration data
        if domain == "finance":
            default_samples = self._get_finance_calibration_data()
        elif domain == "healthcare":
            default_samples = self._get_healthcare_calibration_data()
        elif domain == "analytics":
            default_samples = self._get_analytics_calibration_data()
        else:
            # General domain - mixture of enterprise, technical, and SAP-specific data
            default_samples = self._get_general_calibration_data()
        
        # Calculate how many times we need to repeat the samples
        repeat_count = (num_samples + len(default_samples) - 1) // len(default_samples)
        
        # Generate samples with slight variations for better calibration
        samples = []
        for _ in range(repeat_count):
            for sample in default_samples:
                if len(samples) >= num_samples:
                    break
                
                # Add some variation to samples for better calibration coverage
                if random.random() < 0.3:  # 30% chance of variation
                    # Apply random variations
                    variations = [
                        # Length variation
                        lambda s: s + " " + " ".join(s.split()[:3]),
                        # Question form
                        lambda s: f"Can you explain why {s.lower()}",
                        # Instruction form
                        lambda s: f"Please analyze the following: {s}",
                        # List form
                        lambda s: f"Here are some points to consider:\n1. {s}\n2. Related concepts",
                    ]
                    sample = random.choice(variations)(sample)
                
                samples.append(sample)
        
        return samples[:num_samples]
        
    def _detect_domain(self) -> str:
        """
        Detect the application domain based on environment and available models.
        
        This method analyzes the runtime environment to determine the most appropriate
        domain-specific calibration data to use for model quantization. It examines
        environment variables, model characteristics, and other system indicators to
        make an intelligent decision about which domain the application is operating in.
        
        The detected domain influences the calibration data used for quantization,
        which can significantly impact model performance in domain-specific tasks.
        
        Supported domains:
        - finance: Financial applications, trading, banking, investment
        - healthcare: Medical applications, clinical systems, patient data
        - analytics: Business intelligence, reporting, dashboards
        - general: General enterprise applications (default)
        
        Returns
        -------
        str
            The detected domain name: "finance", "healthcare", "analytics", or "general"
            
        Notes
        -----
        - Domain detection is critical for effective model quantization as it helps
          select appropriate calibration data that matches the target workload
        - The detection process is hierarchical and will check multiple indicators
        - If multiple domains match, the first matching domain in the priority order
          (finance, healthcare, analytics, general) will be selected
        - You can override automatic detection by setting environment variables
          with domain indicators
        """
        import os
        import re
        
        # Check environment variables for domain hints
        env_vars = os.environ.keys()
        
        # Check for finance indicators
        finance_indicators = ['FINANCE', 'BANKING', 'TRADING', 'INVESTMENT']
        for indicator in finance_indicators:
            if any(indicator in var for var in env_vars):
                logger.info(f"Detected finance domain from environment variable containing '{indicator}'")
                return "finance"
                
        # Check for healthcare indicators
        healthcare_indicators = ['HEALTH', 'MEDICAL', 'CLINICAL', 'PATIENT']
        for indicator in healthcare_indicators:
            if any(indicator in var for var in env_vars):
                logger.info(f"Detected healthcare domain from environment variable containing '{indicator}'")
                return "healthcare"
                
        # Check for analytics indicators
        analytics_indicators = ['ANALYTICS', 'DASHBOARD', 'REPORT', 'BI', 'BUSINESS_INTELLIGENCE']
        for indicator in analytics_indicators:
            if any(indicator in var for var in env_vars):
                logger.info(f"Detected analytics domain from environment variable containing '{indicator}'")
                return "analytics"
        
        # Try to detect from SAP-specific environment
        try:
            # Check for SAP HANA environment
            if os.environ.get('HANA_HOST') or os.environ.get('HANA_USERKEY'):
                # This is a HANA environment, further detect the domain
                
                # If we have connection to HANA, check table schemas
                if hasattr(self, 'connection_context') and self.connection_context:
                    # Implementation could query HANA metadata to detect domain
                    pass
            
            # Check file system for domain-specific files
            domain_path_indicators = {
                "finance": ["financial", "trading", "banking", "investment"],
                "healthcare": ["medical", "health", "patient", "clinical"],
                "analytics": ["analytics", "report", "dashboard", "metrics"]
            }
            
            # Look at current directory and parent directories
            current_path = os.getcwd()
            for domain, indicators in domain_path_indicators.items():
                for indicator in indicators:
                    if indicator in current_path.lower():
                        logger.info(f"Detected {domain} domain from current path containing '{indicator}'")
                        return domain
        except:
            pass
                
        # Default to general domain (SAP enterprise)
        logger.info("No specific domain detected, using general domain")
        return "general"
        
    def _get_general_calibration_data(self) -> List[str]:
        """Get general domain calibration data."""
        return [
            "The transformer architecture has revolutionized natural language processing.",
            "Deep learning models can process multiple types of data including text, images, and audio.",
            "NVIDIA's Hopper GPUs feature advanced matrix multiplication capabilities for AI workloads.",
            "Implementing efficient transformers requires attention to memory usage and computational optimizations.",
            "SAP HANA Cloud provides enterprise-grade database capabilities with advanced analytics.",
            "Integrating generative AI with enterprise data requires careful consideration of security and privacy.",
            "Time series forecasting is an important application of machine learning in business intelligence.",
            "What are the main components of a neural network?",
            "Explain the concept of attention mechanisms in transformer models.",
            "How can I optimize my database queries for better performance?",
            "Summarize the key benefits of using cloud-based analytics platforms.",
            "SQL is a standard language for accessing and manipulating databases.",
            "Python has become the dominant programming language for data science and machine learning.",
            "The integration of large language models with enterprise systems creates new opportunities for automation.",
            "Efficient model deployment requires consideration of hardware capabilities and optimization techniques.",
            "Enterprise resource planning systems integrate core business processes.",
            "SAP BTP provides a platform for extending and integrating SAP applications.",
            "Artificial intelligence enhances business processes through automation and data analysis.",
            "Cloud computing offers scalability, flexibility, and cost advantages for enterprise applications.",
            "Data governance ensures that high-quality data is available throughout an organization.",
            "Machine learning models can identify patterns in large datasets to predict future trends.",
            "Digital transformation involves integrating digital technology into all areas of a business.",
            "The Internet of Things connects physical devices to the internet for data exchange.",
            "APIs enable different software systems to communicate and share data.",
            "Cybersecurity protects systems, networks, and programs from digital attacks.",
        ]
        
    def _get_finance_calibration_data(self) -> List[str]:
        """Get finance domain calibration data."""
        return [
            "Financial analysis involves examining financial statements to evaluate performance.",
            "Investment strategies aim to balance risk and return based on investor objectives.",
            "Portfolio diversification reduces risk by allocating investments across different assets.",
            "Asset allocation determines the mix of assets in a portfolio based on risk tolerance.",
            "Financial forecasting uses historical data to predict future financial performance.",
            "Risk management identifies, assesses, and prioritizes financial risks.",
            "Capital budgeting evaluates the profitability of long-term investments.",
            "Derivative securities derive their value from underlying assets or benchmarks.",
            "Financial regulations aim to maintain market integrity and protect investors.",
            "Market efficiency theory suggests that prices reflect all available information.",
            "Financial statement analysis evaluates a company's financial health and performance.",
            "Algorithmic trading uses computer algorithms to execute trading strategies.",
            "Credit analysis evaluates the creditworthiness of individuals or organizations.",
            "Actuarial science applies mathematical and statistical methods to assess risk.",
            "Corporate finance focuses on maximizing shareholder value through financial decisions.",
            "Financial planning helps individuals achieve long-term financial goals.",
            "Investment banking services include underwriting, mergers, and acquisitions.",
            "Financial markets facilitate the exchange of financial securities.",
            "Hedge funds use various strategies to generate returns for investors.",
            "Financial technology (FinTech) uses technology to improve financial services.",
        ]
        
    def _get_healthcare_calibration_data(self) -> List[str]:
        """Get healthcare domain calibration data."""
        return [
            "Electronic health records store patient information digitally for easy access.",
            "Clinical decision support systems assist healthcare providers in making decisions.",
            "Telemedicine enables remote diagnosis and treatment of patients.",
            "Medical imaging uses technology to create visual representations of the body.",
            "Healthcare analytics uses data to improve patient outcomes and reduce costs.",
            "Population health management aims to improve health outcomes for groups of individuals.",
            "Precision medicine tailors medical treatment to individual characteristics.",
            "Healthcare interoperability enables different systems to exchange and use information.",
            "Patient engagement involves patients in their own healthcare decisions.",
            "Medical research aims to develop new treatments and improve existing ones.",
            "Health information exchange allows sharing of patient data across organizations.",
            "Remote patient monitoring tracks patient health outside traditional settings.",
            "Clinical trials evaluate the safety and efficacy of medical interventions.",
            "Healthcare quality metrics measure the performance of healthcare providers.",
            "Medical informatics applies information technology to healthcare delivery.",
            "Preventive care focuses on preventing illness rather than treating it.",
            "Healthcare compliance ensures adherence to laws, regulations, and standards.",
            "Patient safety initiatives aim to prevent harm to patients during care.",
            "Healthcare supply chain management ensures availability of supplies and medications.",
            "Public health initiatives promote health and prevent disease at the population level.",
        ]
        
    def _get_analytics_calibration_data(self) -> List[str]:
        """Get analytics domain calibration data."""
        return [
            "Business intelligence transforms data into actionable insights for decision-making.",
            "Data visualization presents complex data in graphical format for easier understanding.",
            "Predictive analytics uses historical data to forecast future outcomes.",
            "Data mining discovers patterns in large datasets to extract useful information.",
            "Statistical analysis examines data to identify trends and relationships.",
            "Descriptive analytics summarizes historical data to understand what happened.",
            "Prescriptive analytics recommends actions based on predictive insights.",
            "Key performance indicators measure progress toward business objectives.",
            "A/B testing compares two versions to determine which performs better.",
            "Real-time analytics processes data as it's generated for immediate insights.",
            "Text analytics extracts meaning from unstructured text data.",
            "Customer analytics examines customer data to improve business relationships.",
            "Web analytics measures and analyzes website traffic and behavior.",
            "Supply chain analytics optimizes supply chain operations through data analysis.",
            "Marketing analytics measures marketing performance and effectiveness.",
            "Financial analytics assesses financial performance and risks.",
            "Operational analytics improves business operations through data analysis.",
            "Sentiment analysis identifies and categorizes opinions in text data.",
            "Competitive intelligence gathers and analyzes information about competitors.",
            "Social media analytics measures and interprets interactions on social platforms.",
        ]
    
    def quantize_with_gptq(self, 
                          model: Any,
                          model_name: str,
                          save_path: Optional[str] = None) -> Any:
        """
        Quantize a model using GPTQ (Generative Pre-trained Transformer Quantization).
        
        GPTQ is a post-training quantization method specifically designed for transformer 
        models. It quantizes weights to reduce memory usage and inference latency while 
        maintaining model quality by finding optimal quantization parameters through 
        a second-order Hessian approximation.
        
        Parameters
        ----------
        model : Any
            PyTorch model to quantize or model name/path string. The model should be 
            a transformer-based model compatible with Hugging Face's transformers library.
        model_name : str
            Name of the model for caching purposes. Used to generate cache filenames
            and uniquely identify the model.
        save_path : str, optional
            Custom path to save the quantized model. If not provided, the model will be 
            saved to the default quantization cache directory specified during initialization.
            
        Returns
        -------
        Any
            Quantized model that can be used for inference. If quantization fails,
            returns the original model.
            
        Notes
        -----
        - GPTQ typically provides 3-4x memory reduction with minimal accuracy loss
        - The quantization process requires a calibration dataset, which is either provided
          during initialization or generated automatically
        - Models are cached to disk for faster loading in subsequent runs
        - Compatible with 2-bit, 3-bit, 4-bit, and 8-bit quantization (configurable via 
          quantization_bit_width parameter during initialization)
        """
        if not self.enable_gptq or not GPTQ_AVAILABLE:
            logger.warning("GPTQ quantization not available or disabled")
            return model
            
        try:
            logger.info(f"Starting GPTQ quantization for model {model_name}")
            
            # Check if we have a cached version
            cache_path = None
            if self.quantization_cache_dir:
                sanitized_name = model_name.replace('/', '_').replace('-', '_')
                bits_suffix = f"{self.quantization_bit_width}bit"
                cache_path = os.path.join(self.quantization_cache_dir, f"{sanitized_name}_gptq_{bits_suffix}")
                
                if os.path.exists(cache_path):
                    logger.info(f"Loading cached GPTQ model from {cache_path}")
                    try:
                        # Load quantized model from cache
                        quantized_model = AutoGPTQForCausalLM.from_quantized(
                            cache_path,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            use_triton=self.is_hopper  # Triton works best on latest GPUs
                        )
                        logger.info("Successfully loaded cached GPTQ model")
                        return quantized_model
                    except Exception as e:
                        logger.warning(f"Failed to load cached GPTQ model: {str(e)}")
            
            # Get calibration data
            examples = self._get_calibration_data(num_samples=128)
            
            # Configure quantization parameters
            quantize_config = BaseQuantizeConfig(
                bits=self.quantization_bit_width,  # 4-bit or 8-bit quantization
                group_size=128,  # Group size for the quantization
                desc_act=False,  # Whether to quantize activations
                sym=True  # Whether to use symmetric quantization
            )
            
            # Create GPTQ model for quantization
            gptq_model = AutoGPTQForCausalLM.from_pretrained(
                model if isinstance(model, str) else model_name,
                quantize_config=quantize_config,
                use_triton=self.is_hopper  # Triton works best on latest GPUs
            )
            
            # Quantize the model
            gptq_model.quantize(examples)
            
            # Save the quantized model if requested
            save_location = save_path or cache_path
            if save_location:
                gptq_model.save_quantized(save_location)
                logger.info(f"Saved quantized GPTQ model to {save_location}")
            
            logger.info("GPTQ quantization completed successfully")
            return gptq_model
            
        except Exception as e:
            logger.error(f"GPTQ quantization failed: {str(e)}")
            return model
    
    def quantize_with_awq(self, 
                         model: Any,
                         model_name: str,
                         save_path: Optional[str] = None) -> Any:
        """
        Quantize a model using AWQ (Activation-aware Weight Quantization).
        
        AWQ is an advanced quantization technique that preserves model quality by identifying
        and protecting critical weights based on activation patterns. It analyzes each layer's
        activation distribution to determine which weights have the most impact on output
        quality, then applies different quantization strategies to preserve those critical
        weights while aggressively quantizing less important ones.
        
        Parameters
        ----------
        model : Any
            PyTorch model to quantize or model name/path string. The model should be 
            a transformer-based model compatible with Hugging Face's transformers library.
        model_name : str
            Name of the model for caching purposes. Used to generate cache filenames
            and uniquely identify the model.
        save_path : str, optional
            Custom path to save the quantized model. If not provided, the model will be 
            saved to the default quantization cache directory specified during initialization.
            
        Returns
        -------
        Any
            Quantized model that can be used for inference. If quantization fails,
            returns the original model.
            
        Notes
        -----
        - AWQ typically provides superior quality compared to standard quantization methods,
          especially for 4-bit quantization
        - AWQ analyzes per-channel activation patterns to preserve model quality
        - The implementation uses GEMM-based (General Matrix Multiplication) kernel optimizations
          for faster inference on NVIDIA GPUs
        - AWQ works particularly well with H100 Tensor Cores when using 4-bit precision
        - The quantization process requires a calibration dataset, which is either provided
          during initialization or generated automatically based on domain detection
        """
        if not self.enable_awq or not AWQ_AVAILABLE:
            logger.warning("AWQ quantization not available or disabled")
            return model
            
        try:
            logger.info(f"Starting AWQ quantization for model {model_name}")
            
            # Check if we have a cached version
            cache_path = None
            if self.quantization_cache_dir:
                sanitized_name = model_name.replace('/', '_').replace('-', '_')
                bits_suffix = f"{self.quantization_bit_width}bit"
                cache_path = os.path.join(self.quantization_cache_dir, f"{sanitized_name}_awq_{bits_suffix}")
                
                if os.path.exists(cache_path):
                    logger.info(f"Loading cached AWQ model from {cache_path}")
                    try:
                        # Load quantized model from cache
                        quantized_model = AutoAWQForCausalLM.from_quantized(
                            cache_path,
                            device_map="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        logger.info("Successfully loaded cached AWQ model")
                        return quantized_model
                    except Exception as e:
                        logger.warning(f"Failed to load cached AWQ model: {str(e)}")
            
            # Get calibration data
            examples = self._get_calibration_data(num_samples=128)
            
            # Create AWQ model for quantization
            awq_model = AutoAWQForCausalLM.from_pretrained(
                model if isinstance(model, str) else model_name
            )
            
            # Quantize the model
            awq_model.quantize(
                examples,
                bit_width=self.quantization_bit_width,
                export_to="awq",  # Export format
                # AWQ-specific parameters
                zero_point=True,  # Whether to use zero-point quantization
                group_size=128,  # Group size for the quantization
                version="gemm"  # Use GEMM version for best performance
            )
            
            # Save the quantized model if requested
            save_location = save_path or cache_path
            if save_location:
                awq_model.save_quantized(save_location)
                logger.info(f"Saved quantized AWQ model to {save_location}")
            
            logger.info("AWQ quantization completed successfully")
            return awq_model
            
        except Exception as e:
            logger.error(f"AWQ quantization failed: {str(e)}")
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
            "use_nvfuser": self.enable_nvfuser,
            "enable_gptq": self.enable_gptq and GPTQ_AVAILABLE,
            "enable_awq": self.enable_awq and AWQ_AVAILABLE,
            "quantization_bit_width": self.quantization_bit_width
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
            "use_hmma_fp8_accumulation": True,  # Special H100 hardware feature
            "enable_tensorrt": self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE,
            "tensorrt_optimizer": self.tensorrt_optimizer is not None,
            # Add quantization-related arguments
            "enable_gptq": self.enable_gptq and GPTQ_AVAILABLE,
            "enable_awq": self.enable_awq and AWQ_AVAILABLE,
            "quantization_bit_width": self.quantization_bit_width,
            "quantization_cache_dir": self.quantization_cache_dir,
            "preferred_quantization": "awq" if self.enable_awq and AWQ_AVAILABLE else 
                                     "gptq" if self.enable_gptq and GPTQ_AVAILABLE else 
                                     "tensorrt" if self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE else "none"
        }
        
        return args
        
    def optimize_embedding_model(self,
                               model: Any,
                               model_name: str,
                               embedding_dim: int,
                               max_sequence_length: int,
                               quantization_method: str = "auto") -> Any:
        """
        Optimize an embedding model with advanced quantization.
        
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
        quantization_method : str, optional
            Quantization method to use: 'auto', 'gptq', 'awq', 'tensorrt', or 'none'
            
        Returns
        -------
        Any
            Optimized model or TensorRT engine
        """
        # Decide on the quantization method if 'auto'
        if quantization_method == "auto":
            # For embedding models, TensorRT often works better than GPTQ/AWQ
            if self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE and self.tensorrt_optimizer:
                quantization_method = "tensorrt"
            elif self.enable_gptq and GPTQ_AVAILABLE:
                quantization_method = "gptq"
            elif self.enable_awq and AWQ_AVAILABLE:
                quantization_method = "awq"
            else:
                quantization_method = "none"
                
        logger.info(f"Selected embedding model quantization method: {quantization_method}")
        
        # Apply TensorRT optimization if selected
        if quantization_method == "tensorrt" and self.enable_tensorrt and TENSORRT_UTILS_AVAILABLE and self.tensorrt_optimizer:
            try:
                trt_engine = self.tensorrt_optimizer.optimize_embedding_model(
                    model=model,
                    model_name=model_name,
                    embedding_dim=embedding_dim,
                    max_sequence_length=max_sequence_length,
                    precision="fp16" if self.is_hopper else "fp32"
                )
                
                if trt_engine:
                    logger.info(f"Embedding model {model_name} successfully optimized with TensorRT")
                    return trt_engine
                else:
                    logger.warning(f"TensorRT optimization failed for embedding model {model_name}")
            except Exception as e:
                logger.warning(f"Failed to optimize embedding model with TensorRT: {str(e)}")
        
        # Try GPTQ if selected or if TensorRT failed
        if (quantization_method == "gptq" or (quantization_method == "tensorrt" and model is model)) and self.enable_gptq and GPTQ_AVAILABLE:
            try:
                # For embedding models, we need to adjust the quantization parameters
                # GPTQ works better with higher bit width for embedding models
                original_bit_width = self.quantization_bit_width
                if self.quantization_bit_width < 8:
                    self.quantization_bit_width = 8  # Use 8-bit for embeddings
                    
                # Quantize the model
                quantized_model = self.quantize_with_gptq(model, model_name)
                
                # Restore original bit width
                self.quantization_bit_width = original_bit_width
                
                if quantized_model is not model:  # Check if quantization was successful
                    logger.info(f"Embedding model {model_name} successfully optimized with GPTQ")
                    return quantized_model
                else:
                    logger.warning(f"GPTQ optimization failed for embedding model {model_name}")
            except Exception as e:
                logger.warning(f"Failed to optimize embedding model with GPTQ: {str(e)}")
        
        # Try AWQ if selected or if previous methods failed
        if (quantization_method == "awq" or (quantization_method in ["tensorrt", "gptq"] and model is model)) and self.enable_awq and AWQ_AVAILABLE:
            try:
                # For embedding models, we need to adjust the quantization parameters
                original_bit_width = self.quantization_bit_width
                if self.quantization_bit_width < 8:
                    self.quantization_bit_width = 8  # Use 8-bit for embeddings
                    
                # Quantize the model
                quantized_model = self.quantize_with_awq(model, model_name)
                
                # Restore original bit width
                self.quantization_bit_width = original_bit_width
                
                if quantized_model is not model:  # Check if quantization was successful
                    logger.info(f"Embedding model {model_name} successfully optimized with AWQ")
                    return quantized_model
                else:
                    logger.warning(f"AWQ optimization failed for embedding model {model_name}")
            except Exception as e:
                logger.warning(f"Failed to optimize embedding model with AWQ: {str(e)}")
        
        # Return the original model if all optimization attempts fail
        return model


def detect_and_optimize_for_hopper(
    enable_gptq: bool = True,
    enable_awq: bool = True,
    quantization_bit_width: int = 4,
    quantization_dataset_path: Optional[str] = None,
    quantization_cache_dir: Optional[str] = "/tmp/quantization_cache"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect Hopper architecture and return optimized configuration.
    
    Parameters
    ----------
    enable_gptq : bool
        Whether to enable GPTQ quantization
    enable_awq : bool
        Whether to enable AWQ quantization
    quantization_bit_width : int
        Bit width for quantization (4 or 8)
    quantization_dataset_path : str, optional
        Path to calibration dataset for quantization
    quantization_cache_dir : str, optional
        Directory to cache quantized models
    
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_hopper, optimized_config)
    """
    optimizer = HopperOptimizer(
        enable_gptq=enable_gptq,
        enable_awq=enable_awq,
        quantization_bit_width=quantization_bit_width,
        quantization_dataset_path=quantization_dataset_path,
        quantization_cache_dir=quantization_cache_dir
    )
    return optimizer.is_hopper, optimizer.get_hopper_specific_args()