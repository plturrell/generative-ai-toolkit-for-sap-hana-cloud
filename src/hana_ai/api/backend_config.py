"""
Backend configuration system for SAP HANA AI Toolkit.

This module provides the configuration and management for multiple backend
options including NVIDIA LaunchPad, Together.ai, and CPU-only processing.
"""

import os
import logging
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

class BackendType(str, Enum):
    """Enumeration of supported backend types."""
    NVIDIA = "nvidia"
    TOGETHER_AI = "together_ai"
    CPU = "cpu"
    AUTO = "auto"

class BackendPriority(BaseModel):
    """Configuration for backend priority and failover."""
    primary: BackendType = Field(BackendType.AUTO, description="Primary backend to use")
    secondary: Optional[BackendType] = Field(None, description="Secondary backend for failover")
    auto_failover: bool = Field(True, description="Whether to automatically failover to secondary backend")
    failover_attempts: int = Field(3, description="Number of attempts before failing over")
    failover_timeout: float = Field(10.0, description="Timeout in seconds before failing over")
    load_balancing: bool = Field(False, description="Whether to load balance between backends")
    load_ratio: float = Field(0.8, description="Ratio of requests to send to primary backend (0.0-1.0)")
    
    @validator('load_ratio')
    def validate_load_ratio(cls, v):
        """Validate that load_ratio is between 0.0 and 1.0."""
        if v < 0.0 or v > 1.0:
            raise ValueError("load_ratio must be between 0.0 and 1.0")
        return v

class NvidiaBackendConfig(BaseModel):
    """Configuration for NVIDIA backend."""
    enabled: bool = Field(False, description="Whether NVIDIA backend is enabled")
    enable_tensorrt: bool = Field(True, description="Whether to enable TensorRT optimizations")
    enable_flash_attention: bool = Field(True, description="Whether to enable Flash Attention")
    enable_transformer_engine: bool = Field(True, description="Whether to enable Transformer Engine")
    enable_fp8: bool = Field(True, description="Whether to enable FP8 precision")
    enable_gptq: bool = Field(True, description="Whether to enable GPTQ quantization")
    enable_awq: bool = Field(True, description="Whether to enable AWQ quantization")
    default_quant_method: str = Field("gptq", description="Default quantization method")
    quantization_bit_width: int = Field(4, description="Quantization bit width")
    cuda_memory_fraction: float = Field(0.85, description="Fraction of GPU memory to use")
    multi_gpu_strategy: str = Field("auto", description="Multi-GPU parallelism strategy")

class TogetherAIBackendConfig(BaseModel):
    """Configuration for Together.ai backend."""
    enabled: bool = Field(False, description="Whether Together.ai backend is enabled")
    api_key: str = Field("", description="Together.ai API key")
    default_model: str = Field("meta-llama/Llama-2-70b-chat-hf", description="Default model for completions")
    default_embedding_model: str = Field("togethercomputer/m2-bert-80M-8k-retrieval", description="Default model for embeddings")
    timeout: float = Field(60.0, description="Request timeout in seconds")
    endpoint_url: Optional[str] = Field(None, description="URL for dedicated endpoint if available")
    max_retries: int = Field(3, description="Maximum number of retries for API calls")

class CPUBackendConfig(BaseModel):
    """Configuration for CPU-only backend."""
    enabled: bool = Field(True, description="Whether CPU backend is enabled")
    default_model: str = Field("llama-2-7b-chat.Q4_K_M.gguf", description="Default model for CPU inference")
    default_embedding_model: str = Field("all-MiniLM-L6-v2", description="Default model for embeddings")
    num_threads: int = Field(4, description="Number of threads to use for CPU inference")
    context_size: int = Field(2048, description="Context size for CPU inference")

class BackendConfig(BaseModel):
    """Complete backend configuration for SAP HANA AI Toolkit."""
    priority: BackendPriority = Field(default_factory=BackendPriority, description="Backend priority and failover configuration")
    nvidia: NvidiaBackendConfig = Field(default_factory=NvidiaBackendConfig, description="NVIDIA backend configuration")
    together_ai: TogetherAIBackendConfig = Field(default_factory=TogetherAIBackendConfig, description="Together.ai backend configuration")
    cpu: CPUBackendConfig = Field(default_factory=CPUBackendConfig, description="CPU backend configuration")
    
    def determine_active_backends(self) -> List[BackendType]:
        """
        Determine which backends are active based on the configuration.
        
        Returns:
            List[BackendType]: List of active backend types.
        """
        active_backends = []
        
        # Check each backend
        if self.nvidia.enabled:
            active_backends.append(BackendType.NVIDIA)
        
        if self.together_ai.enabled and self.together_ai.api_key:
            active_backends.append(BackendType.TOGETHER_AI)
        
        if self.cpu.enabled:
            active_backends.append(BackendType.CPU)
        
        return active_backends
    
    def get_primary_backend(self) -> BackendType:
        """
        Get the primary backend based on configuration and availability.
        
        Returns:
            BackendType: The primary backend type.
        """
        active_backends = self.determine_active_backends()
        
        # If no backends are active, return CPU
        if not active_backends:
            logger.warning("No backends are active. Falling back to CPU.")
            return BackendType.CPU
        
        # If primary is AUTO, determine the best backend
        if self.priority.primary == BackendType.AUTO:
            # Prefer NVIDIA if available
            if BackendType.NVIDIA in active_backends:
                return BackendType.NVIDIA
            # Then Together.ai
            elif BackendType.TOGETHER_AI in active_backends:
                return BackendType.TOGETHER_AI
            # Then CPU
            else:
                return BackendType.CPU
        
        # If primary is specified and active, use it
        if self.priority.primary in active_backends:
            return self.priority.primary
        
        # If primary is specified but not active, use the first active backend
        logger.warning(f"Primary backend {self.priority.primary} is not active. Using {active_backends[0]} instead.")
        return active_backends[0]
    
    def get_secondary_backend(self) -> Optional[BackendType]:
        """
        Get the secondary backend based on configuration and availability.
        
        Returns:
            Optional[BackendType]: The secondary backend type, or None if not available.
        """
        active_backends = self.determine_active_backends()
        primary_backend = self.get_primary_backend()
        
        # Remove primary from active backends
        if primary_backend in active_backends:
            active_backends.remove(primary_backend)
        
        # If no other backends are active, return None
        if not active_backends:
            return None
        
        # If secondary is AUTO, determine the best remaining backend
        if not self.priority.secondary or self.priority.secondary == BackendType.AUTO:
            # Prefer NVIDIA if available
            if BackendType.NVIDIA in active_backends:
                return BackendType.NVIDIA
            # Then Together.ai
            elif BackendType.TOGETHER_AI in active_backends:
                return BackendType.TOGETHER_AI
            # Then CPU
            elif BackendType.CPU in active_backends:
                return BackendType.CPU
            else:
                return None
        
        # If secondary is specified and active, use it
        if self.priority.secondary in active_backends:
            return self.priority.secondary
        
        # If secondary is specified but not active, use the first active backend
        if active_backends:
            logger.warning(f"Secondary backend {self.priority.secondary} is not active. Using {active_backends[0]} instead.")
            return active_backends[0]
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return {
            "priority": self.priority.dict(),
            "nvidia": self.nvidia.dict(),
            "together_ai": self.together_ai.dict(),
            "cpu": self.cpu.dict(),
            "active_backends": self.determine_active_backends(),
            "primary_backend": self.get_primary_backend(),
            "secondary_backend": self.get_secondary_backend()
        }
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration.
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BackendConfig':
        """
        Load the configuration from a JSON file.
        
        Args:
            file_path: Path to load the configuration from.
            
        Returns:
            BackendConfig: The loaded configuration.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the model configs
        priority_data = data.get("priority", {})
        nvidia_data = data.get("nvidia", {})
        together_ai_data = data.get("together_ai", {})
        cpu_data = data.get("cpu", {})
        
        # Create the config
        return cls(
            priority=BackendPriority(**priority_data),
            nvidia=NvidiaBackendConfig(**nvidia_data),
            together_ai=TogetherAIBackendConfig(**together_ai_data),
            cpu=CPUBackendConfig(**cpu_data)
        )
    
    @classmethod
    def from_environment(cls) -> 'BackendConfig':
        """
        Create a configuration from environment variables.
        
        Returns:
            BackendConfig: The configuration.
        """
        # NVIDIA configuration
        nvidia_enabled = os.environ.get("ENABLE_GPU_ACCELERATION", "").lower() == "true"
        
        # Together.ai configuration
        together_ai_enabled = os.environ.get("ENABLE_TOGETHER_AI", "").lower() == "true"
        together_ai_api_key = os.environ.get("TOGETHER_API_KEY", "")
        
        # Create the config
        config = cls(
            priority=BackendPriority(
                primary=BackendType(os.environ.get("PRIMARY_BACKEND", "auto")),
                secondary=BackendType(os.environ.get("SECONDARY_BACKEND", "auto")) if os.environ.get("SECONDARY_BACKEND") else None,
                auto_failover=os.environ.get("AUTO_FAILOVER", "").lower() != "false",
                failover_attempts=int(os.environ.get("FAILOVER_ATTEMPTS", "3")),
                failover_timeout=float(os.environ.get("FAILOVER_TIMEOUT", "10.0")),
                load_balancing=os.environ.get("LOAD_BALANCING", "").lower() == "true",
                load_ratio=float(os.environ.get("LOAD_RATIO", "0.8"))
            ),
            nvidia=NvidiaBackendConfig(
                enabled=nvidia_enabled,
                enable_tensorrt=os.environ.get("ENABLE_TENSORRT", "").lower() != "false",
                enable_flash_attention=os.environ.get("ENABLE_FLASH_ATTENTION", "").lower() != "false",
                enable_transformer_engine=os.environ.get("ENABLE_TRANSFORMER_ENGINE", "").lower() != "false",
                enable_fp8=os.environ.get("ENABLE_FP8", "").lower() != "false",
                enable_gptq=os.environ.get("ENABLE_GPTQ", "").lower() != "false",
                enable_awq=os.environ.get("ENABLE_AWQ", "").lower() != "false",
                default_quant_method=os.environ.get("DEFAULT_QUANT_METHOD", "gptq"),
                quantization_bit_width=int(os.environ.get("QUANTIZATION_BIT_WIDTH", "4")),
                cuda_memory_fraction=float(os.environ.get("CUDA_MEMORY_FRACTION", "0.85")),
                multi_gpu_strategy=os.environ.get("MULTI_GPU_STRATEGY", "auto")
            ),
            together_ai=TogetherAIBackendConfig(
                enabled=together_ai_enabled,
                api_key=together_ai_api_key,
                default_model=os.environ.get("TOGETHER_DEFAULT_MODEL", "meta-llama/Llama-2-70b-chat-hf"),
                default_embedding_model=os.environ.get("TOGETHER_DEFAULT_EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval"),
                timeout=float(os.environ.get("TOGETHER_TIMEOUT", "60.0")),
                endpoint_url=os.environ.get("TOGETHER_ENDPOINT_URL"),
                max_retries=int(os.environ.get("TOGETHER_MAX_RETRIES", "3"))
            ),
            cpu=CPUBackendConfig(
                enabled=True,  # CPU is always enabled as fallback
                default_model=os.environ.get("CPU_DEFAULT_MODEL", "llama-2-7b-chat.Q4_K_M.gguf"),
                default_embedding_model=os.environ.get("CPU_DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                num_threads=int(os.environ.get("CPU_NUM_THREADS", "4")),
                context_size=int(os.environ.get("CPU_CONTEXT_SIZE", "2048"))
            )
        )
        
        return config


# Create a global backend configuration instance
backend_config = BackendConfig.from_environment()

# Export the backend configuration instance
__all__ = ["backend_config", "BackendConfig", "BackendType", "BackendPriority"]