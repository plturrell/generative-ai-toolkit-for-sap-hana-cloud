"""
Configuration settings for the HANA AI Toolkit API.
"""
import os
import secrets
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field

from .env_constants import (
    SAP_AI_CORE_LLM_MODEL,
    SAP_AI_CORE_EMBEDDING_MODEL,
    BTP_DOMAIN_PATTERNS,
    BTP_IP_RANGES,
    DEFAULT_GPU_MEMORY_FRACTION,
    DEFAULT_NVIDIA_VISIBLE_DEVICES,
    DEFAULT_NVIDIA_DRIVER_CAPABILITIES,
    DEFAULT_RATE_LIMIT,
    DEFAULT_MAX_REQUEST_SIZE_MB,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_CONNECTION_POOL_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_PROMETHEUS_PORT,
    DEFAULT_MEMORY_EXPIRATION_SECONDS,
    DEFAULT_TENSORRT_CACHE_DIR,
    DEFAULT_TENSORRT_PRECISION,
    DEFAULT_TENSORRT_MAX_BATCH_SIZE,
    DEFAULT_TENSORRT_WORKSPACE_SIZE_MB,
    DEFAULT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
    ENABLE_TENSORRT
)

class Settings(BaseSettings):
    """API settings loaded from environment variables with defaults."""
    
    # API Server Settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    DEVELOPMENT_MODE: bool = Field(default=False, env="DEVELOPMENT_MODE")
    LOG_LEVEL: str = Field(default=DEFAULT_LOG_LEVEL, env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default=DEFAULT_LOG_FORMAT, env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security Settings
    API_KEYS: List[str] = Field(
        default_factory=lambda: os.environ.get("API_KEYS", "").split(",") if os.environ.get("API_KEYS") else ["dev-key-only-for-testing"]
    )
    AUTH_REQUIRED: bool = Field(default=True, env="AUTH_REQUIRED")
    # Default CORS to only allow SAP BTP domains
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: os.environ.get("CORS_ORIGINS", "*.cfapps.*.hana.ondemand.com,*.hana.ondemand.com").split(",") 
                        if os.environ.get("CORS_ORIGINS") 
                        else ["*.cfapps.*.hana.ondemand.com", "*.hana.ondemand.com"] # Only SAP BTP domains
    )
    ENFORCE_HTTPS: bool = Field(default=True, env="ENFORCE_HTTPS")
    SESSION_SECRET_KEY: str = Field(
        default_factory=lambda: os.environ.get("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
    )
    # Security boundary enforcement
    RESTRICT_EXTERNAL_CALLS: bool = Field(default=True, env="RESTRICT_EXTERNAL_CALLS")
    
    # Rate Limiting Settings
    RATE_LIMIT_PER_MINUTE: int = Field(default=DEFAULT_RATE_LIMIT, env="RATE_LIMIT_PER_MINUTE")
    
    # Database Connection Settings
    HANA_HOST: str = Field(default="", env="HANA_HOST")
    HANA_PORT: int = Field(default=443, env="HANA_PORT")
    HANA_USER: str = Field(default="", env="HANA_USER")
    HANA_PASSWORD: str = Field(default="", env="HANA_PASSWORD")
    HANA_USERKEY: Optional[str] = Field(default=None, env="HANA_USERKEY")
    
    # LLM Settings - Using SAP AI Core models only
    DEFAULT_LLM_MODEL: str = Field(default=SAP_AI_CORE_LLM_MODEL, env="DEFAULT_LLM_MODEL")
    DEFAULT_LLM_TEMPERATURE: float = Field(default=0.0, env="DEFAULT_LLM_TEMPERATURE")
    DEFAULT_LLM_MAX_TOKENS: int = Field(default=1000, env="DEFAULT_LLM_MAX_TOKENS")
    
    # NVIDIA GPU Optimization Settings - Basic
    ENABLE_GPU_ACCELERATION: bool = Field(default=True, env="ENABLE_GPU_ACCELERATION")
    NVIDIA_VISIBLE_DEVICES: str = Field(default=DEFAULT_NVIDIA_VISIBLE_DEVICES, env="NVIDIA_VISIBLE_DEVICES")
    NVIDIA_DRIVER_CAPABILITIES: str = Field(default=DEFAULT_NVIDIA_DRIVER_CAPABILITIES, env="NVIDIA_DRIVER_CAPABILITIES")
    CUDA_MEMORY_FRACTION: float = Field(default=DEFAULT_GPU_MEMORY_FRACTION, env="CUDA_MEMORY_FRACTION")
    
    # NVIDIA GPU Optimization Settings - Advanced
    NVIDIA_CUDA_DEVICE_ORDER: str = Field(default="PCI_BUS_ID", env="NVIDIA_CUDA_DEVICE_ORDER")
    NVIDIA_CUDA_VISIBLE_DEVICES: str = Field(default="0", env="NVIDIA_CUDA_VISIBLE_DEVICES")
    NVIDIA_TF32_OVERRIDE: int = Field(default=1, env="NVIDIA_TF32_OVERRIDE")
    NVIDIA_CUDA_CACHE_MAXSIZE: int = Field(default=2147483648, env="NVIDIA_CUDA_CACHE_MAXSIZE")
    NVIDIA_CUDA_CACHE_PATH: str = Field(default="/tmp/cuda-cache", env="NVIDIA_CUDA_CACHE_PATH")
    
    # Multi-GPU Distribution Settings
    MULTI_GPU_STRATEGY: str = Field(default="auto", env="MULTI_GPU_STRATEGY")
    ENABLE_TENSOR_PARALLELISM: bool = Field(default=True, env="ENABLE_TENSOR_PARALLELISM")
    ENABLE_PIPELINE_PARALLELISM: bool = Field(default=True, env="ENABLE_PIPELINE_PARALLELISM")
    GPU_BATCH_SIZE_OPTIMIZATION: bool = Field(default=True, env="GPU_BATCH_SIZE_OPTIMIZATION")
    
    # Advanced Kernel Optimization Settings
    ENABLE_CUDA_GRAPHS: bool = Field(default=True, env="ENABLE_CUDA_GRAPHS")
    ENABLE_KERNEL_FUSION: bool = Field(default=True, env="ENABLE_KERNEL_FUSION")
    ENABLE_FLASH_ATTENTION: bool = Field(default=True, env="ENABLE_FLASH_ATTENTION")
    CHECKPOINT_ACTIVATIONS: bool = Field(default=True, env="CHECKPOINT_ACTIVATIONS")
    
    # TensorRT Optimization Settings
    ENABLE_TENSORRT: bool = Field(default=ENABLE_TENSORRT, env="ENABLE_TENSORRT")
    TENSORRT_CACHE_DIR: str = Field(default=DEFAULT_TENSORRT_CACHE_DIR, env="TENSORRT_CACHE_DIR")
    TENSORRT_WORKSPACE_SIZE_MB: int = Field(default=DEFAULT_TENSORRT_WORKSPACE_SIZE_MB, env="TENSORRT_WORKSPACE_SIZE_MB") # 1 GB
    TENSORRT_PRECISION: str = Field(default=DEFAULT_TENSORRT_PRECISION, env="TENSORRT_PRECISION")  # fp16, fp32, or int8
    TENSORRT_MAX_BATCH_SIZE: int = Field(default=DEFAULT_TENSORRT_MAX_BATCH_SIZE, env="TENSORRT_MAX_BATCH_SIZE")
    TENSORRT_BUILDER_OPTIMIZATION_LEVEL: int = Field(default=DEFAULT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL, env="TENSORRT_BUILDER_OPTIMIZATION_LEVEL") # 0-5, higher is better
    
    # Memory Settings
    ENABLE_MEMORY: bool = Field(default=True, env="ENABLE_MEMORY")
    MEMORY_EXPIRATION_SECONDS: int = Field(default=DEFAULT_MEMORY_EXPIRATION_SECONDS, env="MEMORY_EXPIRATION_SECONDS")
    
    # Performance Settings
    CONNECTION_POOL_SIZE: int = Field(default=DEFAULT_CONNECTION_POOL_SIZE, env="CONNECTION_POOL_SIZE")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=DEFAULT_REQUEST_TIMEOUT_SECONDS, env="REQUEST_TIMEOUT_SECONDS")
    MAX_REQUEST_SIZE_MB: int = Field(default=DEFAULT_MAX_REQUEST_SIZE_MB, env="MAX_REQUEST_SIZE_MB")
    
    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=DEFAULT_PROMETHEUS_PORT, env="PROMETHEUS_PORT")
    OPENTELEMETRY_ENABLED: bool = Field(default=False, env="OPENTELEMETRY_ENABLED")
    OPENTELEMETRY_ENDPOINT: Optional[str] = Field(default=None, env="OPENTELEMETRY_ENDPOINT")
    
    # Cache Settings
    ENABLE_CACHING: bool = Field(default=True, env="ENABLE_CACHING")
    CACHE_TTL_SECONDS: int = Field(default=300, env="CACHE_TTL_SECONDS")  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # Load from environment first, then .env file, then init arguments
            return env_settings, file_secret_settings, init_settings

# Create settings instance
settings = Settings()