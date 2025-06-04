"""
Configuration for the SAP HANA AI Toolkit API.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """API configuration settings."""
    
    # API settings
    API_TITLE: str = "SAP HANA AI Toolkit API"
    API_DESCRIPTION: str = "API for the Generative AI Toolkit for SAP HANA Cloud"
    API_VERSION: str = "1.0.0"
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Environment settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEVELOPMENT_MODE: bool = ENVIRONMENT.lower() in ("dev", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = os.getenv("API_KEYS", "dev-key").split(",")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    SESSION_SECRET_KEY: str = os.getenv("SESSION_SECRET_KEY", "supersecretkey")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    ENFORCE_HTTPS: bool = os.getenv("ENFORCE_HTTPS", "False").lower() in ("true", "1", "yes")
    RESTRICT_EXTERNAL_CALLS: bool = os.getenv("RESTRICT_EXTERNAL_CALLS", "True").lower() in ("true", "1", "yes")
    
    # HANA Database settings
    HANA_HOST: Optional[str] = os.getenv("HANA_HOST")
    HANA_PORT: Optional[int] = int(os.getenv("HANA_PORT", "0")) or None
    HANA_USER: Optional[str] = os.getenv("HANA_USER")
    HANA_PASSWORD: Optional[str] = os.getenv("HANA_PASSWORD")
    HANA_ENCRYPT: bool = os.getenv("HANA_ENCRYPT", "True").lower() in ("true", "1", "yes")
    HANA_SSL_VALIDATE_CERT: bool = os.getenv("HANA_SSL_VALIDATE_CERT", "False").lower() in ("true", "1", "yes")
    
    # GPU settings
    ENABLE_GPU_ACCELERATION: bool = os.getenv("ENABLE_GPU_ACCELERATION", "True").lower() in ("true", "1", "yes")
    NVIDIA_CUDA_VISIBLE_DEVICES: str = os.getenv("NVIDIA_CUDA_VISIBLE_DEVICES", "all")
    NVIDIA_CUDA_DEVICE_ORDER: str = os.getenv("NVIDIA_CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    NVIDIA_TF32_OVERRIDE: int = int(os.getenv("NVIDIA_TF32_OVERRIDE", "1"))
    NVIDIA_CUDA_CACHE_MAXSIZE: int = int(os.getenv("NVIDIA_CUDA_CACHE_MAXSIZE", "1073741824"))
    NVIDIA_CUDA_CACHE_PATH: str = os.getenv("NVIDIA_CUDA_CACHE_PATH", "/tmp/cuda_cache")
    CUDA_MEMORY_FRACTION: float = float(os.getenv("CUDA_MEMORY_FRACTION", "0.9"))
    MULTI_GPU_STRATEGY: str = os.getenv("MULTI_GPU_STRATEGY", "data_parallel")
    
    # TensorRT settings
    ENABLE_TENSORRT: bool = os.getenv("ENABLE_TENSORRT", "True").lower() in ("true", "1", "yes")
    TENSORRT_CACHE_DIR: str = os.getenv("TENSORRT_CACHE_DIR", "/tmp/tensorrt_engines")
    TENSORRT_FP16_MODE: bool = os.getenv("TENSORRT_FP16_MODE", "True").lower() in ("true", "1", "yes")
    TENSORRT_INT8_MODE: bool = os.getenv("TENSORRT_INT8_MODE", "False").lower() in ("true", "1", "yes")
    TENSORRT_MAX_WORKSPACE_SIZE: int = int(os.getenv("TENSORRT_MAX_WORKSPACE_SIZE", "1073741824"))
    
    # Model settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    MODEL_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "60"))
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "sap-ai-core-llama3")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sap-ai-core-embeddings")
    
    # Advanced settings
    ENABLE_TELEMETRY: bool = os.getenv("ENABLE_TELEMETRY", "True").lower() in ("true", "1", "yes")
    PROMETHEUS_ENDPOINT: Optional[str] = os.getenv("PROMETHEUS_ENDPOINT")
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Configure logging
logging_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Log basic configuration information
logger = logging.getLogger(__name__)
logger.info(f"Environment: {settings.ENVIRONMENT}")
logger.info(f"Debug mode: {settings.DEBUG}")
logger.info(f"GPU acceleration: {settings.ENABLE_GPU_ACCELERATION}")
logger.info(f"TensorRT enabled: {settings.ENABLE_TENSORRT}")

# Log warning if API keys are using default values
if "dev-key" in settings.API_KEYS:
    logger.warning("Using default API key. For production, set a secure API_KEYS environment variable.")

# Log warning if using default secret key
if settings.SECRET_KEY == "supersecretkey":
    logger.warning("Using default SECRET_KEY. For production, set a secure SECRET_KEY environment variable.")

# Log warning if HANA credentials not provided
if not all([settings.HANA_HOST, settings.HANA_PORT, settings.HANA_USER, settings.HANA_PASSWORD]):
    logger.warning("HANA database credentials not fully configured. Some features may be unavailable.")

def get_tensorrt_config() -> Dict[str, Any]:
    """Get TensorRT configuration from settings."""
    return {
        "fp16_mode": settings.TENSORRT_FP16_MODE,
        "int8_mode": settings.TENSORRT_INT8_MODE,
        "max_workspace_size": settings.TENSORRT_MAX_WORKSPACE_SIZE,
        "cache_dir": settings.TENSORRT_CACHE_DIR
    }