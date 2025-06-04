"""
Environment constants for the HANA AI Toolkit API.

This module defines all the environment variables and constants
used by the application, separated from the code.
"""

# SAP AI Core models
SAP_AI_CORE_LLM_MODEL = "sap-ai-core-llama3"
SAP_AI_CORE_EMBEDDING_MODEL = "sap-ai-core-embeddings"

# SAP BTP domains
BTP_DOMAIN_PATTERNS = [
    r'.*\.hana\.ondemand\.com$',       # SAP BTP domains
    r'.*\.cfapps\..*\.hana\.ondemand\.com$',  # SAP BTP Cloud Foundry apps
    r'.*\.sap\.com$',                  # SAP domains
]

# SAP BTP IP ranges
BTP_IP_RANGES = [
    # SAP Cloud Foundry ranges
    '130.214.0.0/16',   # SAP CF EU10 (Frankfurt)
    '157.133.0.0/16',   # SAP CF US10 (VA)
    '138.112.0.0/16',   # SAP CF AP10 (Sydney) 
    '141.82.0.0/16',    # SAP CF JP10 (Tokyo)
    '159.122.0.0/16',   # SAP CF US20 (Washington/Dallas)
    '169.53.0.0/16',    # SAP CF EU20 (Amsterdam)
    '168.1.0.0/16',     # SAP CF BR10 (São Paulo)
    '161.156.0.0/16',   # SAP CF AP11 (Singapore)
    '147.204.0.0/16',   # SAP CF AP12 (Seoul)
    '149.81.0.0/16',    # SAP CF CH20 (Switzerland)
    '148.77.0.0/16',    # SAP CF CA10 (Montreal)
    # SAP HANA Cloud ranges
    '18.194.0.0/15',    # SAP HANA Cloud EU10
    '13.237.0.0/16',    # SAP HANA Cloud AP10
    '54.198.0.0/16',    # SAP HANA Cloud US10
    # Private network ranges used in BTP VPC
    '10.0.0.0/8',       # Private network range 
    '172.16.0.0/12',    # Private network range
    '192.168.0.0/16',   # Private network range
]

# Default circuit breaker settings
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 30.0  # seconds

# Default retry settings
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0

# Default bulkhead settings
DEFAULT_BULKHEAD_MAX_CONCURRENT = 20
DEFAULT_BULKHEAD_MAX_WAITING = 40

# Default timeout settings
DEFAULT_TIMEOUT_SECONDS = 30.0  # seconds

# GPU Optimization defaults
DEFAULT_GPU_MEMORY_FRACTION = 0.8
DEFAULT_NVIDIA_VISIBLE_DEVICES = "all"
DEFAULT_NVIDIA_DRIVER_CAPABILITIES = "compute,utility"
DEFAULT_MULTI_GPU_STRATEGY = "data_parallel"
GPU_STRATEGIES = ["data_parallel", "model_parallel", "pipeline_parallel", "auto"]

# NVIDIA Hopper (H100) specific settings
HOPPER_ENABLE_FLASH_ATTENTION = True
HOPPER_ENABLE_FP8 = True
HOPPER_ENABLE_TRANSFORMER_ENGINE = True
HOPPER_ENABLE_FSDP = True

# TensorRT optimization settings
DEFAULT_TENSORRT_CACHE_DIR = "/tmp/tensorrt_engines"
DEFAULT_TENSORRT_PRECISION = "fp16"  # Options: fp32, fp16, int8
DEFAULT_TENSORRT_MAX_BATCH_SIZE = 32
DEFAULT_TENSORRT_WORKSPACE_SIZE_MB = 1024  # 1GB workspace size
DEFAULT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL = 3  # 0-5, with 5 being the highest
ENABLE_TENSORRT = True

# API security defaults
DEFAULT_RATE_LIMIT = 100
DEFAULT_MAX_REQUEST_SIZE_MB = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300

# Connection pool defaults
DEFAULT_CONNECTION_POOL_SIZE = 5
DEFAULT_CONNECTION_MAX_OVERFLOW = 20
DEFAULT_CONNECTION_POOL_TIMEOUT = 30.0  # seconds
DEFAULT_CONNECTION_POOL_RECYCLE = 1800.0  # 30 minutes

# Logging and metrics defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"
DEFAULT_PROMETHEUS_PORT = 9090
METRICS_ENABLED = True
METRICS_ENDPOINT = "/metrics"
METRICS_INTERNAL_ONLY = True

# Memory settings
DEFAULT_MEMORY_EXPIRATION_SECONDS = 3600

# Deployment settings
DEPLOYMENT_TYPE_PRODUCTION = "production"
DEPLOYMENT_TYPE_CANARY = "canary"
DEPLOYMENT_TYPE_DEV = "development"
VALID_DEPLOYMENT_TYPES = [DEPLOYMENT_TYPE_PRODUCTION, DEPLOYMENT_TYPE_CANARY, DEPLOYMENT_TYPE_DEV]

# Canary deployment settings
DEFAULT_CANARY_WEIGHT = 20  # percentage
DEFAULT_CANARY_STEP_PERCENTAGE = 20
DEFAULT_CANARY_PROMOTION_DELAY = 3600  # seconds (1 hour)
DEFAULT_CANARY_ROLLBACK_THRESHOLD = 5  # number of errors before rollback

# Configuration settings
DEFAULT_CONFIG_DIR = "/etc/hana-ai-toolkit/config"

# Deployment modes
DEPLOYMENT_MODE_FULL = "full"            # Both API and UI components
DEPLOYMENT_MODE_API_ONLY = "api_only"    # Only API components without UI
DEPLOYMENT_MODE_UI_ONLY = "ui_only"      # Only UI components, connects to external API
DEFAULT_DEPLOYMENT_MODE = DEPLOYMENT_MODE_FULL

# Frontend/Backend configuration
DEFAULT_API_BASE_URL = ""                # Default empty for same-origin
DEFAULT_FRONTEND_URL = ""                # Default empty for same-origin