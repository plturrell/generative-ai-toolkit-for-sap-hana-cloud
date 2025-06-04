"""
Environment detection for the SAP HANA AI Toolkit.

This module provides functions to detect the current deployment environment
and configure the application accordingly.
"""

import os
import sys
import logging
import platform
from enum import Enum
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class DeploymentPlatform(str, Enum):
    """Enumeration of supported deployment platforms."""
    NVIDIA_LAUNCHPAD = "nvidia_launchpad"
    TOGETHER_AI = "together_ai"
    SAP_BTP = "sap_btp"
    SAP_KYMA = "sap_kyma"
    VERCEL = "vercel"
    DOCKER = "docker"
    LOCAL = "local"
    AUTO = "auto"
    UNKNOWN = "unknown"

def detect_deployment_platform() -> DeploymentPlatform:
    """
    Detect the current deployment platform based on environment variables and system properties.
    
    Returns:
        DeploymentPlatform: The detected deployment platform
    """
    # Check for explicit platform setting
    explicit_platform = os.environ.get("DEPLOYMENT_PLATFORM", "")
    if explicit_platform and explicit_platform != "auto":
        try:
            return DeploymentPlatform(explicit_platform)
        except ValueError:
            logger.warning(f"Invalid deployment platform specified: {explicit_platform}")
    
    # Check for NVIDIA LaunchPad
    if os.environ.get("NGC_JOB_ID") or os.environ.get("NGC_WORKSPACE"):
        return DeploymentPlatform.NVIDIA_LAUNCHPAD
    
    # Check for Together.ai
    if os.environ.get("TOGETHER_DEPLOYMENT_ID") or os.environ.get("TOGETHER_ENDPOINT_ID"):
        return DeploymentPlatform.TOGETHER_AI
    
    # Check for SAP BTP Cloud Foundry
    if os.environ.get("VCAP_APPLICATION") or os.environ.get("CF_INSTANCE_GUID"):
        return DeploymentPlatform.SAP_BTP
    
    # Check for SAP Kyma
    if os.environ.get("KUBERNETES_SERVICE_HOST") and os.environ.get("KYMA_RUNTIME"):
        return DeploymentPlatform.SAP_KYMA
    
    # Check for Vercel
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
        return DeploymentPlatform.VERCEL
    
    # Check for Docker
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        return DeploymentPlatform.DOCKER
    
    # Check for local development
    if os.environ.get("DEVELOPMENT_MODE") == "true":
        return DeploymentPlatform.LOCAL
    
    # Fallback to unknown
    return DeploymentPlatform.UNKNOWN

def get_platform_configuration(platform: Optional[DeploymentPlatform] = None) -> Dict[str, Any]:
    """
    Get platform-specific configuration defaults.
    
    Args:
        platform: The deployment platform to get configuration for.
                 If None, the current platform will be detected.
    
    Returns:
        Dict[str, Any]: Platform-specific configuration defaults
    """
    if platform is None:
        platform = detect_deployment_platform()
    
    # Base configuration
    config = {
        "platform": platform.value,
        "deployment_mode": "full",  # Default to full mode
        "cors_origins": ["*"],      # Default to allow all origins
        "api_base_url": "",         # Default to same-origin
        "frontend_url": "",         # Default to same-origin
    }
    
    # Platform-specific configuration
    if platform == DeploymentPlatform.NVIDIA_LAUNCHPAD:
        config.update({
            "deployment_mode": "api_only",  # NGC typically runs API only
            "enable_gpu_acceleration": True,
            "enable_tensorrt": True,
            "enable_flash_attention": True,
            "enable_transformer_engine": True,
            "cors_origins": ["*"],  # Allow all origins by default for API mode
        })
    
    elif platform == DeploymentPlatform.TOGETHER_AI:
        config.update({
            "deployment_mode": "api_only",  # Together.ai runs API only
            "enable_gpu_acceleration": False,  # No local GPU
            "enable_together_ai": True,
            "cors_origins": ["*"],  # Allow all origins by default for API mode
        })
    
    elif platform == DeploymentPlatform.SAP_BTP:
        config.update({
            "cors_origins": ["*.cfapps.*.hana.ondemand.com", "*.hana.ondemand.com"],
            # Can be either API-only or full based on configuration
        })
    
    elif platform == DeploymentPlatform.SAP_KYMA:
        config.update({
            "cors_origins": ["*.kyma-system.svc.cluster.local", "*.kyma.shoot.live.k8s-hana.ondemand.com"],
            # Can be either API-only or full based on configuration
        })
    
    elif platform == DeploymentPlatform.VERCEL:
        config.update({
            "deployment_mode": "ui_only",  # Vercel typically runs UI only
            "enable_gpu_acceleration": False,  # No local GPU
            "enable_together_ai": True,  # Use Together.ai by default on Vercel
            "cors_origins": ["*.vercel.app"],
        })
    
    return config

def apply_platform_defaults(settings: Any) -> None:
    """
    Apply platform-specific defaults to the provided settings object.
    
    Args:
        settings: The settings object to apply defaults to
    """
    platform = detect_deployment_platform()
    logger.info(f"Detected deployment platform: {platform}")
    
    # Get platform-specific configuration
    platform_config = get_platform_configuration(platform)
    
    # Apply defaults for any settings that haven't been explicitly set
    for key, value in platform_config.items():
        env_key = key.upper()
        # Only apply if not already set in environment
        if not os.environ.get(env_key) and hasattr(settings, env_key):
            logger.debug(f"Applying platform default for {env_key}: {value}")
            setattr(settings, env_key, value)
    
    # Special handling for CORS origins
    if platform_config.get("cors_origins") and not os.environ.get("CORS_ORIGINS"):
        settings.CORS_ORIGINS = platform_config["cors_origins"]

def get_platform_info() -> Dict[str, Any]:
    """
    Get detailed information about the current deployment platform.
    
    Returns:
        Dict[str, Any]: Detailed platform information
    """
    platform = detect_deployment_platform()
    
    info = {
        "platform": platform.value,
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
        },
        "environment": {
            "deployment_mode": os.environ.get("DEPLOYMENT_MODE", "full"),
            "gpu_acceleration": os.environ.get("ENABLE_GPU_ACCELERATION", "false").lower() == "true",
            "together_ai": os.environ.get("ENABLE_TOGETHER_AI", "false").lower() == "true",
        }
    }
    
    # Platform-specific info
    if platform == DeploymentPlatform.NVIDIA_LAUNCHPAD:
        try:
            import torch
            info["gpu"] = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": [
                    {
                        "name": torch.cuda.get_device_name(i),
                        "memory": {
                            "total": torch.cuda.get_device_properties(i).total_memory,
                            "free": torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i),
                            "used": torch.cuda.memory_allocated(i),
                        }
                    }
                    for i in range(torch.cuda.device_count()) if torch.cuda.is_available()
                ]
            }
        except ImportError:
            info["gpu"] = {"available": False, "error": "PyTorch not available"}
    
    elif platform == DeploymentPlatform.SAP_BTP:
        vcap_app = os.environ.get("VCAP_APPLICATION", "{}")
        try:
            vcap_app_json = json.loads(vcap_app)
            info["btp"] = {
                "application_id": vcap_app_json.get("application_id"),
                "application_name": vcap_app_json.get("application_name"),
                "space_id": vcap_app_json.get("space_id"),
                "space_name": vcap_app_json.get("space_name"),
                "organization_id": vcap_app_json.get("organization_id"),
                "organization_name": vcap_app_json.get("organization_name"),
            }
        except Exception as e:
            info["btp"] = {"error": f"Failed to parse VCAP_APPLICATION: {str(e)}"}
    
    return info