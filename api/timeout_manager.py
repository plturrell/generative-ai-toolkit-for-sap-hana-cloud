"""
Timeout Manager for SAP HANA Cloud Generative AI Toolkit

This module provides timeout management for API requests to the T4 GPU backend,
allowing dynamic timeout adjustments based on endpoint complexity and load.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default timeouts (in seconds) for different endpoints
DEFAULT_TIMEOUTS = {
    "embeddings": 60,  # Embeddings generation can take longer for large inputs
    "search": 30,      # Vector search is typically faster
    "mmr_search": 45,  # MMR search is more complex than regular search
    "health": 10,      # Health checks should be quick
    "metrics": 15,     # Metrics collection might take a bit longer
    "default": 30      # Default timeout for unspecified endpoints
}

# Environment variable to override timeouts
TIMEOUT_CONFIG_ENV = os.getenv("TIMEOUT_CONFIG", "")

# Try to load custom timeouts from environment variable
try:
    if TIMEOUT_CONFIG_ENV:
        CUSTOM_TIMEOUTS = json.loads(TIMEOUT_CONFIG_ENV)
        # Validate and merge with defaults
        if isinstance(CUSTOM_TIMEOUTS, dict):
            for key, value in CUSTOM_TIMEOUTS.items():
                if isinstance(value, (int, float)) and value > 0:
                    DEFAULT_TIMEOUTS[key] = value
                else:
                    logger.warning(f"Invalid timeout value for {key}: {value}. Using default.")
        else:
            logger.warning(f"Invalid timeout configuration format. Using defaults.")
except json.JSONDecodeError:
    logger.warning(f"Could not parse TIMEOUT_CONFIG environment variable. Using defaults.")
except Exception as e:
    logger.warning(f"Error loading custom timeouts: {str(e)}. Using defaults.")

# Optional scaling factor for all timeouts (for environments with constrained resources)
TIMEOUT_SCALE_FACTOR = float(os.getenv("TIMEOUT_SCALE_FACTOR", "1.0"))

# Apply scaling factor if valid
try:
    if TIMEOUT_SCALE_FACTOR > 0:
        for key in DEFAULT_TIMEOUTS:
            DEFAULT_TIMEOUTS[key] = int(DEFAULT_TIMEOUTS[key] * TIMEOUT_SCALE_FACTOR)
    else:
        logger.warning(f"Invalid TIMEOUT_SCALE_FACTOR: {TIMEOUT_SCALE_FACTOR}. Using 1.0.")
        TIMEOUT_SCALE_FACTOR = 1.0
except:
    logger.warning(f"Error applying timeout scale factor. Using defaults.")
    TIMEOUT_SCALE_FACTOR = 1.0

# Log the final timeout configuration at startup
logger.info(f"Timeout configuration (scale factor: {TIMEOUT_SCALE_FACTOR}):")
for endpoint, timeout in DEFAULT_TIMEOUTS.items():
    logger.info(f"  {endpoint}: {timeout}s")

def get_timeout(endpoint: str = "default") -> int:
    """
    Get the appropriate timeout for a specific endpoint
    
    Args:
        endpoint: The endpoint name or path
        
    Returns:
        Timeout value in seconds
    """
    # Extract endpoint name from path if necessary
    if endpoint.startswith("/"):
        parts = endpoint.strip("/").split("/")
        endpoint = parts[-1] if len(parts) > 0 else "default"
    
    # For paths like vectorstore/search, check the last part first, then the whole path
    if "/" in endpoint:
        parts = endpoint.split("/")
        last_part = parts[-1]
        if last_part in DEFAULT_TIMEOUTS:
            return DEFAULT_TIMEOUTS[last_part]
    
    # Return the specific timeout or the default
    return DEFAULT_TIMEOUTS.get(endpoint, DEFAULT_TIMEOUTS["default"])

def get_all_timeouts() -> Dict[str, int]:
    """
    Get all configured timeouts
    
    Returns:
        Dictionary of all endpoint timeouts
    """
    return DEFAULT_TIMEOUTS.copy()

def update_timeout(endpoint: str, value: int) -> bool:
    """
    Update the timeout for a specific endpoint (for runtime adjustments)
    
    Args:
        endpoint: The endpoint name
        value: New timeout value in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if value > 0:
        DEFAULT_TIMEOUTS[endpoint] = value
        logger.info(f"Updated timeout for {endpoint} to {value}s")
        return True
    else:
        logger.warning(f"Invalid timeout value for {endpoint}: {value}")
        return False

# Auto-adjust timeouts based on load
# This is a simple implementation that could be expanded with more sophisticated logic
def auto_adjust_timeouts(load_factor: float) -> None:
    """
    Automatically adjust timeouts based on system load
    
    Args:
        load_factor: A value between 0.0 and 1.0 representing system load
                    (0.0 = idle, 1.0 = maximum load)
    """
    if not (0.0 <= load_factor <= 1.0):
        logger.warning(f"Invalid load factor: {load_factor}. Must be between 0.0 and 1.0.")
        return
    
    # Increase timeouts as load increases
    # This simple algorithm doubles timeouts at max load
    scale = 1.0 + load_factor
    
    # Apply to all endpoints
    for endpoint in DEFAULT_TIMEOUTS:
        # Get original timeout (before any load adjustment)
        original = DEFAULT_TIMEOUTS[endpoint] / TIMEOUT_SCALE_FACTOR
        
        # Apply load-based scaling
        DEFAULT_TIMEOUTS[endpoint] = int(original * scale * TIMEOUT_SCALE_FACTOR)
    
    logger.info(f"Adjusted timeouts for load factor {load_factor}")