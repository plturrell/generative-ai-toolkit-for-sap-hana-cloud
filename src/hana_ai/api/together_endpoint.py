"""
Together.ai dedicated endpoint integration for deploying GPU-accelerated
models with the SAP HANA AI Toolkit.
"""

import os
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any, Union
import yaml

# Configure logging
logger = logging.getLogger(__name__)

# Together.ai API constants
TOGETHER_API_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_DEPLOYMENTS_ENDPOINT = "/deployments"
TOGETHER_DEPLOYMENT_STATUS_ENDPOINT = "/deployments/{deployment_id}/status"

class TogetherEndpointDeployer:
    """
    Handles deployment of models to Together.ai dedicated endpoints.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the Together.ai endpoint deployer.
        
        Args:
            api_key: Together.ai API key. Defaults to TOGETHER_API_KEY environment variable.
            config_path: Path to the Together.ai deployment config file.
        """
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        self.config_path = config_path or os.environ.get("TOGETHER_CONFIG_PATH", "together.yaml")
        
        # Check if API key is provided
        if not self.api_key:
            logger.warning(
                "Together.ai API key not provided. Set the TOGETHER_API_KEY environment variable "
                "or pass it directly to the constructor."
            )
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load the Together.ai deployment configuration from YAML file.
        
        Returns:
            Dict: The deployment configuration.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def create_deployment(self) -> Dict[str, Any]:
        """
        Create a new dedicated endpoint deployment on Together.ai.
        
        Returns:
            Dict: The deployment response.
        """
        config = self.load_config()
        
        # Extract endpoint configuration
        endpoint_config = config.get("endpoint", {})
        
        # Prepare deployment request
        deployment_request = {
            "name": endpoint_config.get("name", "sap-hana-ai-toolkit"),
            "description": endpoint_config.get("description", "SAP HANA AI Toolkit Deployment"),
            "model": {
                "name": endpoint_config.get("model", {}).get("baseModel", "meta-llama/Llama-2-70b-chat-hf"),
                "quantization": {
                    "enabled": endpoint_config.get("model", {}).get("quantization", {}).get("enabled", True),
                    "method": endpoint_config.get("model", {}).get("quantization", {}).get("method", "awq"),
                    "bits": endpoint_config.get("model", {}).get("quantization", {}).get("bits", 4)
                }
            },
            "hardware": {
                "instance_type": endpoint_config.get("hardware", {}).get("instanceType", "a100-40gb"),
                "count": endpoint_config.get("hardware", {}).get("count", 1)
            },
            "scaling": {
                "min_replicas": endpoint_config.get("scaling", {}).get("minReplicas", 1),
                "max_replicas": endpoint_config.get("scaling", {}).get("maxReplicas", 2),
                "target_utilization": endpoint_config.get("scaling", {}).get("targetUtilization", 80)
            },
            "serving": {
                "max_tokens": endpoint_config.get("model", {}).get("serving", {}).get("maxTokens", 4096),
                "max_batch_size": endpoint_config.get("model", {}).get("serving", {}).get("maxBatchSize", 32),
                "max_concurrent_requests": endpoint_config.get("model", {}).get("serving", {}).get("maxConcurrentRequests", 10),
                "timeout": endpoint_config.get("model", {}).get("serving", {}).get("timeout", 120)
            },
            "advanced": {
                "enable_tensorrt": endpoint_config.get("advanced", {}).get("enableTensorRT", True),
                "enable_flash_attention": endpoint_config.get("advanced", {}).get("enableFlashAttention", True),
                "enable_kv_caching": endpoint_config.get("advanced", {}).get("enableKVCaching", True),
                "enable_continuous_batching": endpoint_config.get("advanced", {}).get("enableContinuousBatching", True),
                "scheduling_strategy": endpoint_config.get("advanced", {}).get("schedulingStrategy", "fair")
            }
        }
        
        # Add network configuration if present
        if endpoint_config.get("network", {}).get("privateAccess", False):
            deployment_request["network"] = {
                "private_access": True,
                "allowed_ips": endpoint_config.get("network", {}).get("allowedIPs", [])
            }
        
        # Send deployment request
        try:
            response = self.session.post(
                f"{TOGETHER_API_BASE_URL}{TOGETHER_DEPLOYMENTS_ENDPOINT}",
                json=deployment_request
            )
            response.raise_for_status()
            
            logger.info(f"Deployment request submitted successfully")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create deployment: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the status of a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            
        Returns:
            Dict: The deployment status.
        """
        try:
            response = self.session.get(
                f"{TOGETHER_API_BASE_URL}{TOGETHER_DEPLOYMENT_STATUS_ENDPOINT.format(deployment_id=deployment_id)}"
            )
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get deployment status: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def wait_for_deployment(self, deployment_id: str, timeout: int = 3600, interval: int = 30) -> Dict[str, Any]:
        """
        Wait for a deployment to be ready.
        
        Args:
            deployment_id: The ID of the deployment.
            timeout: Maximum time to wait in seconds.
            interval: Polling interval in seconds.
            
        Returns:
            Dict: The final deployment status.
        """
        logger.info(f"Waiting for deployment {deployment_id} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_deployment_status(deployment_id)
            
            if status.get("status") == "ready":
                logger.info(f"Deployment {deployment_id} is ready!")
                return status
            elif status.get("status") == "failed":
                logger.error(f"Deployment {deployment_id} failed: {status.get('message')}")
                raise Exception(f"Deployment failed: {status.get('message')}")
            
            logger.info(f"Deployment status: {status.get('status')}. Waiting {interval} seconds...")
            time.sleep(interval)
        
        raise TimeoutError(f"Timed out waiting for deployment {deployment_id} to be ready")
    
    def delete_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Delete a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            
        Returns:
            Dict: The deletion response.
        """
        try:
            response = self.session.delete(
                f"{TOGETHER_API_BASE_URL}{TOGETHER_DEPLOYMENTS_ENDPOINT}/{deployment_id}"
            )
            response.raise_for_status()
            
            logger.info(f"Deployment {deployment_id} deleted successfully")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete deployment: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def update_deployment(self, deployment_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            updates: The updates to apply.
            
        Returns:
            Dict: The update response.
        """
        try:
            response = self.session.patch(
                f"{TOGETHER_API_BASE_URL}{TOGETHER_DEPLOYMENTS_ENDPOINT}/{deployment_id}",
                json=updates
            )
            response.raise_for_status()
            
            logger.info(f"Deployment {deployment_id} updated successfully")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update deployment: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all deployments.
        
        Returns:
            List[Dict]: The list of deployments.
        """
        try:
            response = self.session.get(
                f"{TOGETHER_API_BASE_URL}{TOGETHER_DEPLOYMENTS_ENDPOINT}"
            )
            response.raise_for_status()
            
            return response.json().get("deployments", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list deployments: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise


def deploy_to_together(config_path: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Deploy the SAP HANA AI Toolkit to Together.ai.
    
    Args:
        config_path: Path to the Together.ai deployment config file.
        api_key: Together.ai API key.
        
    Returns:
        Dict: The deployment information.
    """
    deployer = TogetherEndpointDeployer(
        api_key=api_key,
        config_path=config_path
    )
    
    # Create deployment
    deployment = deployer.create_deployment()
    deployment_id = deployment.get("deployment_id")
    
    # Wait for deployment to be ready
    status = deployer.wait_for_deployment(deployment_id)
    
    return {
        "deployment_id": deployment_id,
        "status": status.get("status"),
        "endpoint_url": status.get("endpoint_url"),
        "created_at": deployment.get("created_at"),
        "model": deployment.get("model", {}).get("name")
    }