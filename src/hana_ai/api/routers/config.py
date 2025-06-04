"""
Configuration UI router for managing BTP service connections.

This module provides endpoints for administrators to configure connections to 
SAP HANA, SAP AI Core, and other BTP services. It includes secure credential 
storage and validation functionality.
"""
import os
import time
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr
from cryptography.fernet import Fernet

from hana_ml.dataframe import ConnectionContext

from ..auth import get_admin_api_key
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Secure storage for credentials
CREDENTIALS_FILE = os.path.join(settings.CONFIG_DIR, "credentials.enc")
CONFIG_FILE = os.path.join(settings.CONFIG_DIR, "config.json")

# Create config directory if it doesn't exist
os.makedirs(settings.CONFIG_DIR, exist_ok=True)

# Models for request validation
class HANACredentials(BaseModel):
    """SAP HANA connection credentials."""
    host: str = Field(..., description="HANA host")
    port: int = Field(..., description="HANA port")
    user: str = Field(..., description="HANA username")
    password: SecretStr = Field(..., description="HANA password")
    encrypt: bool = Field(default=True, description="Use encrypted connection")
    userkey: Optional[str] = Field(default=None, description="HANA userkey (optional)")

class AICoreSetting(BaseModel):
    """SAP AI Core settings."""
    api_key: SecretStr = Field(..., description="AI Core API key")
    service_url: str = Field(..., description="AI Core service URL")
    resource_group: str = Field(..., description="AI Core resource group")
    deployment_id: Optional[str] = Field(default=None, description="Model deployment ID")

class OtherBTPService(BaseModel):
    """Configuration for other BTP services."""
    service_type: str = Field(..., description="Type of BTP service")
    name: str = Field(..., description="Service name")
    credentials: Dict[str, Any] = Field(..., description="Service credentials")
    enabled: bool = Field(default=True, description="Whether service is enabled")

class ConfigStatus(BaseModel):
    """Status of configuration."""
    hana_configured: bool = Field(..., description="Whether HANA is configured")
    aicore_configured: bool = Field(..., description="Whether AI Core is configured")
    other_services: List[str] = Field(default_factory=list, description="Other configured services")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")

def get_encryption_key():
    """Get or create encryption key for credentials."""
    key_file = os.path.join(settings.CONFIG_DIR, ".key")
    
    if os.path.exists(key_file):
        with open(key_file, "rb") as f:
            return f.read()
    else:
        # Generate a new key
        key = Fernet.generate_key()
        
        # Save key (in production, use a proper key management service)
        with open(key_file, "wb") as f:
            f.write(key)
            
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
        
        return key

def encrypt_credentials(data: Dict[str, Any]) -> bytes:
    """Encrypt credentials using Fernet symmetric encryption."""
    key = get_encryption_key()
    f = Fernet(key)
    return f.encrypt(json.dumps(data).encode())

def decrypt_credentials() -> Dict[str, Any]:
    """Decrypt stored credentials."""
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
        
    key = get_encryption_key()
    f = Fernet(key)
    
    with open(CREDENTIALS_FILE, "rb") as file:
        encrypted_data = file.read()
        
    try:
        decrypted_data = f.decrypt(encrypted_data)
        return json.loads(decrypted_data)
    except Exception as e:
        logger.error(f"Failed to decrypt credentials: {str(e)}")
        return {}

def save_credentials(credentials: Dict[str, Any]):
    """Save encrypted credentials to file."""
    encrypted_data = encrypt_credentials(credentials)
    
    with open(CREDENTIALS_FILE, "wb") as file:
        file.write(encrypted_data)
    
    # Set restrictive permissions
    os.chmod(CREDENTIALS_FILE, 0o600)

def get_config() -> Dict[str, Any]:
    """Get current configuration (non-sensitive)."""
    if not os.path.exists(CONFIG_FILE):
        return {
            "hana_configured": False,
            "aicore_configured": False,
            "other_services": [],
            "last_updated": None
        }
        
    with open(CONFIG_FILE, "r") as file:
        return json.load(file)

def save_config(config: Dict[str, Any]):
    """Save configuration (non-sensitive) to file."""
    config["last_updated"] = datetime.now().isoformat()
    
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=2)

def test_hana_connection(credentials: Dict[str, Any]) -> bool:
    """Test SAP HANA connection with provided credentials."""
    try:
        # Create connection
        if credentials.get("userkey"):
            conn = ConnectionContext(userkey=credentials["userkey"])
        else:
            conn = ConnectionContext(
                address=credentials["host"],
                port=credentials["port"],
                user=credentials["user"],
                password=credentials["password"],
                encrypt=credentials.get("encrypt", True)
            )
        
        # Test with simple query
        result = conn.sql("SELECT 1 FROM DUMMY").collect()
        
        # Close connection
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"HANA connection test failed: {str(e)}")
        return False

@router.get(
    "/status",
    response_model=ConfigStatus,
    summary="Get configuration status",
    description="Check the status of BTP service configurations"
)
async def get_configuration_status(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Get the current configuration status.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    ConfigStatus
        Configuration status
    """
    config = get_config()
    
    return ConfigStatus(
        hana_configured=config.get("hana_configured", False),
        aicore_configured=config.get("aicore_configured", False),
        other_services=config.get("other_services", []),
        last_updated=datetime.fromisoformat(config["last_updated"]) if config.get("last_updated") else None
    )

@router.post(
    "/hana",
    summary="Configure SAP HANA connection",
    description="Set up connection details for SAP HANA database"
)
async def configure_hana(
    credentials: HANACredentials,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Configure SAP HANA connection details.
    
    Parameters
    ----------
    credentials : HANACredentials
        HANA connection credentials
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Configuration result
    """
    try:
        # Convert model to dict
        creds_dict = credentials.dict()
        
        # Convert SecretStr to plain text for storage
        if isinstance(creds_dict["password"], dict) and "value" in creds_dict["password"]:
            creds_dict["password"] = creds_dict["password"]["value"]
            
        # Test connection
        if not test_hana_connection(creds_dict):
            raise HTTPException(
                status_code=400,
                detail="Failed to connect to HANA with provided credentials"
            )
            
        # Get existing credentials
        all_creds = decrypt_credentials()
        all_creds["hana"] = creds_dict
        
        # Save credentials
        save_credentials(all_creds)
        
        # Update config
        config = get_config()
        config["hana_configured"] = True
        save_config(config)
        
        return {"status": "success", "message": "HANA configuration saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring HANA: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error configuring HANA: {str(e)}"
        )

@router.post(
    "/aicore",
    summary="Configure SAP AI Core",
    description="Set up connection details for SAP AI Core service"
)
async def configure_aicore(
    settings: AICoreSetting,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Configure SAP AI Core settings.
    
    Parameters
    ----------
    settings : AICoreSetting
        AI Core settings
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Configuration result
    """
    try:
        # Convert model to dict
        settings_dict = settings.dict()
        
        # Convert SecretStr to plain text for storage
        if isinstance(settings_dict["api_key"], dict) and "value" in settings_dict["api_key"]:
            settings_dict["api_key"] = settings_dict["api_key"]["value"]
            
        # TODO: Test connection to AI Core (implementation depends on SDK details)
        
        # Get existing credentials
        all_creds = decrypt_credentials()
        all_creds["aicore"] = settings_dict
        
        # Save credentials
        save_credentials(all_creds)
        
        # Update config
        config = get_config()
        config["aicore_configured"] = True
        save_config(config)
        
        return {"status": "success", "message": "AI Core configuration saved successfully"}
        
    except Exception as e:
        logger.error(f"Error configuring AI Core: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error configuring AI Core: {str(e)}"
        )

@router.post(
    "/service",
    summary="Configure other BTP service",
    description="Add configuration for another BTP service"
)
async def configure_service(
    service: OtherBTPService,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Configure other BTP service.
    
    Parameters
    ----------
    service : OtherBTPService
        Service configuration
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Configuration result
    """
    try:
        # Convert model to dict
        service_dict = service.dict()
        
        # Get existing credentials
        all_creds = decrypt_credentials()
        
        # Initialize other_services if it doesn't exist
        if "other_services" not in all_creds:
            all_creds["other_services"] = {}
            
        # Store service credentials
        all_creds["other_services"][service.name] = service_dict
        
        # Save credentials
        save_credentials(all_creds)
        
        # Update config
        config = get_config()
        if "other_services" not in config:
            config["other_services"] = []
            
        if service.name not in config["other_services"]:
            config["other_services"].append(service.name)
            
        save_config(config)
        
        return {"status": "success", "message": f"Service '{service.name}' configured successfully"}
        
    except Exception as e:
        logger.error(f"Error configuring service: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error configuring service: {str(e)}"
        )

@router.delete(
    "/service/{service_name}",
    summary="Remove BTP service configuration",
    description="Delete configuration for a BTP service"
)
async def remove_service_config(
    service_name: str,
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Remove a BTP service configuration.
    
    Parameters
    ----------
    service_name : str
        Name of the service to remove
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Result of the operation
    """
    try:
        # Get existing credentials
        all_creds = decrypt_credentials()
        
        # Remove service if it exists
        if "other_services" in all_creds and service_name in all_creds["other_services"]:
            del all_creds["other_services"][service_name]
            save_credentials(all_creds)
            
        # Update config
        config = get_config()
        if "other_services" in config and service_name in config["other_services"]:
            config["other_services"].remove(service_name)
            save_config(config)
            
        return {"status": "success", "message": f"Service '{service_name}' removed successfully"}
        
    except Exception as e:
        logger.error(f"Error removing service: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error removing service: {str(e)}"
        )

@router.get(
    "/test/hana",
    summary="Test HANA connection",
    description="Test the configured SAP HANA connection"
)
async def test_hana(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the configured HANA connection.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Get credentials
        all_creds = decrypt_credentials()
        
        if "hana" not in all_creds:
            return {
                "status": "not_configured",
                "message": "HANA connection not configured"
            }
            
        hana_creds = all_creds["hana"]
        
        # Test connection
        if test_hana_connection(hana_creds):
            return {
                "status": "success",
                "message": "HANA connection successful",
                "details": {
                    "host": hana_creds["host"],
                    "port": hana_creds["port"],
                    "user": hana_creds["user"]
                }
            }
        else:
            return {
                "status": "error",
                "message": "HANA connection test failed"
            }
            
    except Exception as e:
        logger.error(f"Error testing HANA connection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing HANA connection: {str(e)}"
        )

@router.get(
    "/test/aicore",
    summary="Test AI Core connection",
    description="Test the configured SAP AI Core connection"
)
async def test_aicore(
    admin_api_key: str = Depends(get_admin_api_key)
):
    """
    Test the configured AI Core connection.
    
    Parameters
    ----------
    admin_api_key : str
        Admin API key for authentication
        
    Returns
    -------
    Dict
        Test result
    """
    try:
        # Get credentials
        all_creds = decrypt_credentials()
        
        if "aicore" not in all_creds:
            return {
                "status": "not_configured",
                "message": "AI Core connection not configured"
            }
            
        aicore_settings = all_creds["aicore"]
        
        # TODO: Test connection to AI Core (implementation depends on SDK details)
        # For now, just return the configuration status
        
        return {
            "status": "success",
            "message": "AI Core configuration exists",
            "details": {
                "service_url": aicore_settings["service_url"],
                "resource_group": aicore_settings["resource_group"],
                "has_deployment": aicore_settings.get("deployment_id") is not None
            }
        }
            
    except Exception as e:
        logger.error(f"Error testing AI Core connection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing AI Core connection: {str(e)}"
        )