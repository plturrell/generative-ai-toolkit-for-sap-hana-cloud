#!/usr/bin/env python3
"""
Unified deployment configuration tool for SAP HANA AI Toolkit.

This script generates deployment configurations for different platforms
and deployment modes, simplifying the setup of frontend/backend combinations.
"""

import os
import sys
import argparse
import shutil
import json
import logging
import platform
import subprocess
import re
from typing import Dict, Any, List, Optional, Tuple

# Try to import yaml, install if not available
try:
    import yaml
except ImportError:
    print("PyYAML not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pyyaml", "--break-system-packages"])
        import yaml
    except Exception as e:
        print(f"Error installing PyYAML: {e}")
        print("Please manually install PyYAML: pip install pyyaml")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Deployment modes
DEPLOYMENT_MODES = ["full", "api_only", "ui_only"]

# Backend platforms
BACKEND_PLATFORMS = ["nvidia", "together", "btp", "auto"]

# Frontend platforms
FRONTEND_PLATFORMS = ["vercel", "btp", "auto"]

class DeploymentConfig:
    """Deployment configuration generator."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the deployment configuration generator.
        
        Args:
            base_dir: Base directory for the project.
        """
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.templates_dir = os.path.join(self.base_dir, "deployment", "templates")
        self.output_dir = os.path.join(self.base_dir, "deployment", "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_platforms(self) -> Tuple[str, str]:
        """
        Detect the most suitable backend and frontend platforms based on the environment.
        
        Returns:
            Tuple[str, str]: Detected backend and frontend platforms.
        """
        # Default platforms
        backend_platform = "btp"  # Default to SAP BTP
        frontend_platform = "btp"  # Default to SAP BTP
        
        # Check for NVIDIA GPU
        has_nvidia_gpu = False
        try:
            if platform.system() == "Linux":
                # Try to run nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=False
                )
                has_nvidia_gpu = result.returncode == 0
                
                if has_nvidia_gpu:
                    logger.info("NVIDIA GPU detected")
                    backend_platform = "nvidia"
            
            if not has_nvidia_gpu:
                # Check for environment variables that indicate Together.ai
                if os.environ.get("TOGETHER_API_KEY") or os.path.exists("./together-api-key.txt"):
                    logger.info("Together.ai credentials detected")
                    backend_platform = "together"
        except Exception as e:
            logger.warning(f"Error during platform detection: {str(e)}")
        
        # Check for Vercel-specific environment
        if os.environ.get("VERCEL") or os.environ.get("VERCEL_URL"):
            logger.info("Vercel environment detected")
            frontend_platform = "vercel"
        
        # Check for Cloud Foundry / BTP environment
        cf_env_vars = ["VCAP_APPLICATION", "VCAP_SERVICES", "CF_INSTANCE_INDEX"]
        if any(os.environ.get(var) for var in cf_env_vars):
            logger.info("Cloud Foundry / SAP BTP environment detected")
            backend_platform = "btp"
            frontend_platform = "btp"
        
        return backend_platform, frontend_platform
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a template from the templates directory.
        
        Args:
            template_name: Name of the template file.
            
        Returns:
            Dict[str, Any]: The template content as a dictionary.
        """
        template_path = os.path.join(self.templates_dir, template_name)
        
        if not os.path.exists(template_path):
            logger.error(f"Template not found: {template_path}")
            return {}
        
        # Load based on file extension
        if template_name.endswith(".json"):
            with open(template_path, "r") as f:
                return json.load(f)
        elif template_name.endswith(".yaml") or template_name.endswith(".yml"):
            with open(template_path, "r") as f:
                return yaml.safe_load(f)
        elif template_name.endswith(".env"):
            # Parse .env file into a dictionary
            result = {}
            with open(template_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        result[key.strip()] = value.strip()
            return result
        else:
            logger.error(f"Unsupported template format: {template_name}")
            return {}
    
    def save_config(self, config: Dict[str, Any], output_name: str, format_type: str) -> str:
        """
        Save a configuration to the output directory.
        
        Args:
            config: Configuration dictionary.
            output_name: Name for the output file.
            format_type: Type of output format (json, yaml, env).
            
        Returns:
            str: Path to the saved configuration file.
        """
        output_path = os.path.join(self.output_dir, output_name)
        
        if format_type == "json":
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
        elif format_type == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        elif format_type == "env":
            with open(output_path, "w") as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
        else:
            logger.error(f"Unsupported output format: {format_type}")
            return ""
        
        logger.info(f"Configuration saved to: {output_path}")
        return output_path
    
    def save_platform_detection(self, backend: str, frontend: str) -> None:
        """
        Save the detected platforms to files for CI/CD.
        
        Args:
            backend: Detected backend platform.
            frontend: Detected frontend platform.
        """
        backend_path = os.path.join(self.output_dir, "detected_backend.txt")
        frontend_path = os.path.join(self.output_dir, "detected_frontend.txt")
        
        with open(backend_path, "w") as f:
            f.write(backend)
        
        with open(frontend_path, "w") as f:
            f.write(frontend)
        
        logger.info(f"Saved detected backend platform: {backend}")
        logger.info(f"Saved detected frontend platform: {frontend}")
    
    def merge_configs(self, base_config: Dict[str, Any], overlay_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration.
            overlay_config: Configuration to overlay on top of the base.
            
        Returns:
            Dict[str, Any]: Merged configuration.
        """
        result = base_config.copy()
        
        for key, value in overlay_config.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge nested dictionaries
                result[key] = self.merge_configs(result[key], value)
            else:
                # Otherwise simply overwrite
                result[key] = value
        
        return result
    
    def generate_backend_config(
        self, 
        platform: str, 
        frontend_url: Optional[str] = None,
        custom_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate a backend configuration for the specified platform.
        
        Args:
            platform: Backend platform (nvidia, together, btp).
            frontend_url: URL of the frontend to allow in CORS.
            custom_values: Custom configuration values to apply.
            
        Returns:
            Tuple[Dict[str, Any], str]: Generated configuration and format type.
        """
        template_name = ""
        format_type = ""
        
        if platform == "nvidia":
            template_name = "nvidia-backend.env"
            format_type = "env"
        elif platform == "together":
            template_name = "together-backend.yaml"
            format_type = "yaml"
        elif platform == "btp":
            template_name = "btp-backend.env"
            format_type = "env"
        else:
            logger.error(f"Unsupported backend platform: {platform}")
            return {}, ""
        
        # Load template
        config = self.load_template(template_name)
        
        # Set deployment mode
        if "DEPLOYMENT_MODE" in config:
            config["DEPLOYMENT_MODE"] = "api_only"
        elif "environment" in config and isinstance(config["environment"], dict):
            config["environment"]["DEPLOYMENT_MODE"] = "api_only"
        
        # Set frontend URL if provided
        if frontend_url:
            if "FRONTEND_URL" in config:
                config["FRONTEND_URL"] = frontend_url
            elif "environment" in config and isinstance(config["environment"], dict):
                config["environment"]["FRONTEND_URL"] = frontend_url
        
        # Apply custom values
        if custom_values:
            config = self.merge_configs(config, custom_values)
        
        return config, format_type
    
    def generate_frontend_config(
        self, 
        platform: str, 
        backend_url: Optional[str] = None,
        custom_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate a frontend configuration for the specified platform.
        
        Args:
            platform: Frontend platform (vercel, btp).
            backend_url: URL of the backend API.
            custom_values: Custom configuration values to apply.
            
        Returns:
            Tuple[Dict[str, Any], str]: Generated configuration and format type.
        """
        template_name = ""
        format_type = ""
        
        if platform == "vercel":
            template_name = "vercel-frontend.json"
            format_type = "json"
        elif platform == "btp":
            template_name = "btp-frontend.env"
            format_type = "env"
        else:
            logger.error(f"Unsupported frontend platform: {platform}")
            return {}, ""
        
        # Load template
        config = self.load_template(template_name)
        
        # Set deployment mode
        if platform == "vercel" and "env" in config:
            config["env"]["DEPLOYMENT_MODE"] = "ui_only"
            if backend_url:
                config["env"]["API_BASE_URL"] = backend_url
        elif "DEPLOYMENT_MODE" in config:
            config["DEPLOYMENT_MODE"] = "ui_only"
            if backend_url:
                config["API_BASE_URL"] = backend_url
        
        # Apply custom values
        if custom_values:
            config = self.merge_configs(config, custom_values)
        
        return config, format_type
    
    def generate_full_config(
        self,
        platform: str = "btp",
        custom_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate a full stack configuration.
        
        Args:
            platform: Platform for full stack deployment (currently only btp supported).
            custom_values: Custom configuration values to apply.
            
        Returns:
            Tuple[Dict[str, Any], str]: Generated configuration and format type.
        """
        if platform != "btp":
            logger.error(f"Full stack deployment only supported on BTP, not on {platform}")
            return {}, ""
        
        template_name = "btp-full.env"
        format_type = "env"
        
        # Load template
        config = self.load_template(template_name)
        
        # Set deployment mode
        if "DEPLOYMENT_MODE" in config:
            config["DEPLOYMENT_MODE"] = "full"
        
        # Apply custom values
        if custom_values:
            config = self.merge_configs(config, custom_values)
        
        return config, format_type
    
    def generate_hybrid_config(
        self, 
        backend_platform: str = None,
        frontend_platform: str = None,
        backend_url: str = None,
        frontend_url: str = None,
        custom_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate configurations for a hybrid deployment.
        
        Args:
            backend_platform: Backend platform.
            frontend_platform: Frontend platform.
            backend_url: URL of the backend API.
            frontend_url: URL of the frontend application.
            custom_values: Custom configuration values to apply.
            
        Returns:
            Dict[str, str]: Paths to the generated configuration files.
        """
        result = {}
        
        # Auto-detect platforms if set to "auto"
        detected_backend = None
        detected_frontend = None
        
        if backend_platform == "auto" or frontend_platform == "auto":
            detected_backend, detected_frontend = self.detect_platforms()
            
            if backend_platform == "auto":
                backend_platform = detected_backend
                logger.info(f"Auto-detected backend platform: {backend_platform}")
            
            if frontend_platform == "auto":
                frontend_platform = detected_frontend
                logger.info(f"Auto-detected frontend platform: {frontend_platform}")
            
            # Save detection results for CI/CD
            self.save_platform_detection(detected_backend, detected_frontend)
        
        # Generate backend configuration if specified
        if backend_platform:
            backend_config, format_type = self.generate_backend_config(
                backend_platform, 
                frontend_url=frontend_url,
                custom_values=custom_values
            )
            
            if backend_config:
                output_name = f"{backend_platform}-backend"
                if format_type == "env":
                    output_name += ".env"
                elif format_type == "json":
                    output_name += ".json"
                elif format_type == "yaml":
                    output_name += ".yaml"
                
                result["backend"] = self.save_config(backend_config, output_name, format_type)
        
        # Generate frontend configuration if specified
        if frontend_platform:
            frontend_config, format_type = self.generate_frontend_config(
                frontend_platform, 
                backend_url=backend_url,
                custom_values=custom_values
            )
            
            if frontend_config:
                output_name = f"{frontend_platform}-frontend"
                if format_type == "env":
                    output_name += ".env"
                elif format_type == "json":
                    output_name += ".json"
                elif format_type == "yaml":
                    output_name += ".yaml"
                
                result["frontend"] = self.save_config(frontend_config, output_name, format_type)
        
        # Generate full stack configuration if both platforms are BTP
        if backend_platform == "btp" and frontend_platform == "btp":
            full_config, format_type = self.generate_full_config(
                platform="btp",
                custom_values=custom_values
            )
            
            if full_config:
                output_name = "btp-full.env"
                result["full"] = self.save_config(full_config, output_name, format_type)
        
        return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deployment configurations for different platforms and modes"
    )
    
    parser.add_argument(
        "--mode",
        choices=DEPLOYMENT_MODES + ["hybrid"],
        default="full",
        help="Deployment mode (default: full)"
    )
    
    parser.add_argument(
        "--backend",
        choices=BACKEND_PLATFORMS,
        default="auto",
        help="Backend platform to use (default: auto)"
    )
    
    parser.add_argument(
        "--frontend",
        choices=FRONTEND_PLATFORMS,
        default="auto",
        help="Frontend platform to use (default: auto)"
    )
    
    parser.add_argument(
        "--backend-url",
        help="URL of the backend API for frontend configuration"
    )
    
    parser.add_argument(
        "--frontend-url",
        help="URL of the frontend application for backend CORS configuration"
    )
    
    parser.add_argument(
        "--custom-values",
        help="Path to a JSON file with custom configuration values"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated configurations"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Initialize configuration generator
    config_generator = DeploymentConfig()
    
    # Set output directory if specified
    if args.output_dir:
        config_generator.output_dir = args.output_dir
        os.makedirs(config_generator.output_dir, exist_ok=True)
    
    # Load custom values if specified
    custom_values = None
    if args.custom_values:
        try:
            with open(args.custom_values, "r") as f:
                custom_values = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load custom values: {str(e)}")
            sys.exit(1)
    
    # For hybrid or auto mode - generate both frontend and backend configurations
    if args.mode == "hybrid" or (args.backend == "auto" or args.frontend == "auto"):
        # Determine backend and frontend platforms
        backend_platform = args.backend
        frontend_platform = args.frontend
        
        result = config_generator.generate_hybrid_config(
            backend_platform=backend_platform,
            frontend_platform=frontend_platform,
            backend_url=args.backend_url,
            frontend_url=args.frontend_url,
            custom_values=custom_values
        )
        
        if not result:
            logger.error("Failed to generate hybrid configuration")
            sys.exit(1)
        
        logger.info("Hybrid configuration generated successfully:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
    
    # Generate backend-only configuration
    elif args.mode == "api_only":
        backend_platform = args.backend
        if backend_platform == "auto":
            backend_platform, _ = config_generator.detect_platforms()
            logger.info(f"Auto-detected backend platform: {backend_platform}")
            # Save for CI/CD
            config_generator.save_platform_detection(backend_platform, "none")
        
        backend_config, format_type = config_generator.generate_backend_config(
            backend_platform,
            frontend_url=args.frontend_url,
            custom_values=custom_values
        )
        
        if not backend_config:
            logger.error("Failed to generate backend configuration")
            sys.exit(1)
        
        output_name = f"{backend_platform}-backend"
        if format_type == "env":
            output_name += ".env"
        elif format_type == "json":
            output_name += ".json"
        elif format_type == "yaml":
            output_name += ".yaml"
        
        output_path = config_generator.save_config(backend_config, output_name, format_type)
        logger.info(f"Backend configuration generated: {output_path}")
    
    # Generate frontend-only configuration
    elif args.mode == "ui_only":
        frontend_platform = args.frontend
        if frontend_platform == "auto":
            _, frontend_platform = config_generator.detect_platforms()
            logger.info(f"Auto-detected frontend platform: {frontend_platform}")
            # Save for CI/CD
            config_generator.save_platform_detection("none", frontend_platform)
        
        frontend_config, format_type = config_generator.generate_frontend_config(
            frontend_platform,
            backend_url=args.backend_url,
            custom_values=custom_values
        )
        
        if not frontend_config:
            logger.error("Failed to generate frontend configuration")
            sys.exit(1)
        
        output_name = f"{frontend_platform}-frontend"
        if format_type == "env":
            output_name += ".env"
        elif format_type == "json":
            output_name += ".json"
        elif format_type == "yaml":
            output_name += ".yaml"
        
        output_path = config_generator.save_config(frontend_config, output_name, format_type)
        logger.info(f"Frontend configuration generated: {output_path}")
    
    # Generate full stack configuration
    elif args.mode == "full":
        if args.backend != "btp" or args.frontend != "btp":
            logger.error("Full stack deployment only supported on SAP BTP for both frontend and backend")
            sys.exit(1)
        
        full_config, format_type = config_generator.generate_full_config(
            platform="btp",
            custom_values=custom_values
        )
        
        if not full_config:
            logger.error("Failed to generate full stack configuration")
            sys.exit(1)
        
        output_path = config_generator.save_config(full_config, "btp-full.env", format_type)
        logger.info(f"Full stack configuration generated: {output_path}")

if __name__ == "__main__":
    main()