#!/usr/bin/env python3
"""
NVIDIA T4 deployment configuration tool for SAP HANA AI Toolkit.

This script generates optimal deployment configurations specifically for
NVIDIA T4 GPU environments, automating the setup of the SAP HANA AI Toolkit
with T4-specific optimizations.
"""

import os
import sys
import argparse
import json
import logging
import subprocess
from typing import Dict, Any, Optional

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

class T4DeploymentConfig:
    """NVIDIA T4 deployment configuration generator."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the T4 deployment configuration generator.
        
        Args:
            base_dir: Base directory for the project.
        """
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.templates_dir = os.path.join(self.base_dir, "deployment", "nvidia-t4", "templates")
        self.output_dir = os.path.join(self.base_dir, "deployment", "nvidia-t4", "output")
        
        # Create output directories if they don't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default templates if they don't exist."""
        default_env_template = {
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "json",
            "AUTH_REQUIRED": "true",
            "CORS_ORIGINS": "*",
            "ENFORCE_HTTPS": "false",
            "DEPLOYMENT_MODE": "api_only",
            "DEPLOYMENT_PLATFORM": "nvidia",
            "ENABLE_GPU_ACCELERATION": "true",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_MEMORY_FRACTION": "0.8",
            "NVIDIA_CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "NVIDIA_CUDA_CACHE_MAXSIZE": "2147483648",
            "NVIDIA_CUDA_CACHE_PATH": "/tmp/cuda-cache",
            "MULTI_GPU_STRATEGY": "data_parallel",
            "ENABLE_TENSOR_PARALLELISM": "false",
            "ENABLE_PIPELINE_PARALLELISM": "false",
            "GPU_BATCH_SIZE_OPTIMIZATION": "true",
            "ENABLE_CUDA_GRAPHS": "true",
            "ENABLE_KERNEL_FUSION": "true",
            "ENABLE_FLASH_ATTENTION": "true",
            "CHECKPOINT_ACTIVATIONS": "true",
            "ENABLE_TENSORRT": "true",
            "TENSORRT_CACHE_DIR": "/tmp/tensorrt_engines",
            "TENSORRT_WORKSPACE_SIZE_MB": "1024",
            "TENSORRT_PRECISION": "fp16",
            "TENSORRT_MAX_BATCH_SIZE": "16",
            "TENSORRT_BUILDER_OPTIMIZATION_LEVEL": "3",
            "ENABLE_MEMORY": "true",
            "MEMORY_EXPIRATION_SECONDS": "3600",
            "CONNECTION_POOL_SIZE": "10",
            "REQUEST_TIMEOUT_SECONDS": "300",
            "MAX_REQUEST_SIZE_MB": "10",
            "PROMETHEUS_ENABLED": "true",
            "PROMETHEUS_PORT": "9090",
            "ENABLE_CACHING": "true",
            "CACHE_TTL_SECONDS": "300"
        }
        
        env_template_path = os.path.join(self.templates_dir, "t4-environment.env.template")
        if not os.path.exists(env_template_path):
            with open(env_template_path, "w") as f:
                for key, value in default_env_template.items():
                    f.write(f"{key}={value}\n")
            logger.info(f"Created default T4 environment template at {env_template_path}")
    
    def detect_t4_gpu(self) -> bool:
        """
        Detect if a NVIDIA T4 GPU is available in the system.
        
        Returns:
            bool: True if T4 GPU is detected, False otherwise.
        """
        try:
            # Try to run nvidia-smi to check for T4
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            
            if result.returncode == 0 and "T4" in result.stdout:
                logger.info("NVIDIA T4 GPU detected!")
                return True
            
            # If nvidia-smi didn't show T4, try PyTorch if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        device_name = torch.cuda.get_device_name(i)
                        if "T4" in device_name:
                            logger.info(f"NVIDIA T4 GPU detected via PyTorch: {device_name}")
                            return True
            except ImportError:
                pass
            
            logger.warning("No NVIDIA T4 GPU detected in the system")
            return False
        except Exception as e:
            logger.error(f"Error detecting GPU: {str(e)}")
            return False
    
    def get_gpu_memory(self) -> int:
        """
        Get the T4 GPU memory in GB.
        
        Returns:
            int: Memory in GB, or 16 as default for T4 if detection fails.
        """
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    if "T4" in device_name:
                        props = torch.cuda.get_device_properties(i)
                        memory_gb = round(props.total_memory / (1024**3))
                        logger.info(f"T4 GPU has {memory_gb} GB of memory")
                        return memory_gb
        except Exception as e:
            logger.warning(f"Error detecting GPU memory: {str(e)}")
        
        # Default for T4 is 16GB
        return 16
    
    def generate_environment_file(self, 
                                 frontend_url: Optional[str] = None,
                                 custom_values: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an environment file optimized for T4 GPU.
        
        Args:
            frontend_url: URL of the frontend (for CORS settings)
            custom_values: Custom configuration values to apply
            
        Returns:
            str: Path to the generated environment file
        """
        template_path = os.path.join(self.templates_dir, "t4-environment.env.template")
        output_path = os.path.join(self.output_dir, "t4-environment.env")
        
        # Load template
        env_vars = {}
        with open(template_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
        
        # Set frontend URL if provided (for CORS)
        if frontend_url:
            env_vars["CORS_ORIGINS"] = frontend_url
            
        # Check T4 memory and adjust settings
        memory_gb = self.get_gpu_memory()
        
        # T4-specific tuning based on available memory
        if memory_gb <= 8:
            # Low memory config
            env_vars["CUDA_MEMORY_FRACTION"] = "0.7"
            env_vars["TENSORRT_MAX_BATCH_SIZE"] = "8"
            env_vars["CHECKPOINT_ACTIVATIONS"] = "true"
        elif memory_gb <= 12:
            # Medium memory config
            env_vars["CUDA_MEMORY_FRACTION"] = "0.75"
            env_vars["TENSORRT_MAX_BATCH_SIZE"] = "12"
        else:
            # High memory config (16GB)
            env_vars["CUDA_MEMORY_FRACTION"] = "0.8"
            env_vars["TENSORRT_MAX_BATCH_SIZE"] = "16"
        
        # Apply custom values if provided
        if custom_values:
            for key, value in custom_values.items():
                env_vars[key] = value
        
        # Write to output file
        with open(output_path, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Generated T4-optimized environment file at {output_path}")
        return output_path
    
    def generate_docker_compose(self, env_file: str) -> str:
        """
        Generate a docker-compose file for T4 deployment.
        
        Args:
            env_file: Path to the environment file to use
            
        Returns:
            str: Path to the generated docker-compose file
        """
        output_path = os.path.join(self.output_dir, "docker-compose.yml")
        
        # Create docker-compose configuration
        compose_config = {
            "version": "3.8",
            "services": {
                "hana-ai-api": {
                    "build": {
                        "context": "../../",
                        "dockerfile": "deployment/nvidia-t4/Dockerfile"
                    },
                    "image": "hana-ai-toolkit-t4:latest",
                    "container_name": "hana-ai-toolkit-t4",
                    "env_file": [
                        os.path.basename(env_file)
                    ],
                    "ports": [
                        "${API_PORT:-8000}:8000",
                        "${PROMETHEUS_PORT:-9090}:9090"
                    ],
                    "volumes": [
                        "../../:/app",
                        "nvidia-cache:/tmp/cuda-cache",
                        "tensorrt-cache:/tmp/tensorrt_engines"
                    ],
                    "restart": "unless-stopped",
                    "networks": [
                        "hana-ai-network"
                    ],
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": 1,
                                        "capabilities": ["gpu"]
                                    }
                                ]
                            }
                        }
                    }
                },
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": "prometheus",
                    "volumes": [
                        "../prometheus:/etc/prometheus",
                        "prometheus_data:/prometheus"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/usr/share/prometheus/console_libraries",
                        "--web.console.templates=/usr/share/prometheus/consoles"
                    ],
                    "ports": [
                        "9091:9090"
                    ],
                    "restart": "unless-stopped",
                    "networks": [
                        "hana-ai-network"
                    ]
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": "grafana",
                    "volumes": [
                        "grafana_data:/var/lib/grafana",
                        "../grafana/provisioning:/etc/grafana/provisioning"
                    ],
                    "environment": [
                        "GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}",
                        "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}",
                        "GF_USERS_ALLOW_SIGN_UP=false"
                    ],
                    "ports": [
                        "3000:3000"
                    ],
                    "restart": "unless-stopped",
                    "networks": [
                        "hana-ai-network"
                    ],
                    "depends_on": [
                        "prometheus"
                    ]
                },
                "nvidia-smi-exporter": {
                    "image": "utkuozdemir/nvidia_gpu_exporter:latest",
                    "container_name": "nvidia-smi-exporter",
                    "restart": "unless-stopped",
                    "ports": [
                        "9835:9835"
                    ],
                    "runtime": "nvidia",
                    "networks": [
                        "hana-ai-network"
                    ],
                    "environment": [
                        "NVIDIA_VISIBLE_DEVICES=all"
                    ],
                    "depends_on": [
                        "prometheus"
                    ]
                }
            },
            "networks": {
                "hana-ai-network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "prometheus_data": {},
                "grafana_data": {},
                "nvidia-cache": {},
                "tensorrt-cache": {}
            }
        }
        
        # Write to output file
        with open(output_path, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info(f"Generated docker-compose file at {output_path}")
        return output_path
    
    def generate_deployment_files(self, 
                                 frontend_url: Optional[str] = None,
                                 custom_values: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate all necessary deployment files for T4 GPU deployment.
        
        Args:
            frontend_url: URL of the frontend (for CORS settings)
            custom_values: Custom configuration values to apply
            
        Returns:
            Dict[str, str]: Paths to the generated files
        """
        # Check if T4 GPU is available
        has_t4 = self.detect_t4_gpu()
        
        if not has_t4:
            logger.warning("No T4 GPU detected. Configuration will be generated but may not be optimal.")
        
        # Generate environment file
        env_file = self.generate_environment_file(
            frontend_url=frontend_url,
            custom_values=custom_values
        )
        
        # Generate docker-compose file
        docker_compose_file = self.generate_docker_compose(env_file)
        
        # Check if Dockerfile exists
        dockerfile_path = os.path.join(self.base_dir, "deployment", "nvidia-t4", "Dockerfile")
        if not os.path.exists(dockerfile_path):
            logger.error(f"Dockerfile not found at {dockerfile_path}")
            dockerfile_status = "Not found"
        else:
            dockerfile_status = "Available"
        
        # Check if deployment script exists
        deploy_script_path = os.path.join(self.base_dir, "deployment", "nvidia-t4", "deploy-t4.sh")
        if not os.path.exists(deploy_script_path):
            logger.error(f"Deployment script not found at {deploy_script_path}")
            deploy_script_status = "Not found"
        else:
            deploy_script_status = "Available"
        
        # Return all generated files
        return {
            "environment_file": env_file,
            "docker_compose_file": docker_compose_file,
            "dockerfile": f"{dockerfile_path} ({dockerfile_status})",
            "deployment_script": f"{deploy_script_path} ({deploy_script_status})"
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deployment configurations for NVIDIA T4 GPU"
    )
    
    parser.add_argument(
        "--frontend-url",
        help="URL of the frontend application for CORS configuration"
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
    config_generator = T4DeploymentConfig()
    
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
    
    # Generate deployment files
    files = config_generator.generate_deployment_files(
        frontend_url=args.frontend_url,
        custom_values=custom_values
    )
    
    # Display results
    logger.info("T4 deployment configuration generated successfully:")
    for key, value in files.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()