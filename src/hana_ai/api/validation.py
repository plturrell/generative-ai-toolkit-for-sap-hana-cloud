"""
Environment validation module for the HANA AI Toolkit API.

This module provides validation functions to ensure all required
environment variables and connections are properly configured.
"""
import os
import sys
import logging
import socket
import json
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np
import pandas as pd
from hana_ml.dataframe import ConnectionContext

from .config import settings
from .env_constants import SAP_AI_CORE_LLM_MODEL, SAP_AI_CORE_EMBEDDING_MODEL
from .gpu_utils import GPUProfiler, MultiGPUManager

logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """
    Validates the runtime environment to ensure all required
    components are properly configured.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_results = {
            "system": {"status": "unknown", "details": {}},
            "gpu": {"status": "unknown", "details": {}},
            "hana_connection": {"status": "unknown", "details": {}},
            "ai_core_sdk": {"status": "unknown", "details": {}},
            "environment_variables": {"status": "unknown", "details": {}}
        }
    
    def validate_system(self) -> Dict[str, Any]:
        """
        Validate the system environment.
        
        Returns
        -------
        Dict[str, Any]
            System validation results
        """
        try:
            # Check Python version
            python_version = sys.version
            python_ok = sys.version_info >= (3, 7)
            
            # Check memory
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024 ** 3)
            memory_ok = memory_gb >= 4.0  # Require at least 4GB RAM
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024 ** 3)
            disk_ok = disk_gb >= 2.0  # Require at least 2GB free space
            
            # Check network connectivity
            network_ok = self._check_network_connectivity()
            
            # Check CPU
            cpu_count = psutil.cpu_count(logical=True)
            cpu_ok = cpu_count >= 2  # Require at least 2 CPUs
            
            system_ok = python_ok and memory_ok and disk_ok and network_ok and cpu_ok
            
            self.validation_results["system"] = {
                "status": "ok" if system_ok else "warning",
                "details": {
                    "python_version": python_version,
                    "python_ok": python_ok,
                    "memory_gb": round(memory_gb, 2),
                    "memory_ok": memory_ok,
                    "disk_free_gb": round(disk_gb, 2),
                    "disk_ok": disk_ok,
                    "network_ok": network_ok,
                    "cpu_count": cpu_count,
                    "cpu_ok": cpu_ok
                }
            }
        except Exception as e:
            self.validation_results["system"] = {
                "status": "error",
                "details": {
                    "error": str(e)
                }
            }
        
        return self.validation_results["system"]
    
    def validate_gpu(self) -> Dict[str, Any]:
        """
        Validate GPU availability and configuration using advanced utilities.
        
        Returns
        -------
        Dict[str, Any]
            GPU validation results with detailed diagnostics
        """
        try:
            if not settings.ENABLE_GPU_ACCELERATION:
                self.validation_results["gpu"] = {
                    "status": "disabled",
                    "details": {
                        "message": "GPU acceleration is disabled in configuration"
                    }
                }
                return self.validation_results["gpu"]
            
            # Use GPU profiler for comprehensive diagnostics
            profiler = GPUProfiler()
            gpu_stats = profiler.get_gpu_stats()
            
            # If no GPUs available
            if not gpu_stats:
                self.validation_results["gpu"] = {
                    "status": "error",
                    "details": {
                        "message": "No GPUs available or PyTorch not installed",
                        "visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "Not set")
                    }
                }
                return self.validation_results["gpu"]
            
            # Get GPU count and names
            gpu_count = len(gpu_stats)
            gpu_names = [stats.name for device_id, stats in gpu_stats.items()]
            
            # Get detailed memory information
            gpu_memory = []
            for device_id, stats in gpu_stats.items():
                gpu_memory.append({
                    "device": device_id,
                    "name": stats.name,
                    "total_gb": round(stats.total_memory_mb / 1024, 2),
                    "used_gb": round(stats.used_memory_mb / 1024, 2),
                    "free_gb": round(stats.free_memory_mb / 1024, 2),
                    "utilization_percent": stats.utilization_percent,
                    "temperature": stats.temperature,
                    "power_usage_watts": stats.power_usage_watts,
                    "power_limit_watts": stats.power_limit_watts,
                    "compute_capability": stats.compute_capability
                })
            
            # Check multi-GPU capabilities
            multi_gpu_manager = MultiGPUManager()
            execution_plan = multi_gpu_manager.get_optimal_execution_plan()
            
            # Get performance recommendations
            recommendations = profiler.get_optimization_recommendations()
            
            # Verify GPU computation
            compute_ok = True
            test_error = ""
            try:
                import torch
                test_tensor = torch.rand(10, 10).cuda()
                test_result = test_tensor.sum().item()
                
                # Test mixed precision if available
                has_amp = False
                try:
                    with torch.cuda.amp.autocast():
                        mixed_test = torch.rand(10, 10).cuda() * torch.rand(10, 10).cuda()
                        mixed_result = mixed_test.sum().item()
                    has_amp = True
                except:
                    has_amp = False
                
                # Test NVIDIA optimized operations
                has_cudnn = torch.backends.cudnn.is_available()
                has_tf32 = getattr(torch.backends.cuda, "matmul", None) is not None and \
                           getattr(torch.backends.cuda.matmul, "allow_tf32", False)
                           
            except Exception as e:
                compute_ok = False
                test_error = str(e)
            
            # Test GPU memory bandwidth
            memory_bandwidth = {}
            try:
                import torch
                for device_id in range(gpu_count):
                    # Simple memory bandwidth test
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    
                    # Allocate large tensor
                    size = 1000  # Adjust based on available memory
                    a = torch.ones((size, size), device=f"cuda:{device_id}")
                    b = torch.ones((size, size), device=f"cuda:{device_id}")
                    
                    # Time matrix multiplication which is memory-bound
                    start_time = time.time()
                    for _ in range(10):
                        c = torch.matmul(a, b)
                    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                    end_time = time.time()
                    
                    # Calculate approximate bandwidth (very rough estimate)
                    # Each matrix is size*size*4 bytes (float32)
                    # Matmul reads 2 matrices and writes 1, so ~3 matrices worth of data
                    data_bytes = 3 * size * size * 4 * 10  # 10 iterations
                    duration = end_time - start_time
                    bandwidth_gb_s = (data_bytes / duration) / (1024**3)
                    
                    memory_bandwidth[device_id] = round(bandwidth_gb_s, 2)
            except Exception as e:
                memory_bandwidth = {"error": str(e)}
            
            # Add CUDA kernel optimization check
            has_cuda_graphs = False
            try:
                import torch
                has_cuda_graphs = hasattr(torch.cuda, "make_graphed_callables")
            except:
                pass
            
            # Combined status
            gpu_status = "ok" if compute_ok else "error"
            
            # Check for issues but not critical failures
            if compute_ok:
                # Check for potential throttling
                for device_id, stats in gpu_stats.items():
                    if stats.temperature > 80:  # High temperature
                        gpu_status = "warning"
                    if stats.power_usage_watts > 0.95 * stats.power_limit_watts:  # Near power limit
                        gpu_status = "warning"
            
            self.validation_results["gpu"] = {
                "status": gpu_status,
                "details": {
                    "gpu_count": gpu_count,
                    "gpu_names": gpu_names,
                    "gpu_memory": gpu_memory,
                    "compute_ok": compute_ok,
                    "compute_error": test_error,
                    "gpu_environment": {
                        "visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "Not set"),
                        "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "Not set"),
                        "cuda_device_order": os.environ.get("NVIDIA_CUDA_DEVICE_ORDER", "Not set"),
                        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE", "Not set"),
                        "cuda_cache_size": os.environ.get("NVIDIA_CUDA_CACHE_MAXSIZE", "Not set"),
                        "cuda_cache_path": os.environ.get("NVIDIA_CUDA_CACHE_PATH", "Not set")
                    },
                    "gpu_capabilities": {
                        "has_cudnn": has_cudnn if 'has_cudnn' in locals() else False,
                        "has_tf32": has_tf32 if 'has_tf32' in locals() else False,
                        "has_amp": has_amp if 'has_amp' in locals() else False,
                        "has_cuda_graphs": has_cuda_graphs
                    },
                    "memory_bandwidth_gb_s": memory_bandwidth,
                    "execution_strategy": execution_plan.get("strategy", "unknown"),
                    "optimization_recommendations": recommendations
                }
            }
            
        except Exception as e:
            self.validation_results["gpu"] = {
                "status": "error",
                "details": {
                    "error": str(e)
                }
            }
        
        return self.validation_results["gpu"]
    
    def validate_hana_connection(self) -> Dict[str, Any]:
        """
        Validate HANA database connection.
        
        Returns
        -------
        Dict[str, Any]
            HANA connection validation results
        """
        try:
            # Check if connection credentials are provided
            has_direct_credentials = all([
                settings.HANA_HOST,
                settings.HANA_PORT,
                settings.HANA_USER,
                settings.HANA_PASSWORD
            ])
            
            has_userkey = settings.HANA_USERKEY is not None
            
            if not (has_direct_credentials or has_userkey):
                self.validation_results["hana_connection"] = {
                    "status": "error",
                    "details": {
                        "message": "No HANA connection credentials provided"
                    }
                }
                return self.validation_results["hana_connection"]
            
            # Try to connect to HANA
            try:
                start_time = time.time()
                
                if has_userkey:
                    conn = ConnectionContext(userkey=settings.HANA_USERKEY)
                else:
                    conn = ConnectionContext(
                        address=settings.HANA_HOST,
                        port=settings.HANA_PORT,
                        user=settings.HANA_USER,
                        password=settings.HANA_PASSWORD,
                        encrypt=True
                    )
                
                # Test connection with a simple query
                result = conn.sql("SELECT * FROM DUMMY").collect()
                connection_time = time.time() - start_time
                
                # Get HANA version
                version_result = conn.sql("SELECT VERSION FROM M_DATABASE").collect()
                hana_version = version_result.iloc[0, 0] if not version_result.empty else "Unknown"
                
                # Get current schema
                schema = conn.get_current_schema()
                
                self.validation_results["hana_connection"] = {
                    "status": "ok",
                    "details": {
                        "connection_time_ms": round(connection_time * 1000, 2),
                        "hana_version": hana_version,
                        "current_schema": schema,
                        "connection_method": "userkey" if has_userkey else "direct",
                        "host": settings.HANA_HOST if has_direct_credentials else "Using userkey"
                    }
                }
                
                # Close connection
                conn.close()
                
            except Exception as conn_error:
                self.validation_results["hana_connection"] = {
                    "status": "error",
                    "details": {
                        "message": "Failed to connect to HANA",
                        "error": str(conn_error),
                        "connection_method": "userkey" if has_userkey else "direct",
                        "host": settings.HANA_HOST if has_direct_credentials else "Using userkey"
                    }
                }
        
        except Exception as e:
            self.validation_results["hana_connection"] = {
                "status": "error",
                "details": {
                    "error": str(e)
                }
            }
        
        return self.validation_results["hana_connection"]
    
    def validate_ai_core_sdk(self) -> Dict[str, Any]:
        """
        Validate SAP AI Core SDK integration.
        
        Returns
        -------
        Dict[str, Any]
            AI Core SDK validation results
        """
        try:
            # Check if SAP GenAI Hub SDK is installed
            try:
                import gen_ai_hub
                has_genai_hub = True
                genai_hub_version = gen_ai_hub.__version__
            except ImportError:
                has_genai_hub = False
                genai_hub_version = None
            
            if not has_genai_hub:
                self.validation_results["ai_core_sdk"] = {
                    "status": "error",
                    "details": {
                        "message": "SAP GenAI Hub SDK not installed",
                        "solution": "Install with 'pip install generative-ai-hub-sdk[all]'"
                    }
                }
                return self.validation_results["ai_core_sdk"]
            
            # Check if langchain integration is available
            try:
                from gen_ai_hub.proxy.langchain import init_llm, init_embedding_model
                has_langchain_integration = True
            except ImportError:
                has_langchain_integration = False
            
            # Check model configuration
            model_name = settings.DEFAULT_LLM_MODEL
            is_sap_model = model_name.startswith("sap-ai-core")
            
            # Check environment variables
            ai_core_env_vars = {key: val for key, val in os.environ.items() if key.startswith(("GEN_AI_HUB_", "AI_CORE_", "SAP_AI_"))}
            
            self.validation_results["ai_core_sdk"] = {
                "status": "ok" if (has_genai_hub and has_langchain_integration and is_sap_model) else "warning",
                "details": {
                    "sdk_installed": has_genai_hub,
                    "sdk_version": genai_hub_version,
                    "langchain_integration": has_langchain_integration,
                    "model_name": model_name,
                    "is_sap_model": is_sap_model,
                    "environment_variables": {k: "***" if "KEY" in k or "SECRET" in k else v for k, v in ai_core_env_vars.items()}
                }
            }
        
        except Exception as e:
            self.validation_results["ai_core_sdk"] = {
                "status": "error",
                "details": {
                    "error": str(e)
                }
            }
        
        return self.validation_results["ai_core_sdk"]
    
    def validate_environment_variables(self) -> Dict[str, Any]:
        """
        Validate required environment variables.
        
        Returns
        -------
        Dict[str, Any]
            Environment variables validation results
        """
        try:
            # Check for required environment variables
            required_vars = [
                "DEFAULT_LLM_MODEL",
                "API_PORT",
                "LOG_LEVEL",
                "LOG_FORMAT"
            ]
            
            # Check for conditionally required variables
            if not settings.HANA_USERKEY:
                required_vars.extend([
                    "HANA_HOST",
                    "HANA_PORT",
                    "HANA_USER",
                    "HANA_PASSWORD"
                ])
            
            # Security settings
            if settings.AUTH_REQUIRED and not settings.API_KEYS:
                missing_security = True
                security_message = "AUTH_REQUIRED is true but no API_KEYS provided"
            else:
                missing_security = False
                security_message = ""
            
            # Check for missing variables
            missing_vars = [var for var in required_vars if not getattr(settings, var, None)]
            
            # Check for proper model naming
            model_name = settings.DEFAULT_LLM_MODEL
            correct_model_naming = model_name.startswith("sap-ai-core")
            
            # GPU settings check
            if settings.ENABLE_GPU_ACCELERATION:
                gpu_vars = [
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "CUDA_MEMORY_FRACTION"
                ]
                missing_gpu_vars = [var for var in gpu_vars if not getattr(settings, var, None)]
            else:
                missing_gpu_vars = []
            
            # CORS settings validation
            cors_origins = settings.CORS_ORIGINS
            has_wildcard_cors = "*" in cors_origins
            
            self.validation_results["environment_variables"] = {
                "status": "error" if (missing_vars or missing_security) else ("warning" if not correct_model_naming or has_wildcard_cors or missing_gpu_vars else "ok"),
                "details": {
                    "missing_required": missing_vars,
                    "missing_gpu_settings": missing_gpu_vars if settings.ENABLE_GPU_ACCELERATION else "N/A (GPU disabled)",
                    "security_issues": security_message if missing_security else "None",
                    "model_naming_correct": correct_model_naming,
                    "wildcard_cors": has_wildcard_cors,
                    "development_mode": settings.DEVELOPMENT_MODE
                }
            }
        
        except Exception as e:
            self.validation_results["environment_variables"] = {
                "status": "error",
                "details": {
                    "error": str(e)
                }
            }
        
        return self.validation_results["environment_variables"]
    
    def validate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all validations.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            All validation results
        """
        self.validate_system()
        self.validate_gpu()
        self.validate_hana_connection()
        self.validate_ai_core_sdk()
        self.validate_environment_variables()
        
        # Determine overall status
        error_components = [k for k, v in self.validation_results.items() if v["status"] == "error"]
        warning_components = [k for k, v in self.validation_results.items() if v["status"] == "warning"]
        
        if error_components:
            overall_status = "error"
            message = f"Critical issues found in: {', '.join(error_components)}"
        elif warning_components:
            overall_status = "warning"
            message = f"Warnings found in: {', '.join(warning_components)}"
        else:
            overall_status = "ok"
            message = "All validations passed successfully"
        
        self.validation_results["overall"] = {
            "status": overall_status,
            "message": message,
            "timestamp": time.time()
        }
        
        return self.validation_results
    
    def _check_network_connectivity(self) -> bool:
        """
        Check network connectivity to key services.
        
        Returns
        -------
        bool
            True if network connectivity is OK, False otherwise
        """
        hosts_to_check = []
        
        # Add HANA host if available
        if settings.HANA_HOST:
            hosts_to_check.append(settings.HANA_HOST)
        
        # Add some BTP domains
        hosts_to_check.extend([
            "api.hana.ondemand.com",
            "api.sap.com"
        ])
        
        # Try to connect to hosts
        for host in hosts_to_check:
            try:
                # Try DNS resolution
                socket.gethostbyname(host)
                return True
            except:
                continue
        
        return False


def validate_environment() -> Dict[str, Any]:
    """
    Validate the runtime environment.
    
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    validator = EnvironmentValidator()
    return validator.validate_all()