#!/usr/bin/env python3
"""
Comprehensive Automated Testing Framework for NVIDIA T4 Deployments.

This script provides extensive testing capabilities including:
1. Environment validation and configuration testing
2. TensorRT optimization verification
3. Model performance benchmarking
4. Memory utilization analysis
5. HTML report generation

Usage:
    python run_automated_tests.py [options]
"""
import os
import sys
import argparse
import subprocess
import time
import json
import logging
import requests
import datetime
import platform
import threading
import webbrowser
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing GPU-related libraries
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestResult:
    """Class to store test results with timing information."""
    
    def __init__(self, name: str, category: str):
        """Initialize test result."""
        self.name = name
        self.category = category
        self.status = "not_run"
        self.details = {}
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0
    
    def start(self):
        """Mark test as started."""
        self.start_time = time.time()
        self.status = "running"
        return self
    
    def passed(self, details: Optional[Dict[str, Any]] = None):
        """Mark test as passed."""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.status = "passed"
        if details:
            self.details = details
        return self
    
    def failed(self, details: Optional[Dict[str, Any]] = None):
        """Mark test as failed."""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.status = "failed"
        if details:
            self.details = details
        return self
    
    def skipped(self, reason: str):
        """Mark test as skipped."""
        self.status = "skipped"
        self.details = {"reason": reason}
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "details": self.details,
            "duration_ms": self.duration_ms
        }


class T4AutomatedTester:
    """
    Automated testing framework for NVIDIA T4 deployments.
    
    This class runs comprehensive tests to validate T4 GPU optimization,
    TensorRT configuration, and overall system performance.
    """
    
    def __init__(self, 
                host: str = "localhost",
                api_port: int = 8000,
                prometheus_port: int = 9091,
                grafana_port: int = 3000,
                gpu_metrics_port: int = 9835,
                ssh_key: Optional[str] = None,
                username: Optional[str] = None,
                output_dir: str = "test-reports"):
        """
        Initialize the automated tester.
        
        Args:
            host: Hostname or IP of the NVIDIA instance
            api_port: Port for the API service
            prometheus_port: Port for Prometheus metrics
            grafana_port: Port for Grafana dashboards
            gpu_metrics_port: Port for NVIDIA GPU metrics exporter
            ssh_key: Path to SSH key for remote testing (optional)
            username: Username for SSH connection (optional)
            output_dir: Directory to store test reports
        """
        self.host = host
        self.api_port = api_port
        self.prometheus_port = prometheus_port
        self.grafana_port = grafana_port
        self.gpu_metrics_port = gpu_metrics_port
        self.ssh_key = ssh_key
        self.username = username
        self.output_dir = output_dir
        self.remote_mode = (host != "localhost" and host != "127.0.0.1")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test results
        self.results = {
            "environment": [],
            "tensorrt": [],
            "performance": [],
            "memory": [],
            "integration": []
        }
        
        # Timestamp for reports
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.report_file = os.path.join(self.output_dir, f"t4-test-report-{self.timestamp}.json")
        self.html_report_file = os.path.join(self.output_dir, f"t4-test-report-{self.timestamp}.html")
        
        logger.info(f"Initialized automated tester for host: {host}")
    
    def run_command(self, command: str) -> Tuple[int, str, str]:
        """
        Run a command locally or remotely via SSH.
        
        Args:
            command: Command to execute
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if self.remote_mode:
            ssh_cmd = ["ssh"]
            if self.ssh_key:
                ssh_cmd.extend(["-i", self.ssh_key])
            
            target = f"{self.username}@{self.host}" if self.username else self.host
            ssh_cmd.append(target)
            ssh_cmd.append(command)
            
            logger.debug(f"Running remote command: {' '.join(ssh_cmd)}")
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            logger.debug(f"Running local command: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    
    def api_request(self, endpoint: str, method: str = "GET", 
                   data: Optional[Dict[str, Any]] = None, 
                   timeout: int = 30) -> Optional[requests.Response]:
        """
        Make a request to the API.
        
        Args:
            endpoint: API endpoint (without host/port)
            method: HTTP method
            data: Request data for POST requests
            timeout: Request timeout in seconds
            
        Returns:
            Response object or None if request failed
        """
        url = f"http://{self.host}:{self.api_port}{endpoint}"
        
        try:
            if method.upper() == "GET":
                return requests.get(url, timeout=timeout)
            elif method.upper() == "POST":
                headers = {"Content-Type": "application/json"}
                return requests.post(url, json=data, headers=headers, timeout=timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    # ====== ENVIRONMENT TESTS ======
    
    def test_gpu_detection(self) -> TestResult:
        """
        Test GPU detection to verify T4 availability.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("T4 GPU Detection", "environment").start()
        
        # Test nvidia-smi
        returncode, stdout, stderr = self.run_command("nvidia-smi")
        
        if returncode != 0:
            return result.failed({
                "error": stderr,
                "message": "Failed to run nvidia-smi. NVIDIA drivers may not be installed."
            })
        
        # Check for T4 in output
        t4_detected = "T4" in stdout
        
        if not t4_detected:
            return result.failed({
                "output": stdout,
                "message": "NVIDIA GPU detected but not a T4 model."
            })
        
        # Get detailed GPU info
        returncode, stdout, stderr = self.run_command(
            "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
        )
        
        if returncode == 0:
            # Parse CSV output
            parts = [p.strip() for p in stdout.strip().split(',')]
            
            if len(parts) >= 4:
                details = {
                    "model": parts[0],
                    "memory": parts[1],
                    "driver_version": parts[2],
                    "compute_capability": parts[3],
                    "message": f"NVIDIA T4 GPU detected with {parts[1]} of memory."
                }
                return result.passed(details)
        
        # Fallback to basic info
        return result.passed({
            "model": "T4",
            "message": "NVIDIA T4 GPU detected."
        })
    
    def test_cuda_version(self) -> TestResult:
        """
        Test CUDA version compatibility.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("CUDA Version", "environment").start()
        
        # Check nvcc version
        returncode, stdout, stderr = self.run_command("nvcc --version")
        
        if returncode != 0:
            return result.failed({
                "error": stderr,
                "message": "Failed to get CUDA version. NVCC not found."
            })
        
        # Extract CUDA version from output
        import re
        cuda_version_match = re.search(r"release (\d+\.\d+)", stdout)
        
        if not cuda_version_match:
            return result.failed({
                "output": stdout,
                "message": "Failed to parse CUDA version from nvcc output."
            })
        
        cuda_version = cuda_version_match.group(1)
        cuda_version_float = float(cuda_version)
        
        # Verify CUDA version is 11.0 or higher (required for T4 optimizations)
        if cuda_version_float < 11.0:
            return result.failed({
                "cuda_version": cuda_version,
                "message": f"CUDA version {cuda_version} is below the required minimum of 11.0."
            })
        
        return result.passed({
            "cuda_version": cuda_version,
            "message": f"CUDA version {cuda_version} meets requirements."
        })
    
    def test_docker_nvidia_runtime(self) -> TestResult:
        """
        Test Docker NVIDIA runtime configuration.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Docker NVIDIA Runtime", "environment").start()
        
        # Check if Docker is installed
        returncode, stdout, stderr = self.run_command("docker --version")
        
        if returncode != 0:
            return result.failed({
                "error": stderr,
                "message": "Docker is not installed or accessible."
            })
        
        # Check NVIDIA Docker Runtime
        returncode, stdout, stderr = self.run_command("docker info | grep -i nvidia")
        
        if returncode != 0 or "nvidia" not in stdout.lower():
            return result.failed({
                "docker": True,
                "nvidia_runtime": False,
                "message": "NVIDIA Docker runtime is not installed or configured."
            })
        
        # Test running a NVIDIA container
        test_cmd = 'docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi'
        returncode, stdout, stderr = self.run_command(test_cmd)
        
        if returncode != 0:
            return result.failed({
                "docker": True,
                "nvidia_runtime": True,
                "container_test": False,
                "error": stderr,
                "message": "Failed to run NVIDIA container."
            })
        
        return result.passed({
            "docker": True,
            "nvidia_runtime": True,
            "container_test": True,
            "message": "Docker with NVIDIA runtime is properly configured."
        })
    
    def test_deployment_status(self) -> TestResult:
        """
        Test deployment container status.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Deployment Status", "environment").start()
        
        # Check container status
        returncode, stdout, stderr = self.run_command("docker-compose ps")
        
        if returncode != 0:
            return result.failed({
                "error": stderr,
                "message": "Failed to get container status."
            })
        
        # Parse output to find container status
        containers = {}
        lines = stdout.strip().split('\n')
        
        if len(lines) <= 1:
            return result.failed({
                "message": "No containers found in docker-compose output."
            })
        
        for line in lines[1:]:  # Skip header line
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                status = ' '.join([p for p in parts if "Up" in p or "Exit" in p])
                containers[name] = status
        
        # Check for required containers
        required_containers = ["api", "prometheus", "grafana", "gpu-exporter"]
        missing_containers = []
        
        for container in required_containers:
            found = False
            for name in containers:
                if container in name:
                    found = True
                    break
            
            if not found:
                missing_containers.append(container)
        
        if missing_containers:
            return result.failed({
                "containers": containers,
                "missing": missing_containers,
                "message": f"Missing required containers: {', '.join(missing_containers)}"
            })
        
        # Check container status
        unhealthy_containers = {}
        for name, status in containers.items():
            if "Exit" in status or not "Up" in status:
                unhealthy_containers[name] = status
        
        if unhealthy_containers:
            return result.failed({
                "containers": containers,
                "unhealthy": unhealthy_containers,
                "message": f"Unhealthy containers detected: {', '.join(unhealthy_containers.keys())}"
            })
        
        return result.passed({
            "containers": containers,
            "message": "All required containers are running."
        })
    
    # ====== TENSORRT TESTS ======
    
    def test_tensorrt_availability(self) -> TestResult:
        """
        Test TensorRT availability in the deployment.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("TensorRT Availability", "tensorrt").start()
        
        if not self.remote_mode:
            # For local testing, check if TensorRT is available
            if not TENSORRT_AVAILABLE:
                return result.failed({
                    "message": "TensorRT is not available in the local Python environment."
                })
            
            # Get TensorRT version
            tensorrt_version = trt.__version__
            
            return result.passed({
                "tensorrt_version": tensorrt_version,
                "message": f"TensorRT {tensorrt_version} is available."
            })
        else:
            # For remote testing, check TensorRT in the container
            cmd = "docker exec hana-ai-toolkit-t4 python -c \"import tensorrt; print(tensorrt.__version__)\""
            returncode, stdout, stderr = self.run_command(cmd)
            
            if returncode != 0:
                return result.failed({
                    "error": stderr,
                    "message": "Failed to check TensorRT version in the container."
                })
            
            tensorrt_version = stdout.strip()
            
            return result.passed({
                "tensorrt_version": tensorrt_version,
                "message": f"TensorRT {tensorrt_version} is available in the container."
            })
    
    def test_tensorrt_optimization(self) -> TestResult:
        """
        Test TensorRT optimization settings.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("TensorRT Optimization", "tensorrt").start()
        
        # Use the API to check TensorRT settings
        response = self.api_request("/api/v1/system/gpu")
        
        if not response or response.status_code != 200:
            return result.failed({
                "message": "Failed to get GPU information from API."
            })
        
        try:
            data = response.json()
            
            # Check for TensorRT settings
            if "tensorrt" not in data:
                return result.failed({
                    "message": "TensorRT information not found in API response."
                })
            
            tensorrt_info = data["tensorrt"]
            
            # Check if TensorRT is enabled
            if not tensorrt_info.get("enabled", False):
                return result.failed({
                    "tensorrt_info": tensorrt_info,
                    "message": "TensorRT is not enabled."
                })
            
            # Check precision mode (should be FP16 for T4)
            precision = tensorrt_info.get("precision", "unknown")
            if precision != "fp16":
                return result.failed({
                    "tensorrt_info": tensorrt_info,
                    "message": f"TensorRT precision is {precision}, expected fp16 for T4 optimization."
                })
            
            return result.passed({
                "tensorrt_info": tensorrt_info,
                "message": "TensorRT is properly configured for T4 optimization."
            })
        except Exception as e:
            return result.failed({
                "error": str(e),
                "message": "Failed to parse API response."
            })
    
    def test_fp16_support(self) -> TestResult:
        """
        Test FP16 support for T4 Tensor Cores.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("FP16 Support", "tensorrt").start()
        
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            return result.skipped("Required libraries (PyTorch, NumPy) not available")
        
        try:
            # Check if CUDA and FP16 are available
            if not torch.cuda.is_available():
                return result.failed({
                    "message": "CUDA is not available."
                })
            
            # Create a simple test tensor
            dtype = torch.float16
            a = torch.randn(1024, 1024, dtype=dtype, device="cuda")
            b = torch.randn(1024, 1024, dtype=dtype, device="cuda")
            
            # Warmup
            for _ in range(5):
                c = torch.matmul(a, b)
            
            # Measure performance
            start_time = time.time()
            iterations = 10
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            fp16_time = (end_time - start_time) / iterations
            
            # Compare with FP32
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            
            # Warmup
            for _ in range(5):
                c_fp32 = torch.matmul(a_fp32, b_fp32)
            
            # Measure performance
            start_time = time.time()
            for _ in range(iterations):
                c_fp32 = torch.matmul(a_fp32, b_fp32)
            torch.cuda.synchronize()
            end_time = time.time()
            
            fp32_time = (end_time - start_time) / iterations
            
            # Calculate speedup
            speedup = fp32_time / fp16_time
            
            # T4 should show at least 2x speedup for FP16 operations
            if speedup >= 1.8:
                return result.passed({
                    "fp16_time_ms": fp16_time * 1000,
                    "fp32_time_ms": fp32_time * 1000,
                    "speedup": speedup,
                    "message": f"FP16 operations are {speedup:.2f}x faster than FP32, indicating Tensor Cores are working."
                })
            else:
                return result.failed({
                    "fp16_time_ms": fp16_time * 1000,
                    "fp32_time_ms": fp32_time * 1000,
                    "speedup": speedup,
                    "message": f"FP16 speedup is only {speedup:.2f}x. Expected at least 1.8x for T4 Tensor Cores."
                })
            
        except Exception as e:
            return result.failed({
                "error": str(e),
                "message": "Error testing FP16 support."
            })
    
    # ====== PERFORMANCE TESTS ======
    
    def test_api_latency(self) -> TestResult:
        """
        Test API response latency.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("API Latency", "performance").start()
        
        # Test API health endpoint
        start_time = time.time()
        response = self.api_request("/health")
        end_time = time.time()
        
        if not response or response.status_code != 200:
            return result.failed({
                "message": "API health check failed."
            })
        
        health_latency_ms = (end_time - start_time) * 1000
        
        # Test a more complex endpoint
        start_time = time.time()
        response = self.api_request("/api/v1/system/info")
        end_time = time.time()
        
        if not response or response.status_code != 200:
            return result.failed({
                "health_latency_ms": health_latency_ms,
                "message": "API system info request failed."
            })
        
        info_latency_ms = (end_time - start_time) * 1000
        
        # Analyze latency
        if health_latency_ms > 500 or info_latency_ms > 1000:
            return result.failed({
                "health_latency_ms": health_latency_ms,
                "info_latency_ms": info_latency_ms,
                "message": "API latency is higher than expected."
            })
        
        return result.passed({
            "health_latency_ms": health_latency_ms,
            "info_latency_ms": info_latency_ms,
            "message": "API latency is within acceptable range."
        })
    
    def test_gpu_utilization(self) -> TestResult:
        """
        Test GPU utilization metrics.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("GPU Utilization", "performance").start()
        
        # Check GPU metrics endpoint
        metrics_url = f"http://{self.host}:{self.gpu_metrics_port}/metrics"
        
        try:
            response = requests.get(metrics_url, timeout=5)
            
            if response.status_code != 200:
                return result.failed({
                    "status_code": response.status_code,
                    "message": "GPU metrics endpoint returned an error."
                })
            
            # Check for key metrics
            metrics_text = response.text
            
            # Define key metrics to check
            key_metrics = [
                "nvidia_gpu_duty_cycle",
                "nvidia_gpu_memory_used_bytes",
                "nvidia_gpu_temperature_celsius",
                "nvidia_gpu_power_draw_watts"
            ]
            
            missing_metrics = []
            for metric in key_metrics:
                if metric not in metrics_text:
                    missing_metrics.append(metric)
            
            if missing_metrics:
                return result.failed({
                    "missing_metrics": missing_metrics,
                    "message": f"Missing GPU metrics: {', '.join(missing_metrics)}"
                })
            
            # Extract current utilization
            import re
            
            # Extract utilization
            util_match = re.search(r'nvidia_gpu_duty_cycle{[^}]*}\s+(\d+)', metrics_text)
            utilization = int(util_match.group(1)) if util_match else None
            
            # Extract memory usage
            mem_match = re.search(r'nvidia_gpu_memory_used_bytes{[^}]*}\s+(\d+)', metrics_text)
            memory_bytes = int(mem_match.group(1)) if mem_match else None
            memory_mb = memory_bytes / (1024 * 1024) if memory_bytes else None
            
            # Extract temperature
            temp_match = re.search(r'nvidia_gpu_temperature_celsius{[^}]*}\s+(\d+)', metrics_text)
            temperature = int(temp_match.group(1)) if temp_match else None
            
            # Extract power
            power_match = re.search(r'nvidia_gpu_power_draw_watts{[^}]*}\s+(\d+\.\d+)', metrics_text)
            power = float(power_match.group(1)) if power_match else None
            
            return result.passed({
                "utilization_percent": utilization,
                "memory_used_mb": memory_mb,
                "temperature_celsius": temperature,
                "power_draw_watts": power,
                "message": "GPU metrics are being reported correctly."
            })
            
        except requests.RequestException as e:
            return result.failed({
                "error": str(e),
                "message": "Failed to connect to GPU metrics endpoint."
            })
    
    def test_inference_performance(self) -> TestResult:
        """
        Test inference performance using a sample request.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Inference Performance", "performance").start()
        
        # Make a sample inference request
        sample_request = {
            "prompt": "What is HANA Cloud?",
            "max_tokens": 100
        }
        
        start_time = time.time()
        response = self.api_request("/api/v1/generate", method="POST", data=sample_request, timeout=60)
        end_time = time.time()
        
        if not response or response.status_code != 200:
            return result.failed({
                "message": "Inference request failed."
            })
        
        inference_latency_ms = (end_time - start_time) * 1000
        
        try:
            data = response.json()
            tokens_generated = len(data.get("text", "").split())
            tokens_per_second = tokens_generated / ((end_time - start_time) if end_time > start_time else 0.001)
            
            # Analyze performance
            if tokens_per_second < 5:  # Adjust threshold based on model size
                return result.failed({
                    "inference_latency_ms": inference_latency_ms,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                    "message": f"Inference performance is lower than expected: {tokens_per_second:.2f} tokens/sec."
                })
            
            return result.passed({
                "inference_latency_ms": inference_latency_ms,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "message": f"Inference performance is acceptable: {tokens_per_second:.2f} tokens/sec."
            })
            
        except Exception as e:
            return result.failed({
                "error": str(e),
                "inference_latency_ms": inference_latency_ms,
                "message": "Failed to analyze inference response."
            })
    
    # ====== MEMORY TESTS ======
    
    def test_memory_usage(self) -> TestResult:
        """
        Test GPU memory usage patterns.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Memory Usage", "memory").start()
        
        # Get current memory usage
        returncode, stdout, stderr = self.run_command(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
        )
        
        if returncode != 0:
            return result.failed({
                "error": stderr,
                "message": "Failed to get GPU memory usage."
            })
        
        # Parse CSV output
        try:
            parts = [int(p.strip()) for p in stdout.strip().split(',')]
            
            if len(parts) >= 2:
                used_memory = parts[0]
                total_memory = parts[1]
                usage_percent = (used_memory / total_memory) * 100
                
                memory_status = "optimal"
                memory_message = "Memory usage is optimal."
                
                if usage_percent < 5:
                    memory_status = "underutilized"
                    memory_message = "GPU memory is underutilized. Models may not be loaded."
                elif usage_percent > 95:
                    memory_status = "critical"
                    memory_message = "GPU memory usage is critically high. Risk of OOM errors."
                elif usage_percent > 85:
                    memory_status = "high"
                    memory_message = "GPU memory usage is high but within limits."
                
                return result.passed({
                    "used_memory_mb": used_memory,
                    "total_memory_mb": total_memory,
                    "usage_percent": usage_percent,
                    "memory_status": memory_status,
                    "message": memory_message
                })
            
            return result.failed({
                "output": stdout,
                "message": "Failed to parse GPU memory usage."
            })
            
        except Exception as e:
            return result.failed({
                "error": str(e),
                "output": stdout,
                "message": "Error processing memory usage data."
            })
    
    def test_memory_fragmentation(self) -> TestResult:
        """
        Test for GPU memory fragmentation.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Memory Fragmentation", "memory").start()
        
        if not TORCH_AVAILABLE:
            return result.skipped("PyTorch not available for memory fragmentation test")
        
        try:
            if not torch.cuda.is_available():
                return result.skipped("CUDA not available for memory fragmentation test")
            
            # Get current memory stats
            initial_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            initial_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            initial_free = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
            initial_free_mb = initial_free / (1024 * 1024)  # MB
            
            # Test fragmentation by allocating and freeing different sized tensors
            tensor_sizes = [10, 100, 1000, 10000, 5000, 2000, 500, 100, 50, 20]
            tensors = []
            
            # Allocate tensors of different sizes
            for size in tensor_sizes:
                tensors.append(torch.rand(size, size, device="cuda"))
            
            # Free every other tensor
            for i in range(0, len(tensors), 2):
                tensors[i] = None
            
            # Force garbage collection
            torch.cuda.empty_cache()
            
            # Get new memory stats
            new_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            new_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            # Calculate fragmentation
            fragmentation = new_reserved - new_allocated
            fragmentation_percent = (fragmentation / new_reserved) * 100 if new_reserved > 0 else 0
            
            # Clean up all tensors
            tensors = None
            torch.cuda.empty_cache()
            
            # Analyze fragmentation
            if fragmentation_percent > 20:
                return result.failed({
                    "initial_allocated_mb": initial_allocated,
                    "initial_reserved_mb": initial_reserved,
                    "final_allocated_mb": new_allocated,
                    "final_reserved_mb": new_reserved,
                    "fragmentation_mb": fragmentation,
                    "fragmentation_percent": fragmentation_percent,
                    "message": f"High memory fragmentation detected: {fragmentation_percent:.2f}%."
                })
            
            return result.passed({
                "initial_allocated_mb": initial_allocated,
                "initial_reserved_mb": initial_reserved,
                "final_allocated_mb": new_allocated,
                "final_reserved_mb": new_reserved,
                "fragmentation_mb": fragmentation,
                "fragmentation_percent": fragmentation_percent,
                "message": f"Memory fragmentation is acceptable: {fragmentation_percent:.2f}%."
            })
            
        except Exception as e:
            return result.failed({
                "error": str(e),
                "message": "Error testing memory fragmentation."
            })
    
    # ====== INTEGRATION TESTS ======
    
    def test_prometheus_integration(self) -> TestResult:
        """
        Test Prometheus integration.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Prometheus Integration", "integration").start()
        
        # Check Prometheus endpoint
        prometheus_url = f"http://{self.host}:{self.prometheus_port}/-/healthy"
        
        try:
            response = requests.get(prometheus_url, timeout=5)
            
            if response.status_code != 200:
                return result.failed({
                    "status_code": response.status_code,
                    "message": "Prometheus health check failed."
                })
            
            # Check if Prometheus is scraping GPU metrics
            prometheus_api_url = f"http://{self.host}:{self.prometheus_port}/api/v1/query"
            query_params = {"query": "nvidia_gpu_duty_cycle"}
            
            response = requests.get(prometheus_api_url, params=query_params, timeout=5)
            
            if response.status_code != 200:
                return result.failed({
                    "status_code": response.status_code,
                    "message": "Prometheus query API failed."
                })
            
            try:
                data = response.json()
                
                if data.get("status") != "success":
                    return result.failed({
                        "prometheus_response": data,
                        "message": "Prometheus query failed."
                    })
                
                results = data.get("data", {}).get("result", [])
                
                if not results:
                    return result.failed({
                        "message": "No GPU metrics found in Prometheus. Check exporter configuration."
                    })
                
                return result.passed({
                    "metrics_found": len(results),
                    "message": f"Prometheus is successfully collecting GPU metrics."
                })
                
            except Exception as e:
                return result.failed({
                    "error": str(e),
                    "message": "Failed to parse Prometheus response."
                })
            
        except requests.RequestException as e:
            return result.failed({
                "error": str(e),
                "message": "Failed to connect to Prometheus."
            })
    
    def test_grafana_integration(self) -> TestResult:
        """
        Test Grafana dashboard integration.
        
        Returns:
            TestResult object with test results
        """
        result = TestResult("Grafana Integration", "integration").start()
        
        # Check Grafana endpoint
        grafana_url = f"http://{self.host}:{self.grafana_port}/api/health"
        
        try:
            response = requests.get(grafana_url, timeout=5)
            
            if response.status_code != 200:
                return result.failed({
                    "status_code": response.status_code,
                    "message": "Grafana health check failed."
                })
            
            # Check for T4 dashboard
            grafana_search_url = f"http://{self.host}:{self.grafana_port}/api/search"
            
            response = requests.get(grafana_search_url, timeout=5)
            
            if response.status_code != 200:
                return result.failed({
                    "status_code": response.status_code,
                    "message": "Grafana search API failed."
                })
            
            try:
                dashboards = response.json()
                
                t4_dashboard_found = False
                for dashboard in dashboards:
                    if "t4" in dashboard.get("title", "").lower() or "gpu" in dashboard.get("title", "").lower():
                        t4_dashboard_found = True
                        break
                
                if not t4_dashboard_found:
                    return result.failed({
                        "dashboards": [d.get("title") for d in dashboards],
                        "message": "T4 GPU dashboard not found in Grafana."
                    })
                
                return result.passed({
                    "dashboards": [d.get("title") for d in dashboards],
                    "message": "Grafana is properly configured with T4 GPU dashboard."
                })
                
            except Exception as e:
                return result.failed({
                    "error": str(e),
                    "message": "Failed to parse Grafana response."
                })
            
        except requests.RequestException as e:
            return result.failed({
                "error": str(e),
                "message": "Failed to connect to Grafana."
            })
    
    # ====== RUN ALL TESTS ======
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and generate report.
        
        Returns:
            Dict with test results
        """
        logger.info(f"Starting automated tests on {self.host}...")
        
        # Environment tests
        logger.info("Running environment tests...")
        self.results["environment"].append(self.test_gpu_detection().to_dict())
        self.results["environment"].append(self.test_cuda_version().to_dict())
        self.results["environment"].append(self.test_docker_nvidia_runtime().to_dict())
        self.results["environment"].append(self.test_deployment_status().to_dict())
        
        # TensorRT tests
        logger.info("Running TensorRT tests...")
        self.results["tensorrt"].append(self.test_tensorrt_availability().to_dict())
        self.results["tensorrt"].append(self.test_tensorrt_optimization().to_dict())
        self.results["tensorrt"].append(self.test_fp16_support().to_dict())
        
        # Performance tests
        logger.info("Running performance tests...")
        self.results["performance"].append(self.test_api_latency().to_dict())
        self.results["performance"].append(self.test_gpu_utilization().to_dict())
        self.results["performance"].append(self.test_inference_performance().to_dict())
        
        # Memory tests
        logger.info("Running memory tests...")
        self.results["memory"].append(self.test_memory_usage().to_dict())
        self.results["memory"].append(self.test_memory_fragmentation().to_dict())
        
        # Integration tests
        logger.info("Running integration tests...")
        self.results["integration"].append(self.test_prometheus_integration().to_dict())
        self.results["integration"].append(self.test_grafana_integration().to_dict())
        
        # Compile overall results
        total_tests = sum(len(tests) for tests in self.results.values())
        passed_tests = sum(1 for category in self.results.values() for test in category if test["status"] == "passed")
        failed_tests = sum(1 for category in self.results.values() for test in category if test["status"] == "failed")
        skipped_tests = sum(1 for category in self.results.values() for test in category if test["status"] == "skipped")
        
        if failed_tests > 0:
            overall_status = "failed"
            overall_message = f"Automated tests completed with {failed_tests} failures."
        else:
            overall_status = "passed"
            overall_message = f"All automated tests passed successfully."
        
        # Add overall results
        overall_results = {
            "status": overall_status,
            "message": overall_message,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "pass_rate": (passed_tests / (total_tests - skipped_tests)) * 100 if (total_tests - skipped_tests) > 0 else 0,
            "categories": self.results,
            "timestamp": self.timestamp,
            "host": self.host,
            "system_info": self._get_system_info()
        }
        
        # Save report
        self._save_report(overall_results)
        
        # Generate HTML report
        self._generate_html_report(overall_results)
        
        logger.info(f"Automated tests completed. Results saved to {self.report_file}")
        
        return overall_results
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        # Add CUDA info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            system_info["gpu_count"] = torch.cuda.device_count()
        
        return system_info
    
    def _save_report(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        with open(self.report_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _generate_html_report(self, results: Dict[str, Any]):
        """Generate HTML report from results."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Automated Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #76b900;
        }}
        h1 {{
            margin: 0;
            color: #76b900;
        }}
        .summary {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            flex: 1;
            margin-right: 10px;
            text-align: center;
        }}
        .summary-box:last-child {{
            margin-right: 0;
        }}
        .summary-box h2 {{
            margin-top: 0;
            font-size: 16px;
        }}
        .summary-box .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .passed {{
            color: #28a745;
        }}
        .failed {{
            color: #dc3545;
        }}
        .skipped {{
            color: #ffc107;
        }}
        .category {{
            margin-bottom: 30px;
        }}
        .category-header {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
        }}
        .test {{
            margin-bottom: 15px;
            border-left: 5px solid #ddd;
            padding-left: 15px;
        }}
        .test.passed {{
            border-left-color: #28a745;
        }}
        .test.failed {{
            border-left-color: #dc3545;
        }}
        .test.skipped {{
            border-left-color: #ffc107;
        }}
        .test-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .test-name {{
            font-weight: bold;
        }}
        .test-status {{
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .test-status.passed {{
            background-color: #d4edda;
        }}
        .test-status.failed {{
            background-color: #f8d7da;
        }}
        .test-status.skipped {{
            background-color: #fff3cd;
        }}
        .test-message {{
            margin-bottom: 10px;
        }}
        .test-details {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .system-info {{
            margin-top: 30px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .system-info h2 {{
            margin-top: 0;
            font-size: 16px;
        }}
        .system-info table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .system-info table td {{
            padding: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .system-info table td:first-child {{
            font-weight: bold;
            width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Automated Test Report</h1>
            <p>Host: {results['host']} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="overall {results['status']}">Overall Status: {results['status'].upper()} - {results['message']}</p>
        </header>
        
        <div class="summary">
            <div class="summary-box">
                <h2>Total Tests</h2>
                <div class="value">{results['total_tests']}</div>
            </div>
            <div class="summary-box">
                <h2>Passed</h2>
                <div class="value passed">{results['passed_tests']}</div>
            </div>
            <div class="summary-box">
                <h2>Failed</h2>
                <div class="value failed">{results['failed_tests']}</div>
            </div>
            <div class="summary-box">
                <h2>Skipped</h2>
                <div class="value skipped">{results['skipped_tests']}</div>
            </div>
            <div class="summary-box">
                <h2>Pass Rate</h2>
                <div class="value">{results['pass_rate']:.1f}%</div>
            </div>
        </div>
"""
        
        # Add categories and tests
        for category_name, tests in results["categories"].items():
            # Count statuses for this category
            passed = sum(1 for test in tests if test["status"] == "passed")
            failed = sum(1 for test in tests if test["status"] == "failed")
            skipped = sum(1 for test in tests if test["status"] == "skipped")
            
            html += f"""
        <div class="category">
            <div class="category-header">
                <span>{category_name.replace('_', ' ').title()} Tests</span>
                <span>{passed} passed, {failed} failed, {skipped} skipped</span>
            </div>
"""
            
            for test in tests:
                html += f"""
            <div class="test {test['status']}">
                <div class="test-header">
                    <span class="test-name">{test['name']}</span>
                    <span class="test-status {test['status']}">{test['status'].upper()}</span>
                </div>
"""
                
                if "message" in test["details"]:
                    html += f"""
                <div class="test-message">{test['details']['message']}</div>
"""
                
                # Add other details
                details = {k: v for k, v in test["details"].items() if k != "message"}
                if details:
                    html += f"""
                <div class="test-details">{json.dumps(details, indent=2)}</div>
"""
                
                if test["duration_ms"] > 0:
                    html += f"""
                <div class="test-duration">Duration: {test['duration_ms']} ms</div>
"""
                
                html += """
            </div>
"""
            
            html += """
        </div>
"""
        
        # Add system info
        html += """
        <div class="system-info">
            <h2>System Information</h2>
            <table>
"""
        
        for key, value in results["system_info"].items():
            html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        with open(self.html_report_file, "w") as f:
            f.write(html)
    
    def open_html_report(self):
        """Open the HTML report in a web browser."""
        if os.path.exists(self.html_report_file):
            webbrowser.open(f"file://{os.path.abspath(self.html_report_file)}")
            logger.info(f"Opened HTML report in browser: {self.html_report_file}")
        else:
            logger.warning(f"HTML report not found: {self.html_report_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Automated Testing for NVIDIA T4 Deployments"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Hostname or IP of the NVIDIA instance (default: localhost)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the API service (default: 8000)"
    )
    
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=9091,
        help="Port for Prometheus (default: 9091)"
    )
    
    parser.add_argument(
        "--grafana-port",
        type=int,
        default=3000,
        help="Port for Grafana (default: 3000)"
    )
    
    parser.add_argument(
        "--gpu-metrics-port",
        type=int,
        default=9835,
        help="Port for NVIDIA GPU metrics exporter (default: 9835)"
    )
    
    parser.add_argument(
        "--ssh-key",
        help="Path to SSH key for remote testing"
    )
    
    parser.add_argument(
        "--username",
        help="Username for SSH connection"
    )
    
    parser.add_argument(
        "--output-dir",
        default="test-reports",
        help="Directory to store test reports (default: test-reports)"
    )
    
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open HTML report in browser when tests complete"
    )
    
    parser.add_argument(
        "--category",
        choices=["all", "environment", "tensorrt", "performance", "memory", "integration"],
        default="all",
        help="Test category to run (default: all)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the automated testing framework."""
    args = parse_arguments()
    
    # Initialize tester
    tester = T4AutomatedTester(
        host=args.host,
        api_port=args.api_port,
        prometheus_port=args.prometheus_port,
        grafana_port=args.grafana_port,
        gpu_metrics_port=args.gpu_metrics_port,
        ssh_key=args.ssh_key,
        username=args.username,
        output_dir=args.output_dir
    )
    
    # Run tests
    if args.category == "all":
        results = tester.run_all_tests()
    else:
        # Run specific category tests
        logger.info(f"Running {args.category} tests...")
        
        # This would need to be implemented for each category
        # For now, just run all tests
        results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"NVIDIA T4 AUTOMATED TEST RESULTS: {args.host}")
    print("=" * 80)
    print(f"Status: {results['status'].upper()}")
    print(f"Message: {results['message']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Skipped: {results['skipped_tests']}")
    print(f"Pass Rate: {results['pass_rate']:.1f}%")
    print(f"Report: {tester.report_file}")
    print(f"HTML Report: {tester.html_report_file}")
    print("=" * 80)
    
    # Open HTML report if requested
    if args.open_report:
        tester.open_html_report()
    
    # Return exit code based on test results
    sys.exit(0 if results["status"] == "passed" else 1)


if __name__ == "__main__":
    main()