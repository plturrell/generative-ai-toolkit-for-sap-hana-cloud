#!/usr/bin/env python3
"""
CI/CD Automated Testing for NVIDIA T4 Deployments.

This script is designed to be run in CI/CD pipelines to automatically
validate T4 GPU deployments with no manual intervention.

Usage:
    python ci_t4_test.py [--host HOST] [--port PORT]
"""
import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import datetime
import socket
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configuration
OUTPUT_DIR = "t4-test-results"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TEST_ID = f"ci-test-{TIMESTAMP}"
RESULTS_FILE = os.path.join(OUTPUT_DIR, f"{TEST_ID}.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, f"{TEST_ID}.html")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Results storage
results = {
    "timestamp": TIMESTAMP,
    "test_id": TEST_ID,
    "gpu": {},
    "api": {},
    "monitoring": {}
}

def run_command(command: str, timeout: int = 30) -> Tuple[int, str, str]:
    """
    Run a shell command and capture output.
    
    Args:
        command: Command string to execute
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    print(f"Running: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return 124, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def check_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Check if a port is open on a host.
    
    Args:
        host: Hostname or IP
        port: Port number
        timeout: Timeout in seconds
        
    Returns:
        True if port is open, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def test_nvidia_smi() -> Dict[str, Any]:
    """
    Test NVIDIA GPU detection using nvidia-smi.
    
    Returns:
        Dictionary with test results
    """
    print("\nTesting NVIDIA GPU detection...")
    
    result = {}
    
    # Run nvidia-smi
    returncode, stdout, stderr = run_command("nvidia-smi")
    
    if returncode != 0:
        print("❌ Failed to run nvidia-smi")
        result["detected"] = False
        result["error"] = stderr
        return result
    
    # Check for T4 in output
    t4_detected = "T4" in stdout
    if t4_detected:
        print("✅ NVIDIA T4 GPU detected")
        result["detected"] = True
        result["is_t4"] = True
    else:
        print("⚠️ NVIDIA GPU detected but not a T4")
        result["detected"] = True
        result["is_t4"] = False
    
    # Get more GPU details
    returncode, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader"
    )
    
    if returncode == 0:
        parts = [p.strip() for p in stdout.strip().split(',')]
        
        if len(parts) >= 8:
            result["model"] = parts[0]
            result["memory_total"] = parts[1]
            result["driver_version"] = parts[2]
            result["compute_capability"] = parts[3]
            result["utilization"] = parts[4]
            result["memory_used"] = parts[5]
            result["temperature"] = parts[6]
            result["power_draw"] = parts[7]
            
            print(f"  Model: {parts[0]}")
            print(f"  Memory: {parts[1]}")
            print(f"  Driver: {parts[2]}")
            print(f"  Compute Capability: {parts[3]}")
            print(f"  Utilization: {parts[4]}")
            print(f"  Memory Used: {parts[5]}")
            print(f"  Temperature: {parts[6]}")
            print(f"  Power Draw: {parts[7]}")
    
    # Get raw info
    result["raw_info"] = stdout
    
    return result

def test_cuda_pytorch() -> Dict[str, Any]:
    """
    Test CUDA and PyTorch installation.
    
    Returns:
        Dictionary with test results
    """
    print("\nTesting CUDA and PyTorch...")
    
    result = {}
    
    # Create a temporary Python script for testing CUDA
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        
    # Test tensor operations with timing
    device = torch.device('cuda')
    
    # FP32 test
    a = torch.randn(2048, 2048, dtype=torch.float32, device=device)
    b = torch.randn(2048, 2048, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(10):
        c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    fp32_time = start.elapsed_time(end) / 10
    print(f"FP32 matmul time: {fp32_time:.3f} ms")
    
    # FP16 test (for Tensor Cores)
    if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer supports Tensor Cores
        a_half = a.half()
        b_half = b.half()
        
        # Warmup
        for _ in range(5):
            c_half = torch.matmul(a_half, b_half)
        torch.cuda.synchronize()
        
        # Benchmark
        start.record()
        for _ in range(10):
            c_half = torch.matmul(a_half, b_half)
        end.record()
        torch.cuda.synchronize()
        fp16_time = start.elapsed_time(end) / 10
        print(f"FP16 matmul time: {fp16_time:.3f} ms")
        print(f"Speedup (FP32 → FP16): {fp32_time / fp16_time:.2f}x")
    else:
        print("GPU does not support Tensor Cores (compute capability < 7.0)")
        
    # Memory info
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("CUDA not available")
""")
        script_path = f.name
    
    # Run the script
    returncode, stdout, stderr = run_command(f"python3 {script_path}")
    
    # Clean up
    try:
        os.unlink(script_path)
    except:
        pass
    
    if returncode != 0:
        print("❌ Failed to run CUDA/PyTorch test")
        result["success"] = False
        result["error"] = stderr
        return result
    
    # Parse output
    result["success"] = True
    result["output"] = stdout
    
    # Extract key information
    if "CUDA available: True" in stdout:
        result["cuda_available"] = True
        
        # Extract CUDA version
        import re
        cuda_match = re.search(r"CUDA version: ([\d\.]+)", stdout)
        if cuda_match:
            result["cuda_version"] = cuda_match.group(1)
        
        # Extract GPU count
        gpu_count_match = re.search(r"Number of GPUs: (\d+)", stdout)
        if gpu_count_match:
            result["gpu_count"] = int(gpu_count_match.group(1))
        
        # Extract GPU name
        gpu_name_match = re.search(r"GPU 0: (.*)", stdout)
        if gpu_name_match:
            result["gpu_name"] = gpu_name_match.group(1)
            result["is_t4"] = "T4" in result["gpu_name"]
        
        # Extract performance metrics
        fp32_time_match = re.search(r"FP32 matmul time: ([\d\.]+) ms", stdout)
        if fp32_time_match:
            result["fp32_time_ms"] = float(fp32_time_match.group(1))
        
        fp16_time_match = re.search(r"FP16 matmul time: ([\d\.]+) ms", stdout)
        if fp16_time_match:
            result["fp16_time_ms"] = float(fp16_time_match.group(1))
        
        speedup_match = re.search(r"Speedup \(FP32 → FP16\): ([\d\.]+)x", stdout)
        if speedup_match:
            result["speedup"] = float(speedup_match.group(1))
            
            # Check if Tensor Cores are likely being used
            if result["speedup"] > 1.8:
                result["tensor_cores_active"] = True
                print("✅ Tensor Cores are active (good speedup)")
            else:
                result["tensor_cores_active"] = False
                print("⚠️ Tensor Cores may not be fully utilized (low speedup)")
    else:
        result["cuda_available"] = False
    
    return result

def test_tensorrt() -> Dict[str, Any]:
    """
    Test TensorRT installation and configuration with detailed optimization checks.
    
    Returns:
        Dictionary with test results including optimization capabilities
    """
    print("\nTesting TensorRT...")
    
    result = {}
    
    # Create a temporary Python script for testing TensorRT
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
import sys
import time
import numpy as np

try:
    import tensorrt as trt
    print("TensorRT version:", trt.__version__)
    
    # Get logger and create builder
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Check for precision support
    fp16_supported = builder.platform_has_fast_fp16
    int8_supported = builder.platform_has_fast_int8
    print("FP16 support:", fp16_supported)
    print("INT8 support:", int8_supported)
    
    # Check for DLA (Deep Learning Accelerator)
    dla_supported = hasattr(builder, "num_DLA_cores") and builder.num_DLA_cores > 0
    print("DLA support:", dla_supported)
    print("DLA cores:", builder.num_DLA_cores if dla_supported else 0)
    
    # Create network and config
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    
    # Check workspace size (important for T4 performance)
    default_workspace_size = config.max_workspace_size
    print(f"Default workspace size: {default_workspace_size / (1024*1024):.0f} MB")
    
    # Check for tactic sources
    tactic_sources = config.get_tactic_sources()
    print("Default tactic sources:", tactic_sources)
    
    # Check builder optimization level
    print("Optimization level:", config.builder_optimization_level)
    
    # Test a simple TensorRT model with FP16 if supported
    if fp16_supported:
        try:
            import torch
            
            # Create a simple model for testing
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(512, 512)
                    self.relu = torch.nn.ReLU()
                    self.fc2 = torch.nn.Linear(512, 512)
                
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
            
            # Create model and input
            model = SimpleModel().eval().cuda()
            input_tensor = torch.randn(1, 512, dtype=torch.float32).cuda()
            
            # Trace model
            traced_model = torch.jit.trace(model, input_tensor)
            
            # Test FP32 performance
            warmup = 5
            iterations = 10
            
            # Warmup
            for _ in range(warmup):
                output = traced_model(input_tensor)
                torch.cuda.synchronize()
            
            # Benchmark FP32
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(iterations):
                output = traced_model(input_tensor)
                torch.cuda.synchronize()
            fp32_time = (time.time() - start_time) * 1000 / iterations
            print(f"PyTorch FP32 inference time: {fp32_time:.3f} ms")
            
            # Test with TensorRT FP16
            try:
                import torch_tensorrt
                
                # Compile with TensorRT in FP16
                trt_model = torch_tensorrt.compile(
                    traced_model,
                    inputs=[torch_tensorrt.Input(input_tensor.shape)],
                    enabled_precisions={torch.float16}
                )
                
                # Warmup
                for _ in range(warmup):
                    output = trt_model(input_tensor)
                    torch.cuda.synchronize()
                
                # Benchmark FP16
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(iterations):
                    output = trt_model(input_tensor)
                    torch.cuda.synchronize()
                fp16_time = (time.time() - start_time) * 1000 / iterations
                print(f"TensorRT FP16 inference time: {fp16_time:.3f} ms")
                
                # Calculate speedup
                speedup = fp32_time / fp16_time
                print(f"TensorRT FP16 speedup: {speedup:.2f}x")
                
                # Check for optimal T4 performance (should be at least 1.8x)
                if speedup >= 1.8:
                    print("✅ TensorRT FP16 is properly optimized for T4")
                else:
                    print("⚠️ TensorRT FP16 speedup is lower than expected for T4")
                
                torch_tensorrt_available = True
            except (ImportError, RuntimeError) as e:
                print(f"torch_tensorrt not available or error: {e}")
                torch_tensorrt_available = False
        except ImportError:
            print("PyTorch not available for TensorRT optimization testing")
            torch_tensorrt_available = False
    
    # Check environment variables that affect TensorRT
    env_vars = [
        "TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION",
        "TENSORRT_PRECISION",
        "TENSORRT_MAX_WORKSPACE_SIZE",
        "TENSORRT_BUILDER_OPTIMIZATION_LEVEL",
        "TENSORRT_ENGINE_CACHE_ENABLE"
    ]
    
    for var in env_vars:
        val = os.environ.get(var, "Not set")
        print(f"Environment variable {var}: {val}")
    
    # Success
    print("TensorRT test successful")
    sys.exit(0)
except ImportError:
    print("TensorRT not installed")
    sys.exit(1)
except Exception as e:
    print(f"Error testing TensorRT: {e}")
    sys.exit(2)
""")
        script_path = f.name
    
    # Run the script
    returncode, stdout, stderr = run_command(f"python3 {script_path}", timeout=60)
    
    # Clean up
    try:
        os.unlink(script_path)
    except:
        pass
    
    if returncode != 0:
        print("❌ TensorRT test failed")
        result["available"] = False
        result["error"] = stderr if stderr else stdout
        return result
    
    # Parse output
    result["available"] = True
    result["output"] = stdout
    
    # Extract version
    import re
    version_match = re.search(r"TensorRT version: ([\d\.]+)", stdout)
    if version_match:
        result["version"] = version_match.group(1)
    
    # Extract FP16 support
    fp16_match = re.search(r"FP16 support: (True|False)", stdout)
    if fp16_match:
        result["fp16_support"] = fp16_match.group(1) == "True"
        if result["fp16_support"]:
            print("✅ TensorRT supports FP16 (good for T4)")
        else:
            print("❌ TensorRT does not support FP16")
    
    # Extract INT8 support
    int8_match = re.search(r"INT8 support: (True|False)", stdout)
    if int8_match:
        result["int8_support"] = int8_match.group(1) == "True"
    
    # Extract DLA support
    dla_match = re.search(r"DLA support: (True|False)", stdout)
    if dla_match:
        result["dla_support"] = dla_match.group(1) == "True"
    
    # Extract workspace size
    workspace_match = re.search(r"Default workspace size: ([\d\.]+) MB", stdout)
    if workspace_match:
        result["workspace_mb"] = float(workspace_match.group(1))
        # T4 performs well with at least 1024MB workspace
        result["workspace_sufficient"] = result["workspace_mb"] >= 1024
    
    # Extract optimization level
    opt_level_match = re.search(r"Optimization level: (\d+)", stdout)
    if opt_level_match:
        result["optimization_level"] = int(opt_level_match.group(1))
        # Level 3 is recommended for T4
        result["optimal_optimization_level"] = result["optimization_level"] >= 3
    
    # Extract performance metrics if available
    fp32_time_match = re.search(r"PyTorch FP32 inference time: ([\d\.]+) ms", stdout)
    fp16_time_match = re.search(r"TensorRT FP16 inference time: ([\d\.]+) ms", stdout)
    speedup_match = re.search(r"TensorRT FP16 speedup: ([\d\.]+)x", stdout)
    
    if fp32_time_match and fp16_time_match and speedup_match:
        result["fp32_time_ms"] = float(fp32_time_match.group(1))
        result["fp16_time_ms"] = float(fp16_time_match.group(1))
        result["speedup"] = float(speedup_match.group(1))
        
        # Check if optimization is good for T4
        result["optimal_tensorrt_speedup"] = result["speedup"] >= 1.8
        if result["optimal_tensorrt_speedup"]:
            print(f"✅ Excellent TensorRT optimization (speedup: {result['speedup']:.2f}x)")
        else:
            print(f"⚠️ Suboptimal TensorRT speedup for T4 ({result['speedup']:.2f}x, should be at least 1.8x)")
    
    # Check TensorRT environment variables
    env_vars = {}
    env_var_pattern = r"Environment variable ([\w_]+): (.+)"
    for match in re.finditer(env_var_pattern, stdout):
        var_name = match.group(1)
        var_value = match.group(2)
        env_vars[var_name] = var_value
    
    if env_vars:
        result["environment_variables"] = env_vars
        
        # Check for optimal settings
        if env_vars.get("TENSORRT_PRECISION", "Not set") == "fp16":
            result["optimal_precision_setting"] = True
        
        workspace_setting = env_vars.get("TENSORRT_MAX_WORKSPACE_SIZE", "Not set")
        if workspace_setting != "Not set":
            try:
                workspace_mb = int(workspace_setting) / (1024*1024)
                result["configured_workspace_mb"] = workspace_mb
                result["optimal_workspace_setting"] = workspace_mb >= 1024
            except:
                pass
    
    # Overall optimization assessment
    optimization_checks = [
        result.get("fp16_support", False),
        result.get("optimal_tensorrt_speedup", False),
        result.get("workspace_sufficient", False),
        result.get("optimal_optimization_level", False)
    ]
    
    optimization_score = sum(1 for check in optimization_checks if check) / len(optimization_checks) if optimization_checks else 0
    result["optimization_score"] = optimization_score
    
    if optimization_score >= 0.75:
        print("✅ TensorRT is well optimized for T4 GPU")
    elif optimization_score >= 0.5:
        print("⚠️ TensorRT optimization is acceptable but could be improved")
    else:
        print("❌ TensorRT is not optimally configured for T4 GPU")
    
    return result

def test_api(host: str, port: int, auth_token: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
    """
    Test API endpoints with optional authentication.
    
    Args:
        host: API host
        port: API port
        auth_token: Optional Bearer token for authorization
        username: Optional username for basic auth
        password: Optional password for basic auth
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting API endpoints at {host}:{port}...")
    
    result = {
        "host": host,
        "port": port,
        "endpoints": {},
        "auth_method": "none"
    }
    
    # Test if port is open
    if not check_port(host, port):
        print(f"❌ Port {port} is not open on {host}")
        result["port_open"] = False
        return result
    
    result["port_open"] = True
    
    # Prepare authentication
    headers = {}
    auth = None
    
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        result["auth_method"] = "bearer"
        print(f"Using Bearer token authentication")
    elif username and password:
        auth = (username, password)
        result["auth_method"] = "basic"
        print(f"Using Basic authentication with username: {username}")
    
    # Define endpoints to test
    endpoints = [
        {"path": "/", "name": "Root"},
        {"path": "/health", "name": "Health"},
        {"path": "/api/v1/system/info", "name": "System Info"},
        {"path": "/api/v1/system/gpu", "name": "GPU Info"}
    ]
    
    # First check if auth is needed by testing the root endpoint without auth
    root_url = f"http://{host}:{port}/"
    try:
        root_response = requests.get(root_url, timeout=10)
        if root_response.status_code == 401:
            result["auth_required"] = True
            print("🔒 Authentication is required for this API")
            
            # If no auth provided, but auth is required
            if not auth_token and not (username and password):
                print("❌ Authentication required but no credentials provided")
                result["auth_error"] = "Authentication required but no credentials provided"
                
                # Try to extract auth requirements from response
                auth_header = root_response.headers.get("WWW-Authenticate", "")
                if auth_header:
                    result["auth_header"] = auth_header
                    print(f"Auth header: {auth_header}")
                
                return result
        else:
            result["auth_required"] = False
    except requests.RequestException:
        pass  # Continue with testing as normal
    
    # Test each endpoint
    for endpoint in endpoints:
        path = endpoint["path"]
        name = endpoint["name"]
        url = f"http://{host}:{port}{path}"
        
        print(f"Testing {name} endpoint: {url}")
        
        try:
            # Make request with appropriate auth
            response = requests.get(url, headers=headers, auth=auth, timeout=10)
            
            result["endpoints"][name] = {
                "path": path,
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300
            }
            
            if response.status_code == 200:
                print(f"✅ {name} endpoint returned status 200")
                
                # Try to parse JSON response
                try:
                    data = response.json()
                    result["endpoints"][name]["data"] = data
                    
                    # For GPU endpoint, check for T4
                    if name == "GPU Info" and isinstance(data, dict):
                        if data.get("name", "").find("T4") >= 0:
                            result["endpoints"][name]["is_t4"] = True
                            print("✅ T4 GPU detected in API response")
                        else:
                            result["endpoints"][name]["is_t4"] = False
                except:
                    result["endpoints"][name]["data"] = response.text[:500]
            elif response.status_code == 401:
                print(f"❌ {name} endpoint authentication failed (401 Unauthorized)")
                result["endpoints"][name]["auth_failed"] = True
            elif response.status_code == 403:
                print(f"❌ {name} endpoint permission denied (403 Forbidden)")
                result["endpoints"][name]["permission_denied"] = True
            else:
                print(f"❌ {name} endpoint returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Failed to connect to {name} endpoint: {e}")
            result["endpoints"][name] = {
                "path": path,
                "error": str(e),
                "success": False
            }
    
    # Test additional authenticated endpoints if auth is provided
    if auth_token or (username and password):
        # Add specific authenticated endpoints to test
        auth_endpoints = [
            {"path": "/api/v1/system/status", "name": "System Status"},
            {"path": "/api/v1/models", "name": "Models List"}
        ]
        
        for endpoint in auth_endpoints:
            path = endpoint["path"]
            name = endpoint["name"]
            url = f"http://{host}:{port}{path}"
            
            print(f"Testing authenticated endpoint {name}: {url}")
            
            try:
                response = requests.get(url, headers=headers, auth=auth, timeout=10)
                
                result["endpoints"][name] = {
                    "path": path,
                    "status_code": response.status_code,
                    "success": 200 <= response.status_code < 300,
                    "authenticated": True
                }
                
                if response.status_code == 200:
                    print(f"✅ Authenticated {name} endpoint returned status 200")
                    
                    # Try to parse JSON response
                    try:
                        data = response.json()
                        result["endpoints"][name]["data"] = data
                    except:
                        result["endpoints"][name]["data"] = response.text[:500]
                elif response.status_code == 401:
                    print(f"❌ {name} endpoint authentication failed (401 Unauthorized)")
                    result["endpoints"][name]["auth_failed"] = True
                elif response.status_code == 403:
                    print(f"❌ {name} endpoint permission denied (403 Forbidden)")
                    result["endpoints"][name]["permission_denied"] = True
                else:
                    print(f"❌ {name} endpoint returned status {response.status_code}")
            except requests.RequestException as e:
                print(f"❌ Failed to connect to authenticated {name} endpoint: {e}")
                result["endpoints"][name] = {
                    "path": path,
                    "error": str(e),
                    "success": False,
                    "authenticated": True
                }
    
    return result

def test_monitoring(host: str) -> Dict[str, Any]:
    """
    Test monitoring components (Prometheus, Grafana).
    
    Args:
        host: Host name
        
    Returns:
        Dictionary with test results
    """
    print("\nTesting monitoring components...")
    
    result = {
        "prometheus": {},
        "grafana": {},
        "gpu_exporter": {}
    }
    
    # Test Prometheus
    prometheus_port = 9091
    if check_port(host, prometheus_port):
        print(f"✅ Prometheus port {prometheus_port} is open")
        result["prometheus"]["port_open"] = True
        
        # Test Prometheus API
        prometheus_url = f"http://{host}:{prometheus_port}/-/healthy"
        try:
            response = requests.get(prometheus_url, timeout=5)
            result["prometheus"]["healthy"] = response.status_code == 200
            
            if response.status_code == 200:
                print("✅ Prometheus is healthy")
                
                # Check for GPU metrics
                query_url = f"http://{host}:{prometheus_port}/api/v1/query"
                params = {"query": "nvidia_gpu_duty_cycle"}
                
                try:
                    response = requests.get(query_url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("status") == "success" and len(data.get("data", {}).get("result", [])) > 0:
                            print("✅ Prometheus has GPU metrics")
                            result["prometheus"]["gpu_metrics"] = True
                        else:
                            print("❌ No GPU metrics found in Prometheus")
                            result["prometheus"]["gpu_metrics"] = False
                except:
                    result["prometheus"]["gpu_metrics"] = False
            else:
                print(f"❌ Prometheus health check failed: {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Failed to connect to Prometheus: {e}")
            result["prometheus"]["healthy"] = False
            result["prometheus"]["error"] = str(e)
    else:
        print(f"❌ Prometheus port {prometheus_port} is not open")
        result["prometheus"]["port_open"] = False
    
    # Test Grafana
    grafana_port = 3000
    if check_port(host, grafana_port):
        print(f"✅ Grafana port {grafana_port} is open")
        result["grafana"]["port_open"] = True
        
        # Test Grafana API
        grafana_url = f"http://{host}:{grafana_port}/api/health"
        try:
            response = requests.get(grafana_url, timeout=5)
            result["grafana"]["healthy"] = response.status_code == 200
            
            if response.status_code == 200:
                print("✅ Grafana is healthy")
            else:
                print(f"❌ Grafana health check failed: {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Failed to connect to Grafana: {e}")
            result["grafana"]["healthy"] = False
            result["grafana"]["error"] = str(e)
    else:
        print(f"❌ Grafana port {grafana_port} is not open")
        result["grafana"]["port_open"] = False
    
    # Test GPU metrics exporter
    gpu_exporter_port = 9835
    if check_port(host, gpu_exporter_port):
        print(f"✅ GPU exporter port {gpu_exporter_port} is open")
        result["gpu_exporter"]["port_open"] = True
        
        # Test metrics endpoint
        metrics_url = f"http://{host}:{gpu_exporter_port}/metrics"
        try:
            response = requests.get(metrics_url, timeout=5)
            result["gpu_exporter"]["working"] = response.status_code == 200
            
            if response.status_code == 200:
                print("✅ GPU metrics exporter is working")
                
                # Check for NVIDIA metrics
                if "nvidia_gpu" in response.text:
                    print("✅ NVIDIA GPU metrics found")
                    result["gpu_exporter"]["nvidia_metrics"] = True
                    
                    # Parse metrics
                    import re
                    
                    # GPU utilization
                    util_match = re.search(r'nvidia_gpu_duty_cycle\S*\s+([\d\.]+)', response.text)
                    if util_match:
                        result["gpu_exporter"]["utilization"] = float(util_match.group(1))
                    
                    # Memory usage
                    mem_match = re.search(r'nvidia_gpu_memory_used_bytes\S*\s+([\d\.]+)', response.text)
                    if mem_match:
                        result["gpu_exporter"]["memory_used_bytes"] = float(mem_match.group(1))
                        result["gpu_exporter"]["memory_used_mb"] = float(mem_match.group(1)) / (1024 * 1024)
                    
                    # Temperature
                    temp_match = re.search(r'nvidia_gpu_temperature_celsius\S*\s+([\d\.]+)', response.text)
                    if temp_match:
                        result["gpu_exporter"]["temperature"] = float(temp_match.group(1))
                    
                    # Power
                    power_match = re.search(r'nvidia_gpu_power_draw_watts\S*\s+([\d\.]+)', response.text)
                    if power_match:
                        result["gpu_exporter"]["power_watts"] = float(power_match.group(1))
                else:
                    print("❌ No NVIDIA GPU metrics found")
                    result["gpu_exporter"]["nvidia_metrics"] = False
            else:
                print(f"❌ GPU metrics exporter returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Failed to connect to GPU metrics exporter: {e}")
            result["gpu_exporter"]["working"] = False
            result["gpu_exporter"]["error"] = str(e)
    else:
        print(f"❌ GPU exporter port {gpu_exporter_port} is not open")
        result["gpu_exporter"]["port_open"] = False
    
    return result

def generate_html_report(results: Dict[str, Any], report_file: str) -> None:
    """
    Generate HTML report from test results.
    
    Args:
        results: Test results dictionary
        report_file: Output file path
    """
    print(f"\nGenerating HTML report: {report_file}")
    
    # Determine overall status
    gpu_detected = False
    is_t4 = False
    
    if results.get("gpu", {}).get("is_t4", False):
        gpu_detected = True
        is_t4 = True
    elif results.get("pytorch", {}).get("is_t4", False):
        gpu_detected = True
        is_t4 = True
    elif results.get("api", {}).get("endpoints", {}).get("GPU Info", {}).get("is_t4", False):
        gpu_detected = True
        is_t4 = True
    
    if is_t4:
        status = "PASS"
        status_class = "pass"
        status_message = "T4 GPU detected and validated"
    elif gpu_detected:
        status = "PARTIAL"
        status_class = "partial"
        status_message = "GPU detected but not confirmed as T4"
    else:
        status = "FAIL"
        status_class = "fail"
        status_message = "No GPU detected or tests failed"
    
    # Create HTML content with enhanced styling for visual performance graphs
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 CI Test Report</title>
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
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .summary {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            flex-basis: 48%;
            margin-bottom: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }}
        .summary-box:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .summary-box h2 {{
            margin-top: 0;
            font-size: 16px;
            color: #555;
        }}
        .summary-box .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .pass {{
            color: #28a745;
        }}
        .partial {{
            color: #ffc107;
        }}
        .fail {{
            color: #dc3545;
        }}
        .info {{
            color: #17a2b8;
        }}
        .success {{
            color: #28a745;
        }}
        .warning {{
            color: #ffc107;
        }}
        .failure {{
            color: #dc3545;
        }}
        .section {{
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
            color: #555;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            background-color: white;
            border-radius: 3px;
            overflow: hidden;
        }}
        table th, table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        table th {{
            background-color: #f2f2f2;
            font-weight: 600;
            color: #555;
            position: sticky;
            top: 0;
        }}
        table tr:hover {{
            background-color: #f9f9f9;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.4;
        }}
        
        /* Performance graph styles */
        .perf-bar {{
            height: 20px;
            background-color: #76b900;
            border-radius: 3px;
            margin-top: 5px;
            transition: width 0.5s ease-in-out;
            position: relative;
        }}
        .perf-container {{
            margin-top: 10px;
            background-color: #f5f5f5;
            border-radius: 3px;
            padding: 2px;
            position: relative;
        }}
        .perf-label {{
            position: absolute;
            right: 5px;
            top: 2px;
            font-size: 12px;
            color: #444;
        }}
        .speedup-bar {{
            display: flex;
            align-items: center;
            margin-top: 8px;
            margin-bottom: 8px;
        }}
        .speedup-marker {{
            position: relative;
            height: 30px;
        }}
        .speedup-value {{
            height: 30px;
            line-height: 30px;
            padding-left: 10px;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            text-align: center;
            border-radius: 3px;
            transition: all 0.3s ease;
        }}
        .speedup-target {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #dc3545;
        }}
        .speedup-target::after {{
            content: 'Target (1.8x)';
            position: absolute;
            top: -20px;
            left: -30px;
            font-size: 11px;
            color: #dc3545;
            white-space: nowrap;
        }}
        .optimization-tips {{
            background-color: #fff;
            border-left: 4px solid #17a2b8;
            padding: 10px 15px;
            margin-top: 20px;
            border-radius: 0 3px 3px 0;
        }}
        .optimization-tips h3 {{
            margin-top: 0;
            color: #17a2b8;
        }}
        .optimization-tips ul {{
            margin-bottom: 0;
            padding-left: 20px;
        }}
        .optimization-tips li {{
            margin-bottom: 5px;
        }}
        
        /* Stats circle styles */
        .stats-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .stat-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            margin: 10px;
            position: relative;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .stat-circle:hover {{
            transform: scale(1.05);
        }}
        .stat-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-unit {{
            font-size: 12px;
            opacity: 0.7;
        }}
        .gpu-utilization {{
            background: linear-gradient(135deg, #76b900, #5a8f00);
            color: white;
        }}
        .gpu-memory {{
            background: linear-gradient(135deg, #1a73e8, #1559b7);
            color: white;
        }}
        .gpu-temp {{
            background: linear-gradient(135deg, #f6c142, #f28c38);
            color: white;
        }}
        .gpu-power {{
            background: linear-gradient(135deg, #e53935, #b71c1c);
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 CI Test Report</h1>
            <p>Test ID: {TEST_ID} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="summary">
            <div class="summary-box">
                <h2>Overall Status</h2>
                <div class="value {status_class}">{status}</div>
            </div>
            <div class="summary-box">
                <h2>Summary</h2>
                <div>{status_message}</div>
            </div>
        </div>
"""

    # Add GPU section
    if "gpu" in results and results["gpu"]:
        gpu = results["gpu"]
        gpu_status = "success" if gpu.get("detected", False) else "failure"
        t4_status = "success" if gpu.get("is_t4", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>GPU Detection (nvidia-smi)</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Result</th>
                </tr>
                <tr>
                    <td>GPU Detected</td>
                    <td class="{gpu_status}">{"✅ Yes" if gpu.get("detected", False) else "❌ No"}</td>
                </tr>
                <tr>
                    <td>T4 GPU</td>
                    <td class="{t4_status}">{"✅ Yes" if gpu.get("is_t4", False) else "❌ No"}</td>
                </tr>
"""
        
        if gpu.get("detected", False):
            html += f"""
                <tr>
                    <td>Model</td>
                    <td>{gpu.get("model", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Memory</td>
                    <td>{gpu.get("memory_total", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Driver Version</td>
                    <td>{gpu.get("driver_version", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Compute Capability</td>
                    <td>{gpu.get("compute_capability", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Utilization</td>
                    <td>{gpu.get("utilization", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Memory Used</td>
                    <td>{gpu.get("memory_used", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Temperature</td>
                    <td>{gpu.get("temperature", "Unknown")}</td>
                </tr>
                <tr>
                    <td>Power Draw</td>
                    <td>{gpu.get("power_draw", "Unknown")}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
    
    # Add CUDA/PyTorch section
    if "pytorch" in results and results["pytorch"]:
        pytorch = results["pytorch"]
        cuda_status = "success" if pytorch.get("cuda_available", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>CUDA & PyTorch</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Result</th>
                </tr>
                <tr>
                    <td>CUDA Available</td>
                    <td class="{cuda_status}">{"✅ Yes" if pytorch.get("cuda_available", False) else "❌ No"}</td>
                </tr>
"""
        
        if pytorch.get("cuda_available", False):
            html += f"""
                <tr>
                    <td>CUDA Version</td>
                    <td>{pytorch.get("cuda_version", "Unknown")}</td>
                </tr>
                <tr>
                    <td>GPU Count</td>
                    <td>{pytorch.get("gpu_count", "Unknown")}</td>
                </tr>
                <tr>
                    <td>GPU Name</td>
                    <td>{pytorch.get("gpu_name", "Unknown")}</td>
                </tr>
"""
            
            # Add performance metrics if available
            if "fp32_time_ms" in pytorch and "fp16_time_ms" in pytorch:
                max_time = max(pytorch["fp32_time_ms"], pytorch["fp16_time_ms"])
                fp32_width = (pytorch["fp32_time_ms"] / max_time) * 100
                fp16_width = (pytorch["fp16_time_ms"] / max_time) * 100
                
                speedup_class = "pass" if pytorch.get("speedup", 0) > 1.8 else "partial" if pytorch.get("speedup", 0) > 1.3 else "fail"
                
                html += f"""
                <tr>
                    <td>FP32 Matmul Time</td>
                    <td>
                        {pytorch["fp32_time_ms"]:.2f} ms
                        <div class="perf-bar" style="width: {fp32_width}%"></div>
                    </td>
                </tr>
                <tr>
                    <td>FP16 Matmul Time</td>
                    <td>
                        {pytorch["fp16_time_ms"]:.2f} ms
                        <div class="perf-bar" style="width: {fp16_width}%"></div>
                    </td>
                </tr>
                <tr>
                    <td>Speedup (FP32 → FP16)</td>
                    <td class="{speedup_class}">{pytorch.get("speedup", 0):.2f}x{" (Tensor Cores active)" if pytorch.get("tensor_cores_active", False) else ""}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
    
    # Add TensorRT section with enhanced optimization details
    if "tensorrt" in results and results["tensorrt"]:
        tensorrt = results["tensorrt"]
        tensorrt_status = "success" if tensorrt.get("available", False) else "failure"
        optimization_score = tensorrt.get("optimization_score", 0)
        
        # Determine optimization color class
        if optimization_score >= 0.75:
            optimization_class = "pass"
        elif optimization_score >= 0.5:
            optimization_class = "partial"
        else:
            optimization_class = "fail"
        
        html += f"""
        <div class="section">
            <h2>TensorRT</h2>
            <div class="summary">
                <div class="summary-box">
                    <h2>TensorRT Available</h2>
                    <div class="{tensorrt_status}">{"✅ Yes" if tensorrt.get("available", False) else "❌ No"}</div>
                </div>
                <div class="summary-box">
                    <h2>T4 Optimization Score</h2>
                    <div class="{optimization_class}">{int(optimization_score * 100)}%</div>
                </div>
            </div>
            
            <table>
                <tr>
                    <th>Test</th>
                    <th>Result</th>
                    <th>Details</th>
                </tr>
"""
        
        if tensorrt.get("available", False):
            # Basic information
            fp16_status = "success" if tensorrt.get("fp16_support", False) else "failure"
            int8_status = "success" if tensorrt.get("int8_support", False) else "warning"
            
            html += f"""
                <tr>
                    <td>TensorRT Version</td>
                    <td>{tensorrt.get("version", "Unknown")}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>FP16 Support</td>
                    <td class="{fp16_status}">{"✅ Yes" if tensorrt.get("fp16_support", False) else "❌ No"}</td>
                    <td>{"T4 GPUs require FP16 support for optimal performance" if not tensorrt.get("fp16_support", False) else "Required for Tensor Core acceleration"}</td>
                </tr>
                <tr>
                    <td>INT8 Support</td>
                    <td class="{int8_status}">{"✅ Yes" if tensorrt.get("int8_support", False) else "❌ No"}</td>
                    <td>{"Helpful for additional performance but not critical" if not tensorrt.get("int8_support", False) else "Can provide additional optimization for certain workloads"}</td>
                </tr>
"""

            # DLA support (if available)
            if "dla_support" in tensorrt:
                dla_status = "success" if tensorrt.get("dla_support", False) else "info"
                html += f"""
                <tr>
                    <td>DLA Support</td>
                    <td class="{dla_status}">{"✅ Yes" if tensorrt.get("dla_support", False) else "ℹ️ No"}</td>
                    <td>{"Deep Learning Accelerator cores available" if tensorrt.get("dla_support", False) else "Not available on T4 GPUs (normal)"}</td>
                </tr>
"""

            # Workspace size
            if "workspace_mb" in tensorrt:
                workspace_status = "success" if tensorrt.get("workspace_sufficient", False) else "warning"
                html += f"""
                <tr>
                    <td>Workspace Size</td>
                    <td class="{workspace_status}">{tensorrt.get("workspace_mb", 0):.0f} MB</td>
                    <td>{"Sufficient for T4 optimization" if tensorrt.get("workspace_sufficient", False) else "Increase to at least 1024 MB for better performance"}</td>
                </tr>
"""

            # Optimization level
            if "optimization_level" in tensorrt:
                opt_level_status = "success" if tensorrt.get("optimal_optimization_level", False) else "warning"
                html += f"""
                <tr>
                    <td>Optimization Level</td>
                    <td class="{opt_level_status}">{tensorrt.get("optimization_level", 0)}</td>
                    <td>{"Optimal for T4 (Level 3 or higher)" if tensorrt.get("optimal_optimization_level", False) else "Increase to level 3 for better T4 performance"}</td>
                </tr>
"""

            # Performance metrics with enhanced visual graphs
            if "fp32_time_ms" in tensorrt and "fp16_time_ms" in tensorrt:
                fp32_time = tensorrt.get("fp32_time_ms", 0)
                fp16_time = tensorrt.get("fp16_time_ms", 0)
                speedup = tensorrt.get("speedup", 0)
                
                # Determine max time for scaling bars
                max_time = max(fp32_time, fp16_time)
                # Calculate widths for bars (as percentage)
                fp32_width = 100
                fp16_width = (fp16_time / max_time) * 100 if max_time > 0 else 0
                
                # Determine color based on speedup
                if speedup >= 1.8:
                    speedup_color = "#28a745"  # Green - excellent
                    speedup_status = "success"
                elif speedup >= 1.3:
                    speedup_color = "#ffc107"  # Yellow - acceptable
                    speedup_status = "warning"
                else:
                    speedup_color = "#dc3545"  # Red - poor
                    speedup_status = "failure"
                
                # Calculate position for the target marker (1.8x speedup)
                target_position = (100 / speedup) * 1.8 if speedup > 0 else 100
                target_visible = target_position <= 100  # Only show if it's within the scale
                
                html += f"""
                <tr>
                    <td colspan="3">
                        <h3>TensorRT Performance Comparison</h3>
                        <div style="margin-top: 15px;">
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>FP32 Inference (PyTorch)</span>
                                    <span>{fp32_time:.2f} ms</span>
                                </div>
                                <div class="perf-container">
                                    <div class="perf-bar" style="width: {fp32_width}%; background-color: #888;"></div>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>FP16 Inference (TensorRT)</span>
                                    <span>{fp16_time:.2f} ms</span>
                                </div>
                                <div class="perf-container">
                                    <div class="perf-bar" style="width: {fp16_width}%; background-color: #76b900;"></div>
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px; margin-bottom: 5px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span><strong>Speedup Factor</strong></span>
                                    <span class="{speedup_status}"><strong>{speedup:.2f}x</strong></span>
                                </div>
                                <div class="speedup-bar">
                                    <div class="speedup-marker" style="width: 100%;">
                                        <div class="speedup-value" style="width: {min(100, speedup/2.5*100)}%; background-color: {speedup_color};">
                                            {speedup:.2f}x
                                        </div>
                                        {f'<div class="speedup-target" style="left: {target_position}%;"></div>' if target_visible else ''}
                                    </div>
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    {"✅ Excellent - Tensor Cores are likely active" if speedup >= 1.8 else 
                                    "⚠️ Below expected for T4 GPU - Tensor Cores may not be fully utilized"}
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
"""

            # Environment variables
            if "environment_variables" in tensorrt:
                env_vars = tensorrt["environment_variables"]
                html += f"""
                <tr>
                    <td colspan="3"><strong>Environment Variables</strong></td>
                </tr>
"""
                
                for var_name, var_value in env_vars.items():
                    # Determine if this is an optimal setting
                    var_status = "info"
                    var_note = ""
                    
                    if var_name == "TENSORRT_PRECISION" and var_value == "fp16":
                        var_status = "success"
                        var_note = "Optimal setting for T4"
                    elif var_name == "TENSORRT_PRECISION" and var_value != "fp16" and var_value != "Not set":
                        var_status = "warning"
                        var_note = "Change to 'fp16' for T4 optimization"
                    elif var_name == "TENSORRT_MAX_WORKSPACE_SIZE" and var_value != "Not set":
                        try:
                            workspace_mb = int(var_value) / (1024*1024)
                            if workspace_mb >= 1024:
                                var_status = "success"
                                var_note = "Sufficient workspace size"
                            else:
                                var_status = "warning"
                                var_note = "Increase to at least 1024 MB"
                        except:
                            pass
                    elif var_name == "TENSORRT_BUILDER_OPTIMIZATION_LEVEL" and var_value != "Not set":
                        try:
                            opt_level = int(var_value)
                            if opt_level >= 3:
                                var_status = "success"
                                var_note = "Optimal level for T4"
                            else:
                                var_status = "warning"
                                var_note = "Increase to level 3"
                        except:
                            pass
                    
                    html += f"""
                <tr>
                    <td>{var_name}</td>
                    <td class="{var_status}">{var_value}</td>
                    <td>{var_note}</td>
                </tr>
"""
        
        html += """
            </table>
            
            <div class="optimization-tips">
                <h3>Optimization Tips</h3>
                <ul>
"""

        # Add optimization tips based on test results
        if not tensorrt.get("available", False):
            html += """
                    <li>Install TensorRT for significant performance improvement on T4 GPUs</li>
"""
        else:
            if not tensorrt.get("fp16_support", False):
                html += """
                    <li>Ensure GPU driver supports FP16 operations for Tensor Cores</li>
"""
            
            if tensorrt.get("workspace_mb", 0) < 1024:
                html += f"""
                    <li>Increase TensorRT workspace size from {tensorrt.get("workspace_mb", 0):.0f} MB to at least 1024 MB</li>
"""
            
            if tensorrt.get("optimization_level", 0) < 3:
                html += f"""
                    <li>Set TENSORRT_BUILDER_OPTIMIZATION_LEVEL=3 for better performance</li>
"""
            
            if "speedup" in tensorrt and tensorrt.get("speedup", 0) < 1.8:
                html += f"""
                    <li>Current TensorRT speedup ({tensorrt.get("speedup", 0):.2f}x) is below expected for T4 - ensure Tensor Cores are active</li>
"""
            
            if "environment_variables" in tensorrt:
                env_vars = tensorrt["environment_variables"]
                if env_vars.get("TENSORRT_PRECISION", "Not set") != "fp16":
                    html += """
                    <li>Set TENSORRT_PRECISION=fp16 in environment variables</li>
"""
                
                if env_vars.get("TENSORRT_ENGINE_CACHE_ENABLE", "Not set") == "Not set":
                    html += """
                    <li>Enable TensorRT engine caching with TENSORRT_ENGINE_CACHE_ENABLE=1</li>
"""
        
        html += """
                </ul>
            </div>
        </div>
"""
    
    # Add API section with authentication status
    if "api" in results and results["api"]:
        api = results["api"]
        api_status = "success" if api.get("port_open", False) else "failure"
        auth_method = api.get("auth_method", "none")
        
        # Determine authentication status and message
        auth_required = api.get("auth_required", False)
        auth_error = "auth_error" in api
        
        if auth_method != "none":
            auth_status = "success"
            auth_message = f"Using {auth_method.capitalize()} Authentication"
        elif auth_required and auth_error:
            auth_status = "failure"
            auth_message = "Authentication required but not provided"
        elif auth_required:
            auth_status = "warning"
            auth_message = "Authentication required"
        else:
            auth_status = "info"
            auth_message = "No authentication required"
        
        html += f"""
        <div class="section">
            <h2>API Endpoints</h2>
            <div class="summary">
                <div class="summary-box">
                    <h2>API Status</h2>
                    <div class="{api_status}">{"✅ Running" if api.get("port_open", False) else "❌ Not Available"}</div>
                </div>
                <div class="summary-box">
                    <h2>Authentication</h2>
                    <div class="{auth_status}">{auth_message}</div>
                </div>
            </div>
            
            <p>Host: {api.get("host", "Unknown")}:{api.get("port", "Unknown")}</p>
            
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td>API Service</td>
                    <td class="{api_status}">{"✅ Running" if api.get("port_open", False) else "❌ Not Available"}</td>
                    <td>Port {api.get("port", "Unknown")} {"open" if api.get("port_open", False) else "closed"}</td>
                </tr>
"""
        
        if auth_method != "none":
            html += f"""
                <tr>
                    <td>Authentication</td>
                    <td class="success">✅ Configured</td>
                    <td>{auth_method.capitalize()} authentication</td>
                </tr>
"""
        elif auth_required:
            html += f"""
                <tr>
                    <td>Authentication</td>
                    <td class="{'failure' if auth_error else 'warning'}">{'❌ Missing' if auth_error else '⚠️ Required'}</td>
                    <td>{'Credentials required but not provided' if auth_error else 'API requires authentication'}</td>
                </tr>
"""
            
            if "auth_header" in api:
                html += f"""
                <tr>
                    <td>Auth Requirements</td>
                    <td colspan="2"><code>{api.get("auth_header", "")}</code></td>
                </tr>
"""
        
        if api.get("port_open", False) and "endpoints" in api:
            html += f"""
                <tr>
                    <td colspan="3"><strong>Standard Endpoints</strong></td>
                </tr>
"""
            
            # Group endpoints by authentication
            standard_endpoints = []
            auth_endpoints = []
            
            for name, endpoint in api["endpoints"].items():
                if endpoint.get("authenticated", False):
                    auth_endpoints.append((name, endpoint))
                else:
                    standard_endpoints.append((name, endpoint))
            
            # Standard endpoints
            for name, endpoint in standard_endpoints:
                if endpoint.get("auth_failed", False):
                    endpoint_status = "failure"
                    status_text = "❌ Auth Failed (401)"
                elif endpoint.get("permission_denied", False):
                    endpoint_status = "failure"
                    status_text = "❌ Forbidden (403)"
                else:
                    endpoint_status = "success" if endpoint.get("success", False) else "failure"
                    status_text = "✅ OK" if endpoint.get("success", False) else f"❌ Failed ({endpoint.get('status_code', 'N/A')})"
                
                html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{endpoint_status}">{status_text}</td>
                    <td>{endpoint.get("path", "Unknown")}</td>
                </tr>
"""
                
                # Add GPU info if available
                if name == "GPU Info" and endpoint.get("success", False) and "data" in endpoint:
                    gpu_info = endpoint["data"]
                    if isinstance(gpu_info, dict):
                        t4_detected = endpoint.get("is_t4", False)
                        t4_status = "success" if t4_detected else "failure"
                        
                        html += f"""
                <tr>
                    <td colspan="2">T4 GPU Detected in API</td>
                    <td class="{t4_status}">{"✅ Yes" if t4_detected else "❌ No"}</td>
                </tr>
"""
            
            # Authenticated endpoints section if any
            if auth_endpoints:
                html += f"""
                <tr>
                    <td colspan="3"><strong>Authenticated Endpoints</strong></td>
                </tr>
"""
                
                for name, endpoint in auth_endpoints:
                    if endpoint.get("auth_failed", False):
                        endpoint_status = "failure"
                        status_text = "❌ Auth Failed (401)"
                    elif endpoint.get("permission_denied", False):
                        endpoint_status = "failure"
                        status_text = "❌ Forbidden (403)"
                    else:
                        endpoint_status = "success" if endpoint.get("success", False) else "failure"
                        status_text = "✅ OK" if endpoint.get("success", False) else f"❌ Failed ({endpoint.get('status_code', 'N/A')})"
                    
                    html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{endpoint_status}">{status_text}</td>
                    <td>{endpoint.get("path", "Unknown")}</td>
                </tr>
"""
        
        html += """
            </table>
            
            <div class="optimization-tips">
                <h3>API Testing Tips</h3>
                <ul>
"""
        
        # Add API-specific tips based on test results
        if not api.get("port_open", False):
            html += """
                    <li>Make sure the API service is running and the port is open on the specified host</li>
                    <li>Check if a firewall is blocking access to the API port</li>
"""
        elif auth_required and auth_error:
            html += """
                    <li>Provide authentication credentials to access protected API endpoints</li>
                    <li>Check the API documentation for authentication requirements</li>
"""
        
        if api.get("port_open", False) and "endpoints" in api:
            failed_endpoints = [name for name, endpoint in api["endpoints"].items() if not endpoint.get("success", False)]
            if failed_endpoints:
                html += f"""
                    <li>Some endpoints failed: {', '.join(failed_endpoints)}. Check the API configuration and logs.</li>
"""
        
        html += """
                </ul>
            </div>
        </div>
"""
    
    # Add Monitoring section
    if "monitoring" in results and results["monitoring"]:
        monitoring = results["monitoring"]
        
        html += """
        <div class="section">
            <h2>Monitoring</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
"""
        
        # Prometheus
        prometheus = monitoring.get("prometheus", {})
        prometheus_status = "success" if prometheus.get("port_open", False) and prometheus.get("healthy", False) else "failure"
        
        html += f"""
                <tr>
                    <td>Prometheus</td>
                    <td class="{prometheus_status}">{"✅ Running" if prometheus.get("port_open", False) and prometheus.get("healthy", False) else "❌ Not Available"}</td>
                    <td>
                        {"Port open: " + ("Yes" if prometheus.get("port_open", False) else "No")}<br>
                        {"Health check: " + ("Passed" if prometheus.get("healthy", False) else "Failed")}
                    </td>
                </tr>
"""
        
        if prometheus.get("port_open", False) and prometheus.get("healthy", False):
            gpu_metrics_status = "success" if prometheus.get("gpu_metrics", False) else "failure"
            
            html += f"""
                <tr>
                    <td>GPU Metrics in Prometheus</td>
                    <td class="{gpu_metrics_status}">{"✅ Available" if prometheus.get("gpu_metrics", False) else "❌ Not Found"}</td>
                    <td>nvidia_gpu_duty_cycle metric {"found" if prometheus.get("gpu_metrics", False) else "not found"}</td>
                </tr>
"""
        
        # Grafana
        grafana = monitoring.get("grafana", {})
        grafana_status = "success" if grafana.get("port_open", False) and grafana.get("healthy", False) else "failure"
        
        html += f"""
                <tr>
                    <td>Grafana</td>
                    <td class="{grafana_status}">{"✅ Running" if grafana.get("port_open", False) and grafana.get("healthy", False) else "❌ Not Available"}</td>
                    <td>
                        {"Port open: " + ("Yes" if grafana.get("port_open", False) else "No")}<br>
                        {"Health check: " + ("Passed" if grafana.get("healthy", False) else "Failed")}
                    </td>
                </tr>
"""
        
        # GPU Exporter
        gpu_exporter = monitoring.get("gpu_exporter", {})
        gpu_exporter_status = "success" if gpu_exporter.get("port_open", False) and gpu_exporter.get("working", False) else "failure"
        
        html += f"""
                <tr>
                    <td>GPU Metrics Exporter</td>
                    <td class="{gpu_exporter_status}">{"✅ Running" if gpu_exporter.get("port_open", False) and gpu_exporter.get("working", False) else "❌ Not Available"}</td>
                    <td>
                        {"Port open: " + ("Yes" if gpu_exporter.get("port_open", False) else "No")}<br>
                        {"Metrics: " + ("Available" if gpu_exporter.get("working", False) else "Not Available")}
                    </td>
                </tr>
"""
        
        if gpu_exporter.get("port_open", False) and gpu_exporter.get("working", False) and gpu_exporter.get("nvidia_metrics", False):
            # Get GPU metrics with defaults
            utilization = gpu_exporter.get("utilization", 0)
            memory_used_mb = gpu_exporter.get("memory_used_mb", 0)
            temperature = gpu_exporter.get("temperature", 0)
            power_watts = gpu_exporter.get("power_watts", 0)
            
            # Determine status colors for each metric
            # Utilization
            if utilization > 80:
                util_color = "#28a745"  # Green - excellent utilization
            elif utilization > 30:
                util_color = "#ffc107"  # Yellow - medium utilization
            else:
                util_color = "#17a2b8"  # Blue - low utilization
            
            # Temperature
            if temperature > 80:
                temp_color = "#dc3545"  # Red - high temperature
            elif temperature > 65:
                temp_color = "#ffc107"  # Yellow - medium temperature
            else:
                temp_color = "#28a745"  # Green - good temperature
            
            # Power
            if power_watts > 60:
                power_color = "#ffc107"  # Yellow - high power
            else:
                power_color = "#28a745"  # Green - good power usage
            
            html += f"""
                <tr>
                    <td colspan="3">
                        <h3>NVIDIA T4 GPU Metrics</h3>
                        <div class="stats-container">
                            <div class="stat-circle gpu-utilization">
                                <div class="stat-title">GPU Utilization</div>
                                <div class="stat-value">{utilization}</div>
                                <div class="stat-unit">%</div>
                            </div>
                            
                            <div class="stat-circle gpu-memory">
                                <div class="stat-title">Memory Used</div>
                                <div class="stat-value">{memory_used_mb:.0f}</div>
                                <div class="stat-unit">MB</div>
                            </div>
                            
                            <div class="stat-circle gpu-temp">
                                <div class="stat-title">Temperature</div>
                                <div class="stat-value">{temperature:.1f}</div>
                                <div class="stat-unit">°C</div>
                            </div>
                            
                            <div class="stat-circle gpu-power">
                                <div class="stat-title">Power Usage</div>
                                <div class="stat-value">{power_watts:.1f}</div>
                                <div class="stat-unit">Watts</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 20px;">
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>GPU Utilization</span>
                                    <span style="color: {util_color};">{utilization}%</span>
                                </div>
                                <div class="perf-container">
                                    <div class="perf-bar" style="width: {utilization}%; background-color: {util_color};"></div>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Temperature</span>
                                    <span style="color: {temp_color};">{temperature:.1f}°C</span>
                                </div>
                                <div class="perf-container">
                                    <div class="perf-bar" style="width: {min(100, temperature/100*100)}%; background-color: {temp_color};"></div>
                                </div>
                            </div>
                            
                            <div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Power Draw</span>
                                    <span style="color: {power_color};">{power_watts:.1f} Watts</span>
                                </div>
                                <div class="perf-container">
                                    <div class="perf-bar" style="width: {min(100, power_watts/70*100)}%; background-color: {power_color};"></div>
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    T4 has a maximum power draw of 70W
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
    
    # Add Recommendations section
    html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
"""
    
    # Add recommendations based on test results
    recommendations = []
    
    if not is_t4:
        recommendations.append("Verify this is a T4 GPU instance - no T4 GPU was detected in tests")
    
    if "pytorch" in results and results["pytorch"]:
        pytorch = results["pytorch"]
        if not pytorch.get("cuda_available", False):
            recommendations.append("Install CUDA and PyTorch with CUDA support")
        elif pytorch.get("speedup", 0) < 1.8:
            recommendations.append("Optimize for Tensor Cores - FP16 speedup is lower than expected for T4")
    
    if "tensorrt" in results and results["tensorrt"]:
        tensorrt = results["tensorrt"]
        if not tensorrt.get("available", False):
            recommendations.append("Install TensorRT for improved inference performance")
        elif not tensorrt.get("fp16_support", False):
            recommendations.append("Configure TensorRT with FP16 support for T4 Tensor Cores")
    
    if "api" in results and results["api"]:
        api = results["api"]
        if not api.get("port_open", False):
            recommendations.append("Ensure API service is running and port is accessible")
    
    if "monitoring" in results and results["monitoring"]:
        monitoring = results["monitoring"]
        prometheus = monitoring.get("prometheus", {})
        if not prometheus.get("port_open", False) or not prometheus.get("healthy", False):
            recommendations.append("Configure Prometheus monitoring service")
        elif not prometheus.get("gpu_metrics", False):
            recommendations.append("Set up Prometheus to collect GPU metrics")
        
        grafana = monitoring.get("grafana", {})
        if not grafana.get("port_open", False) or not grafana.get("healthy", False):
            recommendations.append("Configure Grafana dashboards for monitoring")
        
        gpu_exporter = monitoring.get("gpu_exporter", {})
        if not gpu_exporter.get("port_open", False) or not gpu_exporter.get("working", False):
            recommendations.append("Set up NVIDIA GPU metrics exporter")
    
    if not recommendations:
        recommendations.append("All tests passed! No recommendations needed.")
    
    for recommendation in recommendations:
        html += f"                <li>{recommendation}</li>\n"
    
    html += """
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(report_file, "w") as f:
        f.write(html)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CI/CD Automated Testing for NVIDIA T4 Deployments"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Hostname or IP of the NVIDIA instance (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API service (default: 8000)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to store test results (default: {OUTPUT_DIR})"
    )
    
    # Authentication options
    auth_group = parser.add_argument_group('Authentication Options')
    auth_group.add_argument(
        "--auth-token",
        help="Bearer token for API authentication"
    )
    
    auth_group.add_argument(
        "--username",
        help="Username for basic authentication"
    )
    
    auth_group.add_argument(
        "--password",
        help="Password for basic authentication"
    )
    
    # Test control options
    test_group = parser.add_argument_group('Test Control Options')
    test_group.add_argument(
        "--no-api-test",
        action="store_true",
        help="Skip API testing"
    )
    
    test_group.add_argument(
        "--no-monitoring-test",
        action="store_true",
        help="Skip monitoring testing"
    )
    
    test_group.add_argument(
        "--no-cuda-test",
        action="store_true",
        help="Skip CUDA/PyTorch testing"
    )
    
    test_group.add_argument(
        "--no-tensorrt-test",
        action="store_true",
        help="Skip TensorRT testing"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--open-report",
        action="store_true",
        help="Automatically open HTML report in browser after completion"
    )
    
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Update output directory if specified
    global OUTPUT_DIR, RESULTS_FILE, REPORT_FILE
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        RESULTS_FILE = os.path.join(OUTPUT_DIR, f"{TEST_ID}.json")
        REPORT_FILE = os.path.join(OUTPUT_DIR, f"{TEST_ID}.html")
    
    print(f"====== NVIDIA T4 CI Test ======")
    print(f"Host: {args.host}")
    print(f"API Port: {args.port}")
    print(f"Output: {REPORT_FILE}")
    if args.auth_token:
        print(f"Authentication: Bearer Token")
    elif args.username and args.password:
        print(f"Authentication: Basic ({args.username})")
    print(f"===============================")
    
    # Run tests
    results["gpu"] = test_nvidia_smi()
    
    if not args.no_cuda_test:
        results["pytorch"] = test_cuda_pytorch()
    
    if not args.no_tensorrt_test:
        results["tensorrt"] = test_tensorrt()
    
    if not args.no_api_test:
        results["api"] = test_api(
            host=args.host, 
            port=args.port,
            auth_token=args.auth_token,
            username=args.username,
            password=args.password
        )
    
    if not args.no_monitoring_test:
        results["monitoring"] = test_monitoring(args.host)
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Generate HTML report
    generate_html_report(results, REPORT_FILE)
    print(f"HTML report saved to: {REPORT_FILE}")
    
    # Open report in browser if requested
    if args.open_report:
        try:
            import webbrowser
            print(f"Opening HTML report in browser...")
            webbrowser.open(f"file://{os.path.abspath(REPORT_FILE)}")
        except Exception as e:
            print(f"Failed to open browser: {e}")
    
    # Determine exit code
    t4_detected = False
    gpu_detected = False
    
    # Check GPU detection
    if "gpu" in results and results["gpu"]:
        gpu_detected = results["gpu"].get("detected", False)
        t4_detected = t4_detected or results["gpu"].get("is_t4", False)
    
    # Check PyTorch detection
    if "pytorch" in results and results["pytorch"]:
        gpu_detected = gpu_detected or results["pytorch"].get("cuda_available", False)
        t4_detected = t4_detected or results["pytorch"].get("is_t4", False)
    
    # Check API detection
    if "api" in results and results["api"] and "endpoints" in results["api"]:
        for name, endpoint in results["api"]["endpoints"].items():
            if name == "GPU Info" and endpoint.get("is_t4", False):
                t4_detected = True
                gpu_detected = True
    
    if t4_detected:
        print("\n✅ NVIDIA T4 GPU detected and validated")
        return 0
    elif gpu_detected:
        print("\n⚠️ GPU detected but not confirmed as T4")
        return 1
    else:
        print("\n❌ No GPU detected or tests failed")
        return 2

if __name__ == "__main__":
    sys.exit(main())