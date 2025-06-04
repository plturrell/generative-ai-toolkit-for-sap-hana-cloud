#!/usr/bin/env python3
"""
Lightweight Remote Testing Script for NVIDIA T4 Deployments.

This script is designed to be run directly on the server or through
an authenticated session. It performs basic tests to validate the
T4 GPU deployment without requiring direct SSH access.

Usage:
    python remote_test.py
"""
import os
import sys
import json
import time
import datetime
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Output directory
OUTPUT_DIR = "t4-test-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test timestamp
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TEST_ID = f"remote-test-{TIMESTAMP}"

# Output files
RESULTS_JSON = os.path.join(OUTPUT_DIR, f"{TEST_ID}.json")
RESULTS_HTML = os.path.join(OUTPUT_DIR, f"{TEST_ID}.html")

# Results storage
results = {
    "timestamp": TIMESTAMP,
    "test_id": TEST_ID,
    "gpu": {},
    "system": {},
    "api": {},
    "performance": {}
}

print(f"Starting NVIDIA T4 Remote Test {TEST_ID}")
print("-" * 60)

def run_command(command: str) -> Tuple[int, str, str]:
    """Run a shell command and capture output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def test_gpu_detection():
    """Test GPU detection and properties."""
    print("\nTesting GPU Detection:")
    returncode, stdout, stderr = run_command("nvidia-smi")
    
    if returncode != 0:
        print("❌ NVIDIA GPU not detected")
        results["gpu"]["detected"] = False
        results["gpu"]["error"] = stderr
        return
    
    print("✅ NVIDIA GPU detected")
    results["gpu"]["detected"] = True
    results["gpu"]["raw_info"] = stdout
    
    # Check for T4 in output
    if "T4" in stdout:
        print("✅ NVIDIA T4 GPU confirmed")
        results["gpu"]["is_t4"] = True
    else:
        print("❌ Not a T4 GPU")
        results["gpu"]["is_t4"] = False
    
    # Get detailed GPU info
    returncode, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader"
    )
    
    if returncode == 0:
        parts = [p.strip() for p in stdout.strip().split(',')]
        
        if len(parts) >= 8:
            results["gpu"]["model"] = parts[0]
            results["gpu"]["memory_total"] = parts[1]
            results["gpu"]["driver_version"] = parts[2]
            results["gpu"]["compute_capability"] = parts[3]
            results["gpu"]["utilization"] = parts[4]
            results["gpu"]["memory_used"] = parts[5]
            results["gpu"]["temperature"] = parts[6]
            results["gpu"]["power_draw"] = parts[7]
            
            print(f"  Model: {parts[0]}")
            print(f"  Memory: {parts[1]}")
            print(f"  Driver: {parts[2]}")
            print(f"  Compute Capability: {parts[3]}")
            print(f"  Utilization: {parts[4]}")
            print(f"  Memory Used: {parts[5]}")
            print(f"  Temperature: {parts[6]}")
            print(f"  Power Draw: {parts[7]}")

def test_cuda_pytorch():
    """Test CUDA and PyTorch installation."""
    print("\nTesting CUDA and PyTorch:")
    
    # Check CUDA version
    returncode, stdout, stderr = run_command("nvcc --version")
    
    if returncode == 0:
        import re
        cuda_match = re.search(r"release (\d+\.\d+)", stdout)
        if cuda_match:
            cuda_version = cuda_match.group(1)
            results["system"]["cuda_version"] = cuda_version
            print(f"✅ CUDA version: {cuda_version}")
        else:
            results["system"]["cuda_version"] = "unknown"
            print("❌ Could not determine CUDA version")
    else:
        results["system"]["cuda_installed"] = False
        print("❌ CUDA not installed or nvcc not in PATH")
    
    # Check PyTorch with CUDA
    returncode, stdout, stderr = run_command(
        "python3 -c \"import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
    )
    
    if returncode == 0:
        print(stdout)
        
        # Parse output
        results["system"]["pytorch_installed"] = True
        
        if "CUDA available: True" in stdout:
            results["system"]["pytorch_cuda"] = True
            print("✅ PyTorch with CUDA support")
        else:
            results["system"]["pytorch_cuda"] = False
            print("❌ PyTorch without CUDA support")
    else:
        results["system"]["pytorch_installed"] = False
        print("❌ PyTorch not installed or error importing")

def test_tensorrt():
    """Test TensorRT installation."""
    print("\nTesting TensorRT:")
    
    returncode, stdout, stderr = run_command(
        "python3 -c \"import tensorrt; print('TensorRT version:', tensorrt.__version__)\""
    )
    
    if returncode == 0:
        print(stdout)
        results["system"]["tensorrt_installed"] = True
        
        # Extract version
        import re
        version_match = re.search(r"TensorRT version: (\d+\.\d+\.\d+)", stdout)
        if version_match:
            results["system"]["tensorrt_version"] = version_match.group(1)
    else:
        results["system"]["tensorrt_installed"] = False
        print("❌ TensorRT not installed or error importing")

def test_api_endpoints():
    """Test API endpoints."""
    print("\nTesting API Endpoints:")
    
    # Test endpoints to check
    endpoints = [
        {"url": "http://localhost:8000/", "name": "Root"},
        {"url": "http://localhost:8000/health", "name": "Health"},
        {"url": "http://localhost:8000/api/v1/system/info", "name": "System Info"},
        {"url": "http://localhost:8000/api/v1/system/gpu", "name": "GPU Info"}
    ]
    
    results["api"]["endpoints"] = {}
    
    for endpoint in endpoints:
        name = endpoint["name"]
        url = endpoint["url"]
        print(f"Testing {name} endpoint: {url}")
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=5)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                print(f"✅ {name}: Status 200 OK ({latency_ms:.1f}ms)")
                results["api"]["endpoints"][name] = {
                    "status": "ok",
                    "status_code": response.status_code,
                    "latency_ms": latency_ms
                }
                
                # For certain endpoints, capture the response data
                if name == "GPU Info":
                    try:
                        data = response.json()
                        results["api"]["gpu_info"] = data
                    except:
                        pass
            else:
                print(f"❌ {name}: Status {response.status_code}")
                results["api"]["endpoints"][name] = {
                    "status": "error",
                    "status_code": response.status_code,
                    "latency_ms": latency_ms
                }
        except requests.RequestException as e:
            print(f"❌ {name}: Connection error: {str(e)}")
            results["api"]["endpoints"][name] = {
                "status": "error",
                "error": str(e)
            }

def test_gpu_performance():
    """Test GPU performance with PyTorch."""
    print("\nTesting GPU Performance:")
    
    test_script = """
import torch
import time

# Check if CUDA is available
if not torch.cuda.is_available():
    print("❌ CUDA not available")
    exit(1)

print(f"✅ Testing on device: {torch.cuda.get_device_name(0)}")

# Function to measure matrix multiplication performance
def benchmark_matmul(size, dtype, iterations=10):
    # Create random matrices
    a = torch.randn(size, size, dtype=dtype, device="cuda")
    b = torch.randn(size, size, dtype=dtype, device="cuda")
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

# Test matrix multiplication with different precisions
print("Matrix multiplication benchmark (2048x2048):")

# FP32
fp32_time = benchmark_matmul(2048, torch.float32)
print(f"  FP32: {fp32_time:.6f} seconds")

# FP16 (Half precision for Tensor Cores)
fp16_time = benchmark_matmul(2048, torch.float16)
print(f"  FP16: {fp16_time:.6f} seconds")

# Calculate speedup
speedup = fp32_time / fp16_time
print(f"  Speedup (FP32 → FP16): {speedup:.2f}x")

# Test memory allocation
print("\\nMemory test:")
torch.cuda.reset_peak_memory_stats()
try:
    # Try to allocate a large tensor (8GB)
    large_tensor = torch.zeros(1024 * 8, 1024 * 8, dtype=torch.float32, device="cuda")
    print("  ✅ Successfully allocated 8GB tensor")
    del large_tensor
except Exception as e:
    print(f"  ❌ Failed to allocate large tensor: {e}")

# Report peak memory
peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
print(f"  Peak memory usage: {peak_memory:.2f} GB")

results = {
    "fp32_time": fp32_time,
    "fp16_time": fp16_time, 
    "speedup": speedup,
    "peak_memory_gb": peak_memory
}

print(f"\\nRESULTS_JSON: {results}")
"""
    
    # Save test script
    test_script_path = "gpu_benchmark.py"
    with open(test_script_path, "w") as f:
        f.write(test_script)
    
    # Run test script
    returncode, stdout, stderr = run_command(f"python3 {test_script_path}")
    
    if returncode == 0:
        print(stdout)
        
        # Extract results
        import re
        results_match = re.search(r"RESULTS_JSON: (\{.*\})", stdout)
        if results_match:
            try:
                perf_results = eval(results_match.group(1))
                results["performance"] = perf_results
                
                # Check for good speedup
                if perf_results.get("speedup", 0) > 1.8:
                    print("✅ Good FP16 speedup - Tensor Cores working well")
                else:
                    print("❌ Poor FP16 speedup - Tensor Cores may not be optimized")
            except:
                print("❌ Failed to parse performance results")
    else:
        print("❌ GPU performance test failed")
        print(stderr)
    
    # Clean up
    try:
        os.remove(test_script_path)
    except:
        pass

def test_docker_status():
    """Test Docker container status."""
    print("\nTesting Docker Status:")
    
    returncode, stdout, stderr = run_command("docker ps")
    
    if returncode == 0:
        print("✅ Docker is running")
        results["system"]["docker_running"] = True
        results["system"]["docker_containers"] = stdout
        
        # Count containers
        lines = stdout.strip().split('\n')
        container_count = len(lines) - 1 if lines else 0
        print(f"  {container_count} containers running")
        
        # Look for key containers
        if "api" in stdout or "hana-ai" in stdout:
            print("✅ API container detected")
        
        if "prometheus" in stdout:
            print("✅ Prometheus container detected")
        
        if "grafana" in stdout:
            print("✅ Grafana container detected")
    else:
        print("❌ Docker command failed")
        results["system"]["docker_running"] = False
        results["system"]["docker_error"] = stderr

def generate_html_report():
    """Generate HTML report from results."""
    print("\nGenerating HTML Report...")
    
    # Basic template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Remote Test Report</title>
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
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            flex-basis: 23%;
            margin-bottom: 15px;
            text-align: center;
        }}
        .summary-box h2 {{
            margin-top: 0;
            font-size: 16px;
        }}
        .summary-box .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .section {{
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .section h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        table th, table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        table th {{
            background-color: #eee;
        }}
        .success {{
            color: #28a745;
        }}
        .warning {{
            color: #ffc107;
        }}
        .error {{
            color: #dc3545;
        }}
        pre {{
            background-color: #eee;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Remote Test Report</h1>
            <p>Test ID: {TEST_ID} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
    
    # GPU Section
    html += """
        <div class="section">
            <h2>GPU Information</h2>
"""
    
    if results["gpu"].get("detected", False):
        gpu_model = results["gpu"].get("model", "Unknown")
        memory = results["gpu"].get("memory_total", "Unknown")
        driver = results["gpu"].get("driver_version", "Unknown")
        compute = results["gpu"].get("compute_capability", "Unknown")
        
        html += f"""
            <div class="summary">
                <div class="summary-box">
                    <h2>Model</h2>
                    <div class="value">{gpu_model}</div>
                </div>
                <div class="summary-box">
                    <h2>Memory</h2>
                    <div class="value">{memory}</div>
                </div>
                <div class="summary-box">
                    <h2>Driver</h2>
                    <div class="value">{driver}</div>
                </div>
                <div class="summary-box">
                    <h2>Compute</h2>
                    <div class="value">{compute}</div>
                </div>
            </div>
            
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>GPU Type</td>
                    <td class="{'success' if results['gpu'].get('is_t4', False) else 'error'}">{gpu_model} {'(T4 Confirmed)' if results['gpu'].get('is_t4', False) else '(Not T4)'}</td>
                </tr>
                <tr>
                    <td>Utilization</td>
                    <td>{results['gpu'].get('utilization', 'Unknown')}</td>
                </tr>
                <tr>
                    <td>Memory Used</td>
                    <td>{results['gpu'].get('memory_used', 'Unknown')}</td>
                </tr>
                <tr>
                    <td>Temperature</td>
                    <td>{results['gpu'].get('temperature', 'Unknown')}</td>
                </tr>
                <tr>
                    <td>Power Draw</td>
                    <td>{results['gpu'].get('power_draw', 'Unknown')}</td>
                </tr>
            </table>
"""
    else:
        html += """
            <p class="error">❌ No NVIDIA GPU detected</p>
"""
    
    html += """
        </div>
"""
    
    # System Section
    html += """
        <div class="section">
            <h2>System Configuration</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
"""
    
    # CUDA
    cuda_version = results["system"].get("cuda_version", "Not detected")
    cuda_status = "success" if cuda_version != "Not detected" else "error"
    html += f"""
                <tr>
                    <td>CUDA</td>
                    <td class="{cuda_status}">{cuda_version}</td>
                    <td>{cuda_version if cuda_version != "Not detected" else "Not installed or not in PATH"}</td>
                </tr>
"""
    
    # PyTorch
    pytorch_status = "success" if results["system"].get("pytorch_installed", False) else "error"
    pytorch_cuda = results["system"].get("pytorch_cuda", False)
    html += f"""
                <tr>
                    <td>PyTorch</td>
                    <td class="{pytorch_status}">{pytorch_status}</td>
                    <td>{"With CUDA support" if pytorch_cuda else "Without CUDA support"}</td>
                </tr>
"""
    
    # TensorRT
    tensorrt_status = "success" if results["system"].get("tensorrt_installed", False) else "error"
    tensorrt_version = results["system"].get("tensorrt_version", "Not detected")
    html += f"""
                <tr>
                    <td>TensorRT</td>
                    <td class="{tensorrt_status}">{tensorrt_status}</td>
                    <td>{f"Version: {tensorrt_version}" if tensorrt_status == "success" else "Not installed"}</td>
                </tr>
"""
    
    # Docker
    docker_status = "success" if results["system"].get("docker_running", False) else "error"
    html += f"""
                <tr>
                    <td>Docker</td>
                    <td class="{docker_status}">{docker_status}</td>
                    <td>{"Running" if docker_status == "success" else "Not running"}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
"""
    
    # API Section
    html += """
        <div class="section">
            <h2>API Status</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Latency</th>
                </tr>
"""
    
    endpoints = results["api"].get("endpoints", {})
    for name, data in endpoints.items():
        status = data.get("status", "unknown")
        status_class = "success" if status == "ok" else "error"
        latency = f"{data.get('latency_ms', 0):.1f} ms" if "latency_ms" in data else "N/A"
        
        html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{status_class}">{status.upper()}</td>
                    <td>{latency}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
"""
    
    # Performance Section
    html += """
        <div class="section">
            <h2>GPU Performance</h2>
"""
    
    if "performance" in results and results["performance"]:
        fp32_time = results["performance"].get("fp32_time", 0)
        fp16_time = results["performance"].get("fp16_time", 0)
        speedup = results["performance"].get("speedup", 0)
        memory = results["performance"].get("peak_memory_gb", 0)
        
        speedup_class = "success" if speedup > 1.8 else "warning" if speedup > 1.3 else "error"
        
        html += f"""
            <div class="summary">
                <div class="summary-box">
                    <h2>FP32 Time</h2>
                    <div class="value">{fp32_time:.6f}s</div>
                </div>
                <div class="summary-box">
                    <h2>FP16 Time</h2>
                    <div class="value">{fp16_time:.6f}s</div>
                </div>
                <div class="summary-box">
                    <h2>Speedup</h2>
                    <div class="value {speedup_class}">{speedup:.2f}x</div>
                </div>
                <div class="summary-box">
                    <h2>Peak Memory</h2>
                    <div class="value">{memory:.2f} GB</div>
                </div>
            </div>
            
            <div>
                <h3>Analysis</h3>
                <p>T4 GPUs with properly configured Tensor Cores should show at least 1.8x speedup for FP16 operations.</p>
                <p class="{speedup_class}">
                    {
                        "✅ Excellent performance: Tensor Cores are working well." if speedup > 2.0 else
                        "✅ Good performance: Tensor Cores are active." if speedup > 1.8 else
                        "⚠️ Mediocre performance: Tensor Cores may not be fully optimized." if speedup > 1.3 else
                        "❌ Poor performance: Tensor Cores may not be working properly."
                    }
                </p>
            </div>
"""
    else:
        html += """
            <p class="error">❌ Performance test did not complete successfully</p>
"""
    
    html += """
        </div>
"""
    
    # Recommendations Section
    html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
"""
    
    # Add recommendations based on test results
    recommendations = []
    
    # GPU detection
    if not results["gpu"].get("detected", False):
        recommendations.append("Install NVIDIA drivers and CUDA toolkit")
    elif not results["gpu"].get("is_t4", False):
        recommendations.append("Verify this is a T4 GPU instance")
    
    # CUDA/PyTorch
    if not results["system"].get("cuda_version", ""):
        recommendations.append("Install CUDA toolkit or add to PATH")
    
    if not results["system"].get("pytorch_cuda", False):
        recommendations.append("Install PyTorch with CUDA support")
    
    if not results["system"].get("tensorrt_installed", False):
        recommendations.append("Install TensorRT for improved inference performance")
    
    # Performance
    if "performance" in results:
        speedup = results["performance"].get("speedup", 0)
        if speedup < 1.8:
            recommendations.append("Optimize for Tensor Cores: ensure FP16 precision is properly configured")
    
    # API
    api_issues = False
    for name, data in results["api"].get("endpoints", {}).items():
        if data.get("status", "") != "ok":
            api_issues = True
            break
    
    if api_issues:
        recommendations.append("Check API service status and configuration")
    
    # Add recommendations to HTML
    if recommendations:
        for rec in recommendations:
            html += f"                <li>{rec}</li>\n"
    else:
        html += "                <li>✅ All tests passed! No recommendations needed.</li>\n"
    
    html += """
            </ul>
        </div>
"""
    
    # Raw JSON
    html += """
        <div class="section">
            <h2>Raw Test Data</h2>
            <pre>
"""
    html += json.dumps(results, indent=2)
    html += """
            </pre>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(RESULTS_HTML, "w") as f:
        f.write(html)
    
    print(f"HTML report saved to: {RESULTS_HTML}")

def main():
    """Main test function."""
    try:
        # Run tests
        test_gpu_detection()
        test_cuda_pytorch()
        test_tensorrt()
        test_docker_status()
        test_api_endpoints()
        test_gpu_performance()
        
        # Save results
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {RESULTS_JSON}")
        
        # Generate HTML report
        generate_html_report()
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("NVIDIA T4 Test Summary:")
        
        if results["gpu"].get("detected", False) and results["gpu"].get("is_t4", False):
            print("✅ T4 GPU is properly detected")
        else:
            print("❌ T4 GPU validation failed")
        
        if "performance" in results and results["performance"].get("speedup", 0) > 1.8:
            print("✅ Tensor Cores are working well")
        elif "performance" in results:
            print(f"⚠️ Tensor Cores performance: {results['performance'].get('speedup', 0):.2f}x speedup")
        
        api_success = True
        for name, data in results["api"].get("endpoints", {}).items():
            if data.get("status", "") != "ok":
                api_success = False
                break
        
        if api_success:
            print("✅ API endpoints are accessible")
        else:
            print("❌ Some API endpoints are not accessible")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())