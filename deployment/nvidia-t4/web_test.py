#!/usr/bin/env python3
"""
Web-based NVIDIA T4 Testing Script.

This script provides testing capabilities for a T4 GPU deployment
using browser automation. It can be run anywhere with web access
to the deployed environment.

Usage:
    python web_test.py [url]
"""
import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install with: pip install selenium")
    print("Also install Chrome WebDriver: https://chromedriver.chromium.org/downloads")
    sys.exit(1)

# Output directory
OUTPUT_DIR = "t4-test-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test timestamp
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TEST_ID = f"web-test-{TIMESTAMP}"

# Output files
RESULTS_JSON = os.path.join(OUTPUT_DIR, f"{TEST_ID}.json")
RESULTS_HTML = os.path.join(OUTPUT_DIR, f"{TEST_ID}.html")
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Results storage
results = {
    "timestamp": TIMESTAMP,
    "test_id": TEST_ID,
    "gpu": {},
    "system": {},
    "api": {},
    "performance": {},
    "screenshots": []
}

def setup_webdriver(headless: bool = True):
    """Set up Chrome WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        return driver
    except WebDriverException as e:
        print(f"Failed to initialize WebDriver: {e}")
        sys.exit(1)

def take_screenshot(driver, name):
    """Take a screenshot and save it."""
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"{name}-{TIMESTAMP}.png")
    driver.save_screenshot(screenshot_path)
    results["screenshots"].append(screenshot_path)
    print(f"Screenshot saved: {screenshot_path}")
    return screenshot_path

def test_jupyter_access(driver, url):
    """Test access to Jupyter environment."""
    print(f"\nTesting access to Jupyter environment: {url}")
    
    try:
        driver.get(url)
        time.sleep(3)
        
        # Take screenshot of the login page
        screenshot = take_screenshot(driver, "jupyter-login")
        
        # Check if we're at a login page
        if "Sign in" in driver.title or "Login" in driver.title or "Jupyter" in driver.title:
            print("✅ Jupyter login page accessible")
            results["jupyter"] = {
                "accessible": True,
                "title": driver.title,
                "screenshot": screenshot
            }
        else:
            print(f"❌ Not a Jupyter login page. Title: {driver.title}")
            results["jupyter"] = {
                "accessible": False,
                "title": driver.title,
                "screenshot": screenshot
            }
    except Exception as e:
        print(f"❌ Error accessing Jupyter: {e}")
        results["jupyter"] = {
            "accessible": False,
            "error": str(e)
        }

def test_gpu_notebook(driver, url):
    """Test GPU availability through a Jupyter notebook."""
    print("\nTesting GPU availability through Jupyter...")
    
    # First, check if we're logged in or need to login
    if "Sign in" in driver.title or "Login" in driver.title:
        print("Login required - this test requires authentication")
        results["gpu_notebook"] = {
            "tested": False,
            "message": "Login required for this test"
        }
        return
    
    try:
        # Navigate to the new notebook page
        driver.get(f"{url}/notebooks/gpu_test.ipynb")
        time.sleep(2)
        
        # Check if we need to create a new notebook
        if "404" in driver.title or "Not Found" in driver.title:
            print("Creating a new notebook...")
            driver.get(f"{url}/new/notebook")
            time.sleep(3)
            
            # Take screenshot of notebook creation
            take_screenshot(driver, "notebook-creation")
            
            # Find the code cell and enter GPU test code
            try:
                cell = driver.find_element(By.CSS_SELECTOR, ".CodeMirror-code")
                driver.execute_script("""
                    arguments[0].CodeMirror.setValue(`
import torch
import subprocess
import sys

print("Python version:", sys.version)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        
    # Run nvidia-smi
    print("\\nnvidia-smi output:")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        print(result.stdout)
    except:
        print("nvidia-smi command failed")
else:
    print("CUDA not available")
                    `);
                """, cell)
                
                # Execute the cell
                run_button = driver.find_element(By.CSS_SELECTOR, ".run_this_cell")
                run_button.click()
                
                # Wait for execution to complete
                time.sleep(10)
                
                # Take screenshot of results
                screenshot = take_screenshot(driver, "notebook-results")
                
                # Check output
                output = driver.find_element(By.CSS_SELECTOR, ".output_text pre").text
                
                if "CUDA available: True" in output:
                    print("✅ CUDA is available")
                    results["gpu_notebook"] = {
                        "tested": True,
                        "cuda_available": True,
                        "screenshot": screenshot,
                        "output": output
                    }
                    
                    # Check for T4 GPU
                    if "T4" in output:
                        print("✅ NVIDIA T4 GPU detected")
                        results["gpu_notebook"]["is_t4"] = True
                    else:
                        print("❌ NVIDIA T4 GPU not detected")
                        results["gpu_notebook"]["is_t4"] = False
                else:
                    print("❌ CUDA is not available")
                    results["gpu_notebook"] = {
                        "tested": True,
                        "cuda_available": False,
                        "screenshot": screenshot,
                        "output": output
                    }
            except Exception as e:
                print(f"❌ Error interacting with notebook: {e}")
                results["gpu_notebook"] = {
                    "tested": False,
                    "error": str(e)
                }
        else:
            print("Existing notebook found - opening it")
            # Wait for the notebook to load
            time.sleep(5)
            
            # Take screenshot
            screenshot = take_screenshot(driver, "existing-notebook")
            
            results["gpu_notebook"] = {
                "tested": True,
                "message": "Existing notebook opened",
                "screenshot": screenshot
            }
    except Exception as e:
        print(f"❌ Error testing GPU notebook: {e}")
        results["gpu_notebook"] = {
            "tested": False,
            "error": str(e)
        }

def test_api_health(driver, url):
    """Test API health endpoint."""
    print("\nTesting API health endpoint...")
    
    api_url = f"{url.replace('jupyter', 'api')}/health"
    print(f"API URL: {api_url}")
    
    try:
        driver.get(api_url)
        time.sleep(3)
        
        # Take screenshot
        screenshot = take_screenshot(driver, "api-health")
        
        # Check if response contains health data
        page_source = driver.page_source
        
        if "healthy" in page_source or "status" in page_source:
            print("✅ API health endpoint returned data")
            results["api_health"] = {
                "accessible": True,
                "screenshot": screenshot,
                "content": page_source[:500] if len(page_source) > 500 else page_source
            }
        else:
            print("❌ API health endpoint did not return expected data")
            results["api_health"] = {
                "accessible": False,
                "screenshot": screenshot,
                "content": page_source[:500] if len(page_source) > 500 else page_source
            }
    except Exception as e:
        print(f"❌ Error testing API health: {e}")
        results["api_health"] = {
            "accessible": False,
            "error": str(e)
        }

def test_api_gpu_info(driver, url):
    """Test API GPU info endpoint."""
    print("\nTesting API GPU info endpoint...")
    
    api_url = f"{url.replace('jupyter', 'api')}/api/v1/system/gpu"
    print(f"API URL: {api_url}")
    
    try:
        driver.get(api_url)
        time.sleep(3)
        
        # Take screenshot
        screenshot = take_screenshot(driver, "api-gpu-info")
        
        # Check if response contains GPU data
        page_source = driver.page_source
        
        if "T4" in page_source or "NVIDIA" in page_source:
            print("✅ API GPU info endpoint returned data")
            results["api_gpu_info"] = {
                "accessible": True,
                "screenshot": screenshot,
                "content": page_source[:500] if len(page_source) > 500 else page_source
            }
            
            # Check if it's a T4 GPU
            if "T4" in page_source:
                print("✅ NVIDIA T4 GPU detected in API response")
                results["api_gpu_info"]["is_t4"] = True
            else:
                print("❌ NVIDIA T4 GPU not detected in API response")
                results["api_gpu_info"]["is_t4"] = False
        else:
            print("❌ API GPU info endpoint did not return expected data")
            results["api_gpu_info"] = {
                "accessible": False,
                "screenshot": screenshot,
                "content": page_source[:500] if len(page_source) > 500 else page_source
            }
    except Exception as e:
        print(f"❌ Error testing API GPU info: {e}")
        results["api_gpu_info"] = {
            "accessible": False,
            "error": str(e)
        }

def test_prometheus(driver, url):
    """Test Prometheus monitoring."""
    print("\nTesting Prometheus...")
    
    prometheus_url = url.replace("jupyter", "prometheus")
    print(f"Prometheus URL: {prometheus_url}")
    
    try:
        driver.get(prometheus_url)
        time.sleep(3)
        
        # Take screenshot
        screenshot = take_screenshot(driver, "prometheus")
        
        # Check if it's Prometheus
        if "Prometheus" in driver.title or "prometheus" in driver.page_source.lower():
            print("✅ Prometheus page accessible")
            results["prometheus"] = {
                "accessible": True,
                "screenshot": screenshot,
                "title": driver.title
            }
            
            # Try to check for GPU metrics
            try:
                # Search for nvidia metrics
                search_box = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Search']")
                search_box.clear()
                search_box.send_keys("nvidia_gpu")
                time.sleep(2)
                
                # Take screenshot of search results
                search_screenshot = take_screenshot(driver, "prometheus-search")
                
                # Check for results
                page_source = driver.page_source
                if "nvidia_gpu" in page_source:
                    print("✅ NVIDIA GPU metrics found in Prometheus")
                    results["prometheus"]["gpu_metrics"] = True
                    results["prometheus"]["search_screenshot"] = search_screenshot
                else:
                    print("❌ No NVIDIA GPU metrics found in Prometheus")
                    results["prometheus"]["gpu_metrics"] = False
                    results["prometheus"]["search_screenshot"] = search_screenshot
            except Exception as e:
                print(f"❌ Error searching for GPU metrics: {e}")
                results["prometheus"]["gpu_metrics_error"] = str(e)
        else:
            print("❌ Not a Prometheus page")
            results["prometheus"] = {
                "accessible": False,
                "screenshot": screenshot,
                "title": driver.title
            }
    except Exception as e:
        print(f"❌ Error testing Prometheus: {e}")
        results["prometheus"] = {
            "accessible": False,
            "error": str(e)
        }

def test_grafana(driver, url):
    """Test Grafana dashboards."""
    print("\nTesting Grafana...")
    
    grafana_url = url.replace("jupyter", "grafana")
    print(f"Grafana URL: {grafana_url}")
    
    try:
        driver.get(grafana_url)
        time.sleep(3)
        
        # Take screenshot
        screenshot = take_screenshot(driver, "grafana")
        
        # Check if it's Grafana
        if "Grafana" in driver.title or "grafana" in driver.page_source.lower():
            print("✅ Grafana page accessible")
            results["grafana"] = {
                "accessible": True,
                "screenshot": screenshot,
                "title": driver.title
            }
            
            # Look for GPU dashboard if we're logged in
            if "login" not in driver.page_source.lower() and "sign in" not in driver.page_source.lower():
                try:
                    # Try to navigate to dashboards
                    driver.get(f"{grafana_url}/dashboards")
                    time.sleep(3)
                    
                    # Take screenshot of dashboards
                    dashboards_screenshot = take_screenshot(driver, "grafana-dashboards")
                    
                    # Check for GPU dashboard
                    page_source = driver.page_source
                    if "GPU" in page_source or "NVIDIA" in page_source or "T4" in page_source:
                        print("✅ GPU dashboard found in Grafana")
                        results["grafana"]["gpu_dashboard"] = True
                        results["grafana"]["dashboards_screenshot"] = dashboards_screenshot
                    else:
                        print("❌ No GPU dashboard found in Grafana")
                        results["grafana"]["gpu_dashboard"] = False
                        results["grafana"]["dashboards_screenshot"] = dashboards_screenshot
                except Exception as e:
                    print(f"❌ Error checking for GPU dashboards: {e}")
                    results["grafana"]["dashboard_error"] = str(e)
            else:
                print("Login required to check for GPU dashboards")
                results["grafana"]["login_required"] = True
        else:
            print("❌ Not a Grafana page")
            results["grafana"] = {
                "accessible": False,
                "screenshot": screenshot,
                "title": driver.title
            }
    except Exception as e:
        print(f"❌ Error testing Grafana: {e}")
        results["grafana"] = {
            "accessible": False,
            "error": str(e)
        }

def generate_html_report():
    """Generate HTML report from results."""
    print("\nGenerating HTML report...")
    
    # Basic template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Web Test Report</title>
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
            flex-basis: 48%;
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
        .pass {{
            color: #28a745;
        }}
        .fail {{
            color: #dc3545;
        }}
        .unknown {{
            color: #6c757d;
        }}
        .screenshot {{
            max-width: 100%;
            height: auto;
            margin-top: 15px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .screenshot-container {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .result-box {{
            border-left: 5px solid #ddd;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .result-box.success {{
            border-left-color: #28a745;
        }}
        .result-box.failure {{
            border-left-color: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Web Test Report</h1>
            <p>Test ID: {TEST_ID} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
    
    # Determine overall status
    t4_detected = False
    if results.get("gpu_notebook", {}).get("is_t4", False) or results.get("api_gpu_info", {}).get("is_t4", False):
        t4_detected = True
    
    if t4_detected:
        overall_status = "PASS"
        overall_class = "pass"
        overall_message = "NVIDIA T4 GPU detected in the deployment"
    else:
        if "gpu_notebook" in results or "api_gpu_info" in results:
            overall_status = "FAIL"
            overall_class = "fail"
            overall_message = "Tests ran but NVIDIA T4 GPU not detected"
        else:
            overall_status = "UNKNOWN"
            overall_class = "unknown"
            overall_message = "Could not determine GPU status - tests may not have run correctly"
    
    # Add summary
    html += f"""
        <div class="summary">
            <div class="summary-box">
                <h2>Overall Status</h2>
                <div class="value {overall_class}">{overall_status}</div>
            </div>
            <div class="summary-box">
                <h2>Summary</h2>
                <div>{overall_message}</div>
            </div>
        </div>
"""
    
    # Add Jupyter section if tested
    if "jupyter" in results:
        jupyter = results["jupyter"]
        jupyter_status = "success" if jupyter.get("accessible", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>Jupyter Environment</h2>
            <div class="result-box {jupyter_status}">
                <h3>Status: {"✅ Accessible" if jupyter.get("accessible", False) else "❌ Not Accessible"}</h3>
                <p>Title: {jupyter.get("title", "Unknown")}</p>
"""
        
        # Add screenshot if available
        if "screenshot" in jupyter:
            screenshot_path = os.path.relpath(jupyter["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="Jupyter Screenshot" class="screenshot">
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add GPU notebook section if tested
    if "gpu_notebook" in results:
        gpu_notebook = results["gpu_notebook"]
        notebook_status = "success" if gpu_notebook.get("tested", False) and gpu_notebook.get("cuda_available", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>GPU Test Notebook</h2>
            <div class="result-box {notebook_status}">
                <h3>Status: {"✅ CUDA Available" if gpu_notebook.get("cuda_available", False) else "❌ CUDA Not Available"}</h3>
"""
        
        if gpu_notebook.get("is_t4", False):
            html += """
                <p class="pass">✅ NVIDIA T4 GPU Detected</p>
"""
        elif "is_t4" in gpu_notebook:
            html += """
                <p class="fail">❌ NVIDIA T4 GPU Not Detected</p>
"""
        
        # Add screenshot if available
        if "screenshot" in gpu_notebook:
            screenshot_path = os.path.relpath(gpu_notebook["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="GPU Notebook Screenshot" class="screenshot">
                </div>
"""
        
        # Add output if available
        if "output" in gpu_notebook:
            html += f"""
                <h4>Output:</h4>
                <pre>{gpu_notebook["output"]}</pre>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add API health section if tested
    if "api_health" in results:
        api_health = results["api_health"]
        api_status = "success" if api_health.get("accessible", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>API Health</h2>
            <div class="result-box {api_status}">
                <h3>Status: {"✅ Accessible" if api_health.get("accessible", False) else "❌ Not Accessible"}</h3>
"""
        
        # Add screenshot if available
        if "screenshot" in api_health:
            screenshot_path = os.path.relpath(api_health["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="API Health Screenshot" class="screenshot">
                </div>
"""
        
        # Add content if available
        if "content" in api_health:
            html += f"""
                <h4>Response:</h4>
                <pre>{api_health["content"]}</pre>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add API GPU info section if tested
    if "api_gpu_info" in results:
        api_gpu_info = results["api_gpu_info"]
        api_gpu_status = "success" if api_gpu_info.get("accessible", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>API GPU Info</h2>
            <div class="result-box {api_gpu_status}">
                <h3>Status: {"✅ Accessible" if api_gpu_info.get("accessible", False) else "❌ Not Accessible"}</h3>
"""
        
        if api_gpu_info.get("is_t4", False):
            html += """
                <p class="pass">✅ NVIDIA T4 GPU Detected</p>
"""
        elif "is_t4" in api_gpu_info:
            html += """
                <p class="fail">❌ NVIDIA T4 GPU Not Detected</p>
"""
        
        # Add screenshot if available
        if "screenshot" in api_gpu_info:
            screenshot_path = os.path.relpath(api_gpu_info["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="API GPU Info Screenshot" class="screenshot">
                </div>
"""
        
        # Add content if available
        if "content" in api_gpu_info:
            html += f"""
                <h4>Response:</h4>
                <pre>{api_gpu_info["content"]}</pre>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add Prometheus section if tested
    if "prometheus" in results:
        prometheus = results["prometheus"]
        prometheus_status = "success" if prometheus.get("accessible", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>Prometheus</h2>
            <div class="result-box {prometheus_status}">
                <h3>Status: {"✅ Accessible" if prometheus.get("accessible", False) else "❌ Not Accessible"}</h3>
"""
        
        if prometheus.get("gpu_metrics", False):
            html += """
                <p class="pass">✅ NVIDIA GPU Metrics Found</p>
"""
        elif "gpu_metrics" in prometheus:
            html += """
                <p class="fail">❌ No NVIDIA GPU Metrics Found</p>
"""
        
        # Add screenshot if available
        if "screenshot" in prometheus:
            screenshot_path = os.path.relpath(prometheus["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="Prometheus Screenshot" class="screenshot">
                </div>
"""
        
        # Add search screenshot if available
        if "search_screenshot" in prometheus:
            search_screenshot_path = os.path.relpath(prometheus["search_screenshot"], OUTPUT_DIR)
            html += f"""
                <h4>GPU Metrics Search:</h4>
                <div class="screenshot-container">
                    <img src="{search_screenshot_path}" alt="Prometheus Search Screenshot" class="screenshot">
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add Grafana section if tested
    if "grafana" in results:
        grafana = results["grafana"]
        grafana_status = "success" if grafana.get("accessible", False) else "failure"
        
        html += f"""
        <div class="section">
            <h2>Grafana</h2>
            <div class="result-box {grafana_status}">
                <h3>Status: {"✅ Accessible" if grafana.get("accessible", False) else "❌ Not Accessible"}</h3>
"""
        
        if grafana.get("login_required", False):
            html += """
                <p>Login required to check for GPU dashboards</p>
"""
        elif grafana.get("gpu_dashboard", False):
            html += """
                <p class="pass">✅ GPU Dashboard Found</p>
"""
        elif "gpu_dashboard" in grafana:
            html += """
                <p class="fail">❌ No GPU Dashboard Found</p>
"""
        
        # Add screenshot if available
        if "screenshot" in grafana:
            screenshot_path = os.path.relpath(grafana["screenshot"], OUTPUT_DIR)
            html += f"""
                <div class="screenshot-container">
                    <img src="{screenshot_path}" alt="Grafana Screenshot" class="screenshot">
                </div>
"""
        
        # Add dashboards screenshot if available
        if "dashboards_screenshot" in grafana:
            dashboards_screenshot_path = os.path.relpath(grafana["dashboards_screenshot"], OUTPUT_DIR)
            html += f"""
                <h4>Dashboards:</h4>
                <div class="screenshot-container">
                    <img src="{dashboards_screenshot_path}" alt="Grafana Dashboards Screenshot" class="screenshot">
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add recommendations section
    html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
"""
    
    # Add recommendations based on test results
    recommendations = []
    
    if not t4_detected:
        recommendations.append("Verify this is a T4 GPU instance - no T4 GPU was detected in tests")
    
    if "api_health" in results and not results["api_health"].get("accessible", False):
        recommendations.append("Check API service status and configuration")
    
    if "prometheus" in results:
        prometheus = results["prometheus"]
        if not prometheus.get("accessible", False):
            recommendations.append("Check Prometheus service status and configuration")
        elif not prometheus.get("gpu_metrics", False):
            recommendations.append("Configure Prometheus to collect NVIDIA GPU metrics")
    
    if "grafana" in results:
        grafana = results["grafana"]
        if not grafana.get("accessible", False):
            recommendations.append("Check Grafana service status and configuration")
        elif not grafana.get("login_required", False) and not grafana.get("gpu_dashboard", False):
            recommendations.append("Add GPU dashboards to Grafana for monitoring")
    
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
    with open(RESULTS_HTML, "w") as f:
        f.write(html)
    
    print(f"HTML report saved to: {RESULTS_HTML}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Web-based NVIDIA T4 Testing Script"
    )
    
    parser.add_argument(
        "url",
        nargs="?",
        default="https://jupyter0-ipzl7zn0p.brevlab.com",
        help="URL of the Jupyter environment (default: https://jupyter0-ipzl7zn0p.brevlab.com)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (no browser UI)"
    )
    
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open HTML report when testing is complete"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"====== NVIDIA T4 Web Test ======")
    print(f"URL: {args.url}")
    print(f"Output: {RESULTS_HTML}")
    print(f"===============================")
    
    # Initialize WebDriver
    driver = setup_webdriver(headless=args.headless)
    
    try:
        # Run tests
        test_jupyter_access(driver, args.url)
        test_gpu_notebook(driver, args.url)
        test_api_health(driver, args.url)
        test_api_gpu_info(driver, args.url)
        test_prometheus(driver, args.url)
        test_grafana(driver, args.url)
        
        # Save results
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {RESULTS_JSON}")
        
        # Generate HTML report
        generate_html_report()
        
        # Open report if requested
        if args.open_report:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(RESULTS_HTML)}")
                print(f"Opened HTML report in browser")
            except:
                print(f"Failed to open HTML report automatically")
        
        # Determine exit code
        t4_detected = False
        if results.get("gpu_notebook", {}).get("is_t4", False) or results.get("api_gpu_info", {}).get("is_t4", False):
            t4_detected = True
            return 0  # Success
        
        if "gpu_notebook" in results or "api_gpu_info" in results:
            return 1  # Tests ran but no T4 detected
        
        return 2  # Tests failed to run properly
        
    finally:
        # Clean up WebDriver
        driver.quit()

if __name__ == "__main__":
    sys.exit(main())