#!/usr/bin/env python3
"""
Comprehensive Test Suite for NVIDIA T4 Deployments.

This script runs the complete test suite including:
1. Automated environment and optimization tests
2. Load testing with variable user counts
3. HTML report generation with consolidated results

Usage:
    python run_t4_test_suite.py [options]
"""
import os
import sys
import time
import json
import argparse
import logging
import subprocess
import datetime
import webbrowser
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REPORT_DIR = "t4-test-reports"


def run_command(command: str) -> int:
    """
    Run a shell command and return the exit code.
    
    Args:
        command: Command to execute
        
    Returns:
        Command exit code
    """
    logger.info(f"Running command: {command}")
    return subprocess.call(command, shell=True)


def run_automated_tests(args: argparse.Namespace) -> str:
    """
    Run the automated tests for T4 deployment.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the HTML report file
    """
    logger.info("Starting automated environment and optimization tests...")
    
    # Build command
    cmd = [
        f"python3 run_automated_tests.py",
        f"--host {args.host}",
        f"--api-port {args.api_port}",
        f"--prometheus-port {args.prometheus_port}",
        f"--grafana-port {args.grafana_port}",
        f"--gpu-metrics-port {args.gpu_metrics_port}",
        f"--output-dir {os.path.join(args.output_dir, 'automated-tests')}"
    ]
    
    # Add optional arguments
    if args.ssh_key:
        cmd.append(f"--ssh-key {args.ssh_key}")
    
    if args.username:
        cmd.append(f"--username {args.username}")
    
    # Add open-report flag
    cmd.append("--open-report")
    
    # Run command
    exit_code = run_command(" ".join(cmd))
    
    if exit_code != 0:
        logger.warning(f"Automated tests completed with exit code {exit_code}")
    else:
        logger.info("Automated tests completed successfully")
    
    # Find the latest HTML report
    report_dir = os.path.join(args.output_dir, "automated-tests")
    report_files = [f for f in os.listdir(report_dir) if f.endswith(".html")]
    
    if not report_files:
        logger.warning("No HTML report found")
        return ""
    
    # Sort by modification time (newest first)
    report_files.sort(key=lambda f: os.path.getmtime(os.path.join(report_dir, f)), reverse=True)
    
    return os.path.join(report_dir, report_files[0])


def run_load_tests(args: argparse.Namespace) -> str:
    """
    Run load tests for T4 deployment.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the HTML report file
    """
    logger.info("Starting load tests...")
    
    # Build command
    cmd = [
        f"python3 load_test.py",
        f"--host {args.host}",
        f"--port {args.api_port}",
        f"--gpu-metrics-port {args.gpu_metrics_port}",
        f"--output-dir {os.path.join(args.output_dir, 'load-tests')}"
    ]
    
    # Add optional arguments
    if args.api_key:
        cmd.append(f"--api-key {args.api_key}")
    
    # Add custom user counts if specified
    if args.load_test_users:
        user_counts = " ".join(map(str, args.load_test_users))
        cmd.append(f"--users {user_counts}")
    
    # Add custom duration if specified
    if args.load_test_duration:
        cmd.append(f"--duration {args.load_test_duration}")
    
    # Run command
    exit_code = run_command(" ".join(cmd))
    
    if exit_code != 0:
        logger.warning(f"Load tests completed with exit code {exit_code}")
    else:
        logger.info("Load tests completed successfully")
    
    # Find the latest HTML report
    report_dir = os.path.join(args.output_dir, "load-tests")
    
    if not os.path.exists(report_dir):
        logger.warning(f"Report directory not found: {report_dir}")
        return ""
    
    report_files = [f for f in os.listdir(report_dir) if f.endswith(".html")]
    
    if not report_files:
        logger.warning("No HTML report found")
        return ""
    
    # Sort by modification time (newest first)
    report_files.sort(key=lambda f: os.path.getmtime(os.path.join(report_dir, f)), reverse=True)
    
    return os.path.join(report_dir, report_files[0])


def generate_consolidated_report(args: argparse.Namespace, 
                               automated_report: str, 
                               load_report: str) -> str:
    """
    Generate a consolidated HTML report.
    
    Args:
        args: Command-line arguments
        automated_report: Path to automated test report
        load_report: Path to load test report
        
    Returns:
        Path to the consolidated report
    """
    logger.info("Generating consolidated report...")
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create report path
    report_file = os.path.join(args.output_dir, f"t4-consolidated-report-{timestamp}.html")
    
    # Extract information from reports
    automated_info = {}
    load_info = {}
    
    # Read automated test report
    if automated_report and os.path.exists(automated_report):
        with open(automated_report, "r") as f:
            content = f.read()
            # Extract key information using simple parsing
            # This is a simplified approach - in a production tool, we'd parse the JSON data
            automated_info["found"] = True
            
            # Extract overall status
            if "Overall Status: PASSED" in content:
                automated_info["status"] = "passed"
            elif "Overall Status: FAILED" in content:
                automated_info["status"] = "failed"
            else:
                automated_info["status"] = "unknown"
                
            # Extract file path
            automated_info["report_path"] = os.path.basename(automated_report)
    else:
        automated_info["found"] = False
    
    # Read load test report
    if load_report and os.path.exists(load_report):
        with open(load_report, "r") as f:
            content = f.read()
            load_info["found"] = True
            
            # Extract throughput information
            import re
            
            # Extract requests per second
            rps_match = re.search(r'Requests/Second.*?<div class="value">(\d+\.\d+)</div>', content)
            load_info["requests_per_second"] = float(rps_match.group(1)) if rps_match else 0
            
            # Extract tokens per second
            tps_match = re.search(r'Tokens/Second.*?<div class="value">(\d+\.\d+)</div>', content)
            load_info["tokens_per_second"] = float(tps_match.group(1)) if tps_match else 0
            
            # Extract GPU utilization
            gpu_util_match = re.search(r'Avg GPU Utilization.*?<div class="value">(\d+\.\d+)%</div>', content)
            load_info["avg_gpu_utilization"] = float(gpu_util_match.group(1)) if gpu_util_match else 0
            
            # Extract file path
            load_info["report_path"] = os.path.basename(load_report)
    else:
        load_info["found"] = False
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Consolidated Test Report</title>
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
            flex: 1;
            margin-right: 10px;
            text-align: center;
            margin-bottom: 10px;
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
        .unknown {{
            color: #ffc107;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .report-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .report-card h3 {{
            margin: 0;
        }}
        .report-card .btn {{
            background-color: #76b900;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .info-item {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
        }}
        .info-item label {{
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        .info-item .value {{
            font-size: 18px;
        }}
        .recommendation {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 5px solid #007bff;
        }}
        .recommendation h3 {{
            margin-top: 0;
            color: #007bff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Consolidated Test Report</h1>
            <p>Host: {args.host} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="section">
            <h2>Test Suite Summary</h2>
            <div class="summary">
                <div class="summary-box">
                    <h2>Environment Tests</h2>
                    <div class="value {automated_info.get('status', 'unknown') if automated_info.get('found', False) else 'unknown'}">
                        {automated_info.get('status', 'Not Run').upper() if automated_info.get('found', False) else 'NOT RUN'}
                    </div>
                </div>
                <div class="summary-box">
                    <h2>Load Tests</h2>
                    <div class="value {'' if not load_info.get('found', False) else 'passed' if load_info.get('requests_per_second', 0) > 0 else 'failed'}">
                        {('COMPLETED' if load_info.get('requests_per_second', 0) > 0 else 'FAILED') if load_info.get('found', False) else 'NOT RUN'}
                    </div>
                </div>
                <div class="summary-box">
                    <h2>Requests/Second</h2>
                    <div class="value">{load_info.get('requests_per_second', 'N/A')}</div>
                </div>
                <div class="summary-box">
                    <h2>Tokens/Second</h2>
                    <div class="value">{load_info.get('tokens_per_second', 'N/A')}</div>
                </div>
                <div class="summary-box">
                    <h2>Avg GPU Utilization</h2>
                    <div class="value">{load_info.get('avg_gpu_utilization', 'N/A')}%</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Reports</h2>
"""

    # Add automated test report card
    if automated_info.get("found", False):
        html += f"""
            <div class="report-card">
                <div>
                    <h3>Environment & Optimization Tests</h3>
                    <p>Status: <span class="{automated_info.get('status', 'unknown')}">{automated_info.get('status', 'Unknown').upper()}</span></p>
                </div>
                <a href="automated-tests/{automated_info['report_path']}" class="btn" target="_blank">View Report</a>
            </div>
"""
    else:
        html += """
            <div class="report-card">
                <div>
                    <h3>Environment & Optimization Tests</h3>
                    <p>Status: <span class="unknown">NOT RUN</span></p>
                </div>
                <span>No report available</span>
            </div>
"""

    # Add load test report card
    if load_info.get("found", False):
        html += f"""
            <div class="report-card">
                <div>
                    <h3>Load Tests</h3>
                    <p>Performance metrics for various user loads</p>
                </div>
                <a href="load-tests/{load_info['report_path']}" class="btn" target="_blank">View Report</a>
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <label>Requests/Second</label>
                    <div class="value">{load_info.get('requests_per_second', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <label>Tokens/Second</label>
                    <div class="value">{load_info.get('tokens_per_second', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <label>Avg GPU Utilization</label>
                    <div class="value">{load_info.get('avg_gpu_utilization', 'N/A')}%</div>
                </div>
            </div>
"""
    else:
        html += """
            <div class="report-card">
                <div>
                    <h3>Load Tests</h3>
                    <p>Status: <span class="unknown">NOT RUN</span></p>
                </div>
                <span>No report available</span>
            </div>
"""

    # Add recommendations section
    html += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
"""

    # Add automated test recommendations
    if automated_info.get("found", False) and automated_info.get("status") == "failed":
        html += """
            <div class="recommendation">
                <h3>Environment Setup</h3>
                <p>Some environment tests have failed. Please check the detailed report for specific issues.</p>
                <ul>
                    <li>Verify NVIDIA drivers are installed correctly</li>
                    <li>Check TensorRT configuration</li>
                    <li>Ensure Docker with NVIDIA runtime is configured properly</li>
                </ul>
            </div>
"""

    # Add load test recommendations
    if load_info.get("found", False):
        gpu_utilization = load_info.get('avg_gpu_utilization', 0)
        
        if gpu_utilization < 30:
            html += """
            <div class="recommendation">
                <h3>Low GPU Utilization</h3>
                <p>GPU utilization is below optimal levels. Consider these improvements:</p>
                <ul>
                    <li>Increase batch sizes to better utilize GPU resources</li>
                    <li>Optimize request patterns to allow for better batching</li>
                    <li>Check for potential CPU bottlenecks in the pipeline</li>
                    <li>Verify TensorRT optimization settings</li>
                </ul>
            </div>
"""
        elif gpu_utilization > 90:
            html += """
            <div class="recommendation">
                <h3>High GPU Utilization</h3>
                <p>GPU is running at very high utilization, which is good for throughput but may impact latency:</p>
                <ul>
                    <li>Monitor for thermal throttling if sustained for long periods</li>
                    <li>Consider load balancing if latency becomes an issue</li>
                    <li>Ensure adequate cooling for production deployments</li>
                </ul>
            </div>
"""
        
        # Add token throughput recommendations
        tokens_per_second = load_info.get('tokens_per_second', 0)
        if tokens_per_second < 10:
            html += """
            <div class="recommendation">
                <h3>Low Token Throughput</h3>
                <p>Token generation rate is lower than expected for T4 GPUs:</p>
                <ul>
                    <li>Verify FP16 precision is enabled for inference</li>
                    <li>Check model size and whether it's appropriate for T4 memory (16GB)</li>
                    <li>Consider model quantization (INT8) for larger models</li>
                    <li>Implement caching for common queries</li>
                </ul>
            </div>
"""

    html += """
        </div>
    </div>
</body>
</html>
"""

    # Write HTML to file
    with open(report_file, "w") as f:
        f.write(html)
    
    logger.info(f"Consolidated report saved to {report_file}")
    
    return report_file


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Suite for NVIDIA T4 Deployments"
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
        "--api-key",
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Directory to store test reports (default: {DEFAULT_REPORT_DIR})"
    )
    
    parser.add_argument(
        "--load-test-users",
        type=int,
        nargs="+",
        help="Concurrent user counts for load testing (default: 1 5 10 20)"
    )
    
    parser.add_argument(
        "--load-test-duration",
        type=int,
        help="Duration in seconds for each load test (default: 60)"
    )
    
    parser.add_argument(
        "--skip-automated-tests",
        action="store_true",
        help="Skip automated environment and optimization tests"
    )
    
    parser.add_argument(
        "--skip-load-tests",
        action="store_true",
        help="Skip load tests"
    )
    
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open consolidated report in browser when tests complete"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the test suite."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "automated-tests"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "load-tests"), exist_ok=True)
    
    logger.info(f"Starting NVIDIA T4 test suite for {args.host}")
    
    # Initialize report paths
    automated_report = ""
    load_report = ""
    
    # Run automated tests
    if not args.skip_automated_tests:
        automated_report = run_automated_tests(args)
    else:
        logger.info("Skipping automated tests")
    
    # Run load tests
    if not args.skip_load_tests:
        load_report = run_load_tests(args)
    else:
        logger.info("Skipping load tests")
    
    # Generate consolidated report
    consolidated_report = generate_consolidated_report(args, automated_report, load_report)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"NVIDIA T4 TEST SUITE COMPLETED")
    print("=" * 80)
    
    if automated_report:
        print(f"Automated Tests: Completed - {automated_report}")
    else:
        print("Automated Tests: Skipped")
    
    if load_report:
        print(f"Load Tests: Completed - {load_report}")
    else:
        print("Load Tests: Skipped")
    
    print(f"\nConsolidated Report: {consolidated_report}")
    print("=" * 80)
    
    # Open report in browser if requested
    if args.open_report and consolidated_report:
        webbrowser.open(f"file://{os.path.abspath(consolidated_report)}")
        logger.info(f"Opened consolidated report in browser")


if __name__ == "__main__":
    main()