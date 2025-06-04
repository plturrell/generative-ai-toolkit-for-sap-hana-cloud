#!/usr/bin/env python3
"""
Automated CLI testing for NVIDIA T4 deployment.

This script tests the deployment process on an NVIDIA T4 instance,
validating GPU detection, configuration, and monitoring components.
"""
import os
import sys
import argparse
import subprocess
import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NVIDIADeploymentTester:
    """Tests the NVIDIA T4 deployment setup and functionality."""
    
    def __init__(self, 
                host: str = "localhost", 
                api_port: int = 8000, 
                prometheus_port: int = 9091,
                grafana_port: int = 3000, 
                gpu_metrics_port: int = 9835,
                ssh_key: Optional[str] = None,
                username: Optional[str] = None):
        """
        Initialize the deployment tester.
        
        Args:
            host: Hostname or IP of the NVIDIA instance
            api_port: Port for the API service
            prometheus_port: Port for Prometheus metrics
            grafana_port: Port for Grafana dashboards
            gpu_metrics_port: Port for NVIDIA GPU metrics exporter
            ssh_key: Path to SSH key for remote testing (optional)
            username: Username for SSH connection (optional)
        """
        self.host = host
        self.api_port = api_port
        self.prometheus_port = prometheus_port
        self.grafana_port = grafana_port
        self.gpu_metrics_port = gpu_metrics_port
        self.ssh_key = ssh_key
        self.username = username
        self.remote_mode = (host != "localhost" and host != "127.0.0.1")
        self.results = {
            "gpu_detection": {"status": "not_tested", "details": {}},
            "docker_setup": {"status": "not_tested", "details": {}},
            "api_health": {"status": "not_tested", "details": {}},
            "monitoring": {"status": "not_tested", "details": {}},
            "gpu_utilization": {"status": "not_tested", "details": {}}
        }
    
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
            
            logger.info(f"Running remote command: {' '.join(ssh_cmd)}")
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            logger.info(f"Running local command: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    
    def test_gpu_detection(self) -> Dict[str, Any]:
        """
        Test GPU detection to verify T4 availability.
        
        Returns:
            Dict with test results
        """
        logger.info("Testing GPU detection...")
        
        # Test nvidia-smi
        returncode, stdout, stderr = self.run_command("nvidia-smi")
        
        if returncode != 0:
            logger.error(f"nvidia-smi failed: {stderr}")
            self.results["gpu_detection"] = {
                "status": "error",
                "details": {
                    "error": stderr,
                    "message": "Failed to run nvidia-smi. NVIDIA drivers may not be installed."
                }
            }
            return self.results["gpu_detection"]
        
        # Check for T4 in output
        t4_detected = "T4" in stdout
        
        if not t4_detected:
            logger.warning("NVIDIA GPU detected but not a T4 model")
            self.results["gpu_detection"] = {
                "status": "warning",
                "details": {
                    "output": stdout,
                    "message": "NVIDIA GPU detected but not a T4 model."
                }
            }
        else:
            logger.info("NVIDIA T4 GPU detected!")
            
            # Get memory info
            returncode, stdout, stderr = self.run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
            memory_mb = int(stdout.strip()) if returncode == 0 else "unknown"
            
            self.results["gpu_detection"] = {
                "status": "success",
                "details": {
                    "model": "T4",
                    "memory_mb": memory_mb,
                    "message": f"NVIDIA T4 GPU with {memory_mb}MB memory detected successfully."
                }
            }
        
        return self.results["gpu_detection"]
    
    def test_docker_setup(self) -> Dict[str, Any]:
        """
        Test Docker setup for container deployment.
        
        Returns:
            Dict with test results
        """
        logger.info("Testing Docker setup...")
        
        # Check Docker installation
        returncode, stdout, stderr = self.run_command("docker --version")
        
        if returncode != 0:
            logger.error(f"Docker not installed: {stderr}")
            self.results["docker_setup"] = {
                "status": "error",
                "details": {
                    "error": stderr,
                    "message": "Docker is not installed or accessible."
                }
            }
            return self.results["docker_setup"]
        
        # Check Docker Compose installation
        returncode, stdout, stderr = self.run_command("docker-compose --version")
        
        if returncode != 0:
            logger.warning(f"Docker Compose not installed: {stderr}")
            self.results["docker_setup"] = {
                "status": "warning",
                "details": {
                    "docker": stdout.strip(),
                    "docker_compose": "Not installed",
                    "message": "Docker is installed but Docker Compose is missing."
                }
            }
            return self.results["docker_setup"]
        
        # Check NVIDIA Docker Runtime
        returncode, stdout, stderr = self.run_command("docker info | grep -i nvidia")
        
        if returncode != 0 or "nvidia" not in stdout.lower():
            logger.warning("NVIDIA Docker runtime not detected")
            self.results["docker_setup"] = {
                "status": "warning",
                "details": {
                    "docker": True,
                    "docker_compose": True,
                    "nvidia_runtime": False,
                    "message": "Docker and Docker Compose are installed, but NVIDIA runtime is missing."
                }
            }
        else:
            logger.info("Docker with NVIDIA runtime detected!")
            self.results["docker_setup"] = {
                "status": "success",
                "details": {
                    "docker": True,
                    "docker_compose": True,
                    "nvidia_runtime": True,
                    "message": "Docker with NVIDIA runtime is properly configured."
                }
            }
        
        return self.results["docker_setup"]
    
    def test_api_health(self) -> Dict[str, Any]:
        """
        Test API health endpoint.
        
        Returns:
            Dict with test results
        """
        logger.info("Testing API health...")
        
        api_url = f"http://{self.host}:{self.api_port}/health"
        
        try:
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("API health check successful")
                
                # Parse response
                try:
                    health_data = response.json()
                    self.results["api_health"] = {
                        "status": "success",
                        "details": {
                            "response_code": response.status_code,
                            "health_status": health_data.get("status"),
                            "message": "API health check successful."
                        }
                    }
                except:
                    self.results["api_health"] = {
                        "status": "success",
                        "details": {
                            "response_code": response.status_code,
                            "message": "API health check successful but response was not valid JSON."
                        }
                    }
            else:
                logger.warning(f"API health check failed with status code: {response.status_code}")
                self.results["api_health"] = {
                    "status": "warning",
                    "details": {
                        "response_code": response.status_code,
                        "message": f"API health check returned status code {response.status_code}."
                    }
                }
        except requests.RequestException as e:
            logger.error(f"API health check error: {str(e)}")
            self.results["api_health"] = {
                "status": "error",
                "details": {
                    "error": str(e),
                    "message": "Failed to connect to API health endpoint."
                }
            }
        
        return self.results["api_health"]
    
    def test_monitoring(self) -> Dict[str, Any]:
        """
        Test monitoring stack (Prometheus, Grafana).
        
        Returns:
            Dict with test results
        """
        logger.info("Testing monitoring services...")
        
        monitoring_results = {
            "prometheus": False,
            "grafana": False,
            "gpu_metrics": False
        }
        
        # Check Prometheus
        prometheus_url = f"http://{self.host}:{self.prometheus_port}/-/healthy"
        try:
            response = requests.get(prometheus_url, timeout=5)
            monitoring_results["prometheus"] = response.status_code == 200
            logger.info(f"Prometheus check: {'Success' if monitoring_results['prometheus'] else 'Failed'}")
        except requests.RequestException:
            logger.warning("Failed to connect to Prometheus")
        
        # Check Grafana
        grafana_url = f"http://{self.host}:{self.grafana_port}/api/health"
        try:
            response = requests.get(grafana_url, timeout=5)
            monitoring_results["grafana"] = response.status_code == 200
            logger.info(f"Grafana check: {'Success' if monitoring_results['grafana'] else 'Failed'}")
        except requests.RequestException:
            logger.warning("Failed to connect to Grafana")
        
        # Check GPU metrics exporter
        gpu_metrics_url = f"http://{self.host}:{self.gpu_metrics_port}/metrics"
        try:
            response = requests.get(gpu_metrics_url, timeout=5)
            monitoring_results["gpu_metrics"] = (
                response.status_code == 200 and 
                "nvidia_gpu" in response.text
            )
            logger.info(f"GPU metrics check: {'Success' if monitoring_results['gpu_metrics'] else 'Failed'}")
        except requests.RequestException:
            logger.warning("Failed to connect to GPU metrics exporter")
        
        # Determine overall status
        if all(monitoring_results.values()):
            status = "success"
            message = "All monitoring services are running correctly."
        elif any(monitoring_results.values()):
            status = "warning"
            message = "Some monitoring services are running, but not all."
        else:
            status = "error"
            message = "None of the monitoring services are running."
        
        self.results["monitoring"] = {
            "status": status,
            "details": {
                **monitoring_results,
                "message": message
            }
        }
        
        return self.results["monitoring"]
    
    def test_gpu_utilization(self) -> Dict[str, Any]:
        """
        Test GPU utilization reporting.
        
        Returns:
            Dict with test results
        """
        logger.info("Testing GPU utilization metrics...")
        
        if self.results["monitoring"]["details"].get("gpu_metrics", False):
            # GPU metrics exporter is running, check for T4 metrics
            gpu_metrics_url = f"http://{self.host}:{self.gpu_metrics_port}/metrics"
            try:
                response = requests.get(gpu_metrics_url, timeout=5)
                if response.status_code == 200:
                    # Look for key GPU metrics
                    metrics = {
                        "utilization": "nvidia_gpu_duty_cycle" in response.text,
                        "memory": "nvidia_gpu_memory_used_bytes" in response.text,
                        "temperature": "nvidia_gpu_temperature_celsius" in response.text,
                        "power": "nvidia_gpu_power_draw_watts" in response.text
                    }
                    
                    if all(metrics.values()):
                        status = "success"
                        message = "All GPU metrics are being reported correctly."
                    elif any(metrics.values()):
                        status = "warning"
                        message = "Some GPU metrics are being reported, but not all."
                    else:
                        status = "error"
                        message = "No GPU metrics are being reported."
                    
                    self.results["gpu_utilization"] = {
                        "status": status,
                        "details": {
                            "metrics": metrics,
                            "message": message
                        }
                    }
                else:
                    self.results["gpu_utilization"] = {
                        "status": "error",
                        "details": {
                            "response_code": response.status_code,
                            "message": "GPU metrics endpoint returned an error."
                        }
                    }
            except requests.RequestException as e:
                self.results["gpu_utilization"] = {
                    "status": "error",
                    "details": {
                        "error": str(e),
                        "message": "Failed to connect to GPU metrics endpoint."
                    }
                }
        else:
            # Use nvidia-smi to get GPU utilization
            returncode, stdout, stderr = self.run_command(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw " +
                "--format=csv,noheader,nounits"
            )
            
            if returncode == 0:
                try:
                    values = stdout.strip().split(',')
                    if len(values) >= 4:
                        utilization, memory, temperature, power = [float(v.strip()) for v in values]
                        self.results["gpu_utilization"] = {
                            "status": "success",
                            "details": {
                                "utilization_percent": utilization,
                                "memory_used_mb": memory,
                                "temperature_celsius": temperature,
                                "power_draw_watts": power,
                                "message": "GPU utilization metrics retrieved successfully via nvidia-smi."
                            }
                        }
                    else:
                        self.results["gpu_utilization"] = {
                            "status": "warning",
                            "details": {
                                "raw_output": stdout.strip(),
                                "message": "GPU utilization metrics format unexpected."
                            }
                        }
                except ValueError:
                    self.results["gpu_utilization"] = {
                        "status": "warning",
                        "details": {
                            "raw_output": stdout.strip(),
                            "message": "Failed to parse GPU utilization metrics."
                        }
                    }
            else:
                self.results["gpu_utilization"] = {
                    "status": "error",
                    "details": {
                        "error": stderr,
                        "message": "Failed to get GPU utilization metrics via nvidia-smi."
                    }
                }
        
        return self.results["gpu_utilization"]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all deployment tests.
        
        Returns:
            Dict with all test results
        """
        logger.info(f"Starting deployment tests on {self.host}...")
        
        # Run all tests
        self.test_gpu_detection()
        self.test_docker_setup()
        self.test_api_health()
        self.test_monitoring()
        self.test_gpu_utilization()
        
        # Determine overall status
        success_count = sum(1 for result in self.results.values() if result["status"] == "success")
        warning_count = sum(1 for result in self.results.values() if result["status"] == "warning")
        error_count = sum(1 for result in self.results.values() if result["status"] == "error")
        not_tested = sum(1 for result in self.results.values() if result["status"] == "not_tested")
        
        if error_count > 0:
            overall_status = "error"
            overall_message = f"Deployment tests failed with {error_count} errors."
        elif warning_count > 0:
            overall_status = "warning"
            overall_message = f"Deployment tests completed with {warning_count} warnings."
        elif not_tested > 0:
            overall_status = "incomplete"
            overall_message = f"Deployment tests incomplete with {not_tested} untested components."
        else:
            overall_status = "success"
            overall_message = "All deployment tests passed successfully."
        
        # Add overall results
        self.results["overall"] = {
            "status": overall_status,
            "message": overall_message,
            "counts": {
                "success": success_count,
                "warning": warning_count,
                "error": error_count,
                "not_tested": not_tested
            },
            "timestamp": time.time()
        }
        
        return self.results
    
    def print_report(self) -> None:
        """Print a formatted report of test results."""
        if "overall" not in self.results:
            print("No test results available. Run tests first.")
            return
        
        print("\n" + "=" * 80)
        print(f"NVIDIA T4 DEPLOYMENT TEST REPORT: {self.host}")
        print("=" * 80)
        
        overall = self.results["overall"]
        status_icon = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "incomplete": "⏳",
            "not_tested": "🔷"
        }
        
        print(f"\nOverall Status: {status_icon[overall['status']]} {overall['status'].upper()}")
        print(f"Message: {overall['message']}")
        print(f"Success: {overall['counts']['success']}, Warnings: {overall['counts']['warning']}, Errors: {overall['counts']['error']}")
        
        # Print individual test results
        for test_name, result in self.results.items():
            if test_name == "overall":
                continue
            
            print(f"\n{status_icon[result['status']]} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
            if "message" in result["details"]:
                print(f"  {result['details']['message']}")
            
            # Print additional details
            details = {k: v for k, v in result["details"].items() if k != "message"}
            if details:
                print("  Details:")
                for key, value in details.items():
                    if isinstance(value, dict):
                        print(f"    {key}:")
                        for subkey, subvalue in value.items():
                            print(f"      {subkey}: {subvalue}")
                    else:
                        print(f"    {key}: {value}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_file: str) -> None:
        """
        Save test results to a JSON file.
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated CLI testing for NVIDIA T4 deployment"
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
        "--output",
        help="Path to save test results JSON"
    )
    
    parser.add_argument(
        "--test",
        choices=["all", "gpu", "docker", "api", "monitoring", "utilization"],
        default="all",
        help="Specific test to run (default: all)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI testing tool."""
    args = parse_arguments()
    
    # Initialize tester
    tester = NVIDIADeploymentTester(
        host=args.host,
        api_port=args.api_port,
        prometheus_port=args.prometheus_port,
        grafana_port=args.grafana_port,
        gpu_metrics_port=args.gpu_metrics_port,
        ssh_key=args.ssh_key,
        username=args.username
    )
    
    # Run specified tests
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "gpu":
        tester.test_gpu_detection()
    elif args.test == "docker":
        tester.test_docker_setup()
    elif args.test == "api":
        tester.test_api_health()
    elif args.test == "monitoring":
        tester.test_monitoring()
    elif args.test == "utilization":
        tester.test_gpu_utilization()
    
    # Print report
    tester.print_report()
    
    # Save report if requested
    if args.output:
        tester.save_report(args.output)

if __name__ == "__main__":
    main()