#!/usr/bin/env python3
"""
Load Testing Tool for NVIDIA T4 Deployments.

This script simulates concurrent requests to measure system performance
under various load conditions, focusing on GPU utilization and throughput.

Usage:
    python load_test.py [options]
"""
import os
import sys
import time
import json
import uuid
import random
import logging
import argparse
import datetime
import threading
import concurrent.futures
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for matplotlib and numpy
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib and/or numpy not available. Plotting will be disabled.")


@dataclass
class LoadTestConfig:
    """Configuration for load test."""
    host: str = "localhost"
    port: int = 8000
    api_key: Optional[str] = None
    gpu_metrics_port: int = 9835
    concurrent_users: List[int] = None
    duration_seconds: int = 60
    ramp_up_seconds: int = 15
    ramp_down_seconds: int = 5
    request_types: List[str] = None
    request_distribution: Dict[str, float] = None
    think_time_ms: Tuple[int, int] = (500, 2000)
    output_dir: str = "load-test-results"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.concurrent_users is None:
            self.concurrent_users = [1, 5, 10, 20]
        
        if self.request_types is None:
            self.request_types = ["query", "generate", "embeddings", "vectorsearch"]
        
        if self.request_distribution is None:
            # Equal distribution by default
            self.request_distribution = {
                request_type: 1.0 / len(self.request_types) 
                for request_type in self.request_types
            }


@dataclass
class RequestResult:
    """Result of a single request."""
    request_type: str
    start_time: float
    end_time: float
    status_code: int
    success: bool
    error: Optional[str] = None
    response_size: int = 0
    tokens_generated: int = 0
    request_id: str = ""
    
    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


class GPUMetricsCollector(threading.Thread):
    """
    Collects GPU metrics during load test.
    
    This runs in a separate thread and periodically collects
    GPU utilization, memory usage, and other metrics.
    """
    
    def __init__(self, host: str, port: int = 9835, interval_seconds: int = 1):
        """
        Initialize GPU metrics collector.
        
        Args:
            host: Hostname or IP address
            port: GPU metrics exporter port
            interval_seconds: Collection interval in seconds
        """
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.interval = interval_seconds
        self.running = False
        self.metrics = []
        self.lock = threading.Lock()
    
    def run(self):
        """Run metrics collection thread."""
        self.running = True
        
        while self.running:
            try:
                metrics = self._collect_metrics()
                
                with self.lock:
                    self.metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
            
            time.sleep(self.interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current GPU metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics_url = f"http://{self.host}:{self.port}/metrics"
        
        try:
            response = requests.get(metrics_url, timeout=5)
            
            if response.status_code != 200:
                return {
                    "timestamp": time.time(),
                    "error": f"HTTP {response.status_code}",
                    "utilization_percent": 0,
                    "memory_used_mb": 0,
                    "temperature_celsius": 0,
                    "power_watts": 0
                }
            
            # Parse Prometheus metrics format
            metrics_text = response.text
            
            # Extract metrics using simple parsing
            # In a production tool, we'd use a proper Prometheus client library
            import re
            
            # Extract GPU utilization (duty cycle)
            util_match = re.search(r'nvidia_gpu_duty_cycle{[^}]*}\s+(\d+)', metrics_text)
            utilization = int(util_match.group(1)) if util_match else 0
            
            # Extract memory usage
            mem_match = re.search(r'nvidia_gpu_memory_used_bytes{[^}]*}\s+(\d+)', metrics_text)
            memory_bytes = int(mem_match.group(1)) if mem_match else 0
            memory_mb = memory_bytes / (1024 * 1024) if memory_bytes else 0
            
            # Extract temperature
            temp_match = re.search(r'nvidia_gpu_temperature_celsius{[^}]*}\s+(\d+)', metrics_text)
            temperature = int(temp_match.group(1)) if temp_match else 0
            
            # Extract power usage
            power_match = re.search(r'nvidia_gpu_power_draw_watts{[^}]*}\s+(\d+\.\d+)', metrics_text)
            power = float(power_match.group(1)) if power_match else 0
            
            return {
                "timestamp": time.time(),
                "utilization_percent": utilization,
                "memory_used_mb": memory_mb,
                "temperature_celsius": temperature,
                "power_watts": power
            }
            
        except Exception as e:
            logger.debug(f"Error collecting GPU metrics: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "utilization_percent": 0,
                "memory_used_mb": 0,
                "temperature_celsius": 0,
                "power_watts": 0
            }
    
    def stop(self):
        """Stop metrics collection."""
        self.running = False
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """
        Get collected metrics.
        
        Returns:
            List of metrics dictionaries
        """
        with self.lock:
            return self.metrics.copy()


class LoadTester:
    """
    Load testing framework for NVIDIA T4 deployments.
    
    Simulates concurrent users and measures system performance
    under various load conditions.
    """
    
    def __init__(self, config: LoadTestConfig):
        """
        Initialize load tester.
        
        Args:
            config: Load test configuration
        """
        self.config = config
        self.results = []
        self.gpu_metrics = []
        self.metrics_collector = None
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Generate a unique test ID
        self.test_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Request templates
        self.request_templates = {
            "query": {
                "endpoint": "/api/v1/dataframes/query",
                "method": "POST",
                "data": {
                    "query": "SELECT * FROM DUMMY",
                    "limit": 10
                }
            },
            "generate": {
                "endpoint": "/api/v1/generate",
                "method": "POST",
                "data": {
                    "prompt": "What is SAP HANA Cloud?",
                    "max_tokens": 100
                }
            },
            "embeddings": {
                "endpoint": "/api/v1/vectorstore/embed",
                "method": "POST",
                "data": {
                    "text": "How does SAP HANA Cloud integrate with generative AI?",
                    "model": "default"
                }
            },
            "vectorsearch": {
                "endpoint": "/api/v1/vectorstore/query",
                "method": "POST",
                "data": {
                    "query": "time series forecasting",
                    "top_k": 3
                }
            }
        }
        
        logger.info(f"Initialized load tester for {config.host}:{config.port}")
    
    def run_test(self, concurrent_users: int) -> List[RequestResult]:
        """
        Run a load test with the specified number of concurrent users.
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            
        Returns:
            List of request results
        """
        logger.info(f"Starting load test with {concurrent_users} concurrent users")
        
        self.start_time = time.time()
        end_time = self.start_time + self.config.duration_seconds
        results = []
        
        # Start GPU metrics collector
        self.metrics_collector = GPUMetricsCollector(
            host=self.config.host,
            port=self.config.gpu_metrics_port
        )
        self.metrics_collector.start()
        
        # Create thread pool for concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit initial batch of users
            futures = []
            for _ in range(concurrent_users):
                futures.append(executor.submit(self._user_session, end_time, results))
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"User session error: {e}")
        
        # Stop metrics collector and get metrics
        if self.metrics_collector:
            self.metrics_collector.stop()
            self.metrics_collector.join()
            self.gpu_metrics = self.metrics_collector.get_metrics()
        
        self.end_time = time.time()
        
        logger.info(f"Load test completed. {len(results)} requests processed.")
        
        return results
    
    def _user_session(self, end_time: float, results: List[RequestResult]):
        """
        Simulate a user session making requests until the end time.
        
        Args:
            end_time: Timestamp when the test should end
            results: List to collect request results
        """
        session = requests.Session()
        
        # Set up headers
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        
        while time.time() < end_time:
            # Choose a request type based on distribution
            request_type = self._select_request_type()
            
            # Get request template
            template = self.request_templates.get(request_type)
            if not template:
                logger.warning(f"Unknown request type: {request_type}")
                continue
            
            # Prepare request
            url = f"http://{self.config.host}:{self.config.port}{template['endpoint']}"
            method = template["method"]
            data = template["data"].copy()
            
            # Add some randomization to data
            if request_type == "query":
                data["limit"] = random.randint(5, 20)
            elif request_type == "generate":
                prompts = [
                    "What is SAP HANA Cloud?",
                    "Explain time series forecasting in SAP HANA.",
                    "How can I use vector search in SAP HANA?",
                    "Describe SAP HANA's integration with generative AI.",
                    "What are the benefits of using SAP HANA with NVIDIA GPUs?"
                ]
                data["prompt"] = random.choice(prompts)
                data["max_tokens"] = random.randint(50, 200)
            elif request_type == "vectorsearch":
                queries = [
                    "time series forecasting",
                    "machine learning",
                    "vector embeddings",
                    "generative AI",
                    "database optimization"
                ]
                data["query"] = random.choice(queries)
                data["top_k"] = random.randint(1, 5)
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Execute request
            result = self._execute_request(session, url, method, data, headers, request_type, request_id)
            
            # Add to results
            results.append(result)
            
            # Think time between requests
            think_time = random.randint(self.config.think_time_ms[0], self.config.think_time_ms[1]) / 1000.0
            time.sleep(think_time)
    
    def _select_request_type(self) -> str:
        """
        Select a request type based on the configured distribution.
        
        Returns:
            Selected request type
        """
        # Get distribution
        distribution = self.config.request_distribution
        
        # Generate random number
        rand = random.random()
        
        # Select request type
        cumulative = 0.0
        for request_type, probability in distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return request_type
        
        # Fallback to first request type
        return list(distribution.keys())[0]
    
    def _execute_request(self, 
                        session: requests.Session, 
                        url: str, 
                        method: str, 
                        data: Dict[str, Any],
                        headers: Dict[str, str],
                        request_type: str,
                        request_id: str) -> RequestResult:
        """
        Execute a single request and return the result.
        
        Args:
            session: Requests session
            url: Request URL
            method: HTTP method
            data: Request data
            headers: Request headers
            request_type: Type of request
            request_id: Unique request ID
            
        Returns:
            Request result
        """
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = session.get(url, params=data, headers=headers, timeout=30)
            else:
                response = session.post(url, json=data, headers=headers, timeout=30)
            
            end_time = time.time()
            
            # Determine success
            success = 200 <= response.status_code < 300
            
            # Extract response size
            response_size = len(response.content)
            
            # Extract tokens generated for text generation requests
            tokens_generated = 0
            if request_type == "generate" and success:
                try:
                    response_json = response.json()
                    generated_text = response_json.get("text", "")
                    tokens_generated = len(generated_text.split())
                except:
                    pass
            
            return RequestResult(
                request_type=request_type,
                start_time=start_time,
                end_time=end_time,
                status_code=response.status_code,
                success=success,
                response_size=response_size,
                tokens_generated=tokens_generated,
                request_id=request_id
            )
            
        except Exception as e:
            end_time = time.time()
            
            return RequestResult(
                request_type=request_type,
                start_time=start_time,
                end_time=end_time,
                status_code=0,
                success=False,
                error=str(e),
                request_id=request_id
            )
    
    def run_load_test_sequence(self) -> Dict[str, Any]:
        """
        Run the complete load test sequence with increasing user counts.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Starting load test sequence with {len(self.config.concurrent_users)} user levels")
        
        all_results = []
        
        for user_count in self.config.concurrent_users:
            logger.info(f"Testing with {user_count} concurrent users")
            
            # Run test with this user count
            results = self.run_test(user_count)
            
            # Store results
            all_results.extend(results)
            
            # Wait between tests
            if user_count != self.config.concurrent_users[-1]:
                logger.info(f"Waiting for 10 seconds before next test...")
                time.sleep(10)
        
        # Process and save results
        test_results = self._process_results(all_results)
        
        # Save results
        self._save_results(test_results)
        
        # Generate report
        self._generate_report(test_results)
        
        return test_results
    
    def _process_results(self, results: List[RequestResult]) -> Dict[str, Any]:
        """
        Process raw results into aggregate metrics.
        
        Args:
            results: List of request results
            
        Returns:
            Dictionary with processed metrics
        """
        # Basic stats
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Response times
        response_times = [r.duration_ms for r in results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        if response_times:
            sorted_times = sorted(response_times)
            p50_response_time = sorted_times[int(len(sorted_times) * 0.5)]
            p90_response_time = sorted_times[int(len(sorted_times) * 0.9)]
            p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            p50_response_time = p90_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
        
        # Group by request type
        request_types = {}
        for result in results:
            if result.request_type not in request_types:
                request_types[result.request_type] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "response_times": []
                }
            
            request_types[result.request_type]["count"] += 1
            if result.success:
                request_types[result.request_type]["successful"] += 1
            else:
                request_types[result.request_type]["failed"] += 1
            
            request_types[result.request_type]["response_times"].append(result.duration_ms)
        
        # Calculate stats for each request type
        for req_type, stats in request_types.items():
            stats["success_rate"] = (stats["successful"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            
            if stats["response_times"]:
                stats["avg_response_time"] = sum(stats["response_times"]) / len(stats["response_times"])
                
                sorted_times = sorted(stats["response_times"])
                stats["p50_response_time"] = sorted_times[int(len(sorted_times) * 0.5)]
                stats["p90_response_time"] = sorted_times[int(len(sorted_times) * 0.9)]
                stats["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
                stats["min_response_time"] = min(stats["response_times"])
                stats["max_response_time"] = max(stats["response_times"])
            else:
                stats["avg_response_time"] = 0
                stats["p50_response_time"] = 0
                stats["p90_response_time"] = 0
                stats["p95_response_time"] = 0
                stats["min_response_time"] = 0
                stats["max_response_time"] = 0
            
            # Clean up raw data to reduce size
            del stats["response_times"]
        
        # Calculate throughput
        if self.start_time and self.end_time:
            test_duration = self.end_time - self.start_time
            requests_per_second = total_requests / test_duration if test_duration > 0 else 0
            
            # Tokens throughput for generate requests
            total_tokens = sum(r.tokens_generated for r in results if r.request_type == "generate")
            tokens_per_second = total_tokens / test_duration if test_duration > 0 else 0
        else:
            test_duration = 0
            requests_per_second = 0
            tokens_per_second = 0
        
        # Process GPU metrics
        gpu_metrics_summary = self._process_gpu_metrics()
        
        # Build final results
        return {
            "test_id": self.test_id,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "concurrent_users": self.config.concurrent_users,
                "duration_seconds": self.config.duration_seconds,
                "request_types": self.config.request_types,
                "request_distribution": self.config.request_distribution
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "test_duration": test_duration,
                "requests_per_second": requests_per_second,
                "tokens_per_second": tokens_per_second,
                "avg_response_time": avg_response_time,
                "p50_response_time": p50_response_time,
                "p90_response_time": p90_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "request_types": request_types,
            "gpu_metrics": gpu_metrics_summary,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _process_gpu_metrics(self) -> Dict[str, Any]:
        """
        Process GPU metrics into summary statistics.
        
        Returns:
            Dictionary with GPU metrics summary
        """
        if not self.gpu_metrics:
            return {
                "available": False,
                "message": "No GPU metrics collected"
            }
        
        # Extract metrics over time
        timestamps = []
        utilization = []
        memory = []
        temperature = []
        power = []
        
        for metric in self.gpu_metrics:
            timestamps.append(metric["timestamp"])
            utilization.append(metric["utilization_percent"])
            memory.append(metric["memory_used_mb"])
            temperature.append(metric["temperature_celsius"])
            power.append(metric["power_watts"])
        
        # Calculate statistics
        avg_utilization = sum(utilization) / len(utilization) if utilization else 0
        max_utilization = max(utilization) if utilization else 0
        min_utilization = min(utilization) if utilization else 0
        
        avg_memory = sum(memory) / len(memory) if memory else 0
        max_memory = max(memory) if memory else 0
        
        avg_temperature = sum(temperature) / len(temperature) if temperature else 0
        max_temperature = max(temperature) if temperature else 0
        
        avg_power = sum(power) / len(power) if power else 0
        max_power = max(power) if power else 0
        
        return {
            "available": True,
            "samples": len(self.gpu_metrics),
            "avg_utilization_percent": avg_utilization,
            "max_utilization_percent": max_utilization,
            "min_utilization_percent": min_utilization,
            "avg_memory_used_mb": avg_memory,
            "max_memory_used_mb": max_memory,
            "avg_temperature_celsius": avg_temperature,
            "max_temperature_celsius": max_temperature,
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "metrics_over_time": {
                "timestamps": timestamps,
                "utilization_percent": utilization,
                "memory_used_mb": memory,
                "temperature_celsius": temperature,
                "power_watts": power
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save test results to disk.
        
        Args:
            results: Test results dictionary
        """
        # Save JSON results
        results_file = os.path.join(self.config.output_dir, f"load-test-{self.test_id}.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate graphs if plotting is available
        if PLOTTING_AVAILABLE:
            self._generate_graphs(results)
    
    def _generate_graphs(self, results: Dict[str, Any]):
        """
        Generate performance graphs from test results.
        
        Args:
            results: Test results dictionary
        """
        output_dir = os.path.join(self.config.output_dir, f"graphs-{self.test_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # GPU utilization over time
        if results["gpu_metrics"]["available"]:
            metrics = results["gpu_metrics"]["metrics_over_time"]
            
            # Convert timestamps to relative seconds
            start_time = metrics["timestamps"][0] if metrics["timestamps"] else 0
            relative_time = [(t - start_time) for t in metrics["timestamps"]]
            
            # GPU utilization
            plt.figure(figsize=(10, 6))
            plt.plot(relative_time, metrics["utilization_percent"])
            plt.title("GPU Utilization Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Utilization (%)")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "gpu_utilization.png"))
            plt.close()
            
            # GPU memory
            plt.figure(figsize=(10, 6))
            plt.plot(relative_time, metrics["memory_used_mb"])
            plt.title("GPU Memory Usage Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Memory Used (MB)")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "gpu_memory.png"))
            plt.close()
            
            # GPU temperature
            plt.figure(figsize=(10, 6))
            plt.plot(relative_time, metrics["temperature_celsius"])
            plt.title("GPU Temperature Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Temperature (°C)")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "gpu_temperature.png"))
            plt.close()
            
            # GPU power
            plt.figure(figsize=(10, 6))
            plt.plot(relative_time, metrics["power_watts"])
            plt.title("GPU Power Usage Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Power (Watts)")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "gpu_power.png"))
            plt.close()
        
        # Response time by request type
        req_types = list(results["request_types"].keys())
        avg_times = [results["request_types"][req_type]["avg_response_time"] for req_type in req_types]
        
        plt.figure(figsize=(10, 6))
        plt.bar(req_types, avg_times)
        plt.title("Average Response Time by Request Type")
        plt.xlabel("Request Type")
        plt.ylabel("Response Time (ms)")
        plt.grid(True, axis="y")
        plt.savefig(os.path.join(output_dir, "response_time_by_type.png"))
        plt.close()
        
        # Success rate by request type
        success_rates = [results["request_types"][req_type]["success_rate"] for req_type in req_types]
        
        plt.figure(figsize=(10, 6))
        plt.bar(req_types, success_rates)
        plt.title("Success Rate by Request Type")
        plt.xlabel("Request Type")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        plt.grid(True, axis="y")
        plt.savefig(os.path.join(output_dir, "success_rate_by_type.png"))
        plt.close()
        
        logger.info(f"Graphs saved to {output_dir}")
    
    def _generate_report(self, results: Dict[str, Any]):
        """
        Generate HTML report from test results.
        
        Args:
            results: Test results dictionary
        """
        report_file = os.path.join(self.config.output_dir, f"load-test-report-{self.test_id}.html")
        
        # Check if graphs were generated
        graphs_dir = os.path.join(self.config.output_dir, f"graphs-{self.test_id}")
        has_graphs = os.path.exists(graphs_dir) and PLOTTING_AVAILABLE
        
        # Create HTML content
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Load Test Report</title>
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
        .graphs {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .graph {{
            flex-basis: 48%;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .graph img {{
            max-width: 100%;
            height: auto;
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
            background-color: #f8f9fa;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .config {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .config-item {{
            margin-bottom: 10px;
        }}
        .config-item label {{
            font-weight: bold;
            display: inline-block;
            width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Load Test Report</h1>
            <p>Test ID: {results['test_id']} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="section">
            <h2>Test Configuration</h2>
            <div class="config">
                <div class="config-item">
                    <label>Host:</label>
                    <span>{results['config']['host']}:{results['config']['port']}</span>
                </div>
                <div class="config-item">
                    <label>Concurrent Users:</label>
                    <span>{', '.join(map(str, results['config']['concurrent_users']))}</span>
                </div>
                <div class="config-item">
                    <label>Duration:</label>
                    <span>{results['config']['duration_seconds']} seconds</span>
                </div>
                <div class="config-item">
                    <label>Request Types:</label>
                    <span>{', '.join(results['config']['request_types'])}</span>
                </div>
                <div class="config-item">
                    <label>Request Distribution:</label>
                    <span>{', '.join([f"{k}: {v:.1%}" for k, v in results['config']['request_distribution'].items()])}</span>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <div class="summary">
                <div class="summary-box">
                    <h2>Total Requests</h2>
                    <div class="value">{results['summary']['total_requests']}</div>
                </div>
                <div class="summary-box">
                    <h2>Success Rate</h2>
                    <div class="value">{results['summary']['success_rate']:.1f}%</div>
                </div>
                <div class="summary-box">
                    <h2>Requests/Second</h2>
                    <div class="value">{results['summary']['requests_per_second']:.1f}</div>
                </div>
                <div class="summary-box">
                    <h2>Tokens/Second</h2>
                    <div class="value">{results['summary']['tokens_per_second']:.1f}</div>
                </div>
                <div class="summary-box">
                    <h2>Avg Response Time</h2>
                    <div class="value">{results['summary']['avg_response_time']:.1f} ms</div>
                </div>
                <div class="summary-box">
                    <h2>P95 Response Time</h2>
                    <div class="value">{results['summary']['p95_response_time']:.1f} ms</div>
                </div>
                <div class="summary-box">
                    <h2>Max Response Time</h2>
                    <div class="value">{results['summary']['max_response_time']:.1f} ms</div>
                </div>
                <div class="summary-box">
                    <h2>Test Duration</h2>
                    <div class="value">{results['summary']['test_duration']:.1f} s</div>
                </div>
            </div>
        </div>
"""
        
        # Add GPU metrics if available
        if results["gpu_metrics"]["available"]:
            html += f"""
        <div class="section">
            <h2>GPU Performance</h2>
            <div class="summary">
                <div class="summary-box">
                    <h2>Avg GPU Utilization</h2>
                    <div class="value">{results['gpu_metrics']['avg_utilization_percent']:.1f}%</div>
                </div>
                <div class="summary-box">
                    <h2>Max GPU Utilization</h2>
                    <div class="value">{results['gpu_metrics']['max_utilization_percent']:.1f}%</div>
                </div>
                <div class="summary-box">
                    <h2>Avg Memory Usage</h2>
                    <div class="value">{results['gpu_metrics']['avg_memory_used_mb']:.0f} MB</div>
                </div>
                <div class="summary-box">
                    <h2>Max Memory Usage</h2>
                    <div class="value">{results['gpu_metrics']['max_memory_used_mb']:.0f} MB</div>
                </div>
                <div class="summary-box">
                    <h2>Avg Temperature</h2>
                    <div class="value">{results['gpu_metrics']['avg_temperature_celsius']:.1f}°C</div>
                </div>
                <div class="summary-box">
                    <h2>Max Temperature</h2>
                    <div class="value">{results['gpu_metrics']['max_temperature_celsius']:.1f}°C</div>
                </div>
                <div class="summary-box">
                    <h2>Avg Power</h2>
                    <div class="value">{results['gpu_metrics']['avg_power_watts']:.1f} W</div>
                </div>
                <div class="summary-box">
                    <h2>Max Power</h2>
                    <div class="value">{results['gpu_metrics']['max_power_watts']:.1f} W</div>
                </div>
            </div>
        </div>
"""
        
        # Add graphs if available
        if has_graphs:
            html += """
        <div class="section">
            <h2>Performance Graphs</h2>
            <div class="graphs">
"""
            
            # GPU graphs
            if results["gpu_metrics"]["available"]:
                html += f"""
                <div class="graph">
                    <h3>GPU Utilization</h3>
                    <img src="graphs-{self.test_id}/gpu_utilization.png" alt="GPU Utilization">
                </div>
                <div class="graph">
                    <h3>GPU Memory Usage</h3>
                    <img src="graphs-{self.test_id}/gpu_memory.png" alt="GPU Memory Usage">
                </div>
                <div class="graph">
                    <h3>GPU Temperature</h3>
                    <img src="graphs-{self.test_id}/gpu_temperature.png" alt="GPU Temperature">
                </div>
                <div class="graph">
                    <h3>GPU Power Usage</h3>
                    <img src="graphs-{self.test_id}/gpu_power.png" alt="GPU Power Usage">
                </div>
"""
            
            # Request type graphs
            html += f"""
                <div class="graph">
                    <h3>Response Time by Request Type</h3>
                    <img src="graphs-{self.test_id}/response_time_by_type.png" alt="Response Time by Request Type">
                </div>
                <div class="graph">
                    <h3>Success Rate by Request Type</h3>
                    <img src="graphs-{self.test_id}/success_rate_by_type.png" alt="Success Rate by Request Type">
                </div>
            </div>
        </div>
"""
        
        # Add request type details
        html += """
        <div class="section">
            <h2>Request Type Details</h2>
            <table>
                <tr>
                    <th>Request Type</th>
                    <th>Count</th>
                    <th>Success Rate</th>
                    <th>Avg Response Time</th>
                    <th>P50 Response Time</th>
                    <th>P90 Response Time</th>
                    <th>Min Response Time</th>
                    <th>Max Response Time</th>
                </tr>
"""
        
        for req_type, stats in results["request_types"].items():
            html += f"""
                <tr>
                    <td>{req_type}</td>
                    <td>{stats['count']}</td>
                    <td>{stats['success_rate']:.1f}%</td>
                    <td>{stats['avg_response_time']:.1f} ms</td>
                    <td>{stats['p50_response_time']:.1f} ms</td>
                    <td>{stats['p90_response_time']:.1f} ms</td>
                    <td>{stats['min_response_time']:.1f} ms</td>
                    <td>{stats['max_response_time']:.1f} ms</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML to file
        with open(report_file, "w") as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {report_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load Testing Tool for NVIDIA T4 Deployments"
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
        "--api-key",
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--gpu-metrics-port",
        type=int,
        default=9835,
        help="Port for NVIDIA GPU metrics exporter (default: 9835)"
    )
    
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Concurrent user counts to test (default: 1 5 10 20)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds for each user count (default: 60)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="load-test-results",
        help="Directory to store test results (default: load-test-results)"
    )
    
    parser.add_argument(
        "--request-types",
        nargs="+",
        default=["query", "generate", "embeddings", "vectorsearch"],
        help="Request types to test (default: query generate embeddings vectorsearch)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the load testing tool."""
    args = parse_arguments()
    
    # Create config
    config = LoadTestConfig(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        gpu_metrics_port=args.gpu_metrics_port,
        concurrent_users=args.users,
        duration_seconds=args.duration,
        request_types=args.request_types,
        output_dir=args.output_dir
    )
    
    # Initialize load tester
    tester = LoadTester(config)
    
    # Run load test sequence
    results = tester.run_load_test_sequence()
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"NVIDIA T4 LOAD TEST RESULTS: {args.host}:{args.port}")
    print("=" * 80)
    print(f"Total Requests: {results['summary']['total_requests']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Requests per Second: {results['summary']['requests_per_second']:.1f}")
    print(f"Tokens per Second: {results['summary']['tokens_per_second']:.1f}")
    print(f"Average Response Time: {results['summary']['avg_response_time']:.1f} ms")
    print(f"P95 Response Time: {results['summary']['p95_response_time']:.1f} ms")
    
    if results["gpu_metrics"]["available"]:
        print("\nGPU Performance:")
        print(f"Average Utilization: {results['gpu_metrics']['avg_utilization_percent']:.1f}%")
        print(f"Maximum Utilization: {results['gpu_metrics']['max_utilization_percent']:.1f}%")
        print(f"Average Memory Usage: {results['gpu_metrics']['avg_memory_used_mb']:.0f} MB")
    
    print(f"\nFull report: {os.path.join(config.output_dir, f'load-test-report-{tester.test_id}.html')}")
    print("=" * 80)


if __name__ == "__main__":
    main()