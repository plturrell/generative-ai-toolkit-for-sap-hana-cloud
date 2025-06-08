#!/usr/bin/env python3
"""
TensorRT T4 GPU Integration Test Script

This script tests the TensorRT integration on NVIDIA T4 GPUs for the
SAP HANA Generative AI Toolkit. It validates embedding generation,
vector search, and benchmarks performance metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import requests
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "api_base_url": "http://localhost:8000",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64],
    "enable_tensorrt": True,
    "test_timeout": 300,
    "auth": {
        "enabled": False,
        "api_key": ""
    },
    "results_dir": "test_results",
}

class T4TensorRTTest:
    """Test suite for T4 GPU TensorRT integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test suite.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.api_base_url = config.get("api_base_url", DEFAULT_CONFIG["api_base_url"])
        self.embedding_model = config.get("embedding_model", DEFAULT_CONFIG["embedding_model"])
        self.precision = config.get("precision", DEFAULT_CONFIG["precision"])
        self.batch_sizes = config.get("batch_sizes", DEFAULT_CONFIG["batch_sizes"])
        self.enable_tensorrt = config.get("enable_tensorrt", DEFAULT_CONFIG["enable_tensorrt"])
        self.test_timeout = config.get("test_timeout", DEFAULT_CONFIG["test_timeout"])
        self.auth = config.get("auth", DEFAULT_CONFIG["auth"])
        self.results_dir = config.get("results_dir", DEFAULT_CONFIG["results_dir"])
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test results
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            },
            "performance": {
                "tensorrt_speedup": None,
                "optimal_batch_size": None,
                "memory_usage": None,
                "throughput": None
            }
        }
        
        # Prepare headers for API requests
        self.headers = {}
        if self.auth["enabled"] and self.auth.get("api_key"):
            self.headers["Authorization"] = f"Bearer {self.auth['api_key']}"
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all T4 TensorRT tests.
        
        Returns:
            Test results
        """
        tests = [
            self.test_gpu_detection,
            self.test_tensorrt_availability,
            self.test_embedding_generation,
            self.test_embedding_batch_performance,
            self.test_adaptive_batch_sizing,
            self.test_vector_search
        ]
        
        for test_func in tests:
            try:
                test_name = test_func.__name__
                logger.info(f"Running test: {test_name}")
                
                start_time = time.time()
                result = test_func()
                elapsed = time.time() - start_time
                
                result["test_name"] = test_name
                result["elapsed_time"] = elapsed
                
                self.results["tests"].append(result)
                
                # Update summary
                self.results["summary"]["total"] += 1
                if result["status"] == "passed":
                    self.results["summary"]["passed"] += 1
                elif result["status"] == "failed":
                    self.results["summary"]["failed"] += 1
                else:
                    self.results["summary"]["errors"] += 1
                
                logger.info(f"Test {test_name} {result['status']} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Error running test {test_func.__name__}: {str(e)}")
                self.results["tests"].append({
                    "test_name": test_func.__name__,
                    "status": "error",
                    "error": str(e),
                    "elapsed_time": time.time() - start_time
                })
                self.results["summary"]["total"] += 1
                self.results["summary"]["errors"] += 1
        
        # Save results to file
        self.save_results()
        
        return self.results
    
    def test_gpu_detection(self) -> Dict[str, Any]:
        """
        Test GPU detection on the server.
        
        Returns:
            Test result
        """
        result = {
            "name": "GPU Detection",
            "status": "failed",
            "details": {}
        }
        
        try:
            # Check if GPU is available on the server
            url = f"{self.api_base_url}/api/gpu_info"
            response = requests.get(url, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            gpu_info = response.json()
            result["details"]["gpu_info"] = gpu_info
            
            # Check if it's a T4 GPU
            if gpu_info.get("is_t4", False):
                result["status"] = "passed"
                result["message"] = "T4 GPU detected"
            else:
                result["message"] = f"No T4 GPU detected, found: {gpu_info.get('name', 'unknown')}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error checking GPU: {str(e)}"
        
        return result
    
    def test_tensorrt_availability(self) -> Dict[str, Any]:
        """
        Test TensorRT availability on the server.
        
        Returns:
            Test result
        """
        result = {
            "name": "TensorRT Availability",
            "status": "failed",
            "details": {}
        }
        
        try:
            # Check if TensorRT is available on the server
            url = f"{self.api_base_url}/api/gpu_info"
            response = requests.get(url, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            gpu_info = response.json()
            result["details"]["gpu_info"] = gpu_info
            
            # Check TensorRT availability
            if gpu_info.get("tensorrt_available", False):
                result["status"] = "passed"
                result["message"] = f"TensorRT available, version: {gpu_info.get('tensorrt_version', 'unknown')}"
                result["details"]["tensorrt_version"] = gpu_info.get("tensorrt_version")
            else:
                result["message"] = "TensorRT not available"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error checking TensorRT: {str(e)}"
        
        return result
    
    def test_embedding_generation(self) -> Dict[str, Any]:
        """
        Test embedding generation with TensorRT.
        
        Returns:
            Test result
        """
        result = {
            "name": "Embedding Generation",
            "status": "failed",
            "details": {}
        }
        
        try:
            # Test text to embed
            text = "SAP HANA is a high-performance in-memory database that provides fast data processing for complex analytics and transactions."
            
            # Generate embeddings with TensorRT
            url = f"{self.api_base_url}/api/embeddings"
            payload = {
                "texts": [text],
                "model_name": self.embedding_model,
                "use_tensorrt": self.enable_tensorrt,
                "precision": self.precision
            }
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            tensorrt_result = response.json()
            result["details"]["tensorrt"] = tensorrt_result
            
            # Generate embeddings without TensorRT for comparison
            payload["use_tensorrt"] = False
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            pytorch_result = response.json()
            result["details"]["pytorch"] = pytorch_result
            
            # Calculate speedup
            tensorrt_time = tensorrt_result.get("processing_time_ms", 0)
            pytorch_time = pytorch_result.get("processing_time_ms", 0)
            
            if pytorch_time > 0 and tensorrt_time > 0:
                speedup = pytorch_time / tensorrt_time
                result["details"]["speedup"] = speedup
                self.results["performance"]["tensorrt_speedup"] = speedup
                
                # Check if embeddings are similar
                if tensorrt_result.get("embeddings") and pytorch_result.get("embeddings"):
                    tensorrt_embedding = np.array(tensorrt_result["embeddings"][0])
                    pytorch_embedding = np.array(pytorch_result["embeddings"][0])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(tensorrt_embedding, pytorch_embedding) / (
                        np.linalg.norm(tensorrt_embedding) * np.linalg.norm(pytorch_embedding)
                    )
                    result["details"]["similarity"] = similarity
                    
                    # Validate results
                    if similarity > 0.98:  # Allow for small differences due to optimization
                        result["status"] = "passed"
                        result["message"] = f"Embedding generation successful, speedup: {speedup:.2f}x, similarity: {similarity:.4f}"
                    else:
                        result["message"] = f"Embedding similarity too low: {similarity:.4f}"
                else:
                    result["message"] = "Missing embeddings in response"
            else:
                result["message"] = "Invalid timing data"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error generating embeddings: {str(e)}"
        
        return result
    
    def test_embedding_batch_performance(self) -> Dict[str, Any]:
        """
        Test embedding generation with different batch sizes.
        
        Returns:
            Test result
        """
        result = {
            "name": "Batch Performance",
            "status": "failed",
            "details": {
                "batch_results": []
            }
        }
        
        try:
            # Generate sample texts
            sample_text = "SAP HANA is a high-performance in-memory database that provides fast data processing."
            
            best_throughput = 0
            optimal_batch_size = 1
            
            for batch_size in self.batch_sizes:
                texts = [f"{sample_text} Sample {i}." for i in range(batch_size)]
                
                # Generate embeddings with TensorRT
                url = f"{self.api_base_url}/api/embeddings"
                payload = {
                    "texts": texts,
                    "model_name": self.embedding_model,
                    "use_tensorrt": self.enable_tensorrt,
                    "precision": self.precision
                }
                
                start_time = time.time()
                response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
                response.raise_for_status()
                
                tensorrt_result = response.json()
                total_time = time.time() - start_time
                
                # Calculate throughput (samples/second)
                throughput = batch_size / total_time
                
                batch_result = {
                    "batch_size": batch_size,
                    "processing_time_ms": tensorrt_result.get("processing_time_ms", 0),
                    "total_time_ms": total_time * 1000,
                    "throughput": throughput
                }
                
                result["details"]["batch_results"].append(batch_result)
                
                # Update optimal batch size
                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
            
            # Store optimal batch size in results
            result["details"]["optimal_batch_size"] = optimal_batch_size
            result["details"]["best_throughput"] = best_throughput
            self.results["performance"]["optimal_batch_size"] = optimal_batch_size
            self.results["performance"]["throughput"] = best_throughput
            
            result["status"] = "passed"
            result["message"] = f"Batch performance test completed, optimal batch size: {optimal_batch_size}, throughput: {best_throughput:.2f} samples/sec"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error testing batch performance: {str(e)}"
        
        return result
    
    def test_adaptive_batch_sizing(self) -> Dict[str, Any]:
        """
        Test adaptive batch sizing functionality.
        
        Returns:
            Test result
        """
        result = {
            "name": "Adaptive Batch Sizing",
            "status": "failed",
            "details": {}
        }
        
        try:
            # Check if adaptive batch sizing is enabled
            url = f"{self.api_base_url}/api/config/batch_sizing"
            response = requests.get(url, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            batch_config = response.json()
            result["details"]["config"] = batch_config
            
            # Verify T4 optimization is enabled
            if batch_config.get("t4_optimized", False):
                # Test with multiple batch sizes
                url = f"{self.api_base_url}/api/embeddings/adaptive_test"
                
                # Create test data with various lengths
                test_data = [
                    {"texts": ["Short text"] * 10, "expected_batch": "small"},
                    {"texts": ["Medium length text with some more words to process"] * 20, "expected_batch": "medium"},
                    {"texts": ["Very long text " + "with lots of words " * 20] * 10, "expected_batch": "small"}
                ]
                
                results = []
                for data in test_data:
                    # Get embeddings with adaptive batch sizing
                    payload = {
                        "texts": data["texts"],
                        "model_name": self.embedding_model,
                        "use_tensorrt": self.enable_tensorrt,
                        "precision": self.precision,
                        "enable_adaptive_batch": True
                    }
                    
                    response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
                    response.raise_for_status()
                    
                    adaptive_result = response.json()
                    
                    # Get metrics from the response
                    test_result = {
                        "input_length": len(data["texts"]),
                        "text_length": len(data["texts"][0]),
                        "expected_behavior": data["expected_batch"],
                        "batch_size": adaptive_result.get("metadata", {}).get("batch_size"),
                        "processing_time_ms": adaptive_result.get("processing_time_ms"),
                        "adaptive_enabled": adaptive_result.get("metadata", {}).get("adaptive_batch_sizing", False)
                    }
                    
                    results.append(test_result)
                
                # Record test results
                result["details"]["test_results"] = results
                
                # Get performance statistics
                url = f"{self.api_base_url}/api/metrics/batch_performance"
                response = requests.get(url, headers=self.headers, timeout=self.test_timeout)
                response.raise_for_status()
                
                performance_stats = response.json()
                result["details"]["performance_stats"] = performance_stats
                
                # Check that we have some optimal batch sizes recorded
                if performance_stats.get("optimal_batch_sizes"):
                    result["status"] = "passed"
                    result["message"] = "Adaptive batch sizing test successful"
                else:
                    result["message"] = "No optimal batch sizes recorded"
            else:
                result["message"] = "T4 optimization is not enabled for batch sizing"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error testing adaptive batch sizing: {str(e)}"
        
        return result
        
    def test_vector_search(self) -> Dict[str, Any]:
        """
        Test vector search functionality.
        
        Returns:
            Test result
        """
        result = {
            "name": "Vector Search",
            "status": "failed",
            "details": {}
        }
        
        try:
            # Create test documents
            documents = [
                "SAP HANA is a high-performance in-memory database.",
                "TensorRT accelerates deep learning inference.",
                "T4 GPUs provide efficient AI computing.",
                "Vector search enables semantic retrieval of information.",
                "Embeddings represent text in high-dimensional space."
            ]
            
            # Index documents
            url = f"{self.api_base_url}/api/vectorstore/index"
            payload = {
                "documents": documents,
                "collection_name": "test_collection",
                "model_name": self.embedding_model,
                "use_tensorrt": self.enable_tensorrt
            }
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            index_result = response.json()
            result["details"]["index"] = index_result
            
            # Perform search
            query = "How does GPU acceleration work?"
            
            url = f"{self.api_base_url}/api/vectorstore/search"
            payload = {
                "query": query,
                "collection_name": "test_collection",
                "model_name": self.embedding_model,
                "use_tensorrt": self.enable_tensorrt,
                "top_k": 2
            }
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.test_timeout)
            response.raise_for_status()
            
            search_result = response.json()
            result["details"]["search"] = search_result
            
            # Validate search results
            if search_result.get("matches") and len(search_result["matches"]) > 0:
                result["status"] = "passed"
                result["message"] = "Vector search test successful"
            else:
                result["message"] = "No search results returned"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"Error testing vector search: {str(e)}"
        
        return result
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.results_dir, f"t4-tensorrt-test-{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
        
        # Generate HTML report
        self.generate_html_report(timestamp)
    
    def generate_html_report(self, timestamp: str):
        """
        Generate HTML report from test results.
        
        Args:
            timestamp: Timestamp for the report file
        """
        html_filename = os.path.join(self.results_dir, f"t4-tensorrt-report-{timestamp}.html")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>T4 GPU TensorRT Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #0066cc; }}
                h2 {{ color: #333; margin-top: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .test {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background-color: #f0f0f0; padding: 10px; overflow-x: auto; }}
                .performance {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>T4 GPU TensorRT Test Report</h1>
            <div class='summary'>
                <h2>Summary</h2>
                <p><strong>Timestamp:</strong> {self.results["timestamp"]}</p>
                <p><strong>API URL:</strong> {self.api_base_url}</p>
                <p><strong>Model:</strong> {self.embedding_model}</p>
                <p><strong>Precision:</strong> {self.precision}</p>
                <p><strong>Total Tests:</strong> {self.results["summary"]["total"]}</p>
                <p><strong>Passed:</strong> <span class='passed'>{self.results["summary"]["passed"]}</span></p>
                <p><strong>Failed:</strong> <span class='failed'>{self.results["summary"]["failed"]}</span></p>
                <p><strong>Errors:</strong> <span class='error'>{self.results["summary"]["errors"]}</span></p>
            </div>
            
            <div class='performance'>
                <h2>Performance Metrics</h2>
                <p><strong>TensorRT Speedup:</strong> {self.results["performance"]["tensorrt_speedup"]:.2f}x</p>
                <p><strong>Optimal Batch Size:</strong> {self.results["performance"]["optimal_batch_size"]}</p>
                <p><strong>Best Throughput:</strong> {self.results["performance"]["throughput"]:.2f} samples/sec</p>
            </div>
            
            <h2>Test Results</h2>
        """
        
        for test in self.results["tests"]:
            html += f"""
            <div class='test'>
                <h3>{test.get("name", test.get("test_name", "Unknown"))}</h3>
                <p><strong>Status:</strong> <span class='{test["status"]}'>{test["status"]}</span></p>
                <p><strong>Message:</strong> {test.get("message", "")}</p>
                <p><strong>Time:</strong> {test.get("elapsed_time", 0):.2f}s</p>
            """
            
            if "details" in test:
                html += "<h4>Details</h4>"
                if "batch_results" in test["details"]:
                    html += """
                    <h5>Batch Performance</h5>
                    <table>
                        <tr>
                            <th>Batch Size</th>
                            <th>Processing Time (ms)</th>
                            <th>Total Time (ms)</th>
                            <th>Throughput (samples/sec)</th>
                        </tr>
                    """
                    
                    for batch in test["details"]["batch_results"]:
                        html += f"""
                        <tr>
                            <td>{batch["batch_size"]}</td>
                            <td>{batch["processing_time_ms"]:.2f}</td>
                            <td>{batch["total_time_ms"]:.2f}</td>
                            <td>{batch["throughput"]:.2f}</td>
                        </tr>
                        """
                    
                    html += "</table>"
                
                if "speedup" in test["details"]:
                    html += f"<p><strong>TensorRT Speedup:</strong> {test['details']['speedup']:.2f}x</p>"
                
                if "similarity" in test["details"]:
                    html += f"<p><strong>Embedding Similarity:</strong> {test['details']['similarity']:.4f}</p>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(html_filename, "w") as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {html_filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="T4 TensorRT Integration Test")
    parser.add_argument("--config", type=str, default="test_config.json", help="Path to config file")
    parser.add_argument("--url", type=str, help="API base URL")
    parser.add_argument("--model", type=str, help="Embedding model name")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], help="Precision mode")
    parser.add_argument("--output", type=str, help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                config.update(json.load(f))
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    # Override configuration with command line arguments
    if args.url:
        config["api_base_url"] = args.url
    if args.model:
        config["embedding_model"] = args.model
    if args.precision:
        config["precision"] = args.precision
    if args.output:
        config["results_dir"] = args.output
    
    # Run tests
    test_suite = T4TensorRTTest(config)
    results = test_suite.run_all_tests()
    
    # Return exit code based on test results
    if results["summary"]["errors"] > 0:
        return 2  # Error
    elif results["summary"]["failed"] > 0:
        return 1  # Failed
    else:
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())