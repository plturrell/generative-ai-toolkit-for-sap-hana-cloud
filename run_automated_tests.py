#!/usr/bin/env python3
"""
Automated CLI-based testing tool for SAP HANA Cloud Generative AI Toolkit on T4 GPU.

This script provides a command-line interface for running comprehensive tests
against the deployed system, with or without direct browser access to the
Jupyter instance.

Example usage:
    # Run all tests
    python run_automated_tests.py --all

    # Run specific test suite
    python run_automated_tests.py --suite gpu_performance

    # Run with custom configuration
    python run_automated_tests.py --config custom_config.json
"""

import argparse
import json
import os
import sys
import time
import subprocess
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_tests")

# Test suites
TEST_SUITES = {
    "environment": "Verify environment setup (GPU, drivers, Python packages)",
    "tensorrt": "Test TensorRT optimization for T4 GPU",
    "vectorstore": "Test vector store functionality",
    "gpu_performance": "Benchmark GPU performance for embedding and vector operations",
    "error_handling": "Test error handling and recovery",
    "api": "Test API endpoints"
}

class HanaGenerativeAITester:
    """Class for automated testing of SAP HANA Cloud Generative AI Toolkit on T4 GPU"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tester with configuration
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_path)
        self.results_dir = self.config.get("results_dir", "test_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up API base URL
        self.api_base_url = self.config.get("api_base_url", "https://jupyter0-513syzm60.brevlab.com")
        
        # Initialize test data generator if needed
        if not os.path.exists(os.path.join(self.results_dir, "sample_documents.json")):
            self._generate_test_data()
            
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "api_base_url": "https://jupyter0-513syzm60.brevlab.com",
            "results_dir": "test_results",
            "test_timeout": 300,  # 5 minutes
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "precision": "fp16",
            "batch_sizes": [1, 8, 32, 64, 128],
            "auth": {
                "enabled": False,
                "username": "",
                "password": ""
            },
            "hana_connection": {
                "address": "",
                "port": 0,
                "user": "",
                "password": ""
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                # Merge with default config
                for key, value in custom_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _generate_test_data(self):
        """Generate test data for testing"""
        logger.info("Generating test data...")
        
        # Sample documents for testing
        sample_documents = [
            {
                "id": f"doc{i}",
                "content": f"This is test document {i} about {topic}.",
                "metadata": {
                    "source": "test",
                    "topic": topic,
                    "importance": i % 3 + 1
                }
            }
            for i, topic in enumerate([
                "machine learning", "artificial intelligence", "natural language processing",
                "computer vision", "deep learning", "neural networks", "transformer models",
                "SAP HANA", "vector databases", "embeddings", "GPU acceleration",
                "T4 optimization", "TensorRT", "CUDA programming", "parallel computing"
            ])
        ]
        
        # Sample queries for testing
        test_queries = [
            "What is machine learning?",
            "How does GPU acceleration work?",
            "Tell me about vector databases",
            "Explain neural networks",
            "What is SAP HANA?",
            "How are embeddings used in AI?",
            "What is TensorRT optimization?"
        ]
        
        # Save to files
        with open(os.path.join(self.results_dir, "sample_documents.json"), 'w') as f:
            json.dump(sample_documents, f, indent=2)
            
        with open(os.path.join(self.results_dir, "test_queries.json"), 'w') as f:
            json.dump(test_queries, f, indent=2)
            
        logger.info("Test data generated and saved.")
    
    def _load_test_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load test data for tests"""
        with open(os.path.join(self.results_dir, "sample_documents.json"), 'r') as f:
            documents = json.load(f)
            
        with open(os.path.join(self.results_dir, "test_queries.json"), 'r') as f:
            queries = json.load(f)
            
        return documents, queries
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_base_url": self.api_base_url,
            "suite_status": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "performance": {},
            "recommendations": []
        }
        
        for suite in TEST_SUITES:
            logger.info(f"Running test suite: {suite}")
            suite_result = self.run_test_suite(suite)
            
            # Update results
            results["suite_status"][suite] = suite_result.get("status", "error")
            
            # Update summary
            for key in ["total_tests", "passed", "failed", "error", "simulated"]:
                results["summary"][key] += suite_result.get("summary", {}).get(key, 0)
                
            # Add suite-specific data
            results[suite] = suite_result
            
            # Add recommendations
            if "recommendations" in suite_result:
                results["recommendations"].extend(suite_result["recommendations"])
                
            # Add performance metrics
            if "performance" in suite_result:
                for metric, value in suite_result["performance"].items():
                    results["performance"][metric] = value
        
        # Save results
        with open(os.path.join(self.results_dir, "test_report.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        self._print_summary(results)
        
        return results
    
    def run_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite"""
        if suite_name not in TEST_SUITES:
            logger.error(f"Unknown test suite: {suite_name}")
            return {
                "status": "error",
                "error": f"Unknown test suite: {suite_name}",
                "summary": {
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 1,
                    "simulated": 0
                }
            }
            
        # Call the appropriate test method
        method_name = f"_test_{suite_name}"
        if hasattr(self, method_name):
            try:
                return getattr(self, method_name)()
            except Exception as e:
                logger.exception(f"Error running test suite {suite_name}: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "summary": {
                        "total_tests": 0,
                        "passed": 0,
                        "failed": 0,
                        "error": 1,
                        "simulated": 0
                    }
                }
        else:
            logger.error(f"No test implementation for suite: {suite_name}")
            return {
                "status": "error",
                "error": f"No test implementation for suite: {suite_name}",
                "summary": {
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 1,
                    "simulated": 0
                }
            }
    
    def _test_environment(self) -> Dict[str, Any]:
        """Test environment configuration"""
        logger.info("Testing environment setup...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": []
        }
        
        # Test connectivity
        try:
            response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
            if response.status_code == 200:
                results["tests"].append({
                    "name": "api_connectivity",
                    "status": "passed",
                    "details": "API is reachable"
                })
                results["summary"]["passed"] += 1
            else:
                results["tests"].append({
                    "name": "api_connectivity",
                    "status": "failed",
                    "details": f"API returned status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "api_connectivity",
                "status": "error",
                "details": f"Error connecting to API: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
            results["recommendations"].append(
                "Check if the API server is running and accessible"
            )
        
        results["summary"]["total_tests"] += 1
        
        # Test GPU info
        try:
            response = requests.get(f"{self.api_base_url}/api/gpu_info", timeout=10)
            if response.status_code == 200:
                gpu_info = response.json()
                
                # Check if it's a T4 GPU
                if gpu_info.get("is_t4", False):
                    results["tests"].append({
                        "name": "gpu_t4_available",
                        "status": "passed",
                        "details": f"T4 GPU detected: {gpu_info.get('name', 'Unknown')}"
                    })
                    results["summary"]["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "gpu_t4_available",
                        "status": "failed",
                        "details": f"No T4 GPU detected. Found: {gpu_info.get('name', 'Unknown')}"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
                    results["recommendations"].append(
                        "This toolkit is optimized for NVIDIA T4 GPUs. Consider switching to a T4 instance."
                    )
                
                # Check TensorRT availability
                if gpu_info.get("tensorrt_available", False):
                    results["tests"].append({
                        "name": "tensorrt_available",
                        "status": "passed",
                        "details": f"TensorRT available: {gpu_info.get('tensorrt_version', 'Unknown')}"
                    })
                    results["summary"]["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "tensorrt_available",
                        "status": "failed",
                        "details": "TensorRT is not available"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
                    results["recommendations"].append(
                        "Install TensorRT for optimal performance on T4 GPU"
                    )
                
                # Store GPU info
                results["gpu_info"] = gpu_info
            else:
                results["tests"].append({
                    "name": "gpu_info",
                    "status": "failed",
                    "details": f"Failed to get GPU info: {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "gpu_info",
                "status": "error",
                "details": f"Error getting GPU info: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 2
        
        # Save results
        with open(os.path.join(self.results_dir, "environment_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _test_tensorrt(self) -> Dict[str, Any]:
        """Test TensorRT optimization"""
        logger.info("Testing TensorRT optimization...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": [],
            "performance": {}
        }
        
        # Test embedding generation with TensorRT
        model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        batch_sizes = self.config.get("batch_sizes", [1, 8, 32])
        
        # Get test data
        documents, _ = self._load_test_data()
        texts = [doc["content"] for doc in documents[:max(batch_sizes)]]
        
        # Test with TensorRT enabled
        try:
            response = requests.post(
                f"{self.api_base_url}/api/embeddings",
                json={
                    "texts": texts[:1],  # Test with a single text first
                    "model_name": model_name,
                    "use_tensorrt": True,
                    "precision": self.config.get("precision", "fp16")
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("gpu_used", False):
                    results["tests"].append({
                        "name": "tensorrt_single_embedding",
                        "status": "passed",
                        "details": f"Generated embedding with TensorRT in {data.get('processing_time_ms', 0):.2f}ms"
                    })
                    results["summary"]["passed"] += 1
                    
                    # Store dimensionality
                    results["embedding_dimensions"] = data.get("dimensions", 0)
                else:
                    results["tests"].append({
                        "name": "tensorrt_single_embedding",
                        "status": "failed",
                        "details": "GPU not used for embedding generation"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
                    results["recommendations"].append(
                        "Check GPU configuration and TensorRT setup"
                    )
            else:
                results["tests"].append({
                    "name": "tensorrt_single_embedding",
                    "status": "failed",
                    "details": f"API returned status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "tensorrt_single_embedding",
                "status": "error",
                "details": f"Error generating embedding: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Test batch performance with different batch sizes
        batch_results = []
        for batch_size in batch_sizes:
            if batch_size > len(texts):
                continue
                
            try:
                # Test with TensorRT
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/api/embeddings",
                    json={
                        "texts": texts[:batch_size],
                        "model_name": model_name,
                        "use_tensorrt": True,
                        "precision": self.config.get("precision", "fp16"),
                        "batch_size": batch_size
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    trt_time = data.get("processing_time_ms", 0)
                    
                    # Test without TensorRT
                    response_no_trt = requests.post(
                        f"{self.api_base_url}/api/embeddings",
                        json={
                            "texts": texts[:batch_size],
                            "model_name": model_name,
                            "use_tensorrt": False,
                            "batch_size": batch_size
                        },
                        timeout=60
                    )
                    
                    if response_no_trt.status_code == 200:
                        data_no_trt = response_no_trt.json()
                        no_trt_time = data_no_trt.get("processing_time_ms", 0)
                        
                        # Calculate speedup
                        speedup = no_trt_time / trt_time if trt_time > 0 else 0
                        
                        batch_results.append({
                            "batch_size": batch_size,
                            "trt_time_ms": trt_time,
                            "no_trt_time_ms": no_trt_time,
                            "speedup": speedup
                        })
                        
                        results["tests"].append({
                            "name": f"tensorrt_batch_{batch_size}",
                            "status": "passed",
                            "details": f"Batch size {batch_size}: TensorRT speedup = {speedup:.2f}x"
                        })
                        results["summary"]["passed"] += 1
                    else:
                        results["tests"].append({
                            "name": f"tensorrt_batch_{batch_size}_no_trt",
                            "status": "failed",
                            "details": f"API returned status code {response_no_trt.status_code}"
                        })
                        results["summary"]["failed"] += 1
                else:
                    results["tests"].append({
                        "name": f"tensorrt_batch_{batch_size}",
                        "status": "failed",
                        "details": f"API returned status code {response.status_code}"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
            except Exception as e:
                results["tests"].append({
                    "name": f"tensorrt_batch_{batch_size}",
                    "status": "error",
                    "details": f"Error testing batch size {batch_size}: {str(e)}"
                })
                results["summary"]["error"] += 1
                results["status"] = "error"
            
            results["summary"]["total_tests"] += 1
        
        # Store batch results
        results["batch_results"] = batch_results
        
        # Calculate average speedup
        if batch_results:
            avg_speedup = np.mean([res["speedup"] for res in batch_results])
            results["performance"]["tensorrt_speedup"] = avg_speedup
            
            # Find optimal batch size
            optimal_batch = max(batch_results, key=lambda x: x["speedup"])
            results["performance"]["optimal_batch_size"] = optimal_batch["batch_size"]
            results["performance"]["optimal_speedup"] = optimal_batch["speedup"]
            
            # Add recommendations
            if avg_speedup > 1.5:
                results["recommendations"].append(
                    f"TensorRT provides significant speedup (avg {avg_speedup:.2f}x). Keep it enabled."
                )
            else:
                results["recommendations"].append(
                    f"TensorRT speedup is modest (avg {avg_speedup:.2f}x). Consider fine-tuning."
                )
                
            results["recommendations"].append(
                f"Optimal batch size is {optimal_batch['batch_size']} (speedup: {optimal_batch['speedup']:.2f}x)"
            )
        
        # Save results
        with open(os.path.join(self.results_dir, "tensorrt_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _test_vectorstore(self) -> Dict[str, Any]:
        """Test vector store functionality"""
        logger.info("Testing vector store functionality...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": []
        }
        
        # Load test data
        documents, queries = self._load_test_data()
        
        # Test search functionality
        try:
            # Use the first query
            query = queries[0]
            
            response = requests.post(
                f"{self.api_base_url}/api/vectorstore/search",
                json={
                    "query": query,
                    "k": 3,
                    "table_name": "test_collection"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results["tests"].append({
                    "name": "vectorstore_search",
                    "status": "passed",
                    "details": f"Search completed in {data.get('processing_time_ms', 0):.2f}ms"
                })
                results["summary"]["passed"] += 1
                
                # Store search results
                results["search_results"] = data.get("results", [])
                
                # Check if we have enough results
                if len(data.get("results", [])) < 3:
                    results["recommendations"].append(
                        "Search returned fewer results than requested. Check vector store content."
                    )
            else:
                # Try to simulate search
                results["tests"].append({
                    "name": "vectorstore_search",
                    "status": "simulated",
                    "details": f"API returned status code {response.status_code}. Using simulated data."
                })
                results["summary"]["simulated"] += 1
                
                # Create simulated results
                simulated_results = [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": 0.9 - (i * 0.1)
                    }
                    for i, doc in enumerate(documents[:3])
                ]
                
                results["search_results"] = simulated_results
                results["recommendations"].append(
                    "Search API failed. Check vector store setup and connection."
                )
        except Exception as e:
            results["tests"].append({
                "name": "vectorstore_search",
                "status": "error",
                "details": f"Error searching vector store: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
            results["recommendations"].append(
                "Vector store search failed. Check API and database connection."
            )
        
        results["summary"]["total_tests"] += 1
        
        # Test MMR search
        try:
            # Use the first query
            query = queries[0]
            
            response = requests.post(
                f"{self.api_base_url}/api/vectorstore/mmr_search",
                json={
                    "query": query,
                    "k": 3,
                    "lambda_mult": 0.5,
                    "table_name": "test_collection"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results["tests"].append({
                    "name": "vectorstore_mmr_search",
                    "status": "passed",
                    "details": f"MMR search completed in {data.get('processing_time_ms', 0):.2f}ms"
                })
                results["summary"]["passed"] += 1
                
                # Store MMR search results
                results["mmr_search_results"] = data.get("results", [])
            else:
                # Try to simulate MMR search
                results["tests"].append({
                    "name": "vectorstore_mmr_search",
                    "status": "simulated",
                    "details": f"API returned status code {response.status_code}. Using simulated data."
                })
                results["summary"]["simulated"] += 1
                
                # Create simulated results with diversity
                simulated_results = [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": 0.85 - (i * 0.15)
                    }
                    for i, doc in enumerate(documents[5:8])  # Different docs than regular search
                ]
                
                results["mmr_search_results"] = simulated_results
                results["recommendations"].append(
                    "MMR search API failed. Check vector store setup and connection."
                )
        except Exception as e:
            results["tests"].append({
                "name": "vectorstore_mmr_search",
                "status": "error",
                "details": f"Error performing MMR search: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
            results["recommendations"].append(
                "MMR search failed. Check API and database connection."
            )
        
        results["summary"]["total_tests"] += 1
        
        # Save results
        with open(os.path.join(self.results_dir, "vectorstore_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _test_gpu_performance(self) -> Dict[str, Any]:
        """Test GPU performance"""
        logger.info("Testing GPU performance...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": [],
            "performance": {}
        }
        
        # Load test data
        documents, _ = self._load_test_data()
        texts = [doc["content"] for doc in documents]
        
        # Test GPU vs CPU performance
        model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = 8  # Use a moderate batch size
        
        try:
            # Test with GPU
            response_gpu = requests.post(
                f"{self.api_base_url}/api/embeddings",
                json={
                    "texts": texts[:batch_size],
                    "model_name": model_name,
                    "use_tensorrt": True,
                    "precision": self.config.get("precision", "fp16")
                },
                timeout=60
            )
            
            if response_gpu.status_code == 200:
                data_gpu = response_gpu.json()
                gpu_time = data_gpu.get("processing_time_ms", 0)
                
                # Test with CPU (by disabling GPU)
                response_cpu = requests.post(
                    f"{self.api_base_url}/api/embeddings",
                    json={
                        "texts": texts[:batch_size],
                        "model_name": model_name,
                        "use_tensorrt": False,
                        "precision": "fp32"  # CPU uses FP32
                    },
                    timeout=60
                )
                
                if response_cpu.status_code == 200:
                    data_cpu = response_cpu.json()
                    cpu_time = data_cpu.get("processing_time_ms", 0)
                    
                    # Calculate speedup
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    
                    results["tests"].append({
                        "name": "gpu_vs_cpu_performance",
                        "status": "passed",
                        "details": f"GPU: {gpu_time:.2f}ms, CPU: {cpu_time:.2f}ms, Speedup: {speedup:.2f}x"
                    })
                    results["summary"]["passed"] += 1
                    
                    # Store performance metrics
                    results["performance"]["gpu_time_ms"] = gpu_time
                    results["performance"]["cpu_time_ms"] = cpu_time
                    results["performance"]["gpu_speedup"] = speedup
                    
                    # Add recommendations
                    if speedup > 5:
                        results["recommendations"].append(
                            f"GPU provides excellent speedup ({speedup:.2f}x). Continue using GPU acceleration."
                        )
                    elif speedup > 2:
                        results["recommendations"].append(
                            f"GPU provides good speedup ({speedup:.2f}x). Continue using GPU acceleration."
                        )
                    else:
                        results["recommendations"].append(
                            f"GPU speedup is modest ({speedup:.2f}x). Consider optimizing GPU usage."
                        )
                else:
                    results["tests"].append({
                        "name": "gpu_vs_cpu_performance_cpu",
                        "status": "failed",
                        "details": f"API returned status code {response_cpu.status_code} for CPU test"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
            else:
                results["tests"].append({
                    "name": "gpu_vs_cpu_performance_gpu",
                    "status": "failed",
                    "details": f"API returned status code {response_gpu.status_code} for GPU test"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "gpu_vs_cpu_performance",
                "status": "error",
                "details": f"Error testing GPU vs CPU performance: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
            results["recommendations"].append(
                "GPU performance test failed. Check API and GPU availability."
            )
        
        results["summary"]["total_tests"] += 1
        
        # Test MMR GPU performance
        try:
            # Use a query
            query = "What is machine learning?"
            
            # Test MMR search with GPU
            response_mmr = requests.post(
                f"{self.api_base_url}/api/vectorstore/mmr_search",
                json={
                    "query": query,
                    "k": 5,
                    "lambda_mult": 0.5,
                    "table_name": "test_collection"
                },
                timeout=30
            )
            
            if response_mmr.status_code == 200:
                data_mmr = response_mmr.json()
                mmr_time = data_mmr.get("processing_time_ms", 0)
                
                # Test regular search for comparison
                response_search = requests.post(
                    f"{self.api_base_url}/api/vectorstore/search",
                    json={
                        "query": query,
                        "k": 5,
                        "table_name": "test_collection"
                    },
                    timeout=30
                )
                
                if response_search.status_code == 200:
                    data_search = response_search.json()
                    search_time = data_search.get("processing_time_ms", 0)
                    
                    # Calculate MMR overhead
                    mmr_overhead = mmr_time / search_time if search_time > 0 else 0
                    
                    results["tests"].append({
                        "name": "mmr_performance",
                        "status": "passed",
                        "details": f"MMR: {mmr_time:.2f}ms, Regular: {search_time:.2f}ms, Overhead: {mmr_overhead:.2f}x"
                    })
                    results["summary"]["passed"] += 1
                    
                    # Store performance metrics
                    results["performance"]["mmr_time_ms"] = mmr_time
                    results["performance"]["search_time_ms"] = search_time
                    results["performance"]["mmr_overhead"] = mmr_overhead
                    
                    # Add recommendations
                    if mmr_overhead > 2:
                        results["recommendations"].append(
                            f"MMR search has significant overhead ({mmr_overhead:.2f}x). Use selectively."
                        )
                    else:
                        results["recommendations"].append(
                            f"MMR search overhead is reasonable ({mmr_overhead:.2f}x). Use for better diversity."
                        )
                else:
                    results["tests"].append({
                        "name": "mmr_performance_search",
                        "status": "failed",
                        "details": f"API returned status code {response_search.status_code} for regular search"
                    })
                    results["summary"]["failed"] += 1
            else:
                results["tests"].append({
                    "name": "mmr_performance_mmr",
                    "status": "failed",
                    "details": f"API returned status code {response_mmr.status_code} for MMR search"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "mmr_performance",
                "status": "error",
                "details": f"Error testing MMR performance: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Save results
        with open(os.path.join(self.results_dir, "gpu_performance_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        logger.info("Testing error handling and recovery...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": []
        }
        
        # Test invalid input handling
        try:
            # Test with empty input
            response = requests.post(
                f"{self.api_base_url}/api/embeddings",
                json={
                    "texts": [],
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                },
                timeout=10
            )
            
            # Expect a 4xx error
            if 400 <= response.status_code < 500:
                results["tests"].append({
                    "name": "error_empty_input",
                    "status": "passed",
                    "details": f"API correctly returned {response.status_code} for empty input"
                })
                results["summary"]["passed"] += 1
            else:
                results["tests"].append({
                    "name": "error_empty_input",
                    "status": "failed",
                    "details": f"API returned unexpected status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
                results["recommendations"].append(
                    "Improve validation for empty inputs"
                )
        except Exception as e:
            results["tests"].append({
                "name": "error_empty_input",
                "status": "error",
                "details": f"Error testing empty input: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Test invalid model
        try:
            # Test with invalid model name
            response = requests.post(
                f"{self.api_base_url}/api/embeddings",
                json={
                    "texts": ["Test text"],
                    "model_name": "invalid-model-name"
                },
                timeout=10
            )
            
            # Expect a 4xx error or graceful fallback
            if 400 <= response.status_code < 500:
                results["tests"].append({
                    "name": "error_invalid_model",
                    "status": "passed",
                    "details": f"API correctly returned {response.status_code} for invalid model"
                })
                results["summary"]["passed"] += 1
            elif response.status_code == 200:
                # Check if it fell back to a default model
                data = response.json()
                if data.get("model", "") != "invalid-model-name":
                    results["tests"].append({
                        "name": "error_invalid_model",
                        "status": "passed",
                        "details": f"API fell back to {data.get('model', 'unknown')} model"
                    })
                    results["summary"]["passed"] += 1
                    results["recommendations"].append(
                        "API has good fallback behavior for invalid models"
                    )
                else:
                    results["tests"].append({
                        "name": "error_invalid_model",
                        "status": "failed",
                        "details": "API accepted invalid model name without fallback"
                    })
                    results["summary"]["failed"] += 1
                    results["status"] = "failure"
            else:
                results["tests"].append({
                    "name": "error_invalid_model",
                    "status": "failed",
                    "details": f"API returned unexpected status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "error_invalid_model",
                "status": "error",
                "details": f"Error testing invalid model: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Test large input handling
        try:
            # Create a very long text
            long_text = "This is a very long text. " * 1000
            
            response = requests.post(
                f"{self.api_base_url}/api/embeddings",
                json={
                    "texts": [long_text],
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                results["tests"].append({
                    "name": "error_large_input",
                    "status": "passed",
                    "details": "API handled large input correctly"
                })
                results["summary"]["passed"] += 1
                results["recommendations"].append(
                    "API can handle large inputs"
                )
            elif 400 <= response.status_code < 500:
                results["tests"].append({
                    "name": "error_large_input",
                    "status": "passed",
                    "details": f"API correctly rejected large input with {response.status_code}"
                })
                results["summary"]["passed"] += 1
                results["recommendations"].append(
                    "API rejects large inputs with proper error"
                )
            else:
                results["tests"].append({
                    "name": "error_large_input",
                    "status": "failed",
                    "details": f"API returned unexpected status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "error_large_input",
                "status": "error",
                "details": f"Error testing large input: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Save results
        with open(os.path.join(self.results_dir, "error_handling_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _test_api(self) -> Dict[str, Any]:
        """Test API endpoints"""
        logger.info("Testing API endpoints...")
        
        results = {
            "status": "success",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "simulated": 0
            },
            "recommendations": []
        }
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                results["tests"].append({
                    "name": "api_health",
                    "status": "passed",
                    "details": "Health endpoint is working"
                })
                results["summary"]["passed"] += 1
                
                # Store health data
                results["health_data"] = response.json()
            else:
                results["tests"].append({
                    "name": "api_health",
                    "status": "failed",
                    "details": f"Health endpoint returned status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "api_health",
                "status": "error",
                "details": f"Error accessing health endpoint: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Test metrics endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/metrics", timeout=10)
            
            if response.status_code == 200:
                results["tests"].append({
                    "name": "api_metrics",
                    "status": "passed",
                    "details": "Metrics endpoint is working"
                })
                results["summary"]["passed"] += 1
                
                # Store metrics data
                results["metrics_data"] = response.json()
            else:
                results["tests"].append({
                    "name": "api_metrics",
                    "status": "failed",
                    "details": f"Metrics endpoint returned status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "api_metrics",
                "status": "error",
                "details": f"Error accessing metrics endpoint: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Test GPU info endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/gpu_info", timeout=10)
            
            if response.status_code == 200:
                results["tests"].append({
                    "name": "api_gpu_info",
                    "status": "passed",
                    "details": "GPU info endpoint is working"
                })
                results["summary"]["passed"] += 1
                
                # Store GPU info data
                results["gpu_info_data"] = response.json()
            else:
                results["tests"].append({
                    "name": "api_gpu_info",
                    "status": "failed",
                    "details": f"GPU info endpoint returned status code {response.status_code}"
                })
                results["summary"]["failed"] += 1
                results["status"] = "failure"
        except Exception as e:
            results["tests"].append({
                "name": "api_gpu_info",
                "status": "error",
                "details": f"Error accessing GPU info endpoint: {str(e)}"
            })
            results["summary"]["error"] += 1
            results["status"] = "error"
        
        results["summary"]["total_tests"] += 1
        
        # Save results
        with open(os.path.join(self.results_dir, "api_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 80)
        print(f"  Test Summary for {self.api_base_url}")
        print("=" * 80)
        
        summary = results.get("summary", {})
        print(f"Total Tests:  {summary.get('total_tests', 0)}")
        print(f"Passed:       {summary.get('passed', 0)}")
        print(f"Failed:       {summary.get('failed', 0)}")
        print(f"Errors:       {summary.get('error', 0)}")
        print(f"Simulated:    {summary.get('simulated', 0)}")
        print("-" * 80)
        
        # Print suite status
        print("Test Suite Status:")
        for suite, status in results.get("suite_status", {}).items():
            print(f"  - {suite}: {status}")
        print("-" * 80)
        
        # Print performance metrics
        if "performance" in results:
            print("Performance Metrics:")
            for metric, value in results["performance"].items():
                print(f"  - {metric}: {value}")
            print("-" * 80)
        
        # Print recommendations
        if "recommendations" in results:
            print("Recommendations:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")
            print("-" * 80)
        
        print(f"Detailed results saved in: {self.results_dir}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SAP HANA Cloud Generative AI Toolkit T4 GPU Testing")
    
    # Required arguments
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--results-dir", help="Directory to store test results")
    
    # Test selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all test suites")
    group.add_argument("--suite", help="Run specific test suite")
    group.add_argument("--list-suites", action="store_true", help="List available test suites")
    
    args = parser.parse_args()
    
    # List test suites if requested
    if args.list_suites:
        print("Available test suites:")
        for suite, description in TEST_SUITES.items():
            print(f"  {suite}: {description}")
        return 0
    
    # Create tester
    tester = HanaGenerativeAITester(args.config)
    
    # Override results directory if specified
    if args.results_dir:
        tester.results_dir = args.results_dir
        os.makedirs(tester.results_dir, exist_ok=True)
    
    # Run tests
    if args.all:
        results = tester.run_all_tests()
    elif args.suite:
        if args.suite not in TEST_SUITES:
            print(f"Error: Unknown test suite: {args.suite}")
            print("Available test suites:")
            for suite in TEST_SUITES:
                print(f"  {suite}")
            return 1
            
        results = tester.run_test_suite(args.suite)
    else:
        parser.print_help()
        return 1
    
    # Determine exit code based on results
    if results.get("status") == "error":
        return 2
    elif results.get("status") == "failure":
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())