"""
Testing utilities for SAP HANA Cloud Generative AI Toolkit.

This module provides utilities and fixtures for testing the Generative AI Toolkit,
including mock data generation, API simulation, and test result formatting.
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common test data
SAMPLE_TOPICS = [
    "machine learning", "artificial intelligence", "natural language processing",
    "computer vision", "deep learning", "neural networks", "transformer models",
    "SAP HANA", "vector databases", "embeddings", "GPU acceleration",
    "T4 optimization", "TensorRT", "CUDA programming", "parallel computing"
]

SAMPLE_QUERIES = [
    "What is machine learning?",
    "How does GPU acceleration work?",
    "Tell me about vector databases",
    "Explain neural networks",
    "What is SAP HANA?",
    "How are embeddings used in AI?",
    "What is TensorRT optimization?"
]

class TestDataGenerator:
    """Generator for test data used in testing"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_documents(self, count: int = 15) -> List[Dict[str, Any]]:
        """
        Generate sample documents for testing
        
        Args:
            count: Number of documents to generate
        
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for i in range(count):
            # Select a topic
            topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
            
            # Generate document
            document = {
                "id": f"doc{i}",
                "content": self._generate_document_content(topic, i),
                "metadata": {
                    "source": "test",
                    "topic": topic,
                    "importance": i % 3 + 1,
                    "created_at": time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400)),
                    "word_count": random.randint(50, 500)
                }
            }
            
            documents.append(document)
            
        return documents
    
    def _generate_document_content(self, topic: str, index: int) -> str:
        """
        Generate document content based on topic
        
        Args:
            topic: Document topic
            index: Document index
        
        Returns:
            Document content
        """
        # Simple templates
        templates = [
            "This document provides an overview of {topic}. It covers the key concepts and applications.",
            "{topic} is an important field in technology. This document explains its significance and use cases.",
            "An introduction to {topic}. This document explains how it works and why it matters.",
            "Understanding {topic}: This document provides a detailed explanation of the underlying principles.",
            "Advanced concepts in {topic}. This document is designed for users who already have basic knowledge."
        ]
        
        template = templates[index % len(templates)]
        return template.format(topic=topic)
    
    def generate_queries(self, count: int = 7) -> List[str]:
        """
        Generate sample queries for testing
        
        Args:
            count: Number of queries to generate
        
        Returns:
            List of query strings
        """
        if count <= len(SAMPLE_QUERIES):
            return SAMPLE_QUERIES[:count]
        
        # Generate additional queries if needed
        queries = SAMPLE_QUERIES.copy()
        
        for i in range(len(SAMPLE_QUERIES), count):
            # Select a topic
            topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
            
            # Generate query
            query = f"What are the applications of {topic}?"
            queries.append(query)
            
        return queries
    
    def generate_embeddings(self, count: int = 10, dim: int = 384) -> List[List[float]]:
        """
        Generate mock embeddings for testing
        
        Args:
            count: Number of embeddings to generate
            dim: Dimensionality of embeddings
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for _ in range(count):
            # Generate random embedding vector
            vector = np.random.randn(dim).astype(np.float32)
            
            # Normalize
            vector = vector / np.linalg.norm(vector)
            
            embeddings.append(vector.tolist())
            
        return embeddings
    
    def save_test_data(self, output_dir: str = "test_data"):
        """
        Generate and save test data to files
        
        Args:
            output_dir: Directory to save test data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save documents
        documents = self.generate_documents()
        with open(os.path.join(output_dir, "sample_documents.json"), 'w') as f:
            json.dump(documents, f, indent=2)
            
        # Generate and save queries
        queries = self.generate_queries()
        with open(os.path.join(output_dir, "test_queries.json"), 'w') as f:
            json.dump(queries, f, indent=2)
            
        # Generate and save embeddings
        embeddings = self.generate_embeddings(len(documents))
        with open(os.path.join(output_dir, "sample_embeddings.json"), 'w') as f:
            json.dump(embeddings, f, indent=2)
            
        logger.info(f"Test data saved to {output_dir}")


class APISimulator:
    """
    Simulator for API responses when real API is not available
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize API simulator
        
        Args:
            data_dir: Directory with test data
        """
        self.data_dir = data_dir or "test_data"
        self.documents = []
        self.embeddings = []
        self.queries = []
        
        # Load test data
        self._load_test_data()
    
    def _load_test_data(self):
        """Load test data from files"""
        try:
            # Load documents
            doc_path = os.path.join(self.data_dir, "sample_documents.json")
            if os.path.exists(doc_path):
                with open(doc_path, 'r') as f:
                    self.documents = json.load(f)
            
            # Load embeddings
            emb_path = os.path.join(self.data_dir, "sample_embeddings.json")
            if os.path.exists(emb_path):
                with open(emb_path, 'r') as f:
                    self.embeddings = json.load(f)
            
            # Load queries
            query_path = os.path.join(self.data_dir, "test_queries.json")
            if os.path.exists(query_path):
                with open(query_path, 'r') as f:
                    self.queries = json.load(f)
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            
            # Generate test data if loading fails
            generator = TestDataGenerator()
            self.documents = generator.generate_documents()
            self.embeddings = generator.generate_embeddings(len(self.documents))
            self.queries = generator.generate_queries()
    
    def simulate_embedding_response(
        self, 
        texts: List[str], 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_tensorrt: bool = True,
        precision: str = "fp16"
    ) -> Dict[str, Any]:
        """
        Simulate embedding generation response
        
        Args:
            texts: Input texts
            model_name: Embedding model name
            use_tensorrt: Whether TensorRT is used
            precision: Precision for computation
            
        Returns:
            Simulated API response
        """
        # Generate embeddings
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing time
        
        # Use pre-generated embeddings if available, otherwise generate new ones
        if len(self.embeddings) >= len(texts):
            embeddings = self.embeddings[:len(texts)]
        else:
            generator = TestDataGenerator()
            embeddings = generator.generate_embeddings(len(texts))
        
        # Determine embedding dimensions
        dim = len(embeddings[0]) if embeddings else 384
        
        # Simulate processing time based on precision and TensorRT
        if use_tensorrt:
            processing_time = 50 + 5 * len(texts)  # Base + per text cost
            if precision == "int8":
                processing_time *= 0.7  # INT8 is faster
            elif precision == "fp32":
                processing_time *= 1.5  # FP32 is slower
        else:
            processing_time = 200 + 20 * len(texts)  # CPU is slower
        
        # Convert to milliseconds
        processing_time_ms = processing_time
        
        return {
            "embeddings": embeddings,
            "model": model_name,
            "dimensions": dim,
            "processing_time_ms": processing_time_ms,
            "gpu_used": use_tensorrt,
            "batch_size_used": len(texts)
        }
    
    def simulate_search_response(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        mmr: bool = False,
        lambda_mult: float = 0.5
    ) -> Dict[str, Any]:
        """
        Simulate vector search response
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional filter
            mmr: Whether to use MMR for diversity
            lambda_mult: Lambda parameter for MMR
            
        Returns:
            Simulated search response
        """
        # Simulate processing time
        start_time = time.time()
        time.sleep(0.05)  # Simulate processing time
        
        # Select documents to return
        if len(self.documents) > 0:
            # Limit to k documents
            k = min(k, len(self.documents))
            
            if mmr:
                # For MMR, select a more diverse set
                indices = list(range(len(self.documents)))
                random.shuffle(indices)
                selected_indices = indices[:k]
                selected_docs = [self.documents[i] for i in selected_indices]
                
                # Add diversity-based scores
                results = []
                for i, doc in enumerate(selected_docs):
                    score = 0.9 - (i * 0.1 * random.uniform(0.5, 1.0))
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            else:
                # For regular search, select most relevant docs
                selected_docs = self.documents[:k]
                
                # Add similarity-based scores
                results = []
                for i, doc in enumerate(selected_docs):
                    score = 0.95 - (i * 0.05)
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
        else:
            # No documents available
            results = []
        
        # Apply filtering if specified
        if filter and results:
            filtered_results = []
            for result in results:
                # Check if result matches filter
                match = True
                for key, value in filter.items():
                    if key in result["metadata"]:
                        if result["metadata"][key] != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    filtered_results.append(result)
            
            results = filtered_results
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        if mmr:
            processing_time_ms *= 1.5  # MMR is more expensive
        
        return {
            "results": results,
            "processing_time_ms": processing_time_ms,
            "query": query
        }
    
    def simulate_health_response(self) -> Dict[str, Any]:
        """
        Simulate health check response
        
        Returns:
            Simulated health response
        """
        return {
            "status": "healthy",
            "api_version": "1.0.0",
            "backend": {
                "status": "healthy",
                "details": {
                    "gpu": {
                        "available": True,
                        "name": "Tesla T4",
                        "memory_total": 16 * 1024 * 1024 * 1024,
                        "memory_used": 2 * 1024 * 1024 * 1024
                    },
                    "services": {
                        "embedding": "up",
                        "vectorstore": "up",
                        "tensorrt": "up"
                    }
                }
            },
            "timeouts": {
                "default": 30,
                "embeddings": 60,
                "search": 30,
                "mmr_search": 45,
                "health": 10,
                "metrics": 15
            }
        }
    
    def simulate_metrics_response(self) -> Dict[str, Any]:
        """
        Simulate metrics response
        
        Returns:
            Simulated metrics response
        """
        return {
            "requests": {
                "total": 1245,
                "embeddings": 876,
                "search": 328,
                "mmr_search": 41
            },
            "performance": {
                "embedding_avg_ms": 85.4,
                "search_avg_ms": 42.7,
                "mmr_search_avg_ms": 67.9
            },
            "gpu": {
                "utilization_avg": 45.2,
                "memory_usage_avg": 4.2,
                "temperature_avg": 56.5
            },
            "errors": {
                "total": 23,
                "rate": 0.018,
                "types": {
                    "timeout": 8,
                    "invalid_input": 11,
                    "internal": 4
                }
            },
            "uptime_seconds": 345600
        }
    
    def simulate_gpu_info_response(self) -> Dict[str, Any]:
        """
        Simulate GPU info response
        
        Returns:
            Simulated GPU info response
        """
        return {
            "name": "Tesla T4",
            "is_t4": True,
            "compute_capability": "7.5",
            "total_memory_gb": 16.0,
            "allocated_memory_gb": 2.3,
            "reserved_memory_gb": 4.1,
            "multi_processor_count": 40,
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 49152,
            "tensorrt_available": True,
            "tensorrt_version": "8.5.1"
        }


class TestResultFormatter:
    """
    Formatter for test results
    """
    
    @staticmethod
    def format_test_summary(results: Dict[str, Any]) -> str:
        """
        Format test summary as string
        
        Args:
            results: Test results dictionary
            
        Returns:
            Formatted summary string
        """
        summary = results.get("summary", {})
        
        lines = [
            "=" * 80,
            f"  Test Summary",
            "=" * 80,
            f"Total Tests:  {summary.get('total_tests', 0)}",
            f"Passed:       {summary.get('passed', 0)}",
            f"Failed:       {summary.get('failed', 0)}",
            f"Errors:       {summary.get('error', 0)}",
            f"Simulated:    {summary.get('simulated', 0)}",
            "-" * 80
        ]
        
        # Add suite status
        lines.append("Test Suite Status:")
        for suite, status in results.get("suite_status", {}).items():
            lines.append(f"  - {suite}: {status}")
        lines.append("-" * 80)
        
        # Add performance metrics
        if "performance" in results:
            lines.append("Performance Metrics:")
            for metric, value in results["performance"].items():
                if isinstance(value, float):
                    lines.append(f"  - {metric}: {value:.2f}")
                else:
                    lines.append(f"  - {metric}: {value}")
            lines.append("-" * 80)
        
        # Add recommendations
        if "recommendations" in results:
            lines.append("Recommendations:")
            for rec in results["recommendations"]:
                lines.append(f"  - {rec}")
            lines.append("-" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_html_report(results: Dict[str, Any]) -> str:
        """
        Format test results as HTML report
        
        Args:
            results: Test results dictionary
            
        Returns:
            HTML report
        """
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAP HANA Cloud Generative AI Toolkit Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #0066cc; }
                h2 { color: #333; margin-top: 20px; }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .status-success { color: green; }
                .status-partial { color: orange; }
                .status-error, .status-failure { color: red; }
                .status-simulated { color: blue; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .recommendations { background-color: #e6f7ff; padding: 15px; border-radius: 5px; }
                .performance { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>SAP HANA Cloud Generative AI Toolkit Test Report</h1>
            <div class='summary'>
                <h2>Summary</h2>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>API Base URL:</strong> {api_base_url}</p>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Passed:</strong> <span class='status-success'>{passed}</span></p>
                <p><strong>Simulated:</strong> <span class='status-simulated'>{simulated}</span></p>
                <p><strong>Failed:</strong> <span class='status-failure'>{failed}</span></p>
                <p><strong>Error:</strong> <span class='status-error'>{error}</span></p>
            </div>
            
            <h2>Test Suite Status</h2>
            <table>
                <tr>
                    <th>Suite</th>
                    <th>Status</th>
                </tr>
                {suite_status_rows}
            </table>
            
            <div class='performance'>
                <h2>Performance Metrics</h2>
                <p><strong>Average GPU Speedup:</strong> {gpu_speedup}x</p>
                <p><strong>Optimal Batch Size:</strong> {optimal_batch_size}</p>
                <p><strong>MMR Speedup:</strong> {mmr_speedup}x</p>
            </div>
            
            <div class='recommendations'>
                <h2>Recommendations</h2>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </body>
        </html>
        '''
        
        # Generate suite status rows
        suite_status_rows = ""
        for suite_name, status in results.get("suite_status", {}).items():
            suite_status_rows += f'<tr><td>{suite_name}</td><td class="status-{status}">{status}</td></tr>'
            
        # Generate recommendations list
        recommendations_html = ""
        for rec in results.get("recommendations", []):
            recommendations_html += f"<li>{rec}</li>"
            
        # Format performance metrics
        gpu_speedup = "N/A"
        mmr_speedup = "N/A"
        optimal_batch_size = "N/A"
        
        if "performance" in results:
            perf = results["performance"]
            if "gpu_speedup" in perf:
                gpu_speedup = f"{perf['gpu_speedup']:.2f}"
            if "mmr_speedup" in perf:
                mmr_speedup = f"{perf['mmr_speedup']:.2f}"
            if "optimal_batch_size" in perf:
                optimal_batch_size = str(perf["optimal_batch_size"])
            
        # Generate HTML
        html = html_template.format(
            timestamp=results.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            api_base_url=results.get("api_base_url", "N/A"),
            total_tests=results.get("summary", {}).get("total_tests", 0),
            passed=results.get("summary", {}).get("passed", 0),
            simulated=results.get("summary", {}).get("simulated", 0),
            failed=results.get("summary", {}).get("failed", 0),
            error=results.get("summary", {}).get("error", 0),
            suite_status_rows=suite_status_rows,
            gpu_speedup=gpu_speedup,
            optimal_batch_size=optimal_batch_size,
            mmr_speedup=mmr_speedup,
            recommendations=recommendations_html
        )
        
        return html


# Utility functions
def generate_test_data(output_dir: str = "test_data", seed: Optional[int] = None):
    """
    Generate test data and save to files
    
    Args:
        output_dir: Directory to save test data
        seed: Random seed for reproducibility
    """
    generator = TestDataGenerator(seed)
    generator.save_test_data(output_dir)