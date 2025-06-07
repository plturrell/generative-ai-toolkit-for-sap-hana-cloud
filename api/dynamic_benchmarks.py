"""
Dynamic Benchmarking for NGC Blueprint

This module provides functionality to dynamically generate benchmark metrics
for the NGC blueprint based on actual performance measurements, replacing
static values with dynamically computed ones.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkMetric:
    """Class to track benchmark metrics."""
    
    def __init__(self, name: str, unit: str, window_size: int = 1000):
        """
        Initialize benchmark metric.
        
        Args:
            name: Metric name
            unit: Metric unit
            window_size: Number of samples to keep
        """
        self.name = name
        self.unit = unit
        self.window_size = window_size
        self.values = []
        self.timestamps = []
        self.metadata = {}
    
    def add_sample(self, value: float, metadata: Dict[str, Any] = None):
        """
        Add sample to metric.
        
        Args:
            value: Metric value
            metadata: Additional metadata for the sample
        """
        self.values.append(value)
        self.timestamps.append(datetime.now())
        
        if metadata:
            sample_id = len(self.values) - 1
            self.metadata[sample_id] = metadata
        
        # Trim if exceeds window size
        if len(self.values) > self.window_size:
            excess = len(self.values) - self.window_size
            self.values = self.values[excess:]
            self.timestamps = self.timestamps[excess:]
            
            # Update metadata indices
            new_metadata = {}
            for idx, meta in self.metadata.items():
                new_idx = idx - excess
                if new_idx >= 0:
                    new_metadata[new_idx] = meta
            self.metadata = new_metadata
    
    def get_average(self) -> Optional[float]:
        """Get average metric value."""
        if not self.values:
            return None
        
        return float(np.mean(self.values))
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """
        Get percentile value.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value or None if no samples
        """
        if not self.values:
            return None
        
        return float(np.percentile(self.values, percentile))
    
    def get_formatted_value(self, value: float) -> str:
        """
        Format metric value with unit.
        
        Args:
            value: Metric value
            
        Returns:
            Formatted value string
        """
        if self.unit == "ms":
            return f"{value:.1f}ms"
        elif self.unit == "tokens/sec":
            return f"{value:.1f} tokens/sec"
        elif self.unit == "GB":
            return f"{value:.1f} GB"
        elif self.unit == "x":
            return f"{value:.1f}x"
        else:
            return f"{value}"
    
    def get_by_batch_size(self, batch_size: int) -> List[float]:
        """
        Get samples for specific batch size.
        
        Args:
            batch_size: Batch size to filter by
            
        Returns:
            List of values for that batch size
        """
        values = []
        
        for idx, value in enumerate(self.values):
            meta = self.metadata.get(idx, {})
            if meta.get("batch_size") == batch_size:
                values.append(value)
        
        return values
    
    def get_batch_sizes(self) -> List[int]:
        """Get unique batch sizes in samples."""
        batch_sizes = set()
        
        for meta in self.metadata.values():
            if "batch_size" in meta:
                batch_sizes.add(meta["batch_size"])
        
        return sorted(list(batch_sizes))


class BenchmarkManager:
    """Manager for collecting and reporting benchmark metrics."""
    
    def __init__(self, blueprint_file: str, metrics_file: str = "/tmp/ngc_benchmark_metrics.json"):
        """
        Initialize benchmark manager.
        
        Args:
            blueprint_file: Path to NGC blueprint file
            metrics_file: Path to save metrics
        """
        self.blueprint_file = blueprint_file
        self.metrics_file = metrics_file
        
        # Initialize metrics
        self.metrics = {
            "inference_latency": BenchmarkMetric("Inference Latency", "ms"),
            "throughput": BenchmarkMetric("Throughput", "tokens/sec"),
            "memory_usage": BenchmarkMetric("Memory Usage", "GB"),
            "tensorrt_speedup": BenchmarkMetric("TensorRT Speedup", "x")
        }
        
        # Load saved metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load saved metrics from file."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Restore metrics
                for metric_name, metric_data in data.items():
                    if metric_name in self.metrics:
                        self.metrics[metric_name].values = metric_data.get("values", [])
                        self.metrics[metric_name].timestamps = [
                            datetime.fromisoformat(ts) for ts in metric_data.get("timestamps", [])
                        ]
                        self.metrics[metric_name].metadata = {
                            int(k): v for k, v in metric_data.get("metadata", {}).items()
                        }
                
                logger.info(f"Loaded benchmark metrics from {self.metrics_file}")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            data = {}
            
            for metric_name, metric in self.metrics.items():
                data[metric_name] = {
                    "values": metric.values,
                    "timestamps": [ts.isoformat() for ts in metric.timestamps],
                    "metadata": {str(k): v for k, v in metric.metadata.items()}
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved benchmark metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def record_inference_benchmark(self, 
                                  batch_size: int,
                                  sequence_length: int,
                                  latency_ms: float,
                                  throughput: float,
                                  memory_usage_bytes: float,
                                  tensorrt_speedup: float,
                                  model_name: str = None):
        """
        Record inference benchmark metrics.
        
        Args:
            batch_size: Batch size used
            sequence_length: Sequence length processed
            latency_ms: Inference latency in milliseconds
            throughput: Throughput in tokens/second
            memory_usage_bytes: Memory usage in bytes
            tensorrt_speedup: Speedup from TensorRT optimization
            model_name: Optional model name
        """
        metadata = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_name": model_name
        }
        
        # Record metrics
        self.metrics["inference_latency"].add_sample(latency_ms, metadata)
        self.metrics["throughput"].add_sample(throughput, metadata)
        self.metrics["memory_usage"].add_sample(memory_usage_bytes / (1024 * 1024 * 1024), metadata)  # Convert to GB
        self.metrics["tensorrt_speedup"].add_sample(tensorrt_speedup, metadata)
        
        # Save metrics
        self._save_metrics()
        
        logger.info(f"Recorded benchmark: batch_size={batch_size}, "
                   f"latency={latency_ms:.2f}ms, throughput={throughput:.2f} tokens/s, "
                   f"memory={memory_usage_bytes/(1024*1024*1024):.2f}GB, speedup={tensorrt_speedup:.2f}x")
    
    def load_blueprint(self) -> Dict[str, Any]:
        """
        Load NGC blueprint from file.
        
        Returns:
            Blueprint dictionary
        """
        try:
            with open(self.blueprint_file, 'r') as f:
                blueprint = json.load(f)
            return blueprint
        except Exception as e:
            logger.error(f"Error loading blueprint: {str(e)}")
            return {}
    
    def save_blueprint(self, blueprint: Dict[str, Any]):
        """
        Save NGC blueprint to file.
        
        Args:
            blueprint: Blueprint dictionary
        """
        try:
            with open(self.blueprint_file, 'w') as f:
                json.dump(blueprint, f, indent=2)
            logger.info(f"Saved blueprint to {self.blueprint_file}")
        except Exception as e:
            logger.error(f"Error saving blueprint: {str(e)}")
    
    def update_blueprint_benchmarks(self):
        """Update NGC blueprint with dynamic benchmark metrics."""
        blueprint = self.load_blueprint()
        
        if not blueprint:
            logger.error("Failed to load blueprint")
            return
        
        # Ensure benchmarks section exists
        if "benchmarks" not in blueprint:
            blueprint["benchmarks"] = {}
        
        if "t4" not in blueprint["benchmarks"]:
            blueprint["benchmarks"]["t4"] = {}
        
        t4_benchmarks = blueprint["benchmarks"]["t4"]
        
        # Update inference latency benchmarks
        if "inference_latency" not in t4_benchmarks:
            t4_benchmarks["inference_latency"] = {}
        
        # Get common batch sizes
        batch_sizes = set()
        for metric in self.metrics.values():
            batch_sizes.update(metric.get_batch_sizes())
        
        batch_sizes = sorted(list(batch_sizes))
        if not batch_sizes:
            batch_sizes = [1, 8, 32]  # Default batch sizes
        
        # Update latency metrics by batch size
        latency_metric = self.metrics["inference_latency"]
        for batch_size in batch_sizes:
            batch_values = latency_metric.get_by_batch_size(batch_size)
            if batch_values:
                # Use median latency (p50) as representative value
                latency_value = float(np.median(batch_values))
                t4_benchmarks["inference_latency"][f"batch{batch_size}"] = latency_metric.get_formatted_value(latency_value)
        
        # Update throughput metrics by batch size
        if "throughput" not in t4_benchmarks:
            t4_benchmarks["throughput"] = {}
        
        throughput_metric = self.metrics["throughput"]
        for batch_size in batch_sizes:
            batch_values = throughput_metric.get_by_batch_size(batch_size)
            if batch_values:
                # Use p90 throughput as representative value (more optimistic but achievable)
                throughput_value = float(np.percentile(batch_values, 90))
                t4_benchmarks["throughput"][f"batch{batch_size}"] = throughput_metric.get_formatted_value(throughput_value)
        
        # Update memory usage
        memory_metric = self.metrics["memory_usage"]
        if memory_metric.values:
            # Use p95 memory usage as representative value
            memory_value = memory_metric.get_percentile(95)
            t4_benchmarks["memoryUsage"] = memory_metric.get_formatted_value(memory_value)
        
        # Update tensorrt speedup
        speedup_metric = self.metrics["tensorrt_speedup"]
        if speedup_metric.values:
            # Use average speedup as representative value
            speedup_value = speedup_metric.get_average()
            t4_benchmarks["tensorrtSpeedup"] = speedup_metric.get_formatted_value(speedup_value)
        
        # Save updated blueprint
        self.save_blueprint(blueprint)
        
        logger.info("Updated NGC blueprint benchmarks with dynamic metrics")
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """
        Get benchmark report.
        
        Returns:
            Dictionary with benchmark report
        """
        report = {
            "metrics": {},
            "batch_sizes": {},
            "recommendations": []
        }
        
        # Summary metrics
        for metric_name, metric in self.metrics.items():
            if metric.values:
                report["metrics"][metric_name] = {
                    "count": len(metric.values),
                    "mean": metric.get_average(),
                    "p50": metric.get_percentile(50),
                    "p90": metric.get_percentile(90),
                    "p95": metric.get_percentile(95),
                    "min": float(np.min(metric.values)),
                    "max": float(np.max(metric.values)),
                    "unit": metric.unit
                }
        
        # Metrics by batch size
        batch_sizes = set()
        for metric in self.metrics.values():
            batch_sizes.update(metric.get_batch_sizes())
        
        for batch_size in sorted(batch_sizes):
            report["batch_sizes"][str(batch_size)] = {}
            
            for metric_name, metric in self.metrics.items():
                batch_values = metric.get_by_batch_size(batch_size)
                if batch_values:
                    report["batch_sizes"][str(batch_size)][metric_name] = {
                        "count": len(batch_values),
                        "mean": float(np.mean(batch_values)),
                        "p50": float(np.median(batch_values)),
                        "p90": float(np.percentile(batch_values, 90)),
                        "unit": metric.unit
                    }
        
        # Generate recommendations
        recommendations = []
        
        # Check if we have enough samples
        if any(len(metric.values) < 10 for metric in self.metrics.values()):
            recommendations.append("Collect more benchmark samples for more accurate metrics")
        
        # Check if we have diverse batch sizes
        if len(batch_sizes) < 3:
            recommendations.append("Benchmark with more diverse batch sizes for better scaling insights")
        
        # Check TensorRT speedup
        speedup_metric = self.metrics["tensorrt_speedup"]
        if speedup_metric.values and speedup_metric.get_average() < 2.0:
            recommendations.append("TensorRT speedup is below 2x, consider optimizing TensorRT settings")
        
        # Check latency variation
        latency_metric = self.metrics["inference_latency"]
        if latency_metric.values:
            p50 = latency_metric.get_percentile(50)
            p95 = latency_metric.get_percentile(95)
            if p95 / p50 > 2.0:
                recommendations.append("High latency variation observed, consider stabilizing inference")
        
        report["recommendations"] = recommendations
        
        return report


def create_benchmark_manager(blueprint_file: str, 
                           metrics_file: str = "/tmp/ngc_benchmark_metrics.json") -> BenchmarkManager:
    """
    Create benchmark manager.
    
    Args:
        blueprint_file: Path to NGC blueprint file
        metrics_file: Path to save metrics
        
    Returns:
        BenchmarkManager instance
    """
    return BenchmarkManager(
        blueprint_file=blueprint_file,
        metrics_file=metrics_file
    )