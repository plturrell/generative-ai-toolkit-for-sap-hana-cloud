"""
T4 GPU Parameter Auto-Tuner for SAP HANA Cloud Generative AI Toolkit

This module provides dynamic parameter tuning capabilities for T4 GPU optimization,
automatically learning optimal settings based on workload characteristics, hardware
capabilities, and observed performance patterns.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import numpy as np
import torch
from datetime import datetime, timedelta

from api.t4_gpu_optimizer import T4GPUConfig, T4MemoryManager, T4TensorRTOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterHistory:
    """Class to track parameter history and performance."""
    
    def __init__(self, param_name: str, window_size: int = 100):
        """
        Initialize parameter history tracker.
        
        Args:
            param_name: Name of the parameter
            window_size: Number of data points to keep in history
        """
        self.param_name = param_name
        self.window_size = window_size
        self.values = []
        self.performance_metrics = []
        self.timestamps = []
    
    def add_datapoint(self, value: Any, performance_metric: float):
        """
        Add parameter value and associated performance metric.
        
        Args:
            value: Parameter value
            performance_metric: Performance metric (higher is better)
        """
        self.values.append(value)
        self.performance_metrics.append(performance_metric)
        self.timestamps.append(datetime.now())
        
        # Trim history if exceeds window size
        if len(self.values) > self.window_size:
            self.values.pop(0)
            self.performance_metrics.pop(0)
            self.timestamps.pop(0)
    
    def get_best_value(self) -> Any:
        """Get parameter value with best performance metric."""
        if not self.performance_metrics:
            return None
            
        best_idx = np.argmax(self.performance_metrics)
        return self.values[best_idx]
    
    def get_recent_best_value(self, hours: int = 24) -> Any:
        """
        Get best parameter value from recent history.
        
        Args:
            hours: Look back window in hours
            
        Returns:
            Parameter value with best performance in recent history
        """
        if not self.timestamps:
            return None
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_indices = [i for i, ts in enumerate(self.timestamps) if ts > cutoff_time]
        
        if not recent_indices:
            return self.get_best_value()
            
        recent_performance = [self.performance_metrics[i] for i in recent_indices]
        best_recent_idx = recent_indices[np.argmax(recent_performance)]
        
        return self.values[best_recent_idx]
    
    def get_trend(self) -> str:
        """Get trend of parameter performance."""
        if len(self.performance_metrics) < 2:
            return "insufficient_data"
            
        # Simple linear regression to determine trend
        x = np.arange(len(self.performance_metrics))
        y = np.array(self.performance_metrics)
        
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"


class WorkloadProfile:
    """Class to characterize workload patterns for parameter tuning."""
    
    def __init__(self, name: str):
        """
        Initialize workload profile.
        
        Args:
            name: Profile name
        """
        self.name = name
        self.input_shapes = {}
        self.output_shapes = {}
        self.batch_sizes = []
        self.sequence_lengths = []
        self.memory_usage = []
        self.compute_intensity = []
        self.request_frequency = []
        self.timestamps = []
        
    def add_observation(self, 
                         input_shape: Dict[str, List[int]],
                         output_shape: Dict[str, List[int]],
                         batch_size: int,
                         sequence_length: int,
                         memory_usage: float,
                         compute_intensity: float,
                         request_frequency: float):
        """
        Add workload observation.
        
        Args:
            input_shape: Input tensor shapes
            output_shape: Output tensor shapes
            batch_size: Batch size used
            sequence_length: Sequence length processed
            memory_usage: Memory usage in bytes
            compute_intensity: Compute intensity metric
            request_frequency: Requests per second
        """
        self.input_shapes[datetime.now()] = input_shape
        self.output_shapes[datetime.now()] = output_shape
        self.batch_sizes.append(batch_size)
        self.sequence_lengths.append(sequence_length)
        self.memory_usage.append(memory_usage)
        self.compute_intensity.append(compute_intensity)
        self.request_frequency.append(request_frequency)
        self.timestamps.append(datetime.now())
        
    def get_typical_batch_size(self) -> int:
        """Get typical batch size for this workload."""
        if not self.batch_sizes:
            return 1
        
        # Use 90th percentile as typical batch size to account for outliers
        return int(np.percentile(self.batch_sizes, 90))
    
    def get_typical_sequence_length(self) -> int:
        """Get typical sequence length for this workload."""
        if not self.sequence_lengths:
            return 512  # Default
        
        # Use median sequence length as typical
        return int(np.median(self.sequence_lengths))
    
    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage for this workload."""
        if not self.memory_usage:
            return 0
        
        # Use 95th percentile to account for outliers
        return float(np.percentile(self.memory_usage, 95))
    
    def get_average_request_frequency(self, hours: int = 1) -> float:
        """
        Get average request frequency over recent period.
        
        Args:
            hours: Look back window in hours
            
        Returns:
            Average requests per second
        """
        if not self.request_frequency:
            return 0
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_indices = [i for i, ts in enumerate(self.timestamps) if ts > cutoff_time]
        
        if not recent_indices:
            return float(np.mean(self.request_frequency))
            
        recent_frequency = [self.request_frequency[i] for i in recent_indices]
        return float(np.mean(recent_frequency))
    
    def is_memory_bound(self) -> bool:
        """Determine if workload is memory-bound or compute-bound."""
        if not self.memory_usage or not self.compute_intensity:
            return False
            
        # Calculate correlation between memory usage and compute intensity
        correlation = np.corrcoef(self.memory_usage, self.compute_intensity)[0, 1]
        
        # If memory usage increases more than compute intensity, workload is memory-bound
        return correlation > 0.7


class T4ParameterAutoTuner:
    """
    Auto-tuner for T4 GPU parameters that dynamically adjusts
    settings based on workload characteristics and performance feedback.
    """
    
    def __init__(self, 
                 config_path: str = "/tmp/t4_autotuner.json",
                 history_window: int = 100,
                 update_interval: int = 300,
                 enable_auto_update: bool = True):
        """
        Initialize auto-tuner.
        
        Args:
            config_path: Path to save/load tuned parameters
            history_window: Number of data points to keep in history
            update_interval: Seconds between parameter updates
            enable_auto_update: Whether to auto-update parameters
        """
        self.config_path = config_path
        self.history_window = history_window
        self.update_interval = update_interval
        self.enable_auto_update = enable_auto_update
        
        # Parameter history trackers
        self.param_history = {
            "fp16_mode": ParameterHistory("fp16_mode", window_size=history_window),
            "int8_mode": ParameterHistory("int8_mode", window_size=history_window),
            "max_workspace_size": ParameterHistory("max_workspace_size", window_size=history_window),
            "memory_fraction": ParameterHistory("memory_fraction", window_size=history_window),
            "max_batch_size": ParameterHistory("max_batch_size", window_size=history_window),
            "optimization_level": ParameterHistory("optimization_level", window_size=history_window)
        }
        
        # Workload profiles
        self.workload_profiles = {}
        
        # Current parameters
        self.current_params = self._load_parameters()
        
        # Optimization metrics
        self.metrics = {
            "inference_latency": [],
            "throughput": [],
            "memory_usage": [],
            "tensorrt_speedup": []
        }
        
        # Auto-update thread
        self.update_thread = None
        if enable_auto_update:
            self._start_auto_update()
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load parameters from saved config or use defaults."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded tuned parameters from {self.config_path}")
                return params
        except Exception as e:
            logger.warning(f"Error loading parameters: {str(e)}")
        
        # Default parameters
        return {
            "fp16_mode": True,
            "int8_mode": False,
            "max_workspace_size": 4 * (1 << 30),  # 4GB
            "memory_fraction": 0.8,
            "max_batch_size": 64,
            "optimization_level": 3
        }
    
    def _save_parameters(self):
        """Save current parameters to config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.current_params, f, indent=2)
            logger.info(f"Saved tuned parameters to {self.config_path}")
        except Exception as e:
            logger.warning(f"Error saving parameters: {str(e)}")
    
    def _start_auto_update(self):
        """Start background thread for auto-updating parameters."""
        if self.update_thread is not None and self.update_thread.is_alive():
            return
            
        self.update_thread = threading.Thread(target=self._auto_update_thread, daemon=True)
        self.update_thread.start()
        logger.info("Started auto-update thread for T4 GPU parameters")
    
    def _auto_update_thread(self):
        """Background thread for automatic parameter updates."""
        while True:
            try:
                # Update parameters based on history
                self.update_parameters()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in auto-update thread: {str(e)}")
                time.sleep(60)  # Wait a bit longer if there was an error
    
    def update_parameters(self):
        """Update parameters based on performance history."""
        if not any(h.performance_metrics for h in self.param_history.values()):
            logger.info("Insufficient performance data for parameter tuning")
            return
        
        # Get best parameter values from history
        updates = {}
        
        for param_name, history in self.param_history.items():
            best_value = history.get_recent_best_value()
            if best_value is not None and best_value != self.current_params.get(param_name):
                updates[param_name] = best_value
        
        if not updates:
            logger.debug("No parameter updates needed")
            return
        
        # Update parameters
        for param_name, value in updates.items():
            logger.info(f"Updating parameter {param_name}: {self.current_params.get(param_name)} -> {value}")
            self.current_params[param_name] = value
        
        # Save updated parameters
        self._save_parameters()
        
        # Apply changes to running optimizers
        self._apply_parameter_changes()
    
    def _apply_parameter_changes(self):
        """Apply parameter changes to environment variables for running processes."""
        # Update environment variables
        for param_name, value in self.current_params.items():
            env_name = f"T4_GPU_{param_name.upper()}"
            os.environ[env_name] = str(value)
        
        logger.info("Applied parameter changes to environment variables")
    
    def record_inference_performance(self, 
                                     model_name: str,
                                     input_shape: Dict[str, List[int]],
                                     output_shape: Dict[str, List[int]],
                                     batch_size: int,
                                     sequence_length: int,
                                     latency_ms: float,
                                     throughput: float,
                                     memory_usage: float,
                                     tensorrt_speedup: float = 1.0):
        """
        Record inference performance metrics.
        
        Args:
            model_name: Name of the model
            input_shape: Input tensor shapes
            output_shape: Output tensor shapes
            batch_size: Batch size used
            sequence_length: Sequence length processed
            latency_ms: Inference latency in milliseconds
            throughput: Throughput in samples/second
            memory_usage: Memory usage in bytes
            tensorrt_speedup: Speedup from TensorRT optimization
        """
        # Create or get workload profile
        if model_name not in self.workload_profiles:
            self.workload_profiles[model_name] = WorkloadProfile(model_name)
        
        profile = self.workload_profiles[model_name]
        
        # Calculate compute intensity (throughput/memory)
        compute_intensity = throughput / max(1, memory_usage / (1024 * 1024 * 1024))  # per GB
        
        # Update workload profile
        profile.add_observation(
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            memory_usage=memory_usage,
            compute_intensity=compute_intensity,
            request_frequency=throughput / batch_size
        )
        
        # Update metrics
        self.metrics["inference_latency"].append(latency_ms)
        self.metrics["throughput"].append(throughput)
        self.metrics["memory_usage"].append(memory_usage)
        self.metrics["tensorrt_speedup"].append(tensorrt_speedup)
        
        # Calculate performance score (higher is better)
        # Prioritize throughput and TensorRT speedup, penalize latency
        performance_score = (throughput * tensorrt_speedup) / max(1.0, latency_ms / 10.0)
        
        # Record parameter performance
        for param_name, value in self.current_params.items():
            if param_name in self.param_history:
                self.param_history[param_name].add_datapoint(value, performance_score)
        
        # Log performance metrics
        logger.info(f"Recorded inference performance for {model_name}: "
                   f"latency={latency_ms:.2f}ms, throughput={throughput:.2f} samples/s, "
                   f"speedup={tensorrt_speedup:.2f}x, score={performance_score:.2f}")
    
    def get_optimized_config(self, model_name: str = None) -> T4GPUConfig:
        """
        Get optimized T4 GPU configuration.
        
        Args:
            model_name: Name of the model for workload-specific optimization
            
        Returns:
            Optimized T4 GPU configuration
        """
        # Start with current parameters
        config_params = self.current_params.copy()
        
        # Apply workload-specific optimizations if model_name is provided
        if model_name and model_name in self.workload_profiles:
            profile = self.workload_profiles[model_name]
            
            # Adjust batch size based on workload
            if "max_batch_size" in config_params:
                typical_batch = profile.get_typical_batch_size()
                # Add headroom for growth
                config_params["max_batch_size"] = max(typical_batch * 2, config_params["max_batch_size"])
            
            # Adjust memory fraction based on memory usage patterns
            if "memory_fraction" in config_params and profile.is_memory_bound():
                # If memory bound, allocate more memory
                config_params["memory_fraction"] = min(0.95, config_params["memory_fraction"] * 1.1)
            
            # Adjust precision based on model and compute patterns
            if "int8_mode" in config_params and tensorrt_speedup > 2.0:
                # If we get good speedup, enable INT8 mode
                config_params["int8_mode"] = True
        
        # Create config
        return T4GPUConfig(**config_params)
    
    def get_optimal_batch_size(self, 
                              model_name: str,
                              input_size_per_sample: int,
                              output_size_per_sample: int,
                              processing_size_per_sample: Optional[int] = None) -> int:
        """
        Get optimal batch size based on model and memory requirements.
        
        Args:
            model_name: Name of the model
            input_size_per_sample: Memory required for one input sample (bytes)
            output_size_per_sample: Memory required for one output sample (bytes)
            processing_size_per_sample: Additional memory for processing (bytes)
            
        Returns:
            Optimal batch size
        """
        # Use workload profile if available
        if model_name in self.workload_profiles:
            profile = self.workload_profiles[model_name]
            
            # If we have profiling data, use it to estimate processing size
            if not processing_size_per_sample and profile.memory_usage:
                # Estimate processing memory based on observed usage patterns
                peak_memory = profile.get_peak_memory_usage()
                typical_batch = profile.get_typical_batch_size()
                typical_sequence = profile.get_typical_sequence_length()
                
                # Calculate typical input and output size
                typical_input_size = input_size_per_sample * (typical_sequence / max(1, sequence_length))
                typical_output_size = output_size_per_sample * (typical_sequence / max(1, sequence_length))
                
                # Estimate processing size
                if typical_batch > 0:
                    estimated_processing = (peak_memory / typical_batch) - typical_input_size - typical_output_size
                    processing_size_per_sample = max(estimated_processing, input_size_per_sample * 5)
        
        # Create memory manager with optimized config
        memory_manager = T4MemoryManager(self.get_optimized_config(model_name))
        
        # Get optimal batch size
        return memory_manager.calculate_optimal_batch_size(
            input_size_per_sample,
            output_size_per_sample,
            processing_size_per_sample
        )
    
    def get_auto_calibration_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get auto-calibration configuration for INT8 optimization.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Calibration configuration
        """
        config = {
            "calibration_cache": "/tmp/t4_calibration_cache",
            "algorithm": "entropy",
            "batch_size": 8,
            "num_batches": 100,
            "quantization_noise": 0.05
        }
        
        # If we have workload profile, optimize calibration settings
        if model_name in self.workload_profiles:
            profile = self.workload_profiles[model_name]
            
            # Adjust batch size based on workload
            typical_batch = profile.get_typical_batch_size()
            config["batch_size"] = min(typical_batch, 32)  # Keep reasonable for calibration
            
            # Adjust algorithm based on compute patterns
            if profile.is_memory_bound():
                # For memory-bound workloads, use percentile for better accuracy
                config["algorithm"] = "percentile"
                config["percentile"] = 99.99
            else:
                # For compute-bound workloads, use entropy for better performance
                config["algorithm"] = "entropy"
        
        return config
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report of parameter tuning.
        
        Returns:
            Performance report
        """
        report = {
            "current_parameters": self.current_params,
            "metric_averages": {},
            "workload_profiles": {},
            "parameter_trends": {},
            "recommendation": {}
        }
        
        # Calculate metric averages
        for metric_name, values in self.metrics.items():
            if values:
                report["metric_averages"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Summarize workload profiles
        for name, profile in self.workload_profiles.items():
            report["workload_profiles"][name] = {
                "typical_batch_size": profile.get_typical_batch_size(),
                "typical_sequence_length": profile.get_typical_sequence_length(),
                "peak_memory_usage_gb": profile.get_peak_memory_usage() / (1024**3),
                "avg_request_frequency": profile.get_average_request_frequency(),
                "is_memory_bound": profile.is_memory_bound()
            }
        
        # Parameter trends
        for param_name, history in self.param_history.items():
            report["parameter_trends"][param_name] = {
                "trend": history.get_trend(),
                "best_value": history.get_best_value(),
                "current_value": self.current_params.get(param_name)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check for improvement opportunities
        for param_name, history in self.param_history.items():
            best_value = history.get_best_value()
            current_value = self.current_params.get(param_name)
            
            if best_value is not None and best_value != current_value:
                recommendations.append(f"Change {param_name} from {current_value} to {best_value}")
        
        # Check for memory utilization
        avg_memory_usage = np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        
        if total_memory > 0:
            memory_utilization = avg_memory_usage / total_memory
            if memory_utilization < 0.5:
                recommendations.append(f"Memory utilization is low ({memory_utilization:.2%}), consider increasing batch size")
            elif memory_utilization > 0.9:
                recommendations.append(f"Memory utilization is high ({memory_utilization:.2%}), consider model quantization")
        
        report["recommendation"] = {
            "suggestions": recommendations,
            "auto_update_enabled": self.enable_auto_update
        }
        
        return report


def create_autotuner(config_path: str = "/tmp/t4_autotuner.json", 
                    enable_auto_update: bool = True) -> T4ParameterAutoTuner:
    """
    Create T4 parameter auto-tuner.
    
    Args:
        config_path: Path to save/load tuned parameters
        enable_auto_update: Whether to auto-update parameters
        
    Returns:
        T4 parameter auto-tuner instance
    """
    return T4ParameterAutoTuner(
        config_path=config_path,
        enable_auto_update=enable_auto_update
    )


# Singleton instance
_autotuner_instance = None

def get_autotuner() -> T4ParameterAutoTuner:
    """Get or create singleton auto-tuner instance."""
    global _autotuner_instance
    
    if _autotuner_instance is None:
        _autotuner_instance = create_autotuner()
        
    return _autotuner_instance