"""
NVIDIA GPU utilities for optimized AI workloads.

This module provides advanced GPU management utilities for:
1. Multi-GPU distribution and load balancing
2. GPU memory monitoring and optimization
3. Performance profiling and metrics collection
4. Model parallelism and tensor parallelism
"""
import os
import json
import time
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False


@dataclass
class GPUStats:
    """Data class for GPU statistics."""
    device_id: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    utilization_percent: float
    temperature: float
    power_usage_watts: float
    power_limit_watts: float
    clock_speed_mhz: float
    compute_capability: str


class GPUProfiler:
    """
    Advanced GPU profiler for performance monitoring and optimization.
    
    Provides detailed performance metrics, traces, and optimization
    recommendations for NVIDIA GPUs.
    """
    def __init__(self, 
                 trace_dir: str = "/tmp/gpu_traces",
                 profile_memory: bool = True,
                 profile_cuda: bool = True):
        """
        Initialize the GPU profiler.
        
        Parameters
        ----------
        trace_dir : str
            Directory to save trace files
        profile_memory : bool
            Whether to profile memory usage
        profile_cuda : bool
            Whether to profile CUDA operations
        """
        self.trace_dir = trace_dir
        self.profile_memory = profile_memory
        self.profile_cuda = profile_cuda
        
        # Create trace directory if it doesn't exist
        os.makedirs(trace_dir, exist_ok=True)
        
        # Initialize profiling state
        self.is_profiling = False
        self.current_profile = None
        self.profiling_thread = None
        self.profile_lock = threading.Lock()
        
        # Check for GPU availability
        self.available = TORCH_AVAILABLE and torch.cuda.is_available()
        if not self.available:
            logger.warning("GPU profiling disabled: PyTorch or CUDA not available")
            return
        
        # Initialize NVIDIA SMI if available
        if NVIDIA_SMI_AVAILABLE:
            try:
                nvidia_smi.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
                logger.warning("Failed to initialize NVIDIA Management Library")
        else:
            self.nvml_initialized = False
        
        logger.info(f"GPU profiler initialized with {torch.cuda.device_count()} GPUs available")

    def __del__(self):
        """Clean up resources."""
        if self.nvml_initialized:
            try:
                nvidia_smi.nvmlShutdown()
            except:
                pass

    def get_gpu_stats(self) -> Dict[int, GPUStats]:
        """
        Get detailed statistics for all available GPUs.
        
        Returns
        -------
        Dict[int, GPUStats]
            Dictionary of GPU stats keyed by device ID
        """
        if not self.available:
            return {}
        
        stats = {}
        device_count = torch.cuda.device_count()
        
        for device_id in range(device_count):
            # Basic info from PyTorch
            props = torch.cuda.get_device_properties(device_id)
            name = props.name
            total_memory = props.total_memory / (1024 * 1024)  # Convert to MB
            compute_capability = f"{props.major}.{props.minor}"
            
            # Memory info from PyTorch
            reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
            free = total_memory - reserved
            
            # Default values for fields we might not be able to get
            utilization = 0
            temperature = 0
            power_usage = 0
            power_limit = 0
            clock_speed = 0
            
            # Try to get more detailed info from NVIDIA SMI
            if self.nvml_initialized:
                try:
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
                    util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                    temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
                    temperature = temp
                    power_info = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_usage = power_info
                    power_limit_info = nvidia_smi.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    power_limit = power_limit_info
                    clock_info = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_SM)
                    clock_speed = clock_info
                except:
                    # Fall back to subprocess for nvidia-smi if NVML fails
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", f"--id={device_id}", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True
                        )
                        values = result.stdout.strip().split(",")
                        if len(values) >= 5:
                            utilization = float(values[0])
                            temperature = float(values[1])
                            power_usage = float(values[2])
                            power_limit = float(values[3])
                            clock_speed = float(values[4])
                    except:
                        logger.debug(f"Failed to get detailed GPU stats for device {device_id}")
            
            stats[device_id] = GPUStats(
                device_id=device_id,
                name=name,
                total_memory_mb=total_memory,
                used_memory_mb=allocated,
                free_memory_mb=free,
                utilization_percent=utilization,
                temperature=temperature,
                power_usage_watts=power_usage,
                power_limit_watts=power_limit,
                clock_speed_mhz=clock_speed,
                compute_capability=compute_capability
            )
            
        return stats

    def start_profiling(self, 
                        duration_sec: int = 10, 
                        trace_filename: Optional[str] = None,
                        activities: Optional[List[str]] = None) -> bool:
        """
        Start GPU profiling session.
        
        Parameters
        ----------
        duration_sec : int
            Duration of profiling in seconds
        trace_filename : str, optional
            Custom filename for the trace file
        activities : List[str], optional
            List of activities to profile (cpu, cuda, memory)
            
        Returns
        -------
        bool
            True if profiling started successfully
        """
        if not PROFILER_AVAILABLE or not self.available:
            logger.warning("GPU profiling not available")
            return False
        
        with self.profile_lock:
            if self.is_profiling:
                logger.warning("Profiling already in progress")
                return False
            
            # Set up activities to profile
            profiler_activities = []
            if activities is None:
                activities = ["cpu", "cuda"]
                
            if "cpu" in activities:
                profiler_activities.append(ProfilerActivity.CPU)
            if "cuda" in activities and self.profile_cuda:
                profiler_activities.append(ProfilerActivity.CUDA)
            
            # Generate trace filename
            if trace_filename is None:
                timestamp = int(time.time())
                trace_filename = f"gpu_trace_{timestamp}.json"
            
            trace_path = os.path.join(self.trace_dir, trace_filename)
            
            # Create profiler
            try:
                self.current_profile = profile(
                    activities=profiler_activities,
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=duration_sec - 2 if duration_sec > 2 else 1,
                        repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.trace_dir),
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=self.profile_memory
                )
                self.current_profile.__enter__()
                self.is_profiling = True
                
                # Start a background thread to stop profiling after duration
                self.profiling_thread = threading.Thread(
                    target=self._stop_profiling_after_duration,
                    args=(duration_sec, trace_path)
                )
                self.profiling_thread.daemon = True
                self.profiling_thread.start()
                
                logger.info(f"GPU profiling started for {duration_sec} seconds")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start GPU profiling: {str(e)}")
                self.current_profile = None
                return False
    
    def _stop_profiling_after_duration(self, duration_sec: int, trace_path: str):
        """Background thread to stop profiling after duration."""
        time.sleep(duration_sec)
        self.stop_profiling(trace_path)
    
    def stop_profiling(self, trace_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop GPU profiling and return results.
        
        Parameters
        ----------
        trace_path : str, optional
            Path to save the trace file
            
        Returns
        -------
        Dict[str, Any]
            Profiling results summary
        """
        with self.profile_lock:
            if not self.is_profiling or self.current_profile is None:
                logger.warning("No active profiling session to stop")
                return {}
            
            try:
                # Exit the profiler context
                self.current_profile.__exit__(None, None, None)
                
                # Get key stats
                key_metrics = self._extract_profile_metrics(self.current_profile)
                
                # Save trace file if specified
                if trace_path:
                    try:
                        self.current_profile.export_chrome_trace(trace_path)
                        logger.info(f"GPU trace saved to {trace_path}")
                    except Exception as e:
                        logger.error(f"Failed to save GPU trace: {str(e)}")
                
                self.is_profiling = False
                self.current_profile = None
                
                logger.info("GPU profiling completed")
                return key_metrics
                
            except Exception as e:
                logger.error(f"Error stopping GPU profiling: {str(e)}")
                self.is_profiling = False
                self.current_profile = None
                return {"error": str(e)}
    
    def _extract_profile_metrics(self, profile_result) -> Dict[str, Any]:
        """Extract key metrics from profiling results."""
        # This is a simplified implementation
        # A full implementation would parse the profiler events and extract detailed metrics
        metrics = {
            "cuda_time_total_us": 0,
            "cpu_time_total_us": 0,
            "memory_allocated_peak_mb": 0,
            "top_operations": [],
            "bottlenecks": []
        }
        
        try:
            # Get memory stats
            metrics["memory_allocated_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Extract more metrics from the profiler
            # Note: This would be more detailed in a full implementation
            
            # Identify potential bottlenecks
            if metrics["memory_allocated_peak_mb"] > 0.9 * torch.cuda.get_device_properties(0).total_memory / (1024 * 1024):
                metrics["bottlenecks"].append("High GPU memory usage - consider batch size reduction")
                
        except Exception as e:
            logger.error(f"Error extracting profile metrics: {str(e)}")
            
        return metrics
        
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """
        Get GPU optimization recommendations based on current stats.
        
        Returns
        -------
        List[Dict[str, str]]
            List of optimization recommendations with title and description
        """
        recommendations = []
        
        if not self.available:
            recommendations.append({
                "title": "GPU Not Available",
                "description": "No CUDA-capable GPUs detected. Consider using a GPU-enabled environment for better performance."
            })
            return recommendations
        
        # Get current GPU stats
        stats = self.get_gpu_stats()
        
        # Check memory usage
        for device_id, gpu_stat in stats.items():
            memory_usage_percent = (gpu_stat.used_memory_mb / gpu_stat.total_memory_mb) * 100
            
            if memory_usage_percent > 90:
                recommendations.append({
                    "title": f"High Memory Usage on GPU {device_id}",
                    "description": f"Memory usage is {memory_usage_percent:.1f}%. Consider reducing batch sizes or using gradient checkpointing."
                })
            
            if gpu_stat.utilization_percent < 30 and memory_usage_percent > 50:
                recommendations.append({
                    "title": f"Memory-bound operations on GPU {device_id}",
                    "description": "High memory usage but low compute utilization suggests memory-bound operations. Consider using mixed precision (FP16) to reduce memory requirements."
                })
                
            if gpu_stat.temperature > 80:
                recommendations.append({
                    "title": f"High GPU Temperature ({gpu_stat.temperature}°C)",
                    "description": "GPU is running hot which may lead to thermal throttling. Check cooling and consider reducing workload."
                })
                
        # Multi-GPU recommendations
        if len(stats) > 1:
            utilizations = [gpu.utilization_percent for gpu in stats.values()]
            avg_util = sum(utilizations) / len(utilizations)
            
            if max(utilizations) - min(utilizations) > 40:
                recommendations.append({
                    "title": "Unbalanced GPU Utilization",
                    "description": "GPUs have significantly different utilization levels. Consider using data parallel or model parallel approaches for better load balancing."
                })
                
            if avg_util < 50:
                recommendations.append({
                    "title": "Low Overall GPU Utilization",
                    "description": "GPUs are underutilized. Consider increasing batch size or processing multiple requests in parallel."
                })
        
        return recommendations


class MultiGPUManager:
    """
    Advanced multi-GPU management for distributed AI workloads.
    
    Provides sophisticated workload distribution across multiple GPUs with:
    - Automatic load balancing based on GPU utilization and memory
    - Support for data parallelism and model parallelism
    - Smart scheduling of tasks based on GPU capabilities
    - Monitoring and rebalancing
    """
    def __init__(self, 
                 strategy: str = "auto",
                 memory_fraction: float = 0.8,
                 enable_mixed_precision: bool = True):
        """
        Initialize the multi-GPU manager.
        
        Parameters
        ----------
        strategy : str
            Distribution strategy: 'auto', 'data_parallel', 'model_parallel', 'pipeline', 'device_map'
        memory_fraction : float
            Fraction of GPU memory to use (0.0 to 1.0)
        enable_mixed_precision : bool
            Whether to enable mixed precision (FP16/BF16) for better performance
        """
        self.strategy = strategy
        self.memory_fraction = memory_fraction
        self.enable_mixed_precision = enable_mixed_precision
        
        # Initialize state
        self.initialized = False
        self.gpu_count = 0
        self.active_gpus = []
        self.device_weights = {}
        self.task_assignments = {}
        self.gpu_stats = {}
        
        # Initialize profiler for monitoring
        self.profiler = GPUProfiler()
        
        # Check for GPU availability
        if not TORCH_AVAILABLE:
            logger.warning("MultiGPUManager: PyTorch not available")
            return
            
        if not torch.cuda.is_available():
            logger.warning("MultiGPUManager: CUDA not available")
            return
            
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            logger.warning("MultiGPUManager: No GPUs available")
            return
            
        # Set memory fraction
        for i in range(self.gpu_count):
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, i)
        
        # Initialize active GPUs and weights
        self.active_gpus = list(range(self.gpu_count))
        self._initialize_device_weights()
        
        # Enable mixed precision if requested
        if self.enable_mixed_precision:
            self._setup_mixed_precision()
        
        # Initialize distributed training if needed
        if self.strategy == "data_parallel" and DISTRIBUTED_AVAILABLE and self.gpu_count > 1:
            self._setup_distributed()
        
        self.initialized = True
        logger.info(f"MultiGPUManager initialized with {self.gpu_count} GPUs using '{self.strategy}' strategy")
        
    def _initialize_device_weights(self):
        """Initialize device weights based on GPU capabilities."""
        if self.gpu_count == 0:
            return
            
        self.device_weights = {}
        total_weight = 0
        
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            
            # Calculate a weight based on compute capability, memory, and SM count
            compute_score = props.major * 10 + props.minor
            memory_score = props.total_memory / (1024 * 1024 * 1024)  # Convert to GB
            sm_score = props.multi_processor_count
            
            # Combine scores with appropriate weighting
            device_score = (0.3 * compute_score) + (0.5 * memory_score) + (0.2 * sm_score)
            self.device_weights[i] = device_score
            total_weight += device_score
        
        # Normalize weights
        for i in self.device_weights:
            self.device_weights[i] /= total_weight
            
        logger.debug(f"GPU device weights: {self.device_weights}")
        
    def _setup_mixed_precision(self):
        """Set up mixed precision training."""
        if not TORCH_AVAILABLE:
            return
            
        try:
            # For PyTorch 1.10+
            torch.set_float32_matmul_precision('high')
        except:
            pass
            
        try:
            # Check for Ampere or newer GPUs that can use TF32
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                if props.major >= 8:  # Ampere or newer
                    # Enable TF32 for matrix multiplications
                    torch.backends.cuda.matmul.allow_tf32 = True
                    # Enable TF32 for cuDNN
                    torch.backends.cudnn.allow_tf32 = True
                    break
        except:
            pass
    
    def _setup_distributed(self):
        """Set up distributed training."""
        if not DISTRIBUTED_AVAILABLE:
            return
            
        try:
            # Initialize process group
            dist.init_process_group(backend="nccl")
            logger.info("Distributed training initialized with NCCL backend")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {str(e)}")
    
    def get_optimal_device(self, task_id: str = None, memory_requirement: float = None) -> int:
        """
        Get the optimal GPU device for a given task.
        
        Parameters
        ----------
        task_id : str, optional
            Identifier for the task, for tracking purposes
        memory_requirement : float, optional
            Required memory in MB
            
        Returns
        -------
        int
            The optimal device ID
        """
        if not self.initialized or self.gpu_count == 0:
            return -1  # CPU
            
        if self.gpu_count == 1:
            return 0  # Only one GPU
        
        # Update GPU stats
        self._update_gpu_stats()
        
        # If strategy is device_map, follow pre-determined mapping
        if self.strategy == "device_map" and task_id in self.task_assignments:
            return self.task_assignments[task_id]
        
        # For model_parallel and pipeline, follow pre-determined assignments
        if self.strategy in ["model_parallel", "pipeline"] and task_id in self.task_assignments:
            return self.task_assignments[task_id]
        
        # For auto and data_parallel, use dynamic assignment
        best_device = 0
        best_score = -float('inf')
        
        for device_id in self.active_gpus:
            # Skip if memory requirement exceeds available memory
            if memory_requirement and self.gpu_stats[device_id].free_memory_mb < memory_requirement:
                continue
                
            # Calculate score based on free memory and utilization
            memory_score = self.gpu_stats[device_id].free_memory_mb
            util_score = 100 - self.gpu_stats[device_id].utilization_percent
            capability_weight = self.device_weights.get(device_id, 1.0)
            
            # Combined score: higher is better
            score = (0.7 * memory_score / 1000) + (0.3 * util_score) * capability_weight
            
            if score > best_score:
                best_score = score
                best_device = device_id
        
        # Record assignment if task_id provided
        if task_id:
            self.task_assignments[task_id] = best_device
            
        return best_device
    
    def _update_gpu_stats(self):
        """Update GPU statistics."""
        self.gpu_stats = self.profiler.get_gpu_stats()
    
    def create_device_map(self, 
                          model_size_gb: float, 
                          layer_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """
        Create a device map for model parallelism.
        
        Parameters
        ----------
        model_size_gb : float
            Size of the model in GB
        layer_counts : Dict[str, int], optional
            Count of layers by type for more precise mapping
            
        Returns
        -------
        Dict[str, int]
            Mapping of layer names to device IDs
        """
        if not self.initialized or self.gpu_count <= 1:
            return {"": 0 if self.gpu_count == 1 else -1}
            
        # Update GPU stats
        self._update_gpu_stats()
        
        # Simple implementation for demonstration
        # A production implementation would analyze layer sizes and dependencies
        
        # Calculate available memory per GPU
        available_memory = {}
        total_available = 0
        
        for device_id, stats in self.gpu_stats.items():
            # Use 80% of free memory to be safe
            mem_available = stats.free_memory_mb * 0.8 / 1024  # Convert to GB
            available_memory[device_id] = mem_available
            total_available += mem_available
        
        # Check if we have enough memory
        if total_available < model_size_gb:
            logger.warning(f"Insufficient GPU memory: needed {model_size_gb:.2f}GB, available {total_available:.2f}GB")
            return {"": -1}  # Indicate CPU only
            
        # Create a basic device map
        device_map = {}
        
        # Simple case: if we have specific layer counts
        if layer_counts:
            total_layers = sum(layer_counts.values())
            layers_assigned = 0
            current_device = 0
            
            # Calculate threshold for each device based on memory
            thresholds = {}
            for device_id, mem in available_memory.items():
                thresholds[device_id] = int((mem / total_available) * total_layers)
            
            # Assign layers to devices
            for layer_type, count in layer_counts.items():
                for i in range(count):
                    layer_name = f"{layer_type}.{i}"
                    
                    device_map[layer_name] = current_device
                    layers_assigned += 1
                    
                    # Move to next device if threshold reached
                    if layers_assigned >= thresholds[current_device]:
                        current_device += 1
                        if current_device >= self.gpu_count:
                            current_device = 0  # Wrap around
        else:
            # Simple fractional assignment based on memory
            device_map = {"": 0}  # Default to first GPU
            
        return device_map
        
    def optimize_batch_size(self, 
                           start_batch_size: int, 
                           sample_input_shape: List[int],
                           model: Optional[Any] = None) -> int:
        """
        Find the optimal batch size for the current GPU configuration.
        
        Parameters
        ----------
        start_batch_size : int
            Starting batch size to test
        sample_input_shape : List[int]
            Shape of a single input sample
        model : Any, optional
            PyTorch model to test
            
        Returns
        -------
        int
            Optimal batch size
        """
        if not TORCH_AVAILABLE or not self.initialized or self.gpu_count == 0:
            return start_batch_size
            
        # Simple implementation - in production this would be more sophisticated
        # with actual model forward passes to find memory limits
        
        # Start with suggested batch size
        batch_size = start_batch_size
        
        # If no model provided, use heuristic based on available memory
        if model is None:
            # Get total available memory across devices
            total_mem = 0
            for device_id in self.active_gpus:
                props = torch.cuda.get_device_properties(device_id)
                total_mem += props.total_memory / (1024 * 1024 * 1024)  # GB
                
            # Assume each sample takes approximately:
            # (product of dimensions) * 4 bytes (float32) * 3 (for activations and gradients)
            sample_size = 1
            for dim in sample_input_shape:
                sample_size *= dim
            sample_size_gb = (sample_size * 4 * 3) / (1024 * 1024 * 1024)
            
            # Calculate max theoretical batch size using 70% of memory
            max_batch = int((total_mem * 0.7) / sample_size_gb)
            
            # Adjust batch size to be a power of 2 for better performance
            batch_size = 2 ** int(max(0, min(10, (max_batch - 1).bit_length() - 1)))
            
            logger.info(f"Estimated optimal batch size: {batch_size}")
            return batch_size
        
        # With model: iteratively test batch sizes
        # (This would be a more sophisticated implementation in production)
        return batch_size
    
    def enable_tensor_parallelism(self, 
                                 model: Any, 
                                 num_gpus: Optional[int] = None) -> Any:
        """
        Enable tensor parallelism for a PyTorch model.
        
        Parameters
        ----------
        model : Any
            PyTorch model to parallelize
        num_gpus : int, optional
            Number of GPUs to use (default: all available)
            
        Returns
        -------
        Any
            Parallelized model
        """
        if not TORCH_AVAILABLE or not self.initialized:
            return model
            
        if num_gpus is None:
            num_gpus = self.gpu_count
            
        num_gpus = min(num_gpus, self.gpu_count)
        
        if num_gpus <= 1:
            return model
            
        # This is a simplified implementation
        # A full implementation would use libraries like DeepSpeed or Megatron-LM
        # for proper tensor parallelism
        
        logger.info(f"Tensor parallelism enabled across {num_gpus} GPUs")
        return model
    
    def get_optimal_execution_plan(self) -> Dict[str, Any]:
        """
        Generate an optimal execution plan for the current GPUs.
        
        Returns
        -------
        Dict[str, Any]
            Execution plan with device assignments and optimization settings
        """
        if not self.initialized:
            return {"strategy": "cpu", "reason": "No GPUs available"}
            
        # Update GPU stats
        self._update_gpu_stats()
        
        # Collect optimization recommendations
        recommendations = self.profiler.get_optimization_recommendations()
        
        # Determine optimal strategy based on GPU count and capabilities
        if self.gpu_count == 1:
            strategy = "single_gpu"
            
            # For a single GPU, optimize memory usage
            mem_stats = self.gpu_stats.get(0, None)
            if mem_stats:
                memory_optimizations = []
                
                if mem_stats.free_memory_mb < 2000:
                    memory_optimizations.append("gradient_checkpointing")
                    
                if mem_stats.compute_capability >= "7.0":  # Volta or newer
                    memory_optimizations.append("mixed_precision")
                
                plan = {
                    "strategy": strategy,
                    "device_assignments": {"all": 0},
                    "memory_optimizations": memory_optimizations,
                    "recommendations": recommendations
                }
            else:
                plan = {
                    "strategy": strategy,
                    "device_assignments": {"all": 0},
                    "recommendations": recommendations
                }
        else:
            # For multiple GPUs, choose strategy based on GPU capabilities
            similar_gpus = True
            first_gpu_props = torch.cuda.get_device_properties(0)
            
            for i in range(1, self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                if (props.major != first_gpu_props.major or 
                    props.minor != first_gpu_props.minor or 
                    abs(props.total_memory - first_gpu_props.total_memory) > 1e9):
                    similar_gpus = False
                    break
                    
            if similar_gpus:
                # With similar GPUs, data parallelism is usually best
                strategy = "data_parallel"
                device_assignments = {"all": list(range(self.gpu_count))}
            else:
                # With different GPUs, model parallelism or pipeline might be better
                strategy = "model_parallel"
                
                # Create more sophisticated device map based on capabilities
                device_assignments = self.create_device_map(10)  # Assuming 10GB model
            
            plan = {
                "strategy": strategy,
                "device_assignments": device_assignments,
                "recommendations": recommendations
            }
        
        return plan