# NVIDIA T4 GPU Optimization Guide

## Overview

This guide provides detailed instructions for optimizing the SAP HANA Cloud Generative AI Toolkit on NVIDIA T4 GPUs. The T4 GPU is based on the Turing architecture and offers excellent performance/cost ratio for inference workloads with 16GB VRAM, 320 Tensor cores, and support for FP16 and INT8 precision.

## T4 GPU Architecture Specifications

| Feature | Specification |
|---------|---------------|
| Architecture | Turing |
| CUDA Cores | 2,560 |
| Tensor Cores | 320 |
| VRAM | 16 GB GDDR6 |
| Memory Bandwidth | 320 GB/s |
| FP32 Performance | 8.1 TFLOPS |
| FP16 Performance | 65 TFLOPS |
| INT8 Performance | 130 TOPS |
| TDP | 70W |
| Form Factor | PCIe |

## Precision and Performance Optimization

### Supported Precision Modes

The T4 GPU supports the following precision modes:

1. **FP32 (32-bit floating point)**
   - Full precision
   - Used for critical calculations where precision is paramount
   - Lowest computational throughput

2. **FP16 (16-bit floating point)**
   - Half precision
   - Recommended default for T4 GPUs
   - 2x memory efficiency over FP32
   - Up to 8x performance improvement over FP32 using Tensor Cores

3. **INT8 (8-bit integer)**
   - Quarter precision
   - Requires calibration (see calibration section below)
   - 4x memory efficiency over FP32
   - Up to 16x performance improvement over FP32

### Optimizing for Different Workloads

| Workload Type | Recommended Precision | Notes |
|---------------|----------------------|-------|
| Embedding Generation | FP16 | Good balance of accuracy and performance |
| Classification | INT8 | Excellent for high-throughput classification |
| Text Generation | FP16 | Maintains generation quality |
| Quantized Models (GPTQ/AWQ) | INT8/FP16 hybrid | Use TensorRT for optimal execution |

## TensorRT Optimization

### INT8 Calibration Process

T4 GPUs benefit significantly from TensorRT INT8 calibration, which can provide up to 3-4x speedup over standard PyTorch with minimal accuracy loss.

```python
# Example calibration code
from api.t4_gpu_optimizer import T4TensorRTOptimizer, T4GPUConfig

# Configure INT8 calibration
t4_config = T4GPUConfig(
    int8_mode=True,
    fp16_mode=True,
    precision="int8",
    calibration_cache="/tmp/calibration_cache"
)

# Create optimizer
optimizer = T4TensorRTOptimizer(t4_config)

# Prepare calibration dataset
calibration_data = get_representative_data(1000)  # Get ~1000 representative samples

# Calibrate and optimize model
engine = optimizer.optimize_model_with_calibration(
    model,
    "my_model",
    calibration_data,
    calibration_algorithm="entropy"  # Options: entropy, minmax, percentile
)
```

### Best Practices for TensorRT on T4 GPUs

1. **Dynamic Batch Sizing**: Configure TensorRT with dynamic shapes to handle variable batch sizes efficiently.

2. **Workspace Size**: Set `max_workspace_size` to 4GB for optimal balance on 16GB T4 GPUs.

3. **Kernel Tuning**: Use TensorRT's timing cache to save optimal kernel configurations.

4. **Layer Fusion**: Enable layer fusion optimizations with `optimization_level=4`.

5. **Multiple Optimization Profiles**: Create profiles for common input shapes to improve performance across different request sizes.

## Memory Management Strategies

T4 GPUs have 16GB VRAM, which requires careful memory management:

1. **Quantization**: Use model quantization (GPTQ, AWQ) to reduce model size by 2-4x.

2. **Activation Checkpointing**: Reduce memory usage during forward passes with activation checkpointing.

3. **Batch Size Optimization**: Use the `T4MemoryManager` to dynamically calculate optimal batch sizes:

```python
from api.t4_gpu_optimizer import T4MemoryManager

# Initialize memory manager
memory_manager = T4MemoryManager()

# Calculate optimal batch size
batch_size = memory_manager.calculate_optimal_batch_size(
    input_size_per_sample=2048 * 2,  # 2048 tokens in FP16
    output_size_per_sample=1024 * 2,  # 1024 output tokens in FP16
    processing_size_per_sample=None,  # Auto-estimate
    min_batch=1,
    max_batch=64
)
```

4. **Gradient Accumulation**: When fine-tuning on T4, use gradient accumulation to simulate larger batch sizes.

## Performance Benchmarks

| Model | Precision | Batch Size | Latency (ms) | Throughput (tokens/sec) | Memory Usage (GB) |
|-------|-----------|------------|--------------|-------------------------|-------------------|
| BERT-base | FP16 | 1 | 8.2 | 122 | 1.2 |
| BERT-base | FP16 | 32 | 92.5 | 346 | 2.8 |
| BERT-base | INT8 | 1 | 4.7 | 213 | 0.8 |
| BERT-base | INT8 | 32 | 48.3 | 663 | 1.5 |
| Llama-2-7B | FP16 | 1 | 182 | 11.0 | 14.2 |
| Llama-2-7B | INT8+FP16 | 1 | 108 | 18.5 | 7.3 |
| GPTQ-Llama-2-7B | INT8 | 1 | 98 | 20.4 | 4.2 |

## Environment Variables

Configure T4 optimizations with environment variables:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| T4_GPU_FP16_MODE | Enable FP16 precision | true |
| T4_GPU_INT8_MODE | Enable INT8 precision | false |
| T4_GPU_MAX_WORKSPACE_SIZE | Maximum TensorRT workspace size in bytes | 4294967296 (4GB) |
| T4_GPU_DYNAMIC_BATCH_SIZE | Enable dynamic batch sizing | true |
| T4_GPU_ENABLE_PROFILING | Enable TensorRT profiling | false |
| T4_GPU_ENABLE_TENSOR_CORES | Enable Tensor Cores | true |
| T4_GPU_ENABLE_SPARSE | Enable sparse kernels | false |
| T4_GPU_MEMORY_FRACTION | Fraction of GPU memory to use | 0.8 |
| T4_GPU_MAX_BATCH_SIZE | Maximum batch size | 64 |
| T4_GPU_OPTIMIZATION_LEVEL | TensorRT optimization aggressiveness (0-5) | 3 |

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "CUDA out of memory" errors

**Solutions:**
1. Reduce batch size using the `T4MemoryManager`
2. Enable model quantization (INT8 or GPTQ/AWQ)
3. Check for memory leaks using `nvidia-smi` with watch mode
4. Implement gradient checkpointing if training
5. Consider model sharding for very large models

#### Issue: Slow TensorRT optimization

**Solutions:**
1. Increase `max_workspace_size` in T4GPUConfig
2. Enable timing cache to reuse kernel optimizations
3. Limit the number of optimization profiles
4. Disable verbose logging during optimization

#### Issue: Poor INT8 performance after calibration

**Solutions:**
1. Check calibration dataset representativeness
2. Try different calibration algorithms (entropy, minmax, percentile)
3. Ensure activation patterns match real-world inputs
4. Use mixed precision (INT8+FP16) for sensitive layers

#### Issue: Unstable inference results with reduced precision

**Solutions:**
1. Identify problematic layers and keep them in higher precision
2. Use T4's `strict_type_constraints` option in TensorRT
3. Apply quantization-aware fine-tuning
4. Use FP16 instead of INT8 for token generation layers

#### Issue: Driver compatibility problems

**Solutions:**
1. Ensure CUDA 11.8+ and compatible driver (>=450.80.02)
2. Install TensorRT 8.5+ for best performance
3. Update to latest NGC container if using NGC deployment
4. Check compatibility with `nvidia-smi` and verify compute capability 7.5

## Performance Monitoring

Monitor T4 GPU performance using:

1. **NVIDIA-SMI**: Basic GPU monitoring
   ```
   watch -n 0.5 nvidia-smi
   ```

2. **NVPROF**: Detailed profiling
   ```
   nvprof --profile-from-start off -o profile.nvvp python your_script.py
   ```

3. **Nsight Systems**: Timeline-based profiling
   ```
   nsys profile -o profile python your_script.py
   ```

4. **Prometheus/Grafana**: Real-time monitoring
   - Use the provided T4-specific Grafana dashboard at `/deployment/nvidia-t4/grafana/t4-gpu-dashboard.json`
   - Configure DCGM exporter to collect T4-specific metrics

## Deployment Best Practices

1. **Container Configuration**:
   - Use NVIDIA NGC containers as base images
   - Configure with at least 8 CPU cores per GPU
   - Allocate 2x system RAM to GPU memory ratio (32GB RAM for 16GB T4)

2. **Power and Cooling**:
   - T4 is designed for data center use with passive cooling
   - Ensure adequate airflow in server racks
   - Monitor temperature during extended inference sessions

3. **Multi-GPU Scaling**:
   - T4 GPUs scale well for independent inference requests
   - Use container orchestration (Kubernetes) for horizontal scaling
   - Implement load balancing across multiple T4 instances

4. **Cost Optimization**:
   - T4 offers excellent cost/performance ratio for inference
   - Optimize for throughput to maximize cost efficiency
   - Consider auto-scaling based on load patterns

## Additional Resources

- [NVIDIA T4 Technical Specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [NVIDIA NGC Containers](https://catalog.ngc.nvidia.com/containers)
- [TensorRT INT8 Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [T4 Performance Tuning Guide](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)