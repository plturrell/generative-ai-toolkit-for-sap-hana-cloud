# NVIDIA NGC Deployment Guide

This guide provides instructions for deploying the SAP HANA AI Toolkit using NVIDIA NGC containers with optimized GPU acceleration.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [NGC Container Registry Access](#ngc-container-registry-access)
3. [Pulling the NGC Container](#pulling-the-ngc-container)
4. [Running the Container](#running-the-container)
5. [Configuration Options](#configuration-options)
6. [GPU Resource Requirements](#gpu-resource-requirements)
7. [Multi-GPU Deployment](#multi-gpu-deployment)
8. [TensorRT Optimization](#tensorrt-optimization)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

- NVIDIA GPU (A100 or H100 recommended)
- NVIDIA Container Toolkit installed
- Docker 20.10 or later
- NGC account with API key

## NGC Container Registry Access

1. Create an NVIDIA NGC account at https://ngc.nvidia.com
2. Generate an API key from your NGC account
3. Login to the NGC registry:

```bash
docker login nvcr.io
# Enter your NGC username and API key when prompted
```

## Pulling the NGC Container

Pull the optimized SAP HANA AI Toolkit container:

```bash
docker pull nvcr.io/ea-sap/hana-ai-toolkit:latest
```

For a specific version:

```bash
docker pull nvcr.io/ea-sap/hana-ai-toolkit:1.0.0
```

## Running the Container

Basic usage:

```bash
docker run --gpus all -p 8000:8000 -p 9090:9090 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  -v /path/to/data:/app/data \
  nvcr.io/ea-sap/hana-ai-toolkit:latest
```

## Configuration Options

The container supports the following environment variables:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `API_HOST` | API binding address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format (text/json) | `json` |
| `PROMETHEUS_ENABLED` | Enable Prometheus metrics | `true` |
| `PROMETHEUS_PORT` | Prometheus metrics port | `9090` |
| `NVIDIA_VISIBLE_DEVICES` | GPUs to use | `all` |
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `true` |
| `MULTI_GPU_STRATEGY` | Multi-GPU strategy | `auto` |
| `PRECISION` | Computation precision | `bf16` |
| `HOPPER_ENABLE_FLASH_ATTENTION` | Enable Flash Attention | `true` |
| `HOPPER_ENABLE_FP8` | Enable FP8 precision | `true` |
| `HOPPER_ENABLE_TRANSFORMER_ENGINE` | Enable Transformer Engine | `true` |
| `HOPPER_ENABLE_FSDP` | Enable FSDP | `true` |
| `ENABLE_TENSORRT` | Enable TensorRT acceleration | `true` |

## GPU Resource Requirements

Recommended GPU resources for different workloads:

| Workload | GPU Type | Min GPU Memory | Recommended Config |
|----------|----------|----------------|-------------------|
| Small API deployments | T4, A10 | 16GB | Single GPU |
| Medium API deployments | A100 | 40GB | Single A100 or 2x A10 |
| Large-scale inference | A100, H100 | 80GB | Multiple A100/H100 |
| High-throughput | H100 | 80GB | Multiple H100 |

## Multi-GPU Deployment

For multi-GPU deployments, specify the strategy:

```bash
docker run --gpus all -p 8000:8000 \
  -e MULTI_GPU_STRATEGY=data_parallel \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  nvcr.io/ea-sap/hana-ai-toolkit:latest
```

Available strategies:
- `data_parallel`: For similar GPUs with large batch sizes
- `model_parallel`: For very large models that don't fit on a single GPU
- `pipeline_parallel`: For sequence-based processing with dependencies
- `auto`: Automatically select the best strategy

## TensorRT Optimization

The NGC container includes TensorRT optimization for inference acceleration. To enable:

```bash
docker run --gpus all -p 8000:8000 \
  -e ENABLE_TENSORRT=true \
  nvcr.io/ea-sap/hana-ai-toolkit:latest
```

TensorRT provides significant acceleration:
- Up to 3x faster inference on H100 GPUs
- Up to 2x faster inference on A100 GPUs
- Optimized attention mechanisms
- Reduced memory usage

## Monitoring and Metrics

The container exposes Prometheus metrics on port 9090:

```bash
# Access metrics
curl http://localhost:9090/metrics
```

Key metrics available:
- GPU utilization
- Memory usage
- Request throughput
- Request latency
- Error rates

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use model parallelism across multiple GPUs

2. **Slow Performance**
   - Check CUDA driver compatibility
   - Ensure GPU is not being throttled due to temperature
   - Verify TensorRT optimization is enabled

3. **Container Fails to Start**
   - Verify NGC credentials
   - Check GPU drivers are properly installed
   - Ensure NVIDIA Container Toolkit is installed

4. **GPU Not Detected**
   - Run `nvidia-smi` to verify GPU visibility
   - Check Docker has GPU access rights
   - Verify NVIDIA_VISIBLE_DEVICES setting

For more detailed GPU optimization information, refer to [NVIDIA.md](NVIDIA.md) and [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md).