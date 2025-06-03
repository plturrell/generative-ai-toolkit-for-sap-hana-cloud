# NVIDIA Launchable Configuration

This document provides information about the compute requirements and configuration for running the SAP HANA AI Toolkit with NVIDIA optimizations as an NVIDIA Launchable.

## VM Mode Requirement

This project requires VM Mode for the following reasons:

1. **Private Container Registry**: Uses NVIDIA NGC private registry requiring authentication
2. **Direct GPU Access**: Requires direct GPU access for TensorRT optimization
3. **External Authentication**: Connects to SAP HANA Cloud services requiring credentials
4. **System Access**: Needs full system access for container orchestration

## Compute Requirements

### Hardware Requirements

- **GPU**: NVIDIA A100 40GB (preferred) or NVIDIA H100 80GB (optimal)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB SSD

### Software Requirements

- **OS**: Ubuntu 22.04 LTS
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: Latest version
- **Python**: 3.9+
- **CUDA**: 12.2+
- **TensorRT**: 8.6.1+

## Environment Variables

The following environment variables are required:

| Variable | Description | Required |
|----------|-------------|----------|
| `NGC_API_KEY` | NVIDIA NGC API key for container registry access | Yes |
| `HANA_HOST` | SAP HANA Cloud host address | Optional |
| `HANA_PORT` | SAP HANA Cloud port | Optional |
| `HANA_USER` | SAP HANA Cloud username | Optional |
| `HANA_PASSWORD` | SAP HANA Cloud password | Optional |
| `ENABLE_TENSORRT` | Enable TensorRT optimization | No (default: true) |

## Launchable Components

- **Main Notebook**: `nvidia_launchable_setup.ipynb`
- **Docker Images**:
  - `nvcr.io/nvidia/pytorch:24.03-py3`
  - `nvcr.io/ea-sap/hana-ai-toolkit:latest`
- **Key Files**:
  - `AUTHENTICATION.md`: Authentication setup guide
  - `NGC_DEPLOYMENT.md`: NGC deployment instructions
  - `TENSORRT_OPTIMIZATION.md`: TensorRT optimization details
  - `NVIDIA.md`: NVIDIA GPU optimization documentation
  - `publish-to-ngc.sh`: NGC publishing script

## Features

### TensorRT Optimization

- **Precision**: FP16 (default), FP32, INT8 (optional)
- **Workspace Size**: 1024MB
- **Batch Size**: 32 (configurable)
- **Dynamic Shapes**: Enabled
- **Builder Optimization Level**: 3 (range: 0-5)

### Hopper Architecture Optimizations (H100 GPU)

- **FP8 Precision**: Enabled if H100 is available
- **Transformer Engine**: Integrated for transformer models
- **Flash Attention 2**: Optimized attention mechanisms
- **FSDP**: Fully Sharded Data Parallel for distributed training

## Instructions

1. **Startup**: Open and run the `nvidia_launchable_setup.ipynb` notebook
2. **Authentication**: Follow the NGC authentication steps in the notebook
3. **Deployment**: Run the container deployment cells
4. **Benchmarking**: Execute the benchmark cells to measure performance

See the main notebook for step-by-step instructions and detailed explanations.

## Performance Expectations

| Operation | Standard PyTorch | With TensorRT | Speedup |
|-----------|------------------|---------------|---------|
| Embedding Generation | 85 ms | 24 ms | 3.5x |
| LLM Inference | 1450 ms | 580 ms | 2.5x |
| Vector Search | 120 ms | 45 ms | 2.7x |

*Benchmarks performed on NVIDIA A100 GPU with batch size=1