# NVIDIA T4 GPU Deployment Guide

This guide provides detailed instructions for deploying the SAP HANA AI Toolkit optimized for NVIDIA T4 GPUs.

## Overview

NVIDIA T4 GPUs provide a cost-effective solution for AI inference workloads with 16GB GDDR6 memory and 320 Tensor Cores based on the Turing architecture. This deployment configuration is specifically optimized for T4 GPUs to maximize performance.

## System Requirements

- NVIDIA T4 GPU with minimum 16GB memory
- NVIDIA Driver version 450.80.02 or higher
- CUDA 11.0 or higher
- Docker with NVIDIA Container Toolkit installed
- 8+ CPU cores recommended
- 32GB+ system memory recommended

## T4-Specific Optimizations

The deployment includes the following T4-specific optimizations:

1. **FP16 Precision**: T4 GPUs have excellent support for FP16 (half-precision) operations using Tensor Cores, providing up to 2-3x performance improvement
2. **TensorRT Integration**: Optimized TensorRT configuration for T4 Tensor Cores
3. **Memory Management**: Careful memory usage to fit models within T4's 16GB memory limit
4. **Turing-specific Settings**: Architecture-specific settings optimized for Turing (rather than Ampere or Hopper)
5. **Batch Size Optimization**: Optimized batch sizes specifically for T4 characteristics

## Quick Start

### 1. Verify T4 GPU Detection

Ensure your T4 GPU is properly detected:

```bash
nvidia-smi
```

You should see your NVIDIA T4 GPU listed with 16GB memory.

### 2. Generate Deployment Configuration

Run the configuration generation tool:

```bash
cd deployment/nvidia-t4
python3 deploy-config.py
```

This will automatically detect your T4 GPU and generate optimized configurations.

### 3. Deploy with Docker Compose

```bash
cd deployment/nvidia-t4
chmod +x deploy-t4.sh
./deploy-t4.sh
```

The script will:
- Verify T4 GPU availability
- Build and start the containers with T4 optimizations
- Set up monitoring with Prometheus and Grafana

### 4. Access the Services

- API: http://localhost:8000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (default credentials: admin/admin)
- GPU Metrics: http://localhost:9835/metrics

## Configuration Options

### Environment Variables

Key environment variables for T4 optimization:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_MEMORY_FRACTION` | 0.8 | Memory fraction to use (0.0-1.0) |
| `ENABLE_TENSORRT` | true | Enable TensorRT for inference acceleration |
| `TENSORRT_PRECISION` | fp16 | Precision mode (fp16 recommended for T4) |
| `TENSORRT_MAX_BATCH_SIZE` | 16 | Maximum batch size for TensorRT |
| `ENABLE_FLASH_ATTENTION` | true | Enable Flash Attention for faster attention computation |
| `ENABLE_TENSOR_CORES` | true | Ensure Tensor Cores are used for maximum performance |

### Custom Frontend Integration

To connect with a custom frontend, use:

```bash
./deploy-t4.sh --frontend-url="https://your-frontend-url.com"
```

### Custom Environment Variables

To provide custom environment variables:

```bash
./deploy-t4.sh --custom-env=/path/to/custom-env-file.env
```

## Monitoring

The deployment includes comprehensive monitoring specifically designed for T4 GPUs:

1. **Prometheus**: Collects metrics from the API and NVIDIA GPU
2. **Grafana**: Pre-configured dashboard for T4 GPU monitoring showing:
   - GPU Utilization
   - Memory Usage
   - Temperature
   - Power Consumption
   - API Request Metrics

Access the GPU dashboard at: http://localhost:3000/d/t4gpu/nvidia-t4-gpu-dashboard

## Performance Tuning

### Optimal Batch Sizes

T4 GPUs generally perform best with the following batch sizes:

- Small models (<1B parameters): 16-32
- Medium models (1-7B parameters): 4-8
- Large models (>7B parameters): 1-2

### Memory Optimization

If you encounter memory issues:

1. Reduce `CUDA_MEMORY_FRACTION` to 0.7
2. Enable `CHECKPOINT_ACTIVATIONS=true`
3. Reduce batch sizes
4. Consider quantization (INT8) for larger models

### TensorRT Settings

For maximum T4 performance:

1. Set `TENSORRT_PRECISION=fp16`
2. Set `TENSORRT_WORKSPACE_SIZE_MB=1024` (1GB)
3. Set `TENSORRT_BUILDER_OPTIMIZATION_LEVEL=3`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or memory fraction
   ```
   CUDA_MEMORY_FRACTION=0.7
   TENSORRT_MAX_BATCH_SIZE=8
   ```

2. **High GPU Temperature**: Check cooling and consider power limiting
   ```
   nvidia-smi -pl 60  # Limit power to 60W
   ```

3. **TensorRT Failures**: Ensure TensorRT is properly installed with T4 support
   ```
   docker exec -it hana-ai-toolkit-t4 python -c "import tensorrt; print(tensorrt.__version__)"
   ```

4. **Low GPU Utilization**: Check batch sizes and ensure Tensor Cores are being used
   ```
   ENABLE_TENSOR_CORES=true
   ```

### Validating T4 Optimization

The API includes validation endpoints to verify T4 optimization:

```bash
curl http://localhost:8000/validate
```

Check the GPU section of the response for T4-specific information.

## Further Resources

- [NVIDIA T4 GPU Documentation](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [TensorRT Optimization Guide](https://developer.nvidia.com/tensorrt)
- [GPU Optimization.md](../GPU_OPTIMIZATION.md) - General GPU optimization guide
- [NVIDIA.md](../NVIDIA.md) - General NVIDIA deployment information