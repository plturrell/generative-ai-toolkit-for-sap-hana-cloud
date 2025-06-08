# NVIDIA Blueprint for SAP HANA AI Toolkit

This document provides instructions for deploying the Generative AI Toolkit for SAP HANA Cloud using NVIDIA's GPU technologies. The blueprint is optimized for T4 GPUs but will also work with A100 and H100 GPUs with improved performance.

## Overview

This blueprint provides a production-ready deployment configuration for the SAP HANA AI Toolkit with optimizations for NVIDIA T4 GPUs. It includes:

- FastAPI backend optimized for T4 GPUs
- TensorRT integration for accelerated inference
- Quantization support (GPTQ/AWQ)
- Prometheus/Grafana monitoring with GPU metrics
- Adaptive batch sizing for optimal throughput
- NGINX reverse proxy for frontend serving
- Enhanced UI with algorithm transitions and gesture support

## Hardware Requirements

- NVIDIA T4 GPU (minimum)
- 8+ CPU cores
- 16+ GB system RAM
- 50+ GB storage space

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/finsightsap/generative-ai-toolkit-for-sap-hana-cloud.git
   cd generative-ai-toolkit-for-sap-hana-cloud
   ```

2. Configure the environment variables:
   ```bash
   cp nvidia-blueprint-environment.env .env
   # Edit .env with your configuration
   ```

3. Start the containers:
   ```bash
   docker-compose -f nvidia-blueprint-compose.yml up -d
   ```

4. Verify the deployment:
   ```bash
   curl http://localhost:8000/health
   ```

## Configuration Options

The blueprint can be configured through environment variables in the `.env` file. Key configurations include:

### Basic Settings

- `API_PORT`: Port for the FastAPI server (default: 8000)
- `GRAFANA_PORT`: Port for Grafana (default: 3000)
- `PROMETHEUS_WEB_PORT`: Port for Prometheus web interface (default: 9091)
- `NGINX_PORT`: Port for NGINX frontend (default: 80)

### Security Settings

- `AUTH_REQUIRED`: Enable authentication (default: true)
- `JWT_SECRET`: Secret for JWT token generation (CHANGE THIS IN PRODUCTION)
- `CORS_ORIGINS`: Allowed origins for CORS (default: *)

### GPU Optimization Settings

- `ENABLE_GPU_ACCELERATION`: Enable GPU acceleration (default: true)
- `CUDA_MEMORY_FRACTION`: Fraction of GPU memory to use (default: 0.8)
- `ENABLE_TENSORRT`: Enable TensorRT optimization (default: true)
- `TENSORRT_PRECISION`: TensorRT precision (default: fp16)
- `ENABLE_ADAPTIVE_BATCH`: Enable adaptive batch sizing (default: true)

## Monitoring

The blueprint includes Prometheus and Grafana for monitoring:

- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (default credentials: admin/admin)

Pre-configured dashboards are available for:
- API performance metrics
- GPU utilization and memory usage
- Batch sizing optimization
- Model inference latency

## T4 GPU Optimizations

This blueprint includes specific optimizations for NVIDIA T4 GPUs:

1. **FP16 Precision**: Configured for FP16 precision which T4 GPUs handle efficiently
2. **TensorRT Acceleration**: Pre-compiled TensorRT engines for common operations
3. **Kernel Fusion**: Combines multiple CUDA kernels to reduce kernel launch overhead
4. **Flash Attention**: Optimized attention mechanism for better memory efficiency
5. **Adaptive Batch Sizing**: Dynamically adjusts batch sizes based on input characteristics
6. **INT8 Quantization**: Supports INT8 quantization with minimal accuracy loss
7. **GPTQ/AWQ Support**: Post-training quantization methods for smaller memory footprint
8. **CUDA Graphs**: Captures and replays sequences of CUDA operations for reduced overhead

## Advanced Usage

### Multi-GPU Setup

To utilize multiple T4 GPUs:

1. Set `NVIDIA_VISIBLE_DEVICES=all` in the environment file
2. Adjust `MULTI_GPU_STRATEGY=data_parallel` for optimal performance
3. Scale the `hana-ai-backend` service using Docker Compose:
   ```bash
   docker-compose -f nvidia-blueprint-compose.yml up -d --scale hana-ai-backend=2
   ```

### Custom Models

The blueprint supports custom models:

1. Place model files in a directory and mount it to the container:
   ```yaml
   volumes:
     - ./custom-models:/app/models
   ```
2. Set environment variables to use the custom models:
   ```
   CUSTOM_MODELS_DIR=/app/models
   ENABLE_CUSTOM_MODELS=true
   ```

## Testing

Run the automated test suite to verify T4 GPU optimization:

```bash
docker exec hana-ai-toolkit-t4-backend python -m deployment.nvidia-t4.run_t4_test_suite
```

## Troubleshooting

- **GPU Not Detected**: Ensure NVIDIA drivers and Docker GPU support are properly installed
- **Out of Memory Errors**: Reduce `TENSORRT_MAX_BATCH_SIZE` or `CUDA_MEMORY_FRACTION`
- **Slow Performance**: Check GPU utilization with `nvidia-smi` and adjust batch sizes
- **Container Fails to Start**: Verify Docker GPU runtime is installed and available

## License

Apache License 2.0