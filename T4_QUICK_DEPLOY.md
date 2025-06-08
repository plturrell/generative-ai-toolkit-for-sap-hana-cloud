# T4 GPU Quick Deployment Guide

This guide provides a quick reference for deploying the SAP HANA Generative AI Toolkit with NVIDIA T4 GPU acceleration.

## Prerequisites

- Server with NVIDIA T4 GPU
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed

## Deployment Steps

1. **Clone the Repository**

```bash
git clone https://github.com/sap-hana/generative-ai-toolkit
cd generative-ai-toolkit
```

2. **Use T4-Optimized Docker Compose**

```bash
docker-compose -f docker-compose.nvidia.yml up -d
```

3. **Verify Deployment**

```bash
# Check the API health
curl http://localhost:8000/health

# Check GPU information
curl http://localhost:8000/api/gpu_info

# Run T4 TensorRT tests
python test_tensorrt_t4.py
```

## Configuration Options

The following environment variables can be adjusted in `docker-compose.nvidia.yml`:

```yaml
# T4 GPU optimization settings
- ENABLE_GPU_ACCELERATION=true
- ENABLE_TENSORRT=true
- T4_GPU_FP16_MODE=true
- T4_GPU_INT8_MODE=true
- T4_GPU_MAX_WORKSPACE_SIZE=4294967296
- T4_GPU_MEMORY_FRACTION=0.8
- T4_GPU_OPTIMIZATION_LEVEL=4
- PRECISION=fp16
```

## Monitoring

The deployment includes a comprehensive monitoring stack:

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Metrics**: http://localhost:8000/metrics

## Troubleshooting

If you encounter issues, check the following:

1. **GPU Detection**
   ```bash
   nvidia-smi  # Verify T4 GPU is visible
   ```

2. **Docker Logs**
   ```bash
   docker-compose -f docker-compose.nvidia.yml logs api
   ```

3. **TensorRT Cache**
   ```bash
   # Clear TensorRT cache if needed
   docker exec -it <container_id> rm -rf /app/tensorrt_cache/*
   ```

4. **Adaptive Batch Sizing**
   ```bash
   # If experiencing OOM errors, adjust batch size
   # Edit docker-compose.nvidia.yml:
   - ADAPTIVE_BATCH_MAX=64  # Lower value
   - T4_GPU_MEMORY_FRACTION=0.7  # Lower value
   ```

For more detailed information, refer to:
- [T4 GPU Implementation Summary](T4_GPU_IMPLEMENTATION_SUMMARY.md)
- [Full T4 GPU Documentation](T4_GPU_QUICK_START.md)
EOF < /dev/null