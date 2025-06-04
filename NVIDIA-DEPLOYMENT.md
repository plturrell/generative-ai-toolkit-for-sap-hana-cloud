# NVIDIA GPU Deployment Guide

This guide provides instructions for deploying the Generative AI Toolkit for SAP HANA Cloud with NVIDIA GPU acceleration.

## Prerequisites

- NVIDIA GPU with CUDA support (optimized for H100 Hopper architecture)
- NVIDIA Driver (version 525 or higher)
- Docker with NVIDIA Container Toolkit
- Docker Compose

## Installation

### 1. Install NVIDIA Container Toolkit

If you haven't already installed the NVIDIA Container Toolkit, follow these steps:

```bash
# Add the NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Verify NVIDIA Container Toolkit Installation

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

You should see output from `nvidia-smi` showing your GPU details.

## Deployment

### Option 1: Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/finsightsap/generative-ai-toolkit-for-sap-hana-cloud.git
   cd generative-ai-toolkit-for-sap-hana-cloud
   ```

2. Deploy with Docker Compose:
   ```bash
   docker-compose -f docker-compose.nvidia.yml up -d
   ```

3. Access the API at `http://localhost:8000`

### Option 2: Docker (Single Container)

1. Clone the repository:
   ```bash
   git clone https://github.com/finsightsap/generative-ai-toolkit-for-sap-hana-cloud.git
   cd generative-ai-toolkit-for-sap-hana-cloud
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t sap-hana-ai-toolkit -f Dockerfile.nvidia .
   docker run --gpus all -p 8000:8000 sap-hana-ai-toolkit
   ```

3. Access the API at `http://localhost:8000`

## Configuration

The GPU optimizations can be configured using environment variables:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `true` |
| `ENABLE_TENSORRT` | Enable TensorRT optimization | `true` |
| `ENABLE_GPTQ` | Enable GPTQ quantization | `true` |
| `ENABLE_AWQ` | Enable AWQ quantization | `true` |
| `QUANTIZATION_BIT_WIDTH` | Bit width for quantization (4 or 8) | `4` |
| `ENABLE_FP8` | Enable FP8 precision (for H100) | `true` |
| `ENABLE_FLASH_ATTENTION_2` | Enable Flash Attention 2 | `true` |
| `QUANTIZATION_CACHE_DIR` | Directory to cache quantized models | `/tmp/quantization_cache` |
| `CUDA_VISIBLE_DEVICES` | CUDA devices to use | `0` |

## Monitoring

The deployment includes Prometheus and Grafana for monitoring:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default login: admin/admin)

## Troubleshooting

### GPU Not Detected

If the GPU is not detected, check the following:

1. Verify that the NVIDIA driver is installed:
   ```bash
   nvidia-smi
   ```

2. Verify that Docker can access the GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

3. Check that the NVIDIA Container Toolkit is properly installed:
   ```bash
   sudo docker info | grep nvidia
   ```

### Memory Issues

If you encounter memory issues (especially with larger models), try:

1. Reducing the model size or using quantization (enabled by default)
2. Adjusting the Docker memory limits:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G
   ```

3. Reduce the `QUANTIZATION_BIT_WIDTH` from 4 to 3 or 2 (lower values mean more compression but potentially lower quality)

## Performance Optimization

For optimal performance with NVIDIA H100 GPUs:

1. Enable FP8 precision (enabled by default with `ENABLE_FP8=true`)
2. Use Flash Attention 2 (enabled by default with `ENABLE_FLASH_ATTENTION_2=true`)
3. For multi-GPU setups, adjust `CUDA_VISIBLE_DEVICES` to use all GPUs (e.g., `0,1,2,3`)
4. For memory-constrained environments, use AWQ quantization (enabled by default with `ENABLE_AWQ=true`)

## Support

For issues or questions about NVIDIA GPU deployment, please open an issue on the GitHub repository.