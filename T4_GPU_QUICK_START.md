# T4 GPU Quick Start Guide

This guide provides step-by-step instructions for deploying the SAP HANA Generative AI Toolkit on an NVIDIA T4 GPU server.

## Prerequisites

- A server with NVIDIA T4 GPU (e.g., Brev.dev Jupyter VM)
- SSH access to the server
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed

## Deployment Steps

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/generative-ai-toolkit-for-sap-hana-cloud.git
cd generative-ai-toolkit-for-sap-hana-cloud
```

### 2. Configure Deployment

The default configuration assumes a T4 GPU server at `jupyter0-4ckg1m6x0.brevlab.com`. If your server is different, update the settings:

```bash
# Edit the server address in the deploy-to-t4.sh script
export T4_SERVER="your-t4-server-address"
export SSH_USER="your-ssh-username"
export SSH_KEY="path/to/your/ssh/key"
```

### 3. Deploy to T4 GPU Server

Run the deployment script to deploy the toolkit to your T4 GPU server:

```bash
./deploy-to-t4.sh
```

This script will:
- Check SSH connectivity to your T4 GPU server
- Install Docker, Docker Compose, and NVIDIA Container Toolkit if needed
- Copy the project files to the server
- Start the services using Docker Compose
- Verify the deployment

### 4. Verify Deployment

After deployment, you can access the following endpoints:

- **Frontend UI**: `http://<your-t4-server-address>`
- **API Documentation**: `http://<your-t4-server-address>/api/docs`
- **Metrics Dashboard**: `http://<your-t4-server-address>/metrics`
- **Grafana**: `http://<your-t4-server-address>/grafana`

### 5. Run Tests

To verify that the T4 GPU and TensorRT integration are working correctly, run the test suite:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Navigate to the project directory
cd /home/<your-ssh-username>/generative-ai-toolkit

# Run the tests
./run_tests.sh --all
```

Or run the TensorRT-specific tests:

```bash
./run_tests.sh --suite tensorrt
```

### 6. View Logs

To monitor the deployment and view logs:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Navigate to the project directory
cd /home/<your-ssh-username>/generative-ai-toolkit

# View logs
docker-compose logs -f
```

## Common Tasks

### Update the Deployment

To update your deployment after making changes:

```bash
# Run the deployment script again
./deploy-to-t4.sh
```

### Stop the Deployment

To stop the services:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Navigate to the project directory
cd /home/<your-ssh-username>/generative-ai-toolkit

# Stop the services
docker-compose down
```

### Restart the Deployment

To restart the services:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Navigate to the project directory
cd /home/<your-ssh-username>/generative-ai-toolkit

# Restart the services
docker-compose restart
```

## Advanced Configuration

### Adaptive Batch Sizing

The T4 GPU deployment includes a sophisticated adaptive batch sizing system that automatically optimizes batch sizes based on:

1. **GPU memory availability**
2. **Model characteristics** (input/output size, hidden dimensions)
3. **Workload patterns** (request frequency, input token distribution)
4. **Performance history** (throughput at different batch sizes)
5. **T4-specific optimizations** (tensor core alignment, mixed precision)

This feature dynamically adjusts batch sizes to maximize throughput while preventing out-of-memory errors.

#### Configuration Options

To configure adaptive batch sizing, edit the environment variables in `docker-compose.yml`:

```yaml
environment:
  # Basic GPU settings
  - ENABLE_GPU_ACCELERATION=true
  - ENABLE_TENSORRT=true
  - GPU_MEMORY_FRACTION=0.8
  - PRECISION=fp16
  
  # Adaptive batch sizing settings
  - ENABLE_ADAPTIVE_BATCH=true
  - T4_OPTIMIZED=true
  - ADAPTIVE_BATCH_MIN=1
  - ADAPTIVE_BATCH_MAX=128
  - ADAPTIVE_BATCH_DEFAULT=32
  - ADAPTIVE_BATCH_BENCHMARK_INTERVAL=3600
  - ADAPTIVE_BATCH_CACHE_TTL=300
```

Key settings:

- `ENABLE_ADAPTIVE_BATCH`: Enable/disable adaptive batch sizing (true/false)
- `T4_OPTIMIZED`: Apply T4-specific optimizations (true/false)
- `ADAPTIVE_BATCH_MIN`: Minimum batch size to use (default: 1)
- `ADAPTIVE_BATCH_MAX`: Maximum batch size to use (default: 128)
- `ADAPTIVE_BATCH_DEFAULT`: Starting batch size when no performance history exists (default: 32)
- `ADAPTIVE_BATCH_BENCHMARK_INTERVAL`: How often to run benchmarks (in seconds, default: 3600)
- `ADAPTIVE_BATCH_CACHE_TTL`: How long to cache batch size recommendations (in seconds, default: 300)

#### Using the API

The system exposes an API for monitoring and configuring adaptive batch sizing:

```bash
# Register a model for adaptive batch sizing
curl -X POST "http://<your-t4-server-address>/api/v1/batch-sizing/register-model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "text-embedding-3-large",
    "model_type": "embedding",
    "input_size_bytes": 2,
    "hidden_size_bytes": 8,
    "output_size_bytes": 4,
    "min_batch_size": 1,
    "max_batch_size": 64,
    "initial_batch_size": 8
  }'

# Get the recommended batch size for a specific input
curl -X POST "http://<your-t4-server-address>/api/v1/batch-sizing/get-batch-size" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "text-embedding-3-large",
    "input_tokens": 512,
    "output_tokens": 0,
    "force_rebenchmark": false
  }'

# View model statistics and performance data
curl -X GET "http://<your-t4-server-address>/api/v1/batch-sizing/model-stats/text-embedding-3-large"
```

#### Example Code

Check out the example in `examples/adaptive_batch_sizing.py` for a complete demonstration of how to use adaptive batch sizing in your application.

### Adjusting GPU Optimization Settings

To fine-tune additional T4 GPU optimization settings, edit the environment variables in `docker-compose.yml`:

```yaml
environment:
  - ENABLE_GPU_ACCELERATION=true
  - ENABLE_TENSORRT=true
  - GPU_MEMORY_FRACTION=0.8
  - PRECISION=fp16
  - TENSORRT_CACHE_PATH=/app/tensorrt_cache
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Custom Embedding Models

To use custom embedding models, update the configuration in the API settings or specify the model at runtime.

## Performance Monitoring

The toolkit includes a comprehensive performance monitoring system:

1. **Grafana Dashboard**: Access real-time GPU metrics and batch size optimization data at `http://<your-t4-server-address>/grafana`

2. **API Metrics**: Get detailed performance metrics through the API:
   ```bash
   curl -X GET "http://<your-t4-server-address>/api/v1/metrics/gpu"
   ```

3. **Prometheus**: Raw metrics are available at `http://<your-t4-server-address>/metrics`

## Troubleshooting

### Check GPU Status

To verify that the GPU is accessible:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Check GPU status
nvidia-smi
```

### Check Docker Containers

To check the status of the Docker containers:

```bash
# SSH into your T4 GPU server
ssh -i <your-ssh-key> <your-ssh-username>@<your-t4-server-address>

# Check running containers
docker ps
```

### GPU Memory Issues

If you encounter GPU memory issues, try adjusting the `GPU_MEMORY_FRACTION` setting in `docker-compose.yml` to a lower value (e.g., 0.6).

### Batch Size Issues

If you experience problems with batch sizes:

1. Check the adaptive batch sizing logs:
   ```bash
   docker-compose logs | grep adaptive_batch
   ```

2. Try running with a fixed batch size by disabling adaptive batch sizing:
   ```yaml
   environment:
     - ENABLE_ADAPTIVE_BATCH=false
     - FIXED_BATCH_SIZE=16
   ```

3. Force a rebenchmark through the API:
   ```bash
   curl -X POST "http://<your-t4-server-address>/api/v1/batch-sizing/get-batch-size" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "text-embedding-3-large",
       "input_tokens": 512,
       "force_rebenchmark": true
     }'
   ```

## Next Steps

For more detailed information, refer to:

- [T4 GPU Testing Plan](T4_GPU_TESTING_PLAN.md)
- [GPU Optimization Guide](GPU_OPTIMIZATION.md)
- [TensorRT Optimization Guide](TENSORRT_OPTIMIZATION.md)
- [NVIDIA T4 Deployment Guide](NVIDIA-DEPLOYMENT.md)