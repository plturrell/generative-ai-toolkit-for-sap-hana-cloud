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

### Adjusting GPU Optimization Settings

To fine-tune the T4 GPU optimization settings, edit the environment variables in `docker-compose.yml`:

```yaml
environment:
  - ENABLE_GPU_ACCELERATION=true
  - ENABLE_TENSORRT=true
  - GPU_MEMORY_FRACTION=0.8
  - PRECISION=fp16
  - BATCH_SIZE=32
```

### Custom Embedding Models

To use custom embedding models, update the configuration in the API settings or specify the model at runtime.

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

## Next Steps

For more detailed information, refer to:

- [T4 GPU Testing Plan](T4_GPU_TESTING_PLAN.md)
- [GPU Optimization Guide](GPU_OPTIMIZATION.md)
- [TensorRT Optimization Guide](TENSORRT_OPTIMIZATION.md)
- [NVIDIA T4 Deployment Guide](NVIDIA-DEPLOYMENT.md)