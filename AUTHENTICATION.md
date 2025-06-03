# Authentication and Setup Guide

This document provides detailed instructions for setting up authentication and configuring the SAP HANA AI Toolkit in a VM environment.

## Table of Contents
1. [NVIDIA NGC Authentication](#nvidia-ngc-authentication)
2. [SAP HANA Cloud Connection Setup](#sap-hana-cloud-connection-setup)
3. [Environment Variable Configuration](#environment-variable-configuration)
4. [VM Deployment Steps](#vm-deployment-steps)

## NVIDIA NGC Authentication

NVIDIA NGC (NVIDIA GPU Cloud) requires authentication to access private container registries and API services.

### 1. Create NGC Account

1. Go to [NGC website](https://ngc.nvidia.com/) and sign up for an account
2. Verify your email address and complete the registration process

### 2. Generate NGC API Key

1. Log in to your NGC account
2. Navigate to your account settings (click your name in the top-right corner)
3. Select "Setup" from the dropdown menu
4. Click on "Get API Key"
5. Generate a new API key and save it securely

### 3. Log in to NGC Registry

```bash
# Log in to NGC registry using the API key
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### 4. Configure NGC CLI (Optional)

For publishing containers and managing resources:

```bash
# Download and install NGC CLI
wget -O ngc https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngc
chmod u+x ngc

# Configure NGC CLI with your API key
export NGC_API_KEY=<your NGC API key>
export NGC_CLI_AUTO_UPDATE=false
```

## SAP HANA Cloud Connection Setup

SAP HANA Cloud connection requires proper authentication credentials or key files.

### 1. Using Direct Credentials

Set up environment variables for direct authentication:

```bash
export HANA_HOST=<your HANA host>
export HANA_PORT=<your HANA port>
export HANA_USER=<your HANA username>
export HANA_PASSWORD=<your HANA password>
```

### 2. Using HANA User Key (Recommended for Security)

1. Create a HANA user key file with the SAP HANA client:

```bash
hdbuserstore SET <key_name> <host>:<port> <username> <password>
```

2. Configure the application to use the key:

```bash
export HANA_USERKEY=<key_name>
```

### 3. Verify Connection

Test the connection with a simple command:

```bash
# Using Python API
python -c "from hana_ml.dataframe import ConnectionContext; conn = ConnectionContext(userkey='$HANA_USERKEY' if '$HANA_USERKEY' else None, address='$HANA_HOST', port=$HANA_PORT, user='$HANA_USER', password='$HANA_PASSWORD'); print(conn.sql('SELECT 1 FROM DUMMY').collect())"
```

## Environment Variable Configuration

Configure the SAP HANA AI Toolkit with environment variables for optimal performance.

### 1. Basic Configuration

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export AUTH_REQUIRED=true
export API_KEYS=<your-api-key>  # Comma-separated for multiple keys

# Database Connection Pool
export CONNECTION_POOL_SIZE=10
```

### 2. NVIDIA GPU Configuration

```bash
# Basic GPU Settings
export ENABLE_GPU_ACCELERATION=true
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_MEMORY_FRACTION=0.8

# Advanced GPU Settings
export NVIDIA_CUDA_DEVICE_ORDER=PCI_BUS_ID
export MULTI_GPU_STRATEGY=auto
```

### 3. TensorRT Optimization

```bash
# TensorRT Settings
export ENABLE_TENSORRT=true
export TENSORRT_PRECISION=fp16  # Options: fp32, fp16, int8
export TENSORRT_MAX_BATCH_SIZE=32
export TENSORRT_WORKSPACE_SIZE_MB=1024
export TENSORRT_CACHE_DIR=/tmp/tensorrt_engines
export TENSORRT_BUILDER_OPTIMIZATION_LEVEL=3
```

### 4. H100-Specific Optimizations (If Available)

```bash
# Hopper Optimizations
export HOPPER_ENABLE_FLASH_ATTENTION=true
export HOPPER_ENABLE_FP8=true
export HOPPER_ENABLE_TRANSFORMER_ENGINE=true
export HOPPER_ENABLE_FSDP=true
```

## VM Deployment Steps

Step-by-step guide to deploy in a VM environment.

### 1. Prepare the VM

Ensure the VM has the following components installed:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Driver (if not already installed)
sudo apt-get update
sudo apt-get install -y nvidia-driver-XXX  # Replace XXX with appropriate version

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Clone Repository (If Building from Source)

```bash
git clone https://github.com/finsightsap/generative-ai-toolkit-for-sap-hana-cloud.git
cd generative-ai-toolkit-for-sap-hana-cloud
```

### 3. Pull NGC Container (Recommended)

```bash
# Authenticate with NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>

# Pull the container
docker pull nvcr.io/ea-sap/hana-ai-toolkit:latest
```

### 4. Configure Environment Variables

Create a `.env` file or set variables directly:

```bash
# Create .env file
cat > .env << EOL
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
AUTH_REQUIRED=true
API_KEYS=<your-api-key>

# HANA Connection
HANA_HOST=<your HANA host>
HANA_PORT=<your HANA port>
HANA_USER=<your HANA username>
HANA_PASSWORD=<your HANA password>

# GPU Configuration
ENABLE_GPU_ACCELERATION=true
NVIDIA_VISIBLE_DEVICES=all
CUDA_MEMORY_FRACTION=0.8
ENABLE_TENSORRT=true
EOL
```

### 5. Run the Container

```bash
# Run the NGC container with GPU support
docker run --gpus all -p 8000:8000 -p 9090:9090 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env \
  nvcr.io/ea-sap/hana-ai-toolkit:latest
```

### 6. Verify Deployment

```bash
# Check if the API is running
curl http://localhost:8000/

# Check if GPU is being used
curl http://localhost:9090/metrics | grep gpu
```

### 7. Run Performance Tests (Optional)

```bash
# Install load testing tool
npm install -g autocannon

# Run performance test
autocannon -c 10 -d 30 -m POST \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -b '{"model": "sap-ai-core-llama3", "prompt": "Hello, world!"}' \
  http://localhost:8000/api/v1/llm
```

## Troubleshooting

### NGC Authentication Issues

If you encounter NGC authentication errors:

```bash
# Check if NGC API key is valid
curl -H "Authorization: $NGC_API_KEY" https://api.ngc.nvidia.com/v2/org/nvidia

# Re-authenticate with NGC
docker logout nvcr.io
docker login nvcr.io
```

### GPU Detection Issues

If the container cannot detect GPUs:

```bash
# Check if NVIDIA driver is working
nvidia-smi

# Verify NVIDIA Container Toolkit configuration
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Run with debugging enabled
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Database Connection Issues

For HANA connection problems:

```bash
# Test connection directly
python -c "from hana_ml.dataframe import ConnectionContext; conn = ConnectionContext(address='$HANA_HOST', port=$HANA_PORT, user='$HANA_USER', password='$HANA_PASSWORD'); print(conn.sql('SELECT 1 FROM DUMMY').collect())"

# Check if port is accessible
telnet $HANA_HOST $HANA_PORT
```