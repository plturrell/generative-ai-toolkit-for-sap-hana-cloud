#!/bin/bash
set -e

echo "==========================================="
echo "SAP HANA AI Toolkit Deployment for NVIDIA T4"
echo "==========================================="

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
  echo "Error: nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
  exit 1
fi

# Check for required packages
if ! python3 -c "import yaml" &> /dev/null; then
  echo "Warning: PyYAML package is missing. Installing it temporarily..."
  python3 -m pip install --user pyyaml --break-system-packages || echo "Could not install PyYAML. Please install it manually: pip install pyyaml"
fi

# Check if T4 GPU is present
if ! nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -i "T4" &> /dev/null; then
  echo "Warning: NVIDIA T4 GPU not detected. This configuration is optimized for T4 GPUs."
  echo "Do you want to continue anyway? (y/n)"
  read -r CONTINUE
  if [[ "$CONTINUE" != "y" ]]; then
    echo "Deployment cancelled."
    exit 1
  fi
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
  echo "Error: docker command not found. Please install Docker."
  exit 1
fi

# Check if nvidia docker runtime is available
if ! docker info | grep -i "runtimes.*nvidia" &> /dev/null; then
  echo "Error: NVIDIA Docker runtime not found. Please install nvidia-container-toolkit."
  echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  exit 1
fi

# Navigate to the deployment directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create config directory if it doesn't exist
mkdir -p /tmp/tensorrt_engines
mkdir -p /tmp/cuda-cache

# Parse command-line arguments
FRONTEND_URL=""
CUSTOM_ENV=""
REBUILD=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --frontend-url)
      FRONTEND_URL="$2"
      shift 2
      ;;
    --custom-env)
      CUSTOM_ENV="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Update environment file if frontend URL is provided
if [ -n "$FRONTEND_URL" ]; then
  echo "Setting frontend URL to: $FRONTEND_URL"
  sed -i "s|CORS_ORIGINS=.*|CORS_ORIGINS=$FRONTEND_URL|g" t4-environment.env
fi

# Apply custom environment variables if provided
if [ -n "$CUSTOM_ENV" ] && [ -f "$CUSTOM_ENV" ]; then
  echo "Applying custom environment variables from: $CUSTOM_ENV"
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip comments and empty lines
    if [[ ! "$line" =~ ^# ]] && [[ -n "$line" ]]; then
      key=$(echo "$line" | cut -d= -f1)
      value=$(echo "$line" | cut -d= -f2-)
      # Update the key in t4-environment.env
      sed -i "s|^$key=.*|$key=$value|g" t4-environment.env
    fi
  done < "$CUSTOM_ENV"
fi

# Build and deploy the Docker container
if [ "$REBUILD" = true ]; then
  echo "Rebuilding Docker images..."
  docker-compose -f docker-compose.yml build --no-cache
fi

echo "Starting services..."
docker-compose -f docker-compose.yml up -d

echo "Deployment completed!"
echo "API is available at: http://localhost:8000"
echo "Prometheus metrics at: http://localhost:9091"
echo "Grafana dashboard at: http://localhost:3000"
echo "GPU metrics at: http://localhost:9835/metrics"

# Display logs
echo "Showing logs (Ctrl+C to exit)..."
docker-compose -f docker-compose.yml logs -f