#!/bin/bash
# Script to test the T4 GPU integration locally

# Get the absolute path to the project directory
PROJECT_DIR="$(pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}           Testing T4 GPU Integration                    ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker and Docker Compose are required but not found${NC}"
    exit 1
fi

# Check for NVIDIA Docker runtime
echo -e "${BLUE}Checking for NVIDIA Docker runtime...${NC}"
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}⚠️ NVIDIA Docker runtime not found. T4 GPU features may not work.${NC}"
    echo -e "${YELLOW}   This is expected if you're not on a system with NVIDIA GPU.${NC}"
else
    echo -e "${GREEN}✅ NVIDIA Docker runtime found${NC}"
fi

# Stop any running containers
echo -e "${BLUE}Stopping any running containers...${NC}"
docker-compose down 2>/dev/null
docker-compose -f docker-compose.nvidia.yml down 2>/dev/null
docker-compose -f docker-compose.simple.yml down 2>/dev/null

# Start the T4-optimized Docker Compose setup
echo -e "${BLUE}Starting the simplified T4 GPU Docker Compose setup...${NC}"
docker-compose -f docker-compose.simple.yml up -d

# Wait for services to start
echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 10

# Check if the API service is running
echo -e "${BLUE}Checking if the API service is running...${NC}"
if ! docker ps | grep -q "api"; then
    echo -e "${RED}❌ API service is not running${NC}"
    docker-compose -f docker-compose.simple.yml logs api
    exit 1
fi

echo -e "${GREEN}✅ API service is running${NC}"

# Test the API health endpoint
echo -e "${BLUE}Testing API health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8002/health || echo "Connection failed")

if [[ $HEALTH_RESPONSE == *"status"*"healthy"* ]]; then
    echo -e "${GREEN}✅ API health check passed${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    echo -e "${RED}Response: $HEALTH_RESPONSE${NC}"
    exit 1
fi

# Test the GPU info endpoint
echo -e "${BLUE}Testing GPU info endpoint...${NC}"
GPU_RESPONSE=$(curl -s http://localhost:8002/api/gpu_info || echo "Connection failed")

if [[ $GPU_RESPONSE == *"Connection failed"* ]]; then
    echo -e "${RED}❌ GPU info endpoint connection failed${NC}"
else
    echo -e "${GREEN}✅ GPU info endpoint accessible${NC}"
    echo -e "${BLUE}GPU Info:${NC}"
    echo $GPU_RESPONSE | python3 -m json.tool
fi

# Run TensorRT T4 tests
echo -e "${BLUE}Running TensorRT T4 tests...${NC}"
echo -e "${YELLOW}Note: These tests require a T4 GPU to pass fully${NC}"

# Create test config for local testing
cat > test_config_local.json << EOF
{
  "api_base_url": "http://localhost:8002",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "precision": "fp16",
  "batch_sizes": [1, 8, 32],
  "enable_tensorrt": true,
  "test_timeout": 300,
  "auth": {
    "enabled": false,
    "api_key": ""
  },
  "results_dir": "test_results"
}
EOF

python3 test_tensorrt_t4.py --config test_config_local.json

# Check embeddings endpoint
echo -e "${BLUE}Testing basic embedding generation...${NC}"
EMBED_RESPONSE=$(curl -s -X POST http://localhost:8002/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a test for T4 GPU acceleration"],
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "use_tensorrt": true
  }' || echo "Connection failed")

if [[ $EMBED_RESPONSE == *"Connection failed"* ]]; then
    echo -e "${RED}❌ Embeddings endpoint connection failed${NC}"
else
    echo -e "${GREEN}✅ Embeddings endpoint accessible${NC}"
    if [[ $EMBED_RESPONSE == *"embeddings"* ]]; then
        echo -e "${GREEN}✅ Embeddings generated successfully${NC}"
    else
        echo -e "${RED}❌ Embeddings generation failed${NC}"
        echo $EMBED_RESPONSE | python3 -m json.tool
    fi
fi

# Cleanup
echo -e "${BLUE}Cleaning up...${NC}"
rm -f test_config_local.json

# Offer to stop the containers
read -p "Do you want to stop the Docker containers? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.simple.yml down
    echo -e "${GREEN}✅ Docker containers stopped${NC}"
else
    echo -e "${YELLOW}Docker containers are still running. To stop them later, run:${NC}"
    echo -e "${YELLOW}docker-compose -f docker-compose.simple.yml down${NC}"
fi

echo
echo -e "${GREEN}T4 GPU integration testing completed!${NC}"
echo -e "${BLUE}=========================================================${NC}"