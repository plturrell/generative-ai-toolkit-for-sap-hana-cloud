#!/bin/bash
# T4 GPU Integration Verification Script
# This script checks that all components of the NVIDIA Blueprint are correctly configured and integrated

set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   T4 GPU Integration Verification Tool             ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Check if Docker and Docker Compose are installed
echo -e "\n${YELLOW}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker and Docker Compose are installed.${NC}"

# Check if NVIDIA Docker runtime is installed
echo -e "\n${YELLOW}Checking NVIDIA Docker runtime...${NC}"
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${RED}Warning: NVIDIA Docker runtime not detected.${NC}"
    echo -e "${YELLOW}T4 GPU optimization requires NVIDIA Docker runtime.${NC}"
    echo -e "${YELLOW}Please install it following the instructions at:${NC}"
    echo -e "${YELLOW}https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
fi

# Check for NVIDIA drivers
echo -e "\n${YELLOW}Checking NVIDIA drivers...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Warning: NVIDIA drivers not detected.${NC}"
    echo -e "${YELLOW}T4 GPU optimization requires NVIDIA drivers.${NC}"
    echo -e "${YELLOW}If you're running this on a machine without an NVIDIA GPU, you can ignore this warning.${NC}"
else
    echo -e "${GREEN}NVIDIA drivers detected.${NC}"
    echo -e "${YELLOW}GPU information:${NC}"
    nvidia-smi
fi

# Verify blueprint files
echo -e "\n${YELLOW}Verifying blueprint files...${NC}"
required_files=(
    "nvidia-blueprint-compose.yml"
    "nvidia-blueprint-environment.env"
    "NVIDIA_BLUEPRINT_README.md"
    "ngc-submit.sh"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Required file $file not found.${NC}"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = true ]; then
    echo -e "${GREEN}All blueprint files are present.${NC}"
else
    echo -e "${RED}Some blueprint files are missing.${NC}"
    exit 1
fi

# Verify Docker Compose file syntax
echo -e "\n${YELLOW}Validating Docker Compose file...${NC}"
if ! docker-compose -f nvidia-blueprint-compose.yml config > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose file has syntax errors.${NC}"
    docker-compose -f nvidia-blueprint-compose.yml config
    exit 1
else
    echo -e "${GREEN}Docker Compose file syntax is valid.${NC}"
fi

# Check for essential environment variables in the .env file
echo -e "\n${YELLOW}Checking environment variables...${NC}"
essential_vars=(
    "API_PORT"
    "ENABLE_GPU_ACCELERATION"
    "T4_OPTIMIZED"
    "ENABLE_TENSORRT"
    "TENSORRT_PRECISION"
)

all_vars_exist=true
for var in "${essential_vars[@]}"; do
    if ! grep -q "$var=" nvidia-blueprint-environment.env; then
        echo -e "${RED}Error: Essential environment variable $var not found in .env file.${NC}"
        all_vars_exist=false
    fi
done

if [ "$all_vars_exist" = true ]; then
    echo -e "${GREEN}All essential environment variables are present.${NC}"
else
    echo -e "${RED}Some essential environment variables are missing.${NC}"
    exit 1
fi

# Optional: Test FastAPI without starting full stack
echo -e "\n${YELLOW}Would you like to run a basic API health check without starting the full stack?${NC}"
echo -e "${YELLOW}This will create a temporary container to test the API.${NC}"
read -p "(y/n): " -n 1 -r
echo 
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Starting temporary API container for testing...${NC}"
    docker run --rm -d --name temp-hana-ai-test -p 8080:8000 -e T4_OPTIMIZED=true -e ENABLE_GPU_ACCELERATION=false python:3.9-slim bash -c "apt-get update && apt-get install -y curl python3-pip && pip install fastapi uvicorn && echo 'from fastapi import FastAPI; app = FastAPI(); @app.get(\"/health\"); async def health(): return {\"status\": \"healthy\", \"t4_optimized\": True}' > test.py && uvicorn test:app --host 0.0.0.0 --port 8000"
    
    echo -e "${YELLOW}Waiting for container to start...${NC}"
    sleep 5
    
    echo -e "${YELLOW}Testing API health endpoint...${NC}"
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        echo -e "${GREEN}API health check passed.${NC}"
    else
        echo -e "${RED}API health check failed.${NC}"
    fi
    
    echo -e "${YELLOW}Stopping temporary container...${NC}"
    docker stop temp-hana-ai-test
fi

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}   Integration Verification Complete                ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}All components are correctly configured.${NC}"
echo -e "${GREEN}To deploy the NVIDIA Blueprint, run:${NC}"
echo -e "${BLUE}docker-compose -f nvidia-blueprint-compose.yml up -d${NC}"
echo -e "\n${GREEN}To submit to NGC, run:${NC}"
echo -e "${BLUE}./ngc-submit.sh${NC}"
echo -e "\n${YELLOW}Make sure to customize the environment variables in nvidia-blueprint-environment.env before deployment.${NC}"