#!/bin/bash
# Deploy the Neural Additive Models for SAP HANA Generative AI Toolkit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Neural Additive Models Deployment for SAP HANA Cloud   ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker before continuing.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose before continuing.${NC}"
    exit 1
fi

# Check for required files
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}docker-compose.yml not found. Please run this script from the repository root directory.${NC}"
    exit 1
fi

if [ ! -d "src/hana_ai/tools/hana_ml_tools/nam_design_system" ]; then
    echo -e "${RED}NAM Design System directory not found. Please ensure you have the correct repository version.${NC}"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo -e "${YELLOW}Frontend directory not found. Creating it...${NC}"
    mkdir -p frontend
fi

# Create frontend build directory if needed
if [ ! -d "frontend/build" ]; then
    echo -e "${YELLOW}Frontend build directory not found. Creating it...${NC}"
    mkdir -p frontend/build
    
    # Copy index.html to build directory if it exists
    if [ -f "frontend/index.html" ]; then
        cp frontend/index.html frontend/build/
    fi
fi

# Check for NAM components
echo -e "${BLUE}Checking Neural Additive Models components...${NC}"
nam_tools_found=false
if [ -f "src/hana_ai/tools/hana_ml_tools/neural_additive_models_tools.py" ]; then
    nam_tools_found=true
    echo -e "${GREEN}Found NAM tools module.${NC}"
else
    echo -e "${RED}NAM tools module not found. Deployment may fail.${NC}"
fi

nam_viz_found=false
if [ -f "src/hana_ai/tools/hana_ml_tools/nam_visualizer_tools.py" ]; then
    nam_viz_found=true
    echo -e "${GREEN}Found NAM visualizer module.${NC}"
else
    echo -e "${RED}NAM visualizer module not found. Deployment may fail.${NC}"
fi

nam_design_found=false
if [ -d "src/hana_ai/tools/hana_ml_tools/nam_design_system" ]; then
    nam_design_found=true
    echo -e "${GREEN}Found NAM design system.${NC}"
else
    echo -e "${RED}NAM design system not found. Deployment may fail.${NC}"
fi

# Stop any running containers and remove volumes if needed
echo -e "${BLUE}Stopping any running containers...${NC}"
docker-compose down

# Build and start the containers
echo -e "${BLUE}Building and starting containers...${NC}"
docker-compose up -d --build

# Wait for services to start
echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 5

# Check if services are running
api_running=$(docker-compose ps | grep api | grep "Up" | wc -l)
frontend_running=$(docker-compose ps | grep frontend | grep "Up" | wc -l)
nam_viz_running=$(docker-compose ps | grep nam-visualizer | grep "Up" | wc -l)
nam_model_running=$(docker-compose ps | grep nam-model-service | grep "Up" | wc -l)

if [ "$api_running" -eq "1" ] && [ "$frontend_running" -eq "1" ] && [ "$nam_viz_running" -eq "1" ] && [ "$nam_model_running" -eq "1" ]; then
    echo -e "${GREEN}All services are running!${NC}"
else
    echo -e "${RED}Some services failed to start. Please check docker-compose logs for details.${NC}"
    docker-compose ps
    exit 1
fi

# Print success message
echo -e "${GREEN}Neural Additive Models deployment completed successfully!${NC}"
echo -e "${BLUE}You can access the application at: ${NC}"
echo -e "${GREEN}http://localhost${NC}"
echo ""
echo -e "${BLUE}API is available at: ${NC}"
echo -e "${GREEN}http://localhost/api${NC}"
echo ""
echo -e "${BLUE}NAM Visualizer is available at: ${NC}"
echo -e "${GREEN}http://localhost/nam-visualizer${NC}"
echo ""
echo -e "${BLUE}To check logs: ${NC}"
echo -e "${YELLOW}docker-compose logs -f${NC}"
echo ""
echo -e "${BLUE}=========================================================${NC}"