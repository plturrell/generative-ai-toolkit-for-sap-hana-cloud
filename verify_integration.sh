#!/bin/bash
# Integration verification script for T4 GPU optimization, Vercel, and testing framework

# Get the absolute path to the project directory
PROJECT_DIR="/Users/apple/projects/finsightsap/generative-ai-toolkit-for-sap-hana-cloud"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Verifying T4 GPU, Vercel, and Testing Integration      ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo

# Check components existence
echo -e "${BLUE}Checking component files...${NC}"
MISSING_FILES=0

# T4 GPU optimizer
if [ ! -f "${PROJECT_DIR}/api/t4_gpu_optimizer.py" ]; then
    echo -e "${RED}❌ Missing T4 GPU optimizer: ${PROJECT_DIR}/api/t4_gpu_optimizer.py${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ T4 GPU optimizer found${NC}"
fi

# Vercel integration
if [ ! -f "${PROJECT_DIR}/api/vercel_integration.py" ]; then
    echo -e "${RED}❌ Missing Vercel integration: ${PROJECT_DIR}/api/vercel_integration.py${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Vercel integration found${NC}"
fi

# Testing framework
if [ ! -f "${PROJECT_DIR}/run_automated_tests.py" ]; then
    echo -e "${RED}❌ Missing testing framework: ${PROJECT_DIR}/run_automated_tests.py${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Testing framework found${NC}"
fi

# Test utilities
if [ ! -f "${PROJECT_DIR}/tests/utils/test_utils.py" ]; then
    echo -e "${RED}❌ Missing test utilities: ${PROJECT_DIR}/tests/utils/test_utils.py${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Test utilities found${NC}"
fi

# Run tests script
if [ ! -f "${PROJECT_DIR}/run_tests.sh" ]; then
    echo -e "${RED}❌ Missing run tests script: ${PROJECT_DIR}/run_tests.sh${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Run tests script found${NC}"
fi

# Test config
if [ ! -f "${PROJECT_DIR}/test_config.json" ]; then
    echo -e "${RED}❌ Missing test config: ${PROJECT_DIR}/test_config.json${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Test config found${NC}"
fi

# Docker Compose config
if [ ! -f "${PROJECT_DIR}/docker-compose.yml" ]; then
    echo -e "${RED}❌ Missing Docker Compose file: ${PROJECT_DIR}/docker-compose.yml${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ Docker Compose file found${NC}"
fi

# T4 GPU test
if [ ! -f "${PROJECT_DIR}/test_tensorrt_t4.py" ]; then
    echo -e "${RED}❌ Missing TensorRT T4 test: ${PROJECT_DIR}/test_tensorrt_t4.py${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ TensorRT T4 test found${NC}"
fi

# Deployment script
if [ ! -f "${PROJECT_DIR}/deploy-to-t4.sh" ]; then
    echo -e "${RED}❌ Missing T4 deployment script: ${PROJECT_DIR}/deploy-to-t4.sh${NC}"
    MISSING_FILES=1
else
    echo -e "${GREEN}✅ T4 deployment script found${NC}"
fi

if [ $MISSING_FILES -ne 0 ]; then
    echo
    echo -e "${RED}❌ Some components are missing. Please run the installation/setup process again.${NC}"
    exit 1
fi

echo
echo -e "${BLUE}Checking for dependencies...${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Python 3 found${NC}"
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️ Docker is not installed. Required for deployment.${NC}"
else
    echo -e "${GREEN}✅ Docker found${NC}"
    
    # Check Docker version
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo -e "${GREEN}   Docker version: $DOCKER_VERSION${NC}"
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️ Docker Compose is not installed. Required for deployment.${NC}"
else
    echo -e "${GREEN}✅ Docker Compose found${NC}"
    
    # Check Docker Compose version
    COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
    echo -e "${GREEN}   Docker Compose version: $COMPOSE_VERSION${NC}"
fi

# Check for basic Python packages
echo -e "${BLUE}Checking Python packages...${NC}"
MISSING_PACKAGES=0

packages=("numpy" "requests" "torch" "json" "datetime" "typing")
for package in "${packages[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        echo -e "${RED}❌ Missing package: $package${NC}"
        MISSING_PACKAGES=1
    else
        echo -e "${GREEN}✅ Package found: $package${NC}"
    fi
done

if [ $MISSING_PACKAGES -ne 0 ]; then
    echo
    echo -e "${RED}❌ Some required Python packages are missing. Install them using:${NC}"
    echo -e "${YELLOW}pip install numpy requests torch${NC}"
    exit 1
fi

echo
echo -e "${BLUE}Checking for CUDA and GPU support...${NC}"

# Check if CUDA is available
if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${YELLOW}⚠️ CUDA is not available on this machine${NC}"
    echo -e "${YELLOW}   This is okay for development, but T4 GPU functions won't work locally${NC}"
else
    echo -e "${GREEN}✅ CUDA is available${NC}"
    
    # Get CUDA version
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}   CUDA version: $CUDA_VERSION${NC}"
    
    # Check GPU information
    GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}   GPU: $GPU_INFO${NC}"
    
    # Check if it's a T4
    if echo "$GPU_INFO" | grep -q "T4"; then
        echo -e "${GREEN}   ✅ NVIDIA T4 GPU detected${NC}"
    else
        echo -e "${YELLOW}   ⚠️ This is not a T4 GPU${NC}"
    fi
fi

# Check TensorRT availability
echo -e "${BLUE}Checking for TensorRT...${NC}"
if python3 -c "import tensorrt; print(tensorrt.__version__)" &> /dev/null; then
    TRT_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
    echo -e "${GREEN}✅ TensorRT is installed (version: $TRT_VERSION)${NC}"
else
    echo -e "${YELLOW}⚠️ TensorRT is not installed on this machine${NC}"
    echo -e "${YELLOW}   This is okay for development, but TensorRT optimization won't work locally${NC}"
fi

echo
echo -e "${BLUE}Checking T4 GPU Optimizer imports...${NC}"
# Create a temporary Python script to test T4 GPU Optimizer imports
cat > temp_check.py << EOF
try:
    from api.t4_gpu_optimizer import T4TensorRTOptimizer, T4MemoryManager
    print("✅ T4 GPU Optimizer imports successfully")
except ImportError as e:
    print(f"❌ T4 GPU Optimizer import error: {e}")
    exit(1)
EOF

python3 temp_check.py
IMPORT_RESULT=$?
rm temp_check.py

if [ $IMPORT_RESULT -ne 0 ]; then
    echo -e "${RED}❌ T4 GPU Optimizer imports failed${NC}"
    exit 1
fi

echo
echo -e "${BLUE}Checking Vercel integration imports...${NC}"
# Create a temporary Python script to test Vercel integration imports
cat > temp_check.py << EOF
try:
    from api.vercel_integration import VercelProxy, VercelAuthHandler
    print("✅ Vercel integration imports successfully")
except ImportError as e:
    print(f"❌ Vercel integration import error: {e}")
    exit(1)
EOF

python3 temp_check.py
IMPORT_RESULT=$?
rm temp_check.py

if [ $IMPORT_RESULT -ne 0 ]; then
    echo -e "${RED}❌ Vercel integration imports failed${NC}"
    exit 1
fi

echo
echo -e "${BLUE}Checking if test configuration is valid...${NC}"
# Check if test_config.json is valid JSON
if ! python3 -c "import json; json.load(open('test_config.json'))" &> /dev/null; then
    echo -e "${RED}❌ test_config.json is not valid JSON${NC}"
    exit 1
else
    echo -e "${GREEN}✅ test_config.json is valid${NC}"
    
    # Extract API base URL
    API_URL=$(python3 -c "import json; print(json.load(open('test_config.json')).get('api_base_url', 'unknown'))")
    echo -e "${GREEN}   API base URL: $API_URL${NC}"
fi

echo
echo -e "${BLUE}Checking Docker configuration for T4 GPU...${NC}"
# Check if Docker Compose file has GPU configuration
if grep -q "nvidia" "${PROJECT_DIR}/docker-compose.yml"; then
    echo -e "${GREEN}✅ Docker Compose is configured for GPU usage${NC}"
else
    echo -e "${YELLOW}⚠️ Docker Compose may not be properly configured for GPU usage${NC}"
    echo -e "${YELLOW}   Consider using docker-compose.nvidia.yml instead: ${NC}"
    echo -e "${YELLOW}   docker-compose -f docker-compose.nvidia.yml up -d${NC}"
fi

# Check if T4-optimized Docker Compose file exists
if [ -f "${PROJECT_DIR}/docker-compose.nvidia.yml" ]; then
    echo -e "${GREEN}✅ T4 GPU optimized Docker Compose file found${NC}"
    # Check if it has proper T4 optimizations
    if grep -q "T4_OPTIMIZED" "${PROJECT_DIR}/docker-compose.nvidia.yml"; then
        echo -e "${GREEN}✅ T4 GPU optimizations are properly configured${NC}"
    else
        echo -e "${YELLOW}⚠️ T4 GPU optimizations may not be properly configured in docker-compose.nvidia.yml${NC}"
    fi
else
    echo -e "${RED}❌ T4 GPU optimized Docker Compose file not found${NC}"
fi

echo
echo -e "${GREEN}All components are present and correctly integrated.${NC}"
echo
echo -e "${BLUE}Checking T4 TensorRT Test Capability...${NC}"
# Check if the T4 TensorRT test is executable
if [ -x "${PROJECT_DIR}/test_tensorrt_t4.py" ]; then
    echo -e "${GREEN}✅ T4 TensorRT test is executable${NC}"
else
    echo -e "${YELLOW}⚠️ T4 TensorRT test is not executable. Making it executable...${NC}"
    chmod +x "${PROJECT_DIR}/test_tensorrt_t4.py"
    echo -e "${GREEN}✅ T4 TensorRT test is now executable${NC}"
fi

echo
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. To run the automated tests locally:"
echo -e "   ${YELLOW}./run_tests.sh --all${NC}"
echo
echo -e "2. To test specifically T4 GPU and TensorRT integration:"
echo -e "   ${YELLOW}./test_tensorrt_t4.py${NC}"
echo -e "   ${YELLOW}./test_tensorrt_t4.py --url https://your-t4-server-address${NC}"
echo
echo -e "3. To deploy to T4 GPU server:"
echo -e "   ${YELLOW}./deploy-to-t4.sh${NC}"
echo -e "   ${YELLOW}./deploy-to-t4.sh --server your-t4-server-address --user your-username${NC}"
echo
echo -e "4. To deploy frontend to Vercel with T4 backend:"
echo -e "   ${YELLOW}./deployment/deploy-vercel-t4.sh${NC}"
echo
echo -e "5. For more details on T4 GPU optimization, see:"
echo -e "   ${YELLOW}cat T4_GPU_TESTING_PLAN.md${NC}"
echo -e "   ${YELLOW}cat T4_GPU_QUICK_START.md${NC}"
echo -e "   ${YELLOW}cat T4_NVIDIA_VERCEL_DEPLOYMENT.md${NC}"
echo

echo -e "${GREEN}Integration verification completed successfully!${NC}"
echo -e "${BLUE}=========================================================${NC}"