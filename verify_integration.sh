#!/bin/bash
# Integration verification script for T4 GPU optimization, Vercel, and testing framework

# Get the absolute path to the project directory
PROJECT_DIR="/Users/apple/projects/finsightsap/generative-ai-toolkit-for-sap-hana-cloud"

echo "Verifying integration between T4 GPU optimization, Vercel, and testing framework..."
echo

# Check components existence
echo "Checking component files..."
MISSING_FILES=0

# T4 GPU optimizer
if [ ! -f "${PROJECT_DIR}/api/t4_gpu_optimizer.py" ]; then
    echo "❌ Missing T4 GPU optimizer: ${PROJECT_DIR}/api/t4_gpu_optimizer.py"
    MISSING_FILES=1
else
    echo "✅ T4 GPU optimizer found"
fi

# Vercel integration
if [ ! -f "${PROJECT_DIR}/api/vercel_integration.py" ]; then
    echo "❌ Missing Vercel integration: ${PROJECT_DIR}/api/vercel_integration.py"
    MISSING_FILES=1
else
    echo "✅ Vercel integration found"
fi

# Testing framework
if [ ! -f "${PROJECT_DIR}/run_automated_tests.py" ]; then
    echo "❌ Missing testing framework: ${PROJECT_DIR}/run_automated_tests.py"
    MISSING_FILES=1
else
    echo "✅ Testing framework found"
fi

# Test utilities
if [ ! -f "${PROJECT_DIR}/tests/utils/test_utils.py" ]; then
    echo "❌ Missing test utilities: ${PROJECT_DIR}/tests/utils/test_utils.py"
    MISSING_FILES=1
else
    echo "✅ Test utilities found"
fi

# Run tests script
if [ ! -f "${PROJECT_DIR}/run_tests.sh" ]; then
    echo "❌ Missing run tests script: ${PROJECT_DIR}/run_tests.sh"
    MISSING_FILES=1
else
    echo "✅ Run tests script found"
fi

# Test config
if [ ! -f "${PROJECT_DIR}/test_config.json" ]; then
    echo "❌ Missing test config: ${PROJECT_DIR}/test_config.json"
    MISSING_FILES=1
else
    echo "✅ Test config found"
fi

if [ $MISSING_FILES -ne 0 ]; then
    echo
    echo "❌ Some components are missing. Please run the installation/setup process again."
    exit 1
fi

echo
echo "Checking for dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
else
    echo "✅ Python 3 found"
fi

# Check for basic Python packages
echo "Checking Python packages..."
MISSING_PACKAGES=0

packages=("numpy" "requests" "torch")
for package in "${packages[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        echo "❌ Missing package: $package"
        MISSING_PACKAGES=1
    else
        echo "✅ Package found: $package"
    fi
done

if [ $MISSING_PACKAGES -ne 0 ]; then
    echo
    echo "❌ Some required Python packages are missing. Install them using:"
    echo "pip install numpy requests torch"
    exit 1
fi

echo
echo "Checking T4 GPU Optimizer imports..."
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
    echo "❌ T4 GPU Optimizer imports failed"
    exit 1
fi

echo
echo "Checking Vercel integration imports..."
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
    echo "❌ Vercel integration imports failed"
    exit 1
fi

echo
echo "Checking if test configuration is valid..."
# Check if test_config.json is valid JSON
if ! python3 -c "import json; json.load(open('test_config.json'))" &> /dev/null; then
    echo "❌ test_config.json is not valid JSON"
    exit 1
else
    echo "✅ test_config.json is valid"
fi

echo
echo "All components are present and correctly integrated."
echo
echo "To run the automated tests, use:"
echo "./run_tests.sh --all"
echo
echo "To deploy with Vercel and T4 GPU backend, use:"
echo "./deployment/deploy-vercel-t4.sh"
echo

echo "Integration verification completed successfully!"