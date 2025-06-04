#!/bin/bash
set -e

# Setup script for creating a virtual environment with required dependencies
# This is a more robust approach than installing packages system-wide

echo "Setting up Python virtual environment for NVIDIA T4 deployment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi

# Create a virtual environment directory if it doesn't exist
VENV_DIR="./venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install required packages
echo "Installing required packages..."
pip install pyyaml

# Print information about the environment
echo "Virtual environment setup complete!"
echo "To activate the environment, run:"
echo "source $VENV_DIR/bin/activate"
echo ""
echo "To deploy using this environment, run:"
echo "source $VENV_DIR/bin/activate && ./deploy-t4.sh"

# Deactivate the virtual environment
deactivate