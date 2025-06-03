#!/bin/bash
# Script to build and publish the SAP HANA AI Toolkit to NVIDIA NGC

set -e

# Check for NGC CLI
if ! command -v ngc &> /dev/null; then
    echo "NVIDIA NGC CLI not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget -O ngc https://ngc.nvidia.com/downloads/ngccli_linux.zip
        unzip ngc
        chmod u+x ngc
        export PATH=$PATH:$(pwd)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        wget -O ngc https://ngc.nvidia.com/downloads/ngccli_mac.zip
        unzip ngc
        chmod u+x ngc
        export PATH=$PATH:$(pwd)
    else
        echo "Unsupported OS. Please install NGC CLI manually: https://ngc.nvidia.com/setup/installers/cli"
        exit 1
    fi
fi

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    echo "NGC_API_KEY environment variable not set."
    echo "Please get your API key from https://ngc.nvidia.com/setup"
    echo "Then set it with: export NGC_API_KEY=your_api_key"
    exit 1
fi

# Get tag from arguments or use latest
TAG=${1:-latest}
echo "Building and publishing with tag: $TAG"

# Set NGC variables
NGC_ORG="ea-sap"
NGC_REPO="hana-ai-toolkit"
NGC_REGISTRY="nvcr.io"

# Login to NGC
echo "Logging in to NGC..."
echo $NGC_API_KEY | docker login $NGC_REGISTRY --username='$oauthtoken' --password-stdin

# Build the NGC container
echo "Building NGC container..."
docker build -t $NGC_REGISTRY/$NGC_ORG/$NGC_REPO:$TAG -f deployment/Dockerfile.ngc .

# Push to NGC
echo "Pushing to NGC..."
docker push $NGC_REGISTRY/$NGC_ORG/$NGC_REPO:$TAG

# Upload blueprint
echo "Uploading NGC blueprint..."
ngc registry resource upload-blueprint \
  --model-path ./ngc-blueprint.json \
  --description "Generative AI Toolkit for SAP HANA Cloud with NVIDIA GPU optimization" \
  --registry $NGC_REGISTRY \
  --org $NGC_ORG \
  --team "hana-ai" \
  --repository $NGC_REPO \
  --tag $TAG

echo "Successfully published to NGC: $NGC_REGISTRY/$NGC_ORG/$NGC_REPO:$TAG"
echo "Container can be pulled with: docker pull $NGC_REGISTRY/$NGC_ORG/$NGC_REPO:$TAG"