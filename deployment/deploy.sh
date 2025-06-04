#!/bin/bash
set -e

echo "================================"
echo "SAP HANA AI Toolkit Deployment"
echo "================================"

# Default values
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-"hybrid"}
BACKEND_PLATFORM=${BACKEND_PLATFORM:-""}
FRONTEND_PLATFORM=${FRONTEND_PLATFORM:-""}
BACKEND_URL=${BACKEND_URL:-""}
FRONTEND_URL=${FRONTEND_URL:-""}
CUSTOM_VALUES=${CUSTOM_VALUES:-""}
CANARY=${CANARY:-"false"}
CANARY_WEIGHT=${CANARY_WEIGHT:-20}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --backend)
            BACKEND_PLATFORM="$2"
            shift 2
            ;;
        --frontend)
            FRONTEND_PLATFORM="$2"
            shift 2
            ;;
        --backend-url)
            BACKEND_URL="$2"
            shift 2
            ;;
        --frontend-url)
            FRONTEND_URL="$2"
            shift 2
            ;;
        --custom-values)
            CUSTOM_VALUES="$2"
            shift 2
            ;;
        --canary)
            CANARY="true"
            shift
            ;;
        --canary-weight)
            CANARY_WEIGHT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory for configurations
OUTPUT_DIR="./deployment/output"
mkdir -p $OUTPUT_DIR

# Generate deployment configurations
echo "Generating deployment configurations..."

# Check for required packages
if ! python3 -c "import yaml" &> /dev/null; then
    echo "Warning: PyYAML package is missing. Installing it temporarily..."
    python3 -m pip install --user pyyaml --break-system-packages || echo "Could not install PyYAML. Please install it manually: pip install pyyaml"
fi

CONFIG_CMD="./deployment/deploy-config.py --mode $DEPLOYMENT_MODE --output-dir $OUTPUT_DIR"

if [ -n "$BACKEND_PLATFORM" ]; then
    CONFIG_CMD="$CONFIG_CMD --backend $BACKEND_PLATFORM"
fi

if [ -n "$FRONTEND_PLATFORM" ]; then
    CONFIG_CMD="$CONFIG_CMD --frontend $FRONTEND_PLATFORM"
fi

if [ -n "$BACKEND_URL" ]; then
    CONFIG_CMD="$CONFIG_CMD --backend-url $BACKEND_URL"
fi

if [ -n "$FRONTEND_URL" ]; then
    CONFIG_CMD="$CONFIG_CMD --frontend-url $FRONTEND_URL"
fi

if [ -n "$CUSTOM_VALUES" ]; then
    CONFIG_CMD="$CONFIG_CMD --custom-values $CUSTOM_VALUES"
fi

echo "Running: $CONFIG_CMD"
$CONFIG_CMD

# Check if config generation succeeded
if [ $? -ne 0 ]; then
    echo "Configuration generation failed!"
    exit 1
fi

# Deploy backend if specified
if [ -n "$BACKEND_PLATFORM" ]; then
    echo "Deploying backend to $BACKEND_PLATFORM..."
    
    case $BACKEND_PLATFORM in
        nvidia)
            echo "Deploying to NVIDIA LaunchPad..."
            if [ "$CANARY" == "true" ]; then
                echo "Canary deployments not supported for NVIDIA LaunchPad"
            else
                # Replace with actual NVIDIA deployment command
                echo "NVIDIA LaunchPad deployment would be triggered here"
                # Example: python -m src.hana_ai.api
            fi
            ;;
        together_ai)
            echo "Deploying to Together.ai..."
            if [ "$CANARY" == "true" ]; then
                echo "Canary deployments not supported for Together.ai"
            else
                # Deploy to Together.ai using the generated configuration
                echo "Together.ai deployment would be triggered here"
                # Example: python deploy_together.py --config $OUTPUT_DIR/together-backend.yaml
            fi
            ;;
        sap_btp)
            echo "Deploying to SAP BTP..."
            if [ "$CANARY" == "true" ]; then
                echo "Deploying CANARY version with weight: $CANARY_WEIGHT%"
                ./deployment/canary/canary-deployment.sh --env cf --percentage $CANARY_WEIGHT
            else
                # Replace with actual BTP deployment command
                echo "SAP BTP deployment would be triggered here"
                # Example: cf push -f deployment/cloudfoundry/manifest.yml
            fi
            ;;
        *)
            echo "Unknown backend platform: $BACKEND_PLATFORM"
            exit 1
            ;;
    esac
fi

# Deploy frontend if specified
if [ -n "$FRONTEND_PLATFORM" ]; then
    echo "Deploying frontend to $FRONTEND_PLATFORM..."
    
    case $FRONTEND_PLATFORM in
        vercel)
            echo "Deploying to Vercel..."
            # Replace with actual Vercel deployment command
            echo "Vercel deployment would be triggered here"
            # Example: vercel --prod
            ;;
        sap_btp)
            echo "Deploying to SAP BTP..."
            if [ "$CANARY" == "true" ]; then
                echo "Deploying CANARY version with weight: $CANARY_WEIGHT%"
                ./deployment/canary/canary-deployment.sh --env cf --percentage $CANARY_WEIGHT
            else
                # Replace with actual BTP deployment command
                echo "SAP BTP deployment would be triggered here"
                # Example: cf push -f deployment/cloudfoundry/manifest.yml
            fi
            ;;
        *)
            echo "Unknown frontend platform: $FRONTEND_PLATFORM"
            exit 1
            ;;
    esac
fi

echo "Deployment completed!"
exit 0
