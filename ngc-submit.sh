#!/bin/bash
# NGC Submission script for the SAP HANA AI Toolkit T4 Blueprint
# This script prepares and submits the blueprint to the NVIDIA GPU Cloud

set -e

# Configuration
NGC_ORG="ea-sap"
NGC_TEAM="sap-ai-toolkit"
NGC_REPO="hana-ai-toolkit"
NGC_VERSION="1.1.0"
NGC_DESCRIPTION="Generative AI Toolkit for SAP HANA Cloud with NVIDIA GPU optimization and enhanced visualization"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}   NGC Blueprint Submission Tool    ${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if NGC CLI is installed
if ! command -v ngc &> /dev/null; then
    echo -e "${RED}Error: NGC CLI not found. Please install it first.${NC}"
    echo "Visit https://ngc.nvidia.com/setup/installers/cli for installation instructions."
    exit 1
fi

# Check if user is logged in to NGC
echo -e "\n${YELLOW}Checking NGC credentials...${NC}"
if ! ngc config get &> /dev/null; then
    echo -e "${RED}Error: You are not logged in to NGC. Please login first.${NC}"
    echo "Run: ngc config set"
    exit 1
fi
echo -e "${GREEN}NGC credentials verified.${NC}"

# Verify required files exist
echo -e "\n${YELLOW}Verifying required files...${NC}"
required_files=(
    "ngc-blueprint.json"
    "nvidia-blueprint-compose.yml"
    "nvidia-blueprint-environment.env"
    "NVIDIA_BLUEPRINT_README.md"
    "deployment/nvidia-t4/Dockerfile"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Required file $file not found.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}All required files verified.${NC}"

# Create a temporary directory for submission
echo -e "\n${YELLOW}Preparing submission package...${NC}"
TEMP_DIR=$(mktemp -d)
SUBMISSION_DIR="$TEMP_DIR/hana-ai-toolkit-submission"
mkdir -p "$SUBMISSION_DIR"

# Copy required files to submission directory
cp ngc-blueprint.json "$SUBMISSION_DIR/blueprint.json"
cp nvidia-blueprint-compose.yml "$SUBMISSION_DIR/docker-compose.yml"
cp nvidia-blueprint-environment.env "$SUBMISSION_DIR/.env.template"
cp NVIDIA_BLUEPRINT_README.md "$SUBMISSION_DIR/README.md"
cp -r deployment "$SUBMISSION_DIR/"
cp -r src "$SUBMISSION_DIR/"
cp -r examples "$SUBMISSION_DIR/"
cp -r grafana "$SUBMISSION_DIR/"
cp -r prometheus "$SUBMISSION_DIR/"
cp -r nginx "$SUBMISSION_DIR/"
cp requirements*.txt "$SUBMISSION_DIR/"

# Create a tar.gz archive
echo -e "\n${YELLOW}Creating submission archive...${NC}"
ARCHIVE_NAME="hana-ai-toolkit-ngc-submission-$NGC_VERSION.tar.gz"
tar -czf "$ARCHIVE_NAME" -C "$TEMP_DIR" "hana-ai-toolkit-submission"
echo -e "${GREEN}Created submission archive: $ARCHIVE_NAME${NC}"

# Clean up temporary directory
rm -rf "$TEMP_DIR"

# Submit to NGC
echo -e "\n${YELLOW}Submitting to NGC...${NC}"
echo -e "${BLUE}This will initiate the NGC submission process for your blueprint.${NC}"
echo -e "${BLUE}You will need to complete the submission in the NGC web interface.${NC}"

read -p "Do you want to proceed with the submission? (y/n): " -n 1 -r
echo 
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Submission cancelled. The archive $ARCHIVE_NAME has been created for manual submission.${NC}"
    exit 0
fi

# NGC submission
echo -e "\n${YELLOW}Uploading submission...${NC}"
NGC_RESPONSE=$(ngc registry resource upload --source "$ARCHIVE_NAME" --resource-type blueprint)
echo -e "${GREEN}Upload complete. Please complete the submission in the NGC web portal.${NC}"
echo -e "${BLUE}$NGC_RESPONSE${NC}"

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}Submission process initiated. Next steps:${NC}"
echo -e "${GREEN}1. Go to the NGC website: https://ngc.nvidia.com${NC}"
echo -e "${GREEN}2. Navigate to your organization's dashboard${NC}"
echo -e "${GREEN}3. Complete the submission form with additional details${NC}"
echo -e "${GREEN}4. Submit for NVIDIA review${NC}"
echo -e "${GREEN}=======================================================${NC}"

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
echo -e "${BLUE}The local archive $ARCHIVE_NAME has been retained for your records.${NC}"
echo -e "${GREEN}Done.${NC}"