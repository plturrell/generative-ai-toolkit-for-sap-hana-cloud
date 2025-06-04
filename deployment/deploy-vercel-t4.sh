#!/bin/bash
# Deploy SAP HANA Cloud Generative AI Toolkit to Vercel (frontend) with T4 GPU backend

set -e

# Default settings
BACKEND_URL=${BACKEND_URL:-"https://jupyter0-4ckg1m6x0.brevlab.com"}
ENVIRONMENT=${ENVIRONMENT:-"development"}
VERCEL_TOKEN=${VERCEL_TOKEN:-""}
VERCEL_PROJECT=${VERCEL_PROJECT:-"sap-hana-generative-ai-toolkit"}
VERCEL_ORG=${VERCEL_ORG:-""}
VERCEL_TEMPLATE="deployment/templates/vercel-frontend.json"
VERCEL_OUTPUT="deployment/generated/vercel-frontend.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print script header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  SAP HANA Cloud Generative AI Toolkit - Vercel Deployment${NC}"
echo -e "${BLUE}  T4 GPU Backend Integration                            ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Check for required environment variables
if [ -z "$VERCEL_TOKEN" ]; then
    echo -e "${YELLOW}VERCEL_TOKEN is not set. You may need to log in manually.${NC}"
fi

# Check if backend URL is reachable
echo -e "${BLUE}Checking if backend URL is reachable: ${BACKEND_URL}${NC}"
if ! curl --output /dev/null --silent --head --fail "${BACKEND_URL}/api/health"; then
    echo -e "${RED}Error: Backend URL is not reachable. Make sure the T4 GPU backend is running.${NC}"
    echo -e "${YELLOW}Would you like to continue anyway? (y/n)${NC}"
    read -r continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment aborted.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Continuing with deployment despite unreachable backend...${NC}"
else
    echo -e "${GREEN}Backend URL is reachable.${NC}"
fi

# Create deployment directory if it doesn't exist
mkdir -p deployment/generated

# Generate Vercel configuration
echo -e "${BLUE}Generating Vercel configuration...${NC}"

# Replace template variables
sed -e "s|__BACKEND_URL__|${BACKEND_URL}|g" \
    -e "s|__ENVIRONMENT__|${ENVIRONMENT}|g" \
    "${VERCEL_TEMPLATE}" > "${VERCEL_OUTPUT}"

echo -e "${GREEN}Vercel configuration generated: ${VERCEL_OUTPUT}${NC}"

# Set up vercel.json in the root directory
echo -e "${BLUE}Setting up vercel.json in the root directory...${NC}"
cp "${VERCEL_OUTPUT}" ./vercel.json
echo -e "${GREEN}vercel.json created in root directory.${NC}"

# Set up requirements-vercel.txt
echo -e "${BLUE}Setting up requirements for Vercel...${NC}"
cp api/requirements-vercel.txt ./requirements.txt
echo -e "${GREEN}requirements.txt created in root directory.${NC}"

# Create vercel_handler.py for serverless function entrypoint
echo -e "${BLUE}Creating vercel_handler.py...${NC}"

cat > api/vercel_handler.py << EOF
"""
Vercel handler for SAP HANA Cloud Generative AI Toolkit.

This module provides an entry point for Vercel serverless functions.
"""

import os
import sys
from fastapi import FastAPI
from api.vercel_integration import app

# This is the handler that Vercel calls
def handler(request, context):
    # Process the request with FastAPI
    return app(request, context)
EOF

echo -e "${GREEN}vercel_handler.py created.${NC}"

# Prepare for deployment
echo -e "${BLUE}Preparing for deployment to Vercel...${NC}"

# Check if we need to login
if [ -n "$VERCEL_TOKEN" ]; then
    echo -e "${BLUE}Logging in to Vercel with token...${NC}"
    # Use the token for authentication
    vercel login --token "$VERCEL_TOKEN"
else
    echo -e "${YELLOW}No Vercel token provided. You may need to log in manually.${NC}"
fi

# Deploy to Vercel
echo -e "${BLUE}Deploying to Vercel...${NC}"

DEPLOY_ARGS=""
if [ -n "$VERCEL_ORG" ]; then
    DEPLOY_ARGS="$DEPLOY_ARGS --scope $VERCEL_ORG"
fi

if [ -n "$VERCEL_PROJECT" ]; then
    DEPLOY_ARGS="$DEPLOY_ARGS --name $VERCEL_PROJECT"
fi

# Set environment variables
vercel env add T4_GPU_BACKEND_URL production "$BACKEND_URL" $DEPLOY_ARGS || true
vercel env add ENVIRONMENT production "$ENVIRONMENT" $DEPLOY_ARGS || true

# Deploy to Vercel
DEPLOYMENT_URL=$(vercel --prod $DEPLOY_ARGS)

echo -e "${GREEN}Deployment successful!${NC}"
echo -e "${GREEN}Deployed to: ${DEPLOYMENT_URL}${NC}"

# Save deployment URL to file
echo "$DEPLOYMENT_URL" > deployment_url.txt
echo -e "${BLUE}Deployment URL saved to deployment_url.txt${NC}"

echo ""
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}  Deployment completed successfully!                    ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Deployment URL: ${DEPLOYMENT_URL}                      ${NC}"
echo -e "${BLUE}  Backend URL: ${BACKEND_URL}                            ${NC}"
echo -e "${BLUE}=========================================================${NC}"