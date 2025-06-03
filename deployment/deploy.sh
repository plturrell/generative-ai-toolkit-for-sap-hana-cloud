#!/bin/bash
set -e

echo "================================"
echo "Running deployment script"
echo "================================"

# Check if we're deploying a canary version
CANARY=${CANARY:-false}
CANARY_WEIGHT=${CANARY_WEIGHT:-20}

if [ "$CANARY" == "true" ]; then
    echo "Deploying CANARY version with weight: $CANARY_WEIGHT%"
    ./deployment/canary/canary-deployment.sh --env cf --percentage $CANARY_WEIGHT
else
    echo "Deploying PRODUCTION version"
    # Add your production deployment commands here
    # For example:
    # cf push -f deployment/cloudfoundry/manifest.yml
fi

echo "Deployment completed!"
exit 0
