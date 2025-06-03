#!/bin/bash
# Canary deployment script for SAP HANA AI Toolkit
# This script facilitates canary deployments to SAP BTP environments

set -e

# Parse command line arguments
ENVIRONMENT="cf"  # Default to Cloud Foundry
CANARY_PERCENTAGE=20
CANARY_VERSION=$(date +%Y%m%d%H%M%S)
ROLLBACK_ON_ERROR=true
DEPLOYMENT_TIMEOUT=300

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENVIRONMENT="$2"; shift ;;
        --percentage) CANARY_PERCENTAGE="$2"; shift ;;
        --version) CANARY_VERSION="$2"; shift ;;
        --no-rollback) ROLLBACK_ON_ERROR=false ;;
        --timeout) DEPLOYMENT_TIMEOUT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate environment
if [[ "$ENVIRONMENT" != "cf" && "$ENVIRONMENT" != "k8s" ]]; then
    echo "Error: Environment must be either 'cf' (Cloud Foundry) or 'k8s' (Kubernetes)"
    exit 1
fi

# Validate canary percentage
if ! [[ "$CANARY_PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$CANARY_PERCENTAGE" -lt 1 ] || [ "$CANARY_PERCENTAGE" -gt 99 ]; then
    echo "Error: Canary percentage must be between 1 and 99"
    exit 1
fi

echo "╔════════════════════════════════════════════════╗"
echo "║        SAP HANA AI Toolkit Canary Deploy       ║"
echo "╚════════════════════════════════════════════════╝"
echo "Environment:      $ENVIRONMENT"
echo "Canary Version:   $CANARY_VERSION"
echo "Traffic Weight:   $CANARY_PERCENTAGE%"
echo "Auto-rollback:    $ROLLBACK_ON_ERROR"
echo "Deployment Timeout: $DEPLOYMENT_TIMEOUT seconds"
echo "────────────────────────────────────────────────"

# Create deployment timestamp for this canary
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
DEPLOYMENT_ID="canary-$TIMESTAMP"
echo "Deployment ID:    $DEPLOYMENT_ID"

# Source environment variables if available
if [ -f "../.env" ]; then
    source "../.env"
fi

# Deploy based on environment
if [ "$ENVIRONMENT" = "cf" ]; then
    # Cloud Foundry deployment
    echo "Deploying canary to Cloud Foundry..."
    
    # Prepare manifest with dynamic values
    sed "s/\${CF_DOMAIN}/$CF_DOMAIN/g" cf-canary.yml > cf-canary-deploy.yml
    
    # Login to CF if needed (assumes CF CLI is installed)
    if [ -z "$(cf target | grep 'API endpoint')" ]; then
        cf login -a "$CF_API" -u "$CF_USERNAME" -p "$CF_PASSWORD" -o "$CF_ORG" -s "$CF_SPACE"
    fi
    
    # Deploy canary app
    CF_CANARY_APP="hana-ai-toolkit-canary"
    cf push -f cf-canary-deploy.yml
    
    # Set up route mapping if needed for weighted routing
    if [ "$CANARY_PERCENTAGE" -gt 0 ]; then
        echo "Setting up canary routes with $CANARY_PERCENTAGE% traffic weight..."
        # Cloud Foundry doesn't natively support traffic splitting
        # This would require additional route service or external load balancer
        echo "Note: For accurate traffic splitting in CF, an external load balancer or Istio is recommended"
    fi
    
    # Cleanup temporary manifest
    rm cf-canary-deploy.yml
    
    # Start monitoring for errors
    echo "Monitoring canary deployment for errors..."
    MONITORING_END=$(($(date +%s) + DEPLOYMENT_TIMEOUT))
    
    while [ $(date +%s) -lt $MONITORING_END ]; do
        HEALTH_STATUS=$(cf app "$CF_CANARY_APP" | grep 'status:' || echo "unknown")
        if [[ "$HEALTH_STATUS" == *"crashed"* ]]; then
            echo "Error: Canary deployment has crashed!"
            if [ "$ROLLBACK_ON_ERROR" = true ]; then
                echo "Performing automatic rollback..."
                cf delete "$CF_CANARY_APP" -f
                echo "Rollback completed successfully."
            fi
            exit 1
        fi
        sleep 10
    done
    
    echo "Canary deployment to Cloud Foundry completed successfully!"

elif [ "$ENVIRONMENT" = "k8s" ]; then
    # Kubernetes deployment
    echo "Deploying canary to Kubernetes..."
    
    # Prepare manifest with dynamic values
    export CANARY_VERSION=$CANARY_VERSION
    export CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-"docker.io/yourusername"}
    export DOMAIN=${K8S_DOMAIN:-"example.com"}
    
    # Update canary weight in ingress
    sed -i "s/canary-weight: \"[0-9]*\"/canary-weight: \"$CANARY_PERCENTAGE\"/g" k8s-canary.yaml
    
    # Apply Kubernetes manifests
    envsubst < k8s-canary.yaml | kubectl apply -f -
    
    # Wait for deployment to be ready
    echo "Waiting for canary deployment to be ready..."
    kubectl rollout status deployment/hana-ai-toolkit-canary --timeout=${DEPLOYMENT_TIMEOUT}s
    
    # Start monitoring for errors
    echo "Monitoring canary deployment for errors..."
    MONITORING_END=$(($(date +%s) + DEPLOYMENT_TIMEOUT))
    
    while [ $(date +%s) -lt $MONITORING_END ]; do
        # Check pod status
        FAILED_PODS=$(kubectl get pods -l app=hana-ai-toolkit,version=canary -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
        if [ ! -z "$FAILED_PODS" ]; then
            echo "Error: Canary pods are not running properly: $FAILED_PODS"
            # Get logs from failed pods
            for POD in $FAILED_PODS; do
                echo "Logs from $POD:"
                kubectl logs $POD --tail=50
            done
            
            if [ "$ROLLBACK_ON_ERROR" = true ]; then
                echo "Performing automatic rollback..."
                kubectl delete -f <(envsubst < k8s-canary.yaml)
                echo "Rollback completed successfully."
            fi
            exit 1
        fi
        sleep 10
    done
    
    echo "Canary deployment to Kubernetes completed successfully!"
fi

echo "╔════════════════════════════════════════════════╗"
echo "║    Canary Deployment Completed Successfully    ║"
echo "╚════════════════════════════════════════════════╝"
echo "To promote canary to production:"
echo "  $0 --env $ENVIRONMENT --percentage 100 --version $CANARY_VERSION"
echo ""
echo "To rollback canary deployment:"
echo "  $0 --env $ENVIRONMENT --percentage 0 --version stable"
echo "────────────────────────────────────────────────"