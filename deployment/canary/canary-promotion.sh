#!/bin/bash
# Canary promotion script for SAP HANA AI Toolkit
# This script promotes a canary deployment to production

set -e

# Parse command line arguments
ENVIRONMENT="cf"  # Default to Cloud Foundry
CANARY_VERSION=""
PROMOTION_TIMEOUT=300
ROLLBACK_ON_ERROR=true

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENVIRONMENT="$2"; shift ;;
        --version) CANARY_VERSION="$2"; shift ;;
        --no-rollback) ROLLBACK_ON_ERROR=false ;;
        --timeout) PROMOTION_TIMEOUT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate environment
if [[ "$ENVIRONMENT" != "cf" && "$ENVIRONMENT" != "k8s" ]]; then
    echo "Error: Environment must be either 'cf' (Cloud Foundry) or 'k8s' (Kubernetes)"
    exit 1
fi

# Validate canary version
if [ -z "$CANARY_VERSION" ]; then
    echo "Error: Canary version must be specified with --version"
    exit 1
fi

echo "╔════════════════════════════════════════════════╗"
echo "║      SAP HANA AI Toolkit Canary Promotion      ║"
echo "╚════════════════════════════════════════════════╝"
echo "Environment:      $ENVIRONMENT"
echo "Canary Version:   $CANARY_VERSION"
echo "Auto-rollback:    $ROLLBACK_ON_ERROR"
echo "Promotion Timeout: $PROMOTION_TIMEOUT seconds"
echo "────────────────────────────────────────────────"

# Create promotion timestamp
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
PROMOTION_ID="promotion-$TIMESTAMP"
echo "Promotion ID:     $PROMOTION_ID"

# Source environment variables if available
if [ -f "../.env" ]; then
    source "../.env"
fi

# Function to perform rollback
perform_rollback() {
    if [ "$ENVIRONMENT" = "cf" ]; then
        echo "Rolling back to previous version..."
        cf start hana-ai-toolkit-backup
        cf stop hana-ai-toolkit
        cf rename hana-ai-toolkit hana-ai-toolkit-failed
        cf rename hana-ai-toolkit-backup hana-ai-toolkit
    elif [ "$ENVIRONMENT" = "k8s" ]; then
        echo "Rolling back to previous version..."
        kubectl rollout undo deployment/hana-ai-toolkit
    fi
    echo "Rollback completed."
}

# Promote based on environment
if [ "$ENVIRONMENT" = "cf" ]; then
    # Cloud Foundry promotion
    echo "Promoting canary to production in Cloud Foundry..."
    
    # Login to CF if needed
    if [ -z "$(cf target | grep 'API endpoint')" ]; then
        cf login -a "$CF_API" -u "$CF_USERNAME" -p "$CF_PASSWORD" -o "$CF_ORG" -s "$CF_SPACE"
    fi
    
    # Check if canary exists
    CANARY_EXISTS=$(cf app hana-ai-toolkit-canary &>/dev/null && echo "true" || echo "false")
    if [ "$CANARY_EXISTS" = "false" ]; then
        echo "Error: Canary application 'hana-ai-toolkit-canary' not found!"
        exit 1
    fi
    
    # Backup current production app
    echo "Creating backup of current production app..."
    cf stop hana-ai-toolkit 2>/dev/null || true
    cf rename hana-ai-toolkit hana-ai-toolkit-backup 2>/dev/null || true
    
    # Promote canary to production
    echo "Promoting canary to production..."
    cf rename hana-ai-toolkit-canary hana-ai-toolkit
    
    # Verify health of promoted app
    echo "Verifying health of promoted application..."
    RETRY=0
    MAX_RETRIES=5
    
    while [ $RETRY -lt $MAX_RETRIES ]; do
        HEALTH_STATUS=$(cf app hana-ai-toolkit | grep 'status:' || echo "unknown")
        if [[ "$HEALTH_STATUS" == *"running"* ]]; then
            break
        fi
        RETRY=$((RETRY+1))
        echo "Waiting for application to become healthy... ($RETRY/$MAX_RETRIES)"
        sleep 10
    done
    
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "Error: Promoted application is not healthy!"
        if [ "$ROLLBACK_ON_ERROR" = true ]; then
            perform_rollback
            exit 1
        fi
    fi
    
    # Clean up backup after successful promotion
    echo "Promotion successful! Cleaning up backup..."
    cf delete hana-ai-toolkit-backup -f
    
    echo "Canary promotion to Cloud Foundry completed successfully!"

elif [ "$ENVIRONMENT" = "k8s" ]; then
    # Kubernetes promotion
    echo "Promoting canary to production in Kubernetes..."
    
    # Check if canary exists
    CANARY_EXISTS=$(kubectl get deployment hana-ai-toolkit-canary &>/dev/null && echo "true" || echo "false")
    if [ "$CANARY_EXISTS" = "false" ]; then
        echo "Error: Canary deployment 'hana-ai-toolkit-canary' not found!"
        exit 1
    fi
    
    # Update production deployment with canary image
    echo "Updating production deployment with canary image version..."
    export CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-"docker.io/yourusername"}
    kubectl set image deployment/hana-ai-toolkit hana-ai-toolkit=${CONTAINER_REGISTRY}/hana-ai-toolkit:${CANARY_VERSION}
    
    # Wait for rollout to complete
    echo "Waiting for production rollout to complete..."
    kubectl rollout status deployment/hana-ai-toolkit --timeout=${PROMOTION_TIMEOUT}s
    
    # Update ingress to direct all traffic to production
    echo "Updating ingress to direct all traffic to production..."
    # Get the current ingress
    kubectl get ingress hana-ai-toolkit-canary -o yaml | \
        sed 's/canary: "true"/canary: "false"/g' | \
        sed 's/canary-weight: "[0-9]*"/canary-weight: "0"/g' | \
        kubectl apply -f -
    
    # Verify health of promoted deployment
    echo "Verifying health of promoted deployment..."
    FAILED_PODS=$(kubectl get pods -l app=hana-ai-toolkit -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
    if [ ! -z "$FAILED_PODS" ]; then
        echo "Error: Production pods are not running properly: $FAILED_PODS"
        if [ "$ROLLBACK_ON_ERROR" = true ]; then
            perform_rollback
            exit 1
        fi
    fi
    
    # Clean up canary after successful promotion
    echo "Promotion successful! Cleaning up canary deployment..."
    kubectl delete deployment hana-ai-toolkit-canary
    kubectl delete service hana-ai-toolkit-canary
    kubectl delete ingress hana-ai-toolkit-canary
    
    echo "Canary promotion to Kubernetes completed successfully!"
fi

echo "╔════════════════════════════════════════════════╗"
echo "║    Canary Promotion Completed Successfully     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "The canary version $CANARY_VERSION has been successfully"
echo "promoted to production in the $ENVIRONMENT environment."
echo "────────────────────────────────────────────────"