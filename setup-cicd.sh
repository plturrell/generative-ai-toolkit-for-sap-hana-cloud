#!/bin/bash
# CI/CD Pipeline Setup Script
# This script sets up a local CI/CD pipeline using Git hooks

set -e

echo "================================"
echo "Setting up CI/CD Pipeline"
echo "================================"

# Navigate to the project root directory
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Check for syntax errors in Python files
echo "Checking Python syntax..."
git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | while read file; do
    python -m py_compile "$file" || exit 1
    echo "✓ $file"
done

# Run linting if flake8 is available
if command -v flake8 &> /dev/null; then
    echo "Running flake8..."
    git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | xargs flake8
fi

# Run type checking if mypy is available
if command -v mypy &> /dev/null; then
    echo "Running mypy..."
    git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | xargs mypy
fi

echo "Pre-commit checks passed!"
exit 0
EOF
chmod +x .git/hooks/pre-commit

# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
set -e

echo "Running pre-push checks..."

# Run tests if pytest is available
if command -v pytest &> /dev/null; then
    echo "Running tests..."
    python -m pytest -xvs
else
    echo "Pytest not found, skipping tests"
fi

echo "Pre-push checks passed!"
exit 0
EOF
chmod +x .git/hooks/pre-push

# Create post-commit hook for automated deployment
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash

echo "Running post-commit actions..."

# Get the current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Only proceed for main branch
if [ "$BRANCH" == "main" ]; then
    echo "On main branch, running deployment tasks..."
    
    # Automatically push to remote if on main branch
    git push origin main
    
    # Optional: Trigger deployment if deploy script exists
    if [ -f ./deployment/deploy.sh ]; then
        echo "Running deployment script..."
        ./deployment/deploy.sh
    fi
fi

echo "Post-commit actions completed!"
exit 0
EOF
chmod +x .git/hooks/post-commit

# Create a simple deployment script
mkdir -p deployment
cat > deployment/deploy.sh << 'EOF'
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
EOF
chmod +x deployment/deploy.sh

# Create local CI script
cat > run-ci.sh << 'EOF'
#!/bin/bash
set -e

echo "================================"
echo "Running CI checks"
echo "================================"

# Run linting
echo "Running linting..."
if command -v flake8 &> /dev/null; then
    flake8 src/
else
    echo "flake8 not found, skipping linting"
fi

# Run type checking
echo "Running type checking..."
if command -v mypy &> /dev/null; then
    mypy src/
else
    echo "mypy not found, skipping type checking"
fi

# Run tests
echo "Running tests..."
if command -v pytest &> /dev/null; then
    python -m pytest -xvs
else
    echo "pytest not found, skipping tests"
fi

echo "CI checks completed!"
exit 0
EOF
chmod +x run-ci.sh

# Create deployment configuration
cat > .env.cicd << 'EOF'
# CI/CD Configuration
GITHUB_REPO=your-username/your-repo
DEPLOY_ENVIRONMENT=dev
ENABLE_CANARY=false
CANARY_WEIGHT=20

# Deployment Configuration
CF_API=https://api.cf.eu10.hana.ondemand.com
CF_ORG=your-org
CF_SPACE=your-space
CF_DOMAIN=cfapps.eu10.hana.ondemand.com

# Docker Configuration
DOCKER_REGISTRY=docker.io
DOCKER_IMAGE=hana-ai-toolkit
DOCKER_TAG=latest
EOF

echo "================================"
echo "CI/CD Pipeline Setup Complete!"
echo "================================"
echo ""
echo "The following files were created:"
echo "- Git hooks (pre-commit, pre-push, post-commit)"
echo "- Deployment script (deployment/deploy.sh)"
echo "- CI script (run-ci.sh)"
echo "- CI/CD configuration (.env.cicd)"
echo ""
echo "To use the CI/CD pipeline:"
echo "1. Update the configuration in .env.cicd"
echo "2. Run './run-ci.sh' to manually trigger CI checks"
echo "3. Commit and push to trigger the pipeline"
echo ""
echo "NOTE: You need to set up GitHub Actions on your repository"
echo "to enable the full CI/CD pipeline on GitHub."
echo "================================"