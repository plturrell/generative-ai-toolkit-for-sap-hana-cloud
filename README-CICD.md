# CI/CD Pipeline for SAP HANA AI Toolkit

This document describes the CI/CD pipeline setup for the Generative AI Toolkit for SAP HANA Cloud.

## Automated CI/CD Setup

This project includes a fully automated CI/CD pipeline that connects your local development environment to GitHub and deployment targets.

### Setup Instructions

1. Run the setup script to configure the local CI/CD environment:

```bash
./setup-cicd.sh
```

This script sets up:
- Git hooks for automated checks
- Local CI validation
- Deployment automation
- Environment configuration

2. Edit the `.env.cicd` file to configure your specific environment:

```bash
# Update with your GitHub repository
GITHUB_REPO=your-username/your-repo

# Set your deployment preferences
DEPLOY_ENVIRONMENT=dev
ENABLE_CANARY=false
CANARY_WEIGHT=20

# Configure your SAP BTP environment
CF_API=https://api.cf.eu10.hana.ondemand.com
CF_ORG=your-org
CF_SPACE=your-space
```

3. Initialize the Git repository (if not already done):

```bash
git init
git add .
git commit -m "Initial commit"
```

4. Add your GitHub remote:

```bash
git remote add origin https://github.com/USERNAME/REPOSITORY.git
```

5. Push to GitHub:

```bash
git push -u origin main
```

## CI/CD Pipeline Components

### 1. Local Validation

The local pipeline includes:
- Pre-commit hooks: Syntax checking, linting, type checking
- Pre-push hooks: Automated tests
- Post-commit hooks: Automated deployment (configurable)

### 2. GitHub Actions Workflows

The repository includes GitHub Actions workflows for:
- Continuous Integration (`ci.yml`)
- Continuous Deployment (`cd.yml`)
- Release Automation (`release.yml`)

### 3. Deployment Automation

Deployment options include:
- Standard deployment
- Canary deployment with gradual rollout
- Automatic monitoring and rollback

## Usage Examples

### Running CI Checks Locally

```bash
./run-ci.sh
```

### Deploying a Canary Release

```bash
CANARY=true CANARY_WEIGHT=20 ./deployment/deploy.sh
```

### Creating a Release

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Monitoring and Observability

The CI/CD pipeline includes:
- Build status notifications
- Deployment monitoring
- Canary metrics collection
- Performance dashboards

## Security Considerations

- Credentials are stored in GitHub Secrets
- All deployments are validated before promotion
- Code is scanned for vulnerabilities
- Access control is enforced for deployments

## Troubleshooting

If you encounter issues with the CI/CD pipeline:

1. Check the GitHub Actions logs
2. Verify local Git hooks are installed correctly
3. Ensure all dependencies are installed
4. Validate environment configuration

For detailed error logs, check:
- `.git/hooks/log` directory for local hook errors
- GitHub Actions logs for remote pipeline errors