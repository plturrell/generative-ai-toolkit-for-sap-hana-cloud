# GitHub Actions Deployment Instructions

Follow these steps to deploy the SAP HANA AI Toolkit with Together.ai backend and Vercel frontend using GitHub Actions.

## 1. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Add the following secrets:

| Secret Name | Description |
|-------------|-------------|
| `TOGETHER_API_KEY` | Your Together.ai API key |
| `VERCEL_TOKEN` | Your Vercel access token |
| `CF_API` | Cloud Foundry API endpoint (only needed for BTP deployments) |
| `CF_USERNAME` | Cloud Foundry username (only needed for BTP deployments) |
| `CF_PASSWORD` | Cloud Foundry password (only needed for BTP deployments) |
| `CF_ORG` | Cloud Foundry organization (only needed for BTP deployments) |
| `CF_SPACE` | Cloud Foundry space (only needed for BTP deployments) |
| `NVIDIA_API_KEY` | NVIDIA NGC API key (only needed for NVIDIA deployments) |

## 2. Trigger the Deployment Workflow

1. Go to the Actions tab in your GitHub repository
2. Click on the "CI/CD Pipeline" workflow
3. Click "Run workflow"
4. Configure the workflow with the following parameters:
   - Deployment mode: `hybrid`
   - Backend platform: `together`
   - Frontend platform: `vercel`
5. Click "Run workflow"

## 3. Monitor the Deployment

1. The workflow will run the following jobs:
   - Test: Run tests to ensure code quality
   - Build: Build the deployment artifacts
   - Generate Config: Create deployment configurations
   - Deploy Backend to Together.ai: Deploy the backend to Together.ai
   - Deploy Frontend to Vercel: Deploy the frontend to Vercel

2. Click on each job to view the logs and track progress

## 4. Verify the Deployment

### Backend Verification

1. After deployment, you can verify the backend status using the Together.ai dashboard:
   - Visit [Together.ai Dashboard](https://api.together.xyz/dashboard)
   - Check that your endpoint is active
   - Test the backend API endpoint with:
   ```bash
   curl -H "Authorization: Bearer <your-together-api-key>" \
        https://api.together.xyz/v1/sap-hana-ai-toolkit/health
   ```

### Frontend Verification

1. After deployment, you can verify the frontend using the Vercel dashboard:
   - Visit [Vercel Dashboard](https://vercel.com/dashboard)
   - Check that your project deployment is complete
   - Visit your deployed frontend at the provided URL

## 5. Troubleshooting

If the deployment fails, check the workflow logs for error messages. Common issues include:

1. **Invalid API keys or tokens**: Ensure your GitHub Secrets are correctly set
2. **Configuration errors**: Check the configuration files in `deployment/generated/`
3. **Resource limits**: Verify you have sufficient resources in your Together.ai or Vercel accounts
4. **Network issues**: Ensure your GitHub Actions runner has network access to the deployment targets

## 6. Redeployment

If you need to make changes to the deployment:

1. Make your changes to the codebase
2. Commit and push to GitHub
3. The CI/CD pipeline will automatically run for commits to the main branch
4. Alternatively, manually trigger the workflow again with your desired parameters