# Deploying to Together.ai and Vercel

This guide walks through the steps to deploy the SAP HANA AI Toolkit with a Together.ai backend and Vercel frontend.

## Prerequisites

1. Together.ai account with API key
2. Vercel account with access token
3. GitHub repository with the SAP HANA AI Toolkit code

## Deployment Steps

### 1. Generate Deployment Configurations

Generate the configurations for Together.ai backend and Vercel frontend:

```bash
# Create a virtual environment
python3 -m venv deployment_env
source deployment_env/bin/activate

# Install requirements
pip install pyyaml

# Generate configurations
python ./deployment/deploy-config.py \
  --mode hybrid \
  --backend together \
  --frontend vercel \
  --backend-url https://api.together.xyz/v1/sap-hana-ai-toolkit \
  --frontend-url https://sap-hana-ai-toolkit.vercel.app \
  --output-dir ./deployment/generated
```

### 2. Deploy Backend to Together.ai

1. Update the generated Together.ai configuration:

```bash
# Replace the API key placeholder in the configuration
sed -i '' 's/REPLACE_WITH_YOUR_TOGETHER_API_KEY/your-actual-together-api-key/' ./deployment/generated/together-backend.yaml
```

2. Deploy using the Together.ai CLI:

```bash
# Install Together.ai CLI
pip install together-cli

# Login to Together.ai
together login

# Deploy the backend
together deploy --config ./deployment/generated/together-backend.yaml
```

3. Verify the deployment:

```bash
# Check deployment status
together list endpoints
```

### 3. Deploy Frontend to Vercel

1. Update the Content-Security-Policy in the Vercel configuration:

```bash
# Replace the placeholder backend URL with your Together.ai endpoint URL
sed -i '' 's|https://your-nvidia-backend-url.example.com|https://api.together.xyz/v1/sap-hana-ai-toolkit|' ./deployment/generated/vercel-frontend.json
```

2. Deploy using the Vercel CLI:

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy the frontend
vercel deploy --prod --yes --local-config ./deployment/generated/vercel-frontend.json
```

3. Verify the deployment:

```bash
# Open the deployed frontend in a browser
vercel open
```

### 4. Automate Deployments with GitHub Actions

You can also use the CI/CD pipeline to automate deployments:

1. Configure GitHub Secrets:
   - `TOGETHER_API_KEY`: Your Together.ai API key
   - `VERCEL_TOKEN`: Your Vercel access token

2. Trigger the workflow manually:
   - Go to the Actions tab in your GitHub repository
   - Select the "CI/CD Pipeline" workflow
   - Click "Run workflow"
   - Set deployment mode to "hybrid"
   - Set backend platform to "together"
   - Set frontend platform to "vercel"
   - Click "Run workflow"

## Monitoring and Maintenance

### Backend Monitoring

1. View metrics and logs in the Together.ai dashboard:
   - https://api.together.xyz/dashboard

2. Check backend health:
   ```bash
   curl https://api.together.xyz/v1/sap-hana-ai-toolkit/health
   ```

### Frontend Monitoring

1. View deployment details in the Vercel dashboard:
   - https://vercel.com/dashboard

2. Check frontend analytics:
   ```bash
   vercel analytics
   ```

## Troubleshooting

### Backend Issues

1. Check Together.ai logs:
   ```bash
   together logs sap-hana-ai-toolkit-backend
   ```

2. Verify API connectivity:
   ```bash
   curl -H "Authorization: Bearer your-together-api-key" https://api.together.xyz/v1/sap-hana-ai-toolkit/health
   ```

### Frontend Issues

1. Check Vercel deployment logs:
   ```bash
   vercel logs
   ```

2. Verify CORS configuration if you encounter cross-origin issues.

## Scaling

### Backend Scaling

Adjust the scaling parameters in `together-backend.yaml`:

```yaml
scaling:
  maxReplicas: 4  # Increase for higher load
  minReplicas: 1
  targetUtilization: 80
```

### Frontend Scaling

Vercel automatically scales the frontend deployment based on traffic.

## Security Considerations

1. Never commit API keys or tokens to the repository
2. Use environment variables for sensitive information
3. Regularly rotate API keys
4. Monitor usage to detect unauthorized access
5. Configure appropriate CORS settings to restrict access