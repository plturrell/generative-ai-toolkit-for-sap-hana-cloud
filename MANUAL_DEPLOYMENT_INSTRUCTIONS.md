# Manual Deployment Instructions

This guide provides instructions for manually deploying the SAP HANA AI Toolkit with Together.ai backend and Vercel frontend using API calls and web interfaces.

## 1. Deploy Backend to Together.ai

### Option 1: Using Together.ai Web Interface

1. Log in to [Together.ai Console](https://api.together.xyz/console)
2. Navigate to "Deployment" > "Endpoints"
3. Click "Create Endpoint"
4. Configure the endpoint with the following settings:
   - Name: `sap-hana-ai-toolkit-backend`
   - Model: `meta-llama/Llama-2-70b-chat-hf`
   - Hardware: `a100-40gb`
   - Quantization: Enabled, 4-bit AWQ
   - Scaling: Min 1, Max 2 replicas
   - Advanced Settings:
     - Enable Continuous Batching
     - Enable Flash Attention
     - Enable KV Caching
     - Enable TensorRT
   - Environment Variables:
     ```
     DEPLOYMENT_MODE=api_only
     DEPLOYMENT_PLATFORM=together_ai
     FRONTEND_URL=https://sap-hana-ai-toolkit.vercel.app
     CORS_ORIGINS=*
     AUTH_REQUIRED=true
     ENABLE_TOGETHER_AI=true
     ```
5. Click "Create Endpoint"

### Option 2: Using Together.ai API

1. Send a POST request to create the endpoint:

```bash
curl -X POST "https://api.together.xyz/v1/endpoints" \
  -H "Authorization: Bearer YOUR_TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sap-hana-ai-toolkit-backend",
    "model": {
      "baseModel": "meta-llama/Llama-2-70b-chat-hf",
      "quantization": {
        "enabled": true,
        "bits": 4,
        "method": "awq"
      },
      "serving": {
        "maxBatchSize": 32,
        "maxConcurrentRequests": 10,
        "maxTokens": 4096,
        "timeout": 120
      }
    },
    "hardware": {
      "instanceType": "a100-40gb",
      "count": 1
    },
    "scaling": {
      "minReplicas": 1,
      "maxReplicas": 2,
      "targetUtilization": 80
    },
    "advanced": {
      "enableContinuousBatching": true,
      "enableFlashAttention": true,
      "enableKVCaching": true,
      "enableTensorRT": true,
      "schedulingStrategy": "fair"
    },
    "environment": {
      "DEPLOYMENT_MODE": "api_only",
      "DEPLOYMENT_PLATFORM": "together_ai",
      "FRONTEND_URL": "https://sap-hana-ai-toolkit.vercel.app",
      "CORS_ORIGINS": "*",
      "AUTH_REQUIRED": "true",
      "ENABLE_TOGETHER_AI": "true"
    }
  }'
```

2. Check the status of your endpoint:

```bash
curl -X GET "https://api.together.xyz/v1/endpoints/sap-hana-ai-toolkit-backend" \
  -H "Authorization: Bearer YOUR_TOGETHER_API_KEY"
```

## 2. Deploy Frontend to Vercel

### Option 1: Using Vercel Web Interface

1. Log in to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" > "Project"
3. Import your GitHub repository
4. Configure the project:
   - Framework Preset: "Other"
   - Root Directory: "/"
   - Build Command: `pip install -r api/requirements-vercel.txt`
   - Output Directory: `public`
   - Environment Variables:
     ```
     DEPLOYMENT_MODE=ui_only
     DEPLOYMENT_PLATFORM=vercel
     API_BASE_URL=https://api.together.xyz/v1/sap-hana-ai-toolkit
     AUTH_REQUIRED=true
     ENABLE_TOGETHER_AI=true
     CORS_ORIGINS=https://*.vercel.app,https://*.sap.com
     ```
5. Click "Deploy"

### Option 2: Using Vercel CLI (If Available)

1. Install Vercel CLI if you haven't already:
```bash
npm install -g vercel
```

2. Create a `vercel.json` file with the following content:
```json
{
  "version": 2,
  "buildCommand": "pip install -r api/requirements-vercel.txt",
  "outputDirectory": "public",
  "env": {
    "DEPLOYMENT_MODE": "ui_only",
    "DEPLOYMENT_PLATFORM": "vercel",
    "API_BASE_URL": "https://api.together.xyz/v1/sap-hana-ai-toolkit",
    "AUTH_REQUIRED": "true",
    "ENABLE_TOGETHER_AI": "true",
    "CORS_ORIGINS": "https://*.vercel.app,https://*.sap.com"
  },
  "build": {
    "env": {
      "PYTHON_VERSION": "3.9"
    }
  },
  "regions": ["iad1"],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        },
        {
          "key": "Strict-Transport-Security",
          "value": "max-age=31536000; includeSubDomains"
        }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/",
      "destination": "/api/index.py"
    },
    {
      "source": "/health",
      "destination": "/api/health.py"
    },
    {
      "source": "/api/:path*",
      "destination": "/api/:path*"
    }
  ]
}
```

3. Login and deploy:
```bash
vercel login
vercel --prod
```

## 3. Configure Cross-Origin Communication

After both deployments are complete, you need to ensure proper CORS configuration:

1. Update the Together.ai backend to allow the Vercel frontend origin:

```bash
curl -X PATCH "https://api.together.xyz/v1/endpoints/sap-hana-ai-toolkit-backend" \
  -H "Authorization: Bearer YOUR_TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "environment": {
      "FRONTEND_URL": "https://your-vercel-deployment-url.vercel.app",
      "CORS_ORIGINS": "https://your-vercel-deployment-url.vercel.app"
    }
  }'
```

2. Update the Vercel frontend to point to the correct backend API URL:
   - Go to your Vercel project
   - Navigate to Settings > Environment Variables
   - Update `API_BASE_URL` with your Together.ai endpoint URL

## 4. Verify the Deployment

### Backend Verification

1. Test the backend health endpoint:
```bash
curl https://api.together.xyz/v1/sap-hana-ai-toolkit-backend/health \
  -H "Authorization: Bearer YOUR_TOGETHER_API_KEY"
```

2. Expected response:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "deployment_mode": "api_only",
  "deployment_platform": "together_ai"
}
```

### Frontend Verification

1. Visit your Vercel deployment URL in a browser
2. Verify that the UI loads correctly
3. Try to authenticate and interact with the backend

## 5. Troubleshooting

### Backend Issues

1. Check Together.ai logs:
   - Visit the Together.ai console
   - Navigate to your endpoint
   - Click on "Logs" tab

2. Common issues:
   - Insufficient quota or resources
   - API key permissions
   - Configuration errors

### Frontend Issues

1. Check Vercel deployment logs:
   - Visit the Vercel dashboard
   - Navigate to your project
   - Click on the latest deployment
   - Check the "Functions" and "Build" logs

2. Common issues:
   - Build failures
   - Environment variable misconfiguration
   - CORS errors when communicating with backend

## 6. Ongoing Maintenance

1. Monitor your Together.ai usage and costs
2. Monitor your Vercel deployment metrics
3. Set up alerts for backend health status
4. Regularly update your deployment with the latest code by redeploying