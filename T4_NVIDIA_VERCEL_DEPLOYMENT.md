# T4 GPU and Vercel Deployment Guide

This document provides detailed instructions for deploying the SAP HANA Generative AI Toolkit with a T4 GPU-accelerated backend on NVIDIA infrastructure and a frontend on Vercel.

## Architecture Overview

The deployment architecture follows a split approach:

1. **T4 GPU Backend**: Deployed on NVIDIA infrastructure (Brev.dev Jupyter VM) using Docker Compose
2. **Frontend**: Deployed on Vercel for global availability and easy scaling

This separation allows us to:
- Leverage specialized GPU hardware for compute-intensive operations
- Provide a globally available, low-latency user interface
- Scale each component independently based on its specific requirements

## Prerequisites

### For Backend Deployment (T4 GPU)

- NVIDIA T4 GPU-equipped virtual machine (Brev.dev Jupyter VM or equivalent)
- SSH access to the virtual machine
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed

### For Frontend Deployment (Vercel)

- Vercel account
- Vercel CLI installed (`npm install -g vercel`)
- Vercel authentication token (optional, for CI/CD)

## Backend Deployment Steps

### 1. Configure Backend Settings

Update the T4 GPU backend settings in `docker-compose.yml`:

```yaml
services:
  # API service with T4 GPU acceleration
  api:
    build:
      context: .
      dockerfile: Dockerfile.nvidia
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      # T4 GPU optimization settings
      - ENABLE_GPU_ACCELERATION=true
      - ENABLE_TENSORRT=true
      - GPU_MEMORY_FRACTION=0.8
      - PRECISION=fp16
      - BATCH_SIZE=32
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Deploy to T4 GPU Server

Use the provided deployment script:

```bash
./deploy-to-t4.sh
```

This script will:
- Check SSH connectivity to the T4 GPU server
- Install Docker, Docker Compose, and NVIDIA Container Toolkit if needed
- Copy project files to the server
- Start services using Docker Compose
- Verify deployment status

#### Manual Deployment

If you prefer to deploy manually:

1. SSH into your T4 GPU server:
   ```bash
   ssh -i ~/.ssh/your_key.pem user@your-t4-server
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/generative-ai-toolkit-for-sap-hana-cloud.git
   cd generative-ai-toolkit-for-sap-hana-cloud
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Verify the deployment:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

### 3. Verify Backend Deployment

After deployment, verify that the backend is running properly:

1. Check container status:
   ```bash
   docker-compose ps
   ```

2. Check API health:
   ```bash
   curl https://your-t4-server/api/health
   ```

3. Check GPU info:
   ```bash
   curl https://your-t4-server/api/gpu_info
   ```

4. Test embedding generation:
   ```bash
   curl -X POST https://your-t4-server/api/embeddings \
     -H "Content-Type: application/json" \
     -d '{"texts": ["SAP HANA is a high-performance in-memory database."], "use_tensorrt": true}'
   ```

## Frontend Deployment Steps

### 1. Configure Frontend Settings

Update the Vercel frontend settings in `vercel.json`:

```json
{
  "env": {
    "T4_GPU_BACKEND_URL": "https://your-t4-server",
    "ENVIRONMENT": "production",
    "DEFAULT_TIMEOUT": "60"
  }
}
```

### 2. Deploy to Vercel

Use the provided deployment script:

```bash
./deployment/deploy-vercel-t4.sh
```

This script will:
- Check if the T4 GPU backend is reachable
- Generate Vercel configuration files
- Set up environment variables
- Deploy the frontend to Vercel
- Update CORS settings on the backend to allow requests from the Vercel domain

#### Manual Deployment

If you prefer to deploy manually:

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy to Vercel:
   ```bash
   vercel --prod
   ```

### 3. Verify Frontend Deployment

After deployment, verify that the frontend is running properly:

1. Open the Vercel deployment URL in a web browser
2. Check that the frontend can connect to the backend by clicking "Check Backend Health"
3. Test embedding generation by entering text and clicking "Generate Embeddings"

## Combined Deployment

To deploy both backend and frontend in one step, use:

```bash
./deployment/deploy-combined.sh
```

This script will:
1. Deploy the backend to the T4 GPU server
2. Deploy the frontend to Vercel
3. Configure cross-origin settings
4. Verify the complete deployment

## Environment Variables

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `true` |
| `ENABLE_TENSORRT` | Enable TensorRT optimization | `true` |
| `GPU_MEMORY_FRACTION` | Fraction of GPU memory to use | `0.8` |
| `PRECISION` | Floating point precision | `fp16` |
| `BATCH_SIZE` | Batch size for inference | `32` |
| `AUTH_REQUIRED` | Require authentication | `false` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `T4_GPU_BACKEND_URL` | URL of the T4 GPU backend | `https://jupyter0-4ckg1m6x0.brevlab.com` |
| `ENVIRONMENT` | Deployment environment | `production` |
| `DEFAULT_TIMEOUT` | Default request timeout | `60` |
| `AUTH_REQUIRED` | Require authentication | `false` |

## Testing the Deployment

Run the automated tests to verify the deployment:

```bash
# Test the backend
ssh user@your-t4-server "cd /path/to/project && ./run_tests.sh --all"

# Test the integration
./test_tensorrt_t4.py --url https://your-t4-server
```

## Monitoring and Metrics

### Backend Monitoring

Access Prometheus metrics:
```
https://your-t4-server/metrics
```

Access Grafana dashboard:
```
https://your-t4-server/grafana
```

### Frontend Monitoring

Monitor the frontend using Vercel Analytics:
1. Go to the Vercel dashboard
2. Select your project
3. Navigate to the Analytics tab

## Troubleshooting

### Backend Issues

1. **Container failures**:
   Check container logs:
   ```bash
   docker-compose logs -f api
   ```

2. **GPU not detected**:
   Verify NVIDIA drivers and container toolkit:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
   ```

3. **TensorRT failures**:
   Check TensorRT installation:
   ```bash
   python -c "import tensorrt; print(tensorrt.__version__)"
   ```

### Frontend Issues

1. **Backend connectivity issues**:
   Check CORS settings and backend URL:
   ```bash
   curl -v -H "Origin: https://your-frontend-url" https://your-t4-server/api/health
   ```

2. **Deployment failures**:
   Check Vercel deployment logs in the Vercel dashboard

## Security Considerations

1. **Backend Security**:
   - Enable authentication in production
   - Restrict CORS origins to your Vercel domain
   - Use HTTPS for all communications

2. **Frontend Security**:
   - Use Vercel's built-in security headers
   - Implement proper error handling
   - Validate all user inputs

## Conclusion

This deployment setup provides an optimal combination of GPU acceleration for compute-intensive operations and global availability for the user interface. By leveraging NVIDIA T4 GPUs for the backend and Vercel for the frontend, you can achieve high performance and scalability for your SAP HANA Generative AI Toolkit.