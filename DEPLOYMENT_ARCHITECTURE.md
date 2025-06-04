# Multi-Platform Deployment Architecture

This document describes the flexible deployment architecture for the SAP HANA AI Toolkit, which supports deploying components across multiple platforms while maintaining seamless integration.

## Architecture Overview

The SAP HANA AI Toolkit supports three deployment modes:

1. **Full Mode**: All components (frontend and backend) deployed on the same platform
2. **API-Only Mode**: Only backend services, designed for high-performance computing environments
3. **UI-Only Mode**: Only frontend components, connecting to an external API backend

**Deployment Architecture Diagram:**

To view the interactive diagram, open `img/deployment-architecture.md` in a Mermaid-compatible viewer or visit [Mermaid Live Editor](https://mermaid.live) and paste the content.

This architecture enables maximum flexibility, allowing deployment of:
- Backend on NVIDIA LaunchPad with frontend on Vercel
- Backend on Together.ai with frontend on SAP BTP
- Both frontend and backend on SAP BTP
- And many other combinations

## Deployment Platforms

### Backend Platforms

| Platform | Description | Best For |
|----------|-------------|----------|
| **NVIDIA LaunchPad** | High-performance GPU infrastructure | Maximum performance, large model inference |
| **Together.ai** | Cloud GPU service via API | Simplified GPU access without hardware management |
| **SAP BTP** | SAP Business Technology Platform | Integration with SAP ecosystem, direct HANA access |

### Frontend Platforms

| Platform | Description | Best For |
|----------|-------------|----------|
| **Vercel** | Global edge network for web hosting | Fast global access, simplified deployment |
| **SAP BTP** | SAP Business Technology Platform | Enterprise compliance, SAP authentication |

## Deployment Modes

### 1. Full Mode

In full mode, both frontend and backend components are deployed on the same platform. This is the simplest deployment model but may not utilize the strengths of each platform.

**Configuration:**
```
DEPLOYMENT_MODE=full
```

### 2. API-Only Mode

In API-only mode, only the backend services are deployed, without the UI components. This is optimized for compute-intensive environments like NVIDIA LaunchPad.

**Configuration:**
```
DEPLOYMENT_MODE=api_only
FRONTEND_URL=https://your-frontend-url.example.com
```

### 3. UI-Only Mode

In UI-only mode, only the frontend components are deployed, connecting to an external backend API. This is ideal for platforms like Vercel that excel at global content delivery.

**Configuration:**
```
DEPLOYMENT_MODE=ui_only
API_BASE_URL=https://your-backend-url.example.com
```

## Deployment Combinations

### NVIDIA Backend + Vercel Frontend

**Use Case:** Maximum performance with simplified frontend deployment

**Backend Configuration:**
```
DEPLOYMENT_MODE=api_only
DEPLOYMENT_PLATFORM=nvidia
FRONTEND_URL=https://your-app.vercel.app
ENABLE_GPU_ACCELERATION=true
```

**Frontend Configuration:**
```
DEPLOYMENT_MODE=ui_only
DEPLOYMENT_PLATFORM=vercel
API_BASE_URL=https://your-nvidia-backend.example.com
```

### Together.ai Backend + Vercel Frontend

**Use Case:** Cloud GPU without hardware management, optimal global delivery

**Backend Configuration:**
```yaml
endpoint:
  name: "sap-hana-ai-toolkit-backend"
  
environment:
  DEPLOYMENT_MODE: "api_only"
  DEPLOYMENT_PLATFORM: "together"
  FRONTEND_URL: "https://your-app.vercel.app"
```

**Frontend Configuration:**
```json
{
  "env": {
    "DEPLOYMENT_MODE": "ui_only",
    "DEPLOYMENT_PLATFORM": "vercel",
    "API_BASE_URL": "https://your-together-endpoint.together.xyz"
  }
}
```

### SAP BTP Backend + SAP BTP Frontend

**Use Case:** Enterprise compliance, unified SAP ecosystem

**Backend Configuration:**
```
DEPLOYMENT_MODE=api_only
DEPLOYMENT_PLATFORM=btp
FRONTEND_URL=https://your-frontend.cfapps.eu10.hana.ondemand.com
```

**Frontend Configuration:**
```
DEPLOYMENT_MODE=ui_only
DEPLOYMENT_PLATFORM=btp
API_BASE_URL=https://your-backend.cfapps.eu10.hana.ondemand.com
```

## Cross-Origin Communication

When deploying frontend and backend on different platforms, cross-origin resource sharing (CORS) is automatically configured based on the deployment mode and URLs.

- In API-only mode, the CORS configuration automatically includes the frontend URL
- In UI-only mode, API requests are proxied to the backend URL

## Deployment Process

The deployment process is simplified using the unified deployment script:

```bash
./deployment/deploy.sh --mode hybrid --backend nvidia --frontend vercel \
  --backend-url https://your-backend.example.com \
  --frontend-url https://your-frontend.vercel.app
```

This script:
1. Generates appropriate configurations for each platform
2. Applies environment-specific optimizations
3. Deploys components to their respective platforms

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions CI/CD pipeline that automates testing, building, and deploying to multiple platforms.

### Pipeline Structure

```
┌────────────────┐     ┌─────────────┐     ┌─────────────────┐
│                │     │             │     │                 │
│ GitHub Actions │────►│ Config Gen  │────►│ Platform Deploy │
│                │     │             │     │                 │
└────────────────┘     └─────────────┘     └─────────────────┘
```

### Pipeline Jobs

1. **Test**: Runs automated tests, linting, and type checking
2. **Build**: Creates Docker images and other deployment artifacts
3. **Generate Config**: Detects platforms and generates deployment configurations
4. **Deploy Backend**: Deploys to the selected backend platform (NVIDIA, Together.ai, or SAP BTP)
5. **Deploy Frontend**: Deploys to the selected frontend platform (Vercel or SAP BTP)

### Triggering Deployments

The pipeline can be triggered in two ways:

1. **Automatic**: On push to the main branch
2. **Manual**: Using GitHub Actions workflow_dispatch with parameters:
   - Deployment mode (full, api_only, ui_only)
   - Backend platform (nvidia, together, btp, auto)
   - Frontend platform (vercel, btp, auto)

### Auto-Detection

The pipeline supports automatic platform detection when `auto` is selected, using environment variables and system capabilities to determine the optimal deployment platform.

### Secret Management

Sensitive configuration values are stored as GitHub Secrets:
- `NVIDIA_API_KEY`: For NVIDIA NGC deployments
- `TOGETHER_API_KEY`: For Together.ai deployments
- `CF_API`, `CF_USERNAME`, `CF_PASSWORD`, etc.: For SAP BTP deployments
- `VERCEL_TOKEN`: For Vercel deployments

## Environment Detection

The application includes automatic environment detection to apply platform-specific optimizations:

- NVIDIA LaunchPad: Enables GPU-specific optimizations (TensorRT, Flash Attention, etc.)
- Together.ai: Configures API access and authentication
- SAP BTP: Sets up HANA connectivity and BTP-specific security

## Configuration Templates

Pre-configured templates are available for common deployment scenarios:

- `deployment/templates/nvidia-backend.env`: NVIDIA LaunchPad backend configuration
- `deployment/templates/together-backend.yaml`: Together.ai backend configuration
- `deployment/templates/btp-backend.env`: SAP BTP backend configuration
- `deployment/templates/vercel-frontend.json`: Vercel frontend configuration
- `deployment/templates/btp-frontend.env`: SAP BTP frontend configuration
- `deployment/templates/btp-full.env`: Full stack deployment on SAP BTP

## Configuration Generator

The `deploy-config.py` script provides a powerful configuration generation tool:

```bash
python ./deployment/deploy-config.py \
  --mode [full|api_only|ui_only|hybrid] \
  --backend [nvidia|together|btp|auto] \
  --frontend [vercel|btp|auto] \
  --backend-url https://your-backend.example.com \
  --frontend-url https://your-frontend.vercel.app \
  --output-dir ./deployment/generated
```

This tool:
1. Automatically detects suitable platforms if set to `auto`
2. Generates appropriate configuration files for each platform
3. Sets proper cross-origin settings for frontend/backend communication
4. Optimizes configurations for each specific platform

## Security Considerations

1. **Authentication**: API keys are required for all backend requests
2. **CORS**: Only configured origins can access the API
3. **HTTPS**: Enforced in production environments
4. **Cross-Origin Isolation**: Strict CSP headers in frontend deployments

## Monitoring and Logging

Regardless of deployment platform, consistent monitoring is available:

- Prometheus metrics endpoint at `/metrics`
- Health check endpoint at `/health`
- Backend status endpoint at `/health/backend-status`
- Structured JSON logging to standard output
- Optional OpenTelemetry integration

## Performance Optimization

Each deployment combination is optimized for its specific environment:

- NVIDIA: GPU memory management, mixed precision, and model quantization
- Together.ai: Optimized model selection and request batching
- SAP BTP: Connection pooling and efficient HANA integration

## Recommended Deployments

| Scenario | Recommended Deployment |
|----------|------------------------|
| Maximum performance | NVIDIA Backend + Vercel Frontend |
| Simplified deployment | Together.ai Backend + Vercel Frontend |
| Enterprise integration | SAP BTP Backend + SAP BTP Frontend |
| Low latency requirements | NVIDIA Backend + Same-region Vercel Frontend |
| Global distribution | Together.ai Backend + Edge-optimized Vercel Frontend |

## Multi-Backend Failover

The system supports automatic failover between multiple backend platforms:

```
                         ┌─────────────────┐
                    ┌───►│ NVIDIA LaunchPad│
                    │    │    (Primary)    │
                    │    └─────────────────┘
┌─────────────┐     │    ┌─────────────────┐
│             │     ├───►│   Together.ai   │
│  Frontend   │◄────┤    │   (Secondary)   │
│             │     │    └─────────────────┘
└─────────────┘     │    ┌─────────────────┐
                    └───►│    CPU-based    │
                         │    (Fallback)   │
                         └─────────────────┘
```

This provides high availability and resilience in production environments.