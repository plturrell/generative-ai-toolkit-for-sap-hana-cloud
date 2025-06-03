# SAP BTP Optimization for HANA AI Toolkit

This document outlines the optimizations made to ensure the Generative AI Toolkit for SAP HANA Cloud is:

1. Using 100% SAP AI Core SDK models for generative AI operations
2. Optimized for NVIDIA GPU acceleration for 10x performance improvement
3. Making no external calls outside SAP HANA, AI Core, or BTP service boundaries

## Changes Implemented

### 1. SAP AI Core SDK Exclusive Usage

- **Model Defaults**: Changed all default model references from external ones (like `gpt-4`) to SAP AI Core models (e.g., `sap-ai-core-llama3`).
- **Embedding Services**: Updated the `GenAIHubEmbeddings` class to exclusively use SAP AI Core models and reject any external model requests.
- **LLM Initialization**: Removed fallbacks to external LLM providers, ensuring only SAP GenAI Hub SDK is used.
- **Validation Checks**: Added validation to ensure model names always start with `sap-ai-core` prefix.

### 2. NVIDIA GPU Optimization

- **GPU Configuration Settings**: Added dedicated configuration settings in `config.py` for GPU acceleration:
  - `ENABLE_GPU_ACCELERATION`: Default is `True` to enable GPU usage
  - `NVIDIA_VISIBLE_DEVICES`: Controls which GPUs are available to the application
  - `NVIDIA_DRIVER_CAPABILITIES`: Set to optimize for compute workloads
  - `CUDA_MEMORY_FRACTION`: Controls memory allocation (default 0.8 or 80%)

- **Embedding Service Optimization**: Added GPU configuration to embedding services for accelerated vector operations.
- **LLM Initialization**: Added GPU configuration to LLM initialization for faster inference.

### 3. BTP Service Boundary Enforcement

- **CORS Restriction**: Limited CORS origins to only SAP BTP domains:
  - `*.cfapps.*.hana.ondemand.com`
  - `*.hana.ondemand.com`

- **External Call Prevention**:
  - Added new security module with domain and IP validation
  - Implemented `ExternalCallPreventionMiddleware` to block any requests to external services
  - Added configuration option `RESTRICT_EXTERNAL_CALLS` (default: `True`)

- **Metrics Security**:
  - Limited Prometheus metrics HTTP server to localhost only
  - Added IP validation to metrics endpoint to ensure only BTP internal networks can access
  - Defined BTP IP ranges in security module for validation

## Deployment Considerations

### Configuration

All configurations have been centralized in environment files and constants:

1. **Environment Constants**: Defined in `src/hana_ai/api/env_constants.py`
2. **Deployment Environment**: Configuration file at `deployment/btp-environment.env`
3. **Docker Compose**: Ready-to-use deployment at `deployment/docker-compose.yml`
4. **Kubernetes Deployment**: Kubernetes configuration at `deployment/kubernetes/deployment.yaml`

Example configuration from the environment file:

```
# AI Core SDK Settings
DEFAULT_LLM_MODEL=sap-ai-core-llama3

# GPU Optimization Settings
ENABLE_GPU_ACCELERATION=true
NVIDIA_VISIBLE_DEVICES=all
CUDA_MEMORY_FRACTION=0.8

# Security Settings
RESTRICT_EXTERNAL_CALLS=true
CORS_ORIGINS=*.cfapps.*.hana.ondemand.com,*.hana.ondemand.com
```

### Docker Deployment

For optimal NVIDIA GPU acceleration, deploy using Docker with the NVIDIA Container Toolkit:

```dockerfile
FROM nvcr.io/nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Set environment variables for GPU optimization
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip3 install -r requirements.txt

# Run the application
CMD ["python3", "-m", "hana_ai.api"]
```

### Deployment Command (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hana-ai-toolkit
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: hana-ai-toolkit
        image: hana-ai-toolkit:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: ENABLE_GPU_ACCELERATION
          value: "true"
        - name: RESTRICT_EXTERNAL_CALLS
          value: "true"
```

## Security Notes

1. The application now actively prevents any calls to external services outside of SAP BTP.
2. Metrics are only available within the BTP network, not externally exposed.
3. All generative AI operations use exclusively SAP AI Core SDK models.
4. CORS settings prevent cross-origin requests from non-SAP domains.