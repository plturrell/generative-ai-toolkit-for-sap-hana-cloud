# Deployment Guide for SAP HANA AI Toolkit

This directory contains files for deploying the Generative AI Toolkit for SAP HANA Cloud in a production environment, optimized for:

1. Using 100% SAP AI Core SDK models for generative AI
2. NVIDIA GPU acceleration for 10x performance improvement
3. No external calls outside of SAP HANA, AI Core, or BTP service boundaries

## Deployment Files

- **btp-environment.env**: Environment configuration file with all settings
- **docker-compose.yml**: Docker Compose configuration for containerized deployment
- **Dockerfile**: Container definition optimized for NVIDIA GPU acceleration

## Deployment Options

### 1. Docker Compose (Recommended for Testing)

```bash
# First, configure your environment variables
cp btp-environment.env.example btp-environment.env
# Edit btp-environment.env with your specific configuration

# Build and start the services
docker-compose up -d
```

### 2. Kubernetes Deployment in SAP BTP

For production deployment in SAP BTP Kubernetes:

1. Build the Docker image:
   ```bash
   docker build -t your-registry/hana-ai-toolkit:latest -f Dockerfile ..
   docker push your-registry/hana-ai-toolkit:latest
   ```

2. Create a Kubernetes deployment:
   ```bash
   # Generate a ConfigMap from environment file
   kubectl create configmap hana-ai-config --from-env-file=btp-environment.env
   
   # Create secrets for sensitive data
   kubectl create secret generic hana-ai-secrets \
     --from-literal=HANA_PASSWORD=your-password \
     --from-literal=API_KEYS=your-api-keys
   
   # Apply the deployment
   kubectl apply -f kubernetes/deployment.yaml
   ```

## Environment Configuration

The `btp-environment.env` file contains all configuration parameters. Key settings include:

### SAP AI Core SDK Configuration

```
DEFAULT_LLM_MODEL=sap-ai-core-llama3
```

### NVIDIA GPU Optimization

```
ENABLE_GPU_ACCELERATION=true
NVIDIA_VISIBLE_DEVICES=all
CUDA_MEMORY_FRACTION=0.8
```

### Security Settings

```
RESTRICT_EXTERNAL_CALLS=true
CORS_ORIGINS=*.cfapps.*.hana.ondemand.com,*.hana.ondemand.com
```

## Monitoring

The deployment includes Prometheus for metrics collection and Grafana for visualization. Both services are configured to only be accessible within the internal network and not exposed externally.

Access Grafana at http://localhost:3000 (default credentials: admin/admin).

## Security Considerations

1. **API Keys**: Update the `API_KEYS` setting with strong, randomly generated keys
2. **HANA Credentials**: Use `HANA_USERKEY` when possible instead of username/password
3. **CORS Settings**: Only allow trusted domains in production
4. **Network Security**: Ensure the API is behind proper network security in BTP

## Troubleshooting

1. **GPU Issues**: Check GPU availability with `nvidia-smi` inside the container
2. **Connection Issues**: Verify HANA credentials and connectivity
3. **Performance Issues**: Check Prometheus metrics for bottlenecks