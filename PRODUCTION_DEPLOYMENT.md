# Production Deployment Guide

This guide covers the production deployment of the Generative AI Toolkit for SAP HANA Cloud with NVIDIA T4 GPU optimization.

## Prerequisites

- NVIDIA T4 GPU (or better)
- NVIDIA drivers and Docker runtime installed
- Docker and Docker Compose
- Access to SAP HANA Cloud instance
- Git access to this repository

## SAP HANA Cloud Connection

The toolkit is configured to connect to the following SAP HANA Cloud instance:

- **Host**: d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com
- **Port**: 443
- **User**: DBADMIN
- **Schema**: SAP_HANA_AI (optional)
- **Encryption**: Enabled

> **IMPORTANT**: The password is stored in the production-environment.env file and should be changed in production.

## Deployment Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud.git
   cd generative-ai-toolkit-for-sap-hana-cloud
   ```

2. Test the HANA connection:
   ```bash
   ./test_hana_connection.py --env-file production-environment.env
   ```

3. Deploy the production stack:
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

4. Verify deployment:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v1/config/test/hana
   ```

## Production Configuration

The production deployment uses the following files:

- `docker-compose.production.yml`: Docker Compose configuration for production
- `production-environment.env`: Environment variables with production settings
- `deployment/nvidia-t4/Dockerfile`: Dockerfile optimized for T4 GPUs

## Security Considerations

1. **Password Management**: 
   - Change the default passwords in production-environment.env
   - Consider using Docker secrets or environment variables for sensitive information

2. **Network Security**:
   - The production setup exposes several ports (8000, 9090, 3000, 80, 443)
   - Configure firewall rules to restrict access to these ports
   - Use HTTPS for all external access

3. **Authentication**:
   - The API authentication is enabled (AUTH_REQUIRED=true)
   - Configure the JWT secret for secure token generation

## Monitoring

The production deployment includes:

- Prometheus metrics at http://localhost:9091
- Grafana dashboard at http://localhost:3000
- T4 GPU-specific monitoring through nvidia-smi-exporter

## Scaling

For horizontal scaling in production:

1. Use Docker Swarm or Kubernetes (configuration files provided in deployment/)
2. Configure load balancing for the API endpoints
3. Use a shared volume for model caching across instances

## Maintenance

Regular maintenance tasks:

1. **Log Rotation**: Configure log rotation for Docker container logs
2. **Updates**: Regularly update the Docker images for security patches
3. **Backups**: Back up the Prometheus and Grafana data volumes
4. **Performance Tuning**: Adjust TensorRT and batch sizing parameters based on workload

## Troubleshooting

Common issues and solutions:

1. **HANA Connection Issues**: 
   - Verify network connectivity to the HANA instance
   - Check credentials in the environment file
   - Use ./test_hana_connection.py for detailed error information

2. **GPU Issues**: 
   - Verify NVIDIA drivers are installed and working (nvidia-smi)
   - Check Docker GPU runtime is enabled (docker info)
   - Monitor GPU usage with the Grafana dashboard

3. **API Performance Issues**: 
   - Adjust batch sizing parameters in the environment file
   - Optimize TensorRT engine settings
   - Consider increasing the CUDA memory fraction

## Support

For support issues, contact:

- SAP HANA Cloud support for database issues
- NVIDIA support for GPU-related issues
- Open issues on the GitHub repository for toolkit-specific problems