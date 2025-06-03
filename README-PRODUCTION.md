# Production Deployment Guide for HANA AI Toolkit API

This guide provides comprehensive instructions for deploying the HANA AI Toolkit API in a production environment. Follow these steps to ensure a secure, scalable, and monitored deployment.

## System Requirements

- Docker and Docker Compose
- 4+ CPU cores
- 8+ GB RAM
- 20+ GB disk space
- Network access to SAP HANA Cloud
- HTTPS certificates for production use

## Security Checklist

Before deploying to production, ensure you have addressed these security requirements:

- [ ] Generate strong API keys (not default values)
- [ ] Configure HTTPS with valid certificates
- [ ] Use a secrets management solution for credentials
- [ ] Configure proper network security (firewall rules, etc.)
- [ ] Set up monitoring and alerting
- [ ] Enable rate limiting
- [ ] Implement proper logging
- [ ] Review and secure all API endpoints

## Configuration

### Environment Variables

Create a `.env` file based on these production values:

```
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEVELOPMENT_MODE=false
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security Settings
API_KEYS=your-strong-key-1,your-strong-key-2
AUTH_REQUIRED=true
CORS_ORIGINS=https://yourapplication.example.com
ENFORCE_HTTPS=true
SESSION_SECRET_KEY=your-random-secret-key

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Database Connection
HANA_HOST=your-hana-host.example.com
HANA_PORT=443
HANA_USER=your-hana-user
HANA_PASSWORD=your-hana-password
# Or use HANA userkey instead
# HANA_USERKEY=YOUR_KEY

# Performance Settings
CONNECTION_POOL_SIZE=20
REQUEST_TIMEOUT_SECONDS=300
MAX_REQUEST_SIZE_MB=10

# Monitoring Settings
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

### HTTPS Certificates

Place your SSL certificates in the `nginx/ssl/` directory:

- `nginx/ssl/fullchain.pem` - Your certificate chain
- `nginx/ssl/privkey.pem` - Your private key

For automated certificate management, consider using Let's Encrypt with Certbot.

## Deployment Options

### Docker Compose

The simplest way to deploy the API is using Docker Compose:

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Scale the API service if needed
docker-compose up -d --scale api=3
```

### Kubernetes

For production deployments, Kubernetes is recommended:

1. Build and push the Docker image:
   ```bash
   docker build -t your-registry/hana-ai-api:latest .
   docker push your-registry/hana-ai-api:latest
   ```

2. Apply Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/
   ```

3. Check deployment status:
   ```bash
   kubectl get pods -n hana-ai
   ```

Kubernetes manifests are available in the `kubernetes/` directory.

## Monitoring

The API exposes Prometheus metrics at `/metrics`. You can visualize these with Grafana.

### Available Metrics

- `api_requests_total` - Total number of API requests
- `api_request_latency_seconds` - API request latency
- `db_query_latency_seconds` - Database query latency
- `db_active_connections` - Number of active database connections
- `llm_request_latency_seconds` - LLM request latency
- `api_errors_total` - Total number of API errors

### Alerts

The Prometheus configuration includes alert rules for:

- High error rates
- Slow response times
- Database connection issues
- Memory and CPU usage

## Load Testing

Before full production deployment, perform load testing:

```bash
# Using k6 (https://k6.io/)
k6 run load-tests/api-load-test.js
```

Target metrics:
- Support 50+ concurrent users
- Handle 100+ requests per second
- Maintain 95th percentile response time under 1 second

## Backup and Disaster Recovery

1. **Database Backup**:
   - HANA Cloud native backups
   - Export vector stores periodically

2. **API Configuration Backup**:
   - Environment variables
   - SSL certificates
   - Prometheus and Grafana dashboards

3. **Recovery Plan**:
   - Document failover procedures
   - Maintain deployment scripts in version control

## Scaling Guidelines

- **Horizontal Scaling**: Add more API containers
- **Vertical Scaling**: Increase container resources
- **Database Connection Pooling**: Adjust `CONNECTION_POOL_SIZE`
- **Cache Tuning**: Configure `CACHE_TTL_SECONDS` 

## Security Hardening

- **Network Security**: Deploy in private subnet
- **API Key Rotation**: Rotate keys regularly
- **Vulnerability Scanning**: Scan containers
- **Audit Logging**: Enable for all endpoints

## Troubleshooting

Common issues and solutions:

1. **Connection issues**:
   - Check HANA credentials
   - Verify network connectivity

2. **High latency**:
   - Check database query performance
   - Monitor LLM response times
   - Increase connection pool size

3. **Memory issues**:
   - Adjust container memory limits
   - Check for memory leaks in logs

## Maintenance Procedures

1. **Updates**:
   - Deploy during low-traffic periods
   - Use blue-green deployment for zero downtime

2. **Monitoring**:
   - Review logs daily
   - Check Grafana dashboards
   - Set up alerts for critical metrics

3. **Backup**:
   - Daily database backup
   - Weekly configuration backup