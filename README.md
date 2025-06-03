# Generative AI Toolkit for SAP HANA Cloud

[![CI](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/ci.yml)
[![CD](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/cd.yml/badge.svg)](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/cd.yml)
[![API CI/CD](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/api-ci.yml/badge.svg)](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/actions/workflows/api-ci.yml)
[![NVIDIA Optimized](https://img.shields.io/badge/NVIDIA-Optimized-76B900)](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NVIDIA.md)
[![NGC Ready](https://img.shields.io/badge/NGC-Ready-76B900)](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NGC_DEPLOYMENT.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Project Board](https://img.shields.io/badge/Project-Board-0052CC)](https://github.com/users/plturrell/projects/1)

## About this project

Generative AI Client for SAP HANA Cloud is an extension of the existing HANA ML Python client library, mainly focusing on GenAI and related use cases. It includes many leading-edge GenAI related open source libraries and provides seamless integration with HANA ML, HANA vector engine, and other SAP GenAI Hub SDK.

### Key Features

- **NVIDIA GPU Acceleration**: 10x performance improvement with optimized GPU utilization - [GPU Optimization](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/GPU_OPTIMIZATION.md)
- **TensorRT Integration**: 2-3x faster inference with optimized model execution - [TensorRT Optimization](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/TENSORRT_OPTIMIZATION.md)
- **NGC Container Publishing**: Pre-optimized containers with NVIDIA NGC integration - [NGC Deployment](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NGC_DEPLOYMENT.md)
- **Canary Deployment**: Advanced deployment strategies with automated verification - [Canary Deployment](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/tree/main/deployment/canary)
- **Failover Handling**: Robust failover mechanisms for production workloads
- **SAP BTP Optimization**: Specialized configurations for SAP Business Technology Platform - [BTP Optimization](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/BTP_OPTIMIZATION.md)
- **Production-Ready**: Enterprise-grade deployment configurations - [Production Guide](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/PRODUCTION_READY.md)

## Requirements and Setup

The prerequisites for using the Generative AI Toolkit for SAP HANA Cloud are listed at [Prerequisites](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/wiki/Prerequisites).

The Generative AI Toolkit for SAP HANA Cloud is available as a Python package. You can install it via `pip`:

```bash
pip install hana-ai
```

For API deployments, see our [API Documentation](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README-API.md).

## Deployment Options

### Docker Deployment

```bash
# Build and run with Docker
docker build -t hana-ai-toolkit .
docker run -p 8000:8000 -p 9090:9090 hana-ai-toolkit
```

### NVIDIA NGC Container (Recommended for GPU Workloads)

```bash
# Pull the NGC optimized container
docker pull nvcr.io/ea-sap/hana-ai-toolkit:latest

# Run with GPU acceleration
docker run --gpus all -p 8000:8000 -p 9090:9090 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/ea-sap/hana-ai-toolkit:latest
```

For detailed NGC deployment options, see our [NGC Deployment Guide](NGC_DEPLOYMENT.md).

### SAP BTP Deployment

```bash
# Deploy to Cloud Foundry
./deployment/deploy.sh

# Deploy with Canary strategy (20% traffic)
CANARY=true CANARY_WEIGHT=20 ./deployment/deploy.sh
```

## Project Links

- [Project Board](https://github.com/users/plturrell/projects/1)
- [Wiki Documentation](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/wiki)
- [Authentication & Setup Guide](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/AUTHENTICATION.md)
- [CI/CD Pipeline](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README-CICD.md)
- [NVIDIA Integration](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NVIDIA.md)
- [NGC Deployment](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/NGC_DEPLOYMENT.md)
- [TensorRT Optimization](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/TENSORRT_OPTIMIZATION.md)
- [API Documentation](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README-API.md)

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/CONTRIBUTING.md).

## Security / Disclosure

If you find any bug that may be a security problem, please follow our instructions at [our security policy](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and contributors. Please see our [LICENSE](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/LICENSE) for copyright and license information.