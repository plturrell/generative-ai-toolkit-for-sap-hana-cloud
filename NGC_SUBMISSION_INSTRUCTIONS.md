# NVIDIA NGC Submission Instructions

Follow these steps to submit the SAP HANA AI Toolkit to NVIDIA NGC:

## Prerequisites

1. NVIDIA Developer Program account (sign up at https://developer.nvidia.com/)
2. NGC API key (obtain from your NVIDIA NGC account settings)
3. NGC CLI tool installed (`pip install nvidia-ngccli`)

## Submission Steps

### 1. Login to NGC

```bash
ngc config set
```

Follow the prompts to enter your API key.

### 2. Create and Push the Docker Image

```bash
# Build the Docker image using the Dockerfile.ngc
docker build -t hana-ai-toolkit:latest -f deployment/Dockerfile.ngc .

# Tag the image for NGC
docker tag hana-ai-toolkit:latest nvcr.io/ea-sap/hana-ai-toolkit:latest

# Login to the NGC container registry
docker login nvcr.io

# Push the image
docker push nvcr.io/ea-sap/hana-ai-toolkit:latest
```

### 3. Submit the NGC Blueprint

```bash
# Validate the blueprint file
ngc blueprint validate ngc-blueprint.json

# Submit the blueprint
ngc blueprint submit --file ngc-blueprint.json --name "SAP HANA AI Toolkit" --version 1.0.0
```

### 4. Complete the NGC LaunchPad Submission Form

Visit the NGC LaunchPad submission portal and complete the following information:

1. **Basic Information**
   - Solution Name: SAP HANA AI Toolkit
   - Version: 1.0.0
   - Description: Generative AI Toolkit for SAP HANA Cloud with NVIDIA GPU optimization
   - Categories: Enterprise AI, Machine Learning, Database

2. **System Requirements**
   - GPU: NVIDIA A100, H100 (preferred)
   - CPU: 8+ cores
   - Memory: 32GB+
   - Storage: 50GB+
   - OS: Ubuntu 22.04
   - CUDA: 12.2

3. **Features and Benefits**
   - Enterprise-ready Generative AI integration with SAP HANA Cloud
   - Optimized for NVIDIA H100 GPUs with FP8 and Transformer Engine
   - Advanced multi-GPU distribution for large models
   - Comprehensive monitoring and observability
   - Production-ready deployment configurations

4. **Supporting Materials**
   - Upload screenshots from the /img directory
   - Link to the GitHub repository
   - Link to documentation

5. **Contact Information**
   - Primary contact details
   - Support contact details

### 5. Finalize Submission

After completing the form and uploading all materials:

1. Review the submission for accuracy
2. Accept the NGC terms and conditions
3. Submit for review

The NVIDIA team will review the submission and contact you with any questions or when the solution is approved for publication on NGC LaunchPad.

## Additional Resources

- [NVIDIA NGC Documentation](https://docs.nvidia.com/ngc/)
- [NVIDIA LaunchPad Documentation](https://docs.nvidia.com/launchpad/)
- [Docker Documentation](https://docs.docker.com/)
- [NGC Blueprint Format](https://docs.nvidia.com/ngc/ngc-catalog-api-user-guide/index.html#blueprint-format)

For any questions regarding the submission process, contact the NVIDIA Enterprise AI team at ea-support@nvidia.com.