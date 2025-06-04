# Deploying SAP HANA AI Toolkit on Together.ai

This guide provides detailed instructions for deploying the SAP HANA AI Toolkit on Together.ai's dedicated endpoint service to leverage their GPU infrastructure.

## Overview

Together.ai offers dedicated GPU endpoints that allow you to deploy your own custom models and applications with GPU acceleration. This deployment option provides several benefits:

- Access to high-performance NVIDIA GPUs (A100, H100)
- Scalable infrastructure with autoscaling capabilities
- Optimized inference performance with TensorRT, Flash Attention, etc.
- Custom deployment configurations tailored to your needs

## Prerequisites

Before deploying to Together.ai, ensure you have:

1. Together.ai account with API key
2. Python 3.8 or later
3. Required Python packages:
   - `requests`
   - `pyyaml`
   - `logging`

## Deployment Configuration

The deployment configuration is defined in the `together.yaml` file, which specifies:

- Hardware requirements (GPU type, count)
- Model configuration (base model, quantization)
- Scaling parameters
- Network settings
- Advanced optimization options

Here's an example configuration:

```yaml
# Together.ai Dedicated Endpoint Configuration

# API Authentication
apiKey: "your-api-key-here"

# Endpoint Configuration
endpoint:
  name: "sap-hana-ai-toolkit"
  description: "SAP HANA AI Toolkit with GPU Acceleration"
  
  # Hardware Configuration
  hardware:
    instanceType: "a100-40gb"  # Options: a100-40gb, a100-80gb, h100-80gb
    count: 1                   # Number of GPUs
    
  # Scaling Configuration
  scaling:
    minReplicas: 1
    maxReplicas: 2
    targetUtilization: 80      # Percentage
    
  # Model Configuration
  model:
    # Base model to use
    baseModel: "meta-llama/Llama-2-70b-chat-hf"
    
    # Quantization (optional)
    quantization:
      enabled: true
      method: "awq"           # Options: awq, gptq, none
      bits: 4                 # Options: 4, 8
```

## Deployment Process

### Option 1: Using the Deployment Script

The toolkit includes a deployment script that handles the deployment process:

```bash
# Set your API key as an environment variable
export TOGETHER_API_KEY="your-api-key-here"

# Run the deployment script
./deploy_together.py --config together.yaml --output deployment_info.json
```

The script will:
1. Validate your configuration
2. Create a deployment on Together.ai
3. Wait for the deployment to be ready
4. Output the deployment information (ID, endpoint URL, etc.)

### Option 2: Using the Python API

You can also deploy programmatically using the Python API:

```python
from src.hana_ai.api.together_endpoint import deploy_to_together

# Deploy to Together.ai
deployment_info = deploy_to_together(
    config_path="together.yaml",
    api_key="your-api-key-here"
)

print(f"Deployment ID: {deployment_info.get('deployment_id')}")
print(f"Endpoint URL: {deployment_info.get('endpoint_url')}")
```

## Configuration Options

### Hardware Options

Together.ai offers several GPU options:

| Instance Type | Description | Use Case |
|---------------|-------------|----------|
| a100-40gb     | NVIDIA A100 with 40GB memory | General purpose |
| a100-80gb     | NVIDIA A100 with 80GB memory | Larger models |
| h100-80gb     | NVIDIA H100 with 80GB memory | Maximum performance |

### Quantization Options

Quantization reduces model size and improves inference speed:

| Method | Description | Trade-offs |
|--------|-------------|------------|
| awq    | Activation-aware Weight Quantization | Better quality, slower inference |
| gptq   | Generative Pre-trained Transformer Quantization | Faster inference, slightly lower quality |
| none   | No quantization | Highest quality, requires more GPU memory |

### Scaling Options

Configure how your deployment scales:

- `minReplicas`: Minimum number of instances (1 recommended)
- `maxReplicas`: Maximum number of instances for scaling
- `targetUtilization`: GPU utilization percentage that triggers scaling

## Post-Deployment

After successful deployment:

1. Update your application configuration to use the Together.ai endpoint URL
2. Set the API key for authentication
3. Monitor usage and performance through the Together.ai dashboard

### Example Code for Using the Endpoint

```python
import requests

# Together.ai endpoint details
endpoint_url = "https://api.together.xyz/v1/deployments/your-deployment-id/completions"
api_key = "your-api-key-here"

# Make a request to the endpoint
response = requests.post(
    endpoint_url,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "prompt": "Explain SAP HANA Cloud in simple terms.",
        "max_tokens": 100,
        "temperature": 0.7
    }
)

# Print the response
print(response.json())
```

## Monitoring and Management

### Monitoring Your Deployment

Monitor your deployment through:

1. Together.ai dashboard
2. Metrics endpoint (if enabled)
3. Logs (if enabled)

### Updating Your Deployment

To update an existing deployment:

```python
from src.hana_ai.api.together_endpoint import TogetherEndpointDeployer

# Initialize the deployer
deployer = TogetherEndpointDeployer(api_key="your-api-key-here")

# Update the deployment
deployer.update_deployment(
    deployment_id="your-deployment-id",
    updates={
        "scaling": {
            "max_replicas": 3
        }
    }
)
```

### Deleting Your Deployment

To delete a deployment:

```python
from src.hana_ai.api.together_endpoint import TogetherEndpointDeployer

# Initialize the deployer
deployer = TogetherEndpointDeployer(api_key="your-api-key-here")

# Delete the deployment
deployer.delete_deployment(deployment_id="your-deployment-id")
```

## Troubleshooting

Common issues and solutions:

1. **Deployment fails to start**:
   - Check your API key
   - Verify your configuration
   - Ensure you have sufficient quota for the requested hardware

2. **Performance issues**:
   - Adjust quantization settings
   - Increase hardware resources
   - Enable optimizations like TensorRT and Flash Attention

3. **High costs**:
   - Reduce the minimum number of replicas
   - Use more aggressive quantization
   - Monitor and adjust max batch size

## Best Practices

1. **Cost Optimization**:
   - Use quantization to reduce GPU memory needs
   - Configure proper autoscaling
   - Set appropriate instance count

2. **Performance Optimization**:
   - Enable Flash Attention for faster inference
   - Use TensorRT for optimized performance
   - Configure appropriate batch sizes

3. **Reliability**:
   - Monitor deployment health
   - Set up alerts for issues
   - Have a fallback strategy

## Resources

- [Together.ai Documentation](https://docs.together.ai/)
- [Together.ai API Reference](https://docs.together.ai/reference/)
- [Together.ai Pricing](https://www.together.ai/pricing)