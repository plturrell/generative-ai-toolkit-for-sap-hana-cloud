# Multi-Backend Deployment Guide

This guide explains how to deploy and configure the SAP HANA AI Toolkit with multiple GPU backends for flexibility, performance, and reliability.

## Overview

The SAP HANA AI Toolkit supports three backend options:

1. **NVIDIA LaunchPad** - High-performance dedicated GPU hardware
2. **Together.ai Cloud** - Managed GPU service through API
3. **CPU-only** - Fallback processing without GPU acceleration

You can configure the system to use:
- A single backend
- Multiple backends with automatic failover
- Multiple backends with load balancing

## Configuration Options

### 1. Environment Variables

Configure backends through environment variables:

```
# Global GPU Settings
ENABLE_GPU_ACCELERATION=true

# Backend Priority
PRIMARY_BACKEND=nvidia     # Options: nvidia, together_ai, cpu, auto
SECONDARY_BACKEND=together_ai  # Options: nvidia, together_ai, cpu, auto
AUTO_FAILOVER=true
LOAD_BALANCING=false
LOAD_RATIO=0.8            # Percentage of requests to primary backend

# NVIDIA Backend
ENABLE_TENSORRT=true
ENABLE_FLASH_ATTENTION=true
ENABLE_TRANSFORMER_ENGINE=true
ENABLE_FP8=true
ENABLE_GPTQ=true
ENABLE_AWQ=true
DEFAULT_QUANT_METHOD=gptq
QUANTIZATION_BIT_WIDTH=4
CUDA_MEMORY_FRACTION=0.85

# Together.ai Backend
ENABLE_TOGETHER_AI=true
TOGETHER_API_KEY=your-api-key
TOGETHER_DEFAULT_MODEL=meta-llama/Llama-2-70b-chat-hf
TOGETHER_DEFAULT_EMBEDDING_MODEL=togethercomputer/m2-bert-80M-8k-retrieval

# CPU Backend
CPU_DEFAULT_MODEL=llama-2-7b-chat.Q4_K_M.gguf
CPU_DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
CPU_NUM_THREADS=4
```

### 2. Configuration File

Alternatively, use a JSON configuration file for more detailed settings:

```json
{
  "priority": {
    "primary": "nvidia",
    "secondary": "together_ai",
    "auto_failover": true,
    "failover_attempts": 3,
    "failover_timeout": 10.0,
    "load_balancing": false,
    "load_ratio": 0.8
  },
  "nvidia": {
    "enabled": true,
    "enable_tensorrt": true,
    "enable_flash_attention": true,
    "enable_transformer_engine": true,
    "enable_fp8": true,
    "enable_gptq": true,
    "enable_awq": true,
    "default_quant_method": "gptq",
    "quantization_bit_width": 4,
    "cuda_memory_fraction": 0.85,
    "multi_gpu_strategy": "auto"
  },
  "together_ai": {
    "enabled": true,
    "api_key": "your-api-key-here",
    "default_model": "meta-llama/Llama-2-70b-chat-hf",
    "default_embedding_model": "togethercomputer/m2-bert-80M-8k-retrieval",
    "timeout": 60.0,
    "max_retries": 3
  },
  "cpu": {
    "enabled": true,
    "default_model": "llama-2-7b-chat.Q4_K_M.gguf",
    "default_embedding_model": "all-MiniLM-L6-v2",
    "num_threads": 4,
    "context_size": 2048
  }
}
```

## Deployment Scenarios

### Scenario 1: NVIDIA LaunchPad Only

For maximum performance with dedicated hardware:

```
PRIMARY_BACKEND=nvidia
SECONDARY_BACKEND=cpu      # Fallback to CPU if needed
ENABLE_TOGETHER_AI=false   # Disable Together.ai

# NVIDIA optimizations
ENABLE_TENSORRT=true
ENABLE_FLASH_ATTENTION=true
ENABLE_TRANSFORMER_ENGINE=true
```

Deploy using:
```bash
# Deploy on NGC
python -m hana_ai.api
```

### Scenario 2: Together.ai Cloud Only

For easy deployment without managing hardware:

```
PRIMARY_BACKEND=together_ai
SECONDARY_BACKEND=cpu       # Fallback to CPU if needed
ENABLE_GPU_ACCELERATION=false  # Disable local GPU

# Together.ai settings
ENABLE_TOGETHER_AI=true
TOGETHER_API_KEY=your-api-key
```

Deploy using:
```bash
# Deploy on Vercel
vercel --prod
```

### Scenario 3: Multiple Backends with Failover

For high availability with automatic failover:

```
PRIMARY_BACKEND=nvidia
SECONDARY_BACKEND=together_ai
AUTO_FAILOVER=true
LOAD_BALANCING=false

# Enable both backends
ENABLE_GPU_ACCELERATION=true
ENABLE_TOGETHER_AI=true
TOGETHER_API_KEY=your-api-key
```

### Scenario 4: Multiple Backends with Load Balancing

For distributing load across backends:

```
PRIMARY_BACKEND=nvidia
SECONDARY_BACKEND=together_ai
AUTO_FAILOVER=true
LOAD_BALANCING=true
LOAD_RATIO=0.7  # 70% to NVIDIA, 30% to Together.ai

# Enable both backends
ENABLE_GPU_ACCELERATION=true
ENABLE_TOGETHER_AI=true
TOGETHER_API_KEY=your-api-key
```

## Administration UI

The toolkit provides a web-based administration interface for configuring and monitoring the multi-backend system:

1. Access the admin panel at `https://your-deployment-url/admin`
2. Click on "GPU Acceleration & Backends" in the Advanced Configuration section
3. Use the interface to configure backend priorities, NVIDIA settings, Together.ai credentials, and CPU fallback options
4. Test individual backends and failover mechanisms

The admin UI provides a user-friendly way to manage all backend settings without editing configuration files or setting environment variables directly.

## Monitoring Backend Status

The toolkit provides a status endpoint to monitor backend health:

```bash
curl https://your-deployment-url/api/v1/backend/status
```

Response example:
```json
{
  "primary_backend": "nvidia",
  "secondary_backend": "together_ai",
  "backends": {
    "nvidia": {
      "status": "available",
      "error": null
    },
    "together_ai": {
      "status": "available",
      "error": null
    },
    "cpu": {
      "status": "available",
      "error": null
    }
  },
  "load_balancing": true,
  "auto_failover": true,
  "initialized_backends": ["nvidia", "together_ai", "cpu"],
  "failover": {
    "nvidia_backend": {
      "is_healthy": true,
      "failure_count": 0,
      "circuit_state": "CLOSED"
    },
    "together_ai_backend": {
      "is_healthy": true,
      "failure_count": 0,
      "circuit_state": "CLOSED"
    },
    "cpu_backend": {
      "is_healthy": true,
      "failure_count": 0,
      "circuit_state": "CLOSED"
    }
  }
}
```

## Failover Behavior

The multi-backend system provides several resilience features:

1. **Automatic Failover**: If the primary backend fails, requests automatically route to the secondary backend.

2. **Circuit Breaker**: After multiple failures, a backend is marked as unavailable and requests stop being sent to it.

3. **Health Checks**: Regular health checks attempt to recover failed backends.

4. **Retries**: Failed requests are automatically retried with exponential backoff.

5. **Timeouts**: All operations have configurable timeouts to prevent blocking.

## Performance Considerations

When using multiple backends:

1. **Latency**: Together.ai has higher latency than local NVIDIA GPUs due to network overhead.

2. **Throughput**: NVIDIA LaunchPad typically offers higher throughput for batch operations.

3. **Cost**: Together.ai charges per token, while NVIDIA has fixed infrastructure costs.

4. **Memory**: Large models may require NVIDIA A100/H100 or model quantization.

## Best Practices

1. **Testing**: Test both backends independently before enabling load balancing.

2. **Monitoring**: Monitor backend status and performance metrics.

3. **Fallback Models**: Configure appropriate models for CPU fallback.

4. **API Keys**: Secure your Together.ai API key as an environment variable.

5. **Quota Management**: Monitor Together.ai usage to avoid exceeding quotas.

## Troubleshooting

Common issues and solutions:

1. **Backend Unavailable**:
   - Check network connectivity
   - Verify API keys
   - Check hardware availability

2. **Slow Performance**:
   - Adjust load balancing ratio
   - Enable model quantization
   - Check for resource contention

3. **Failover Not Working**:
   - Verify AUTO_FAILOVER is enabled
   - Check backend initialization
   - Review error logs

4. **Backend Not Initializing**:
   - Check required libraries are installed
   - Verify hardware requirements
   - Check configuration parameters