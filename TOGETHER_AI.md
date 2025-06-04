# Together.ai Integration for SAP HANA AI Toolkit

This document describes how to use Together.ai's GPU-accelerated API with the SAP HANA AI Toolkit to provide high-performance generative AI capabilities without requiring local GPU hardware.

## Overview

Together.ai provides access to powerful GPU resources in the cloud, allowing you to run state-of-the-art large language models without managing your own GPU infrastructure. This integration enables the SAP HANA AI Toolkit to use Together.ai's API for:

- Generative AI text completions
- Embeddings generation
- Chat completions
- Access to latest models like Llama 2, Mixtral, and more

## Getting Started

### 1. Sign Up for Together.ai

1. Create an account at [Together.ai](https://together.ai/)
2. Obtain your API key from your account dashboard
3. Choose a pricing plan that matches your needs

### 2. Configure the SAP HANA AI Toolkit

Add the following environment variables to your deployment:

```
ENABLE_GPU_ACCELERATION=true
ENABLE_TOGETHER_AI=true
TOGETHER_API_KEY=your_api_key_here
TOGETHER_DEFAULT_MODEL=meta-llama/Llama-2-70b-chat-hf
TOGETHER_DEFAULT_EMBEDDING_MODEL=togethercomputer/m2-bert-80M-8k-retrieval
```

These can be set in:
- `.env` file for local development
- Environment variables in your deployment platform
- Vercel environment configuration
- Kubernetes deployment YAML

### 3. Verify the Integration

To verify that the integration is working correctly:

```python
from hana_ai.api.together_ai import get_together_ai_client

# Get the client
client = get_together_ai_client()

# Generate a completion
response = client.generate_completion(
    prompt="Explain SAP HANA Cloud in simple terms.",
    max_tokens=100
)

print(response)
```

## Using Together.ai with the API

The SAP HANA AI Toolkit will automatically use Together.ai for:

1. **Embedding Generation**: When creating vector embeddings for text
2. **LLM Completions**: When generating text completions for agents
3. **Chat Completions**: For conversational interfaces

You don't need to change your code - the toolkit automatically detects the Together.ai configuration and routes requests appropriately.

## Available Models

Together.ai provides access to various models. Some popular options include:

| Model | Type | Use Case |
|-------|------|----------|
| meta-llama/Llama-2-70b-chat-hf | Chat/Completion | General purpose chat and completions |
| togethercomputer/m2-bert-80M-8k-retrieval | Embeddings | Text embeddings for vector search |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | Chat/Completion | Instruction-following tasks |
| meta-llama/Llama-2-13b-chat-hf | Chat/Completion | Lighter model for faster responses |

To use a specific model:

```python
client = get_together_ai_client()
response = client.generate_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key features of SAP HANA Cloud?"}
    ],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1"
)
```

## Production Configuration

For production deployments, consider:

1. **API Key Security**: Store your Together.ai API key securely using environment variables or a secrets manager
2. **Rate Limiting**: Implement rate limiting to control API usage costs
3. **Fallback Strategy**: Configure fallback options if Together.ai is unavailable
4. **Caching**: Cache common queries to reduce API calls

## Cost Management

Together.ai charges based on token usage. To manage costs:

1. Set appropriate token limits for completions
2. Cache results where appropriate
3. Monitor usage with Together.ai's dashboard
4. Consider batching similar requests

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**: Ensure your API key is correctly set in environment variables
2. **Model Not Found**: Verify the model name is correct and available on Together.ai
3. **Timeout Errors**: Increase the `TOGETHER_TIMEOUT` setting for complex operations
4. **Rate Limiting**: If you encounter rate limits, implement exponential backoff

## Limitations

Be aware of these limitations:

1. **Internet Dependency**: Requires internet access to Together.ai's API
2. **API Latency**: External API calls add latency compared to local processing
3. **Model Availability**: Only models supported by Together.ai can be used
4. **Cost Scaling**: Costs scale with usage, unlike fixed-cost GPU hardware

## Further Resources

- [Together.ai Documentation](https://docs.together.ai/)
- [Together.ai Model Library](https://together.ai/models)
- [API Reference](https://docs.together.ai/reference/inference)