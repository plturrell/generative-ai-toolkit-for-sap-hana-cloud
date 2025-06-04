# Examples for Generative AI Toolkit for SAP HANA Cloud

This directory contains example scripts that demonstrate how to use the Generative AI Toolkit for SAP HANA Cloud.

## Basic Usage

The [basic_usage.py](basic_usage.py) script demonstrates how to use the core functionality of the toolkit:

- Connecting to SAP HANA Cloud
- Using the vector store for embeddings and retrieval
- Working with smart dataframes
- Creating and using SQL agents

```bash
# Set environment variables
export HANA_HOST=your-hana-host
export HANA_PORT=your-hana-port
export HANA_USER=your-hana-user
export HANA_PASSWORD=your-hana-password
export OPENAI_API_KEY=your-openai-api-key

# Run the example
python basic_usage.py
```

## Advanced GPU Optimization

The [advanced_gpu_optimization.py](advanced_gpu_optimization.py) script demonstrates how to use NVIDIA GPU optimizations including:

- GPTQ and AWQ quantization
- Flash Attention 2
- Transformer Engine
- Mixed precision with FP8/FP16
- Performance benchmarking

```bash
# Set environment variables
export HANA_HOST=your-hana-host
export HANA_PORT=your-hana-port
export HANA_USER=your-hana-user
export HANA_PASSWORD=your-hana-password
export OPENAI_API_KEY=your-openai-api-key
export HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
export HF_API_TOKEN=your-huggingface-token

# GPU optimization settings
export ENABLE_GPU_OPTIMIZATIONS=true
export DEFAULT_QUANT_METHOD=gptq # or awq
export ENABLE_FLASH_ATTENTION=true
export ENABLE_TRANSFORMER_ENGINE=true
export USE_INT4_PRECISION=true

# Run the example
python advanced_gpu_optimization.py
```

## SAP HANA Integration

The [sap_hana_integration.py](sap_hana_integration.py) script demonstrates advanced integration with SAP HANA Cloud:

- Time series forecasting
- Dataset reporting
- Time series checks
- Smart dataframe analysis

```bash
# Set environment variables
export HANA_HOST=your-hana-host
export HANA_PORT=your-hana-port
export HANA_USER=your-hana-user
export HANA_PASSWORD=your-hana-password
export OPENAI_API_KEY=your-openai-api-key

# Run the example
python sap_hana_integration.py
```

## Requirements

These examples require the full installation of the Generative AI Toolkit for SAP HANA Cloud:

```bash
pip install generative-ai-toolkit-for-sap-hana-cloud[all]
```

For GPU optimizations, you'll need:
- NVIDIA GPU with CUDA 12.2+
- NVIDIA driver 535.104.05+
- PyTorch 2.1.0+
- Transformer Engine 1.2+
- Flash Attention 2.3.0+

## Documentation

For more information, see the [official documentation](https://github.com/finsightsap/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README.md).