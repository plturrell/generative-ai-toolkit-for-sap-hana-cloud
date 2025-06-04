# T4 GPU Testing Framework

This document describes the comprehensive testing framework for the Generative AI Toolkit for SAP HANA Cloud, with a focus on NVIDIA T4 GPU optimization and TensorRT integration.

## Overview

The testing framework provides automated validation for:

- Environment configuration
- TensorRT optimization
- Vector store functionality 
- GPU performance benchmarks
- Error handling and recovery
- API endpoints
- AI toolkit-specific tools

## Requirements

- Python 3.8 or higher
- NVIDIA T4 GPU (for GPU-specific tests)
- TensorRT 8.0 or higher (for TensorRT tests)
- Required Python packages: requests, numpy, pandas, torch

## Test Configuration

The testing framework uses a configuration file (`test_config.json`) to specify test parameters. Here's an example configuration:

```json
{
    "api_base_url": "https://your-deployment-url.com",
    "results_dir": "test_results",
    "test_timeout": 300,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64, 128],
    "auth": {
        "enabled": false,
        "username": "",
        "password": ""
    },
    "hana_connection": {
        "address": "",
        "port": 0,
        "user": "",
        "password": ""
    },
    "tools": {
        "test_hanaml_tools": true,
        "test_code_template_tools": true,
        "test_agent_tools": true
    }
}
```

## Running Tests

The testing framework can be executed using the provided shell script:

```bash
# Run all test suites
./run_tests.sh --all

# Run a specific test suite
./run_tests.sh --suite gpu_performance

# Get verbose output
./run_tests.sh --all --verbose

# Specify a custom config file
./run_tests.sh --all --config my_config.json

# Specify a custom results directory
./run_tests.sh --all --results-dir my_results
```

## Test Suites

### Environment Tests

Validates the deployment environment:
- NVIDIA driver installation and version
- CUDA version
- TensorRT installation
- Python package versions
- GPU memory availability

### TensorRT Tests

Tests TensorRT optimization for the T4 GPU:
- Model loading and optimization
- Engine creation and serialization
- Precision modes (FP32, FP16, INT8)
- Inference performance

### Vector Store Tests

Tests vector store functionality:
- Embedding generation
- Vector similarity search
- MMR (Maximum Marginal Relevance) search
- Metadata filtering

### GPU Performance Tests

Benchmarks GPU performance:
- Embedding generation with various batch sizes
- CPU vs. GPU performance comparison
- Memory usage optimization
- Throughput and latency measurements

### Error Handling Tests

Tests error handling and recovery:
- Invalid input handling
- Timeout handling
- Resource exhaustion scenarios
- Graceful degradation

### API Tests

Tests API endpoints:
- Health check
- Embedding generation
- Vector search
- Authentication
- Rate limiting

### Tools Tests

Tests toolkit-specific tools:
- HANA ML tools
- Code template tools
- Agent tools

## Test Results

The testing framework generates detailed test results in multiple formats:

- JSON report: `test_results/test_report.json`
- HTML report: `test_results/test_report.html`
- Individual test results: `test_results/<suite_name>_results.json`

The HTML report includes:

- Summary of test results
- Performance metrics
- Recommendations for optimization
- Detailed test results by suite
- Tools performance metrics

## Performance Metrics

The testing framework measures several key performance metrics:

- **GPU Speedup**: Ratio of CPU time to GPU time for embedding generation
- **Optimal Batch Size**: Batch size that provides the best throughput/latency tradeoff
- **MMR Speedup**: Performance improvement for MMR search compared to baseline
- **Memory Efficiency**: GPU memory usage optimization

## Troubleshooting

If tests fail, check the following:

1. **Environment Issues**:
   - Verify NVIDIA drivers are installed and functioning
   - Check CUDA and TensorRT installations
   - Verify Python package versions

2. **API Connectivity**:
   - Ensure the API base URL is correct
   - Check for network connectivity issues
   - Verify authentication credentials if enabled

3. **Resource Limitations**:
   - Check for GPU memory exhaustion
   - Verify CPU resources are sufficient
   - Check disk space for test results

## Fallback Mechanisms

The testing framework includes fallback mechanisms for common issues:

- **API Simulation**: If the API is unavailable, tests can run in simulation mode
- **CPU Fallback**: GPU tests can fall back to CPU if no GPU is available
- **Reduced Batch Sizes**: Automatically reduces batch sizes if out-of-memory errors occur

## Adding Custom Tests

To add custom tests to the framework:

1. Create a new test suite file in the appropriate directory
2. Implement the test suite class with test methods
3. Register the test suite in the main test runner
4. Update the configuration file to include parameters for the new tests

## CI/CD Integration

The testing framework is designed to integrate with CI/CD pipelines:

- GitHub Actions integration
- GitLab CI integration
- Jenkins integration

For more information on CI/CD integration, see the [CI/CD documentation](https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README-CICD.md).