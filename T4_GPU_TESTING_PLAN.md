# T4 GPU Testing Plan for SAP HANA Generative AI Toolkit

This document outlines the comprehensive testing approach for validating the T4 GPU optimization and TensorRT integration in the SAP HANA Generative AI Toolkit. The plan covers automated testing scripts, performance benchmarking, and validation procedures.

## Testing Environment

- **Hardware**: NVIDIA T4 GPU (16GB VRAM)
- **Server**: Brev.dev Jupyter VM (jupyter0-4ckg1m6x0.brevlab.com)
- **CUDA Version**: 12.3.0
- **TensorRT Version**: 8.6.1
- **Python Version**: 3.10+

## Prerequisites

Before running the tests, ensure the following components are available:

1. Access to a server with NVIDIA T4 GPU
2. CUDA Toolkit 12.x installed
3. TensorRT 8.x installed
4. Docker and Docker Compose installed
5. Python 3.10+ with required packages

## Testing Suite Overview

The testing suite includes the following components:

1. **Environment Validation Tests**
   - GPU detection and verification
   - TensorRT availability and version check
   - Driver and library compatibility tests

2. **TensorRT Optimization Tests**
   - Embedding model optimization
   - FP16/INT8 precision validation
   - Memory management tests
   - Dynamic batch sizing tests
   - Adaptive batch sizing performance

3. **Performance Benchmarks**
   - Embedding generation throughput
   - Batch size optimization
   - GPU memory utilization
   - CPU vs. GPU comparison

4. **API Integration Tests**
   - Embedding generation API
   - Vector store operations
   - Hybrid search functionality
   - Error handling and recovery

## Running the Tests

### 1. Using the Automated Test Script

The `test_tensorrt_t4.py` script provides a comprehensive test suite for T4 GPU validation:

```bash
# Run all tests with default configuration
python test_tensorrt_t4.py

# Run tests with custom configuration
python test_tensorrt_t4.py --config custom_config.json

# Run tests against a specific API endpoint
python test_tensorrt_t4.py --url https://jupyter0-4ckg1m6x0.brevlab.com/api

# Run tests with specific model and precision
python test_tensorrt_t4.py --model "sentence-transformers/all-MiniLM-L6-v2" --precision fp16
```

### 2. Using the Shell Script Wrapper

The `run_tests.sh` script provides a convenient wrapper for running tests:

```bash
# Run all test suites
./run_tests.sh --all

# Run only TensorRT tests
./run_tests.sh --suite tensorrt

# Run GPU performance tests
./run_tests.sh --suite gpu_performance

# Run with verbose output
./run_tests.sh --all --verbose
```

### 3. Docker Compose Testing

To test the complete deployment with Docker Compose:

```bash
# Deploy with Docker Compose
docker-compose up -d

# Run tests against the deployed services
./run_tests.sh --config test_config.json

# View logs
docker-compose logs -f

# Tear down deployment
docker-compose down
```

## Test Configuration

The test configuration is stored in a JSON file (`test_config.json` by default) with the following structure:

```json
{
    "api_base_url": "https://jupyter0-4ckg1m6x0.brevlab.com",
    "results_dir": "test_results",
    "test_timeout": 300,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64, 128],
    "adaptive_batch": {
        "enabled": true,
        "min_batch_size": 1,
        "max_batch_size": 128,
        "default_batch_size": 32,
        "t4_optimized": true
    },
    "auth": {
        "enabled": false,
        "username": "",
        "password": ""
    },
    "tools": {
        "test_hanaml_tools": true,
        "test_code_template_tools": true,
        "test_agent_tools": true
    }
}
```

## Test Results and Reports

Test results are stored in the `test_results` directory (configurable) with the following components:

1. **JSON Results**: Raw test data in JSON format
2. **HTML Report**: User-friendly HTML report with test results and performance metrics
3. **Performance Metrics**: GPU utilization, memory usage, throughput, and speedup metrics
4. **Visualizations**: Charts and graphs showing performance comparisons

## Continuous Integration

The testing framework is designed to integrate with CI/CD pipelines:

1. **Automated Testing**: Triggered on pull requests and commits
2. **Regression Testing**: Ensures no performance degradation across versions
3. **Performance Metrics**: Tracks performance trends over time
4. **Deployment Verification**: Validates deployments to testing and production environments

## Performance Expectations

For NVIDIA T4 GPUs, the following performance targets are expected:

1. **TensorRT Speedup**: 2-5x faster than CPU-only or PyTorch processing
2. **Optimal Batch Size**: Expected in the range of 32-64 for embedding models
3. **Memory Utilization**: Efficient memory management with dynamic batch sizing
4. **Throughput**: 100+ embedding generations per second for small models
5. **Adaptive Batch Sizing**: Dynamically optimized batch sizes based on model size, input length, and memory availability to maximize throughput while preventing OOM errors

## Troubleshooting

If tests fail, consider the following troubleshooting steps:

1. **GPU Availability**: Ensure NVIDIA T4 GPU is visible to the container (`nvidia-smi`)
2. **TensorRT Installation**: Verify TensorRT is properly installed (`python -c "import tensorrt; print(tensorrt.__version__)"`)
3. **Memory Issues**: Check for GPU memory exhaustion during tests
4. **API Connectivity**: Ensure API endpoints are accessible
5. **Log Analysis**: Review Docker and application logs for errors

## Additional Resources

- [GPU Optimization Guide](GPU_OPTIMIZATION.md)
- [TensorRT Optimization Guide](TENSORRT_OPTIMIZATION.md)
- [NVIDIA T4 Deployment Guide](NVIDIA-DEPLOYMENT.md)
- [Docker Compose Setup](docker-compose.yml)

## Conclusion

This testing plan ensures comprehensive validation of T4 GPU optimization and TensorRT integration in the SAP HANA Generative AI Toolkit. By following this plan, you can verify that the toolkit is leveraging the full capabilities of NVIDIA T4 GPUs for optimal performance.