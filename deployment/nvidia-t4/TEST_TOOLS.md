# NVIDIA T4 Automated Testing Tools

This directory contains a comprehensive suite of testing tools specifically designed for validating NVIDIA T4 deployments. These tools provide extensive testing capabilities for environment validation, performance benchmarking, and load testing.

## Available Tools

### 1. Automated Testing Framework (`run_automated_tests.py`)

A comprehensive testing framework that validates:
- Environment configuration (GPU detection, CUDA, Docker)
- TensorRT optimization settings
- FP16 precision and Tensor Core utilization
- Memory management and fragmentation
- API health and performance
- Monitoring system integration (Prometheus, Grafana)

```bash
./run_automated_tests.py --host <hostname> [options]
```

### 2. Load Testing Tool (`load_test.py`)

A sophisticated load testing tool that measures system performance under various workloads:
- Simulates concurrent users with configurable patterns
- Tests multiple request types (queries, generation, embeddings)
- Measures throughput, latency, and error rates
- Monitors GPU utilization, memory, temperature, and power
- Generates detailed reports with visualizations

```bash
./load_test.py --host <hostname> [options]
```

### 3. Complete Test Suite (`run_t4_test_suite.py`)

A wrapper script that executes both tools and generates a consolidated report:
- Runs all automated tests
- Performs load testing with various user counts
- Generates a unified HTML report with key metrics
- Provides optimization recommendations

```bash
./run_t4_test_suite.py --host <hostname> [options]
```

## Usage Examples

### Basic Testing (Local Deployment)

```bash
# Run the complete test suite on localhost
./run_t4_test_suite.py

# Run only the automated tests
./run_automated_tests.py

# Run only load tests
./load_test.py
```

### Remote Testing (Brev Lab Instance)

```bash
# Test a remote Brev lab instance
./run_t4_test_suite.py --host jupyter0-ipzl7zn0p.brevlab.com --username ubuntu --ssh-key ~/.ssh/id_rsa

# Run automated tests on remote instance
./run_automated_tests.py --host jupyter0-ipzl7zn0p.brevlab.com --username ubuntu --ssh-key ~/.ssh/id_rsa

# Run load tests on remote instance
./load_test.py --host jupyter0-ipzl7zn0p.brevlab.com
```

### Custom Test Configurations

```bash
# Run tests with custom ports
./run_t4_test_suite.py --api-port 8080 --prometheus-port 9090 --grafana-port 3030

# Run load tests with specific user counts and duration
./load_test.py --users 1 10 50 100 --duration 120

# Skip specific test types
./run_t4_test_suite.py --skip-automated-tests
```

## Output and Reports

All test results are saved to the `t4-test-reports` directory by default (configurable with `--output-dir`):

- `automated-tests/` - Environment and optimization test reports
- `load-tests/` - Load test reports and graphs
- `t4-consolidated-report-TIMESTAMP.html` - Unified HTML report

Reports include:
- Detailed test results and metrics
- Performance graphs and visualizations
- GPU utilization and memory metrics
- Optimization recommendations

## Dependencies

These tools require:
- Python 3.8+
- Requests
- Matplotlib and NumPy (for visualization)
- SSH client (for remote testing)

## Adding to CI/CD Pipeline

To integrate these tests into your CI/CD pipeline, add a stage like:

```yaml
test_t4_deployment:
  stage: test
  script:
    - cd deployment/nvidia-t4
    - ./run_t4_test_suite.py --host $T4_HOST --username $T4_USERNAME --ssh-key $T4_SSH_KEY
  artifacts:
    paths:
      - deployment/nvidia-t4/t4-test-reports/
```

## Troubleshooting

If tests fail, check:
- Network connectivity to the T4 instance
- API availability on the specified port
- NVIDIA driver and Docker configuration
- SSH authentication for remote testing