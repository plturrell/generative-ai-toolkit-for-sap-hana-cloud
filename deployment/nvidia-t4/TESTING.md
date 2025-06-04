# NVIDIA T4 Deployment Testing Guide

This guide explains how to use the automated testing tools for validating NVIDIA T4 GPU deployments.

## Overview

The testing framework validates the following aspects of the deployment:

1. **GPU Detection**: Verifies NVIDIA T4 GPU availability and capabilities
2. **Docker Setup**: Checks Docker with NVIDIA runtime configuration
3. **API Health**: Tests the health endpoint of the deployed API
4. **Monitoring**: Validates Prometheus, Grafana, and GPU metrics exporter
5. **GPU Utilization**: Checks if GPU metrics are being properly reported

## Testing Tools

### 1. Automated Test Script

The `test-deployment.py` script provides comprehensive validation of all deployment components.

```bash
./test-deployment.py --host [hostname] --username [user] --ssh-key [path/to/key]
```

#### Options:

- `--host`: Hostname or IP of the NVIDIA instance (default: localhost)
- `--api-port`: Port for the API service (default: 8000)
- `--prometheus-port`: Port for Prometheus (default: 9091)
- `--grafana-port`: Port for Grafana (default: 3000)
- `--gpu-metrics-port`: Port for NVIDIA GPU metrics exporter (default: 9835)
- `--ssh-key`: Path to SSH key for remote testing
- `--username`: Username for SSH connection
- `--output`: Path to save test results JSON
- `--test`: Specific test to run (choices: all, gpu, docker, api, monitoring, utilization)

### 2. Brev Lab Testing Script

For testing on Brev Lab instances (like jupyter0-ipzl7zn0p.brevlab.com), use the `test-brevlab.sh` script:

```bash
./test-brevlab.sh [username] [ssh_key_path]
```

This script automatically sets the right host for Brev Lab and runs the complete test suite.

## Running Tests

### Local Testing

If testing on the same machine where the deployment is running:

```bash
./test-deployment.py
```

### Remote Testing

For testing a remote deployment:

```bash
./test-deployment.py --host jupyter0-ipzl7zn0p.brevlab.com --username ubuntu --ssh-key ~/.ssh/id_rsa
```

### Individual Component Testing

Test specific components:

```bash
# Test only GPU detection
./test-deployment.py --test gpu

# Test only API health
./test-deployment.py --test api

# Test only monitoring setup
./test-deployment.py --test monitoring
```

## Test Results

The test script provides:

1. A detailed console report with status of each component
2. JSON output file (if `--output` is specified) for programmatic analysis
3. Exit code based on test success (0 for success, non-zero for failures)

## Continuous Integration

Add this test to your CI pipeline:

```yaml
test_deployment:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Test NVIDIA Deployment
      run: |
        cd deployment/nvidia-t4
        ./test-deployment.py --host ${{ secrets.NVIDIA_HOST }} --username ${{ secrets.NVIDIA_USERNAME }} --ssh-key ${{ secrets.NVIDIA_SSH_KEY_PATH }} --output test-results.json
    - name: Upload Test Results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: deployment/nvidia-t4/test-results.json
```

## Troubleshooting

If tests fail, check:

1. **GPU Detection Failure**: Ensure NVIDIA drivers are installed correctly
2. **Docker Setup Failure**: Verify NVIDIA Container Toolkit installation
3. **API Health Failure**: Check if the API service is running on the specified port
4. **Monitoring Failure**: Verify Prometheus and Grafana container status
5. **GPU Utilization Failure**: Check NVIDIA SMI exporter configuration

Run `docker-compose ps` to check the status of all containers in the deployment.