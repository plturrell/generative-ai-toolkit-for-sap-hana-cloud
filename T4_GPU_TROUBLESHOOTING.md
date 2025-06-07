# NVIDIA T4 GPU Troubleshooting Guide

This document provides comprehensive troubleshooting steps for common issues encountered when deploying and running the SAP HANA Cloud Generative AI Toolkit on NVIDIA T4 GPUs.

## Diagnostic Tools

Before diving into specific issues, here are essential diagnostic tools for troubleshooting T4 GPU deployments:

### Basic GPU Information

```bash
# Check GPU availability and basic info
nvidia-smi

# Monitor GPU usage in real-time
watch -n 0.5 nvidia-smi

# Get detailed information about the GPU
nvidia-smi -q

# Check CUDA version
nvcc --version

# Verify TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

### Advanced Diagnostics

```bash
# Check for errors in GPU operation
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv

# Verify compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Check GPU power and thermal metrics
nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,fan.speed,power.draw,power.limit --format=csv
```

## Common Issues and Solutions

### 1. T4 GPU Not Detected

#### Symptoms:
- `nvidia-smi` shows no GPUs
- `torch.cuda.is_available()` returns `False`
- "No CUDA GPUs are available" errors

#### Solutions:

**Check PCIe connection:**
```bash
lspci | grep NVIDIA
```

**Verify driver installation:**
```bash
nvidia-smi
```

**Check driver compatibility:**
- T4 requires driver version >= 450.80.02
- Update drivers if needed:
```bash
sudo apt purge nvidia*
sudo apt install nvidia-driver-535  # or latest compatible driver
```

**Verify CUDA installation:**
```bash
nvcc --version
```

**Check Docker configuration (if using containers):**
- Ensure `--gpus all` flag is used
- Verify the container has GPU access:
```bash
docker run --gpus all --rm nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 2. CUDA Out of Memory Errors

#### Symptoms:
- "CUDA out of memory" error messages
- Process termination during model loading or inference
- Unexpected crashes during high batch size operations

#### Solutions:

**Check current memory usage:**
```bash
nvidia-smi
```

**Monitor memory during operations:**
```bash
watch -n 0.5 "nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv"
```

**Memory optimization strategies:**

1. **Reduce batch size:**
```python
# Use T4MemoryManager to calculate optimal batch size
from api.t4_gpu_optimizer import T4MemoryManager
memory_manager = T4MemoryManager()
batch_size = memory_manager.calculate_optimal_batch_size(
    input_size_per_sample,
    output_size_per_sample
)
```

2. **Enable model quantization:**
```python
# Configure environment for INT8 or FP16
os.environ["T4_GPU_INT8_MODE"] = "true"
os.environ["T4_GPU_FP16_MODE"] = "true"
```

3. **Clear cache if using PyTorch:**
```python
import torch
torch.cuda.empty_cache()
```

4. **Use lighter model variants**
   - Switch from larger to smaller models (e.g., Llama-2-13B to Llama-2-7B)
   - Use quantized models (GPTQ, AWQ)

5. **Implement gradient checkpointing for training:**
```python
model.gradient_checkpointing_enable()
```

### 3. Performance Lower Than Expected

#### Symptoms:
- Inference is slower than benchmarks
- No observable speedup from T4 optimizations
- High latency despite low GPU utilization

#### Solutions:

**Check GPU utilization patterns:**
```bash
nvidia-smi dmon -s u
```

**Verify Tensor Core usage:**
```python
# Enable Tensor Cores explicitly
os.environ["T4_GPU_ENABLE_TENSOR_CORES"] = "true"
```

**Check precision configuration:**
```python
# For best performance on T4, ensure FP16 is enabled
t4_config = T4GPUConfig(
    fp16_mode=True,
    int8_mode=True if calibrated else False,
    precision="fp16"
)
```

**Enable TensorRT optimization:**
```python
# Configure TensorRT optimization
optimizer = T4TensorRTOptimizer(t4_config)
engine = optimizer.optimize_embedding_model(
    model, 
    model_name, 
    max_sequence_length, 
    embedding_dim
)
```

**Increase TensorRT workspace:**
```python
# Increase workspace for better optimization
t4_config = T4GPUConfig(
    max_workspace_size=6 * (1 << 30),  # 6GB
    optimization_level=4
)
```

**Profile the model execution:**
```python
# Use PyTorch profiler to identify bottlenecks
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(inputs)
    
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 4. TensorRT Optimization Failures

#### Symptoms:
- "Failed to optimize model with TensorRT" errors
- "Layer not supported" messages
- Long optimization times followed by failures

#### Solutions:

**Check TensorRT compatibility:**
```python
import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")
# Should be 8.5+ for best compatibility
```

**Simplify model for initial testing:**
- Try optimizing a simpler model first
- Verify individual layers are supported

**Check for dynamic shapes issues:**
```python
# Try with fixed shapes first
input_shapes = {
    "input_ids": [1, 512],
    "attention_mask": [1, 512]
}
```

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tensorrt")
logger.setLevel(logging.DEBUG)
```

**Try different optimization levels:**
```python
# Start with lower optimization level
t4_config = T4GPUConfig(optimization_level=2)
```

### 5. INT8 Calibration Issues

#### Symptoms:
- Poor accuracy after INT8 calibration
- "Failed to build INT8 engine" errors
- Calibration process hanging or crashing

#### Solutions:

**Check calibration dataset:**
- Ensure calibration data is representative of real inputs
- Use at least 100-500 diverse samples

**Try different calibration algorithms:**
```python
# Test different calibration methods
calibration_algorithms = ["entropy", "minmax", "percentile"]
for algorithm in calibration_algorithms:
    try:
        engine = optimizer.optimize_model_with_calibration(
            model,
            "model_name",
            calibration_data,
            calibration_algorithm=algorithm
        )
        if engine:
            print(f"Successful with {algorithm}")
            break
    except Exception as e:
        print(f"Failed with {algorithm}: {e}")
```

**Implement mixed precision calibration:**
```python
# Keep sensitive layers in higher precision
t4_config = T4GPUConfig(
    int8_mode=True,
    fp16_mode=True,
    strict_type_constraints=False
)
```

**Cache calibration results:**
```python
# Use calibration cache
os.environ["T4_GPU_CALIBRATION_CACHE"] = "/path/to/calibration.cache"
```

### 6. Container and Deployment Issues

#### Symptoms:
- Container starts but GPU is not accessible
- Resource allocation errors in container environment
- Docker-related GPU permission issues

#### Solutions:

**Check Docker GPU runtime:**
```bash
# Verify NVIDIA container toolkit installation
dpkg -l | grep nvidia-container-toolkit

# Check if runtime is properly configured
sudo cat /etc/docker/daemon.json | grep nvidia
```

**Verify container GPU access:**
```bash
docker run --gpus all --rm nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Proper resource allocation:**
```yaml
# Sample Kubernetes configuration with proper resource requests
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 32Gi
    cpu: 8
  requests:
    nvidia.com/gpu: 1
    memory: 16Gi
    cpu: 4
```

**Use non-root user with proper permissions:**
```dockerfile
# In Dockerfile
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser

# Add user to video group for GPU access
RUN usermod -a -G video appuser

# Set permissions
RUN chown -R appuser:appgroup /app
USER appuser
```

### 7. Monitoring and Observability Issues

#### Symptoms:
- Missing GPU metrics in Prometheus
- Incomplete dashboard visualization
- Unable to track GPU performance in production

#### Solutions:

**Verify DCGM exporter is running:**
```bash
# Check if DCGM exporter container is running
docker ps | grep dcgm-exporter
```

**Test metrics endpoint:**
```bash
curl http://localhost:9400/metrics | grep gpu
```

**Configure Prometheus scraping:**
```yaml
# Ensure this is in prometheus.yml
scrape_configs:
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['api:9400']
```

**Install NVIDIA Container Toolkit with monitoring support:**
```bash
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2 nvidia-container-runtime
```

**Enable additional metrics collection:**
```bash
# Run DCGM exporter with expanded metrics
docker run -d --gpus all --restart always \
  --name dcgm-exporter \
  -p 9400:9400 \
  nvidia/dcgm-exporter:latest \
  -f /etc/dcgm-exporter/dcp-metrics-included.csv
```

## T4-Specific Errors and Solutions

### 1. Turing Architecture-Specific Errors

#### Symptoms:
- "Unsupported CUDA architecture 'sm_75'" errors
- "Cannot find code for architecture sm_75" messages
- CUDA kernel launch failures

#### Solutions:

**Verify CUDA compilation flags:**
```bash
# When building custom CUDA extensions, include sm_75 architecture
TORCH_CUDA_ARCH_LIST="7.5" pip install -e .
```

**Check for Turing-compatible libraries:**
```bash
# For PyTorch, ensure CUDA 10.2+ compatibility
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

**Enable Turing-specific optimizations:**
```python
# In code, detect and enable Turing-specific paths
device_properties = torch.cuda.get_device_properties(0)
is_turing = device_properties.major == 7 and device_properties.minor >= 5
if is_turing:
    # Enable Turing-specific optimizations
    enable_turing_optimizations()
```

### 2. Memory Bandwidth Limitations

#### Symptoms:
- Performance plateaus despite low utilization
- High memory transfer latency
- Poor scaling with larger batch sizes

#### Solutions:

**Optimize memory access patterns:**
```python
# Use contiguous tensors
tensor = tensor.contiguous()

# Pin memory for faster host-device transfers
tensor = tensor.pin_memory()
```

**Minimize host-device transfers:**
```python
# Keep data on GPU when possible
def process_batch(batch):
    # Move to GPU once
    batch = {k: v.to('cuda') for k, v in batch.items()}
    
    # Process on GPU
    results = model(batch)
    
    # Only move results back to CPU when needed
    return {k: v.cpu() for k, v in results.items()}
```

**Use CUDA Graphs for repeated operations:**
```python
# Capture computation graph for repeated inference
static_input = {k: torch.ones_like(v).to('cuda') for k, v in sample_input.items()}
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay graph with new inputs
def optimized_inference(inputs):
    for k, v in inputs.items():
        static_input[k].copy_(v)
    g.replay()
    return static_output
```

## Advanced Troubleshooting Techniques

### 1. CUDA Error Decoding

When encountering CUDA errors with numeric codes, use this reference:

| Error Code | Description | Typical Causes |
|------------|-------------|----------------|
| 2 | Out of memory | Memory allocation failure, fragmentation |
| 4 | Launch failure | Invalid parameters, kernel configuration |
| 7 | Invalid device function | Unsupported GPU architecture, driver mismatch |
| 8 | Invalid configuration | Invalid launch configuration |
| 11 | Invalid value | API received invalid parameters |
| 77 | Illegal address | Memory access violation |

### 2. TensorRT Layer Compatibility

For TensorRT optimization issues, check layer support:

```python
# Function to check TensorRT layer support
def check_layer_support(model, sample_input):
    import torch_tensorrt
    
    # Try to compile with diagnostics
    try:
        torch_tensorrt.compile(
            model,
            inputs=[sample_input],
            enabled_precisions={torch.float16, torch.int8},
            diagnostics=True
        )
    except Exception as e:
        print(f"Layer support issue: {e}")
        # Extract unsupported layers from error message
```

### 3. Performance Regression Testing

To systematically identify performance regressions:

```bash
# Run automated performance testing
cd deployment/nvidia-t4
python run_t4_test_suite.py --compare-baseline
```

This will execute a series of benchmarks and compare against baseline performance metrics to identify any regressions.

## Known Issues and Limitations

1. **Quantization Accuracy**: T4 INT8 quantization can lead to accuracy issues with text generation models; prefer FP16 for generative tasks.

2. **Memory Fragmentation**: Long-running T4 inference servers can experience memory fragmentation; consider periodic service restarts.

3. **TensorRT Compatibility**: Some custom PyTorch operators are not supported by TensorRT; simplify model architecture for best compatibility.

4. **Maximum Model Size**: T4's 16GB memory limits model size; use quantization for models larger than 7B parameters.

5. **Thermal Throttling**: In poorly ventilated environments, T4 GPUs may experience thermal throttling; ensure proper airflow.

## Getting Additional Help

If you encounter persistent issues not covered in this guide:

1. **Check NVIDIA Developer Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

2. **SAP HANA Cloud Support**: Open a ticket with SAP support for HANA Cloud integration issues

3. **GitHub Issues**: Check for similar issues in the GitHub repository

4. **T4 GPU Slack Channel**: Join #t4-gpu-optimization in the project Slack workspace

5. **Documentation**: Refer to [T4 GPU Optimization Guide](./T4_GPU_OPTIMIZATION_GUIDE.md) for detailed optimization strategies