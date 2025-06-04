# Use NVIDIA CUDA base image for GPU acceleration
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements-api.txt

# Install TensorRT for better NVIDIA T4 performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-libnvinfer \
    python3-libnvinfer-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Set up a non-root user for better security
RUN groupadd -r hanaaiuser && useradd -r -g hanaaiuser hanaaiuser
RUN chown -R hanaaiuser:hanaaiuser /app
USER hanaaiuser

# Set proper environment variables for T4 GPU optimization
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.0"
ENV CUDA_MEMORY_FRACTION=0.8

# T4-specific optimizations
ENV NVIDIA_TF32_OVERRIDE=0
ENV NVIDIA_CUDA_CACHE_MAXSIZE=2147483648
ENV NVIDIA_CUDA_CACHE_PATH=/tmp/cuda-cache

# Optimizations for Turing architecture (T4)
ENV ENABLE_TENSOR_CORES=1
ENV ENABLE_FLASH_ATTENTION=1
ENV ENABLE_KERNEL_FUSION=1
ENV ENABLE_TENSORRT=1
ENV TENSORRT_PRECISION=fp16
ENV CHECKPOINT_ACTIVATIONS=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/ || exit 1

# Run the application
CMD ["python3", "-m", "hana_ai.api"]

# Expose API port
EXPOSE ${API_PORT:-8000}