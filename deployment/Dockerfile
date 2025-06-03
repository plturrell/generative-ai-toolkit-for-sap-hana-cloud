# Use NVIDIA CUDA base image for GPU acceleration
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up a non-root user for better security
RUN groupadd -r hanaaiuser && useradd -r -g hanaaiuser hanaaiuser
RUN chown -R hanaaiuser:hanaaiuser /app
USER hanaaiuser

# Set proper environment variables for GPU acceleration
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/ || exit 1

# Run the application
CMD ["python3", "-m", "hana_ai.api"]

# Expose API port
EXPOSE ${API_PORT:-8000}