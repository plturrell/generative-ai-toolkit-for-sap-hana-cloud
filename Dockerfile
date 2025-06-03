FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt
RUN pip install prometheus-client==0.16.0 python-jose[cryptography]==3.3.0 

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
ENV AUTH_REQUIRED=true
ENV ENFORCE_HTTPS=true
ENV PROMETHEUS_ENABLED=true
ENV PROMETHEUS_PORT=9090
ENV CONNECTION_POOL_SIZE=10

# Expose ports for API and metrics
EXPOSE 8000
EXPOSE 9090

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run server
CMD ["uvicorn", "hana_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]