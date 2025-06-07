#!/bin/bash
# Comprehensive health check script for T4 GPU deployment

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting comprehensive health check for T4 GPU deployment...${NC}"

# Function to check HTTP endpoint
check_endpoint() {
  local name=$1
  local url=$2
  local expected_status=$3
  
  echo -n "Checking $name endpoint ($url)... "
  
  # Try up to 3 times with increasing delay
  for i in {1..3}; do
    status=$(curl -s -o /dev/null -w "%{http_code}" $url || echo "Failed")
    
    if [ "$status" == "$expected_status" ]; then
      echo -e "${GREEN}OK ($status)${NC}"
      return 0
    else
      if [ $i -lt 3 ]; then
        echo -n "Retrying in ${i}s... "
        sleep $i
      fi
    fi
  done
  
  echo -e "${RED}FAILED (Got: $status, Expected: $expected_status)${NC}"
  return 1
}

# Function to check if container is running and healthy
check_container() {
  local name=$1
  
  echo -n "Checking $name container... "
  
  # Check if container exists and is running
  if docker-compose ps | grep $name | grep -q "Up"; then
    # Check health status if container has healthcheck
    health=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q $name) 2>/dev/null || echo "NoHealthCheck")
    
    if [ "$health" == "healthy" ] || [ "$health" == "NoHealthCheck" ]; then
      echo -e "${GREEN}RUNNING${NC}"
      return 0
    else
      echo -e "${RED}UNHEALTHY ($health)${NC}"
      return 1
    fi
  else
    echo -e "${RED}NOT RUNNING${NC}"
    return 1
  fi
}

# Function to check GPU availability
check_gpu() {
  echo -n "Checking NVIDIA GPU availability... "
  
  if command -v nvidia-smi &>/dev/null; then
    # Check if nvidia-smi returns without error
    if nvidia-smi &>/dev/null; then
      # Get GPU info
      gpu_info=$(nvidia-smi --query-gpu=gpu_name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader)
      echo -e "${GREEN}AVAILABLE${NC}"
      echo "  $gpu_info"
      return 0
    else
      echo -e "${RED}ERROR: nvidia-smi command failed${NC}"
      return 1
    fi
  else
    echo -e "${RED}NOT AVAILABLE: nvidia-smi command not found${NC}"
    return 1
  fi
}

# Function to check logs for errors
check_logs() {
  local container=$1
  local error_pattern=$2
  local lines=${3:-100}
  
  echo -n "Checking $container logs for errors... "
  
  # Get logs and check for error pattern
  log_errors=$(docker-compose logs --tail=$lines $container | grep -i "$error_pattern" | wc -l)
  
  if [ $log_errors -gt 0 ]; then
    echo -e "${RED}FOUND $log_errors ERROR(S)${NC}"
    echo "Log sample:"
    docker-compose logs --tail=$lines $container | grep -i "$error_pattern" | head -n 5
    return 1
  else
    echo -e "${GREEN}NO ERRORS${NC}"
    return 0
  fi
}

# Function to check TensorRT optimization
check_tensorrt() {
  echo -n "Checking TensorRT optimization... "
  
  # Request TensorRT status from API
  tensorrt_status=$(curl -s http://localhost:8000/v1/status | grep -o '"tensorrt_enabled":[^,}]*' || echo "Not found")
  
  if [[ $tensorrt_status == *"true"* ]]; then
    echo -e "${GREEN}ENABLED${NC}"
    return 0
  else
    echo -e "${YELLOW}NOT ENABLED ($tensorrt_status)${NC}"
    return 1
  fi
}

# Function to check auto-tuning status
check_autotuning() {
  echo -n "Checking auto-tuning status... "
  
  # Request auto-tuning status from API
  autotuning_status=$(curl -s http://localhost:8000/v1/status | grep -o '"autotuning_enabled":[^,}]*' || echo "Not found")
  
  if [[ $autotuning_status == *"true"* ]]; then
    echo -e "${GREEN}ENABLED${NC}"
    return 0
  else
    echo -e "${YELLOW}NOT ENABLED ($autotuning_status)${NC}"
    return 1
  fi
}

# Check containers
echo "Checking container status..."
check_container "api" || FAILURES=$((FAILURES+1))
check_container "prometheus" || FAILURES=$((FAILURES+1))
check_container "grafana" || FAILURES=$((FAILURES+1))
check_container "dcgm-exporter" || FAILURES=$((FAILURES+1))
check_container "alertmanager" || FAILURES=$((FAILURES+1))
check_container "threshold-adapter" || FAILURES=$((FAILURES+1))
check_container "benchmark-collector" || FAILURES=$((FAILURES+1))
echo

# Check GPU
echo "Checking GPU..."
check_gpu || FAILURES=$((FAILURES+1))
echo

# Check endpoints
echo "Checking endpoints..."
check_endpoint "API health" "http://localhost:8000/health" "200" || FAILURES=$((FAILURES+1))
check_endpoint "API metrics" "http://localhost:9090/metrics" "200" || FAILURES=$((FAILURES+1))
check_endpoint "Prometheus" "http://localhost:9091/-/healthy" "200" || FAILURES=$((FAILURES+1))
check_endpoint "Grafana" "http://localhost:3000/api/health" "200" || FAILURES=$((FAILURES+1))
check_endpoint "DCGM Exporter metrics" "http://localhost:9400/metrics" "200" || FAILURES=$((FAILURES+1))
check_endpoint "Alertmanager" "http://localhost:9093/-/healthy" "200" || FAILURES=$((FAILURES+1))
echo

# Check logs for errors
echo "Checking container logs for errors..."
check_logs "api" "error|exception|fatal" 200 || FAILURES=$((FAILURES+1))
check_logs "prometheus" "error|fatal" || FAILURES=$((FAILURES+1))
check_logs "grafana" "error|fatal" || FAILURES=$((FAILURES+1))
check_logs "dcgm-exporter" "error|fatal" || FAILURES=$((FAILURES+1))
check_logs "alertmanager" "error|fatal" || FAILURES=$((FAILURES+1))
check_logs "threshold-adapter" "error|exception|fatal" || FAILURES=$((FAILURES+1))
check_logs "benchmark-collector" "error|exception|fatal" || FAILURES=$((FAILURES+1))
echo

# Check TensorRT and auto-tuning
echo "Checking optimizations..."
check_tensorrt || WARNINGS=$((WARNINGS+1))
check_autotuning || WARNINGS=$((WARNINGS+1))
echo

# Summary
echo -e "${YELLOW}Health check summary:${NC}"
if [ "$FAILURES" -gt 0 ]; then
  echo -e "${RED}$FAILURES critical issues found!${NC}"
  exit 1
elif [ "$WARNINGS" -gt 0 ]; then
  echo -e "${YELLOW}$WARNINGS warnings found, but all critical checks passed.${NC}"
  exit 0
else
  echo -e "${GREEN}All checks passed successfully!${NC}"
  exit 0
fi