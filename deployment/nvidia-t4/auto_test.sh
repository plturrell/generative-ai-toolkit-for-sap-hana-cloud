#!/bin/bash
# Automated T4 Testing Script
# Usage: ./auto_test.sh [host] [api_port]
set -e

# Default values
HOST=${1:-"localhost"}
API_PORT=${2:-8000}
PROMETHEUS_PORT=9091
GRAFANA_PORT=3000
GPU_METRICS_PORT=9835
OUTPUT_DIR="t4-test-results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/t4-report-${TIMESTAMP}.html"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "====== NVIDIA T4 Automated Test ======"
echo "Host: $HOST"
echo "API Port: $API_PORT"
echo "Output: $REPORT_FILE"
echo "====================================="

# Function to check API health
check_api() {
  echo "Testing API health..."
  if curl -s -f -m 5 "http://${HOST}:${API_PORT}/health" > /dev/null; then
    echo "✓ API is healthy"
    API_STATUS="OK"
  else
    echo "✗ API health check failed"
    API_STATUS="FAIL"
  fi
}

# Function to check GPU info endpoint
check_gpu_info() {
  echo "Testing GPU info endpoint..."
  GPU_INFO=$(curl -s -f -m 5 "http://${HOST}:${API_PORT}/api/v1/system/gpu" 2>/dev/null || echo '{"error":"Failed to connect"}')
  
  if [[ $GPU_INFO == *"error"* ]]; then
    echo "✗ GPU info check failed"
    GPU_DETECTED="UNKNOWN"
    GPU_MODEL="UNKNOWN"
    TENSOR_CORES="UNKNOWN"
  else
    echo "✓ GPU info endpoint accessible"
    # Extract GPU details
    GPU_DETECTED=$(echo $GPU_INFO | grep -o '"detected":[^,}]*' | cut -d':' -f2)
    GPU_MODEL=$(echo $GPU_INFO | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    TENSOR_CORES=$(echo $GPU_INFO | grep -o '"tensor_cores":[^,}]*' | cut -d':' -f2)
    
    if [[ $GPU_MODEL == *"T4"* ]]; then
      echo "✓ NVIDIA T4 GPU detected"
      IS_T4="YES"
    else
      echo "✗ Not a T4 GPU: $GPU_MODEL"
      IS_T4="NO"
    fi
    
    if [[ $TENSOR_CORES == "true" ]]; then
      echo "✓ Tensor Cores available"
    else
      echo "✗ Tensor Cores not detected"
    fi
  fi
}

# Function to check TensorRT
check_tensorrt() {
  echo "Testing TensorRT configuration..."
  TENSORRT_INFO=$(curl -s -f -m 5 "http://${HOST}:${API_PORT}/api/v1/system/info" 2>/dev/null | grep -o '"tensorrt":{[^}]*}' || echo '{"error":"Failed"}')
  
  if [[ $TENSORRT_INFO == *"error"* ]]; then
    echo "✗ TensorRT info check failed"
    TENSORRT_ENABLED="UNKNOWN"
    TENSORRT_PRECISION="UNKNOWN"
  else
    # Extract TensorRT details
    TENSORRT_ENABLED=$(echo $TENSORRT_INFO | grep -o '"enabled":[^,}]*' | cut -d':' -f2)
    TENSORRT_PRECISION=$(echo $TENSORRT_INFO | grep -o '"precision":"[^"]*"' | cut -d'"' -f4)
    
    if [[ $TENSORRT_ENABLED == "true" ]]; then
      echo "✓ TensorRT is enabled"
      
      if [[ $TENSORRT_PRECISION == "fp16" ]]; then
        echo "✓ TensorRT using FP16 precision (optimal for T4)"
      else
        echo "✗ TensorRT not using FP16 precision: $TENSORRT_PRECISION"
      fi
    else
      echo "✗ TensorRT is not enabled"
    fi
  fi
}

# Function to check Prometheus
check_prometheus() {
  echo "Testing Prometheus..."
  if curl -s -f -m 5 "http://${HOST}:${PROMETHEUS_PORT}/-/healthy" > /dev/null; then
    echo "✓ Prometheus is healthy"
    PROMETHEUS_STATUS="OK"
    
    # Check for GPU metrics
    QUERY_RESULT=$(curl -s -f -m 5 "http://${HOST}:${PROMETHEUS_PORT}/api/v1/query?query=nvidia_gpu_duty_cycle" 2>/dev/null || echo '{"error":"Failed"}')
    
    if [[ $QUERY_RESULT == *'"result":['* ]]; then
      echo "✓ GPU metrics found in Prometheus"
      GPU_METRICS_STATUS="OK"
    else
      echo "✗ No GPU metrics in Prometheus"
      GPU_METRICS_STATUS="FAIL"
    fi
  else
    echo "✗ Prometheus health check failed"
    PROMETHEUS_STATUS="FAIL"
    GPU_METRICS_STATUS="UNKNOWN"
  fi
}

# Function to check Grafana
check_grafana() {
  echo "Testing Grafana..."
  if curl -s -f -m 5 "http://${HOST}:${GRAFANA_PORT}/api/health" > /dev/null; then
    echo "✓ Grafana is healthy"
    GRAFANA_STATUS="OK"
    
    # Check for dashboards
    DASHBOARD_RESULT=$(curl -s -f -m 5 "http://${HOST}:${GRAFANA_PORT}/api/search" 2>/dev/null || echo '{"error":"Failed"}')
    
    if [[ $DASHBOARD_RESULT == *'"title":'* ]]; then
      echo "✓ Grafana dashboards available"
      
      if [[ $DASHBOARD_RESULT == *'"title":"GPU'* || $DASHBOARD_RESULT == *'"title":"NVIDIA'* || $DASHBOARD_RESULT == *'"title":"T4'* ]]; then
        echo "✓ GPU dashboard found"
        GPU_DASHBOARD="YES"
      else
        echo "✗ No GPU dashboard found"
        GPU_DASHBOARD="NO"
      fi
    else
      echo "✗ No dashboards found in Grafana"
      GPU_DASHBOARD="UNKNOWN"
    fi
  else
    echo "✗ Grafana health check failed"
    GRAFANA_STATUS="FAIL"
    GPU_DASHBOARD="UNKNOWN"
  fi
}

# Function to check GPU metrics exporter
check_gpu_metrics_exporter() {
  echo "Testing GPU metrics exporter..."
  METRICS=$(curl -s -f -m 5 "http://${HOST}:${GPU_METRICS_PORT}/metrics" 2>/dev/null || echo "Failed")
  
  if [[ $METRICS == *"nvidia_gpu"* ]]; then
    echo "✓ GPU metrics exporter is working"
    
    # Extract utilization
    UTILIZATION=$(echo "$METRICS" | grep 'nvidia_gpu_duty_cycle' | head -1 | awk '{print $2}')
    MEMORY_USED=$(echo "$METRICS" | grep 'nvidia_gpu_memory_used_bytes' | head -1 | awk '{print $2}')
    MEMORY_USED_MB=$(echo "scale=2; $MEMORY_USED / 1024 / 1024" | bc)
    TEMPERATURE=$(echo "$METRICS" | grep 'nvidia_gpu_temperature_celsius' | head -1 | awk '{print $2}')
    POWER=$(echo "$METRICS" | grep 'nvidia_gpu_power_draw_watts' | head -1 | awk '{print $2}')
    
    echo "  GPU Utilization: ${UTILIZATION}%"
    echo "  Memory Used: ${MEMORY_USED_MB}MB"
    echo "  Temperature: ${TEMPERATURE}°C"
    echo "  Power: ${POWER}W"
    
    GPU_EXPORTER_STATUS="OK"
  else
    echo "✗ GPU metrics exporter check failed"
    GPU_EXPORTER_STATUS="FAIL"
    UTILIZATION="UNKNOWN"
    MEMORY_USED_MB="UNKNOWN"
    TEMPERATURE="UNKNOWN"
    POWER="UNKNOWN"
  fi
}

# Perform all checks
check_api
check_gpu_info
check_tensorrt
check_prometheus
check_grafana
check_gpu_metrics_exporter

# Determine overall status
if [[ $API_STATUS == "OK" && $IS_T4 == "YES" && $TENSORRT_ENABLED == "true" && $TENSORRT_PRECISION == "fp16" ]]; then
  OVERALL_STATUS="PASS"
  OVERALL_MESSAGE="T4 GPU is properly configured with optimized settings"
elif [[ $API_STATUS == "OK" && $IS_T4 == "YES" ]]; then
  OVERALL_STATUS="PARTIAL"
  OVERALL_MESSAGE="T4 GPU detected but some optimizations may be missing"
else
  OVERALL_STATUS="FAIL"
  OVERALL_MESSAGE="Critical tests failed - see detailed results"
fi

# Generate HTML report
cat > $REPORT_FILE << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA T4 Test Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #76b900;
        }
        h1 {
            margin: 0;
            color: #76b900;
        }
        .summary {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .summary-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            flex-basis: 48%;
            margin-bottom: 15px;
            text-align: center;
        }
        .summary-box h2 {
            margin-top: 0;
            font-size: 16px;
        }
        .summary-box .value {
            font-size: 24px;
            font-weight: bold;
        }
        .pass {
            color: #28a745;
        }
        .partial {
            color: #ffc107;
        }
        .fail {
            color: #dc3545;
        }
        .unknown {
            color: #6c757d;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        table th, table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        table th {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NVIDIA T4 Automated Test Report</h1>
            <p>Host: ${HOST} | Date: $(date '+%Y-%m-%d %H:%M:%S')</p>
        </header>
        
        <div class="summary">
            <div class="summary-box">
                <h2>Overall Status</h2>
                <div class="value">${OVERALL_STATUS}</div>
            </div>
            <div class="summary-box">
                <h2>Summary</h2>
                <div>${OVERALL_MESSAGE}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>GPU Details</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Result</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td>GPU Detection</td>
                    <td>${IS_T4}</td>
                    <td>${GPU_MODEL}</td>
                </tr>
                <tr>
                    <td>TensorRT Enabled</td>
                    <td>${TENSORRT_ENABLED}</td>
                    <td>Precision: ${TENSORRT_PRECISION}</td>
                </tr>
                <tr>
                    <td>GPU Utilization</td>
                    <td>${UTILIZATION}%</td>
                    <td>Memory: ${MEMORY_USED_MB}MB | Temp: ${TEMPERATURE}°C | Power: ${POWER}W</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Monitoring</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td>API</td>
                    <td>${API_STATUS}</td>
                    <td>Health endpoint: http://${HOST}:${API_PORT}/health</td>
                </tr>
                <tr>
                    <td>Prometheus</td>
                    <td>${PROMETHEUS_STATUS}</td>
                    <td>GPU Metrics: ${GPU_METRICS_STATUS}</td>
                </tr>
                <tr>
                    <td>Grafana</td>
                    <td>${GRAFANA_STATUS}</td>
                    <td>GPU Dashboard: ${GPU_DASHBOARD}</td>
                </tr>
                <tr>
                    <td>GPU Metrics Exporter</td>
                    <td>${GPU_EXPORTER_STATUS}</td>
                    <td>Endpoint: http://${HOST}:${GPU_METRICS_PORT}/metrics</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
EOF

# Add recommendations based on test results
if [[ $IS_T4 != "YES" ]]; then
  echo "<li>Verify this is a T4 GPU instance</li>" >> $REPORT_FILE
fi

if [[ $TENSORRT_ENABLED != "true" ]]; then
  echo "<li>Enable TensorRT for improved performance</li>" >> $REPORT_FILE
fi

if [[ $TENSORRT_PRECISION != "fp16" ]]; then
  echo "<li>Configure TensorRT to use FP16 precision for optimal T4 performance</li>" >> $REPORT_FILE
fi

if [[ $PROMETHEUS_STATUS != "OK" || $GPU_METRICS_STATUS != "OK" ]]; then
  echo "<li>Check Prometheus configuration for GPU metrics collection</li>" >> $REPORT_FILE
fi

if [[ $GRAFANA_STATUS != "OK" || $GPU_DASHBOARD != "YES" ]]; then
  echo "<li>Configure Grafana with T4 GPU dashboards</li>" >> $REPORT_FILE
fi

if [[ $GPU_EXPORTER_STATUS != "OK" ]]; then
  echo "<li>Check NVIDIA GPU metrics exporter service</li>" >> $REPORT_FILE
fi

if [[ $OVERALL_STATUS == "PASS" ]]; then
  echo "<li>All tests passed! No recommendations needed.</li>" >> $REPORT_FILE
fi

# Finish HTML
cat >> $REPORT_FILE << EOF
            </ul>
        </div>
    </div>
</body>
</html>
EOF

echo "====== Test Completed ======"
echo "Overall Status: $OVERALL_STATUS"
echo "Report saved to: $REPORT_FILE"
echo "==========================="

# Open report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  open $REPORT_FILE
fi

# Exit with appropriate code
if [[ $OVERALL_STATUS == "PASS" ]]; then
  exit 0
elif [[ $OVERALL_STATUS == "PARTIAL" ]]; then
  exit 1
else
  exit 2
fi