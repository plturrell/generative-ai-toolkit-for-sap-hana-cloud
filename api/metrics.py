from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import time
import platform
import socket
import psutil

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """
        Handle GET requests to the metrics endpoint.
        
        Returns Prometheus-compatible metrics for monitoring systems.
        """
        try:
            # Check if metrics are enabled
            if os.environ.get("PROMETHEUS_ENABLED", "").lower() != "true":
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Metrics endpoint is disabled")
                return
            
            # Check authorization
            # In a production environment, restrict access to internal networks only
            client_ip = self.client_address[0]
            if client_ip not in ['127.0.0.1', 'localhost']:
                # Get X-Forwarded-For header if available
                forwarded_for = self.headers.get('X-Forwarded-For', '')
                if forwarded_for:
                    client_ip = forwarded_for.split(',')[0].strip()
                
                # Add additional checks for internal networks if needed
                # For simplicity, we'll allow all IPs in this example
            
            # Set response headers
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            
            # Generate metrics
            metrics = []
            
            # Add service info metric
            metrics.append('# HELP service_info Service information')
            metrics.append('# TYPE service_info gauge')
            metrics.append(f'service_info{{name="hana_ai_toolkit",version="1.0.0",environment="{os.environ.get("ENVIRONMENT", "production")}"}} 1')
            
            # Add system metrics
            metrics.append('# HELP system_cpu_usage CPU usage percentage')
            metrics.append('# TYPE system_cpu_usage gauge')
            metrics.append(f'system_cpu_usage {psutil.cpu_percent(interval=None)}')
            
            metrics.append('# HELP system_memory_usage Memory usage in bytes')
            metrics.append('# TYPE system_memory_usage gauge')
            memory = psutil.virtual_memory()
            metrics.append(f'system_memory_usage{{type="total"}} {memory.total}')
            metrics.append(f'system_memory_usage{{type="used"}} {memory.used}')
            metrics.append(f'system_memory_usage{{type="available"}} {memory.available}')
            
            metrics.append('# HELP system_disk_usage Disk usage in bytes')
            metrics.append('# TYPE system_disk_usage gauge')
            disk = psutil.disk_usage('/')
            metrics.append(f'system_disk_usage{{type="total"}} {disk.total}')
            metrics.append(f'system_disk_usage{{type="used"}} {disk.used}')
            metrics.append(f'system_disk_usage{{type="free"}} {disk.free}')
            
            # Add uptime metric
            metrics.append('# HELP system_uptime System uptime in seconds')
            metrics.append('# TYPE system_uptime counter')
            metrics.append(f'system_uptime {int(time.time() - psutil.boot_time())}')
            
            # Add GPU metrics if GPU acceleration is enabled
            if os.environ.get("ENABLE_GPU_ACCELERATION") == "true":
                try:
                    # Mock GPU metrics for Vercel environment
                    # In a real environment, these would come from nvidia-smi or similar
                    metrics.append('# HELP gpu_enabled GPU acceleration enabled')
                    metrics.append('# TYPE gpu_enabled gauge')
                    metrics.append('gpu_enabled 1')
                    
                    metrics.append('# HELP gpu_features GPU features enabled')
                    metrics.append('# TYPE gpu_features gauge')
                    
                    # Report enabled GPU features
                    if os.environ.get("ENABLE_TENSORRT") == "true":
                        metrics.append('gpu_features{feature="tensorrt"} 1')
                    else:
                        metrics.append('gpu_features{feature="tensorrt"} 0')
                        
                    if os.environ.get("ENABLE_TRANSFORMER_ENGINE") == "true":
                        metrics.append('gpu_features{feature="transformer_engine"} 1')
                    else:
                        metrics.append('gpu_features{feature="transformer_engine"} 0')
                        
                    if os.environ.get("ENABLE_FLASH_ATTENTION_2") == "true":
                        metrics.append('gpu_features{feature="flash_attention_2"} 1')
                    else:
                        metrics.append('gpu_features{feature="flash_attention_2"} 0')
                        
                    if os.environ.get("ENABLE_FP8") == "true":
                        metrics.append('gpu_features{feature="fp8"} 1')
                    else:
                        metrics.append('gpu_features{feature="fp8"} 0')
                        
                    if os.environ.get("ENABLE_GPTQ") == "true":
                        metrics.append('gpu_features{feature="gptq"} 1')
                    else:
                        metrics.append('gpu_features{feature="gptq"} 0')
                        
                    if os.environ.get("ENABLE_AWQ") == "true":
                        metrics.append('gpu_features{feature="awq"} 1')
                    else:
                        metrics.append('gpu_features{feature="awq"} 0')
                        
                except Exception as e:
                    # Log GPU metrics error but continue
                    print(f"Error collecting GPU metrics: {str(e)}")
            
            # Add API metrics (mock values for Vercel)
            metrics.append('# HELP api_requests_total Total number of API requests')
            metrics.append('# TYPE api_requests_total counter')
            metrics.append('api_requests_total{endpoint="all"} 0')
            
            metrics.append('# HELP api_request_latency_seconds API request latency in seconds')
            metrics.append('# TYPE api_request_latency_seconds histogram')
            metrics.append('api_request_latency_seconds{quantile="0.5"} 0.0')
            metrics.append('api_request_latency_seconds{quantile="0.9"} 0.0')
            metrics.append('api_request_latency_seconds{quantile="0.99"} 0.0')
            
            metrics.append('# HELP api_errors_total Total number of API errors')
            metrics.append('# TYPE api_errors_total counter')
            metrics.append('api_errors_total{type="all"} 0')
            
            # Write all metrics
            self.wfile.write('\n'.join(metrics).encode('utf-8'))
            
        except Exception as e:
            # Log the error
            print(f"Error in metrics endpoint: {str(e)}")
            
            # Return an error response
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error collecting metrics: {str(e)}".encode('utf-8'))