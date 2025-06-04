from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """
        Handle GET requests to the health endpoint.
        
        Returns a simple health status response for monitoring systems.
        """
        try:
            # Set response headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Create health status response
            health_status = {
                "status": "healthy",
                "service": "HANA AI Toolkit API",
                "version": "1.0.0",
                "environment": os.environ.get("ENVIRONMENT", "production")
            }
            
            # Check for additional environment info
            if os.environ.get("ENABLE_GPU_ACCELERATION") == "true":
                health_status["gpu_acceleration"] = "enabled"
                
                # Add GPU-specific info if available
                gpu_features = []
                if os.environ.get("ENABLE_TENSORRT") == "true":
                    gpu_features.append("TensorRT")
                if os.environ.get("ENABLE_TRANSFORMER_ENGINE") == "true":
                    gpu_features.append("Transformer Engine")
                if os.environ.get("ENABLE_FLASH_ATTENTION_2") == "true":
                    gpu_features.append("Flash Attention 2")
                if os.environ.get("ENABLE_FP8") == "true":
                    gpu_features.append("FP8")
                if os.environ.get("ENABLE_GPTQ") == "true" or os.environ.get("ENABLE_AWQ") == "true":
                    quantization = []
                    if os.environ.get("ENABLE_GPTQ") == "true":
                        quantization.append("GPTQ")
                    if os.environ.get("ENABLE_AWQ") == "true":
                        quantization.append("AWQ")
                    gpu_features.append(f"Quantization ({', '.join(quantization)})")
                
                if gpu_features:
                    health_status["gpu_features"] = gpu_features
            
            # Return the health status as JSON
            self.wfile.write(json.dumps(health_status).encode('utf-8'))
            
        except Exception as e:
            # Log the error
            print(f"Error in health endpoint: {str(e)}")
            
            # Return an error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }).encode('utf-8'))