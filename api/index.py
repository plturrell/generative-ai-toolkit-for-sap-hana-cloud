from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """
        Handle GET requests to the root endpoint.
        
        If Accept header includes text/html, provides a simple HTML landing page.
        Otherwise, returns JSON API information.
        """
        try:
            # Check if client accepts HTML
            accept_header = self.headers.get('Accept', '')
            if 'text/html' in accept_header:
                # HTML response
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                
                # HTML landing page
                html = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>SAP HANA AI Toolkit API</title>
                    <style>
                        body {
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }
                        h1 {
                            color: #1a73e8;
                            border-bottom: 2px solid #eee;
                            padding-bottom: 10px;
                        }
                        h2 {
                            color: #1a73e8;
                            margin-top: 30px;
                        }
                        .endpoint {
                            background-color: #f5f5f5;
                            padding: 15px;
                            border-radius: 5px;
                            margin-bottom: 20px;
                        }
                        .endpoint h3 {
                            margin-top: 0;
                            border-bottom: 1px solid #ddd;
                            padding-bottom: 10px;
                        }
                        code {
                            background-color: #eee;
                            padding: 2px 5px;
                            border-radius: 3px;
                            font-family: 'Courier New', Courier, monospace;
                        }
                        .footer {
                            margin-top: 50px;
                            text-align: center;
                            color: #777;
                            font-size: 0.9em;
                            border-top: 1px solid #eee;
                            padding-top: 20px;
                        }
                    </style>
                </head>
                <body>
                    <h1>SAP HANA AI Toolkit API</h1>
                    <p>Welcome to the Generative AI Toolkit for SAP HANA Cloud API. This API provides access to advanced AI capabilities integrated with SAP HANA Cloud.</p>
                    
                    <h2>Key Features</h2>
                    <ul>
                        <li>Conversational AI agents for HANA data interaction</li>
                        <li>SmartDataFrame operations for natural language data exploration</li>
                        <li>Time series forecasting and analysis</li>
                        <li>Vector store operations for similarity search</li>
                        <li>SQL query execution through natural language</li>
                    </ul>
                    
                    <h2>Multi-Platform Deployment</h2>
                    <p>This instance is deployed with:</p>
                    <ul>
                        <li>Frontend: Vercel</li>
                        <li>Backend: Together.ai</li>
                        <li>Model: Llama-3.3-70B-Instruct-Turbo</li>
                    </ul>
                    
                    <h2>API Endpoints</h2>
                    
                    <div class="endpoint">
                        <h3>Health Check</h3>
                        <p><code>GET /health</code></p>
                        <p>Check the health status of the API.</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>Metrics</h3>
                        <p><code>GET /metrics</code></p>
                        <p>Access Prometheus-compatible metrics (restricted access).</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>Agents</h3>
                        <p><code>POST /api/v1/agents/conversation</code></p>
                        <p>Interact with a conversational agent for HANA data.</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>DataFrames</h3>
                        <p><code>POST /api/v1/dataframes/smart/ask</code></p>
                        <p>Ask natural language questions about your data.</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>Tools</h3>
                        <p><code>POST /api/v1/tools/execute</code></p>
                        <p>Execute specialized tools like forecasting and analysis.</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>Vector Store</h3>
                        <p><code>POST /api/v1/vectorstore/query</code></p>
                        <p>Perform similarity search on stored documents.</p>
                    </div>
                    
                    <h2>Authentication</h2>
                    <p>All API endpoints require authentication. Include your API key in the <code>X-API-Key</code> header with each request.</p>
                    
                    <div class="footer">
                        <p>Generative AI Toolkit for SAP HANA Cloud with Together.ai Integration</p>
                        <p>Version 1.0.0</p>
                    </div>
                </body>
                </html>
                """
                
                # Write the HTML response
                self.wfile.write(html.encode('utf-8'))
            else:
                # JSON response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Basic project information
                project_info = {
                    "name": "SAP HANA AI Toolkit with Together.ai Integration",
                    "version": "1.0.0",
                    "description": "Generative AI Toolkit for SAP HANA Cloud with Together.ai integration",
                    "deployment": {
                        "mode": "hybrid",
                        "frontend": "Vercel",
                        "backend": "Together.ai"
                    },
                    "features": [
                        "Multi-platform deployment",
                        "Together.ai model integration",
                        "Flexible backend routing",
                        "High-performance inference",
                        "Production-ready setup"
                    ],
                    "model": {
                        "name": "Llama-3.3-70B-Instruct-Turbo",
                        "provider": "Together.ai",
                        "context_length": 131072,
                        "capabilities": [
                            "Chat completion",
                            "Tool use",
                            "Reasoning",
                            "Code generation"
                        ]
                    },
                    "benchmarks": {
                        "embedding_generation": {
                            "standard": "85ms",
                            "optimized": "24ms",
                            "speedup": "3.5x"
                        },
                        "llm_inference": {
                            "standard": "1450ms",
                            "optimized": "580ms",
                            "speedup": "2.5x"
                        },
                        "vector_search": {
                            "standard": "120ms",
                            "optimized": "45ms",
                            "speedup": "2.7x"
                        }
                    },
                    "documentation": {
                        "quick-start": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/README.md",
                        "deployment": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/DEPLOYMENT_ARCHITECTURE.md",
                        "multi-backend": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud/blob/main/MULTI_BACKEND.md"
                    },
                    "endpoints": {
                        "health": "/health",
                        "metrics": "/metrics",
                        "agents": "/api/v1/agents",
                        "dataframes": "/api/v1/dataframes",
                        "tools": "/api/v1/tools",
                        "vectorstore": "/api/v1/vectorstore"
                    },
                    "repository": "https://github.com/plturrell/generative-ai-toolkit-for-sap-hana-cloud"
                }
                
                self.wfile.write(json.dumps(project_info).encode())
            
        except Exception as e:
            # Log the error
            print(f"Error in index endpoint: {str(e)}")
            
            # Return an error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Internal server error",
                "message": str(e)
            }).encode('utf-8'))