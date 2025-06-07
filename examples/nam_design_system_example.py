"""
Neural Additive Models Design System Example

This example demonstrates how to integrate the NAM design system into
a FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import HANA ML connection
from hana_ml import ConnectionContext

# Import toolkit
from hana_ai.tools.toolkit import HANAMLToolkit

# Create FastAPI app
app = FastAPI(title="Neural Additive Models Design System Example")

# Create connection to HANA
# In a real application, you would use your own connection details
connection_context = ConnectionContext(
    address='your-hana-address',
    port=443,
    user='your-username',
    password='your-password',
    encrypt=True
)

# Create toolkit and register NAM design system
toolkit = HANAMLToolkit(connection_context=connection_context)
nam_integration = toolkit.register_nam_design_system(app)

# Add a simple home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Additive Models Design System Example</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                line-height: 1.6;
            }
            h1 {
                color: #0070D2;
                border-bottom: 1px solid #E5EAEF;
                padding-bottom: 1rem;
            }
            .card {
                background-color: #F9FBFD;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 2rem 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }
            .btn {
                display: inline-block;
                background-color: #0070D2;
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
                margin-top: 1rem;
            }
            .btn:hover {
                background-color: #004B98;
            }
        </style>
    </head>
    <body>
        <h1>Neural Additive Models Design System</h1>
        
        <p>
            This example demonstrates the Neural Additive Models Design System integration
            with a FastAPI application. The design system provides a cohesive, emotionally
            resonant experience for users interacting with Neural Additive Models.
        </p>
        
        <div class="card">
            <h2>Design System Demo</h2>
            <p>
                Experience the Jony Ive-level design language of the Neural Additive Models
                interface. This demo showcases the comprehensive design system with fluid
                animations, precise interactions, and a unified visual language.
            </p>
            <a href="/api/nam/ui" class="btn">Launch Design System</a>
        </div>
        
        <div class="card">
            <h2>API Documentation</h2>
            <p>
                The NAM Design System includes a comprehensive API for integrating with
                your applications. View the API documentation to learn about available
                endpoints and data schemas.
            </p>
            <a href="/docs" class="btn">View API Docs</a>
        </div>
    </body>
    </html>
    """

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)