# SAP HANA AI Toolkit REST API

This document provides instructions for deploying and using the REST API for the Generative AI Toolkit for SAP HANA Cloud.

## Overview

The REST API extends the Generative AI Toolkit for SAP HANA Cloud to support integration with any application capable of making HTTP requests. It provides endpoints for:

- Conversational AI agents for HANA data interaction
- SmartDataFrame operations for natural language data exploration
- Time series forecasting and analysis
- Vector store operations for similarity search
- SQL query execution through natural language

## Prerequisites

- Python 3.8 or higher
- Access to SAP HANA Cloud instance
- SAP HANA Secure User Store key or credentials

## Installation

```bash
# Install the API package
pip install hana-ai[api]

# Or install from requirements
pip install -r requirements-api.txt
```

## Configuration

The API server can be configured using environment variables:

```bash
# Server settings
export API_HOST=0.0.0.0
export API_PORT=8000
export DEVELOPMENT_MODE=false
export LOG_LEVEL=INFO

# Security settings (comma-separated API keys)
export API_KEYS=your-api-key-1,your-api-key-2
export AUTH_REQUIRED=true
export CORS_ORIGINS=*

# Database connection (either userkey or direct credentials)
export HANA_USERKEY=YOUR_KEY
# OR
export HANA_HOST=your-hana-host.com
export HANA_PORT=443
export HANA_USER=your-username
export HANA_PASSWORD=your-password

# LLM settings
export DEFAULT_LLM_MODEL=gpt-4
export DEFAULT_LLM_TEMPERATURE=0.0
export DEFAULT_LLM_MAX_TOKENS=1000
```

You can also create a `.env` file with these settings in the directory where you run the API.

## Running the API Server

```bash
# Run directly with Python
python -m hana_ai.api

# Or use uvicorn
uvicorn hana_ai.api.app:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger documentation of all endpoints.

## Endpoints

The API provides the following key endpoints:

### Agents

- `POST /api/v1/agents/conversation` - Interact with a conversational agent
- `POST /api/v1/agents/sql` - Execute natural language SQL queries

### DataFrames

- `POST /api/v1/dataframes/query` - Execute SQL queries
- `POST /api/v1/dataframes/smart/ask` - Ask questions about data
- `GET /api/v1/dataframes/tables` - List available tables

### Tools

- `GET /api/v1/tools/list` - List available tools
- `POST /api/v1/tools/execute` - Execute a specific tool
- `POST /api/v1/tools/forecast` - Run time series forecasting

### Vector Store

- `POST /api/v1/vectorstore/query` - Search for similar documents
- `POST /api/v1/vectorstore/store` - Add documents to the vector store
- `POST /api/v1/vectorstore/embed` - Generate embeddings for text

## Authentication

All API endpoints are secured using API keys. Include your API key in requests using the `X-API-Key` header:

```
X-API-Key: your-api-key
```

## Example Usage

### Python Client

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"
HEADERS = {"X-API-Key": API_KEY}

# Ask a question about data
response = requests.post(
    f"{API_URL}/api/v1/dataframes/smart/ask",
    headers=HEADERS,
    json={
        "table_name": "SALES_DATA",
        "question": "What are the average monthly sales in 2023?",
        "transform": False
    }
)
print(response.json())
```

### JavaScript/TypeScript Client

```javascript
const apiUrl = "http://localhost:8000";
const apiKey = "your-api-key";

// Execute a forecasting operation
async function forecast() {
  const response = await fetch(`${apiUrl}/api/v1/tools/forecast`, {
    method: "POST",
    headers: {
      "X-API-Key": apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      table_name: "SALES_DATA",
      key_column: "DATE",
      value_column: "SALES",
      horizon: 12,
      model_name: "sales_forecast"
    })
  });
  
  const result = await response.json();
  console.log(result);
}
```

## Docker Deployment

You can deploy the API using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "hana_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Security Considerations

- Always deploy behind a reverse proxy with SSL/TLS in production
- Use strong, unique API keys and rotate them regularly
- Set up proper network security groups to restrict access
- Store sensitive credentials securely using environment variables or secrets management
- Consider integrating with your organization's OAuth or SAML identity provider

## Troubleshooting

- Check connection to HANA database if query endpoints fail
- Verify API key is correctly set in request headers
- Inspect logs for detailed error information
- Ensure environment variables are correctly configured