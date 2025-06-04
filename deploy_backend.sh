#\!/bin/bash

# Deploy backend to Together.ai
echo "Deploying backend to Together.ai..."

TOGETHER_API_KEY="tgp_v1_Yxsd-Ud8_JfNVm5lcnSmXvvWpNBMPq1_KY6ZxQLRZYI"

# Get available models
echo "Fetching available models..."
curl -s "https://api.together.xyz/api/models" \
  -H "Authorization: Bearer ${TOGETHER_API_KEY}" | head -n 5

# Create the Together.ai endpoint using the newer API
RESPONSE=$(curl -s -X POST "https://api.together.xyz/instances/create" \
  -H "Authorization: Bearer ${TOGETHER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sap-hana-ai-toolkit-backend",
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "description": "SAP HANA AI Toolkit Backend API",
    "hardware": "a100-40gb",
    "environment": {
      "DEPLOYMENT_MODE": "api_only",
      "DEPLOYMENT_PLATFORM": "together",
      "FRONTEND_URL": "https://sap-hana-ai-toolkit.vercel.app",
      "CORS_ORIGINS": "*",
      "AUTH_REQUIRED": "true",
      "ENABLE_TOGETHER_AI": "true"
    }
  }')

echo "Response: $RESPONSE"

# Check if endpoint was created successfully
if echo "$RESPONSE" | grep -q "id"; then
  echo "Backend deployed successfully\!"
  # Extract endpoint ID
  INSTANCE_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
  echo "Instance ID: $INSTANCE_ID"
else
  echo "Failed to deploy backend."
  exit 1
fi
