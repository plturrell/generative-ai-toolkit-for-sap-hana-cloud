#!/usr/bin/env python3
"""
Simple healthcheck script for Docker healthcheck.
Uses requests to check the health endpoint.
"""
import sys
import requests

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200 and response.json().get("status") == "healthy":
        # Healthcheck passed
        sys.exit(0)
    else:
        # Healthcheck failed
        print(f"Healthcheck failed: Status code {response.status_code}")
        sys.exit(1)
except Exception as e:
    # Healthcheck failed due to exception
    print(f"Healthcheck failed with exception: {str(e)}")
    sys.exit(1)