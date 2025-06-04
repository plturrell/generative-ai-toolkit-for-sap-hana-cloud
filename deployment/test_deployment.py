#!/usr/bin/env python3
"""
Deployment verification script for SAP HANA AI Toolkit.

This script tests the connectivity and functionality of the deployed backend and frontend.
"""

import argparse
import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional, Tuple

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test deployment of SAP HANA AI Toolkit"
    )
    
    parser.add_argument(
        "--backend-url",
        required=True,
        help="URL of the backend API (e.g., https://api.together.xyz/v1/sap-hana-ai-toolkit)"
    )
    
    parser.add_argument(
        "--frontend-url",
        required=True,
        help="URL of the frontend (e.g., https://sap-hana-ai-toolkit.vercel.app)"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for authentication with the backend"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def test_backend_health(backend_url: str, api_key: Optional[str] = None, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Test the backend health endpoint."""
    health_url = f"{backend_url.rstrip('/')}/health"
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        if verbose:
            print(f"Testing backend health: {health_url}")
        
        response = requests.get(health_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if verbose:
            print(f"Backend health response: {json.dumps(result, indent=2)}")
        
        # Check for expected fields
        required_fields = ["status"]
        for field in required_fields:
            if field not in result:
                if verbose:
                    print(f"Missing required field '{field}' in health response")
                return False, result
        
        # Check status
        if result["status"] != "ok":
            if verbose:
                print(f"Backend status is not 'ok': {result['status']}")
            return False, result
        
        return True, result
    
    except Exception as e:
        if verbose:
            print(f"Error testing backend health: {str(e)}")
        return False, {"error": str(e)}

def test_backend_status(backend_url: str, api_key: Optional[str] = None, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Test the backend status endpoint."""
    status_url = f"{backend_url.rstrip('/')}/health/backend-status"
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        if verbose:
            print(f"Testing backend status: {status_url}")
        
        response = requests.get(status_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if verbose:
            print(f"Backend status response: {json.dumps(result, indent=2)}")
        
        return True, result
    
    except Exception as e:
        if verbose:
            print(f"Error testing backend status: {str(e)}")
        return False, {"error": str(e)}

def test_frontend(frontend_url: str, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Test the frontend."""
    try:
        if verbose:
            print(f"Testing frontend: {frontend_url}")
        
        response = requests.get(frontend_url, timeout=10)
        response.raise_for_status()
        
        # Check for HTML response
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            if verbose:
                print(f"Frontend response is not HTML: {content_type}")
            return False, {"error": f"Unexpected content type: {content_type}"}
        
        if verbose:
            print(f"Frontend response status: {response.status_code}")
        
        return True, {"status": "ok", "status_code": response.status_code}
    
    except Exception as e:
        if verbose:
            print(f"Error testing frontend: {str(e)}")
        return False, {"error": str(e)}

def test_cors(backend_url: str, frontend_url: str, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Test CORS configuration."""
    options_url = f"{backend_url.rstrip('/')}/health"
    
    headers = {
        "Origin": frontend_url,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type"
    }
    
    try:
        if verbose:
            print(f"Testing CORS configuration: {options_url}")
        
        response = requests.options(options_url, headers=headers, timeout=10)
        
        cors_headers = {
            "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
            "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers")
        }
        
        if verbose:
            print(f"CORS headers: {json.dumps(cors_headers, indent=2)}")
        
        # Check for expected CORS headers
        if not cors_headers["Access-Control-Allow-Origin"]:
            if verbose:
                print("Missing Access-Control-Allow-Origin header")
            return False, cors_headers
        
        # Check if frontend origin is allowed
        origin_header = cors_headers["Access-Control-Allow-Origin"]
        if origin_header != "*" and origin_header != frontend_url:
            if verbose:
                print(f"Frontend origin not allowed: {origin_header}")
            return False, cors_headers
        
        return True, cors_headers
    
    except Exception as e:
        if verbose:
            print(f"Error testing CORS: {str(e)}")
        return False, {"error": str(e)}

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    backend_url = args.backend_url
    frontend_url = args.frontend_url
    api_key = args.api_key
    verbose = args.verbose
    
    print(f"Testing deployment:")
    print(f"  Backend URL: {backend_url}")
    print(f"  Frontend URL: {frontend_url}")
    print(f"  API Key: {'Provided' if api_key else 'Not provided'}")
    print("")
    
    # Test backend health
    print("Testing backend health...")
    backend_health_success, backend_health_result = test_backend_health(backend_url, api_key, verbose)
    if backend_health_success:
        print("✅ Backend health check succeeded")
    else:
        print("❌ Backend health check failed")
        if verbose:
            print(f"  Error: {json.dumps(backend_health_result, indent=2)}")
    print("")
    
    # Test backend status
    print("Testing backend status...")
    backend_status_success, backend_status_result = test_backend_status(backend_url, api_key, verbose)
    if backend_status_success:
        print("✅ Backend status check succeeded")
    else:
        print("❌ Backend status check failed")
        if verbose:
            print(f"  Error: {json.dumps(backend_status_result, indent=2)}")
    print("")
    
    # Test frontend
    print("Testing frontend...")
    frontend_success, frontend_result = test_frontend(frontend_url, verbose)
    if frontend_success:
        print("✅ Frontend check succeeded")
    else:
        print("❌ Frontend check failed")
        if verbose:
            print(f"  Error: {json.dumps(frontend_result, indent=2)}")
    print("")
    
    # Test CORS
    print("Testing CORS configuration...")
    cors_success, cors_result = test_cors(backend_url, frontend_url, verbose)
    if cors_success:
        print("✅ CORS configuration check succeeded")
    else:
        print("❌ CORS configuration check failed")
        if verbose:
            print(f"  Error: {json.dumps(cors_result, indent=2)}")
    print("")
    
    # Overall result
    all_success = backend_health_success and frontend_success and cors_success
    
    if all_success:
        print("🎉 All deployment tests passed! The system is correctly deployed.")
        return 0
    else:
        print("❌ Some deployment tests failed. Please check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())