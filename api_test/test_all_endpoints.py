#!/usr/bin/env python3
"""
Test script to verify all API endpoints in the test environment.

Usage:
    python test_all_endpoints.py [--host localhost] [--port 8002]

This script will test all the implemented API endpoints in the test environment
and report their status, response times, and any issues detected.
"""
import argparse
import json
import requests
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text: str, color: str) -> None:
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")

def make_request(
    method: str, 
    endpoint: str, 
    base_url: str, 
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[int, Any, float]:
    """Make a request to the API endpoint and return status code, response, and time taken."""
    url = f"{base_url}{endpoint}"
    start_time = time.time()
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, json=data)
        else:
            return 500, {"error": f"Unsupported method {method}"}, 0
            
        time_taken = time.time() - start_time
        
        try:
            response_data = response.json()
        except Exception:
            response_data = {"text": response.text}
            
        return response.status_code, response_data, time_taken
    except Exception as e:
        return 500, {"error": str(e)}, time.time() - start_time

def test_endpoint(
    method: str,
    endpoint: str, 
    base_url: str, 
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    expected_status: int = 200,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """Test a specific endpoint and return the results."""
    print_colored(f"Testing {method} {endpoint} {'(' + name + ')' if name else ''}...", Colors.CYAN)
    status, response, time_taken = make_request(method, endpoint, base_url, data, params)
    
    result = {
        "endpoint": endpoint,
        "method": method,
        "name": name or endpoint,
        "status": status,
        "expected_status": expected_status,
        "time_taken": time_taken,
        "success": status == expected_status,
        "response": response
    }
    
    if result["success"]:
        print_colored(f"  ✓ {result['name']} - {status} - {time_taken:.3f}s", Colors.GREEN)
    else:
        print_colored(f"  ✗ {result['name']} - {status} (expected {expected_status}) - {time_taken:.3f}s", Colors.RED)
        print_colored(f"    Response: {json.dumps(response, indent=2)}", Colors.YELLOW)
    
    return result

def run_all_tests(host: str, port: int) -> List[Dict[str, Any]]:
    """Run tests for all endpoints and return the results."""
    base_url = f"http://{host}:{port}"
    results = []
    
    # Basic health endpoints
    results.append(test_endpoint("GET", "/", base_url, name="Root endpoint"))
    results.append(test_endpoint("GET", "/health", base_url, name="Health check"))
    results.append(test_endpoint("GET", "/validate", base_url, name="Validation check"))
    
    # Health endpoints
    results.append(test_endpoint("GET", "/api/v1/health/backend-status", base_url, name="Backend status"))
    results.append(test_endpoint("GET", "/api/v1/health/validate", base_url, name="API validation"))
    results.append(test_endpoint("GET", "/api/v1/health/backend-check/cpu", base_url, name="Backend check (CPU)"))
    results.append(test_endpoint("GET", "/api/v1/health/ping", base_url, name="Health ping"))
    results.append(test_endpoint("GET", "/api/v1/health/metrics", base_url, name="Health metrics"))
    results.append(test_endpoint("GET", "/api/v1/health/platform-info", base_url, name="Platform info"))
    
    # Hardware endpoints
    results.append(test_endpoint("GET", "/api/v1/hardware/gpu", base_url, name="GPU info"))
    results.append(test_endpoint("GET", "/api/v1/hardware/fallback-status", base_url, name="Fallback status"))
    results.append(test_endpoint("GET", "/api/v1/hardware/tensorrt-status", base_url, name="TensorRT status"))
    
    # Hardware optimization endpoints
    results.append(test_endpoint(
        "POST", 
        "/api/v1/hardware/tensorrt/optimize", 
        base_url, 
        data={"model_id": "test-model"}, 
        name="TensorRT optimize"
    ))
    results.append(test_endpoint(
        "PUT", 
        "/api/v1/hardware/fallback-config", 
        base_url, 
        data={"enabled": True, "threshold": 0.85}, 
        name="Update fallback config"
    ))
    
    # Configuration endpoints
    results.append(test_endpoint("GET", "/api/v1/config/status", base_url, name="Config status"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/config/hana", 
        base_url, 
        data={
            "host": "hana-server",
            "port": 30015,
            "user": "DBADMIN",
            "password": "Password123",
            "encrypt": True
        }, 
        name="Config HANA"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/config/aicore", 
        base_url, 
        data={
            "url": "https://aicore.example.com",
            "api_key": "test-api-key",
            "model_name": "gpt-3.5-turbo"
        }, 
        name="Config AI Core"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/config/service", 
        base_url, 
        data={
            "name": "test-service",
            "type": "custom",
            "url": "https://service.example.com",
            "credentials": {"api_key": "test-api-key"}
        }, 
        name="Config service"
    ))
    results.append(test_endpoint(
        "DELETE", 
        "/api/v1/config/service/test-service", 
        base_url, 
        name="Delete service"
    ))
    results.append(test_endpoint("GET", "/api/v1/config/test/hana", base_url, name="Test HANA connection"))
    results.append(test_endpoint("GET", "/api/v1/config/test/aicore", base_url, name="Test AI Core connection"))
    results.append(test_endpoint("GET", "/api/v1/config/batch_sizing", base_url, name="Get batch sizing config"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/config/batch_sizing", 
        base_url, 
        data={
            "strategy": "adaptive",
            "min_batch_size": 1,
            "max_batch_size": 128,
            "default_batch_size": 32
        }, 
        name="Update batch sizing config"
    ))
    results.append(test_endpoint("GET", "/api/v1/config/batch_performance", base_url, name="Get batch performance"))
    results.append(test_endpoint("GET", "/api/v1/config/ui", base_url, name="UI config"))
    results.append(test_endpoint("GET", "/api/v1/config/environment", base_url, name="Environment config"))
    
    # Batch sizing endpoints
    results.append(test_endpoint(
        "POST", 
        "/api/v1/batch-sizing/register-model", 
        base_url, 
        data={
            "model_id": "test-model",
            "description": "Test model",
            "default_batch_size": 32,
            "min_batch_size": 1,
            "max_batch_size": 128
        }, 
        name="Register model for batch sizing"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/batch-sizing/record-performance", 
        base_url, 
        data={
            "model_id": "test-model",
            "batch_size": 32,
            "latency_ms": 120.5,
            "throughput_tokens_per_sec": 1500.8,
            "memory_usage_mb": 2048.0
        }, 
        name="Record batch sizing performance"
    ))
    results.append(test_endpoint(
        "GET", 
        "/api/v1/batch-sizing/get-batch-size", 
        base_url, 
        params={"model_id": "test-model", "input_tokens": 256}, 
        name="Get batch size recommendation"
    ))
    results.append(test_endpoint(
        "GET", 
        "/api/v1/batch-sizing/model-stats/test-model", 
        base_url, 
        name="Get model stats"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/batch-sizing/clear-cache", 
        base_url, 
        data={"model_id": "test-model"}, 
        name="Clear batch cache"
    ))
    results.append(test_endpoint(
        "GET", 
        "/api/v1/batch-sizing/list-models", 
        base_url, 
        name="List batch models"
    ))
    
    # Optimization endpoints
    results.append(test_endpoint("GET", "/api/v1/optimization/status", base_url, name="Optimization status"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/optimization/sparsify", 
        base_url, 
        data={
            "model_id": "test-model",
            "target_sparsity": 0.8,
            "use_quantization": True,
            "precision": "int8"
        }, 
        name="Sparsify model"
    ))
    results.append(test_endpoint(
        "GET", 
        "/api/v1/optimization/models/test-model/stats", 
        base_url, 
        name="Model optimization stats"
    ))
    results.append(test_endpoint(
        "DELETE", 
        "/api/v1/optimization/models/test-model/cache", 
        base_url, 
        name="Clear optimization cache"
    ))
    results.append(test_endpoint(
        "PUT", 
        "/api/v1/optimization/config", 
        base_url, 
        data={
            "enabled": True,
            "target_sparsity": 0.8,
            "use_quantization": True,
            "quantization_bits": 8
        }, 
        name="Update optimization config"
    ))
    
    # Agents endpoints
    results.append(test_endpoint(
        "POST", 
        "/api/v1/agents/conversation", 
        base_url, 
        data={
            "query": "What is HANA Cloud?",
            "history": [],
            "tools": ["GetCodeTemplateFromVectorDB", "HANAMLToolkit"]
        }, 
        name="Agent conversation"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/agents/sql", 
        base_url, 
        data={
            "query": "Show me the top 10 customers by revenue",
            "connection_id": "test-connection"
        }, 
        name="SQL agent"
    ))
    results.append(test_endpoint("GET", "/api/v1/agents/types", base_url, name="Agent types"))
    
    # Tools endpoints
    results.append(test_endpoint("GET", "/api/v1/tools/list", base_url, name="List tools"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/tools/execute", 
        base_url, 
        data={
            "tool_name": "GetCodeTemplateFromVectorDB",
            "parameters": {
                "query": "HANA SQL connection",
                "top_k": 3
            }
        }, 
        name="Execute tool"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/tools/forecast", 
        base_url, 
        data={
            "data": [
                {"date": "2025-01-01", "value": 100},
                {"date": "2025-01-02", "value": 110},
                {"date": "2025-01-03", "value": 120}
            ],
            "time_column": "date",
            "value_column": "value",
            "horizon": 5
        }, 
        name="Run forecast"
    ))
    
    # Vectorstore endpoints
    results.append(test_endpoint("GET", "/api/v1/vectorstore/providers", base_url, name="Vector store providers"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/vectorstore/query", 
        base_url, 
        data={
            "query": "HANA SQL connection example",
            "top_k": 3,
            "provider": "HANAVectorEmbeddings"
        }, 
        name="Vector store query"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/vectorstore/store", 
        base_url, 
        data={
            "documents": [
                {"content": "HANA SQL connection example", "title": "Example 1"},
                {"content": "HANA ML toolkit usage", "title": "Example 2"}
            ],
            "provider": "HANAVectorEmbeddings"
        }, 
        name="Vector store document storage"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/vectorstore/embed", 
        base_url, 
        data={
            "text": ["HANA SQL connection example", "HANA ML toolkit usage"],
            "provider": "HANAVectorEmbeddings"
        }, 
        name="Vector store embed"
    ))
    
    # Dataframes endpoints
    results.append(test_endpoint(
        "POST", 
        "/api/v1/dataframes/query", 
        base_url, 
        data={
            "query": "SELECT * FROM SALES LIMIT 10",
            "connection_id": "default"
        }, 
        name="Dataframe query"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/dataframes/smart/ask", 
        base_url, 
        data={
            "question": "What is the average sales amount?",
            "table_name": "SALES",
            "connection_id": "default"
        }, 
        name="Dataframe smart ask"
    ))
    results.append(test_endpoint(
        "GET", 
        "/api/v1/dataframes/tables", 
        base_url, 
        params={"connection_id": "default"}, 
        name="Dataframe tables"
    ))
    
    # Developer endpoints
    results.append(test_endpoint(
        "POST", 
        "/api/v1/developer/generate-code", 
        base_url, 
        data={
            "flow": {
                "nodes": [
                    {"id": "n1", "type": "input"},
                    {"id": "n2", "type": "process"},
                    {"id": "n3", "type": "output"}
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2"},
                    {"id": "e2", "source": "n2", "target": "n3"}
                ]
            },
            "language": "python"
        }, 
        name="Generate code"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/developer/execute-code", 
        base_url, 
        data={
            "code": "print('Hello, World!')",
            "language": "python"
        }, 
        name="Execute code"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/developer/generate-query", 
        base_url, 
        data={
            "schema": {
                "SALES": [
                    {"name": "id", "type": "int"},
                    {"name": "date", "type": "date"},
                    {"name": "amount", "type": "decimal"}
                ],
                "CUSTOMERS": [
                    {"name": "id", "type": "int"},
                    {"name": "name", "type": "string"}
                ]
            },
            "requirements": "Get average sales amount",
            "dialect": "hana"
        }, 
        name="Generate query"
    ))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/developer/execute-query", 
        base_url, 
        data={
            "query": "SELECT AVG(amount) FROM SALES",
            "connection_id": "default"
        }, 
        name="Execute query"
    ))
    results.append(test_endpoint("GET", "/api/v1/developer/flows", base_url, name="List flows"))
    results.append(test_endpoint(
        "POST", 
        "/api/v1/developer/flows", 
        base_url, 
        data={
            "name": "New Test Flow",
            "flow_type": "data_processing",
            "nodes": [{"id": "n1", "type": "input"}],
            "edges": []
        }, 
        name="Save flow"
    ))
    results.append(test_endpoint("GET", "/api/v1/developer/flows/flow1", base_url, name="Get flow"))
    results.append(test_endpoint("DELETE", "/api/v1/developer/flows/flow1", base_url, name="Delete flow"))
    
    return results

def generate_report(results: List[Dict[str, Any]], host: str, port: int) -> str:
    """Generate a report from the test results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    total_time = sum(r["time_taken"] for r in results)
    
    report = [
        f"# API Test Report - {now}",
        f"\nTarget: http://{host}:{port}",
        f"\n## Summary",
        f"\n- Total endpoints tested: {total_tests}",
        f"- Successful tests: {successful_tests}",
        f"- Failed tests: {total_tests - successful_tests}",
        f"- Success rate: {successful_tests/total_tests*100:.1f}%",
        f"- Total execution time: {total_time:.2f}s",
        f"- Average response time: {total_time/total_tests:.3f}s",
        f"\n## Endpoint Results\n"
    ]
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        report.append(f"### {status} {r['method']} {r['endpoint']} ({r['name']})")
        report.append(f"- Status: {r['status']}")
        report.append(f"- Time: {r['time_taken']:.3f}s")
        if not r["success"]:
            report.append(f"- Expected status: {r['expected_status']}")
            report.append("- Response:")
            report.append("```json")
            report.append(json.dumps(r["response"], indent=2))
            report.append("```")
        report.append("")
    
    return "\n".join(report)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test all API endpoints in the test environment")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", default=8002, type=int, help="API port (default: 8002)")
    parser.add_argument("--report", default="api_test_report.md", help="Path to save the test report (default: api_test_report.md)")
    args = parser.parse_args()
    
    print_colored(f"Starting API endpoint tests on {args.host}:{args.port}", Colors.HEADER + Colors.BOLD)
    
    try:
        results = run_all_tests(args.host, args.port)
        
        # Generate and save the report
        report = generate_report(results, args.host, args.port)
        with open(args.report, "w") as f:
            f.write(report)
        
        # Print summary
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        print_colored("\nTest Summary:", Colors.HEADER)
        print_colored(f"Total endpoints tested: {total}", Colors.BOLD)
        print_colored(f"Successful tests: {successful}", Colors.GREEN)
        print_colored(f"Failed tests: {total - successful}", Colors.RED if total - successful > 0 else Colors.GREEN)
        print_colored(f"Success rate: {successful/total*100:.1f}%", Colors.GREEN if successful == total else Colors.YELLOW)
        print_colored(f"\nDetailed report saved to {args.report}", Colors.BLUE)
        
        if successful < total:
            sys.exit(1)
    except Exception as e:
        print_colored(f"Error during testing: {str(e)}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()