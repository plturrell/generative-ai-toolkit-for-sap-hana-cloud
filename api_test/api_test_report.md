# API Test Report - 2025-06-08 08:51:22

Target: http://localhost:8002

## Summary

- Total endpoints tested: 21
- Successful tests: 21
- Failed tests: 0
- Success rate: 100.0%
- Total execution time: 2.70s
- Average response time: 0.129s

## Endpoint Results

### ✅ GET / (Root endpoint)
- Status: 200
- Time: 0.010s

### ✅ GET /health (Health check)
- Status: 200
- Time: 0.003s

### ✅ GET /validate (Validation check)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/health/backend-status (Backend status)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/health/validate (API validation)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/hardware/gpu (GPU info)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/hardware/fallback-status (Fallback status)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/hardware/tensorrt-status (TensorRT status)
- Status: 200
- Time: 0.003s

### ✅ POST /api/v1/hardware/tensorrt/optimize (TensorRT optimize)
- Status: 200
- Time: 1.010s

### ✅ PUT /api/v1/hardware/fallback-config (Update fallback config)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/config/ui (UI config)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/config/environment (Environment config)
- Status: 200
- Time: 0.003s

### ✅ POST /api/v1/batch-sizing/register-model (Register model for batch sizing)
- Status: 200
- Time: 0.003s

### ✅ POST /api/v1/batch-sizing/record-performance (Record batch sizing performance)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/batch-sizing/get-batch-size (Get batch size recommendation)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/optimization/status (Optimization status)
- Status: 200
- Time: 0.003s

### ✅ POST /api/v1/optimization/sparsify (Sparsify model)
- Status: 200
- Time: 0.811s

### ✅ POST /api/v1/agents/conversation (Agent conversation)
- Status: 200
- Time: 0.509s

### ✅ POST /api/v1/agents/sql (SQL agent)
- Status: 200
- Time: 0.308s

### ✅ GET /api/v1/tools/list (List tools)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/vectorstore/providers (Vector store providers)
- Status: 200
- Time: 0.003s
