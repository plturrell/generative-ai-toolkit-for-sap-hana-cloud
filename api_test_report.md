# API Test Report - 2025-06-08 09:17:07

Target: http://localhost:8002

## Summary

- Total endpoints tested: 58
- Successful tests: 58
- Failed tests: 0
- Success rate: 100.0%
- Total execution time: 8.63s
- Average response time: 0.149s

## Endpoint Results

### ✅ GET / (Root endpoint)
- Status: 200
- Time: 0.012s

### ✅ GET /health (Health check)
- Status: 200
- Time: 0.003s

### ✅ GET /validate (Validation check)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/health/backend-status (Backend status)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/health/validate (API validation)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/health/backend-check/cpu (Backend check (CPU))
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/health/ping (Health ping)
- Status: 200
- Time: 0.003s

### ✅ GET /api/v1/health/metrics (Health metrics)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/health/platform-info (Platform info)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/hardware/gpu (GPU info)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/hardware/fallback-status (Fallback status)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/hardware/tensorrt-status (TensorRT status)
- Status: 200
- Time: 0.002s

### ✅ POST /api/v1/hardware/tensorrt/optimize (TensorRT optimize)
- Status: 200
- Time: 1.006s

### ✅ PUT /api/v1/hardware/fallback-config (Update fallback config)
- Status: 200
- Time: 0.006s

### ✅ GET /api/v1/config/status (Config status)
- Status: 200
- Time: 0.008s

### ✅ POST /api/v1/config/hana (Config HANA)
- Status: 200
- Time: 0.008s

### ✅ POST /api/v1/config/aicore (Config AI Core)
- Status: 200
- Time: 0.004s

### ✅ POST /api/v1/config/service (Config service)
- Status: 200
- Time: 0.006s

### ✅ DELETE /api/v1/config/service/test-service (Delete service)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/config/test/hana (Test HANA connection)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/config/test/aicore (Test AI Core connection)
- Status: 200
- Time: 0.005s

### ✅ GET /api/v1/config/batch_sizing (Get batch sizing config)
- Status: 200
- Time: 0.004s

### ✅ POST /api/v1/config/batch_sizing (Update batch sizing config)
- Status: 200
- Time: 0.005s

### ✅ GET /api/v1/config/batch_performance (Get batch performance)
- Status: 200
- Time: 0.005s

### ✅ GET /api/v1/config/ui (UI config)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/config/environment (Environment config)
- Status: 200
- Time: 0.005s

### ✅ POST /api/v1/batch-sizing/register-model (Register model for batch sizing)
- Status: 200
- Time: 0.005s

### ✅ POST /api/v1/batch-sizing/record-performance (Record batch sizing performance)
- Status: 200
- Time: 0.005s

### ✅ GET /api/v1/batch-sizing/get-batch-size (Get batch size recommendation)
- Status: 200
- Time: 0.007s

### ✅ GET /api/v1/batch-sizing/model-stats/test-model (Get model stats)
- Status: 200
- Time: 0.007s

### ✅ POST /api/v1/batch-sizing/clear-cache (Clear batch cache)
- Status: 200
- Time: 0.006s

### ✅ GET /api/v1/batch-sizing/list-models (List batch models)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/optimization/status (Optimization status)
- Status: 200
- Time: 0.006s

### ✅ POST /api/v1/optimization/sparsify (Sparsify model)
- Status: 200
- Time: 0.810s

### ✅ GET /api/v1/optimization/models/test-model/stats (Model optimization stats)
- Status: 200
- Time: 0.013s

### ✅ DELETE /api/v1/optimization/models/test-model/cache (Clear optimization cache)
- Status: 200
- Time: 0.009s

### ✅ PUT /api/v1/optimization/config (Update optimization config)
- Status: 200
- Time: 0.007s

### ✅ POST /api/v1/agents/conversation (Agent conversation)
- Status: 200
- Time: 0.508s

### ✅ POST /api/v1/agents/sql (SQL agent)
- Status: 200
- Time: 0.307s

### ✅ GET /api/v1/agents/types (Agent types)
- Status: 200
- Time: 0.004s

### ✅ GET /api/v1/tools/list (List tools)
- Status: 200
- Time: 0.005s

### ✅ POST /api/v1/tools/execute (Execute tool)
- Status: 200
- Time: 0.505s

### ✅ POST /api/v1/tools/forecast (Run forecast)
- Status: 200
- Time: 0.809s

### ✅ GET /api/v1/vectorstore/providers (Vector store providers)
- Status: 200
- Time: 0.004s

### ✅ POST /api/v1/vectorstore/query (Vector store query)
- Status: 200
- Time: 0.504s

### ✅ POST /api/v1/vectorstore/store (Vector store document storage)
- Status: 200
- Time: 0.509s

### ✅ POST /api/v1/vectorstore/embed (Vector store embed)
- Status: 200
- Time: 0.310s

### ✅ POST /api/v1/dataframes/query (Dataframe query)
- Status: 200
- Time: 0.306s

### ✅ POST /api/v1/dataframes/smart/ask (Dataframe smart ask)
- Status: 200
- Time: 0.506s

### ✅ GET /api/v1/dataframes/tables (Dataframe tables)
- Status: 200
- Time: 0.006s

### ✅ POST /api/v1/developer/generate-code (Generate code)
- Status: 200
- Time: 0.807s

### ✅ POST /api/v1/developer/execute-code (Execute code)
- Status: 200
- Time: 0.513s

### ✅ POST /api/v1/developer/generate-query (Generate query)
- Status: 200
- Time: 0.611s

### ✅ POST /api/v1/developer/execute-query (Execute query)
- Status: 200
- Time: 0.412s

### ✅ GET /api/v1/developer/flows (List flows)
- Status: 200
- Time: 0.003s

### ✅ POST /api/v1/developer/flows (Save flow)
- Status: 200
- Time: 0.002s

### ✅ GET /api/v1/developer/flows/flow1 (Get flow)
- Status: 200
- Time: 0.002s

### ✅ DELETE /api/v1/developer/flows/flow1 (Delete flow)
- Status: 200
- Time: 0.002s
