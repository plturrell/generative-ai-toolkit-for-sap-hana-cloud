import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const queryLatency = new Trend('query_latency');
const agentLatency = new Trend('agent_latency');
const toolLatency = new Trend('tool_latency');
const vectorLatency = new Trend('vector_latency');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 }, // Ramp up to 10 users
    { duration: '3m', target: 50 }, // Ramp up to 50 users
    { duration: '5m', target: 50 }, // Stay at 50 users
    { duration: '1m', target: 0 },  // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests should be below 1s
    'query_latency': ['p(95)<500'],    // 95% of queries should be below 500ms
    'agent_latency': ['p(95)<5000'],   // 95% of agent requests should be below 5s
    'errors': ['rate<0.1'],            // Error rate should be less than 10%
  },
};

// Shared parameters
const BASE_URL = __ENV.API_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';
const HEADERS = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

export function setup() {
  // Check that the API is available
  const res = http.get(`${BASE_URL}/`);
  check(res, {
    'API is available': (r) => r.status === 200,
    'API is healthy': (r) => r.json('status') === 'healthy',
  });
  
  // Return test context
  return {
    sessionId: `load-test-${Date.now()}`,
  };
}

export default function(data) {
  group('Health Check', function() {
    const res = http.get(`${BASE_URL}/`);
    check(res, {
      'status is 200': (r) => r.status === 200,
      'API is healthy': (r) => r.json('status') === 'healthy',
    });
    errorRate.add(res.status !== 200);
  });
  
  group('Database Queries', function() {
    const payload = JSON.stringify({
      query: 'SELECT * FROM DUMMY',
      limit: 10,
      offset: 0
    });
    
    const start = new Date();
    const res = http.post(`${BASE_URL}/api/v1/dataframes/query`, payload, { headers: HEADERS });
    const duration = new Date() - start;
    
    check(res, {
      'status is 200': (r) => r.status === 200,
      'returns data': (r) => r.json('data').length > 0,
    });
    
    queryLatency.add(duration);
    errorRate.add(res.status !== 200);
  });
  
  group('List Tools', function() {
    const res = http.get(`${BASE_URL}/api/v1/tools/list`, { headers: HEADERS });
    check(res, {
      'status is 200': (r) => r.status === 200,
      'returns tools': (r) => r.json('tools').length > 0,
    });
    errorRate.add(res.status !== 200);
  });
  
  group('Execute Tool', function() {
    const payload = JSON.stringify({
      tool_name: 'fetch_data',
      parameters: {
        table_name: 'DUMMY',
        top_n: 5
      }
    });
    
    const start = new Date();
    const res = http.post(`${BASE_URL}/api/v1/tools/execute`, payload, { headers: HEADERS });
    const duration = new Date() - start;
    
    check(res, {
      'status is 200': (r) => r.status === 200,
      'returns result': (r) => r.json('result') !== undefined,
    });
    
    toolLatency.add(duration);
    errorRate.add(res.status !== 200);
  });
  
  group('Agent Conversation', function() {
    // Only run this test for a subset of users to avoid overwhelming the LLM
    if (Math.random() < 0.2) {
      const payload = JSON.stringify({
        message: 'Show me the database schema',
        session_id: data.sessionId,
        return_intermediate_steps: false,
        verbose: false
      });
      
      const start = new Date();
      const res = http.post(`${BASE_URL}/api/v1/agents/conversation`, payload, { headers: HEADERS });
      const duration = new Date() - start;
      
      check(res, {
        'status is 200': (r) => r.status === 200,
        'returns response': (r) => r.json('response') !== undefined,
      });
      
      agentLatency.add(duration);
      errorRate.add(res.status !== 200);
    }
  });
  
  group('Vector Store Query', function() {
    // Only run this test for a subset of users to avoid overwhelming the vector store
    if (Math.random() < 0.3) {
      const payload = JSON.stringify({
        query: 'Time series forecast',
        top_k: 1,
        collection_name: 'test_collection'
      });
      
      const start = new Date();
      const res = http.post(`${BASE_URL}/api/v1/vectorstore/query`, payload, { headers: HEADERS });
      const duration = new Date() - start;
      
      check(res, {
        'status is 200': (r) => r.status === 200,
        'returns results': (r) => r.json('results') !== undefined,
      });
      
      vectorLatency.add(duration);
      errorRate.add(res.status !== 200);
    }
  });
  
  // Add think time between iterations to simulate real user behavior
  sleep(Math.random() * 3 + 1);
}