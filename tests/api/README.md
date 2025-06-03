# API Tests

This directory contains tests for the HANA AI Toolkit API.

## Test Organization

The tests are organized by component:

- `test_app.py` - Tests for the main FastAPI application
- `test_agents.py` - Tests for agent endpoints
- `test_dataframes.py` - Tests for dataframe endpoints
- `test_tools.py` - Tests for tool endpoints
- `test_vectorstore.py` - Tests for vector store endpoints
- `test_auth.py` - Tests for authentication
- `test_config.py` - Tests for configuration handling
- `test_dependencies.py` - Tests for dependency injection
- `test_integration.py` - Integration tests for combined functionality
- `test_main.py` - Tests for the entry point

## Running the Tests

To run all API tests:

```bash
pytest tests/api
```

To run tests for a specific component:

```bash
pytest tests/api/test_agents.py
```

To run tests with coverage:

```bash
pytest tests/api --cov=hana_ai.api
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `test_client` - FastAPI TestClient instance
- `mock_connection_context` - Mock HANA connection
- `mock_llm` - Mock language model
- `mock_tools` - Mock toolkit tools
- `mock_agent` - Mock agent instance
- `mock_smart_dataframe` - Mock SmartDataFrame
- `mock_vector_store` - Mock vector store
- `mock_embedding_model` - Mock embedding model

## Testing Strategy

1. **Unit Tests**: Testing individual components in isolation
2. **Integration Tests**: Testing interaction between components
3. **API Tests**: Testing HTTP endpoints directly

Most tests use mocking to isolate components and avoid actual database connections.

## Test Coverage Goals

- Code coverage: >90%
- All error handling paths tested
- All authentication scenarios tested
- All endpoint parameters tested