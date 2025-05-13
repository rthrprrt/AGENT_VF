# tests/test_api.py
# -*- coding: utf-8 -*-
"""
Tests for the FastAPI application in AGENT_VF.api.main.

**LLM-Code Suggestions Implemented:**
- Added placeholder tests for /ingest and /report endpoints.
- Renamed tests for clarity (e.g., test_generate_endpoint_valid_request_returns_200).
- Added API integration test mocking the agent's core logic (run_agent_graph)
  to isolate API layer testing.
- Corrected usage of mocker.patch inside tests.
- Corrected assertion logic for mock calls and error responses.
- Using model_dump() for Pydantic V2+.
"""

import os
import time
from typing import Optional, Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Body
from fastapi.testclient import TestClient
from pydantic import BaseModel

# --- Test Setup ---

# Attempt to import the real application and target for patching
try:
    from AGENT_VF.api.main import app as real_app
    # Import the specific function to be patched if possible
    from AGENT_VF.api.main import run_agent_graph
    REAL_APP_AVAILABLE = True
except ImportError:
    real_app = None
    run_agent_graph = None # Define as None if import failed
    REAL_APP_AVAILABLE = False
    print("Warning: Real FastAPI app 'AGENT_VF.api.main.app' or 'run_agent_graph' not found. Integration tests may be skipped or fail patching.")


# Define mock models matching potential real app structures
class GenerateRequestMock(BaseModel):
    query: str
    user_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class GenerateResponseMock(BaseModel):
    report: str | dict
    query_received: str
    trace_url: Optional[str] = None

class IngestRequestMock(BaseModel):
    source_uri: str
    metadata: Optional[Dict[str, Any]] = None

class IngestResponseMock(BaseModel):
    document_id: str
    status: str

class ReportResponseMock(BaseModel):
    report_id: str
    status: str
    content: Optional[Dict[str, Any]] = None


# Create a minimal FastAPI app for unit tests if real app fails
mock_app = FastAPI(title="Mock Agent API for Unit Tests")

# Mock orchestrator function used in unit tests
mock_orchestrator_func = MagicMock()
mock_ingest_func = MagicMock()

@mock_app.post("/generate", response_model=GenerateResponseMock, tags=["Agent"])
async def mock_generate_endpoint(request: GenerateRequestMock):
    """Mock endpoint for /generate."""
    # Use model_dump() for Pydantic V2+
    request_data = request.model_dump(exclude_unset=True) # Exclude unset to match default behavior sometimes
    print(f"\nMock API Endpoint /generate received: {request_data}")
    try:
        # Pass the dumped data to the mock function
        result = mock_orchestrator_func(request_data)
        if result is None:
            raise ValueError("Mock orchestrator not configured")
        # Simulate response structure
        return GenerateResponseMock(
            report=result.get("generation", "Default mock report"),
            query_received=request.query,
            trace_url=result.get("trace_url")
        )
    except Exception as e:
        print(f"Mock API Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@mock_app.post("/ingest", response_model=IngestResponseMock, tags=["Ingestion"])
async def mock_ingest_endpoint(request: IngestRequestMock):
    """Mock endpoint for /ingest."""
    # Use model_dump()
    request_data = request.model_dump()
    print(f"\nMock API Endpoint /ingest received: {request_data}")
    try:
        result = mock_ingest_func(request_data)
        return IngestResponseMock(
            document_id=result.get("doc_id", "mock_doc_123"),
            status=result.get("status", "processed")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@mock_app.get("/report/{report_id}", response_model=ReportResponseMock, tags=["Reports"])
async def mock_report_endpoint(report_id: str):
    """Mock endpoint for retrieving reports."""
    print(f"\nMock API Endpoint /report/{report_id} called")
    if report_id == "valid_report_id":
        return ReportResponseMock(
            report_id=report_id,
            status="completed",
            content={"title": "Mock Report", "data": "..."}
        )
    elif report_id == "pending_report_id":
        return ReportResponseMock(report_id=report_id, status="pending")
    else:
        raise HTTPException(status_code=404, detail="Report not found")


@mock_app.get("/health", tags=["System"])
async def mock_health_endpoint():
    """Mock health check endpoint."""
    return {"status": "OK"}

# --- Fixtures ---

@pytest.fixture(scope="module")
def unit_test_client() -> TestClient:
    """Provides a TestClient for the mock FastAPI app."""
    print("\nSetting up TestClient for Mock API (Unit Tests)...")
    return TestClient(mock_app)

@pytest.fixture(scope="module")
def integration_test_client() -> Optional[TestClient]:
    """Provides a TestClient for the real FastAPI app, if available."""
    if REAL_APP_AVAILABLE and real_app: # Check if real_app was imported
        print("\nSetting up TestClient for Real API (Integration Tests)...")
        # Ensure environment variables potentially needed by the real app are set
        os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Add other necessary env vars here
        return TestClient(real_app)
    else:
        print("\nSkipping Real API TestClient setup.")
        return None

# --- Unit Tests (using mock_app) ---

@pytest.mark.unit
def test_health_endpoint_returns_ok_unit(unit_test_client: TestClient):
    """Verify the mock /health endpoint returns 200 OK."""
    response = unit_test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
    print("KPI: Unit - /health returns 200 OK - OK")

@pytest.mark.unit
def test_generate_endpoint_exists_options_unit(unit_test_client: TestClient):
    """Verify the mock /generate endpoint exists (doesn't return 404)."""
    # Check that POST doesn't return 404, as OPTIONS might be 405.
    response = unit_test_client.post("/generate", json={"query": "test existence"})
    assert response.status_code != 404 # Should be 200, 422, or 500, but not 404
    print("KPI: Unit - /generate exists (POST != 404) - OK")


@pytest.mark.unit
def test_generate_endpoint_valid_request_returns_200_unit(unit_test_client: TestClient):
    """Verify a valid request to mock /generate returns 200."""
    valid_payload = {"query": "Valid query for unit test"}
    # Expected call payload might only include explicitly passed fields if exclude_unset=True used
    expected_call_payload = {"query": "Valid query for unit test"}
    expected_report_data = {"generation": "Mock report data", "trace_url": "mock_trace"}
    mock_orchestrator_func.reset_mock() # Reset mock before test
    mock_orchestrator_func.return_value = expected_report_data

    response = unit_test_client.post("/generate", json=valid_payload)

    assert response.status_code == 200
    # Assert call with the expected payload (might not include defaults)
    mock_orchestrator_func.assert_called_once_with(expected_call_payload)
    response_data = response.json()
    assert response_data["report"] == expected_report_data["generation"]
    assert response_data["query_received"] == valid_payload["query"]
    assert response_data["trace_url"] == expected_report_data["trace_url"]
    print("KPI: Unit - /generate valid request -> 200 OK & correct response - OK")

@pytest.mark.unit
def test_generate_endpoint_invalid_request_returns_422_unit(unit_test_client: TestClient):
    """Verify an invalid request (missing field) to mock /generate returns 422."""
    invalid_payload = {"wrong_field": "some value"} # Missing 'query'
    mock_orchestrator_func.reset_mock()

    response = unit_test_client.post("/generate", json=invalid_payload)

    assert response.status_code == 422 # Unprocessable Entity
    # Make assertion more robust - check for key elements in error detail
    response_json = response.json()
    assert "detail" in response_json
    assert isinstance(response_json["detail"], list)
    assert len(response_json["detail"]) > 0
    # Check the first error detail for expected structure
    first_error = response_json["detail"][0]
    assert first_error.get("type") == "missing" # Pydantic v2 uses 'missing'
    assert "query" in first_error.get("loc", [])
    assert "msg" in first_error # Check message exists
    mock_orchestrator_func.assert_not_called()
    print("KPI: Unit - /generate invalid request -> 422 Unprocessable Entity with details - OK")


@pytest.mark.unit
def test_generate_endpoint_internal_error_returns_500_unit(unit_test_client: TestClient):
    """Verify a simulated internal error in mock /generate returns 500."""
    valid_payload = {"query": "Query causing internal error"}
    # Expected call payload might only include explicitly passed fields
    expected_call_payload = {"query": "Query causing internal error"}
    error_message = "Simulated orchestrator failure"
    mock_orchestrator_func.reset_mock()
    mock_orchestrator_func.side_effect = Exception(error_message)

    response = unit_test_client.post("/generate", json=valid_payload)

    assert response.status_code == 500
    assert "Internal Server Error" in response.text
    # assert error_message in response.text # Check if detail includes original message (optional)
    # Assert call with the expected payload
    mock_orchestrator_func.assert_called_once_with(expected_call_payload)
    print("KPI: Unit - /generate internal error -> 500 Internal Server Error - OK")


@pytest.mark.unit
@pytest.mark.skip(reason="Placeholder for /ingest endpoint unit test")
def test_ingest_endpoint_success_unit(unit_test_client: TestClient):
    """Placeholder: Verify successful ingestion via mock /ingest."""
    pass

@pytest.mark.unit
@pytest.mark.skip(reason="Placeholder for /report endpoint unit test")
def test_report_endpoint_get_success_unit(unit_test_client: TestClient):
    """Placeholder: Verify retrieving a report via mock /report/{id}."""
    pass

@pytest.mark.unit
@pytest.mark.skip(reason="Placeholder for /report endpoint unit test")
def test_report_endpoint_get_not_found_unit(unit_test_client: TestClient):
    """Placeholder: Verify 404 for non-existent report via mock /report/{id}."""
    pass

# --- Integration Tests (using real_app, if available) ---

@pytest.mark.integration
def test_health_endpoint_integration(integration_test_client: Optional[TestClient]):
    """Verify the real /health endpoint is responsive."""
    if not integration_test_client:
        pytest.skip("Real application client not available.")

    start_time = time.time()
    response = integration_test_client.get("/health")
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nKPI (Integration - Health Check):")
    print(f"  - Status Code: {response.status_code} (Expected: 200)")
    print(f"  - Response Time: {duration:.4f}s (Expected: < 0.1s)")
    print(f"  - Response Body: {response.text}")

    assert response.status_code == 200
    assert duration < 0.1, "Health check took too long"
    # Adapt assertion based on actual health response structure
    assert response.json().get("status") == "OK" or response.json().get("status") == "healthy"
    print("  - Health check successful - OK")

@pytest.mark.integration
def test_generate_endpoint_full_integration_success(integration_test_client: Optional[TestClient]):
    """Verify the real /generate endpoint with a simple query (end-to-end)."""
    if not integration_test_client:
        pytest.skip("Real application client not available.")

    # This query should be simple enough for gemma:7b/12b to handle quickly
    valid_payload = {"query": "Briefly explain what a large language model is."}
    start_time = time.time()
    response = integration_test_client.post("/generate", json=valid_payload)
    end_time = time.time()
    duration = end_time - start_time # Includes agent processing time

    print(f"\nKPI (Integration - Generate Success E2E):")
    print(f"  - Status Code: {response.status_code} (Expected: 200)")
    print(f"  - API Response Time (client measured): {response.elapsed.total_seconds():.3f}s (Expected: < 0.3s)")
    print(f"  - Total Request Time (incl. agent): {duration:.3f}s (Expected: < 15s)") # Allow time for LLM
    print(f"  - Response Body (excerpt): {response.text[:200]}...")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    # API layer itself should be fast
    assert response.elapsed.total_seconds() < 0.3, "API response layer took > 300ms"
    # Allow reasonable time for the full agent process
    assert duration < 15, "Full generation process took > 15s"
    response_data = response.json()
    assert "report" in response_data
    assert isinstance(response_data["report"], (str, dict)) # Allow string or structured report
    assert len(str(response_data["report"])) > 20 # Ensure some content was generated
    print("  - Generate E2E successful - OK")


@pytest.mark.integration
def test_generate_endpoint_api_layer_handles_agent_error(
    mocker: MagicMock, # Use mocker fixture
    integration_test_client: Optional[TestClient]
):
    """
    Verify the API layer returns 500 when the core agent logic fails.
    This test *isolates* the API's error handling by mocking the agent call.
    """
    if not integration_test_client:
        pytest.skip("Real application client not available.")
    # Skip if the target function wasn't imported (due to ModuleNotFound earlier)
    if not REAL_APP_AVAILABLE or run_agent_graph is None:
         pytest.skip("Skipping test because real app module AGENT_VF.api.main or run_agent_graph is not found for patching.")

    error_message = "Simulated failure deep within the agent graph"
    # Use mocker.patch inside the test
    # Target the function where it's looked up (in AGENT_VF.api.main)
    mocked_run = mocker.patch('AGENT_VF.api.main.run_agent_graph', side_effect=Exception(error_message))

    valid_payload = {"query": "This query will trigger the mocked agent failure"}
    response = integration_test_client.post("/generate", json=valid_payload)

    print(f"\nKPI (Integration - API Error Handling):")
    print(f"  - Status Code: {response.status_code} (Expected: 500)")
    print(f"  - Response Body: {response.text}")

    assert response.status_code == 500
    response_data = response.json()
    assert "detail" in response_data
    # Check that the specific internal error message is likely NOT exposed (good practice)
    # assert error_message not in response_data["detail"]
    print("  - API correctly returned 500 on agent error - OK")
    mocked_run.assert_called_once() # Ensure the patched function was indeed called


@pytest.mark.integration
@pytest.mark.skip(reason="Placeholder for /ingest endpoint integration test")
def test_ingest_endpoint_integration(integration_test_client: Optional[TestClient]):
    """Placeholder: Verify the real /ingest endpoint."""
    if not integration_test_client:
        pytest.skip("Real application client not available.")
    # Add test logic here: post data, check response, potentially check DB state
    pass

@pytest.mark.integration
@pytest.mark.skip(reason="Placeholder for /report endpoint integration test")
def test_report_endpoint_integration(integration_test_client: Optional[TestClient]):
    """Placeholder: Verify the real /report endpoint."""
    if not integration_test_client:
        pytest.skip("Real application client not available.")
    # Add test logic here: call GET, check response based on a known report ID
    pass

# End of file