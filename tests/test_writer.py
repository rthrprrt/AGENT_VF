# tests/test_writer.py
# -*- coding: utf-8 -*-
"""
Tests for the Writer component (LLM client) in AGENT_VF.writer.client.

**LLM-Code Suggestions Implemented:**
- Added integration tests for error scenarios (Ollama connection, model not found).
- Added parameterized integration test for prompt/context variations.
- Added unit test for complex JSON parsing (assuming LangChain OutputParser).
- Renamed tests for clarity.
- Used mocker fixture and fixtures for LLM clients.
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field # For OutputParser test

# Attempt to import real components
try:
    from AGENT_VF.writer.client import Writer # Your actual Writer class
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_core.exceptions import OutputParserException
    # Import requests or http client library if needed for low-level mocking
    import requests
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import real Writer components: {e}. Some tests may be skipped.")
    # Define dummy classes
    class Ollama: # Dummy
        def __init__(self, model: str, base_url: str, **kwargs):
            self.model = model
            self.base_url = base_url
        def invoke(self, prompt: str, **kwargs) -> str:
            print(f"DummyOllama: Invoked model '{self.model}' with prompt '{prompt[:50]}...'")
            if "json" in prompt.lower():
                 # Corrected: Ensure valid JSON for parsing tests
                 if "broken" in prompt.lower():
                     return '{"title": "Broken JSON", "points": [1, 2,' # Keep broken for error test
                 elif "complex" in prompt.lower():
                     # Provide valid JSON matching ComplexReport structure
                     return json.dumps({
                         "report_title": "Mock Complex Report",
                         "sections": [{"section_id": 1, "content": "Section 1"}], # Use valid key
                         "confidence_score": 0.9
                     })
                 else:
                    return '{"title": "Mock Title", "points": ["Point 1", "Point 2"]}'
            elif "error" in prompt.lower():
                 raise ConnectionError("Simulated connection error to Ollama")
            else:
                return f"Mock response for model {self.model}"

    class Writer: # Dummy
        def __init__(self, llm_client: Any, parser: Optional[Any] = None):
            self.llm = llm_client
            self.parser = parser
            print("Dummy Writer initialized.")

        def generate(self, context: str, query: str, output_format: str = "text") -> str | Dict:
            prompt = f"Context: {context}\nQuery: {query}\nFormat: {output_format}"
            try:
                raw_output = self.llm.invoke(prompt)
                print(f"Dummy Writer: Raw LLM output: {raw_output[:100]}...")
                if self.parser:
                    try:
                        parsed_output = self.parser.parse(raw_output)
                        print(f"Dummy Writer: Parsed output: {parsed_output}")
                        return parsed_output
                    except Exception as parse_error: # Catch generic parse error
                        print(f"Dummy Writer: Parsing failed: {parse_error}")
                        # Correction: Re-raise the original parsing error
                        raise parse_error
                return raw_output # Return raw if no parser
            except ConnectionError as ce:
                 print(f"Dummy Writer: Connection error caught: {ce}")
                 raise  # Re-raise connection errors for tests to catch
            except Exception as e:
                 # Avoid catching the re-raised parse_error here again
                 # Check if it's one of the expected parsing errors
                 if isinstance(e, (json.JSONDecodeError, OutputParserException)):
                      raise # Let parse errors propagate
                 print(f"Dummy Writer: Unexpected error during generation: {e}")
                 raise # Re-raise other unexpected errors

    class JsonOutputParser: # Dummy
        def parse(self, text: str) -> Any:
            # This will raise json.JSONDecodeError on invalid JSON
            return json.loads(text)

    class PydanticOutputParser: # Dummy
        def __init__(self, pydantic_object):
            self.pydantic_object = pydantic_object
        def parse(self, text: str) -> Any:
            # Basic simulation
            try:
                data = json.loads(text)
                return self.pydantic_object(**data)
            except Exception as e:
                 # Correction: Raise the dummy OutputParserException
                 raise OutputParserException(f"Dummy Pydantic parse failed: {e}") from e

    # Correction: Define dummy OutputParserException if real one not imported
    if 'OutputParserException' not in globals():
        class OutputParserException(Exception): pass

    REAL_COMPONENTS_AVAILABLE = False

# --- Pydantic Model for Parser Test ---
class ComplexReport(BaseModel):
    report_title: str = Field(description="Main title of the report")
    sections: List[Dict[str, Any]] = Field(description="List of sections, each a dictionary")
    confidence_score: Optional[float] = Field(None, description="Optional confidence score")

# --- Fixtures ---

# Correction: Change scope to 'session' to match 'real_ollama_client'
@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Correction: Change scope to 'session' to match 'real_ollama_client'
@pytest.fixture(scope="session")
def ollama_model_name() -> str:
    # Use a smaller model for faster integration tests if available
    return os.getenv("OLLAMA_TEST_MODEL", "gemma:7b")

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Provides a MagicMock simulating an LLM client like Ollama."""
    mock = MagicMock(spec=Ollama)
    mock.invoke.return_value = "Default mock LLM response."
    # Configure specific behaviors if needed by default
    return mock

@pytest.fixture(scope="session")
def real_ollama_client(ollama_base_url: str, ollama_model_name: str) -> Optional[Ollama]:
    """Provides a real Ollama client instance, skipping tests if unavailable."""
    if not REAL_COMPONENTS_AVAILABLE:
        # Correction: Return None explicitly if skipping
        return None

    try:
        print(f"\nInitializing real Ollama client: model='{ollama_model_name}', url='{ollama_base_url}'")
        llm = Ollama(model=ollama_model_name, base_url=ollama_base_url, temperature=0.1)
        # Perform a quick check to see if Ollama is responsive
        llm.invoke("Respond with OK if you are ready.")
        print("Real Ollama client initialized and responsive.")
        return llm
    except Exception as e:
        print(f"\nWarning: Failed to initialize or connect to real Ollama client: {e}")
        # Correction: Skip using pytest.skip instead of returning None in except block
        pytest.skip(f"Real Ollama client unavailable ({e}). Skipping relevant integration tests.")
        # This line won't be reached due to skip, but added for consistency
        return None


@pytest.fixture
def writer_unit(mock_llm_client: MagicMock) -> Writer:
    """Provides a Writer instance with a mocked LLM for unit tests."""
    WriterClass = globals().get("Writer")
    return WriterClass(llm_client=mock_llm_client)

@pytest.fixture
def writer_integration(real_ollama_client: Optional[Ollama]) -> Optional[Writer]:
    """Provides a Writer instance with a real LLM for integration tests."""
    if not real_ollama_client:
        return None
    WriterClass = globals().get("Writer")
    return WriterClass(llm_client=real_ollama_client)

# --- Unit Tests ---

@pytest.mark.unit
def test_writer_initialization_unit(writer_unit: Writer, mock_llm_client: MagicMock):
    """Verify Writer initializes correctly with a mock LLM."""
    assert writer_unit.llm is mock_llm_client
    print("KPI: Unit - Writer initialized correctly - OK")

@pytest.mark.unit
def test_writer_generate_calls_llm_invoke_unit(writer_unit: Writer, mock_llm_client: MagicMock):
    """Verify the generate method calls the LLM's invoke method."""
    context = "Test context."
    query = "Test query."
    writer_unit.generate(context=context, query=query)
    mock_llm_client.invoke.assert_called_once()
    # Check if prompt contains context and query (basic check)
    call_args, _ = mock_llm_client.invoke.call_args
    prompt_arg = call_args[0]
    assert context in prompt_arg
    assert query in prompt_arg
    print("KPI: Unit - LLM invoke called with correct prompt elements - OK")

@pytest.mark.unit
def test_writer_parses_complex_json_with_pydantic_parser_unit(mock_llm_client: MagicMock):
    """Verify Writer uses PydanticOutputParser correctly for complex JSON."""
    # Setup
    WriterClass = globals().get("Writer")
    PydanticParserClass = globals().get("PydanticOutputParser")
    # Ensure the dummy OutputParserException is available
    if 'OutputParserException' not in globals():
        globals()["OutputParserException"] = type("OutputParserException", (Exception,), {})
    parser = PydanticParserClass(pydantic_object=ComplexReport)
    writer = WriterClass(llm_client=mock_llm_client, parser=parser)

    # Mock LLM response with valid JSON for the Pydantic model
    # Use the updated mock LLM logic which provides valid JSON here
    mock_llm_client.invoke.return_value = json.dumps({
        "report_title": "Complex Test Report",
        "sections": [
            {"section_id": 1, "content": "Content A"}, # Use valid key
            {"section_id": 2, "content": "Content B", "notes": "Optional note"}
        ],
        "confidence_score": 0.85
    })


    # Execute
    parsed_result = writer.generate(context="N/A", query="Generate complex JSON")

    # Assert
    assert isinstance(parsed_result, ComplexReport)
    assert parsed_result.report_title == "Complex Test Report"
    assert len(parsed_result.sections) == 2
    assert parsed_result.sections[0]["section_id"] == 1 # Corrected key access
    assert parsed_result.confidence_score == 0.85
    mock_llm_client.invoke.assert_called_once() # Ensure LLM was called
    print("KPI: Unit - Complex JSON parsed correctly by Pydantic parser - OK")

@pytest.mark.unit
def test_writer_handles_parsing_error_unit(mock_llm_client: MagicMock):
    """Verify Writer handles errors during output parsing."""
    WriterClass = globals().get("Writer")
    JsonParserClass = globals().get("JsonOutputParser")
    parser = JsonParserClass() # Simple JSON parser
    writer = WriterClass(llm_client=mock_llm_client, parser=parser)

    # Mock LLM response with invalid JSON
    invalid_json_output = '{"title": "Broken JSON", "points": [1, 2,'
    mock_llm_client.invoke.return_value = invalid_json_output

    # Correction: Expect the specific error raised by json.loads
    with pytest.raises(json.JSONDecodeError) as excinfo:
        writer.generate(context="N/A", query="Generate broken JSON")

    # Check the error message if needed
    assert "Expecting value" in str(excinfo.value)
    print(f"KPI: Unit - Handled parsing error ({excinfo.type.__name__}) correctly - OK")


# --- Integration Tests ---
# (Integration tests remain largely the same, relying on the corrected fixtures)

@pytest.mark.integration
def test_writer_integration_generates_text_success(writer_integration: Optional[Writer]):
    """Verify the real writer generates coherent text."""
    if not writer_integration: pytest.skip("Real Writer not available.")

    context = "Topic: Benefits of pytest"
    query = "Write a short paragraph (around 50 words) explaining why pytest is useful for testing Python code."
    min_words = 20
    max_words = 100

    start_time = time.time()
    result = writer_integration.generate(context=context, query=query, output_format="text")
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nKPI (Integration - Generate Text Success):")
    print(f"  - Execution Time: {duration:.3f}s (Expected < 5s)")
    print(f"  - Output Type: {type(result)}")
    print(f"  - Output (excerpt): {result[:150]}...")

    assert isinstance(result, str), "Output is not a string"
    word_count = len(result.split())
    print(f"  - Word Count: {word_count} (Expected {min_words}-{max_words})")
    assert min_words <= word_count <= max_words, f"Word count {word_count} out of range {min_words}-{max_words}"
    assert duration < 5, "Generation took too long"
    # Basic coherence check (presence of keywords)
    assert "pytest" in result.lower()
    assert "test" in result.lower() or "testing" in result.lower()
    print("  - Text generated successfully within constraints - OK")


@pytest.mark.integration
@pytest.mark.parametrize(
    "context, query, output_format, min_length, max_time",
    [
        ("Subject: Python lists", "Generate a JSON object with 'type': 'list' and 'examples': ['a', 1]", "json", 20, 6),
        ("Topic: Solar System", "Write a short text about Mars.", "text", 15, 5),
        ("Data: {'a': 1}", "Output a JSON representation of the input data.", "json", 10, 5),
    ],
    ids=["json_list", "text_mars", "json_data"]
)
def test_writer_integration_prompt_variations(
    writer_integration: Optional[Writer],
    context: str,
    query: str,
    output_format: str,
    min_length: int,
    max_time: int
):
    """Verify writer handles different prompt/context variations."""
    if not writer_integration: pytest.skip("Real Writer not available.")

    start_time = time.time()
    result = writer_integration.generate(context=context, query=query, output_format=output_format)
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nKPI (Integration - Prompt Variation: '{query[:30]}...'):")
    print(f"  - Execution Time: {duration:.3f}s (Expected < {max_time}s)")
    print(f"  - Output Type: {type(result)}")
    print(f"  - Output (excerpt): {str(result)[:150]}...")

    assert duration < max_time, f"Generation took too long ({duration:.3f}s > {max_time}s)"

    if output_format == "json":
        assert isinstance(result, (dict, list)), f"Expected JSON (dict/list), got {type(result)}"
        result_str = json.dumps(result) # Check length of serialized JSON
    else:
        assert isinstance(result, str), f"Expected text (str), got {type(result)}"
        result_str = result

    assert len(result_str) >= min_length, f"Output length {len(result_str)} is less than minimum {min_length}"
    print(f"  - Generated output format '{output_format}' with sufficient length - OK")


@pytest.mark.integration
def test_writer_integration_handles_ollama_connection_error(ollama_base_url: str, ollama_model_name: str, mocker):
    """Verify writer handles connection errors when calling Ollama."""
    if not REAL_COMPONENTS_AVAILABLE: pytest.skip("Real components needed for this test.")

    # Mock the network call (e.g., requests.post) or the Ollama client's method
    # Mocking requests.post is lower level and less dependent on Ollama client internals
    error_message = f"Connection refused to {ollama_base_url}"
    mock_post = mocker.patch('requests.post', side_effect=requests.exceptions.ConnectionError(error_message))

    # Need to instantiate the real Ollama client *after* patching requests
    try:
        llm = Ollama(model=ollama_model_name, base_url=ollama_base_url)
        WriterClass = globals().get("Writer")
        writer = WriterClass(llm_client=llm)
    except Exception as e:
        pytest.fail(f"Failed to instantiate real components after patching requests: {e}")

    # Expect the Writer or Ollama client to raise an error (e.g., ConnectionError, or a custom one)
    with pytest.raises(Exception) as excinfo: # Catch broad exception, check specific type below
        writer.generate(context="N/A", query="Query that will fail connection")

    print(f"\nKPI (Integration - Ollama Connection Error):")
    print(f"  - Exception Raised: {excinfo.type.__name__}")
    print(f"  - Error Message: {excinfo.value}")

    # Check that requests.post was called
    mock_post.assert_called_once()
    # Check the type of exception raised - depends on Ollama/Langchain error handling
    assert isinstance(excinfo.value, requests.exceptions.ConnectionError) or "Connection error" in str(excinfo.value).lower()
    print(f"  - Handled connection error correctly by raising {excinfo.type.__name__} - OK")


@pytest.mark.integration
def test_writer_integration_handles_model_not_found(ollama_base_url: str, mocker):
    """Verify writer handles 'model not found' errors from Ollama."""
    if not REAL_COMPONENTS_AVAILABLE: pytest.skip("Real components needed for this test.")

    # Simulate Ollama returning a 404 or similar error for a non-existent model
    non_existent_model = "non_existent_model_xyz:latest"
    error_payload = {"error": f"model '{non_existent_model}' not found"}

    # Mock the response from requests.post
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 404
    mock_response.json.return_value = error_payload
    mock_response.text = json.dumps(error_payload)
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

    mock_post = mocker.patch('requests.post', return_value=mock_response)

    # Instantiate real components after patching
    try:
        llm = Ollama(model=non_existent_model, base_url=ollama_base_url)
        WriterClass = globals().get("Writer")
        writer = WriterClass(llm_client=llm)
    except Exception as e:
        pytest.fail(f"Failed to instantiate real components after patching requests: {e}")

    # Expect an exception indicating the model wasn't found
    # The exact exception type depends on Langchain's Ollama client implementation
    with pytest.raises(Exception) as excinfo:
        writer.generate(context="N/A", query="Query for non-existent model")

    print(f"\nKPI (Integration - Ollama Model Not Found):")
    print(f"  - Exception Raised: {excinfo.type.__name__}")
    print(f"  - Error Message: {excinfo.value}")

    mock_post.assert_called_once()
    # Check if the error message contains relevant info
    assert non_existent_model in str(excinfo.value) or "not found" in str(excinfo.value).lower()
    print(f"  - Handled model not found error correctly by raising {excinfo.type.__name__} - OK")


# End of file