"""
Tests for the API main entry point.
"""
import sys
import pytest
from unittest.mock import patch, MagicMock

def test_main_function():
    """Test the main function that starts the uvicorn server."""
    from hana_ai.api.__main__ import main
    
    with patch("uvicorn.run") as mock_run:
        # Run the main function
        with patch.object(sys, "exit"):  # Prevent actual exit
            main()
        
        # Check that uvicorn.run was called with correct parameters
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        
        # Check positional and keyword arguments
        assert args[0] == "hana_ai.api.app:app"
        assert kwargs["host"] is not None
        assert kwargs["port"] is not None
        assert isinstance(kwargs["port"], int)
        assert kwargs["log_level"] is not None

def test_logging_configuration():
    """Test that logging is configured correctly."""
    with patch("logging.basicConfig") as mock_logging:
        # Import the module to trigger logging configuration
        import hana_ai.api.__main__
        
        # Check that logging was configured
        mock_logging.assert_called_once()
        
        # Verify log format
        call_kwargs = mock_logging.call_args[1]
        assert "format" in call_kwargs
        assert "level" in call_kwargs

@pytest.mark.skipif(sys.platform != "linux", reason="Integration test requires Linux")
def test_uvicorn_integration():
    """Test that uvicorn can actually load the application."""
    import subprocess
    import time
    import requests
    import signal
    
    # Start the server in a subprocess
    process = subprocess.Popen(
        ["python", "-m", "hana_ai.api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"API_PORT": "8765"}  # Use a non-default port for testing
    )
    
    try:
        # Give the server a moment to start
        time.sleep(2)
        
        # Try to connect to the server
        try:
            response = requests.get("http://localhost:8765/")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
        except requests.ConnectionError:
            pytest.fail("Failed to connect to the server")
    finally:
        # Terminate the server
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
        
        # Check for any errors in server output
        stderr = process.stderr.read().decode("utf-8")
        if stderr and "Error" in stderr:
            pytest.fail(f"Server reported errors: {stderr}")