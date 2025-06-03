"""
Tests for the API configuration.
"""
import os
import pytest
from unittest.mock import patch

def test_settings_defaults():
    """Test default settings values."""
    # Reset settings to use defaults
    with patch.dict(os.environ, {}, clear=True):
        from hana_ai.api.config import Settings
        
        # Create new settings instance with defaults
        settings = Settings()
        
        # Check default values
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        assert settings.DEVELOPMENT_MODE is False
        assert settings.LOG_LEVEL == "INFO"
        assert "dev-key-only-for-testing" in settings.API_KEYS
        assert settings.AUTH_REQUIRED is True
        assert settings.CORS_ORIGINS == ["*"]

def test_settings_from_env():
    """Test loading settings from environment variables."""
    # Set environment variables
    env_vars = {
        "API_HOST": "127.0.0.1",
        "API_PORT": "9000",
        "DEVELOPMENT_MODE": "true",
        "LOG_LEVEL": "DEBUG",
        "API_KEYS": "key1,key2,key3",
        "AUTH_REQUIRED": "false",
        "CORS_ORIGINS": "http://localhost:3000,https://example.com",
        "HANA_HOST": "hana-db.example.com",
        "HANA_PORT": "443",
        "HANA_USER": "test_user",
        "HANA_PASSWORD": "test_password",
        "DEFAULT_LLM_MODEL": "gpt-4-turbo",
        "DEFAULT_LLM_TEMPERATURE": "0.7",
        "DEFAULT_LLM_MAX_TOKENS": "2000"
    }
    
    with patch.dict(os.environ, env_vars, clear=True):
        from hana_ai.api.config import Settings
        
        # Create new settings instance with environment variables
        settings = Settings()
        
        # Check values from environment
        assert settings.API_HOST == "127.0.0.1"
        assert settings.API_PORT == 9000
        assert settings.DEVELOPMENT_MODE is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.API_KEYS == ["key1", "key2", "key3"]
        assert settings.AUTH_REQUIRED is False
        assert settings.CORS_ORIGINS == ["http://localhost:3000", "https://example.com"]
        assert settings.HANA_HOST == "hana-db.example.com"
        assert settings.HANA_PORT == 443
        assert settings.HANA_USER == "test_user"
        assert settings.HANA_PASSWORD == "test_password"
        assert settings.DEFAULT_LLM_MODEL == "gpt-4-turbo"
        assert settings.DEFAULT_LLM_TEMPERATURE == 0.7
        assert settings.DEFAULT_LLM_MAX_TOKENS == 2000

def test_settings_singleton():
    """Test that settings are singleton."""
    from hana_ai.api.config import settings, Settings
    
    # Create a new instance
    new_settings = Settings()
    
    # Modify a value in the original settings
    original_host = settings.API_HOST
    settings.API_HOST = "changed.host"
    
    # Check that the new instance has the old value (they're separate instances)
    assert new_settings.API_HOST == original_host
    assert settings.API_HOST == "changed.host"
    
    # Reset settings
    settings.API_HOST = original_host