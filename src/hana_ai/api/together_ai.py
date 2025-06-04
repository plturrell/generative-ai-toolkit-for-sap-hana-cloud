"""
Together.ai integration for GPU processing in the SAP HANA AI Toolkit.

This module provides a client for Together.ai's API, allowing access to 
powerful GPU resources for model inference and embeddings.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
import httpx
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Together.ai API constants
TOGETHER_API_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_CHAT_ENDPOINT = "/chat/completions"
TOGETHER_COMPLETIONS_ENDPOINT = "/completions"
TOGETHER_EMBEDDINGS_ENDPOINT = "/embeddings"
TOGETHER_MODELS_ENDPOINT = "/models"

# Default timeout for API calls
DEFAULT_TIMEOUT = 60.0

class TogetherAIConfig(BaseModel):
    """Configuration for Together.ai API."""
    api_key: str = Field(..., description="Together.ai API key")
    default_model: str = Field("meta-llama/Llama-2-70b-chat-hf", 
                              description="Default model for completions")
    default_embedding_model: str = Field("togethercomputer/m2-bert-80M-8k-retrieval", 
                                        description="Default model for embeddings")
    timeout: float = Field(DEFAULT_TIMEOUT, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries for API calls")

class TogetherAIClient:
    """Client for the Together.ai API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ):
        """
        Initialize the Together.ai client.
        
        Args:
            api_key: Together.ai API key. Defaults to TOGETHER_API_KEY environment variable.
            default_model: Default model for completions.
            default_embedding_model: Default model for embeddings.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for API calls.
        """
        # Initialize configuration from environment variables and parameters
        self.config = TogetherAIConfig(
            api_key=api_key or os.environ.get("TOGETHER_API_KEY", ""),
            default_model=default_model or os.environ.get(
                "TOGETHER_DEFAULT_MODEL", "meta-llama/Llama-2-70b-chat-hf"
            ),
            default_embedding_model=default_embedding_model or os.environ.get(
                "TOGETHER_DEFAULT_EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval"
            ),
            timeout=timeout or float(os.environ.get("TOGETHER_TIMEOUT", DEFAULT_TIMEOUT)),
            max_retries=max_retries or int(os.environ.get("TOGETHER_MAX_RETRIES", "3"))
        )
        
        # Check if API key is provided
        if not self.config.api_key:
            logger.warning(
                "Together.ai API key not provided. Set the TOGETHER_API_KEY environment variable "
                "or pass it directly to the constructor."
            )
        
        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "POST", 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the Together.ai API.
        
        Args:
            endpoint: API endpoint to call.
            method: HTTP method to use.
            data: Request data.
            params: Query parameters.
            
        Returns:
            API response as a dictionary.
            
        Raises:
            Exception: If the API call fails.
        """
        url = f"{TOGETHER_API_BASE_URL}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.client.get(url, params=params)
                else:
                    response = self.client.post(url, json=data, params=params)
                
                response.raise_for_status()
                return response.json()
            
            except httpx.HTTPStatusError as e:
                error_message = f"Together.ai API error: {e.response.status_code} - {e.response.text}"
                
                # If we've tried max_retries times, raise the exception
                if attempt == self.config.max_retries - 1:
                    logger.error(error_message)
                    raise Exception(error_message) from e
                
                logger.warning(f"{error_message}. Retrying ({attempt + 1}/{self.config.max_retries})...")
                
            except httpx.RequestError as e:
                error_message = f"Together.ai request error: {str(e)}"
                
                # If we've tried max_retries times, raise the exception
                if attempt == self.config.max_retries - 1:
                    logger.error(error_message)
                    raise Exception(error_message) from e
                
                logger.warning(f"{error_message}. Retrying ({attempt + 1}/{self.config.max_retries})...")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from Together.ai.
        
        Returns:
            List of available models with their details.
        """
        response = self._make_request(TOGETHER_MODELS_ENDPOINT, method="GET")
        return response.get("data", [])
    
    def generate_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            model: Model to use for embeddings. Defaults to default_embedding_model.
            
        Returns:
            Dictionary containing the embeddings and related metadata.
        """
        model = model or self.config.default_embedding_model
        
        data = {
            "input": texts,
            "model": model
        }
        
        return self._make_request(TOGETHER_EMBEDDINGS_ENDPOINT, data=data)
    
    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Text prompt to complete.
            model: Model to use for completion. Defaults to default_model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            presence_penalty: Penalty for token presence.
            stop: List of strings that signal the end of completion.
            
        Returns:
            Dictionary containing the completion and related metadata.
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
        if stop:
            data["stop"] = stop
        
        return self._make_request(TOGETHER_COMPLETIONS_ENDPOINT, data=data)
    
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: Model to use for completion. Defaults to default_model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            presence_penalty: Penalty for token presence.
            stop: List of strings that signal the end of completion.
            
        Returns:
            Dictionary containing the chat completion and related metadata.
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
        if stop:
            data["stop"] = stop
        
        return self._make_request(TOGETHER_CHAT_ENDPOINT, data=data)
    
    def close(self):
        """Close the HTTP client session."""
        if self.client:
            self.client.close()
            
    def __del__(self):
        """Ensure the HTTP client is closed when the object is garbage collected."""
        self.close()


# Initialize the Together.ai client when the module is imported
# to ensure it's ready for use in other modules
default_client = None

def get_together_ai_client() -> TogetherAIClient:
    """
    Get the default Together.ai client instance.
    
    Returns:
        TogetherAIClient instance.
    """
    global default_client
    
    if default_client is None:
        # Only initialize if the API key is set
        api_key = os.environ.get("TOGETHER_API_KEY")
        if api_key:
            default_client = TogetherAIClient(api_key=api_key)
        else:
            logger.warning(
                "Together.ai API key not set. Set the TOGETHER_API_KEY environment variable "
                "to use the default client."
            )
            default_client = TogetherAIClient()
    
    return default_client