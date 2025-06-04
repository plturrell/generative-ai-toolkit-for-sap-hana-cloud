"""
Backend manager for SAP HANA AI Toolkit.

This module provides a central management system for multiple backends
including NVIDIA LaunchPad, Together.ai, and CPU-only processing.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar
import importlib
import time
from enum import Enum
import threading

from .backend_config import backend_config, BackendType
from .backend_router import (
    with_backend_router,
    get_backend_status,
    mark_backend_available,
    mark_backend_unavailable,
)
from .failover import with_retry, with_circuit_breaker, with_timeout, failover_manager

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic result type
T = TypeVar('T')

class BackendManager:
    """
    Manager for multiple backend types in the SAP HANA AI Toolkit.
    
    This class provides a centralized way to manage and use multiple backends
    for AI operations, including NVIDIA, Together.ai, and CPU processing.
    """
    
    def __init__(self):
        """Initialize the backend manager."""
        self.initialized_backends = set()
        self._initialization_lock = threading.RLock()
        
        # Register with failover manager
        self._register_with_failover_manager()
        
        # Initialize default backends
        self._initialize_default_backends()
    
    def _register_with_failover_manager(self):
        """Register backends with the failover manager."""
        # Register NVIDIA backend
        failover_manager.register_service(
            service_name="nvidia_backend",
            health_check=self._check_nvidia_health,
        )
        
        # Register Together.ai backend
        failover_manager.register_service(
            service_name="together_ai_backend",
            health_check=self._check_together_ai_health,
        )
        
        # Register CPU backend
        failover_manager.register_service(
            service_name="cpu_backend",
            health_check=lambda: True,  # CPU is always available
        )
    
    def _check_nvidia_health(self) -> bool:
        """
        Check if NVIDIA backend is healthy.
        
        Returns:
            bool: True if healthy, False otherwise.
        """
        if not backend_config.nvidia.enabled:
            return False
        
        try:
            # Try to import torch and check CUDA availability
            import torch
            if not torch.cuda.is_available():
                return False
            
            # Try a simple operation on GPU
            device = torch.device("cuda")
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            y = x + x
            
            return True
        except Exception as e:
            logger.warning(f"NVIDIA health check failed: {str(e)}")
            return False
    
    def _check_together_ai_health(self) -> bool:
        """
        Check if Together.ai backend is healthy.
        
        Returns:
            bool: True if healthy, False otherwise.
        """
        if not backend_config.together_ai.enabled or not backend_config.together_ai.api_key:
            return False
        
        try:
            # Try to import and initialize the Together.ai client
            module_name = "src.hana_ai.api.together_ai"
            try:
                module = importlib.import_module(module_name)
                client = module.get_together_ai_client()
                
                # Try a simple API call
                models = client.get_available_models()
                
                return True
            except ImportError:
                # Try alternative import path
                module_name = "hana_ai.api.together_ai"
                module = importlib.import_module(module_name)
                client = module.get_together_ai_client()
                
                # Try a simple API call
                models = client.get_available_models()
                
                return True
        except Exception as e:
            logger.warning(f"Together.ai health check failed: {str(e)}")
            return False
    
    def _initialize_default_backends(self):
        """Initialize default backends based on configuration."""
        # Get active backends
        active_backends = backend_config.determine_active_backends()
        
        # Initialize active backends
        for backend_type in active_backends:
            self.initialize_backend(backend_type)
    
    def initialize_backend(self, backend_type: BackendType) -> bool:
        """
        Initialize a specific backend.
        
        Args:
            backend_type: The backend type to initialize.
            
        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        with self._initialization_lock:
            # Skip if already initialized
            if backend_type in self.initialized_backends:
                return True
            
            try:
                logger.info(f"Initializing backend: {backend_type}")
                
                # Initialize NVIDIA backend
                if backend_type == BackendType.NVIDIA:
                    if not backend_config.nvidia.enabled:
                        logger.warning("NVIDIA backend is disabled in configuration")
                        return False
                    
                    # Try to import torch and check CUDA availability
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA is not available for NVIDIA backend")
                        return False
                    
                    # Initialize TensorRT if enabled
                    if backend_config.nvidia.enable_tensorrt:
                        try:
                            import tensorrt
                            logger.info("TensorRT initialized successfully")
                        except ImportError:
                            logger.warning("TensorRT is not available")
                    
                    # Initialize Flash Attention if enabled
                    if backend_config.nvidia.enable_flash_attention:
                        try:
                            import flash_attn
                            logger.info("Flash Attention initialized successfully")
                        except ImportError:
                            logger.warning("Flash Attention is not available")
                    
                    # Initialize Transformer Engine if enabled
                    if backend_config.nvidia.enable_transformer_engine:
                        try:
                            import transformer_engine
                            logger.info("Transformer Engine initialized successfully")
                        except ImportError:
                            logger.warning("Transformer Engine is not available")
                    
                    # Mark backend as initialized
                    self.initialized_backends.add(backend_type)
                    mark_backend_available(backend_type)
                    return True
                
                # Initialize Together.ai backend
                elif backend_type == BackendType.TOGETHER_AI:
                    if not backend_config.together_ai.enabled:
                        logger.warning("Together.ai backend is disabled in configuration")
                        return False
                    
                    if not backend_config.together_ai.api_key:
                        logger.warning("Together.ai API key is not set")
                        return False
                    
                    # Try to import and initialize the Together.ai client
                    try:
                        module_name = "src.hana_ai.api.together_ai"
                        module = importlib.import_module(module_name)
                    except ImportError:
                        # Try alternative import path
                        module_name = "hana_ai.api.together_ai"
                        module = importlib.import_module(module_name)
                    
                    # Get client
                    client = module.get_together_ai_client()
                    
                    # Mark backend as initialized
                    self.initialized_backends.add(backend_type)
                    mark_backend_available(backend_type)
                    return True
                
                # Initialize CPU backend
                elif backend_type == BackendType.CPU:
                    if not backend_config.cpu.enabled:
                        logger.warning("CPU backend is disabled in configuration")
                        return False
                    
                    # CPU backend is always available
                    self.initialized_backends.add(backend_type)
                    mark_backend_available(backend_type)
                    return True
                
                else:
                    logger.warning(f"Unknown backend type: {backend_type}")
                    return False
            
            except Exception as e:
                logger.error(f"Failed to initialize backend {backend_type}: {str(e)}")
                mark_backend_unavailable(backend_type, error=e)
                return False
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get the status of all backends.
        
        Returns:
            Dict[str, Any]: The status of all backends.
        """
        # Get backend status from router
        router_status = get_backend_status()
        
        # Add initialized backends information
        router_status["initialized_backends"] = list(self.initialized_backends)
        
        # Add failover manager status
        router_status["failover"] = failover_manager.get_service_status()
        
        return router_status
    
    def wrap_with_backend_routing(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Wrap a function with backend routing.
        
        Args:
            func: The function to wrap.
            
        Returns:
            Callable: The wrapped function.
        """
        return with_backend_router()(func)
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        backend: Optional[BackendType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the configured backends.
        
        Args:
            prompt: The prompt to generate text from.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            backend: Specific backend to use. If None, uses the router.
            **kwargs: Additional parameters for text generation.
            
        Returns:
            Dict[str, Any]: The generated text and metadata.
        """
        # Define backend-specific functions
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def nvidia_generate():
            # Implementation for NVIDIA backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.NVIDIA):
                    raise Exception("Failed to initialize NVIDIA backend")
                
                # Import the necessary modules
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Load model and tokenizer
                model_name = kwargs.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                
                # Generate text
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() if k not in ["model_name"]}
                )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "text": generated_text,
                    "backend": "nvidia",
                    "model": model_name,
                }
            except Exception as e:
                logger.error(f"NVIDIA backend failed to generate text: {str(e)}")
                raise
        
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def together_ai_generate():
            # Implementation for Together.ai backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.TOGETHER_AI):
                    raise Exception("Failed to initialize Together.ai backend")
                
                # Import the Together.ai client
                try:
                    module_name = "src.hana_ai.api.together_ai"
                    module = importlib.import_module(module_name)
                except ImportError:
                    # Try alternative import path
                    module_name = "hana_ai.api.together_ai"
                    module = importlib.import_module(module_name)
                
                # Get client
                client = module.get_together_ai_client()
                
                # Generate text
                model_name = kwargs.get("model_name", backend_config.together_ai.default_model)
                response = client.generate_completion(
                    prompt=prompt,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() if k not in ["model_name"]}
                )
                
                return {
                    "text": response.get("choices", [{}])[0].get("text", ""),
                    "backend": "together_ai",
                    "model": model_name,
                }
            except Exception as e:
                logger.error(f"Together.ai backend failed to generate text: {str(e)}")
                raise
        
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def cpu_generate():
            # Implementation for CPU backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.CPU):
                    raise Exception("Failed to initialize CPU backend")
                
                # Use a simpler model for CPU
                try:
                    from transformers import pipeline
                    
                    # Use a smaller model suitable for CPU
                    model_name = kwargs.get("model_name", backend_config.cpu.default_model)
                    
                    # Create pipeline
                    generator = pipeline("text-generation", model=model_name)
                    
                    # Generate text
                    response = generator(
                        prompt,
                        max_length=len(prompt.split()) + max_tokens,
                        temperature=temperature,
                        **{k: v for k, v in kwargs.items() if k not in ["model_name"]}
                    )
                    
                    return {
                        "text": response[0]["generated_text"],
                        "backend": "cpu",
                        "model": model_name,
                    }
                except Exception as e:
                    # Fallback to a very simple generation if transformers fails
                    logger.warning(f"CPU transformers generation failed, using simple fallback: {str(e)}")
                    return {
                        "text": f"{prompt}\n\nI'm sorry, but I'm currently running in CPU-only mode with limited capabilities.",
                        "backend": "cpu",
                        "model": "fallback",
                    }
            except Exception as e:
                logger.error(f"CPU backend failed to generate text: {str(e)}")
                raise
        
        # If specific backend is requested, use it directly
        if backend:
            if backend == BackendType.NVIDIA:
                return nvidia_generate()
            elif backend == BackendType.TOGETHER_AI:
                return together_ai_generate()
            elif backend == BackendType.CPU:
                return cpu_generate()
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        # Otherwise, use the backend router
        request = with_backend_router(
            nvidia_func=nvidia_generate,
            together_ai_func=together_ai_generate,
            cpu_func=cpu_generate,
        )
        
        return request(prompt, max_tokens, temperature, **kwargs)
    
    def generate_embeddings(
        self,
        texts: List[str],
        backend: Optional[BackendType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings using the configured backends.
        
        Args:
            texts: The texts to generate embeddings for.
            backend: Specific backend to use. If None, uses the router.
            **kwargs: Additional parameters for embedding generation.
            
        Returns:
            Dict[str, Any]: The generated embeddings and metadata.
        """
        # Define backend-specific functions
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def nvidia_generate_embeddings():
            # Implementation for NVIDIA backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.NVIDIA):
                    raise Exception("Failed to initialize NVIDIA backend")
                
                # Import the necessary modules
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                # Load model and tokenizer
                model_name = kwargs.get("model_name", "sentence-transformers/all-mpnet-base-v2")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to("cuda")
                
                # Generate embeddings
                embeddings = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Use mean pooling for sentence embeddings
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]
                    embeddings.append(embedding)
                
                return {
                    "embeddings": embeddings,
                    "backend": "nvidia",
                    "model": model_name,
                }
            except Exception as e:
                logger.error(f"NVIDIA backend failed to generate embeddings: {str(e)}")
                raise
        
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def together_ai_generate_embeddings():
            # Implementation for Together.ai backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.TOGETHER_AI):
                    raise Exception("Failed to initialize Together.ai backend")
                
                # Import the Together.ai client
                try:
                    module_name = "src.hana_ai.api.together_ai"
                    module = importlib.import_module(module_name)
                except ImportError:
                    # Try alternative import path
                    module_name = "hana_ai.api.together_ai"
                    module = importlib.import_module(module_name)
                
                # Get client
                client = module.get_together_ai_client()
                
                # Generate embeddings
                model_name = kwargs.get("model_name", backend_config.together_ai.default_embedding_model)
                response = client.generate_embeddings(
                    texts=texts,
                    model=model_name,
                )
                
                return {
                    "embeddings": response.get("data", [{}])[0].get("embedding", []),
                    "backend": "together_ai",
                    "model": model_name,
                }
            except Exception as e:
                logger.error(f"Together.ai backend failed to generate embeddings: {str(e)}")
                raise
        
        @with_retry(max_retries=2)
        @with_circuit_breaker(failure_threshold=3)
        @with_timeout(seconds=60.0)
        def cpu_generate_embeddings():
            # Implementation for CPU backend
            try:
                # Ensure backend is initialized
                if not self.initialize_backend(BackendType.CPU):
                    raise Exception("Failed to initialize CPU backend")
                
                # Use a simpler model for CPU
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    # Use a smaller model suitable for CPU
                    model_name = kwargs.get("model_name", backend_config.cpu.default_embedding_model)
                    
                    # Create model
                    model = SentenceTransformer(model_name)
                    
                    # Generate embeddings
                    embeddings = model.encode(texts).tolist()
                    
                    return {
                        "embeddings": embeddings,
                        "backend": "cpu",
                        "model": model_name,
                    }
                except Exception as e:
                    # Fallback to a very simple embedding if sentence_transformers fails
                    logger.warning(f"CPU embeddings generation failed: {str(e)}")
                    # Create random embeddings as a last resort
                    import random
                    embeddings = [[random.random() for _ in range(384)] for _ in range(len(texts))]
                    return {
                        "embeddings": embeddings,
                        "backend": "cpu",
                        "model": "fallback",
                    }
            except Exception as e:
                logger.error(f"CPU backend failed to generate embeddings: {str(e)}")
                raise
        
        # If specific backend is requested, use it directly
        if backend:
            if backend == BackendType.NVIDIA:
                return nvidia_generate_embeddings()
            elif backend == BackendType.TOGETHER_AI:
                return together_ai_generate_embeddings()
            elif backend == BackendType.CPU:
                return cpu_generate_embeddings()
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        # Otherwise, use the backend router
        request = with_backend_router(
            nvidia_func=nvidia_generate_embeddings,
            together_ai_func=together_ai_generate_embeddings,
            cpu_func=cpu_generate_embeddings,
        )
        
        return request(texts, **kwargs)

# Create global backend manager instance
backend_manager = BackendManager()

# Export the backend manager instance
__all__ = ["backend_manager", "BackendManager"]