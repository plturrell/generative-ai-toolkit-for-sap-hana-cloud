#!/usr/bin/env python
"""
Advanced GPU Optimization Example for the Generative AI Toolkit for SAP HANA Cloud
This example demonstrates how to use the NVIDIA GPU optimizations including GPTQ and AWQ
quantization methods, Flash Attention 2, Transformer Engine, and TensorRT integration.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
import torch

# Import toolkit components
from hana_ai.vectorstore import HANAMLinVectorEngine
from hana_ai.agents import HANAMLAgentWithMemory
from hana_ai.api.gpu_utils import get_gpu_info, is_gpu_available
from hana_ai.api.gpu_utils_hopper import HopperOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAP HANA Connection parameters
HOST = os.environ.get("HANA_HOST", "localhost")
PORT = os.environ.get("HANA_PORT", "39015")
USER = os.environ.get("HANA_USER", "SYSTEM")
PASSWORD = os.environ.get("HANA_PASSWORD", "")
SCHEMA = os.environ.get("HANA_SCHEMA", "SYSTEM")

# LLM configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

# GPU Optimization settings
ENABLE_GPU = os.environ.get("ENABLE_GPU_OPTIMIZATIONS", "true").lower() == "true"
QUANT_METHOD = os.environ.get("DEFAULT_QUANT_METHOD", "gptq")  # or "awq"
USE_FLASH_ATTN = os.environ.get("ENABLE_FLASH_ATTENTION", "true").lower() == "true"
USE_TRANSFORMER_ENGINE = os.environ.get("ENABLE_TRANSFORMER_ENGINE", "true").lower() == "true"
INT4_PRECISION = os.environ.get("USE_INT4_PRECISION", "true").lower() == "true"


def benchmark_inference(model: Any, input_text: str, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark inference speed with the given model"""
    input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(input_ids, max_length=20)
    
    # Benchmark
    latencies = []
    token_counts = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100)
        end_time = time.time()
        
        latencies.append(end_time - start_time)
        token_counts.append(output.shape[1])
    
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    tokens_per_second = avg_tokens / avg_latency
    
    return {
        "avg_latency_seconds": avg_latency,
        "tokens_per_second": tokens_per_second,
        "avg_tokens_generated": avg_tokens
    }


def main():
    """Main function demonstrating advanced GPU optimizations"""
    logger.info("Starting advanced GPU optimization example")
    
    # Check for required credentials
    if not PASSWORD or not OPENAI_API_KEY or not HF_API_TOKEN:
        logger.error("Missing required credentials. Please set HANA_PASSWORD, OPENAI_API_KEY, and HF_API_TOKEN environment variables.")
        return
    
    # Check GPU availability
    if not is_gpu_available():
        logger.warning("No GPU detected. Optimizations will not be applied.")
        ENABLE_GPU = False
    else:
        gpu_info = get_gpu_info()
        logger.info(f"GPU detected: {gpu_info.get('name', 'Unknown')} with {gpu_info.get('memory_total', 0)/1024:.1f} GB memory")
    
    try:
        # Connect to SAP HANA
        connection_string = f"hana://{USER}:{PASSWORD}@{HOST}:{PORT}/?schema={SCHEMA}"
        logger.info(f"Connecting to SAP HANA at {HOST}:{PORT} with user {USER}")
        
        # Initialize Hopper optimizer if GPU is available
        model = None
        if ENABLE_GPU:
            logger.info(f"Initializing GPU optimizations with {QUANT_METHOD} quantization")
            optimizer = HopperOptimizer(
                model_id=HF_MODEL_ID,
                hf_token=HF_API_TOKEN,
                quantization_method=QUANT_METHOD,
                precision="int4" if INT4_PRECISION else "int8",
                use_flash_attention=USE_FLASH_ATTN,
                use_transformer_engine=USE_TRANSFORMER_ENGINE,
                domain="financial",  # Domain-specific calibration
                cache_dir="./model-cache"
            )
            
            # Load and quantize the model
            logger.info("Loading and quantizing model...")
            model = optimizer.load_and_optimize_model()
            
            # Benchmark the optimized model
            logger.info("Benchmarking optimized model...")
            prompt = "Explain how generative AI can be used with SAP HANA Cloud for financial insights."
            benchmark_results = benchmark_inference(model, prompt)
            
            logger.info(f"Benchmark results:")
            logger.info(f"  Average latency: {benchmark_results['avg_latency_seconds']:.4f} seconds")
            logger.info(f"  Tokens per second: {benchmark_results['tokens_per_second']:.2f}")
            logger.info(f"  Average tokens generated: {benchmark_results['avg_tokens_generated']:.1f}")
        
        # Initialize vector engine for embeddings and RAG
        vector_engine = HANAMLinVectorEngine(
            connection_string=connection_string,
            api_key=OPENAI_API_KEY,
            collection_name="GPU_OPTIMIZED_COLLECTION",
            use_gpu=ENABLE_GPU
        )
        
        # Create an agent with memory
        logger.info("Creating an agent with memory")
        agent = HANAMLAgentWithMemory(
            connection_string=connection_string,
            llm_api_key=OPENAI_API_KEY,
            memory_key="chat_history",
            optimized_model=model if ENABLE_GPU else None
        )
        
        # Run some agent queries
        if agent:
            logger.info("Running agent queries...")
            queries = [
                "What are the key features of SAP HANA Cloud?",
                "How can generative AI enhance data analysis?",
                "Can you explain quantization methods for LLMs?"
            ]
            
            for query in queries:
                logger.info(f"Query: {query}")
                start_time = time.time()
                response = agent.run(query)
                end_time = time.time()
                logger.info(f"Response time: {end_time - start_time:.2f} seconds")
                logger.info(f"Response: {response[:100]}...")
        
        logger.info("Advanced GPU optimization example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in advanced GPU optimization example: {str(e)}")
        raise

if __name__ == "__main__":
    main()