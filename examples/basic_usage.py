#!/usr/bin/env python
"""
Basic usage example for the Generative AI Toolkit for SAP HANA Cloud
This example demonstrates how to connect to SAP HANA Cloud and use the basic functionalities
of the toolkit without GPU optimizations.
"""

import os
import logging
from hana_ai.vectorstore import HANAMLinVectorEngine
from hana_ai.smart_dataframe import HanaSmartDataFrame
from hana_ai.agents import create_hana_sql_agent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAP HANA Connection parameters (from environment variables or defaults)
HOST = os.environ.get("HANA_HOST", "localhost")
PORT = os.environ.get("HANA_PORT", "39015")
USER = os.environ.get("HANA_USER", "SYSTEM")
PASSWORD = os.environ.get("HANA_PASSWORD", "")
SCHEMA = os.environ.get("HANA_SCHEMA", "SYSTEM")

# OpenAI API Key for LLM access
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def main():
    """Main function demonstrating basic usage of the toolkit"""
    logger.info("Starting basic usage example")
    
    # Check for required credentials
    if not PASSWORD or not OPENAI_API_KEY:
        logger.error("Missing required credentials. Please set HANA_PASSWORD and OPENAI_API_KEY environment variables.")
        return
    
    try:
        # Connect to SAP HANA
        connection_string = f"hana://{USER}:{PASSWORD}@{HOST}:{PORT}/?schema={SCHEMA}"
        logger.info(f"Connecting to SAP HANA at {HOST}:{PORT} with user {USER}")
        
        # Initialize vector engine for embeddings and RAG
        vector_engine = HANAMLinVectorEngine(
            connection_string=connection_string,
            api_key=OPENAI_API_KEY,
            collection_name="SAMPLE_COLLECTION"
        )
        
        # Create example documents
        documents = [
            "SAP HANA is an in-memory, column-oriented, relational database management system.",
            "SAP HANA Cloud offers database as a service with advanced analytics capabilities.",
            "Generative AI can be integrated with SAP HANA for intelligent data processing."
        ]
        
        # Add documents to vector store
        logger.info("Adding documents to vector store")
        vector_engine.add_texts(documents)
        
        # Perform a similarity search
        logger.info("Performing similarity search")
        results = vector_engine.similarity_search("What is SAP HANA Cloud?", k=2)
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.page_content}")
        
        # Create and use a smart dataframe
        logger.info("Creating a smart dataframe")
        df_query = """
        SELECT TOP 10 
            'Product' || ROW_NUMBER() OVER (ORDER BY RAND()) AS PRODUCT_NAME,
            ROUND(RAND() * 1000, 2) AS PRICE,
            ROUND(RAND() * 100, 0) AS QUANTITY
        FROM DUMMY
        """
        
        smart_df = HanaSmartDataFrame(
            connection_string=connection_string,
            query=df_query,
            api_key=OPENAI_API_KEY
        )
        
        # Ask a question about the dataframe
        logger.info("Asking a question about the dataframe")
        response = smart_df.ask("What is the total value of all products?")
        logger.info(f"Response: {response}")
        
        # Create an SQL agent
        logger.info("Creating an SQL agent")
        sql_agent = create_hana_sql_agent(
            connection_string=connection_string,
            llm_api_key=OPENAI_API_KEY
        )
        
        # Run a query through the agent
        logger.info("Running a query through the agent")
        agent_response = sql_agent.run("List all tables in the current schema")
        logger.info(f"Agent response: {agent_response}")
        
        logger.info("Basic usage example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {str(e)}")
        raise

if __name__ == "__main__":
    main()