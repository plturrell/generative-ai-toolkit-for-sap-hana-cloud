"""
API endpoints for working with HANA dataframes and SmartDataFrame functionality.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from hana_ml.dataframe import ConnectionContext, DataFrame
from langchain.llms.base import BaseLLM

from hana_ai.smart_dataframe import SmartDataFrame
from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

from ..dependencies import get_connection_context, get_llm
from ..auth import get_api_key
from ..models import SmartDataFrameRequest, DataResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Request for querying a dataframe."""
    query: str
    limit: Optional[int] = 1000
    offset: Optional[int] = 0

@router.post(
    "/query",
    response_model=DataResponse,
    summary="Execute SQL query on HANA database",
    description="Run SQL query against the connected HANA database and return results"
)
async def query_database(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Execute SQL query and return results.
    
    Parameters
    ----------
    request : QueryRequest
        The query request
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    DataResponse
        Query results
    """
    try:
        start_time = time.time()
        
        # Create DataFrame from query
        df = DataFrame(connection_context, request.query)
        
        # Apply pagination if provided
        if request.limit or request.offset:
            paginated_query = f"SELECT * FROM ({df.select_statement}) LIMIT {request.limit} OFFSET {request.offset}"
            df = DataFrame(connection_context, paginated_query)
        
        # Collect results
        result = df.collect()
        columns = df.columns
        
        query_time = time.time() - start_time
        
        return DataResponse(
            columns=columns,
            data=result,
            row_count=len(result),
            query_time=query_time
        )
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing query: {str(e)}"
        )

@router.post(
    "/smart/ask",
    summary="Ask questions about a dataframe",
    description="Ask natural language questions about data in a HANA database table or query"
)
async def smart_dataframe_ask(
    request: SmartDataFrameRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Ask a natural language question about data.
    
    Parameters
    ----------
    request : SmartDataFrameRequest
        The request containing the table and question
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
    llm : BaseLLM
        Language model
        
    Returns
    -------
    Dict
        Response with analysis and results
    """
    try:
        start_time = time.time()
        
        # Get dataframe from table or SQL query
        if request.is_sql_query:
            df = DataFrame(connection_context, request.table_name)
        else:
            df = connection_context.table(request.table_name)
        
        # Create SmartDataFrame
        smart_df = SmartDataFrame(df)
        
        # Set up vector store for code templates if possible
        try:
            hana_vec = HANAMLinVectorEngine(connection_context, "hana_vec_hana_ml_knowledge")
            code_tool = GetCodeTemplateFromVectorDB()
            code_tool.set_vectordb(hana_vec)
            tools = [code_tool]
        except Exception as e:
            logger.warning(f"Could not initialize vector store for code templates: {str(e)}")
            tools = []
        
        # Configure SmartDataFrame
        smart_df.configure(llm=llm, tools=tools)
        
        # Ask question or transform
        if request.transform:
            result_df = smart_df.transform(request.question)
            # Get the SQL for the result
            sql = result_df.select_statement
            # Get a sample of the result data
            result_data = result_df.head(10).collect()
            result_columns = result_df.columns
            
            return {
                "type": "transform",
                "sql": sql,
                "columns": result_columns,
                "data": result_data,
                "query_time": time.time() - start_time
            }
        else:
            result = smart_df.ask(request.question)
            
            return {
                "type": "ask",
                "result": result,
                "query_time": time.time() - start_time
            }
            
    except Exception as e:
        logger.error(f"SmartDataFrame operation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error with SmartDataFrame operation: {str(e)}"
        )

@router.get(
    "/tables",
    summary="List available tables",
    description="Get a list of tables available in the connected HANA database"
)
async def list_tables(
    schema: Optional[str] = Query(None, description="Schema name to filter tables"),
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    List available tables in the database.
    
    Parameters
    ----------
    schema : str, optional
        Schema name to filter tables
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    Dict
        List of available tables
    """
    try:
        if schema:
            # List tables in specific schema
            query = f"""
                SELECT TABLE_NAME 
                FROM SYS.TABLES 
                WHERE SCHEMA_NAME = '{schema}'
                ORDER BY TABLE_NAME
            """
        else:
            # List tables in current schema
            query = """
                SELECT TABLE_NAME 
                FROM SYS.TABLES 
                WHERE SCHEMA_NAME = CURRENT_SCHEMA
                ORDER BY TABLE_NAME
            """
        
        result = connection_context.sql(query).collect()
        tables = [row["TABLE_NAME"] for row in result]
        
        return {
            "tables": tables,
            "count": len(tables),
            "schema": schema or connection_context.get_current_schema()
        }
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tables: {str(e)}"
        )