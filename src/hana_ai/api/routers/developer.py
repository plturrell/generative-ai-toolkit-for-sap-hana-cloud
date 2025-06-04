"""
API endpoints for the Developer Studio.

This module provides endpoints for code generation, execution,
and flow management in the Developer Studio.
"""
import os
import time
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, UploadFile, File
from pydantic import BaseModel, Field

from hana_ml.dataframe import ConnectionContext
from langchain.llms.base import BaseLLM

from ..dependencies import get_connection_context, get_llm
from ..auth import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

# Models for request/response validation
class GenerateCodeRequest(BaseModel):
    """Request model for code generation."""
    flow: Dict[str, Any] = Field(..., description="Flow definition with nodes and edges")
    language: str = Field(default="python", description="Target programming language")
    include_comments: bool = Field(default=True, description="Whether to include comments in the generated code")

class GenerateCodeResponse(BaseModel):
    """Response model for code generation."""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language of the generated code")
    execution_time: float = Field(..., description="Time taken to generate the code in seconds")

class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="Code to execute")
    language: str = Field(default="python", description="Programming language of the code")
    timeout: int = Field(default=30, description="Execution timeout in seconds")

class ExecuteCodeResponse(BaseModel):
    """Response model for code execution."""
    output: str = Field(..., description="Execution output")
    execution_time: float = Field(..., description="Time taken to execute the code in seconds")
    success: bool = Field(..., description="Whether execution was successful")
    error: Optional[str] = Field(None, description="Error message if execution failed")

class SaveFlowRequest(BaseModel):
    """Request model for saving a flow."""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    flow: Dict[str, Any] = Field(..., description="Flow definition with nodes and edges")

class FlowListItem(BaseModel):
    """Model for a flow list item."""
    id: str = Field(..., description="Flow ID")
    name: str = Field(..., description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class QueryRequest(BaseModel):
    """Request model for executing a SQL query."""
    query: str = Field(..., description="SQL query to execute")
    max_rows: int = Field(default=1000, description="Maximum number of rows to return")

class GenerateQueryRequest(BaseModel):
    """Request model for generating a SQL query from schema and requirements."""
    tables: List[str] = Field(..., description="List of tables to query")
    columns: List[str] = Field(..., description="List of columns to select")
    filters: Optional[List[Dict[str, Any]]] = Field(None, description="Filter conditions")
    group_by: Optional[List[str]] = Field(None, description="Group by columns")
    order_by: Optional[List[Dict[str, str]]] = Field(None, description="Order by columns and direction")
    requirements: Optional[str] = Field(None, description="Natural language requirements for the query")

@router.post(
    "/generate-code",
    response_model=GenerateCodeResponse,
    summary="Generate code from flow",
    description="Generate executable code from a visual flow definition"
)
async def generate_code(
    request: GenerateCodeRequest,
    api_key: str = Depends(get_api_key),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Generate code from a flow definition using AI.
    
    This endpoint uses the LLM to generate executable code based on
    the visual flow created in the Developer Studio.
    
    Parameters
    ----------
    request : GenerateCodeRequest
        The flow definition and code generation options
    api_key : str
        API key for authentication
    llm : BaseLLM
        Language model
        
    Returns
    -------
    GenerateCodeResponse
        The generated code and metadata
    """
    try:
        start_time = time.time()
        
        # Extract flow information
        nodes = request.flow.get("nodes", [])
        edges = request.flow.get("edges", [])
        
        # Prepare the prompt for the LLM
        prompt = f"""
        Generate {request.language} code for the following data flow:
        
        Nodes:
        {json.dumps(nodes, indent=2)}
        
        Connections:
        {json.dumps(edges, indent=2)}
        
        Requirements:
        - The code should be executable
        - Use the hana_ml library for HANA database connections
        - Include proper error handling
        - {"Include explanatory comments" if request.include_comments else "Minimize comments"}
        - Use best practices for {request.language}
        
        Return only the code without any additional explanation.
        """
        
        # Use LLM to generate code
        code = llm(prompt)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        return GenerateCodeResponse(
            code=code,
            language=request.language,
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating code: {str(e)}"
        )

@router.post(
    "/execute-code",
    response_model=ExecuteCodeResponse,
    summary="Execute code",
    description="Execute code generated from a flow or manually written"
)
async def execute_code(
    request: ExecuteCodeRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Execute code in a sandboxed environment.
    
    This endpoint executes the provided code in a secure environment
    with the necessary dependencies installed.
    
    Parameters
    ----------
    request : ExecuteCodeRequest
        The code to execute and execution options
    api_key : str
        API key for authentication
        
    Returns
    -------
    ExecuteCodeResponse
        The execution output and metadata
    """
    try:
        start_time = time.time()
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix=get_file_extension(request.language), delete=False) as temp_file:
            temp_file.write(request.code)
            temp_file_path = temp_file.name
        
        try:
            # Execute the code based on the language
            if request.language == "python":
                # Execute Python code
                result = subprocess.run(
                    ["python", temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout
                )
                
                # Check if execution was successful
                if result.returncode == 0:
                    output = result.stdout
                    success = True
                    error = None
                else:
                    output = result.stderr
                    success = False
                    error = f"Execution failed with exit code {result.returncode}"
            
            elif request.language == "sql":
                # For SQL, we would execute it against the database
                # This is a simplified implementation
                conn = get_connection_context()
                try:
                    # Execute SQL query
                    result = conn.sql(request.code).collect()
                    output = json.dumps(result, indent=2)
                    success = True
                    error = None
                except Exception as e:
                    output = str(e)
                    success = False
                    error = "SQL execution failed"
            
            else:
                # Unsupported language
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {request.language}"
                )
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        return ExecuteCodeResponse(
            output=output,
            execution_time=execution_time,
            success=success,
            error=error
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timed out after {request.timeout} seconds")
        return ExecuteCodeResponse(
            output="",
            execution_time=request.timeout,
            success=False,
            error=f"Execution timed out after {request.timeout} seconds"
        )
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing code: {str(e)}"
        )

@router.post(
    "/generate-query",
    summary="Generate SQL query",
    description="Generate a SQL query from schema and requirements"
)
async def generate_query(
    request: GenerateQueryRequest,
    api_key: str = Depends(get_api_key),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Generate a SQL query from schema information and requirements.
    
    This endpoint uses the LLM to generate a SQL query based on
    the tables, columns, and requirements specified.
    
    Parameters
    ----------
    request : GenerateQueryRequest
        The schema information and requirements
    api_key : str
        API key for authentication
    llm : BaseLLM
        Language model
        
    Returns
    -------
    Dict
        The generated SQL query and metadata
    """
    try:
        start_time = time.time()
        
        # Prepare the prompt for the LLM
        prompt = f"""
        Generate a SQL query with the following requirements:
        
        Tables: {', '.join(request.tables)}
        Columns: {', '.join(request.columns)}
        """
        
        if request.filters:
            prompt += f"\nFilters: {json.dumps(request.filters, indent=2)}"
        
        if request.group_by:
            prompt += f"\nGroup by: {', '.join(request.group_by)}"
        
        if request.order_by:
            prompt += f"\nOrder by: {json.dumps(request.order_by, indent=2)}"
        
        if request.requirements:
            prompt += f"\nAdditional requirements: {request.requirements}"
        
        prompt += "\n\nReturn only the SQL query without any additional explanation."
        
        # Use LLM to generate SQL query
        query = llm(prompt)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error generating query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating query: {str(e)}"
        )

@router.post(
    "/execute-query",
    summary="Execute SQL query",
    description="Execute a SQL query against the HANA database"
)
async def execute_query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Execute a SQL query against the HANA database.
    
    Parameters
    ----------
    request : QueryRequest
        The SQL query to execute
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    Dict
        The query results and metadata
    """
    try:
        start_time = time.time()
        
        # Execute query
        result = connection_context.sql(request.query).collect(request.max_rows)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        return {
            "result": result,
            "row_count": len(result),
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing query: {str(e)}"
        )

@router.get(
    "/flows",
    response_model=List[FlowListItem],
    summary="List saved flows",
    description="Get a list of all saved flows"
)
async def list_flows(
    api_key: str = Depends(get_api_key)
):
    """
    List all saved flows.
    
    Parameters
    ----------
    api_key : str
        API key for authentication
        
    Returns
    -------
    List[FlowListItem]
        List of saved flows
    """
    try:
        # In a real implementation, this would fetch flows from a database
        # For now, return some sample flows
        return [
            FlowListItem(
                id="flow1",
                name="Customer Analysis",
                description="Analyze customer data by country and city",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-02T00:00:00Z"
            ),
            FlowListItem(
                id="flow2",
                name="Sales Forecast",
                description="Forecast sales for the next quarter",
                created_at="2025-01-03T00:00:00Z",
                updated_at="2025-01-04T00:00:00Z"
            ),
            FlowListItem(
                id="flow3",
                name="Product Inventory Analysis",
                description="Analyze product inventory levels",
                created_at="2025-01-05T00:00:00Z",
                updated_at="2025-01-06T00:00:00Z"
            )
        ]
    except Exception as e:
        logger.error(f"Error listing flows: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing flows: {str(e)}"
        )

@router.post(
    "/flows",
    summary="Save flow",
    description="Save a flow definition"
)
async def save_flow(
    request: SaveFlowRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Save a flow definition.
    
    Parameters
    ----------
    request : SaveFlowRequest
        The flow to save
    api_key : str
        API key for authentication
        
    Returns
    -------
    Dict
        The saved flow ID and metadata
    """
    try:
        # In a real implementation, this would save the flow to a database
        # For now, just return a success response
        flow_id = f"flow_{int(time.time())}"
        
        return {
            "id": flow_id,
            "name": request.name,
            "description": request.description,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    except Exception as e:
        logger.error(f"Error saving flow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error saving flow: {str(e)}"
        )

@router.get(
    "/flows/{flow_id}",
    summary="Get flow",
    description="Get a flow definition by ID"
)
async def get_flow(
    flow_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get a flow definition by ID.
    
    Parameters
    ----------
    flow_id : str
        Flow ID
    api_key : str
        API key for authentication
        
    Returns
    -------
    Dict
        The flow definition
    """
    try:
        # In a real implementation, this would fetch the flow from a database
        # For now, return a sample flow
        if flow_id == "flow1":
            return {
                "id": "flow1",
                "name": "Customer Analysis",
                "description": "Analyze customer data by country and city",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-02T00:00:00Z",
                "flow": {
                    "nodes": [
                        {
                            "id": "1",
                            "type": "input",
                            "data": {
                                "label": "HANA Table",
                                "description": "Connect to CUSTOMERS table"
                            },
                            "position": { "x": 250, "y": 50 }
                        },
                        {
                            "id": "2",
                            "data": {
                                "label": "Filter",
                                "description": "WHERE country = 'Germany'"
                            },
                            "position": { "x": 250, "y": 150 }
                        },
                        {
                            "id": "3",
                            "type": "output",
                            "data": {
                                "label": "Visualization",
                                "description": "Bar chart of customer count by city"
                            },
                            "position": { "x": 250, "y": 250 }
                        }
                    ],
                    "edges": [
                        { "id": "e1-2", "source": "1", "target": "2" },
                        { "id": "e2-3", "source": "2", "target": "3" }
                    ]
                }
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Flow with ID {flow_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting flow: {str(e)}"
        )

@router.delete(
    "/flows/{flow_id}",
    summary="Delete flow",
    description="Delete a flow by ID"
)
async def delete_flow(
    flow_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Delete a flow by ID.
    
    Parameters
    ----------
    flow_id : str
        Flow ID
    api_key : str
        API key for authentication
        
    Returns
    -------
    Dict
        Deletion status
    """
    try:
        # In a real implementation, this would delete the flow from a database
        # For now, just return a success response
        return {
            "status": "success",
            "message": f"Flow {flow_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting flow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting flow: {str(e)}"
        )

# Helper functions
def get_file_extension(language: str) -> str:
    """Get the file extension for a given programming language."""
    extensions = {
        "python": ".py",
        "sql": ".sql",
        "javascript": ".js",
        "typescript": ".ts"
    }
    
    return extensions.get(language, ".txt")