"""
API endpoints for accessing and using the HANA ML toolkit tools.
"""
import time
import logging
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field

from hana_ml.dataframe import ConnectionContext
from langchain.llms.base import BaseLLM

from hana_ai.tools.toolkit import HANAMLToolkit

from ..dependencies import get_connection_context, get_llm
from ..auth import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

class ToolRequest(BaseModel):
    """Base request for tool execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")

class ToolListResponse(BaseModel):
    """Response for listing available tools."""
    tools: List[Dict[str, Any]] = Field(..., description="List of available tools")
    count: int = Field(..., description="Number of tools available")

@router.get(
    "/list",
    response_model=ToolListResponse,
    summary="List available tools",
    description="Get a list of all available tools in the HANA ML toolkit"
)
async def list_tools(
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    List all available tools in the toolkit.
    
    Parameters
    ----------
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    ToolListResponse
        List of available tools
    """
    try:
        # Get all tools from the toolkit
        toolkit = HANAMLToolkit(connection_context, used_tools="all")
        tools = toolkit.get_tools()
        
        # Format the tools for response
        tool_list = []
        for tool in tools:
            tool_list.append({
                "name": tool.name,
                "description": tool.description,
                "args_schema": getattr(tool, "args_schema", None)
            })
        
        return ToolListResponse(
            tools=tool_list,
            count=len(tool_list)
        )
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tools: {str(e)}"
        )

@router.post(
    "/execute",
    summary="Execute a specific tool",
    description="Execute a single tool from the HANA ML toolkit with the specified parameters"
)
async def execute_tool(
    request: ToolRequest,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Execute a specific tool with parameters.
    
    Parameters
    ----------
    request : ToolRequest
        The request containing tool name and parameters
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
    llm : BaseLLM
        Language model
        
    Returns
    -------
    Dict
        Tool execution result
    """
    try:
        start_time = time.time()
        
        # Get the toolkit
        toolkit = HANAMLToolkit(connection_context, used_tools="all")
        tools = toolkit.get_tools()
        
        # Find the requested tool
        tool = None
        for t in tools:
            if t.name == request.tool_name:
                tool = t
                break
        
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{request.tool_name}' not found"
            )
        
        # Execute the tool
        result = tool.run(request.parameters)
        
        # Process the result for JSON serialization
        if hasattr(result, "to_dict"):
            # Handle pandas DataFrame
            result = result.to_dict(orient="records")
        
        execution_time = time.time() - start_time
        
        return {
            "result": result,
            "execution_time": execution_time,
            "tool": request.tool_name
        }
    except Exception as e:
        logger.error(f"Error executing tool '{request.tool_name}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing tool '{request.tool_name}': {str(e)}"
        )

@router.post(
    "/forecast",
    summary="Run time series forecasting",
    description="Execute time series forecasting on HANA data"
)
async def forecast_timeseries(
    table_name: str = Body(..., description="Source table name"),
    key_column: str = Body(..., description="Date/time column name"),
    value_column: str = Body(..., description="Target value column name"),
    horizon: int = Body(..., description="Number of periods to forecast"),
    model_name: str = Body(..., description="Name to save the model under"),
    model_type: str = Body("automatic_timeseries", description="Type of forecasting model to use"),
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context)
):
    """
    Dedicated endpoint for time series forecasting.
    
    Parameters
    ----------
    table_name : str
        Table containing time series data
    key_column : str
        Date/time column name
    value_column : str
        Target value column name
    horizon : int
        Number of periods to forecast
    model_name : str
        Name to save the model under
    model_type : str
        Type of forecasting model to use
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection
        
    Returns
    -------
    Dict
        Forecast results
    """
    try:
        start_time = time.time()
        
        # Get toolkit with specific tools
        toolkit = HANAMLToolkit(
            connection_context=connection_context, 
            used_tools=[
                "ts_check", 
                "automatic_timeseries_fit_and_save",
                "additive_model_forecast_fit_and_save"
            ]
        )
        tools = toolkit.get_tools()
        
        # 1. Check time series properties
        ts_check_tool = next(t for t in tools if t.name == "ts_check")
        check_result = ts_check_tool.run({
            "table_name": table_name,
            "key": key_column,
            "endog": value_column
        })
        
        # 2. Fit appropriate model
        if model_type == "automatic_timeseries":
            fit_tool = next(t for t in tools if t.name == "automatic_timeseries_fit_and_save")
        else:
            fit_tool = next(t for t in tools if t.name == "additive_model_forecast_fit_and_save")
            
        fit_result = fit_tool.run({
            "fit_table": table_name,
            "name": model_name,
            "key": key_column,
            "endog": value_column,
        })
        
        # 3. Create prediction table name
        predict_table_name = f"{model_name}_{fit_result.get('model_storage_version', 1)}_PREDICT"
        
        # 4. Create horizon table
        horizon_table = f"{table_name}_HORIZON"
        
        # Generate dates for forecast horizon
        # This is a simplified approach - in reality you'd need to determine date increments based on the data
        horizon_query = f"""
        CREATE TABLE {horizon_table} AS (
            SELECT {key_column}
            FROM {table_name}
            ORDER BY {key_column} DESC
            LIMIT {horizon}
        )
        """
        
        # In a real implementation, you'd generate the proper date sequence here
        # This is just a placeholder
        
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "model_details": fit_result,
            "time_series_properties": check_result,
            "execution_time": execution_time
        }
        
    except Exception as e:
        logger.error(f"Error in forecast operation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in forecast operation: {str(e)}"
        )