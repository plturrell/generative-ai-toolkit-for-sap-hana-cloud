"""
API endpoints for interacting with HANA AI agents.
"""
import time
import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from hana_ml.dataframe import ConnectionContext
from langchain.llms.base import BaseLLM

from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory, stateless_call
from hana_ai.agents.hana_sql_agent import create_hana_sql_agent
from hana_ai.tools.toolkit import HANAMLToolkit

from ..models import ConversationRequest, ConversationResponse, ErrorResponse
from ..dependencies import get_connection_context, get_llm
from ..auth import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

# Store for active agent sessions
AGENT_SESSIONS: Dict[str, HANAMLAgentWithMemory] = {}
CHAT_HISTORIES: Dict[str, BaseChatMessageHistory] = {}

@router.post(
    "/conversation", 
    response_model=ConversationResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def process_conversation(
    request: ConversationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Process a conversation message with the HANA ML agent.
    
    This endpoint supports both stateful conversations using session_id
    and stateless one-off interactions.
    
    Parameters
    ----------
    request : ConversationRequest
        The conversation request containing the user message
    background_tasks : BackgroundTasks
        FastAPI background tasks for async operations
    api_key : str
        The API key for authentication
    connection_context : ConnectionContext
        The database connection context
    llm : BaseLLM
        The language model to use
        
    Returns
    -------
    ConversationResponse
        The agent's response to the user message
    """
    session_id = request.session_id
    start_time = time.time()
    
    try:
        # Get or create agent for this session
        if session_id not in AGENT_SESSIONS:
            # Create a new toolkit with all tools
            toolkit = HANAMLToolkit(
                connection_context=connection_context, 
                used_tools="all"
            )
            tools = toolkit.get_tools()
            
            # Create a new agent with memory
            AGENT_SESSIONS[session_id] = HANAMLAgentWithMemory(
                llm=llm,
                tools=tools,
                session_id=session_id,
                n_messages=30,  # Remember 30 previous messages
                verbose=request.verbose
            )
            CHAT_HISTORIES[session_id] = InMemoryChatMessageHistory(session_id=session_id)
        
        # Get the agent for this session
        agent = AGENT_SESSIONS[session_id]
        
        # Process the message
        if request.return_intermediate_steps:
            # Use stateless mode to get intermediate steps
            chat_history = CHAT_HISTORIES[session_id].messages
            response = stateless_call(
                llm=llm,
                tools=agent.tools,
                question=request.message,
                chat_history=chat_history,
                verbose=request.verbose,
                return_intermediate_steps=True
            )
            
            # Extract response and steps
            result = response.get("output", "")
            intermediate_steps = response.get("intermediate_steps", None)
            
            # Update chat history in background
            background_tasks.add_task(
                update_chat_history,
                session_id=session_id, 
                user_message=request.message,
                ai_message=result
            )
            
            return ConversationResponse(
                response=result,
                conversation_id=session_id,
                intermediate_steps=intermediate_steps
            )
        else:
            # Use stateful mode with memory
            result = agent.run(request.message)
            
            return ConversationResponse(
                response=result,
                conversation_id=session_id,
                intermediate_steps=None
            )
            
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing conversation: {str(e)}"
        )

@router.post(
    "/sql",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def execute_sql_agent(
    query: str,
    api_key: str = Depends(get_api_key),
    connection_context: ConnectionContext = Depends(get_connection_context),
    llm: BaseLLM = Depends(get_llm)
):
    """
    Execute a natural language query using the HANA SQL agent.
    
    Parameters
    ----------
    query : str
        Natural language query to execute
    api_key : str
        API key for authentication
    connection_context : ConnectionContext
        Database connection context
    llm : BaseLLM
        Language model to use
        
    Returns
    -------
    Dict
        The SQL agent's response
    """
    try:
        # Create SQL agent
        agent = create_hana_sql_agent(
            llm=llm,
            connection_context=connection_context,
            verbose=False
        )
        
        # Execute the query
        start_time = time.time()
        result = agent.invoke(query)
        execution_time = time.time() - start_time
        
        return {
            "result": result,
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error executing SQL agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing SQL agent: {str(e)}"
        )

def update_chat_history(session_id: str, user_message: str, ai_message: str):
    """
    Update the chat history for a session.
    
    Parameters
    ----------
    session_id : str
        The session identifier
    user_message : str
        The user's message
    ai_message : str
        The AI's response
    """
    if session_id in CHAT_HISTORIES:
        history = CHAT_HISTORIES[session_id]
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)