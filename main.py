"""
Manufacturing Agent MVP - Main Application

This module serves as the entry point for the Manufacturing Agent MVP.
It sets up the FastAPI application, defines the agent tools, and handles API endpoints.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation modules
from src.llm_client import LLMClient
from src.evaluator import ResponseEvaluator, with_evaluation

# LangSmith tracing
try:
    import langsmith
    from langsmith import Client
    from langchain_core.tracers.context import tracing_v2_enabled
    
    # Initialize LangSmith client
    try:
        client = Client()
        logger.info("LangSmith client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith client: {e}")
        client = None
except ImportError as e:
    logger.warning(f"LangSmith not installed: {e}")
    client = None
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# RAG Imports
from rag import get_retriever, get_qa_chain
from rag.vector_store import get_vector_store, initialize_pinecone
from rag.document_loader import load_and_chunk_documents, save_documents_to_vector_store

# Import configuration
from config import (
    APP_NAME, APP_VERSION, DEBUG, MODEL_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
    GEMINI_SAFETY_SETTINGS, ModelProvider
)
from workflows import run_expedite_order_workflow, WorkflowError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="API for the Manufacturing Agent MVP",
    version=APP_VERSION,
    debug=DEBUG
)

# Tool 1: Expedite Order Workflow
@tool
def expedite_order_tool(order_id: str) -> Dict[str, Any]:
    """
    Use this tool when a user asks to expedite, rush, or prioritize a specific order.
    This tool will run a multi-step process to check the order, verify inventory,
    and notify the relevant teams.
    
    Args:
        order_id (str): The ID of the order to expedite.
        
    Returns:
        Dict[str, Any]: The result of the expedite order workflow.
    """
    return run_expedite_order_workflow(order_id)

# Tool 2: Document Search with RAG
@tool
def document_search_tool(query: str) -> str:
    """
    Use this tool to answer general questions about company policies, procedures, 
    or any other information from the knowledge base.
    
    Args:
        query (str): The user's question or query.
        
    Returns:
        str: The answer to the query or a message if no information is found.
    """
    try:
        # Initialize vector store if not already done
        if get_vector_store() is None:
            initialize_pinecone()
        
        # Get the QA chain
        qa_chain = get_qa_chain(llm=get_llm())
        
        if not qa_chain:
            logger.error("Failed to initialize QA chain")
            return "I'm having trouble accessing the knowledge base. Please try again later."
        
        # Get the response
        result = qa_chain({"query": query})
        
        # Format the response with sources
        response = result["result"]
        
        # Add source documents if available
        if "source_documents" in result and result["source_documents"]:
            sources = set(doc.metadata.get("source", "Unknown") for doc in result["source_documents"])
            if sources:
                response += "\n\nSources: " + ", ".join(sources)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in document search: {str(e)}", exc_info=True)
        return "I encountered an error while searching the knowledge base. Please try again later."

def setup_agent() -> AgentExecutor:
    """
    Set up and configure the LangChain agent with tools.
    
    Returns:
        AgentExecutor: The configured agent executor.
    """
    # Initialize the language model with configuration from config
    from config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_API_KEY
    
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Define the tools available to the agent
    tools = [expedite_order_tool, document_search_tool]
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful manufacturing operations assistant. 
        You can help with expediting orders or answering general questions.
        Always be concise and professional in your responses.
        When using tools, provide all necessary parameters."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Return the agent executor
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_llm(eval_mode: bool = False):
    """Initialize and return the Gemini LLM with evaluation support.
    
    Args:
        eval_mode: If True, use evaluation-specific model settings
        
    Returns:
        Configured LLMClient instance with evaluation support
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import google.generativeai as genai
        
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            raise ValueError("Gemini API key is required but not set")
            
        # Configure model parameters based on mode
        if eval_mode:
            model_name = os.getenv("EVAL_GEMINI_MODEL", "gemini-1.5-flash-8b")
            temperature = float(os.getenv("EVAL_GEMINI_TEMPERATURE", 0.3))
            max_tokens = int(os.getenv("EVAL_GEMINI_MAX_TOKENS", 4096))
        else:
            model_name = GEMINI_MODEL
            temperature = GEMINI_TEMPERATURE
            max_tokens = 2048
            
        try:
            # Configure the Gemini client with the API key
            genai.configure(api_key=GEMINI_API_KEY)
        
            # Test the API key by making a simple call
            models = genai.list_models()
            if not models:
                raise ValueError("Failed to fetch models from Gemini API")
        
            # Create a GenerativeModel instance
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=GEMINI_SAFETY_SETTINGS
            )
        
            # Initialize the base LLM
            base_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=GEMINI_API_KEY,
                client=model,
                convert_system_message_to_human=True
            )
            
            # If not in eval mode, wrap with our LLMClient
            if not eval_mode:
                # Initialize the evaluator
                evaluator = ResponseEvaluator()
                # Create and return the LLM client with evaluation
                return LLMClient(base_llm, evaluator)
                
            return base_llm
            
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_llm: {str(e)}")
        raise

def setup_agent() -> AgentExecutor:
    """Set up and return the agent executor with the configured LLM and evaluation."""
    try:
        # Get the LLM with evaluation support
        llm = get_llm()
        
        # Define tools
        tools = [expedite_order_tool, document_search_tool]
        tool_names = [tool.name for tool in tools]
        
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful manufacturing operations assistant. 
            You can help with expediting orders or answering general questions.
            Always be concise and professional in your responses.
            When using tools, provide all necessary parameters.
            
            You have access to the following tools:
            {tools}
            
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Format the prompt with the required variables
        formatted_prompt = prompt.partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=", ".join(tool_names)
        )
        
        # Create the agent
        if MODEL_PROVIDER == ModelProvider.OPENAI:
            from langchain.agents import create_openai_tools_agent
            agent = create_openai_tools_agent(llm, tools, formatted_prompt)
        else:
            from langchain.agents import create_structured_chat_agent
            agent = create_structured_chat_agent(llm, tools, formatted_prompt)
        
        # Initialize the agent executor with proper handling of the scratchpad
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=DEBUG,
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        if DEBUG:
            logger.exception("Agent initialization error:")
        raise

# Initialize the agent
try:
    # Enable tracing if LangSmith is configured
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT", "Manufacturing-Agent-MVP")):
            agent_executor = setup_agent()
        logger.info(f"Successfully initialized {MODEL_PROVIDER.value.upper()} agent with LangSmith tracing")
    else:
        agent_executor = setup_agent()
        logger.info(f"Successfully initialized {MODEL_PROVIDER.value.upper()} agent")
except Exception as e:
    logger.error(f"Failed to initialize agent: {str(e)}")
    agent_executor = None

# Pydantic models
class AgentQuery(BaseModel):
    """Model for agent query requests with evaluation support."""
    query: str = Field(..., description="The user's query or request")
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking multi-turn conversations"
    )
    evaluate: bool = Field(
        True,
        description="Whether to evaluate the response (default: True)"
    )

class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    status: str
    model_provider: str
    model: str
    version: str
    debug: bool

# API endpoints
@app.post("/chat")
async def chat_with_agent(query: AgentQuery):
    """
    Endpoint to interact with the manufacturing agent with built-in evaluation.
    
    Args:
        query (AgentQuery): The user's query and optional conversation ID.
        
    Returns:
        Dict: The agent's response and metadata, including evaluation results.
    """
    try:
        logger.info(f"Received query: {query.query}")
        
        # Process the query using the agent
        response = await agent_executor.ainvoke({
            "input": query.query,
            "chat_history": []  # Add conversation history if available
        })
        
        # Extract the response text
        response_text = response.get("output", "I'm sorry, I couldn't process your request.")
        
        # If the agent is an LLMClient, it will handle evaluation automatically
        if hasattr(agent_executor, 'llm_chain') and hasattr(agent_executor.llm_chain, 'llm'):
            llm = agent_executor.llm_chain.llm
            if hasattr(llm, 'evaluator'):
                # Evaluation runs in the background
                asyncio.create_task(
                    llm.evaluator.evaluate_response(
                        inputs={"question": query.query},
                        outputs={"answer": response_text}
                    )
                )
        
        return {
            "response": response_text,
            "conversation_id": query.conversation_id or "",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        if DEBUG:
            logger.exception("Error details:")
        # Log the error for evaluation
        if hasattr(agent_executor, 'llm_chain') and hasattr(agent_executor.llm_chain, 'llm'):
            llm = agent_executor.llm_chain.llm
            if hasattr(llm, 'evaluator'):
                asyncio.create_task(
                    llm.evaluator.evaluate_response(
                        inputs={"question": query.query},
                        outputs={"error": str(e)}
                    )
                )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/evaluation/metrics", response_model=Dict[str, Any])
async def get_evaluation_metrics():
    """
    Get evaluation metrics for the LLM responses.
    
    Returns:
        Dict: Evaluation metrics including total responses, success rate, etc.
    """
    try:
        if not hasattr(agent_executor, 'llm_chain') or not hasattr(agent_executor.llm_chain, 'llm'):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Evaluation metrics not available with current configuration"
            )
            
        llm = agent_executor.llm_chain.llm
        if not hasattr(llm, 'get_evaluation_metrics'):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Evaluation metrics not available with current LLM client"
            )
            
        metrics = llm.get_evaluation_metrics()
        return {
            "status": "success",
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation metrics: {str(e)}"
        )

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        HealthCheckResponse: The health status of the API and current configuration.
    """
    model = OPENAI_MODEL if MODEL_PROVIDER == ModelProvider.OPENAI else GEMINI_MODEL
    
    return {
        "status": "healthy" if agent_executor else "unhealthy",
        "model_provider": MODEL_PROVIDER.value,
        "model": model,
        "version": APP_VERSION,
        "debug": DEBUG
    }

if __name__ == "__main__":
    import uvicorn
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8001))  # Changed default port to 8001
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="debug" if DEBUG else "info"
    )
