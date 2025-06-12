"""
Manufacturing Agent MVP - Main Application

This module serves as the entry point for the Manufacturing Agent MVP.
It sets up the FastAPI application, defines the agent tools, and handles API endpoints.
"""
import os
import logging
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import configuration
from config import (
    APP_NAME, APP_VERSION, DEBUG, MODEL_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
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

# Tool 2: Document Search (Mock)
@tool
def document_search_tool(query: str) -> str:
    """
    Use this tool to answer general questions about company policies or procedures.
    
    Args:
        query (str): The user's question or query.
        
    Returns:
        str: The answer to the query or a message if no information is found.
    """
    query = query.lower()
    if "return" in query:
        return "Our return policy states that items can be returned within 30 days if defective."
    elif "shipping" in query:
        return "Standard shipping takes 3-5 business days. Expedited shipping is available."
    elif "hours" in query or "time" in query:
        return "Our customer service is available Monday to Friday, 9 AM to 5 PM EST."
    return "I couldn't find specific information about that topic in our documents."

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

def get_llm():
    """Initialize and return the appropriate LLM based on configuration."""
    try:
        if MODEL_PROVIDER == ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
                raise ValueError("OpenAI API key is required but not set")
            return ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                openai_api_key=OPENAI_API_KEY
            )
        elif MODEL_PROVIDER == ModelProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai
            
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
                raise ValueError("Gemini API key is required but not set")
                
            try:
                # Configure the Gemini client with the API key
                genai.configure(api_key=GEMINI_API_KEY)
            
                # Test the API key by making a simple call
                models = genai.list_models()
                if not models:
                    raise ValueError("Failed to fetch models from Gemini API")
            
                # Create a GenerativeModel instance
                model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    generation_config={
                        "temperature": GEMINI_TEMPERATURE,
                    },
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
            
                # Initialize the chat model
                chat = model.start_chat(history=[])
            
                # Return the configured ChatGoogleGenerativeAI instance
                return ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    temperature=GEMINI_TEMPERATURE,
                    google_api_key=GEMINI_API_KEY,
                    client=model,
                    convert_system_message_to_human=True
                )
            except Exception as e:
                logger.error(f"Error initializing Gemini client: {str(e)}")
                raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
        else:
            raise ValueError(f"Unsupported model provider: {MODEL_PROVIDER}")
    except Exception as e:
        logger.error(f"Error in get_llm: {str(e)}")
        raise

def setup_agent() -> AgentExecutor:
    """Set up and return the agent executor with the configured LLM."""
    try:
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
    agent_executor = setup_agent()
    logger.info(f"Successfully initialized {MODEL_PROVIDER.value.upper()} agent")
except Exception as e:
    logger.error(f"Failed to initialize agent: {str(e)}")
    agent_executor = None

# Pydantic models
class AgentQuery(BaseModel):
    """Model for agent query requests."""
    query: str = Field(..., description="The user's query or request")
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking multi-turn conversations"
    )

class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    status: str
    model_provider: str
    model: str
    version: str
    debug: bool

# API endpoints
@app.post("/api/chat")
async def chat_with_agent(query: AgentQuery):
    """
    Endpoint to interact with the manufacturing agent.
    
    Args:
        query (AgentQuery): The user's query and optional conversation ID.
        
    Returns:
        Dict: The agent's response and metadata.
    """
    try:
        if MODEL_PROVIDER == ModelProvider.GEMINI:
            # For Gemini, we'll use a simpler approach without tools for now
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Create a chat session
            chat = model.start_chat(history=[])
            
            # Send the message and get response
            response = chat.send_message(query.query)
            
            # Return the response
            return {
                "response": response.text,
                "conversation_id": query.conversation_id or "default-conversation",
                "metadata": {
                    "model": GEMINI_MODEL,
                    "provider": "Gemini"
                }
            }
        else:
            # Process the query using the OpenAI agent
            result = agent_executor.invoke({
                "input": query.query,
                "agent_scratchpad": []
            })
            
            # Log the interaction
            logger.info(f"Processed query: {query.query}")
            
            # Return the response
            return {
                "response": result.get("output", "No response generated"),
                "conversation_id": query.conversation_id or "default-conversation",
                "metadata": {
                    "model": OPENAI_MODEL,
                    "provider": "OpenAI"
                }
            }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your request: {str(e)}"
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
