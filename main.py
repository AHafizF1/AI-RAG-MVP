"""
Manufacturing Agent MVP - Main Application

This module serves as the entry point for the Manufacturing Agent MVP.
It sets up the FastAPI application, defines the agent tools, and handles API endpoints.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from workflows import run_expedite_order_workflow

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Manufacturing Agent MVP",
    description="API for the Manufacturing Agent MVP",
    version="0.1.0"
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

# Initialize the agent
agent_executor = setup_agent()

# Pydantic model for the API request
class AgentQuery(BaseModel):
    query: str

# API endpoint for the agent
@app.post("/invoke_agent")
async def invoke_agent(query: AgentQuery):
    """
    Endpoint to interact with the manufacturing agent.
    
    Args:
        query (AgentQuery): The user's query.
        
    Returns:
        Dict[str, str]: The agent's response.
    """
    try:
        response = agent_executor.invoke({"input": query.query})
        return {"response": response['output']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
