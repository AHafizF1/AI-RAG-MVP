"""
Test script to verify LangSmith tracing is working correctly.
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from main import setup_agent, get_llm
from langchain_core.tracers.context import tracing_v2_enabled

# Load environment variables
load_dotenv()

async def test_tracing():
    """Test function to verify LangSmith tracing."""
    print("Starting LangSmith tracing test...")
    
    # Test queries
    test_queries = [
        "What is the status of order #12345?",
        "Can you help me find information about quality control procedures?",
        "What are the safety guidelines for operating the CNC machine?",
    ]
    
    # Initialize the LLM and agent
    llm = get_llm()
    
    # Run with tracing enabled
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        project_name = os.getenv("LANGCHAIN_PROJECT", "Manufacturing-Agent-MVP")
        print(f"Running tests with LangSmith tracing (Project: {project_name})")
        
        with tracing_v2_enabled(project_name=project_name):
            agent = setup_agent()
            
            for query in test_queries:
                print(f"\nTesting query: {query}")
                try:
                    response = await agent.ainvoke({"input": query, "chat_history": []})
                    print(f"Response: {response['output']}")
                except Exception as e:
                    print(f"Error processing query: {e}")
    else:
        print("LangSmith tracing is not enabled. Set LANGCHAIN_TRACING_V2=true in .env")

if __name__ == "__main__":
    asyncio.run(test_tracing())
