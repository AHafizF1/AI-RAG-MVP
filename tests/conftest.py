"""
Test configuration and fixtures for the Manufacturing Agent MVP.
"""
import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables from the project root .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Verify required environment variables are set
REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "GEMINI_MODEL"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    pytest.exit(f"Missing required environment variables: {', '.join(missing_vars)}")

# Import LLM and evaluator after environment is loaded
from llm_client import LLMClient
from evaluator import ResponseEvaluator

@pytest.fixture(scope="session")
def llm_client():
    """Fixture to provide an initialized LLM client."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-pro"),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", 0.7)),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )
    return LLMClient(llm=llm)

@pytest.fixture(scope="session")
def evaluator():
    """Fixture to provide an initialized evaluator."""
    return ResponseEvaluator()
