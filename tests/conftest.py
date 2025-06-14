"""
Test configuration and fixtures for the Manufacturing Agent MVP.
This file also handles NLTK data path configuration for tests.
"""
import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Add NLTK for data path configuration
import nltk

# 1. --- EXISTING LOGIC (UNCHANGED) ---
# Add the src directory to the Python path
# NOTE: If your 'rag' module is in 'src', this is correct. If not, adjust as needed.
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables from the project root .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Verify required environment variables are set
# NOTE: Added PINECONE_API_KEY as it's also required for the hybrid search tests
REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "GEMINI_MODEL", "PINECONE_API_KEY"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    pytest.exit(f"Missing required environment variables: {', '.join(missing_vars)}")


# 2. --- NEW LOGIC: PYTEST HOOK FOR NLTK SETUP ---
@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """
    Called at the start of the test session to configure the NLTK data path.
    """
    print("\nConfiguring NLTK data path for test session...")
    
    project_root = Path(__file__).parent.parent
    nltk_data_path = project_root / "nltk_data"
    
    if str(nltk_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_path))
        print(f"Added to NLTK path: {nltk_data_path}")
    
    # --- THIS IS THE FIX ---
    # Update the check to look for the specific resource that fails.
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
        print("NLTK 'punkt_tab' package found successfully.")
    except LookupError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! NLTK 'punkt_tab' package NOT FOUND.              !!!")
        print("!!! Please run the setup script before testing:      !!!")
        print("!!! `python scripts/download_nltk_data.py`           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pytest.exit("NLTK data missing, aborting tests.")
def pytest_sessionstart(session):
    """
    Called at the start of the test session to configure the NLTK data path.
    This ensures that tests can find the locally downloaded 'punkt' package.
    """
    print("\nConfiguring NLTK data path for test session...")
    
    # Define the path to your project's local NLTK data directory
    project_root = Path(__file__).parent.parent
    nltk_data_path = project_root / "nltk_data"
    
    # Prepend this path to NLTK's data path list
    # This ensures NLTK looks here first
    if str(nltk_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_path))
        print(f"Added to NLTK path: {nltk_data_path}")
    
    # Verify that 'punkt' can be found now. If not, abort with a clear message.
    try:
        nltk.data.find("tokenizers/punkt")
        print("NLTK 'punkt' package found successfully.")
    except LookupError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! NLTK 'punkt' package NOT FOUND.                  !!!")
        print("!!! Please run the setup script before testing:      !!!")
        print("!!! `python scripts/download_nltk_data.py`           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pytest.exit("NLTK data missing, aborting tests.")


# 3. --- EXISTING FIXTURES (UNCHANGED) ---
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