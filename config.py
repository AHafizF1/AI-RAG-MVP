"""
Application configuration settings.

This module contains all the configuration settings for the application,
including environment variables and default values.
"""
import os
from typing import Dict, Any, List, Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class ModelProvider(str, Enum):
    GEMINI = "gemini"

# Application Settings
APP_NAME = "Manufacturing Agent MVP"
APP_VERSION = "0.2.0"  # Bumped version for Gemini support
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API Settings
API_PREFIX = "/api"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model Provider Selection
MODEL_PROVIDER = ModelProvider(os.getenv("MODEL_PROVIDER", "gemini").lower())

# RAG Configuration - Pinecone Serverless
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "samtest")
PINECONE_HOST = os.getenv("PINECONE_HOST")  # e.g., "your-index-name-12345.svc.region.pinecone.io"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_DIMENSIONS = int(os.getenv("PINECONE_DIMENSIONS", "1024"))
PINECONE_INDEX_TYPE = os.getenv("PINECONE_INDEX_TYPE", "serverless")

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")
# Llama Text Embed v2 supports 256, 512, 1024 (default), and 2048 dimensions
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Validation
if not PINECONE_API_KEY and DEBUG:
    logger.warning("PINECONE_API_KEY is not set. RAG features will be disabled.")
    
# Export settings for easier imports
__all__ = [
    'APP_NAME', 'APP_VERSION', 'DEBUG', 'API_PREFIX', 'HOST', 'PORT',
    'MODEL_PROVIDER', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT',
    'PINECONE_INDEX_NAME', 'CHUNK_SIZE', 'CHUNK_OVERLAP',
    'EMBEDDING_MODEL', 'EMBEDDING_DIMENSION', 'ModelProvider',
    'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_TEMPERATURE', 'GEMINI_SAFETY_SETTINGS'
]

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
# Load safety settings from environment variable
GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]
# Try to load from environment variable if it exists
safety_settings_env = os.getenv("GEMINI_SAFETY_SETTINGS")
if safety_settings_env:
    try:
        GEMINI_SAFETY_SETTINGS = json.loads(safety_settings_env)
    except json.JSONDecodeError as e:
        print(f"Error parsing GEMINI_SAFETY_SETTINGS: {e}. Using default safety settings.")

# Mock Data Settings
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "URG-456": {
        "status": "Awaiting Production",
        "parts_needed": ["part_a", "part_b"],
        "customer": "Acme Corp"
    },
    "STD-789": {
        "status": "In Production",
        "parts_needed": ["part_c", "part_d"],
        "customer": "Globex Inc"
    }
}

# Workflow Settings
DEFAULT_ORDER_STATUS = "Not Found"
REQUIRED_ORDER_STATUS = "Awaiting Production"

# Slack Notification Settings
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#manufacturing-alerts")

# MES Settings
MES_API_URL = os.getenv("MES_API_URL", "https://mes-api.example.com")
MES_API_KEY = os.getenv("MES_API_KEY", "")

# ERP Settings
ERP_API_URL = os.getenv("ERP_API_URL", "https://erp-api.example.com")
ERP_API_KEY = os.getenv("ERP_API_KEY", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Feature Flags
FEATURE_FLAGS = {
    "enable_slack_notifications": os.getenv("ENABLE_SLACK_NOTIFICATIONS", "true").lower() == "true",
    "enable_mes_integration": os.getenv("ENABLE_MES_INTEGRATION", "true").lower() == "true",
    "enable_erp_integration": os.getenv("ENABLE_ERP_INTEGRATION", "true").lower() == "true",
}
