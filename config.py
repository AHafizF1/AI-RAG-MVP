"""
Application configuration settings.

This module contains all the configuration settings for the application,
including environment variables and default values.
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application Settings
APP_NAME = "Manufacturing Agent MVP"
APP_VERSION = "0.1.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API Settings
API_PREFIX = "/api"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

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
