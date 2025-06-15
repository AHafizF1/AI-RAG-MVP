# In: app/models/chat.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    # Add any other fields that might be part of your request
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    # Add any other fields for the response
    error: Optional[str] = None

print("Pydantic models for ChatRequest and ChatResponse defined.")
