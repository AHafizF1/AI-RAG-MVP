from fastapi import FastAPI
from pydantic import BaseModel
# Ensure correct import path based on directory structure
from .chat_service import simple_rag_answer
# from .chat_service import agentic_task_handler # Commented out as per plan

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    user_id: str
    use_agent: bool = False

class ChatResponse(BaseModel):
    answer: str
    details: list | dict # Adjusted based on simple_rag_answer output

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # if request.use_agent: # Commented out as per plan
    #     result = agentic_task_handler(request.query, request.user_id)
    #     return ChatResponse(answer=result["answer"], details=result["reasoning"])
    # else:
    result = simple_rag_answer(request.query, request.user_id)
    return ChatResponse(answer=result["answer"], details=result["sources"])
