# In: app/services/chat_service.py
import os
from functools import lru_cache

try:
    from morphik import Morphik
except ImportError:
    print("Warning: morphik library not found. Using a dummy Morphik object for chat_service.")
    # Define a dummy Morphik if the actual one is not available
    class DummyMorphikPipeline:
        def run(self, user_input: dict) -> dict:
            print(f"DummyMorphikPipeline.run called with user_input: {user_input}")
            # Simulate a response structure similar to what the pipeline would return
            return {
                "gemini_generator": {
                    "answer": f"This is a dummy response from a placeholder Morphik pipeline for query: {user_input.get('query', '')}"
                }
            }

    class DummyMorphik:
        @staticmethod
        def init(config: str):
            print(f"DummyMorphik.init called with config: {config}")
            return DummyMorphikPipeline()

    Morphik = DummyMorphik

try:
    from app.models.chat import ChatRequest, ChatResponse
except ImportError:
    print("Warning: ChatRequest or ChatResponse models not found. Using placeholder classes.")
    # Define dummy models if actual ones are not available
    # These should ideally match the structure of app.models.chat
    class ChatRequest:
        def __init__(self, query: str, conversation_id: str = None, user_id: str = None):
            self.query = query
            self.conversation_id = conversation_id
            self.user_id = user_id

    class ChatResponse:
        def __init__(self, response: str, conversation_id: str = None, metadata: dict = None, error: str = None):
            self.response = response
            self.conversation_id = conversation_id
            self.metadata = metadata if metadata is not None else {}
            self.error = error

# Best Practice: Set the components path via an environment variable
# This should be set before Morphik is initialized.
# In a real application, this might be set when the application starts,
# e.g., in main.py or via the environment itself.
os.environ["MORPHIK__components__path"] = "morphik_pipeline/components"
print(f"MORPHIK__components__path set to: {os.environ.get('MORPHIK__components__path')}")

@lru_cache(maxsize=1)
def get_rag_pipeline():
    """Initializes the Morphik pipeline once and caches it for performance."""
    print("Initializing RAG pipeline...")
    try:
        pipeline = Morphik.init(config="morphik_pipeline/pipeline.yaml")
        print("RAG pipeline initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        # Fallback to a dummy pipeline if initialization fails
        # This ensures the service can still run, albeit without full functionality.
        return DummyMorphik.init(config="morphik_pipeline/pipeline.yaml")


async def process_chat_query(request: ChatRequest) -> ChatResponse:
    print(f"Processing chat query for conversation_id: {request.conversation_id}, query: '{request.query}'")
    # 1. Get the cached pipeline instance
    rag_pipeline = get_rag_pipeline()

    # 2. Execute the pipeline
    print("Executing RAG pipeline...")
    try:
        result = rag_pipeline.run(
            user_input={"query": request.query} # Pass data to the 'user_input' component
        )
        print(f"RAG pipeline execution result: {result}")
    except Exception as e:
        print(f"Error executing RAG pipeline: {e}")
        return ChatResponse(
            response=f"Error processing your query: {e}",
            conversation_id=request.conversation_id,
            metadata={"model_provider": "gemini", "source": "morphik_pipeline", "error": True},
            error=f"Error processing your query: {e}" # Populate error field
        )

    # 3. Extract the final answer
    # The structure of 'result' depends on your pipeline's output components.
    # Assuming the final answer is from 'gemini_generator' and under the key 'answer'.
    final_answer = result.get("gemini_generator", {}).get("answer", "No answer found.")
    print(f"Final answer extracted: {final_answer}")

    # 4. (CRITICAL) Integrate with our existing evaluation system
    try:
        from src.evaluator import ResponseEvaluator
        evaluator = ResponseEvaluator()
        print("Evaluating response...")
        await evaluator.evaluate_response(
            inputs={"question": request.query}, # Ensure 'question' matches expected key if any
            outputs={"answer": final_answer}
        )
        print("Response evaluation complete.")
    except ImportError:
        print("Warning: src.evaluator.ResponseEvaluator not found. Skipping evaluation.")
    except Exception as e:
        print(f"Error during response evaluation: {e}")


    # 5. Return the response in our standard API format
    response = ChatResponse(
        response=final_answer,
        conversation_id=request.conversation_id,
        metadata={"model_provider": "gemini", "source": "morphik_pipeline"}
    )
    print(f"Returning ChatResponse: {response}")
    return response

print("chat_service.py with Morphik integration defined.")
