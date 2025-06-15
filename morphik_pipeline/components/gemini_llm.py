# In: morphik_pipeline/components/gemini_llm.py
import os
try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google.generativeai not found. GeminiLLM component will not be fully functional.")
    # Define a dummy genai if the actual one is not available for basic script integrity
    class DummyGenAI:
        def configure(self, api_key):
            print(f"DummyGenAI configured with API key: {'*' * len(api_key) if api_key else 'None'}")

        class GenerativeModel:
            def __init__(self, model_name):
                self.model_name = model_name
                print(f"DummyGenerativeModel initialized with model_name: {model_name}")

            def generate_content(self, prompt, generation_config):
                print(f"DummyGenerativeModel generate_content called with prompt: {prompt[:50]}...")
                class DummyResponse:
                    def __init__(self, text):
                        self.text = text
                return DummyResponse(f"Dummy response for prompt: {prompt[:30]}...")

        def GenerativeModel(self, model_name): # Method to mimic genai.GenerativeModel
            return self.GenerativeModel(model_name)

    genai = DummyGenAI()


from morphik import Component, register # Assuming morphik library is available

@register
class GeminiLLM(Component):
    """A Morphik component for interacting with the Google Gemini API."""

    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        print(f"GeminiLLM initialized with model_name={model_name}, temperature={temperature}")

        # Best Practice: Configure the client from environment variables, not config files.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # In a real scenario, this might raise an error or log a more critical warning.
            # For this migration script, we'll print a warning to allow initialization.
            print("Warning: GEMINI_API_KEY not found in environment. GeminiLLM may not function correctly.")
        else:
            try:
                genai.configure(api_key=api_key)
                print("Gemini API configured successfully.")
            except Exception as e:
                print(f"Error configuring Gemini API: {e}")


    def run(self, prompt: str) -> dict:
        """Takes a formatted prompt and returns the LLM's generated text."""
        print(f"GeminiLLM running with prompt: {prompt[:50]}...")
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )
            print("GeminiLLM received response from API.")
            return {"answer": response.text}
        except Exception as e:
            print(f"Error during GeminiLLM run: {e}")
            return {"answer": f"Error generating response: {e}"}
