# In: tests/components/test_gemini_llm.py
import unittest
import os
from unittest.mock import patch

# Attempt to import the component
try:
    from morphik_pipeline.components.gemini_llm import GeminiLLM
except ImportError as e:
    print(f"Warning: Could not import GeminiLLM for testing due to: {e}. Test class will be a placeholder.")
    GeminiLLM = None # type: ignore

class TestGeminiLLM(unittest.TestCase):

    def setUp(self):
        # Set a dummy API key for tests if not already set
        self.original_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.original_api_key:
            os.environ["GEMINI_API_KEY"] = "dummy_test_api_key"

        if GeminiLLM:
            self.llm_component = GeminiLLM(model_name="gemini-pro-test", temperature=0.1)
        else:
            self.llm_component = None # type: ignore
        print("TestGeminiLLM setUp complete.")

    def tearDown(self):
        # Restore original API key state
        if self.original_api_key is None:
            if "GEMINI_API_KEY" in os.environ: # Check before deleting, in case it was set by something else
                del os.environ["GEMINI_API_KEY"]
        elif self.original_api_key : # if it was not None ensure it is set back
            os.environ["GEMINI_API_KEY"] = self.original_api_key


    @unittest.skipIf(GeminiLLM is None, "GeminiLLM component not available for testing")
    @patch('morphik_pipeline.components.gemini_llm.genai.GenerativeModel') # Mock the external API call
    def test_run_successful_response(self, mock_generative_model):
        # Configure the mock to return a specific response structure
        class MockResponse:
            def __init__(self, text):
                self.text = text

        mock_model_instance = mock_generative_model.return_value
        mock_model_instance.generate_content.return_value = MockResponse("Mocked LLM answer")

        prompt = "What is the capital of France?"
        result = self.llm_component.run(prompt=prompt)

        self.assertIn("answer", result, "Result should contain 'answer' key")
        self.assertEqual(result["answer"], "Mocked LLM answer")
        mock_model_instance.generate_content.assert_called_once_with(
            prompt,
            generation_config={"temperature": 0.1}
        )
        print("test_run_successful_response passed.")

    @unittest.skipIf(GeminiLLM is None, "GeminiLLM component not available for testing")
    def test_initialization_without_api_key(self):
        # Temporarily remove API key for this test
        current_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            # This test primarily checks that initialization doesn't crash if the key is missing,
            # as the component currently prints a warning.
            llm_no_key = GeminiLLM()
            self.assertIsNotNone(llm_no_key, "Component should initialize even without API key (with warning)")
            print("test_initialization_without_api_key passed (component initialized with warning).")
        finally:
            if current_key: # Restore key if it was present
                os.environ["GEMINI_API_KEY"] = current_key
            elif "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"] == "dummy_test_api_key":
                # if current_key was None, and we set a dummy, remove it.
                del os.environ["GEMINI_API_KEY"]

    # Add more tests for error handling, different configurations, etc.

if __name__ == "__main__":
    print("Running tests for GeminiLLM...")
    unittest.main()
