# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

# This fixture aims to set environment variables early.
# However, module imports might happen before this in a full test suite.
@pytest.fixture(autouse=True)
def set_initial_env_vars_for_api_module(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai") # Attempt to set default for when main is first imported
    monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key")
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini_key") # To prevent ADC error if Gemini path taken on initial import

# Import main and its components after attempting to set environment variables.
import main # Import the main module itself
from main import app # Import 'app' from main for TestClient

class MockAgentExecutor:
    async def ainvoke(self, inputs: dict) -> dict:
        return {"output": "response"}
    def invoke(self, inputs: dict) -> dict: # Synchronous version
        return {"output": "response"}

@pytest.fixture
def mock_and_configure_main_for_openai_chat(monkeypatch):
    """
    Mocks dependencies within main.py and ensures it's configured for OpenAI
    for the /api/chat endpoint test.
    This runs AFTER main has been imported but BEFORE the test execution.
    """

    # 1. Force MODEL_PROVIDER in 'main' module to be OpenAI.
    # 'main.ModelProvider' is the enum class imported from config.
    monkeypatch.setattr(main, "MODEL_PROVIDER", main.ModelProvider.OPENAI)

    # 2. Ensure relevant API keys in 'main' module are set for OpenAI.
    # (main.py imports OPENAI_API_KEY from config directly into its namespace)
    monkeypatch.setattr(main, "OPENAI_API_KEY", "fixture_openai_key")

    # 3. Mock the agent_executor in 'main' module.
    # This is what the OpenAI path in the endpoint will use.
    monkeypatch.setattr(main, "agent_executor", MockAgentExecutor())

    # 4. Mock get_llm in 'main' module.
    # setup_agent() in main.py calls get_llm(). If MODEL_PROVIDER was OpenAI upon
    # main.py's initial import, setup_agent would have tried to create a real OpenAI LLM.
    # This mock prevents issues if get_llm is called again or if agent_executor was None.
    class MockLLM:
        pass
    monkeypatch.setattr(main, "get_llm", lambda: MockLLM())

    # 5. Safeguard: Mock Gemini components in main.py's 'genai' alias.
    # This is to prevent errors if any Gemini-related code in main.py (at module level)
    # was executed during initial import due to MODEL_PROVIDER being 'gemini' at that specific moment.
    if hasattr(main, 'genai'): # 'genai' is the alias for google.generativeai in main.py
        monkeypatch.setattr(main.genai, "configure", lambda api_key=None: None)

        class MockChat:
            def send_message(self, query):
                # This response should NOT be seen if the test is successful.
                class MockResponse: text = "gemini_path_safeguard_mock_response"
                return MockResponse()

        class MockGenModel:
            def start_chat(self, history=None):
                return MockChat()

        monkeypatch.setattr(main.genai, "GenerativeModel", lambda model_name, generation_config=None, safety_settings=None: MockGenModel())

        # Also, ensure main.chat (the Gemini chat object) is None if it exists,
        # as the OpenAI path should be taken.
        if hasattr(main, 'chat'):
            monkeypatch.setattr(main, "chat", None)


def test_chat_endpoint_empty(mock_and_configure_main_for_openai_chat): # Use the fixture
    client = TestClient(app) # app is from the already imported main module

    res = client.post("/api/chat", json={"query": "policy"})

    # Debug: print response if test fails
    if res.status_code != 200:
        print(f"Response status: {res.status_code}")
        try:
            print(f"Response JSON: {res.json()}")
        except:
            print(f"Response text: {res.text}")

    assert res.status_code == 200
    json_response = res.json()
    assert "response" in json_response
    assert json_response["response"] == "response"
