# tests/test_document_tool.py
import pytest
from main import document_search_tool # Adjusted import
from langchain.chains import RetrievalQA # This is the import path used in main.py for the object we want to mock

@pytest.fixture(autouse=True)
def mock_retrieval(monkeypatch):
    class FakeQA:
        def run(self, q): return "Fake answer" # main.py's document_search_tool expects qa_chain({"query": query}) and then result["result"]
        def __call__(self, inputs: dict) -> dict: # Match the expected call signature
            # q = inputs["query"] # The query is available here if needed for more complex fake logic
            return {"result": "Fake answer"}


    # The document_search_tool in main.py calls get_qa_chain, which then calls RetrievalQA.from_chain_type
    # So we need to mock rag.retriever.RetrievalQA (the actual location of from_chain_type)
    # However, main.document_search_tool itself calls get_qa_chain() which returns an *instance* of RetrievalQA.
    # The .run method is called on this instance.
    # The issue's example `monkeypatch.setattr(RetrievalQA, "from_chain_type", lambda **kwargs: FakeQA())`
    # aims to make `from_chain_type` return our FakeQA instance.

    # The document_search_tool in main.py does:
    # qa_chain = get_qa_chain(llm=get_llm())
    # result = qa_chain({"query": query})
    # So, we need get_qa_chain to return our FakeQA instance.

    # Let's mock the get_qa_chain function in main.py to return our FakeQA instance.
    # This is simpler and directly controls what document_search_tool uses.
    # We also need to mock get_llm() because document_search_tool calls it before get_qa_chain.
    monkeypatch.setattr("main.get_llm", lambda: "fake_llm_instance") # Actual LLM instance doesn't matter
    monkeypatch.setattr("main.get_qa_chain", lambda llm: FakeQA())

def test_document_search():
    ans = document_search_tool("Any query")
    assert ans == "Fake answer"
