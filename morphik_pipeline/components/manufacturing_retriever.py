# In: morphik_pipeline/components/manufacturing_retriever.py
import pickle
from typing import List
from morphik import Component, register # Assuming morphik library is available
from langchain_core.documents import Document

# Placeholder for the actual get_retriever import
# This import will likely need to be adjusted based on the actual project structure
try:
    from rag.retriever import get_retriever
except ImportError:
    print("Warning: rag.retriever.get_retriever not found. Using a dummy function instead for ManufacturingHybridRetriever.")
    # Define a dummy get_retriever if the actual one is not available
    # This allows the component to be defined, though it won't be fully functional
    # without the actual retriever logic.
    class DummyRetriever:
        def invoke(self, query: str) -> List[Document]:
            print(f"DummyRetriever invoked with query: {query}")
            return [Document(page_content=f"Dummy document for query: {query}")]

    def get_retriever(bm25_encoder, search_type="hybrid", k=5):
        print(f"Dummy get_retriever called with search_type={search_type}, k={k}, encoder={bm25_encoder}")
        return DummyRetriever()

@register
class ManufacturingHybridRetriever(Component):
    """A Morphik component wrapping our production hybrid retriever."""

    def __init__(self, encoder_path: str, search_type: str = "hybrid", k: int = 5):
        # Load the pre-fitted encoder once during initialization
        try:
            with open(encoder_path, "rb") as f:
                fitted_encoder = pickle.load(f)
            print(f"Successfully loaded BM25 encoder from {encoder_path}")
        except FileNotFoundError:
            print(f"Warning: Encoder file not found at {encoder_path}. Using None for fitted_encoder.")
            fitted_encoder = None # Allow component to initialize even if file is missing
        except Exception as e:
            print(f"Error loading BM25 encoder from {encoder_path}: {e}. Using None for fitted_encoder.")
            fitted_encoder = None

        # Use our existing, debugged get_retriever factory
        self.retriever = get_retriever(
            bm25_encoder=fitted_encoder,
            search_type=search_type,
            k=k
        )
        print(f"ManufacturingHybridRetriever initialized with search_type={search_type}, k={k}")

    def run(self, query: str) -> dict:
        """Takes a query and returns a dictionary with retrieved documents."""
        if not query:
            print("ManufacturingHybridRetriever received empty query.")
            return {"documents": []}

        print(f"ManufacturingHybridRetriever running with query: {query}")
        results = self.retriever.invoke(query)
        # Best Practice: Return a structured dictionary, not just the raw list.
        return {"documents": results}

# Ensure the __init__.py in components directory makes these components discoverable
# For example, by importing them in morphik_pipeline/components/__init__.py if necessary
