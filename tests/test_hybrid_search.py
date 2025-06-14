# In tests/test_hybrid_search.py

import logging
from pathlib import Path
import pytest
from dotenv import load_dotenv
from rag.retriever import get_retriever
from rag.vector_store import clear_vector_store, get_vector_store, create_bm25_encoder
from langchain_core.documents import Document

# ... (logging and TEST_DOCUMENTS constant are the same) ...
TEST_DOCUMENTS = [
    {"text": "Pinecone is a vector database for building AI applications.", "source": "docs", "category": "introduction"},
    {"text": "Hybrid search combines semantic and keyword search for better results.", "source": "blog", "category": "search"},
    {"text": "Vector databases enable efficient similarity search.", "source": "tutorial", "category": "introduction"},
    {"text": "Pinecone's hybrid search uses both sparse and dense vectors.", "source": "docs", "category": "search"},
    {"text": "Semantic search understands the meaning behind queries.", "source": "blog", "category": "search"}
]


# This fixture now has ONE job: provide a FITTED encoder.
@pytest.fixture(scope="module")
def fitted_bm25_encoder():
    clear_vector_store()
    vector_store = get_vector_store()
    documents = [Document(page_content=doc["text"], metadata=doc) for doc in TEST_DOCUMENTS]
    vector_store.add_documents(documents)
    
    bm25_encoder = create_bm25_encoder()
    bm25_encoder.fit([doc.page_content for doc in documents])
    return bm25_encoder

# --- All tests now take the fitted_bm25_encoder as an argument ---

def test_hybrid_search_basic(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, k=3)
    results = retriever.invoke("How does Pinecone's hybrid search work?")
    assert len(results) > 0

def test_hybrid_search_with_alpha(fitted_bm25_encoder):
    query = "vector database search"
    result_contents = {}
    
    # We create a new retriever for each alpha, PASSING IN the fitted encoder.
    for alpha in [0.0, 0.3, 0.7, 1.0]:
        # NOTE: The alpha parameter is not used by the new `hybrid_search`, but we keep it
        # to show how one might pass other parameters.
        retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, k=5)
        results = retriever.invoke(query)
        result_contents[alpha] = {doc.page_content for doc in results}
        if alpha == 0.0:
            assert len(results) > 0, "Sparse search should now return results"
    
    assert result_contents[0.0] != result_contents[1.0]

def test_hybrid_search_edge_cases(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder)
    assert len(retriever.invoke("")) == 0
    assert len(retriever.invoke(" " * 1000 + "vector database")) > 0

def test_hybrid_search_with_filters(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, search_kwargs={'filter': {'source': 'docs'}})
    results = retriever.invoke("search")
    assert len(results) > 0
    for doc in results:
        assert doc.metadata.get('source') == 'docs'

def test_hybrid_search_vs_dense_only(fitted_bm25_encoder):
    query = "search with semantic understanding"
    
    # Create a hybrid retriever by passing the fitted encoder
    hybrid_retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, k=5)
    hybrid_content = {doc.page_content for doc in hybrid_retriever.invoke(query)}
    
    # Create a dense-only retriever by passing an unfitted encoder
    dense_retriever = get_retriever(bm25_encoder=create_bm25_encoder(), k=5)
    dense_content = {doc.page_content for doc in dense_retriever.invoke(query)}
    
    assert hybrid_content != dense_content