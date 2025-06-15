# tests/test_hybrid_search.py

import logging
from pathlib import Path
import pytest
import time  # Import the time module
from dotenv import load_dotenv
from rag.retriever import get_retriever
from rag.vector_store import clear_vector_store, create_bm25_encoder, upsert_hybrid_documents
from langchain_core.documents import Document

# Load environment variables if you have a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DOCUMENTS = [
    {"text": "Pinecone is a vector database for building AI applications.", "source": "docs", "category": "introduction"},
    {"text": "Hybrid search combines semantic and keyword search for better results.", "source": "blog", "category": "search"},
    {"text": "Vector databases enable efficient similarity search.", "source": "tutorial", "category": "introduction"},
    {"text": "Pinecone's hybrid search uses both sparse and dense vectors.", "source": "docs", "category": "search"},
    {"text": "Semantic search understands the meaning behind queries.", "source": "blog", "category": "search"}
]

@pytest.fixture(scope="module")
def fitted_bm25_encoder():
    clear_vector_store()
    
    docs = [Document(page_content=doc["text"], metadata=doc) for doc in TEST_DOCUMENTS]
    
    bm25_encoder = create_bm25_encoder()
    bm25_encoder.fit([d.page_content for d in docs])
    
    upsert_hybrid_documents(docs, bm25_encoder)
    
    # Add a delay to allow Pinecone's index to become consistent.
    # This is crucial when writing and then immediately reading from a cloud database.
    logger.info("Waiting 10 seconds for Pinecone index to update...")
    time.sleep(10)
    
    return bm25_encoder

def test_hybrid_search_basic(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, k=3)
    results = retriever.invoke("How does Pinecone's hybrid search work?")
    assert len(results) > 0

def test_hybrid_search_is_different_from_sparse_only(fitted_bm25_encoder):
    query = "semantic understanding of vector databases"
    
    hybrid_retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, search_type='hybrid', k=5)
    hybrid_content = {doc.page_content for doc in hybrid_retriever.invoke(query)}
    
    sparse_retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, search_type='sparse', k=5)
    sparse_content = {doc.page_content for doc in sparse_retriever.invoke(query)}
    
    assert len(hybrid_content) > 0
    assert len(sparse_content) > 0
    assert hybrid_content != sparse_content

def test_hybrid_search_edge_cases(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder)
    # An empty query should not error and should return no results.
    assert len(retriever.invoke("")) == 0
    # A valid query should return results.
    assert len(retriever.invoke("vector database")) > 0

def test_hybrid_search_with_filters(fitted_bm25_encoder):
    retriever = get_retriever(bm25_encoder=fitted_bm25_encoder, search_kwargs={'filter': {'source': 'docs'}})
    results = retriever.invoke("search")
    assert len(results) > 0
    for doc in results:
        assert doc.metadata.get('source') == 'docs'