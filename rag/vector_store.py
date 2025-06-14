# In rag/vector_store.py

import logging
from typing import Optional, Dict, Any, List
from pinecone import Pinecone, ServerlessSpec, Index
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone.exceptions import NotFoundException
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, PINECONE_METRIC, PINECONE_DIMENSIONS
)

logger = logging.getLogger(__name__)
SERVER_SIDE_EMBEDDING_MODEL = "llama-text-embed-v2"
pc: Optional[Pinecone] = None

def initialize_pinecone():
    """Initializes the Pinecone client and ensures the index exists."""
    global pc
    if pc: return
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSIONS, metric=PINECONE_METRIC, spec=spec)
    logger.info("Pinecone initialized and index is ready.")

def get_pinecone_index() -> Index:
    """Gets a Pinecone index instance."""
    if not pc: initialize_pinecone()
    return pc.Index(PINECONE_INDEX_NAME)

def create_bm25_encoder() -> BM25Encoder:
    """Creates a new, unfitted BM25 encoder."""
    return BM25Encoder()

def get_embeddings() -> PineconeEmbeddings:
    """Gets the server-side embeddings model."""
    return PineconeEmbeddings(
        model=SERVER_SIDE_EMBEDDING_MODEL,
        query_params={"input_type": "query"},
        document_params={"input_type": "passage"}
    )

def get_vector_store() -> PineconeVectorStore:
    """Gets a new LangChain PineconeVectorStore instance."""
    if not pc: initialize_pinecone()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        text_key="text"
    )

def clear_vector_store():
    """Clears all vectors from the default namespace in the index."""
    try:
        index = get_pinecone_index()
        index.delete(delete_all=True)
    except Exception as e:
        logger.warning(f"Could not clear vector store (it may be empty): {e}")

def hybrid_search(
    *, dense_vector: List[float], sparse_vector: Optional[Dict], top_k: int,
    filter: Optional[Dict] = None
) -> List[Dict]:
    """Performs a hybrid search using pre-computed vectors."""
    index = get_pinecone_index()
    results = index.query(
        vector=dense_vector, sparse_vector=sparse_vector,
        top_k=top_k, filter=filter, include_metadata=True
    )
    formatted_results = [
        {"id": m.id, "score": m.score, "metadata": m.get("metadata", {})}
        for m in results.get('matches', [])
    ]
    return formatted_results