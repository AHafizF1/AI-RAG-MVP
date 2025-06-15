# rag/vector_store.py

import logging
from typing import Optional, Dict, Any, List
from pinecone import Pinecone, ServerlessSpec, Index
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.documents import Document
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

def upsert_hybrid_documents(documents: List[Document], bm25_encoder: BM25Encoder):
    """
    Performs a robust two-step upsert for hybrid search with server-side embeddings.
    Step 1: Adds documents with dense vectors using LangChain's PineconeVectorStore.
    Step 2: Updates the vectors with sparse values using the raw Pinecone client.
    """
    logger.info(f"Starting robust hybrid upsert for {len(documents)} documents.")
    index = get_pinecone_index()
    vector_store = get_vector_store()

    # Generate predictable IDs so we can update the correct vectors later.
    doc_ids = [f"test_doc_{i}" for i in range(len(documents))]

    # Step 1: Add documents using the robust LangChain method.
    # This correctly handles server-side dense vector embeddings.
    vector_store.add_documents(documents, ids=doc_ids)
    logger.info(f"Upserted {len(documents)} documents for dense embeddings.")

    # Step 2: Create sparse vectors and update the records in Pinecone.
    logger.info("Encoding and updating with sparse vectors...")
    doc_contents = [d.page_content for d in documents]
    sparse_vectors = bm25_encoder.encode_documents(doc_contents)

    # The `update` operation is done one-by-one. For a small number of test
    # documents, this is perfectly fine.
    for i, doc_id in enumerate(doc_ids):
        index.update(id=doc_id, sparse_values=sparse_vectors[i])
        
    logger.info(f"Updated {len(doc_ids)} vectors with their sparse values.")


def hybrid_search(
    *, top_k: int,
    dense_vector: Optional[List[float]] = None,
    sparse_vector: Optional[Dict] = None,
    filter: Optional[Dict] = None
) -> List[Dict]:
    """Performs search in Pinecone using one or both vector types."""
    index = get_pinecone_index()
    
    if dense_vector is None and sparse_vector is None:
        raise ValueError("At least one of dense_vector or sparse_vector must be provided.")
        
    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        filter=filter,
        include_metadata=True
    )
    formatted_results = [
        {"id": m.id, "score": m.score, "metadata": m.get("metadata", {})}
        for m in results.get('matches', [])
    ]
    return formatted_results