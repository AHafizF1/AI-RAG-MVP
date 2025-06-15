# rag/retriever.py

import logging
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pinecone_text.sparse import BM25Encoder
from .vector_store import hybrid_search, get_embeddings

logger = logging.getLogger(__name__)

# The fix is to swap the order of inheritance here.
# BaseRetriever must come before BaseModel to resolve the metaclass conflict.
class VectorStoreRetriever(BaseRetriever, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    k: int = Field(default=4, ge=1)
    embedding_function: Any
    bm25_encoder: BM25Encoder
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    search_type: str = Field(default='hybrid', description="One of 'hybrid', 'dense', 'sparse'")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        if not query or not query.strip():
            return []
        
        dense_vector, sparse_vector = None, None
        
        if self.search_type in ['hybrid', 'dense']:
            dense_vector = self.embedding_function.embed_query(query)
        
        if self.search_type in ['hybrid', 'sparse']:
            # A fitted encoder has a 'bm25' object that is not None.
            if hasattr(self.bm25_encoder, 'bm25') and self.bm25_encoder.bm25 is not None:
                sparse_results = self.bm25_encoder.encode_queries([query])
                if sparse_results:
                    sparse_vector = sparse_results[0]
            else:
                logger.warning("BM25 encoder is not fitted. Cannot generate sparse vectors for query.")

        if dense_vector is None and sparse_vector is None:
            logger.error("Could not generate any query vectors. Returning no results.")
            return []

        # Handle filters from both initialization and the invoke() call
        combined_filter = self.search_kwargs.get('filter', {}).copy()
        if 'filter' in kwargs:
            combined_filter.update(kwargs['filter'])

        search_results = hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=self.k,
            filter=combined_filter or None
        )
        
        documents = []
        for result in search_results:
            metadata = result.get('metadata', {})
            page_content = metadata.pop('text', '')
            metadata['score'] = result.get('score', 0.0)
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

def get_retriever(
    *, # Enforce keyword arguments for clarity
    bm25_encoder: Optional[BM25Encoder] = None,
    search_type: str = 'hybrid',
    k: int = 4,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> VectorStoreRetriever:
    """
    Creates and returns a configured VectorStoreRetriever.
    """
    from .vector_store import create_bm25_encoder
    
    final_encoder = bm25_encoder
    # If no encoder is passed, create a new, empty one.
    if final_encoder is None:
        logger.info("No BM25 encoder provided; creating a new, unfitted one.")
        final_encoder = create_bm25_encoder()

    return VectorStoreRetriever(
        embedding_function=get_embeddings(),
        bm25_encoder=final_encoder,
        search_type=search_type,
        k=k,
        search_kwargs=search_kwargs or {}
    )