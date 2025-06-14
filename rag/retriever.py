# In rag/retriever.py

import logging
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pinecone_text.sparse import BM25Encoder
from .vector_store import hybrid_search, get_embeddings, create_bm25_encoder

logger = logging.getLogger(__name__)

class VectorStoreRetriever(BaseRetriever, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    k: int = Field(default=4, ge=1)
    embedding_function: Any
    bm25_encoder: BM25Encoder
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        if not query or not query.strip(): return []
        
        dense_vector = self.embedding_function.embed_query(query)
        sparse_vector = None
        if hasattr(self.bm25_encoder, 'idf_'):
            sparse_vector = self.bm25_encoder.encode_queries(query)
        
        combined_filter = self.search_kwargs.get('filter', {})
        if 'filter' in kwargs: combined_filter.update(kwargs['filter'])

        search_results = hybrid_search(
            dense_vector=dense_vector, sparse_vector=sparse_vector,
            top_k=self.k, filter=combined_filter or None
        )
        
        documents = []
        for result in search_results:
            metadata = result.get('metadata', {})
            page_content = metadata.pop('text', '')
            metadata['score'] = result.get('score', 0.0)
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

def get_retriever(bm25_encoder: BM25Encoder, **kwargs: Any) -> VectorStoreRetriever:
    """Factory to create a retriever, injecting the dependencies."""
    config = {
        "embedding_function": get_embeddings(),
        "bm25_encoder": bm25_encoder,
    }
    config.update(kwargs)
    return VectorStoreRetriever(**config)