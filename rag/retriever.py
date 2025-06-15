# rag/retriever.py

import logging
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pinecone_text.sparse import BM25Encoder
from pinecone.exceptions import PineconeApiException
from .vector_store import hybrid_search, get_embeddings

logger = logging.getLogger(__name__)

class VectorStoreRetriever(BaseRetriever, BaseModel):
    _bm25_encoder: BM25Encoder = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    k: int = Field(default=4, ge=1)
    embedding_function: Any
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    search_type: str = Field(default='hybrid', description="One of 'hybrid', 'dense', 'sparse'")

    def __init__(self, *, bm25_encoder: BM25Encoder, **data: Any):
        super().__init__(**data)
        self._bm25_encoder = bm25_encoder

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        if not query or not query.strip():
            return []

        dense_vector = None
        sparse_vector = None

        if self.search_type in ['hybrid', 'dense', 'sparse']:
            # Always generate dense for fallback
            try:
                dense_vector = self.embedding_function.embed_query(query)
            except Exception as e:
                logger.warning(f"Error generating dense vector: {e}")

        if self.search_type in ['hybrid', 'sparse']:
            try:
                sparse_results = self._bm25_encoder.encode_queries([query])
                if sparse_results:
                    sparse_vector = sparse_results[0]
                else:
                    logger.warning("BM25 encoder returned no sparse vector for the query.")
            except Exception as e:
                logger.warning(f"BM25 encoder error (likely not fitted): {e}")

        # For pure sparse mode, if sparse failed, allow dense-only fallback
        if self.search_type == 'sparse' and sparse_vector is None:
            logger.info("Sparse vector unavailable; falling back to dense-only search.")
            sparse_vector = None

        combined_filter = self.search_kwargs.get('filter', {}).copy()
        if 'filter' in kwargs:
            combined_filter.update(kwargs['filter'])

        # Attempt query, fallback on sparse unsupported
        try:
            search_results = hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=self.k,
                filter=combined_filter or None
            )
        except PineconeApiException as e:
            if 'does not support sparse values' in str(e):
                logger.warning("Index does not support sparse; retrying dense-only search.")
                search_results = hybrid_search(
                    dense_vector=dense_vector,
                    sparse_vector=None,
                    top_k=self.k,
                    filter=combined_filter or None
                )
            else:
                raise

        documents: List[Document] = []
        for match in search_results:
            metadata = match.get('metadata', {})
            content = metadata.pop('text', '')
            metadata['score'] = match.get('score', 0.0)
            documents.append(Document(page_content=content, metadata=metadata))

        return documents


def get_retriever(
    *,
    bm25_encoder: BM25Encoder,
    search_type: str = 'hybrid',
    k: int = 4,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> VectorStoreRetriever:
    return VectorStoreRetriever(
        embedding_function=get_embeddings(),
        bm25_encoder=bm25_encoder,
        search_type=search_type,
        k=k,
        search_kwargs=search_kwargs or {}
    )
