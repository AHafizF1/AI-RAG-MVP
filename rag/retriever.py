"""
Retriever Module

This module provides functionality for retrieving relevant documents using Pinecone's
built-in hybrid search (dense + sparse vector search).
"""
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import Field, BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
import numpy as np

from config import (
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY
)

logger = logging.getLogger(__name__)


def normalize_l2(x):
    """Normalize a vector to unit length using L2 norm."""
    import numpy as np
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / (norm + 1e-6)

class VectorStoreRetriever(BaseRetriever, BaseModel):
    """
    Custom retriever that uses Pinecone's built-in hybrid search.
    
    Hybrid search combines dense vector search (semantic) with sparse vector search (lexical)
    to provide more relevant search results.
    """
    vector_store: Any = Field(..., description="The vector store to use for retrieval")
    k: int = Field(default=4, ge=1, description="Number of documents to retrieve")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, 
                        description="Weight for combining dense and sparse search scores (0.0 = sparse only, 1.0 = dense only)")
    use_hybrid: bool = Field(default=True, description="Whether to use hybrid search")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The Pinecone vector store to use for retrieval.
            k: Number of documents to retrieve.
            alpha: Weight for combining dense and sparse search scores (0.0 = sparse only, 1.0 = dense only).
            use_hybrid: Whether to use hybrid search. If False, falls back to dense search.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**data)
        logger.info(f"Initialized VectorStoreRetriever with hybrid search {'enabled' if self.use_hybrid else 'disabled'}")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        filter: Optional[dict] = None,
        search_type: str = "hybrid",  # 'hybrid', 'dense', or 'sparse'
        **kwargs
    ) -> List[Document]:
        """
        Get documents relevant to the query using Pinecone's hybrid search.
        
        Args:
            query: The query string.
            run_manager: Callback manager for the retriever run.
            filter: Optional filter to apply to the search.
            search_type: Type of search to perform ('hybrid', 'dense', or 'sparse').
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of relevant documents.
        """
        if not self.vector_store:
            logger.warning("Vector store not available for retrieval")
            return []
        
        # Get the underlying Pinecone index
        try:
            pinecone_index = self.vector_store._index
            if not pinecone_index:
                logger.error("Could not access Pinecone index")
                return []
                
            # Prepare query parameters
            query_kwargs = {
                "top_k": self.k,
                "include_metadata": True,
                **kwargs
            }
            
            if search_type == "dense" or not self.use_hybrid:
                # Dense search only
                query_embedding = self.vector_store.embedding_function.embed_query(query)
                query_kwargs["vector"] = query_embedding
                query_kwargs["alpha"] = 1.0  # Dense only
                
            elif search_type == "sparse":
                # Sparse search only - use the query text directly, Pinecone will handle sparse vector generation
                query_kwargs["query_text"] = query
                query_kwargs["alpha"] = 0.0  # Sparse only
                    
            else:  # hybrid
                # Hybrid search - use both dense and sparse vectors
                query_embedding = self.vector_store.embedding_function.embed_query(query)
                query_kwargs["vector"] = query_embedding
                query_kwargs["query_text"] = query  # For sparse vector generation
                query_kwargs["alpha"] = self.alpha  # Balance between dense and sparse
                
            # Apply any additional filters
            if filter:
                query_kwargs["filter"] = filter
                
            # Execute the query
            results = pinecone_index.query(**query_kwargs)
            
            # Convert results to Documents
            documents = []
            for match in results.matches:
                doc_metadata = match.metadata or {}
                doc_metadata["score"] = match.score
                
                # Get the text content from the most likely field
                text = doc_metadata.get("text") or doc_metadata.get("content") or ""
                
                documents.append(
                    Document(
                        page_content=text,
                        metadata=doc_metadata
                    )
                )
                
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
            return []

def get_retriever(
    k: int = 4, 
    alpha: float = 0.5,
    use_hybrid: bool = True,
    **kwargs
) -> Optional[VectorStoreRetriever]:
    """
    Get a retriever instance with support for hybrid search.
    
    Args:
        k: Number of documents to retrieve.
        alpha: Weight for combining dense and sparse search scores (0.0 = sparse only, 1.0 = dense only).
              Values between 0 and 1 balance between sparse and dense search results.
        use_hybrid: Whether to enable hybrid search. If False, falls back to dense search only.
        **kwargs: Additional keyword arguments to pass to the retriever.
        
    Returns:
        A VectorStoreRetriever instance or None if initialization fails.
        
    Example:
        ```python
        # Basic usage with default parameters
        retriever = get_retriever()
        
        # Custom alpha value (0.7 = 70% dense, 30% sparse)
        retriever = get_retriever(alpha=0.7)
        
        # Disable hybrid search (dense search only)
        retriever = get_retriever(use_hybrid=False)
        ```
    """
    from .vector_store import get_vector_store
    
    vector_store = get_vector_store()
    if not vector_store:
        logger.warning("Cannot create retriever: Vector store not available")
        return None
    
    return VectorStoreRetriever(
        vector_store=vector_store, 
        k=k, 
        alpha=alpha,
        use_hybrid=use_hybrid,
        **kwargs
    )

def get_qa_chain(
    llm,
    retriever: Optional[BaseRetriever] = None,
    chain_type: str = "stuff",
    retriever_kwargs: Optional[dict] = None,
    **kwargs
) -> Optional[BaseRetrievalQA]:
    """
    Create a QA chain for question answering with support for hybrid search.
    
    Args:
        llm: The language model to use for answering questions.
        retriever: The retriever to use. If None, a default retriever will be created.
        chain_type: Type of retrieval chain to use (e.g., "stuff", "map_reduce").
        retriever_kwargs: Additional keyword arguments to pass to get_retriever()
                        if a retriever is not provided. This allows configuring
                        hybrid search parameters like 'alpha' and 'use_hybrid'.
        **kwargs: Additional keyword arguments to pass to RetrievalQA.
        
    Returns:
        A RetrievalQA instance or None if initialization fails.
        
    Example:
        ```python
        # Basic usage with default parameters
        qa_chain = get_qa_chain(llm)
        
        # Customize retriever parameters
        qa_chain = get_qa_chain(
            llm,
            chain_type="map_reduce",
            retriever_kwargs={
                "alpha": 0.7,  # 70% dense, 30% sparse
                "k": 5,        # Return top 5 documents
                "use_hybrid": True
            }
        )
        
        # Use a pre-configured retriever
        retriever = get_retriever(alpha=0.3, k=3)
        qa_chain = get_qa_chain(llm, retriever=retriever)
        ```
    """
    if retriever is None:
        retriever_kwargs = retriever_kwargs or {}
        retriever = get_retriever(**retriever_kwargs)
    
    if not retriever:
        logger.error("Cannot create QA chain: Retriever initialization failed")
        return None
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            **kwargs
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}", exc_info=True)
        return None
