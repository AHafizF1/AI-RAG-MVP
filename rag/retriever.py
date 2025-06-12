"""
Retriever Module

This module provides functionality for retrieving relevant documents using the vector store.
"""
import logging
from typing import List, Optional, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA

from config import (
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY
)

logger = logging.getLogger(__name__)

class VectorStoreRetriever(BaseRetriever):
    """Custom retriever that uses the Pinecone vector store."""
    
    def __init__(self, vector_store, k: int = 4, **kwargs):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The vector store to use for retrieval.
            k: Number of documents to retrieve.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs
    ) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: The query string.
            run_manager: Callback manager for the retriever run.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of relevant documents.
        """
        if not self.vector_store:
            logger.warning("Vector store not available for retrieval")
            return []
            
        try:
            # Use similarity search to find relevant documents
            docs = self.vector_store.similarity_search(
                query=query,
                k=self.k,
                **kwargs
            )
            return docs
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return []

def get_retriever(k: int = 4, **kwargs) -> Optional[VectorStoreRetriever]:
    """
    Get a retriever instance.
    
    Args:
        k: Number of documents to retrieve.
        **kwargs: Additional keyword arguments to pass to the retriever.
        
    Returns:
        A VectorStoreRetriever instance or None if initialization fails.
    """
    from .vector_store import get_vector_store
    
    vector_store = get_vector_store()
    if not vector_store:
        logger.warning("Cannot create retriever: Vector store not available")
        return None
    
    return VectorStoreRetriever(vector_store=vector_store, k=k, **kwargs)

def get_qa_chain(
    llm,
    retriever: Optional[BaseRetriever] = None,
    chain_type: str = "stuff",
    **kwargs
) -> Optional[BaseRetrievalQA]:
    """
    Create a QA chain for question answering.
    
    Args:
        llm: The language model to use for answering questions.
        retriever: The retriever to use. If None, a default retriever will be created.
        chain_type: Type of retrieval chain to use (e.g., "stuff", "map_reduce").
        **kwargs: Additional keyword arguments to pass to RetrievalQA.
        
    Returns:
        A RetrievalQA instance or None if initialization fails.
    """
    if retriever is None:
        retriever = get_retriever()
    
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
        logger.error(f"Error creating QA chain: {str(e)}")
        return None
