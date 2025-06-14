"""
RAG (Retrieval-Augmented Generation) Module

This module provides functionality for document retrieval and question answering
using Pinecone as the vector store with server-side embeddings.
"""

from .vector_store import get_vector_store, initialize_vector_store
from .document_loader import load_and_chunk_documents
from .retriever import get_retriever

__all__ = [
    'get_vector_store',
    'initialize_vector_store',
    'load_and_chunk_documents',
    'get_retriever'
]
