"""
RAG (Retrieval-Augmented Generation) Module

This module provides functionality for document retrieval and question answering
using Pinecone as the vector store and OpenAI for embeddings.
"""

from .vector_store import get_vector_store
from .document_loader import load_and_chunk_documents
from .retriever import get_retriever, get_qa_chain

__all__ = [
    'get_vector_store',
    'load_and_chunk_documents',
    'get_retriever',
    'get_qa_chain'
]
