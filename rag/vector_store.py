"""
Vector Store Module

This module handles the initialization and management of the Pinecone vector store.
"""
import os
import logging
from typing import Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)

# Initialize Pinecone client
pc = None
vector_store = None

def initialize_pinecone() -> None:
    """
    Initialize the Pinecone client and create the index if it doesn't exist.
    """
    global pc
    
    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY not set. Vector store will not be available.")
        return
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("Successfully connected to Pinecone")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def get_vector_store() -> Optional[PineconeVectorStore]:
    """
    Get or create the Pinecone vector store.
    
    Returns:
        Optional[PineconeVectorStore]: The vector store instance or None if initialization failed.
    """
    global vector_store
    
    if vector_store is not None:
        return vector_store
    
    if not PINECONE_API_KEY:
        return None
    
    try:
        # Initialize Pinecone client if not already done
        if pc is None:
            initialize_pinecone()
        
        # Check if index exists, create if not
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone serverless index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            import time
            logger.info("Waiting for index to be ready...")
            time.sleep(30)  # Wait for index to be ready
        
        # Initialize embeddings
        # For llama-text-embed-v2, we use the default embedding model
        # as Pinecone will handle the embeddings server-side
        embeddings = OpenAIEmbeddings()
        
        # Initialize vector store
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        
        logger.info(f"Successfully initialized vector store with index: {PINECONE_INDEX_NAME}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return None

def clear_vector_store() -> bool:
    """
    Clear all vectors from the vector store.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if pc and PINECONE_INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(PINECONE_INDEX_NAME)
            index.delete(delete_all=True)
            logger.info("Successfully cleared vector store")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to clear vector store: {str(e)}")
        return False
