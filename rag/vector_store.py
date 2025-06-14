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
# Server-side embeddings will be used instead of client-side

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_HOST,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_METRIC,
    PINECONE_DIMENSIONS,
    PINECONE_INDEX_TYPE,
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
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("Successfully connected to Pinecone")
        
        # Check if index exists, create if it doesn't
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            
            # Configure serverless spec for serverless indexes
            if PINECONE_INDEX_TYPE.lower() == 'serverless':
                spec = ServerlessSpec(
                    cloud=PINECONE_CLOUD.lower(),
                    region=PINECONE_REGION.lower()
                )
            else:
                # For pod-based indexes
                spec = PodSpec(
                    environment=PINECONE_CLOUD.lower(),
                    pod_type="p1.x1",
                    pods=1,
                    pod_region=PINECONE_REGION.lower()
                )
            
            # Create the index
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSIONS,
                metric=PINECONE_METRIC,
                spec=spec
            )
            logger.info(f"Created Pinecone index: {PINECONE_INDEX_NAME}")
        else:
            logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
            
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def get_vector_store() -> Optional[PineconeVectorStore]:
    """
    Get or create the Pinecone vector store.
    
    For serverless indexes, we need to use the host parameter when connecting.
    
    Returns:
        Optional[PineconeVectorStore]: The vector store instance or None if initialization failed.
    """
    global vector_store
    
    if vector_store is not None:
        return vector_store
        
    if not pc:
        initialize_pinecone()
        if not pc:
            return None
            
    # For serverless, we need to use the host parameter
    index_kwargs = {}
    if PINECONE_INDEX_TYPE.lower() == 'serverless' and PINECONE_HOST:
        index_kwargs['host'] = PINECONE_HOST
        
    try:
        # Initialize Pinecone client if not already done
        if pc is None:
            initialize_pinecone()
        
        # Check if index exists, create if not
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone serverless index: {PINECONE_INDEX_NAME}")
            
            # Create index with server-side embeddings configuration for Llama Text Embed v2
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_REGION
                ),
                metadata_config={"indexed": ["*"]}  # Ensure all metadata is indexed for filtering
            )
            
            # Wait for index to be ready
            import time
            logger.info("Waiting for index to be ready...")
            time.sleep(30)  # Wait for index to be ready
        
        # Initialize vector store with server-side embeddings using Llama Text Embed v2
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=None,  # Server-side embeddings will be used
            text_key="text"  # Ensure text key is set for documents
        )
        
        # Configure the index to use Llama Text Embed v2 for server-side embeddings
        index = pc.Index(PINECONE_INDEX_NAME)
        index.configure_metadata(
            index_configuration={
                "indexed": ["*"]  # Ensure all metadata is indexed for filtering
            }
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
