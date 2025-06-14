"""
Pinecone utility functions for vector storage and retrieval with serverless support.
"""
import os
import logging
from typing import List, Optional, Dict, Any, Union
from pinecone import Pinecone, ServerlessSpec, PodSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_HOST,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_METRIC,
    PINECONE_DIMENSIONS,
    PINECONE_INDEX_TYPE
)

logger = logging.getLogger(__name__)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone() -> None:
    """Initialize Pinecone and ensure the index exists."""
    global pc
    
    try:
        # Check if index exists
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
        logger.error(f"Error initializing Pinecone: {str(e)}")
        raise

def get_pinecone_index():
    """Get a connection to the Pinecone index."""
    try:
        # For serverless, we need to connect using the host
        if PINECONE_INDEX_TYPE.lower() == 'serverless' and PINECONE_HOST:
            return pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
        # For pod-based, we can connect directly
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index: {str(e)}")
        raise

def upsert_vectors(vectors: List[Dict[str, Any]], namespace: str = "") -> None:
    """Upsert vectors into the Pinecone index.
    
    Args:
        vectors: List of vector dictionaries with 'id', 'values', and 'metadata' keys
        namespace: Optional namespace for the vectors
    """
    try:
        index = get_pinecone_index()
        # Upsert in batches of 100 (Pinecone's limit)
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            index.upsert(vectors=batch, namespace=namespace)
        logger.info(f"Upserted {len(vectors)} vectors to namespace: {namespace}")
    except Exception as e:
        logger.error(f"Error upserting vectors: {str(e)}")
        raise

def query_vectors(
    query_vector: List[float],
    top_k: int = 5,
    namespace: str = "",
    include_metadata: bool = True
) -> List[Dict[str, Any]]:
    """Query the Pinecone index for similar vectors.
    
    Args:
        query_vector: The query embedding vector
        top_k: Number of results to return
        namespace: Optional namespace to query
        include_metadata: Whether to include metadata in results
        
    Returns:
        List of matching vectors with scores and metadata
    """
    try:
        index = get_pinecone_index()
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata
        )
        return results.get('matches', [])
    except Exception as e:
        logger.error(f"Error querying vectors: {str(e)}")
        raise
