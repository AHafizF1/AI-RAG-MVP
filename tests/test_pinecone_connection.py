"""
Test script to verify Pinecone connection and basic operations.
"""
import logging
import sys
import os
from dotenv import load_dotenv
from config import PINECONE_INDEX_NAME

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pinecone_connection():
    """Test Pinecone connection and basic operations."""
    from rag.pinecone_utils import init_pinecone, get_pinecone_index
    
    logger.info("=== Testing Pinecone Connection ===")
    
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    init_pinecone()
    
    # Get the index
    logger.info("Connecting to Pinecone index...")
    index = get_pinecone_index()
    
    # Get index stats
    logger.info("Fetching index statistics...")
    stats = index.describe_index_stats()
    
    # Log index stats
    logger.info("Pinecone Index Stats:")
    logger.info(f"- Index Name: {PINECONE_INDEX_NAME}")
    logger.info(f"- Dimension: {stats.dimension}")
    logger.info(f"- Total Vectors: {stats.total_vector_count}")
    
    # Log additional stats if available
    if hasattr(stats, 'index_type'):
        logger.info(f"- Index Type: {stats.index_type}")
    if hasattr(stats, 'metric'):
        logger.info(f"- Metric: {stats.metric}")
    
    # Basic assertions
    assert index is not None, "Failed to get Pinecone index"
    assert hasattr(stats, 'dimension'), "Stats missing dimension"
    assert stats.dimension > 0, "Index dimension should be greater than 0"
    assert stats.total_vector_count is not None, "Total vector count is None"
    
    logger.info("✅ Pinecone connection test completed successfully!")

def test_add_document_to_pinecone():
    """Test adding a document to the Pinecone index."""
    from rag.pinecone_utils import init_pinecone, get_pinecone_index, upsert_vectors
    from pathlib import Path
    import hashlib
    
    logger.info("=== Testing Document Addition to Pinecone ===")
    
    # Initialize Pinecone
    init_pinecone()
    index = get_pinecone_index()
    
    # Read test document
    test_doc_path = Path(__file__).parent / "test_document.txt"
    with open(test_doc_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Create a simple embedding (in a real scenario, you'd use your embedding model)
    # For testing, we'll use a dummy embedding with the correct dimension (1024)
    dummy_embedding = [0.1] * 1024  # Replace with actual embedding in production
    
    # Create a unique ID for the document
    doc_id = hashlib.md5(document_text.encode()).hexdigest()
    
    # Prepare the vector data
    vector_data = {
        'id': doc_id,
        'values': dummy_embedding,
        'metadata': {
            'text': document_text[:1000] + '...' if len(document_text) > 1000 else document_text,
            'source': 'test_document.txt',
            'type': 'test_document'
        }
    }
    
    # Add to Pinecone
    logger.info(f"Adding document to Pinecone index: {PINECONE_INDEX_NAME}")
    upsert_vectors([vector_data])
    
    # Verify the document was added
    stats = index.describe_index_stats()
    logger.info(f"Total vectors in index after addition: {stats.total_vector_count}")
    
    # In a real test, you would query the index to verify the document was added
    # For now, we'll just log that we attempted to add it
    logger.info("✅ Document addition test completed. Check Pinecone dashboard for verification.")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the tests
    test_pinecone_connection()
    test_add_document_to_pinecone()
    
    # Exit with appropriate status code
    sys.exit(0)
