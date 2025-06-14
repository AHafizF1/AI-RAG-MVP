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
    assert stats.dimension > 0, f"Invalid dimension in stats: {stats.dimension}"
    assert stats.total_vector_count is not None, "Total vector count is None"
    
    logger.info("✅ Pinecone connection test completed successfully!")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the test
    success = test_pinecone_connection()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
