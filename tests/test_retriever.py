"""
Tests for the retriever module with Pinecone's hybrid search functionality.
"""
import os
import unittest
import logging
import numpy as np
from unittest.mock import patch, MagicMock, call
from langchain_core.documents import Document
from rag.retriever import VectorStoreRetriever, get_retriever, normalize_l2

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TestVectorStoreRetriever(unittest.TestCase):
    """Test cases for VectorStoreRetriever with hybrid search."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock vector store with necessary methods
        self.vector_store = MagicMock()
        self.vector_store.embedding_function = MagicMock()
        self.vector_store.embedding_function.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock Pinecone index
        self.pinecone_index = MagicMock()
        self.vector_store._index = self.pinecone_index
        
        # Mock search results
        self.mock_results = MagicMock()
        self.mock_results.matches = [
            MagicMock(
                id="doc1",
                score=0.9,
                metadata={"text": "Test document 1", "source": "test"}
            ),
            MagicMock(
                id="doc2",
                score=0.8,
                metadata={"text": "Test document 2", "source": "test"}
            )
        ]
        self.pinecone_index.query.return_value = self.mock_results
        
        # Initialize retriever with test config
        self.retriever = VectorStoreRetriever(
            vector_store=self.vector_store,
            k=2,
            alpha=0.5,
            use_hybrid=True
        )
    
    def test_normalize_l2(self):
        """Test L2 normalization of vectors."""
        logger.info("=== Starting L2 normalization test ===")
        
        # Test vector
        vector = [1, 2, 3]
        logger.info(f"Original vector: {vector}")
        
        # Normalize
        normalized = normalize_l2(np.array(vector))
        norm = np.linalg.norm(normalized)
        
        logger.info(f"Normalized vector: {normalized}")
        logger.info(f"L2 norm: {norm:.6f} (expected: ~1.0)")
        
        # Verify
        self.assertAlmostEqual(
            first=norm,
            second=1.0,
            places=5,
            msg="L2 norm of normalized vector should be approximately 1.0"
        )
    
    def test_hybrid_search(self):
        """Test hybrid search with both dense and sparse vectors."""
        logger.info("=== Starting hybrid search test ===")
        
        # Perform search
        query = "test query"
        results = self.retriever._get_relevant_documents(query, search_type="hybrid")
        
        # Log search results
        logger.info(f"Search query: {query}")
        logger.info(f"Retrieved {len(results)} documents")
        for i, doc in enumerate(results, 1):
            logger.info(f"Document {i} - Score: {doc.metadata.get('score')}")
            logger.info(f"Content: {doc.page_content[:100]}...")
        
        # Verify results
        self.assertEqual(len(results), 2, "Should return exactly 2 documents")
        self.assertIsInstance(results[0], Document, "Result should be a Document object")
        self.assertEqual(results[0].page_content, "Test document 1", 
                        "First document content should match expected")
        
        # Verify Pinecone query was called with correct parameters
        self.pinecone_index.query.assert_called_once()
        call_args = self.pinecone_index.query.call_args[1]
        
        logger.info("Verifying Pinecone query parameters:")
        logger.info(f"- top_k: {call_args.get('top_k')} (expected: 2)")
        logger.info(f"- alpha: {call_args.get('alpha')} (expected: 0.5)")
        logger.info(f"- include_metadata: {call_args.get('include_metadata')} (expected: True)")
        
        self.assertEqual(call_args["top_k"], 2, "top_k should be 2")
        self.assertIn("vector", call_args, "Dense vector should be included")
        self.assertIn("query_text", call_args, "Query text should be included for sparse vector")
        self.assertEqual(call_args["alpha"], 0.5, "Alpha should be 0.5 for hybrid search")
        self.assertTrue(call_args["include_metadata"], "Metadata should be included")
    
    def test_dense_search(self):
        """Test dense search only."""
        logger.info("=== Starting dense search test ===")
        
        # Perform search with dense only
        query = "dense vector test"
        results = self.retriever._get_relevant_documents(query, search_type="dense")
        
        # Log search results
        logger.info(f"Dense search query: {query}")
        logger.info(f"Retrieved {len(results)} documents")
        
        # Verify results
        self.assertEqual(len(results), 2, "Should return exactly 2 documents")
        
        # Verify Pinecone query was called with dense vector only
        call_args = self.pinecone_index.query.call_args[1]
        
        logger.info("Verifying dense search parameters:")
        logger.info(f"- alpha: {call_args.get('alpha')} (expected: 1.0 for dense only)")
        
        self.assertEqual(call_args["alpha"], 1.0, "Alpha should be 1.0 for dense search")
        self.assertIn("vector", call_args, "Dense vector should be included")
        self.assertNotIn("query_text", call_args, 
                        "Query text should not be included for dense-only search")
    
    def test_sparse_search(self):
        """Test sparse-only search."""
        logger.info("=== Starting sparse search test ===")
        
        # Perform sparse search
        query = "sparse vector test"
        results = self.retriever._get_relevant_documents(query, search_type="sparse")
        
        # Log search results
        logger.info(f"Sparse search query: {query}")
        logger.info(f"Retrieved {len(results)} documents")
        
        # Verify results
        self.assertEqual(len(results), 2, "Should return exactly 2 documents")
        
        # Verify Pinecone query was called with query_text for sparse vector generation
        call_args = self.pinecone_index.query.call_args[1]
        
        logger.info("Verifying sparse search parameters:")
        logger.info(f"- alpha: {call_args.get('alpha')} (expected: 0.0 for sparse only)")
        
        self.assertEqual(call_args["alpha"], 0.0, "Alpha should be 0.0 for sparse search")
        self.assertIn("query_text", call_args, "Query text should be included for sparse search")
        self.assertNotIn("vector", call_args, 
                        "Dense vector should not be included in sparse-only search")

class TestGetRetriever(unittest.TestCase):
    """Test cases for get_retriever function."""
    
    @patch('rag.vector_store.get_vector_store')
    def test_get_retriever_default(self, mock_get_vector_store):
        """Test get_retriever with default parameters."""
        logger.info("=== Testing get_retriever with default parameters ===")
        
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        logger.info("Calling get_retriever() with default parameters")
        retriever = get_retriever()
        
        # Verify retriever was created with default parameters
        logger.info("Verifying default parameters:")
        logger.info(f"- Type: {type(retriever).__name__} (expected: VectorStoreRetriever)")
        logger.info(f"- k: {retriever.k} (expected: 4)")
        logger.info(f"- alpha: {retriever.alpha} (expected: 0.5)")
        logger.info(f"- use_hybrid: {retriever.use_hybrid} (expected: True)")
        
        self.assertIsInstance(retriever, VectorStoreRetriever, 
                            "Should return VectorStoreRetriever instance")
        self.assertEqual(retriever.k, 4, "Default k should be 4")
        self.assertEqual(retriever.alpha, 0.5, "Default alpha should be 0.5")
        self.assertTrue(retriever.use_hybrid, "Hybrid search should be enabled by default")
    
    @patch('rag.vector_store.get_vector_store')
    def test_get_retriever_custom_params(self, mock_get_vector_store):
        """Test get_retriever with custom parameters."""
        logger.info("=== Testing get_retriever with custom parameters ===")
        
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        # Custom parameters
        custom_params = {
            'k': 5,
            'alpha': 0.7,
            'use_hybrid': False
        }
        
        logger.info(f"Calling get_retriever() with custom parameters: {custom_params}")
        retriever = get_retriever(**custom_params)
        
        # Verify retriever was created with custom parameters
        logger.info("Verifying custom parameters:")
        logger.info(f"- k: {retriever.k} (expected: 5)")
        logger.info(f"- alpha: {retriever.alpha} (expected: 0.7)")
        logger.info(f"- use_hybrid: {retriever.use_hybrid} (expected: False)")
        
        self.assertEqual(retriever.k, 5, "Custom k should be 5")
        self.assertEqual(retriever.alpha, 0.7, "Custom alpha should be 0.7")
        self.assertFalse(retriever.use_hybrid, "Hybrid search should be disabled with custom params")

if __name__ == "__main__":
    unittest.main()
