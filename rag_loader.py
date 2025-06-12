"""
Document ingestion pipeline for RAG system.
Handles loading, splitting, and embedding documents into Pinecone.
"""
import os
import hashlib
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import logging
import pinecone

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from typing import Optional, Dict, Any
import pinecone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DocumentIngestionError(Exception):
    """Custom exception for document ingestion errors."""
    pass

class DocumentLoader:
    """Handles loading documents from various file formats."""
    
    @staticmethod
    def get_loader(file_path: str):
        """Get appropriate loader based on file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return PyPDFLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif file_ext == '.csv':
            return CSVLoader(file_path)
        elif file_ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        else:
            # Fallback to unstructured loader for other file types
            return UnstructuredFileLoader(file_path)
    
    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document from file path."""
        try:
            file_path = str(file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            loader = cls.get_loader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise DocumentIngestionError(f"Failed to load document: {str(e)}")

class RAGIngestionPipeline:
    """Pipeline for ingesting documents into Pinecone vector store."""
    
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        namespace: str = "default"
    ):
        """Initialize the ingestion pipeline with Pinecone's integrated embeddings."""
        # Load environment variables if not provided
        load_dotenv()
        
        # Initialize instance variables
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = pinecone_index_name or os.getenv("PINECONE_INDEX_NAME")
        self.namespace = namespace
        
        if not all([self.pinecone_api_key, self.pinecone_index_name]):
            raise ValueError("Missing required Pinecone configuration. Please provide API key and index name.")
        
        # Initialize Pinecone client
        self.pinecone = pinecone.Pinecone(api_key=self.pinecone_api_key)
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize Pinecone index
        self._init_pinecone_index()
    
    def _init_pinecone_index(self):
        """Initialize or create Pinecone index with integrated embeddings."""
        try:
            # Get list of existing indexes
            existing_indexes = self.pinecone.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            # Create index if it doesn't exist
            if self.pinecone_index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.pinecone_index_name}")
                self.pinecone.create_index(
                    name=self.pinecone_index_name,
                    dimension=1024,  # Dimension for llama-text-embed-v2
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(10)  # Give it some time to initialize
            
            # Connect to the index
            self.index = self.pinecone.Index(self.pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {self.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise DocumentIngestionError(f"Pinecone index initialization failed: {str(e)}")
    
    def process_document(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process a single document: load, split, and return chunks."""
        try:
            # Load document
            docs = DocumentLoader.load_document(file_path)
            
            # Add source metadata
            source_name = Path(file_path).name
            for doc in docs:
                doc.metadata.update({
                    "source": source_name,
                    "page": doc.metadata.get("page", 0),
                    "source_type": Path(file_path).suffix.lower().lstrip('.')
                })
                if metadata:
                    doc.metadata.update(metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Split {len(docs)} pages into {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise DocumentIngestionError(f"Document processing failed: {str(e)}")
    
    def _get_embedding_payload(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the payload for Pinecone's integrated embeddings."""
        return {
            'id': f"doc_{hashlib.md5(text.encode()).hexdigest()}",
            'values': text,  # Text will be embedded by Pinecone
            'metadata': metadata
        }
    
    def ingest_documents(
        self,
        file_paths: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents into Pinecone using integrated embeddings.
        
        Args:
            file_paths: List of file paths to process
            metadata: Additional metadata to add to all documents
            namespace: Pinecone namespace to use (overrides default)
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            all_chunks = []
            target_namespace = namespace or self.namespace
            
            # Process each file
            for file_path in file_paths:
                file_path = str(file_path)
                logger.info(f"Processing file: {file_path}")
                
                # Load document
                docs = DocumentLoader.load_document(file_path)
                if not docs:
                    logger.warning(f"No content found in {file_path}")
                    continue
                
                # Split document into chunks
                splits = self.text_splitter.split_documents(docs)
                logger.info(f"Split into {len(splits)} chunks")
                
                # Add file metadata to each chunk
                for chunk in splits:
                    chunk_metadata = {
                        'source': file_path,
                        'chunk_id': hashlib.md5(chunk.page_content.encode()).hexdigest(),
                        'text': chunk.page_content
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunk.metadata = chunk_metadata
                
                all_chunks.extend(splits)
            
            if not all_chunks:
                raise DocumentIngestionError("No valid content found in any documents")
            
            # Prepare records for Pinecone
            records = [
                self._get_embedding_payload(
                    text=chunk.page_content,
                    metadata=chunk.metadata
                )
                for chunk in all_chunks
            ]
            
            # Process in batches
            total_ingested = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=target_namespace
                )
                total_ingested += len(batch)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
            
            logger.info(f"Successfully ingested {total_ingested} chunks into Pinecone")
            
            return {
                "status": "success",
                "chunks_ingested": total_ingested,
                "documents_processed": len(file_paths),
                "index_name": self.pinecone_index_name,
                "namespace": target_namespace
            }
            
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            raise DocumentIngestionError(f"Failed to ingest documents: {str(e)}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone for RAG")
    parser.add_argument("files", nargs="+", help="Files to ingest")
    parser.add_argument("--namespace", default="default", help="Pinecone namespace")
    args = parser.parse_args()
    
    try:
        pipeline = RAGIngestionPipeline(namespace=args.namespace)
        result = pipeline.ingest_documents(args.files)
        print(f"Ingestion result: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
