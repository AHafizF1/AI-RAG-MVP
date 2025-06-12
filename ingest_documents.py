#!/usr/bin/env python3
"""
Document Ingestion Script

This script helps you upload documents to the Pinecone vector store for RAG.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from the project
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from rag.document_loader import load_and_chunk_documents, save_documents_to_vector_store
from rag.vector_store import clear_vector_store, get_vector_store, initialize_pinecone

def main():
    """Main function to handle document ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "paths",
        nargs="+",
        help="File or directory paths to process"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store before ingesting new documents"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively process directories"
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".txt", ".pdf", ".docx", ".md"],
        help="File extensions to include (with leading .)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize Pinecone
        initialize_pinecone()
        
        # Clear vector store if requested
        if args.clear:
            if input("Are you sure you want to clear the vector store? (y/n): ").lower() == 'y':
                if clear_vector_store():
                    logger.info("Vector store cleared successfully")
                else:
                    logger.warning("Failed to clear vector store")
                    return
            else:
                logger.info("Operation cancelled by user")
                return
        
        # Collect files to process
        files_to_process = []
        for path in args.paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Path not found: {path}")
                continue
                
            if path.is_file():
                if path.suffix.lower() in args.extensions:
                    files_to_process.append(path)
                else:
                    logger.warning(f"Skipping unsupported file type: {path}")
            elif path.is_dir():
                if args.recursive:
                    for ext in args.extensions:
                        files_to_process.extend(path.rglob(f"*{ext}"))
                else:
                    for ext in args.extensions:
                        files_to_process.extend(path.glob(f"*{ext}"))
        
        if not files_to_process:
            logger.warning("No files found to process")
            return
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files in batches
        batch_size = 10
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(files_to_process)-1)//batch_size + 1}")
            
            try:
                # Load and chunk documents
                documents = load_and_chunk_documents(batch)
                
                # Save to vector store
                saved_count = save_documents_to_vector_store(documents)
                logger.info(f"Saved {saved_count} document chunks to vector store")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                continue
        
        logger.info("Document ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
