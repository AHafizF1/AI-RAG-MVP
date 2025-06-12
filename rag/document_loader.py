"""
Document Loader Module

This module handles loading and processing documents for the RAG system.
"""
import os
import logging
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Supported file extensions and their corresponding loaders
SUPPORTED_EXTENSIONS = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.md': UnstructuredMarkdownLoader,
}

class UnsupportedFileTypeError(ValueError):
    """Exception raised when a file type is not supported."""
    pass

def get_loader_for_file(file_path: Union[str, Path]):
    """
    Get the appropriate loader for the given file.
    
    Args:
        file_path: Path to the file to load.
        
    Returns:
        A document loader instance.
        
    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext](str(file_path))
    
    # Try with UnstructuredFileLoader as fallback
    try:
        return UnstructuredFileLoader(str(file_path))
    except Exception as e:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}") from e

def load_and_chunk_documents(
    file_paths: Union[str, List[Union[str, Path]]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    show_progress: bool = True
) -> List[Document]:
    """
    Load and chunk documents from the given file paths.
    
    Args:
        file_paths: A single file path or list of file paths to process.
        chunk_size: Size of each chunk in characters. Uses config value if None.
        chunk_overlap: Overlap between chunks in characters. Uses config value if None.
        show_progress: Whether to show a progress bar.
        
    Returns:
        List of document chunks.
        
    Raises:
        FileNotFoundError: If any file is not found.
        UnsupportedFileTypeError: If any file type is not supported.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_docs = []
    
    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files", disable=not show_progress):
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Get appropriate loader
            loader = get_loader_for_file(file_path)
            
            # Load and split documents
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "source": str(file_path.name),
                    "source_path": str(file_path)
                })
            
            all_docs.extend(chunks)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            if isinstance(e, UnsupportedFileTypeError):
                raise
            # Continue with other files on other types of errors
    
    logger.info(f"Processed {len(file_paths)} files into {len(all_docs)} chunks")
    return all_docs

def save_documents_to_vector_store(
    documents: List[Document],
    batch_size: int = 100,
    show_progress: bool = True
) -> int:
    """
    Save documents to the vector store in batches.
    
    Args:
        documents: List of documents to save.
        batch_size: Number of documents to process in each batch.
        show_progress: Whether to show a progress bar.
        
    Returns:
        int: Number of documents successfully saved.
    """
    from .vector_store import get_vector_store
    
    vector_store = get_vector_store()
    if not vector_store:
        logger.error("Vector store not available. Cannot save documents.")
        return 0
    
    total_docs = len(documents)
    if total_docs == 0:
        return 0
    
    saved_count = 0
    
    # Process in batches
    for i in tqdm(
        range(0, total_docs, batch_size),
        desc="Saving to vector store",
        disable=not show_progress
    ):
        batch = documents[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            saved_count += len(batch)
        except Exception as e:
            logger.error(f"Error saving batch {i//batch_size + 1}: {str(e)}")
    
    logger.info(f"Saved {saved_count} documents to vector store")
    return saved_count
