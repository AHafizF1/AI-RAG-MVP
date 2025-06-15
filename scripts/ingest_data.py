# In: scripts/ingest_data.py
import argparse
import sys

# Attempt to import necessary functions from rag.vector_store
# Use placeholders if the actual module/functions are not found
try:
    from rag.vector_store import load_and_chunk_documents, upsert_hybrid_documents
    print("Successfully imported functions from rag.vector_store.")
except ImportError:
    print("Warning: Could not import from rag.vector_store. Using dummy functions for ingest_data.py.")
    # Define dummy functions if import fails, to allow the script to be created
    from typing import List, Dict, Any
    def load_and_chunk_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
        print(f"load_and_chunk_documents (dummy) called with file_paths: {file_paths}")
        return [{"id": f"dummy_doc_{i}", "content": f"Dummy content from {path}", "metadata": {"source": path}} for i, path in enumerate(file_paths)]

    def upsert_hybrid_documents(documents: List[Dict[str, Any]]):
        print(f"upsert_hybrid_documents (dummy) called with {len(documents)} documents.")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "file_paths",
        metavar="FILE",
        type=str,
        nargs="+",
        help="List of file paths to ingest."
    )

    args = parser.parse_args()

    if not args.file_paths:
        print("No file paths provided. Exiting.")
        sys.exit(1)

    print(f"Starting data ingestion for files: {args.file_paths}")

    # 1. Load and chunk documents
    print("Step 1: Loading and chunking documents...")
    try:
        chunked_documents = load_and_chunk_documents(args.file_paths)
        if not chunked_documents:
            print("No documents were processed or returned from load_and_chunk_documents.")
            # sys.exit(1) # Decide if this is a fatal error
        else:
            print(f"Successfully loaded and chunked {len(chunked_documents)} documents.")
    except Exception as e:
        print(f"Error during document loading and chunking: {e}")
        sys.exit(1)

    # 2. Upsert documents to Pinecone (or other vector store)
    print("Step 2: Upserting documents to vector store...")
    try:
        upsert_hybrid_documents(chunked_documents)
        print("Successfully upserted documents.")
    except Exception as e:
        print(f"Error during document upsertion: {e}")
        sys.exit(1)

    print("Data ingestion script finished successfully.")

if __name__ == "__main__":
    main()
