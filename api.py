"""
FastAPI application for the RAG ingestion service.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag_loader import RAGIngestionPipeline, DocumentIngestionError

# Initialize FastAPI app
app = FastAPI(
    title="RAG Ingestion API",
    description="API for ingesting documents into Pinecone for RAG systems",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class IngestResponse(BaseModel):
    status: str
    message: Optional[str] = None
    chunks_ingested: Optional[int] = None
    documents_processed: Optional[int] = None
    namespace: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

# Initialize pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on application startup."""
    try:
        # The pipeline will be initialized lazily when needed
        pass
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {str(e)}")
        raise

def get_ingestion_pipeline():
    """Dependency to get an instance of the RAG ingestion pipeline."""
    try:
        return RAGIngestionPipeline()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize ingestion pipeline: {str(e)}"
        )

@app.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def ingest_document(
    files: List[UploadFile] = File(..., description="Files to ingest"),
    metadata: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = "default",
    pipeline: RAGIngestionPipeline = Depends(get_ingestion_pipeline)
):
    """
    Ingest one or more documents into Pinecone.
    
    Supports various document formats including PDF, DOCX, TXT, and CSV.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for ingestion"
        )
    
    temp_files = []
    try:
        # Save uploaded files to temporary location
        for file in files:
            # Validate file type
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in ['.pdf', '.docx', '.doc', '.txt', '.csv']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_extension}"
                )
            
            # Create a temporary file with the same extension
            temp_file = tempfile.NamedTemporaryFile(
                suffix=file_extension,
                delete=False
            )
            temp_files.append(temp_file.name)
            
            # Write uploaded file to temporary file
            contents = file.file.read()
            temp_file.write(contents)
            temp_file.close()
        
        # Process the documents
        result = pipeline.ingest_documents(
            file_paths=temp_files,
            metadata=metadata,
            namespace=namespace
        )
        
        return {
            "status": "success",
            "chunks_ingested": result.get("chunks_ingested", 0),
            "documents_processed": len(files),
            "namespace": namespace,
            "message": "Documents processed successfully"
        }
        
    except DocumentIngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ingestion failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during ingestion: {str(e)}"
        )
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_file}: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
