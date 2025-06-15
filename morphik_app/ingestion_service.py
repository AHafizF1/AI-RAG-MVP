from .database import db
from morphik.rules import MetadataExtractionRule
from pydantic import BaseModel

# Optional: Define a schema for automatic data extraction
class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float

def ingest_new_document(file_path: str, user_id: str):
    """Ingests a document with rich metadata and rules."""

    # Define rules for this ingestion
    # This rule tells Morphik to find and extract Invoice data.
    rules = [MetadataExtractionRule(schema=Invoice)]

    # Define metadata for filtering and access control
    metadata = {"owner_id": user_id, "category": "invoices"}

    # Ingest the file. Morphik does the rest.
    doc = db.ingest_file(
        file_path=file_path,
        metadata=metadata,
        rules=rules
    )
    doc.wait_for_completion() # Ensure ingestion is complete before proceeding
    print(f"Successfully ingested {file_path}. Extracted data: {doc.metadata}")
    return doc.id
