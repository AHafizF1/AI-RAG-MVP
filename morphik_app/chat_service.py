from .database import db

def simple_rag_answer(query: str, user_id: str):
    """Gets a direct answer from the knowledge base."""

    # Use filters to only search documents the user has access to
    filters = {"owner_id": user_id}

    response = db.query(
        query=query,
        k=4,  # Retrieve the top 4 most relevant chunks
        filters=filters
    )

    return {
        "answer": response.text,
        "sources": [s.document_name for s in response.sources]
    }
