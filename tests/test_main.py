import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure Python can find the morphik_app package
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morphik_app.main import app

# Define the content of our "golden document"
GOLDEN_DOCUMENT_CONTENT = "The quick brown fox jumps over the lazy dog. This is a unique sentence for testing."
GOLDEN_DOCUMENT_NAME = "golden_document.txt"
USER_ID = "e2e_test_user"

class TestMainE2E(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch('morphik_app.database.db') # Mock the db instance used by services
    def test_chat_endpoint_golden_document_scenario(self, mock_morphik_db):
        # --- Arrange ---

        # 1. Mock Morphik's ingest_file behavior
        mock_ingested_doc = MagicMock()
        mock_ingested_doc.id = "golden_doc_id_123"
        # We don't strictly need to check doc.metadata in this E2E test's focus,
        # but it's good practice if ingestion service relies on it.
        mock_ingested_doc.metadata = {"owner_id": USER_ID, "category": "invoices"}

        # Simulate that wait_for_completion does nothing problematic
        mock_ingested_doc.wait_for_completion = MagicMock()

        mock_morphik_db.ingest_file.return_value = mock_ingested_doc

        # 2. Mock Morphik's query behavior for the specific question
        mock_query_response_source = MagicMock()
        mock_query_response_source.document_name = GOLDEN_DOCUMENT_NAME

        mock_db_query_response = MagicMock()
        mock_db_query_response.text = "The answer is related to the quick brown fox."
        mock_db_query_response.sources = [mock_query_response_source]

        # Set up the mock for the query method
        # This will be called by simple_rag_answer via the /chat endpoint
        mock_morphik_db.query.return_value = mock_db_query_response

        # --- Act ---

        # Simulate Ingestion (though it's mocked, we call the endpoint/service that would do it)
        # For a true E2E, you might have a separate ingestion endpoint or call the service directly.
        # Here, we're focusing on the /chat part, assuming ingestion happened.
        # To simplify, we'll assume the document is "ingested" by the mock setup above.
        # If there was an actual /ingest endpoint, we would call:
        # self.client.post("/ingest", json={"file_path": GOLDEN_DOCUMENT_NAME, "user_id": USER_ID})
        # For now, the mock `db.ingest_file` is ready if any service calls it.

        # Query the /chat endpoint
        chat_payload = {
            "query": "What do you know about a quick brown fox?",
            "user_id": USER_ID,
            "use_agent": False
        }
        response = self.client.post("/chat", json=chat_payload)

        # --- Assert ---
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["answer"], "The answer is related to the quick brown fox.")
        self.assertIn(GOLDEN_DOCUMENT_NAME, response_data["details"]) # 'details' contains sources

        # Verify that db.query was called with expected parameters by the chat_service
        mock_morphik_db.query.assert_called_once_with(
            query="What do you know about a quick brown fox?",
            k=4, # As defined in simple_rag_answer
            filters={"owner_id": USER_ID}
        )

        # Optional: Verify ingestion mock if an ingestion step was explicitly called
        # mock_morphik_db.ingest_file.assert_called_once()
        # This depends on whether the test flow includes an explicit ingestion call.
        # Based on the "Golden Document" strategy, ingestion is a prerequisite.
        # If we had an /ingest endpoint, we'd check its call.
        # If ingestion is done by directly calling `ingest_new_document` in test setup,
        # then this mock patch on `morphik_app.database.db` covers that too.

if __name__ == '__main__':
    # Create a dummy golden_document.txt for illustrative purposes if needed by actual ingestion.
    # However, with full mocking of db.ingest_file, the file itself isn't read by the mock.
    # if not os.path.exists(GOLDEN_DOCUMENT_NAME):
    #    with open(GOLDEN_DOCUMENT_NAME, "w") as f:
    #        f.write(GOLDEN_DOCUMENT_CONTENT)

    unittest.main()
