import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

# Ensure Python can find the morphik_app package
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morphik_app.ingestion_service import ingest_new_document, Invoice
from morphik.rules import MetadataExtractionRule


class TestIngestionService(unittest.TestCase):

    @patch('morphik_app.ingestion_service.db')
    def test_ingest_new_document(self, mock_db):
        # Arrange
        mock_doc = MagicMock()
        mock_doc.id = "test_doc_id"
        mock_doc.metadata = {"extracted_field": "value"} # Simulate some extracted metadata

        mock_db.ingest_file.return_value = mock_doc

        file_path = "dummy/path/to/file.txt"
        user_id = "test_user_123"

        # Act
        doc_id = ingest_new_document(file_path, user_id)

        # Assert
        # 1. Check if db.ingest_file was called correctly
        expected_rules = [MetadataExtractionRule(schema=Invoice)]
        # Since MetadataExtractionRule instances won't be identical by default comparison,
        # we check the type and the schema of the rule.

        args, kwargs = mock_db.ingest_file.call_args

        self.assertEqual(kwargs['file_path'], file_path)
        self.assertEqual(kwargs['metadata'], {"owner_id": user_id, "category": "invoices"})

        actual_rules = kwargs['rules']
        self.assertEqual(len(actual_rules), 1)
        self.assertIsInstance(actual_rules[0], MetadataExtractionRule)
        self.assertEqual(actual_rules[0].schema, Invoice)

        # 2. Check if doc.wait_for_completion was called
        mock_doc.wait_for_completion.assert_called_once()

        # 3. Check if the document ID is returned
        self.assertEqual(doc_id, "test_doc_id")

if __name__ == '__main__':
    unittest.main()
