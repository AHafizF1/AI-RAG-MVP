import unittest
from unittest.mock import MagicMock, patch

# Ensure Python can find the morphik_app package
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morphik_app.chat_service import simple_rag_answer

class TestChatService(unittest.TestCase):

    @patch('morphik_app.chat_service.db')
    def test_simple_rag_answer(self, mock_db):
        # Arrange
        mock_response_source = MagicMock()
        mock_response_source.document_name = "source_doc.txt"

        mock_db_response = MagicMock()
        mock_db_response.text = "This is a test answer."
        mock_db_response.sources = [mock_response_source]

        mock_db.query.return_value = mock_db_response

        query_text = "What is testing?"
        user_id = "test_user_456"

        # Act
        result = simple_rag_answer(query_text, user_id)

        # Assert
        # 1. Check if db.query was called correctly
        mock_db.query.assert_called_once_with(
            query=query_text,
            k=4,
            filters={"owner_id": user_id}
        )

        # 2. Check the structure and content of the returned dictionary
        expected_result = {
            "answer": "This is a test answer.",
            "sources": ["source_doc.txt"]
        }
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
