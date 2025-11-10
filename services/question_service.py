"""
Simplified Question Service for managing question operations
Works with the current DatabaseManager instead of old client classes
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class QuestionService:
    """Service for managing question operations with ChromaDB"""

    def __init__(self, admin_id: str = None):
        self.admin_id = admin_id
        # Use the database manager from the app state instead of old clients
        from main_async import app
        self.db = app.state.db if hasattr(app, 'state') and hasattr(app.state, 'db') else None

    def search_questions(
        self,
        query: str = None,
        limit: int = 100,
        subject: str = None,
        difficulty: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search questions from ChromaDB
        Returns a list of question dictionaries
        """
        try:
            if not self.db or not self.db.chroma_collection:
                logger.warning("ChromaDB not available")
                return []

            # Build metadata filter
            where_filter = {}
            if self.admin_id:
                where_filter["admin_id"] = self.admin_id
            if subject:
                where_filter["subject"] = subject
            if difficulty:
                where_filter["difficulty"] = difficulty

            # Query ChromaDB
            if query:
                results = self.db.chroma_collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_filter if where_filter else None
                )
            else:
                # Get all questions with filter
                results = self.db.chroma_collection.get(
                    limit=limit,
                    where=where_filter if where_filter else None
                )

            # Convert results to question dictionaries
            questions = []
            if 'metadatas' in results:
                metadatas = results['metadatas']
                ids = results.get('ids', [])
                documents = results.get('documents', [])

                for i, metadata in enumerate(metadatas):
                    question = {
                        'id': ids[i] if i < len(ids) else None,
                        'text': documents[i] if i < len(documents) else '',
                        **metadata
                    }
                    questions.append(question)

            return questions

        except Exception as e:
            logger.error(f"Error searching questions: {str(e)}")
            return []

    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get a single question by ID"""
        try:
            if not self.db or not self.db.chroma_collection:
                logger.warning("ChromaDB not available")
                return None

            results = self.db.chroma_collection.get(
                ids=[question_id]
            )

            if results and results.get('metadatas'):
                metadata = results['metadatas'][0]
                document = results['documents'][0] if results.get('documents') else ''
                return {
                    'id': question_id,
                    'text': document,
                    **metadata
                }

            return None

        except Exception as e:
            logger.error(f"Error getting question {question_id}: {str(e)}")
            return None

    def count_questions(self, subject: str = None, difficulty: str = None) -> int:
        """Count questions with optional filters"""
        try:
            questions = self.search_questions(
                limit=10000,  # Get all questions for count
                subject=subject,
                difficulty=difficulty
            )
            return len(questions)

        except Exception as e:
            logger.error(f"Error counting questions: {str(e)}")
            return 0
