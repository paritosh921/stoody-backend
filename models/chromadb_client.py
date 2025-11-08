import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from config import CHROMADB_PATH, CHROMADB_COLLECTION_NAME
from .question import Question

# Shared ChromaDB client to avoid multiple instance conflicts
_shared_chromadb_client = None

class ChromaDBClient:
    """ChromaDB client for managing question storage and retrieval"""

    def __init__(self, admin_id: str = None):
        self.admin_id = admin_id
        # Handle case where admin_id might already be a collection name
        if admin_id and admin_id.startswith("questions_admin_"):
            self.collection_name = admin_id
        elif admin_id and admin_id.startswith("questions"):
            # Handle legacy collection names
            self.collection_name = admin_id
        elif admin_id:
            self.collection_name = f"questions_admin_{admin_id}"
        else:
            self.collection_name = "questions"

        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Use shared ChromaDB client instance to avoid conflicts
            # Multiple PersistentClient instances with different settings cause errors
            global _shared_chromadb_client
            if '_shared_chromadb_client' not in globals() or _shared_chromadb_client is None:
                _shared_chromadb_client = chromadb.PersistentClient(
                    path=str(CHROMADB_PATH),
                    settings=Settings(
                        anonymized_telemetry=False,
                        # Allow reset to avoid schema conflicts across versions
                        allow_reset=True,
                        is_persistent=True,
                    )
                )
            self.client = _shared_chromadb_client

            # Get or create collection for this admin's questions
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"SkillBot questions for admin {self.admin_id}"}
            )

            logging.info(f"ChromaDB initialized for admin {self.admin_id}. Collection: {self.collection_name}")

        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB for admin {self.admin_id}: {str(e)}")
            raise
    
    def save_question(self, question: Question) -> bool:
        """Save a question to ChromaDB"""
        try:
            document, metadata, question_id = question.to_chromadb_format()
            
            # Check if question already exists
            existing = self.collection.get(ids=[question_id])
            
            if existing['ids']:
                # Update existing question
                self.collection.update(
                    ids=[question_id],
                    documents=[document],
                    metadatas=[metadata]
                )
                logging.info(f"Updated question {question_id}")
            else:
                # Add new question
                self.collection.add(
                    documents=[document],
                    metadatas=[metadata],
                    ids=[question_id]
                )
                logging.info(f"Added new question {question_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save question {question.id}: {str(e)}")
            return False
    
    def save_questions_batch(self, questions: List[Question]) -> Tuple[int, int]:
        """Save multiple questions to ChromaDB"""
        success_count = 0
        total_count = len(questions)
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for question in questions:
                document, metadata, question_id = question.to_chromadb_format()
                documents.append(document)
                metadatas.append(metadata)
                ids.append(question_id)
            
            # Check for existing questions
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing['ids'])
            
            # Separate new and existing questions
            new_docs, new_metas, new_ids = [], [], []
            update_docs, update_metas, update_ids = [], [], []
            
            for i, question_id in enumerate(ids):
                if question_id in existing_ids:
                    update_docs.append(documents[i])
                    update_metas.append(metadatas[i])
                    update_ids.append(question_id)
                else:
                    new_docs.append(documents[i])
                    new_metas.append(metadatas[i])
                    new_ids.append(question_id)
            
            # Add new questions
            if new_docs:
                self.collection.add(
                    documents=new_docs,
                    metadatas=new_metas,
                    ids=new_ids
                )
                success_count += len(new_docs)
            
            # Update existing questions
            if update_docs:
                self.collection.update(
                    documents=update_docs,
                    metadatas=update_metas,
                    ids=update_ids
                )
                success_count += len(update_docs)
            
            logging.info(f"Batch operation completed: {success_count}/{total_count} questions saved")
            return success_count, total_count
            
        except Exception as e:
            logging.error(f"Failed to save questions batch: {str(e)}")
            return success_count, total_count
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Retrieve a specific question by ID"""
        try:
            result = self.collection.get(
                ids=[question_id],
                include=["documents", "metadatas"]
            )
            
            if not result['ids']:
                return None
            
            question = Question.from_chromadb_result(
                document=result['documents'][0],
                metadata=result['metadatas'][0],
                id=result['ids'][0]
            )
            
            return question
            
        except Exception as e:
            logging.error(f"Failed to get question {question_id}: {str(e)}")
            return None
    
    def search_questions(
        self, 
        query: str = None, 
        subject: str = None, 
        difficulty: str = None,
        has_images: bool = None,
        document_type: str = None,
        limit: int = 50
    ) -> List[Question]:
        """Search questions with optional filters including document_type for mode-specific retrieval"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if subject:
                where_clause["subject"] = subject
            if difficulty:
                where_clause["difficulty"] = difficulty
            if has_images is not None:
                where_clause["hasImages"] = has_images
            if document_type:
                where_clause["document_type"] = document_type
            
            if query:
                # Semantic search with query
                results = self.collection.query(
                    query_texts=[query],
                    where=where_clause if where_clause else None,
                    n_results=limit,
                    include=["documents", "metadatas"]
                )
                
                questions = []
                for i, doc_id in enumerate(results['ids'][0]):
                    question = Question.from_chromadb_result(
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        id=doc_id
                    )
                    questions.append(question)
                
                return questions
            else:
                # Get all questions with filters - get all for better randomization
                results = self.collection.get(
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas"]
                )

                questions = []
                for i, doc_id in enumerate(results['ids']):
                    question = Question.from_chromadb_result(
                        document=results['documents'][i],
                        metadata=results['metadatas'][i],
                        id=doc_id
                    )
                    questions.append(question)

                # Apply limit after getting all results for better randomization
                if limit and len(questions) > limit:
                    import random
                    random.shuffle(questions)
                    questions = questions[:limit]

                return questions
                
        except Exception as e:
            logging.error(f"Failed to search questions: {str(e)}")
            return []
    
    def delete_question(self, question_id: str) -> bool:
        """Delete a question from ChromaDB"""
        try:
            self.collection.delete(ids=[question_id])
            logging.info(f"Deleted question {question_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete question {question_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the questions collection"""
        try:
            # Get all questions to calculate stats
            all_questions = self.collection.get(include=["metadatas"])
            
            total_count = len(all_questions['ids'])
            
            if total_count == 0:
                return {
                    "total_questions": 0,
                    "subjects": {},
                    "difficulties": {},
                    "with_images": 0,
                    "total_images": 0
                }
            
            subjects = {}
            difficulties = {}
            with_images = 0
            total_images = 0
            
            for metadata in all_questions['metadatas']:
                # Count subjects
                subject = metadata.get('subject', 'Unknown')
                subjects[subject] = subjects.get(subject, 0) + 1
                
                # Count difficulties
                difficulty = metadata.get('difficulty', 'Unknown')
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                # Count images
                if metadata.get('hasImages', False):
                    with_images += 1
                total_images += metadata.get('imageCount', 0)
            
            return {
                "total_questions": total_count,
                "subjects": subjects,
                "difficulties": difficulties,
                "with_images": with_images,
                "total_images": total_images
            }
            
        except Exception as e:
            logging.error(f"Failed to get collection stats: {str(e)}")
            return {}

# Global instances cache
_chromadb_clients = {}

def get_chromadb_client(admin_id: str = None) -> ChromaDBClient:
    """Get or create ChromaDB client instance for specific admin"""
    if admin_id not in _chromadb_clients:
        _chromadb_clients[admin_id] = ChromaDBClient(admin_id)
    return _chromadb_clients[admin_id]