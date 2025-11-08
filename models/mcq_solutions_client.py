import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from config import CHROMADB_PATH
from .mcq_solution import MCQSolution

# Collection name for MCQ solutions
MCQ_SOLUTIONS_COLLECTION_NAME = "mcq_solutions"

class MCQSolutionsClient:
    """ChromaDB client for managing MCQ solution storage and retrieval"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(CHROMADB_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for MCQ solutions
            self.collection = self.client.get_or_create_collection(
                name=MCQ_SOLUTIONS_COLLECTION_NAME,
                metadata={"description": "MCQ solutions and explanations"}
            )
            
            logging.info(f"MCQ Solutions ChromaDB initialized successfully. Collection: {MCQ_SOLUTIONS_COLLECTION_NAME}")
            
        except Exception as e:
            logging.error(f"Failed to initialize MCQ Solutions ChromaDB: {str(e)}")
            raise
    
    def save_solution(self, solution: MCQSolution) -> bool:
        """Save an MCQ solution to ChromaDB"""
        try:
            document, metadata, solution_id = solution.to_chromadb_format()
            
            # Check if solution already exists
            existing = self.collection.get(ids=[solution_id])
            
            if existing['ids']:
                # Update existing solution
                self.collection.update(
                    ids=[solution_id],
                    documents=[document],
                    metadatas=[metadata]
                )
                logging.info(f"Updated MCQ solution {solution_id}")
            else:
                # Add new solution
                self.collection.add(
                    documents=[document],
                    metadatas=[metadata],
                    ids=[solution_id]
                )
                logging.info(f"Added new MCQ solution {solution_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save MCQ solution {solution.id}: {str(e)}")
            return False
    
    def get_solution(self, question_id: str) -> Optional[MCQSolution]:
        """Retrieve a solution by question ID"""
        try:
            result = self.collection.get(
                where={"question_id": question_id},
                include=["documents", "metadatas"]
            )
            
            if not result['ids']:
                return None
            
            solution = MCQSolution.from_chromadb_result(
                document=result['documents'][0],
                metadata=result['metadatas'][0],
                id=result['ids'][0]
            )
            
            return solution
            
        except Exception as e:
            logging.error(f"Failed to get MCQ solution for question {question_id}: {str(e)}")
            return None
    
    def get_solution_by_id(self, solution_id: str) -> Optional[MCQSolution]:
        """Retrieve a solution by solution ID"""
        try:
            result = self.collection.get(
                ids=[solution_id],
                include=["documents", "metadatas"]
            )
            
            if not result['ids']:
                return None
            
            solution = MCQSolution.from_chromadb_result(
                document=result['documents'][0],
                metadata=result['metadatas'][0],
                id=result['ids'][0]
            )
            
            return solution
            
        except Exception as e:
            logging.error(f"Failed to get MCQ solution {solution_id}: {str(e)}")
            return None
    
    def search_solutions(
        self, 
        query: str = None, 
        generated_by: str = None,
        validated: bool = None,
        limit: int = 50
    ) -> List[MCQSolution]:
        """Search solutions with optional filters"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if generated_by:
                where_clause["generated_by"] = generated_by
            if validated is not None:
                where_clause["validated"] = validated
            
            if query:
                # Semantic search with query
                results = self.collection.query(
                    query_texts=[query],
                    where=where_clause if where_clause else None,
                    n_results=limit,
                    include=["documents", "metadatas"]
                )
                
                solutions = []
                for i, doc_id in enumerate(results['ids'][0]):
                    solution = MCQSolution.from_chromadb_result(
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        id=doc_id
                    )
                    solutions.append(solution)
                
                return solutions
            else:
                # Get all solutions with filters
                results = self.collection.get(
                    where=where_clause if where_clause else None,
                    limit=limit,
                    include=["documents", "metadatas"]
                )
                
                solutions = []
                for i, doc_id in enumerate(results['ids']):
                    solution = MCQSolution.from_chromadb_result(
                        document=results['documents'][i],
                        metadata=results['metadatas'][i],
                        id=doc_id
                    )
                    solutions.append(solution)
                
                return solutions
                
        except Exception as e:
            logging.error(f"Failed to search MCQ solutions: {str(e)}")
            return []
    
    def delete_solution(self, solution_id: str) -> bool:
        """Delete a solution from ChromaDB"""
        try:
            self.collection.delete(ids=[solution_id])
            logging.info(f"Deleted MCQ solution {solution_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete MCQ solution {solution_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the solutions collection"""
        try:
            # Get all solutions to calculate stats
            all_solutions = self.collection.get(include=["metadatas"])
            
            total_count = len(all_solutions['ids'])
            
            if total_count == 0:
                return {
                    "total_solutions": 0,
                    "generated_by_llm": 0,
                    "generated_by_database": 0,
                    "validated_count": 0,
                    "average_confidence": 0.0
                }
            
            generated_by_llm = 0
            generated_by_database = 0
            validated_count = 0
            confidence_scores = []
            
            for metadata in all_solutions['metadatas']:
                # Count generation source
                if metadata.get('generated_by') == 'llm':
                    generated_by_llm += 1
                elif metadata.get('generated_by') == 'database':
                    generated_by_database += 1
                
                # Count validated solutions
                if metadata.get('validated', False):
                    validated_count += 1
                
                # Collect confidence scores
                score = metadata.get('confidence_score', 0.0)
                if score > 0:
                    confidence_scores.append(score)
            
            average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                "total_solutions": total_count,
                "generated_by_llm": generated_by_llm,
                "generated_by_database": generated_by_database,
                "validated_count": validated_count,
                "average_confidence": round(average_confidence, 3)
            }
            
        except Exception as e:
            logging.error(f"Failed to get MCQ solutions collection stats: {str(e)}")
            return {}

# Global instance
mcq_solutions_client = None

def get_mcq_solutions_client() -> MCQSolutionsClient:
    """Get or create MCQ Solutions client instance"""
    global mcq_solutions_client
    if mcq_solutions_client is None:
        mcq_solutions_client = MCQSolutionsClient()
    return mcq_solutions_client
