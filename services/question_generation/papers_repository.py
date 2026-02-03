"""
Papers Repository

Repository for CRUD operations on generated papers in MongoDB.
Handles document persistence, querying, and status updates.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models.paper import GeneratedPaper, PaperStatus

logger = logging.getLogger(__name__)

# Collection name for generated papers
COLLECTION_NAME = "generated_papers"


class PapersRepository:
    """
    Repository for managing generated papers in MongoDB.
    
    Provides CRUD operations and querying capabilities for the
    generated_papers collection.
    """
    
    def __init__(self, db_manager):
        """
        Initialize repository with database manager.
        
        Args:
            db_manager: DatabaseManager instance from app.state.db
        """
        self.db_manager = db_manager
    
    async def _get_collection(self, tenant_id: str):
        """Get the MongoDB collection for a tenant."""
        tenant_db = await self.db_manager.get_tenant_db(tenant_id)
        if tenant_db is None:
            raise ValueError(f"Tenant database not found: {tenant_id}")
        return tenant_db[COLLECTION_NAME]
    
    async def create(self, paper: GeneratedPaper) -> str:
        """
        Create a new paper record.
        
        Args:
            paper: GeneratedPaper instance
            
        Returns:
            The paper ID
        """
        try:
            collection = await self._get_collection(paper.tenant_id)
            paper.created_at = datetime.utcnow()
            
            await collection.insert_one(paper.to_mongo_dict())
            logger.info(f"Created paper record: {paper.paper_id}")
            return paper.paper_id
            
        except Exception as e:
            logger.error(f"Failed to create paper record: {e}")
            raise
    
    async def get_by_id(self, tenant_id: str, paper_id: str) -> Optional[GeneratedPaper]:
        """
        Get a paper by ID.
        
        Args:
            tenant_id: Tenant identifier
            paper_id: Paper identifier
            
        Returns:
            GeneratedPaper or None if not found
        """
        try:
            collection = await self._get_collection(tenant_id)
            doc = await collection.find_one({"_id": paper_id})
            
            if doc:
                return GeneratedPaper.from_mongo_dict(doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            raise
    
    async def update(self, paper: GeneratedPaper) -> bool:
        """
        Update an existing paper record.
        
        Args:
            paper: GeneratedPaper with updated fields
            
        Returns:
            True if updated, False if not found
        """
        try:
            collection = await self._get_collection(paper.tenant_id)
            
            result = await collection.replace_one(
                {"_id": paper.paper_id},
                paper.to_mongo_dict()
            )
            
            if result.modified_count > 0:
                logger.debug(f"Updated paper record: {paper.paper_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update paper {paper.paper_id}: {e}")
            raise
    
    async def update_status(
        self,
        tenant_id: str,
        paper_id: str,
        status: PaperStatus,
        error_message: Optional[str] = None,
        extra_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the status of a paper.
        
        Args:
            tenant_id: Tenant identifier
            paper_id: Paper identifier
            status: New status
            error_message: Optional error message
            extra_updates: Optional additional fields to update
            
        Returns:
            True if updated, False if not found
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            update_doc = {
                "$set": {
                    "status": status.value,
                }
            }
            
            if error_message:
                update_doc["$set"]["error_message"] = error_message
            
            if status == PaperStatus.COMPLETED:
                update_doc["$set"]["completed_at"] = datetime.utcnow()
            
            if extra_updates:
                update_doc["$set"].update(extra_updates)
            
            result = await collection.update_one(
                {"_id": paper_id},
                update_doc
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update status for {paper_id}: {e}")
            raise
    
    async def update_urls(
        self,
        tenant_id: str,
        paper_id: str,
        question_paper_url: Optional[str] = None,
        answer_key_url: Optional[str] = None,
        marking_scheme_url: Optional[str] = None,
    ) -> bool:
        """
        Update the PDF URLs for a paper.
        
        Args:
            tenant_id: Tenant identifier
            paper_id: Paper identifier
            question_paper_url: URL to question paper PDF
            answer_key_url: URL to answer key PDF
            marking_scheme_url: URL to marking scheme PDF
            
        Returns:
            True if updated
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            update_doc = {"$set": {}}
            
            if question_paper_url:
                update_doc["$set"]["question_paper_url"] = question_paper_url
            if answer_key_url:
                update_doc["$set"]["answer_key_url"] = answer_key_url
            if marking_scheme_url:
                update_doc["$set"]["marking_scheme_url"] = marking_scheme_url
            
            if not update_doc["$set"]:
                return False
            
            result = await collection.update_one(
                {"_id": paper_id},
                update_doc
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update URLs for {paper_id}: {e}")
            raise
    
    async def delete(self, tenant_id: str, paper_id: str) -> bool:
        """
        Delete a paper record.
        
        Args:
            tenant_id: Tenant identifier
            paper_id: Paper identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            collection = await self._get_collection(tenant_id)
            result = await collection.delete_one({"_id": paper_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted paper record: {paper_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            raise
    
    async def list_papers(
        self,
        tenant_id: str,
        teacher_id: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[PaperStatus] = None,
        skip: int = 0,
        limit: int = 50,
        sort_by: str = "created_at",
        sort_order: int = -1,  # -1 for descending
    ) -> List[GeneratedPaper]:
        """
        List papers with optional filters.
        
        Args:
            tenant_id: Tenant identifier
            teacher_id: Optional filter by teacher
            subject: Optional filter by subject
            status: Optional filter by status
            skip: Number of records to skip
            limit: Maximum records to return
            sort_by: Field to sort by
            sort_order: 1 for ascending, -1 for descending
            
        Returns:
            List of GeneratedPaper instances
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            # Build query
            query = {}
            if teacher_id:
                query["teacher_id"] = teacher_id
            if subject:
                query["paper_config.subject"] = {"$regex": subject, "$options": "i"}
            if status:
                query["status"] = status.value
            
            cursor = collection.find(query).sort(sort_by, sort_order).skip(skip).limit(limit)
            
            papers = []
            async for doc in cursor:
                papers.append(GeneratedPaper.from_mongo_dict(doc))
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            raise
    
    async def count_papers(
        self,
        tenant_id: str,
        teacher_id: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[PaperStatus] = None,
    ) -> int:
        """
        Count papers with optional filters.
        
        Returns:
            Count of matching papers
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            query = {}
            if teacher_id:
                query["teacher_id"] = teacher_id
            if subject:
                query["paper_config.subject"] = {"$regex": subject, "$options": "i"}
            if status:
                query["status"] = status.value
            
            return await collection.count_documents(query)
            
        except Exception as e:
            logger.error(f"Failed to count papers: {e}")
            raise
    
    async def get_papers_by_teacher(
        self,
        tenant_id: str,
        teacher_id: str,
        limit: int = 50,
    ) -> List[GeneratedPaper]:
        """
        Get papers by teacher ID.
        
        Returns:
            List of papers for the teacher
        """
        return await self.list_papers(
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            limit=limit,
        )
    
    async def get_recent_papers(
        self,
        tenant_id: str,
        limit: int = 10,
    ) -> List[GeneratedPaper]:
        """
        Get most recent papers.
        
        Returns:
            List of recent papers
        """
        return await self.list_papers(
            tenant_id=tenant_id,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )
    
    async def get_published_papers(
        self,
        tenant_id: str,
        subject: Optional[str] = None,
        limit: int = 50,
    ) -> List[GeneratedPaper]:
        """
        Get published papers.
        
        Returns:
            List of published papers
        """
        return await self.list_papers(
            tenant_id=tenant_id,
            subject=subject,
            status=PaperStatus.PUBLISHED,
            limit=limit,
        )
    
    async def get_stats(self, tenant_id: str, teacher_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for papers.
        
        Returns:
            Dictionary with aggregated stats
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            match_stage = {}
            if teacher_id:
                match_stage["teacher_id"] = teacher_id
            
            pipeline = [
                {"$match": match_stage} if match_stage else {"$match": {}},
                {
                    "$group": {
                        "_id": None,
                        "total_papers": {"$sum": 1},
                        "completed_papers": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "published_papers": {
                            "$sum": {"$cond": [{"$eq": ["$status", "published"]}, 1, 0]}
                        },
                        "draft_papers": {
                            "$sum": {"$cond": [{"$eq": ["$status", "draft"]}, 1, 0]}
                        },
                        "failed_papers": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                        "last_paper_at": {"$max": "$created_at"},
                    }
                }
            ]
            
            result = await collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                data = result[0]
                return {
                    "total_papers": data.get("total_papers", 0),
                    "completed_papers": data.get("completed_papers", 0),
                    "published_papers": data.get("published_papers", 0),
                    "draft_papers": data.get("draft_papers", 0),
                    "failed_papers": data.get("failed_papers", 0),
                    "last_paper_at": data.get("last_paper_at"),
                }
            
            return {
                "total_papers": 0,
                "completed_papers": 0,
                "published_papers": 0,
                "draft_papers": 0,
                "failed_papers": 0,
                "last_paper_at": None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    async def ensure_indexes(self, tenant_id: str) -> None:
        """
        Ensure required indexes exist on the collection.
        
        Should be called during tenant initialization.
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            # Create indexes
            await collection.create_index("teacher_id")
            await collection.create_index("status")
            await collection.create_index("created_at")
            await collection.create_index("paper_config.subject")
            await collection.create_index([("status", 1), ("created_at", -1)])
            await collection.create_index([("teacher_id", 1), ("status", 1)])
            await collection.create_index([("teacher_id", 1), ("created_at", -1)])
            
            logger.info(f"Ensured indexes for generated_papers in tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to ensure indexes: {e}")
            raise


# Singleton instance holder
_papers_repository: Optional[PapersRepository] = None


def get_papers_repository(db_manager) -> PapersRepository:
    """
    Get or create the papers repository singleton.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        PapersRepository instance
    """
    global _papers_repository
    if _papers_repository is None:
        _papers_repository = PapersRepository(db_manager)
    return _papers_repository
