"""
Knowledge Base Repository

Repository for CRUD operations on knowledge base uploads in MongoDB.
Handles document persistence, querying, and status updates.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.knowledge_base import (
    KnowledgeBaseUpload,
    KnowledgeBaseStats,
    UploadStatus,
)

logger = logging.getLogger(__name__)

# Collection name for knowledge base uploads
COLLECTION_NAME = "knowledge_base_uploads"


class KnowledgeBaseRepository:
    """
    Repository for managing knowledge base uploads in MongoDB.
    
    Provides CRUD operations and querying capabilities for the
    knowledge_base_uploads collection.
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
        # Knowledge base uploads are stored in tenant databases
        tenant_db = await self.db_manager.get_tenant_db(tenant_id)
        if tenant_db is None:
            raise ValueError(f"Tenant database not found: {tenant_id}")
        return tenant_db[COLLECTION_NAME]
    
    async def create(self, upload: KnowledgeBaseUpload) -> str:
        """
        Create a new knowledge base upload record.
        
        Args:
            upload: KnowledgeBaseUpload instance
            
        Returns:
            The upload ID
        """
        try:
            collection = await self._get_collection(upload.tenant_id)
            upload.created_at = datetime.utcnow()
            upload.updated_at = datetime.utcnow()
            
            await collection.insert_one(upload.to_dict())
            logger.info(f"Created upload record: {upload.id}")
            return upload.id
            
        except Exception as e:
            logger.error(f"Failed to create upload record: {e}")
            raise
    
    async def get_by_id(self, tenant_id: str, upload_id: str) -> Optional[KnowledgeBaseUpload]:
        """
        Get an upload by ID.
        
        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
            
        Returns:
            KnowledgeBaseUpload or None if not found
        """
        try:
            collection = await self._get_collection(tenant_id)
            doc = await collection.find_one({"_id": upload_id})
            
            if doc:
                return KnowledgeBaseUpload.from_dict(doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get upload {upload_id}: {e}")
            raise
    
    async def update(self, upload: KnowledgeBaseUpload) -> bool:
        """
        Update an existing upload record.
        
        Args:
            upload: KnowledgeBaseUpload with updated fields
            
        Returns:
            True if updated, False if not found
        """
        try:
            collection = await self._get_collection(upload.tenant_id)
            upload.updated_at = datetime.utcnow()
            
            result = await collection.replace_one(
                {"_id": upload.id},
                upload.to_dict()
            )
            
            if result.modified_count > 0:
                logger.debug(f"Updated upload record: {upload.id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update upload {upload.id}: {e}")
            raise
    
    async def update_status(
        self,
        tenant_id: str,
        upload_id: str,
        status: UploadStatus,
        error_message: Optional[str] = None,
        extra_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the status of an upload.
        
        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
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
                    "updated_at": datetime.utcnow(),
                }
            }
            
            if error_message:
                update_doc["$set"]["error_message"] = error_message
            
            if status == UploadStatus.PROCESSING:
                update_doc["$set"]["processing_started_at"] = datetime.utcnow()
            elif status in (UploadStatus.COMPLETED, UploadStatus.FAILED, UploadStatus.PARTIALLY_COMPLETED):
                update_doc["$set"]["processing_completed_at"] = datetime.utcnow()
            
            if extra_updates:
                update_doc["$set"].update(extra_updates)
            
            result = await collection.update_one(
                {"_id": upload_id},
                update_doc
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update status for {upload_id}: {e}")
            raise
    
    async def update_processing_result(
        self,
        tenant_id: str,
        upload_id: str,
        extracted_text: str,
        total_chunks: int,
        chunks_embedded: int,
        chunks_stored: int,
        qdrant_point_ids: List[str],
        status: UploadStatus,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update upload with processing results.
        
        Args:
            Various processing result fields
            
        Returns:
            True if updated
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            update_doc = {
                "$set": {
                    "extracted_text": extracted_text,
                    "text_length": len(extracted_text) if extracted_text else 0,
                    "total_chunks": total_chunks,
                    "chunks_embedded": chunks_embedded,
                    "chunks_stored": chunks_stored,
                    "qdrant_point_ids": qdrant_point_ids,
                    "status": status.value,
                    "updated_at": datetime.utcnow(),
                    "processing_completed_at": datetime.utcnow(),
                }
            }
            
            if metrics:
                update_doc["$set"]["metrics"] = metrics
            
            if error_message:
                update_doc["$set"]["error_message"] = error_message
            
            result = await collection.update_one(
                {"_id": upload_id},
                update_doc
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update processing result for {upload_id}: {e}")
            raise
    
    async def delete(self, tenant_id: str, upload_id: str) -> bool:
        """
        Delete an upload record.
        
        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            collection = await self._get_collection(tenant_id)
            result = await collection.delete_one({"_id": upload_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted upload record: {upload_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete upload {upload_id}: {e}")
            raise
    
    async def list_uploads(
        self,
        tenant_id: str,
        teacher_id: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[UploadStatus] = None,
        skip: int = 0,
        limit: int = 50,
        sort_by: str = "created_at",
        sort_order: int = -1,  # -1 for descending
    ) -> List[KnowledgeBaseUpload]:
        """
        List uploads with optional filters.
        
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
            List of KnowledgeBaseUpload instances
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            # Build query
            query = {}
            if teacher_id:
                query["teacher_id"] = teacher_id
            if subject:
                query["subject"] = subject
            if status:
                query["status"] = status.value
            
            cursor = collection.find(query).sort(sort_by, sort_order).skip(skip).limit(limit)
            
            uploads = []
            async for doc in cursor:
                uploads.append(KnowledgeBaseUpload.from_dict(doc))
            
            return uploads
            
        except Exception as e:
            logger.error(f"Failed to list uploads: {e}")
            raise
    
    async def count_uploads(
        self,
        tenant_id: str,
        teacher_id: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[UploadStatus] = None,
    ) -> int:
        """
        Count uploads with optional filters.
        
        Returns:
            Count of matching uploads
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            query = {}
            if teacher_id:
                query["teacher_id"] = teacher_id
            if subject:
                query["subject"] = subject
            if status:
                query["status"] = status.value
            
            return await collection.count_documents(query)
            
        except Exception as e:
            logger.error(f"Failed to count uploads: {e}")
            raise
    
    async def get_pending_uploads(
        self,
        tenant_id: str,
        limit: int = 10,
    ) -> List[KnowledgeBaseUpload]:
        """
        Get pending uploads that need processing.
        
        Returns:
            List of pending uploads ordered by creation time
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            cursor = collection.find({
                "status": UploadStatus.PENDING.value
            }).sort("created_at", 1).limit(limit)
            
            uploads = []
            async for doc in cursor:
                uploads.append(KnowledgeBaseUpload.from_dict(doc))
            
            return uploads
            
        except Exception as e:
            logger.error(f"Failed to get pending uploads: {e}")
            raise
    
    async def get_failed_uploads_for_retry(
        self,
        tenant_id: str,
        limit: int = 10,
    ) -> List[KnowledgeBaseUpload]:
        """
        Get failed uploads that can be retried.
        
        Returns:
            List of failed uploads that haven't exceeded retry limit
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            cursor = collection.find({
                "status": UploadStatus.FAILED.value,
                "$expr": {"$lt": ["$retry_count", "$max_retries"]}
            }).sort("updated_at", 1).limit(limit)
            
            uploads = []
            async for doc in cursor:
                uploads.append(KnowledgeBaseUpload.from_dict(doc))
            
            return uploads
            
        except Exception as e:
            logger.error(f"Failed to get failed uploads for retry: {e}")
            raise
    
    async def get_stats(self, tenant_id: str) -> KnowledgeBaseStats:
        """
        Get statistics for a tenant's knowledge base.
        
        Returns:
            KnowledgeBaseStats with aggregated metrics
        """
        try:
            collection = await self._get_collection(tenant_id)
            
            # Aggregation pipeline for stats
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_uploads": {"$sum": 1},
                        "completed_uploads": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "failed_uploads": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                        "pending_uploads": {
                            "$sum": {"$cond": [{"$eq": ["$status", "pending"]}, 1, 0]}
                        },
                        "total_chunks": {"$sum": "$chunks_stored"},
                        "total_text_chars": {"$sum": "$text_length"},
                        "total_tokens_used": {"$sum": "$metrics.tokens_used"},
                        "last_upload_at": {"$max": "$created_at"},
                    }
                }
            ]
            
            result = await collection.aggregate(pipeline).to_list(length=1)
            
            stats = KnowledgeBaseStats(tenant_id=tenant_id)
            
            if result:
                data = result[0]
                stats.total_uploads = data.get("total_uploads", 0)
                stats.completed_uploads = data.get("completed_uploads", 0)
                stats.failed_uploads = data.get("failed_uploads", 0)
                stats.pending_uploads = data.get("pending_uploads", 0)
                stats.total_chunks = data.get("total_chunks", 0)
                stats.total_text_chars = data.get("total_text_chars", 0)
                stats.total_tokens_used = data.get("total_tokens_used", 0)
                stats.last_upload_at = data.get("last_upload_at")
            
            # Get uploads by subject
            subject_pipeline = [
                {"$group": {"_id": "$subject", "count": {"$sum": 1}}}
            ]
            subject_result = await collection.aggregate(subject_pipeline).to_list(length=100)
            stats.uploads_by_subject = {
                item["_id"]: item["count"] 
                for item in subject_result 
                if item["_id"]
            }
            
            # Get uploads by status
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_result = await collection.aggregate(status_pipeline).to_list(length=10)
            stats.uploads_by_status = {
                item["_id"]: item["count"] 
                for item in status_result 
                if item["_id"]
            }
            
            return stats
            
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
            await collection.create_index("subject")
            await collection.create_index("status")
            await collection.create_index("created_at")
            await collection.create_index([("status", 1), ("created_at", 1)])
            await collection.create_index([("teacher_id", 1), ("subject", 1)])
            
            logger.info(f"Ensured indexes for knowledge_base_uploads in tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to ensure indexes: {e}")
            raise
