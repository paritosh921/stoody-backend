"""
Jobs Repository for async task management.

Handles CRUD operations for jobs in MongoDB.
Provides idempotent operations and status tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from .models.job import (
    Job,
    JobType,
    JobStatus,
    JobProgress,
    PaperGenerationState,
)

logger = logging.getLogger(__name__)


class JobsRepository:
    """
    Repository for managing async jobs.
    
    Provides:
    - Job CRUD operations
    - Status tracking and updates
    - Progress management
    - Job history queries
    """
    
    COLLECTION_NAME = "jobs"
    STATE_COLLECTION_NAME = "paper_generation_states"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize repository with database connection."""
        self.db = db
        self.collection = db[self.COLLECTION_NAME]
        self.state_collection = db[self.STATE_COLLECTION_NAME]
    
    # =========================================================================
    # Job CRUD Operations
    # =========================================================================
    
    async def create_job(self, job: Job) -> Job:
        """
        Create a new job.
        
        Args:
            job: Job to create
            
        Returns:
            Created job with assigned ID
        """
        try:
            doc = job.to_mongo_dict()
            await self.collection.insert_one(doc)
            logger.info(f"Created job {job.job_id} (type={job.job_type.value})")
            return job
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job if found, None otherwise
        """
        try:
            doc = await self.collection.find_one({"_id": job_id})
            if doc:
                return Job.from_mongo_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise
    
    async def update_job(self, job: Job) -> bool:
        """
        Update an existing job.
        
        Args:
            job: Job with updated fields
            
        Returns:
            True if updated successfully
        """
        try:
            doc = job.to_mongo_dict()
            result = await self.collection.replace_one(
                {"_id": job.job_id},
                doc
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update job {job.job_id}: {e}")
            raise
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update job status atomically.
        
        Args:
            job_id: Job identifier
            status: New status
            error: Error message if failed
            result: Result data if succeeded
            
        Returns:
            True if updated successfully
        """
        try:
            update_doc: Dict[str, Any] = {"status": status.value}
            
            if status == JobStatus.RUNNING:
                update_doc["started_at"] = datetime.utcnow()
            elif status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
                update_doc["completed_at"] = datetime.utcnow()
            
            if error:
                update_doc["error"] = error
            if result:
                update_doc["result"] = result
            
            updated = await self.collection.update_one(
                {"_id": job_id},
                {"$set": update_doc}
            )
            
            logger.info(f"Updated job {job_id} status to {status.value}")
            return updated.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update job status {job_id}: {e}")
            raise
    
    async def update_job_progress(
        self,
        job_id: str,
        current_step: int,
        total_steps: Optional[int] = None,
        step_name: str = "",
        details: Optional[str] = None,
    ) -> bool:
        """
        Update job progress atomically.
        
        Args:
            job_id: Job identifier
            current_step: Current step number
            total_steps: Total steps (optional update)
            step_name: Name of current step
            details: Additional details
            
        Returns:
            True if updated successfully
        """
        try:
            update_doc = {
                "progress.current_step": current_step,
                "progress.step_name": step_name,
            }
            
            if total_steps is not None:
                update_doc["progress.total_steps"] = total_steps
            if details is not None:
                update_doc["progress.details"] = details
            
            updated = await self.collection.update_one(
                {"_id": job_id},
                {"$set": update_doc}
            )
            
            return updated.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update job progress {job_id}: {e}")
            raise
    
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            result = await self.collection.delete_one({"_id": job_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise
    
    # =========================================================================
    # Job Queries
    # =========================================================================
    
    async def list_jobs(
        self,
        institution_id: str,
        user_id: Optional[str] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Job]:
        """
        List jobs with filters.
        
        Args:
            institution_id: Institution filter (required)
            user_id: Optional user filter
            job_type: Optional job type filter
            status: Optional status filter
            limit: Maximum results
            skip: Offset for pagination
            
        Returns:
            List of matching jobs
        """
        try:
            query: Dict[str, Any] = {"institution_id": institution_id}
            
            if user_id:
                query["user_id"] = user_id
            if job_type:
                query["job_type"] = job_type.value
            if status:
                query["status"] = status.value
            
            cursor = self.collection.find(query).sort(
                "created_at", -1
            ).skip(skip).limit(limit)
            
            jobs = []
            async for doc in cursor:
                jobs.append(Job.from_mongo_dict(doc))
            
            return jobs
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise
    
    async def get_pending_jobs(
        self,
        job_type: Optional[JobType] = None,
        limit: int = 100,
    ) -> List[Job]:
        """
        Get jobs that are queued or need retry.
        
        Args:
            job_type: Optional job type filter
            limit: Maximum results
            
        Returns:
            List of pending jobs
        """
        try:
            query: Dict[str, Any] = {
                "status": {"$in": [JobStatus.QUEUED.value]},
            }
            
            if job_type:
                query["job_type"] = job_type.value
            
            cursor = self.collection.find(query).sort(
                "created_at", 1  # FIFO order
            ).limit(limit)
            
            jobs = []
            async for doc in cursor:
                jobs.append(Job.from_mongo_dict(doc))
            
            return jobs
        except Exception as e:
            logger.error(f"Failed to get pending jobs: {e}")
            raise
    
    async def get_stalled_jobs(
        self,
        stall_threshold_minutes: int = 30,
    ) -> List[Job]:
        """
        Get jobs that appear to be stalled (running for too long).
        
        Args:
            stall_threshold_minutes: Minutes after which a running job is considered stalled
            
        Returns:
            List of potentially stalled jobs
        """
        try:
            threshold = datetime.utcnow() - timedelta(minutes=stall_threshold_minutes)
            
            query = {
                "status": JobStatus.RUNNING.value,
                "started_at": {"$lt": threshold},
            }
            
            cursor = self.collection.find(query)
            
            jobs = []
            async for doc in cursor:
                jobs.append(Job.from_mongo_dict(doc))
            
            return jobs
        except Exception as e:
            logger.error(f"Failed to get stalled jobs: {e}")
            raise
    
    async def cleanup_old_jobs(
        self,
        older_than_days: int = 30,
    ) -> int:
        """
        Delete old completed jobs.
        
        Args:
            older_than_days: Delete jobs older than this many days
            
        Returns:
            Number of deleted jobs
        """
        try:
            threshold = datetime.utcnow() - timedelta(days=older_than_days)
            
            result = await self.collection.delete_many({
                "status": {"$in": [
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ]},
                "completed_at": {"$lt": threshold},
            })
            
            logger.info(f"Cleaned up {result.deleted_count} old jobs")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            raise
    
    # =========================================================================
    # Paper Generation State Management
    # =========================================================================
    
    async def create_generation_state(self, state: PaperGenerationState) -> PaperGenerationState:
        """
        Create a new paper generation state.
        
        Args:
            state: State to create
            
        Returns:
            Created state
        """
        try:
            doc = state.to_mongo_dict()
            await self.state_collection.insert_one(doc)
            logger.info(f"Created generation state for paper {state.paper_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to create generation state: {e}")
            raise
    
    async def get_generation_state(self, paper_id: str) -> Optional[PaperGenerationState]:
        """
        Get paper generation state.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            State if found, None otherwise
        """
        try:
            doc = await self.state_collection.find_one({"_id": paper_id})
            if doc:
                return PaperGenerationState.from_mongo_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get generation state {paper_id}: {e}")
            raise
    
    async def update_generation_state(self, state: PaperGenerationState) -> bool:
        """
        Update paper generation state.
        
        Args:
            state: Updated state
            
        Returns:
            True if updated successfully
        """
        try:
            state.last_updated_at = datetime.utcnow()
            doc = state.to_mongo_dict()
            result = await self.state_collection.replace_one(
                {"_id": state.paper_id},
                doc
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update generation state {state.paper_id}: {e}")
            raise
    
    async def delete_generation_state(self, paper_id: str) -> bool:
        """
        Delete paper generation state.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            result = await self.state_collection.delete_one({"_id": paper_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete generation state {paper_id}: {e}")
            raise
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    async def ensure_indexes(self) -> None:
        """Create necessary indexes for performance."""
        try:
            # Jobs collection indexes
            await self.collection.create_index("institution_id")
            await self.collection.create_index("user_id")
            await self.collection.create_index("job_type")
            await self.collection.create_index("status")
            await self.collection.create_index("created_at")
            await self.collection.create_index([
                ("institution_id", 1),
                ("status", 1),
                ("created_at", -1),
            ])
            
            # State collection indexes
            await self.state_collection.create_index("last_updated_at")
            
            logger.info("Jobs indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise


# Singleton instance
_jobs_repository: Optional[JobsRepository] = None
_db_manager = None


def init_jobs_repository(db_manager) -> None:
    """Initialize the global database manager for jobs repository."""
    global _db_manager
    _db_manager = db_manager


async def get_jobs_repository() -> JobsRepository:
    """
    Get the singleton JobsRepository instance.

    Note: init_jobs_repository must be called first during app startup.
    """
    global _jobs_repository
    if _jobs_repository is None:
        if _db_manager is None:
            raise RuntimeError("Jobs repository not initialized. Call init_jobs_repository first.")
        # Get the underlying motor database from the db_manager
        # DatabaseManager uses mongo_db (the database instance directly)
        db = await _db_manager.get_mongo_db()
        if db is None:
            raise RuntimeError("MongoDB not available")
        _jobs_repository = JobsRepository(db)
        await _jobs_repository.ensure_indexes()
    return _jobs_repository


def get_jobs_repository_from_db(db: AsyncIOMotorDatabase) -> JobsRepository:
    """Get a JobsRepository instance from a database connection."""
    return JobsRepository(db)
