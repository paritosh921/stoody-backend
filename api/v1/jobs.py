"""
Jobs API Endpoints

Provides endpoints for async job status polling.
All heavy operations return job_id immediately; frontend polls for status.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user
from services.question_generation.jobs_repository import get_jobs_repository
from services.question_generation.models.job import Job, JobType, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# =============================================================================
# Response Models
# =============================================================================

class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    type: str
    status: str
    progress: int = Field(ge=0, le=100)
    progress_details: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: List[JobStatusResponse]
    total: int
    has_more: bool


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get the status of a specific job.
    
    This endpoint is designed for polling. Frontend should call this
    every 2-5 seconds while job is running.
    
    Returns:
        Job status with progress information
    """
    try:
        jobs_repo = await get_jobs_repository()
        job = await jobs_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Verify user has access to this job
        user_tenant = current_user.get("tenant_id", "")
        if job.institution_id != user_tenant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return JobStatusResponse(**job.to_api_response())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")


@router.get("", response_model=JobListResponse)
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: dict = Depends(get_current_user),
):
    """
    List jobs for the current user's institution.
    
    Args:
        job_type: Optional filter (ingest_pdf, generate_paper)
        status: Optional filter (queued, running, succeeded, failed)
        limit: Maximum results (default 20)
        offset: Pagination offset
        
    Returns:
        List of jobs with pagination info
    """
    try:
        jobs_repo = await get_jobs_repository()
        
        user_tenant = current_user.get("tenant_id", "")
        user_id = current_user.get("user_id", current_user.get("id", ""))
        
        # Parse filters
        type_filter = None
        if job_type:
            try:
                type_filter = JobType(job_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}")
        
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Get jobs
        jobs = await jobs_repo.list_jobs(
            institution_id=user_tenant,
            user_id=user_id,
            job_type=type_filter,
            status=status_filter,
            limit=limit + 1,  # Get one extra to check if there are more
            skip=offset,
        )
        
        has_more = len(jobs) > limit
        if has_more:
            jobs = jobs[:limit]
        
        job_responses = [JobStatusResponse(**job.to_api_response()) for job in jobs]
        
        return JobListResponse(
            jobs=job_responses,
            total=len(job_responses),
            has_more=has_more,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Cancel a running or queued job.
    
    Note: Cancellation is best-effort. Jobs that have already completed
    or are in a terminal state cannot be cancelled.
    
    Returns:
        Updated job status
    """
    try:
        jobs_repo = await get_jobs_repository()
        job = await jobs_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Verify user has access
        user_tenant = current_user.get("tenant_id", "")
        if job.institution_id != user_tenant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if job can be cancelled
        if job.is_terminal:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in terminal state: {job.status.value}"
            )
        
        # Cancel the job
        job.cancel()
        await jobs_repo.update_job(job)
        
        logger.info(f"Job {job_id} cancelled by user {current_user.get('user_id')}")
        
        return {"message": "Job cancelled", "job_id": job_id, "status": job.status.value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a completed job record.
    
    Only jobs in terminal states (succeeded, failed, cancelled) can be deleted.
    
    Returns:
        Confirmation message
    """
    try:
        jobs_repo = await get_jobs_repository()
        job = await jobs_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Verify user has access
        user_tenant = current_user.get("tenant_id", "")
        if job.institution_id != user_tenant:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if job can be deleted
        if not job.is_terminal:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete job in non-terminal state: {job.status.value}"
            )
        
        # Delete the job
        await jobs_repo.delete_job(job_id)
        
        logger.info(f"Job {job_id} deleted by user {current_user.get('user_id')}")
        
        return {"message": "Job deleted", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")


# =============================================================================
# Health Check Endpoint
# =============================================================================

@router.get("/health/check")
async def jobs_health_check():
    """
    Check if the jobs system is healthy.
    
    Returns:
        Health status
    """
    try:
        jobs_repo = await get_jobs_repository()
        
        # Try to count jobs (simple health check)
        await jobs_repo.collection.count_documents({}, limit=1)
        
        return {
            "status": "healthy",
            "service": "jobs",
        }
        
    except Exception as e:
        logger.error(f"Jobs health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "jobs",
            "error": str(e),
        }
