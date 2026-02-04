"""
Papers API Endpoints

Provides endpoints for managing generated question papers:
- List papers
- Get paper details
- Download PDFs (question paper, answer key, marking scheme)
- Regenerate paper
- Delete paper
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import io

from api.v1.auth_async import get_current_user
from core.database import DatabaseManager
from services.question_generation import (
    PaperAssemblyService,
    get_paper_assembly_service,
    PaperAssemblyError,
    QuestionGeneratorService,
    get_question_generator,
    GeneratedPaper,
    PaperStatus,
    PaperConfig,
    QuestionGenerationConfig,
    PapersRepository,
    get_papers_repository_sync,
)


logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/papers", tags=["Papers"])


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_database(request: Request) -> DatabaseManager:
    """Get database manager from app state."""
    return request.app.state.db


# ============================================================================
# Request/Response Models
# ============================================================================

class PaperSummary(BaseModel):
    """Summary of a generated paper."""
    paper_id: str
    title: str
    subject: str
    grade: str
    total_questions: int
    total_marks: int
    duration_minutes: int
    status: str
    source_type: str
    created_at: str
    question_paper_url: Optional[str] = None
    answer_key_url: Optional[str] = None


class PaperDetails(BaseModel):
    """Full details of a generated paper."""
    paper_id: str
    tenant_id: str
    teacher_id: str
    title: str
    subject: str
    grade: str
    total_questions: int
    total_marks: int
    duration_minutes: int
    status: str
    source_type: str
    source_topic: Optional[str] = None
    source_upload_ids: Optional[List[str]] = None
    sections: List[Dict[str, Any]]
    question_paper_url: Optional[str] = None
    answer_key_url: Optional[str] = None
    marking_scheme_url: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    generation_stats: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class PaperListResponse(BaseModel):
    """Response for list papers endpoint."""
    papers: List[PaperSummary]
    total: int
    page: int
    page_size: int


class RegeneratePaperRequest(BaseModel):
    """Request to regenerate a paper with new configuration."""
    # Optional new configuration - if not provided, uses original
    include_diagrams: Optional[bool] = None
    include_solutions: Optional[bool] = None
    include_marking_scheme: Optional[bool] = None


class CreatePaperFromQuestionsRequest(BaseModel):
    """Request to create a paper from manually selected questions."""
    title: str = Field(..., description="Paper title")
    subject: str = Field(..., description="Subject name")
    grade: str = Field(..., description="Class/Grade")
    duration_minutes: int = Field(default=90, ge=15, le=300)
    tenant_id: str = Field(..., description="Tenant identifier")
    teacher_id: str = Field(..., description="Teacher identifier")
    question_ids: List[str] = Field(..., description="List of question IDs to include")
    school_name: Optional[str] = None
    exam_name: Optional[str] = None


# ============================================================================
# Repository Access
# ============================================================================

async def get_repository(
    db: DatabaseManager = Depends(get_database),
) -> PapersRepository:
    """Get papers repository instance."""
    return get_papers_repository_sync(db)


async def _store_paper_async(
    paper: GeneratedPaper,
    repository: PapersRepository,
) -> None:
    """Store paper in MongoDB."""
    existing = await repository.get_by_id(paper.tenant_id, paper.paper_id)
    if existing:
        await repository.update(paper)
    else:
        await repository.create(paper)


async def _get_paper_async(
    tenant_id: str,
    paper_id: str,
    repository: PapersRepository,
) -> Optional[GeneratedPaper]:
    """Get paper from MongoDB."""
    return await repository.get_by_id(tenant_id, paper_id)


async def _delete_paper_async(
    tenant_id: str,
    paper_id: str,
    repository: PapersRepository,
) -> bool:
    """Delete paper from MongoDB."""
    return await repository.delete(tenant_id, paper_id)


async def _list_papers_async(
    tenant_id: str,
    repository: PapersRepository,
    teacher_id: Optional[str] = None,
    subject: Optional[str] = None,
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
) -> List[GeneratedPaper]:
    """List papers from MongoDB with filters."""
    status_enum = PaperStatus(status_filter) if status_filter else None
    return await repository.list_papers(
        tenant_id=tenant_id,
        teacher_id=teacher_id,
        subject=subject,
        status=status_enum,
        skip=skip,
        limit=limit,
    )


async def _count_papers_async(
    tenant_id: str,
    repository: PapersRepository,
    teacher_id: Optional[str] = None,
    subject: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> int:
    """Count papers in MongoDB with filters."""
    status_enum = PaperStatus(status_filter) if status_filter else None
    return await repository.count_papers(
        tenant_id=tenant_id,
        teacher_id=teacher_id,
        subject=subject,
        status=status_enum,
    )


# ============================================================================
# Role-based Access Control
# ============================================================================

def require_teacher_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require teacher or admin role."""
    # Check multiple possible fields for user role
    user_role = (
        current_user.get("role") or
        current_user.get("userType") or
        current_user.get("user_type") or
        ""
    ).lower()

    # Also check boolean flags
    is_admin = current_user.get("isAdmin", False) or current_user.get("is_admin", False)
    is_teacher = current_user.get("isTeacher", False) or current_user.get("is_teacher", False)
    is_tutor = current_user.get("isTutor", False) or current_user.get("is_tutor", False)

    if user_role not in ["admin", "tutor", "teacher", "b2c_admin"] and not (is_admin or is_teacher or is_tutor):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers and admins can access papers"
        )
    return current_user


async def get_current_user_with_query_token(
    request: Request,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get current user supporting both Bearer token and query parameter.
    
    This allows downloading PDFs via direct URL with token as query param.
    """
    from api.v1.auth_async import get_auth_manager
    from core.token_blacklist import token_blacklist
    
    # Try to get token from query parameter first
    auth_token = token
    
    # If not in query, try Authorization header
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token is revoked
    if token_blacklist.is_revoked(auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    auth_manager = await get_auth_manager()
    user_data = await auth_manager.verify_token_and_get_user(auth_token)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_data


def require_teacher_or_admin_with_query_token(
    current_user: Dict[str, Any] = Depends(get_current_user_with_query_token)
) -> Dict[str, Any]:
    """Require teacher or admin role, supporting query token."""
    # Check multiple possible fields for user role
    user_role = (
        current_user.get("role") or
        current_user.get("userType") or
        current_user.get("user_type") or
        ""
    ).lower()

    # Also check boolean flags
    is_admin = current_user.get("isAdmin", False) or current_user.get("is_admin", False)
    is_teacher = current_user.get("isTeacher", False) or current_user.get("is_teacher", False)
    is_tutor = current_user.get("isTutor", False) or current_user.get("is_tutor", False)

    if user_role not in ["admin", "tutor", "teacher", "b2c_admin"] and not (is_admin or is_teacher or is_tutor):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers and admins can access papers"
        )
    return current_user


def _verify_paper_access(paper: GeneratedPaper, current_user: Dict[str, Any]) -> None:
    """Verify user has access to the paper."""
    user_tenant = current_user.get("tenant_id")
    user_id = current_user.get("user_id")
    user_role = current_user.get("role", "").lower()
    
    if user_tenant and paper.tenant_id != user_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this paper"
        )
    
    if user_role != "admin" and paper.teacher_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this paper"
        )


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_assembly_service() -> PaperAssemblyService:
    """Get the paper assembly service."""
    try:
        return await get_paper_assembly_service()
    except Exception as e:
        logger.error(f"Failed to get paper assembly service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper assembly service unavailable"
        )


async def get_generator_service() -> QuestionGeneratorService:
    """Get the question generator service."""
    try:
        return await get_question_generator()
    except Exception as e:
        logger.error(f"Failed to get question generator service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Question generator service unavailable"
        )


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "",
    response_model=PaperListResponse,
    summary="List generated papers",
    description="Get a paginated list of generated papers with optional filters."
)
@limiter.limit("60/minute")
async def list_papers(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
    tenant_id: Optional[str] = None,
    teacher_id: Optional[str] = None,
    subject: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> PaperListResponse:
    """List generated papers with filters."""
    # Use tenant_id from token, filter by teacher if not admin
    user_tenant = current_user.get("tenant_id")
    user_id = current_user.get("user_id")
    user_role = current_user.get("role", "").lower()
    
    # Teachers can only see their own papers, admins can see all
    effective_tenant_id = user_tenant or tenant_id
    if not effective_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    effective_teacher_id = teacher_id if user_role == "admin" else user_id
    
    skip = (page - 1) * page_size
    papers = await _list_papers_async(
        tenant_id=effective_tenant_id,
        repository=repository,
        teacher_id=effective_teacher_id,
        subject=subject,
        status_filter=status,
        skip=skip,
        limit=page_size,
    )
    
    # Get total count
    total_count = await _count_papers_async(
        tenant_id=effective_tenant_id,
        repository=repository,
        teacher_id=effective_teacher_id,
        subject=subject,
        status_filter=status,
    )
    
    summaries = [
        PaperSummary(
            paper_id=p.paper_id,
            title=p.title,
            subject=p.subject,
            grade=p.grade,
            total_questions=p.total_questions,
            total_marks=p.total_marks,
            duration_minutes=p.duration_minutes,
            status=p.status.value,
            source_type=p.source_type,
            created_at=p.created_at.isoformat(),
            question_paper_url=p.question_paper_url,
            answer_key_url=p.answer_key_url,
        )
        for p in papers
    ]
    
    return PaperListResponse(
        papers=summaries,
        total=total_count,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{paper_id}",
    response_model=PaperDetails,
    summary="Get paper details",
    description="Get full details of a generated paper including all questions."
)
@limiter.limit("60/minute")
async def get_paper_details(
    request: Request,
    paper_id: str,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
) -> PaperDetails:
    """Get full paper details."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    # Verify user has access to this paper
    _verify_paper_access(paper, current_user)
    
    return PaperDetails(
        paper_id=paper.paper_id,
        tenant_id=paper.tenant_id,
        teacher_id=paper.teacher_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        total_questions=paper.total_questions,
        total_marks=paper.total_marks,
        duration_minutes=paper.duration_minutes,
        status=paper.status.value,
        source_type=paper.source_type,
        source_topic=paper.source_topic,
        source_upload_ids=paper.source_upload_ids,
        sections=[s.to_dict() for s in paper.sections],
        question_paper_url=paper.question_paper_url,
        answer_key_url=paper.answer_key_url,
        marking_scheme_url=paper.marking_scheme_url,
        created_at=paper.created_at.isoformat(),
        completed_at=paper.completed_at.isoformat() if paper.completed_at else None,
        generation_stats=paper.generation_stats,
        error_message=paper.error_message,
    )


@router.get(
    "/{paper_id}/question-paper",
    summary="Download question paper PDF",
    description="Download the question paper PDF for a generated paper."
)
@limiter.limit("30/minute")
async def download_question_paper(
    request: Request,
    paper_id: str,
    token: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin_with_query_token),
    repository: PapersRepository = Depends(get_repository),
    assembly_service: PaperAssemblyService = Depends(get_assembly_service),
) -> StreamingResponse:
    """Download question paper PDF."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    # Verify user has access
    _verify_paper_access(paper, current_user)
    
    try:
        pdf_bytes = await assembly_service.generate_question_paper_pdf(paper)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{paper.title.replace(" ", "_")}_question_paper.pdf"'
            }
        )
    except PaperAssemblyError as e:
        logger.error(f"Failed to generate question paper PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}"
        )


@router.get(
    "/{paper_id}/answer-key",
    summary="Download answer key PDF",
    description="Download the answer key PDF for a generated paper."
)
@limiter.limit("30/minute")
async def download_answer_key(
    request: Request,
    paper_id: str,
    token: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin_with_query_token),
    repository: PapersRepository = Depends(get_repository),
    assembly_service: PaperAssemblyService = Depends(get_assembly_service),
) -> StreamingResponse:
    """Download answer key PDF."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    try:
        pdf_bytes = await assembly_service.generate_answer_key_pdf(paper)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{paper.title.replace(" ", "_")}_answer_key.pdf"'
            }
        )
    except PaperAssemblyError as e:
        logger.error(f"Failed to generate answer key PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}"
        )


@router.get(
    "/{paper_id}/marking-scheme",
    summary="Download marking scheme PDF",
    description="Download the detailed marking scheme PDF for a generated paper."
)
@limiter.limit("30/minute")
async def download_marking_scheme(
    request: Request,
    paper_id: str,
    token: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin_with_query_token),
    repository: PapersRepository = Depends(get_repository),
    assembly_service: PaperAssemblyService = Depends(get_assembly_service),
) -> StreamingResponse:
    """Download marking scheme PDF."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    try:
        pdf_bytes = await assembly_service.generate_marking_scheme_pdf(paper)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{paper.title.replace(" ", "_")}_marking_scheme.pdf"'
            }
        )
    except PaperAssemblyError as e:
        logger.error(f"Failed to generate marking scheme PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}"
        )


@router.post(
    "/{paper_id}/regenerate",
    response_model=PaperDetails,
    summary="Regenerate paper PDFs",
    description="Regenerate the PDFs for an existing paper with optional new settings."
)
@limiter.limit("5/minute")
async def regenerate_paper(
    request: Request,
    paper_id: str,
    body: RegeneratePaperRequest,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
    assembly_service: PaperAssemblyService = Depends(get_assembly_service),
) -> PaperDetails:
    """Regenerate paper PDFs."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    try:
        # Update generation config if new settings provided
        if paper.generation_config:
            if body.include_diagrams is not None:
                paper.generation_config.include_diagrams = body.include_diagrams
            if body.include_solutions is not None:
                paper.generation_config.include_solutions = body.include_solutions
            if body.include_marking_scheme is not None:
                paper.generation_config.include_marking_scheme = body.include_marking_scheme
        
        # Regenerate PDFs
        paper.status = PaperStatus.GENERATING
        paper = await assembly_service.assemble_paper(
            paper,
            generate_pdfs=True,
            upload_to_s3=True,
        )
        
        # Update storage
        await _store_paper_async(paper, repository)
        
        return PaperDetails(
            paper_id=paper.paper_id,
            tenant_id=paper.tenant_id,
            teacher_id=paper.teacher_id,
            title=paper.title,
            subject=paper.subject,
            grade=paper.grade,
            total_questions=paper.total_questions,
            total_marks=paper.total_marks,
            duration_minutes=paper.duration_minutes,
            status=paper.status.value,
            source_type=paper.source_type,
            source_topic=paper.source_topic,
            source_upload_ids=paper.source_upload_ids,
            sections=[s.to_dict() for s in paper.sections],
            question_paper_url=paper.question_paper_url,
            answer_key_url=paper.answer_key_url,
            marking_scheme_url=paper.marking_scheme_url,
            created_at=paper.created_at.isoformat(),
            completed_at=paper.completed_at.isoformat() if paper.completed_at else None,
            generation_stats=paper.generation_stats,
            error_message=paper.error_message,
        )
        
    except PaperAssemblyError as e:
        logger.error(f"Failed to regenerate paper: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate paper: {str(e)}"
        )


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete paper",
    description="Delete a generated paper and its associated files."
)
@limiter.limit("10/minute")
async def delete_paper(
    request: Request,
    paper_id: str,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
) -> None:
    """Delete a paper."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    # TODO: Delete S3 files as well
    
    await _delete_paper_async(user_tenant, paper_id, repository)


@router.post(
    "/{paper_id}/publish",
    response_model=PaperDetails,
    summary="Publish paper",
    description="Mark a paper as published (available to students)."
)
@limiter.limit("20/minute")
async def publish_paper(
    request: Request,
    paper_id: str,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
) -> PaperDetails:
    """Publish a paper."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    if paper.status != PaperStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed papers can be published"
        )
    
    paper.status = PaperStatus.PUBLISHED
    await _store_paper_async(paper, repository)
    
    return PaperDetails(
        paper_id=paper.paper_id,
        tenant_id=paper.tenant_id,
        teacher_id=paper.teacher_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        total_questions=paper.total_questions,
        total_marks=paper.total_marks,
        duration_minutes=paper.duration_minutes,
        status=paper.status.value,
        source_type=paper.source_type,
        source_topic=paper.source_topic,
        source_upload_ids=paper.source_upload_ids,
        sections=[s.to_dict() for s in paper.sections],
        question_paper_url=paper.question_paper_url,
        answer_key_url=paper.answer_key_url,
        marking_scheme_url=paper.marking_scheme_url,
        created_at=paper.created_at.isoformat(),
        completed_at=paper.completed_at.isoformat() if paper.completed_at else None,
        generation_stats=paper.generation_stats,
        error_message=paper.error_message,
    )


@router.post(
    "/{paper_id}/archive",
    response_model=PaperDetails,
    summary="Archive paper",
    description="Archive a paper (hide from active list)."
)
@limiter.limit("20/minute")
async def archive_paper(
    request: Request,
    paper_id: str,
    current_user: Dict[str, Any] = Depends(require_teacher_or_admin),
    repository: PapersRepository = Depends(get_repository),
) -> PaperDetails:
    """Archive a paper."""
    user_tenant = current_user.get("tenant_id")
    if not user_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    
    paper = await _get_paper_async(user_tenant, paper_id, repository)
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )
    
    _verify_paper_access(paper, current_user)
    
    paper.status = PaperStatus.ARCHIVED
    await _store_paper_async(paper, repository)
    
    return PaperDetails(
        paper_id=paper.paper_id,
        tenant_id=paper.tenant_id,
        teacher_id=paper.teacher_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        total_questions=paper.total_questions,
        total_marks=paper.total_marks,
        duration_minutes=paper.duration_minutes,
        status=paper.status.value,
        source_type=paper.source_type,
        source_topic=paper.source_topic,
        source_upload_ids=paper.source_upload_ids,
        sections=[s.to_dict() for s in paper.sections],
        question_paper_url=paper.question_paper_url,
        answer_key_url=paper.answer_key_url,
        marking_scheme_url=paper.marking_scheme_url,
        created_at=paper.created_at.isoformat(),
        completed_at=paper.completed_at.isoformat() if paper.completed_at else None,
        generation_stats=paper.generation_stats,
        error_message=paper.error_message,
    )


@router.get(
    "/health",
    summary="Health check",
    description="Check the health of the papers service."
)
@limiter.limit("100/minute")
async def health_check(
    request: Request,
    assembly_service: PaperAssemblyService = Depends(get_assembly_service),
) -> Dict[str, Any]:
    """Check service health."""
    return await assembly_service.health_check()


# ============================================================================
# Helper function to store papers from question generation
# ============================================================================

async def store_generated_paper_async(paper: GeneratedPaper, db_manager) -> None:
    """Store a generated paper (called from question_generation endpoints)."""
    repository = get_papers_repository(db_manager)
    await _store_paper_async(paper, repository)
