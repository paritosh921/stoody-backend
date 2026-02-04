"""
Question Generation API Endpoints

Provides endpoints for generating question papers from:
1. Uploaded notes (Mode 1)
2. Topic-based search (Mode 2)
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user
from core.database import DatabaseManager
from services.question_generation import (
    QuestionGeneratorService,
    get_question_generator,
    QuestionGeneratorError,
    PaperConfig,
    GeneratedPaper,
    get_papers_repository,
)
from services.question_generation.paper_generation_worker import (
    create_paper_generation_job,
    run_paper_generation_worker,
)

# Import schemas from new modular location
from api.v1.schemas.question_generation import (
    GenerateFromNotesRequest,
    GenerateFromTopicRequest,
    QuestionPreviewRequest,
    GeneratedPaperResponse,
    PaperSectionResponse,
    GeneratedQuestionResponse,
    QuestionPreviewResponse,
    HealthCheckResponse,
)

# Import templates from new modular location
from api.v1.templates.paper_templates import (
    get_all_templates,
    get_question_types_info,
)


logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/question-generation", tags=["Question Generation"])


# ============================================================================
# Constants
# ============================================================================

ALLOWED_ROLES = ["admin", "tutor", "teacher", "master_admin", "operator", "superadmin"]
ALLOWED_TYPES = ["admin", "tutor"]


# ============================================================================
# Dependencies
# ============================================================================

async def get_database(request: Request) -> DatabaseManager:
    """Get database manager from app state."""
    return request.app.state.db


async def get_generator() -> QuestionGeneratorService:
    """Get the question generator service."""
    try:
        return await get_question_generator()
    except Exception as e:
        logger.error(f"Failed to get question generator: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Question generation service unavailable"
        )


def verify_user_role(current_user: Dict[str, Any]) -> None:
    """Verify user has teacher/admin role. Raises HTTPException if unauthorized."""
    user_role = (current_user.get("role") or current_user.get("admin_role") or "").lower()
    user_type = current_user.get("user_type", "").lower()
    
    if user_role not in ALLOWED_ROLES and user_type not in ALLOWED_TYPES:
        logger.warning(
            f"User role '{user_role}' / type '{user_type}' not authorized. "
            f"Allowed roles: {ALLOWED_ROLES}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Only teachers and admins can access this resource. "
                   f"Your role: {user_role}, type: {user_type}"
        )


def get_user_ids(current_user: Dict[str, Any], body_tenant_id: str, body_teacher_id: str) -> tuple:
    """Extract tenant_id and teacher_id from token or request body."""
    tenant_id = current_user.get("tenant_id") or body_tenant_id
    teacher_id = current_user.get("user_id") or body_teacher_id
    return tenant_id, teacher_id


# ============================================================================
# Endpoints - Generation
# ============================================================================

@router.post(
    "/from-notes",
    response_model=GeneratedPaperResponse,
    summary="Generate questions from uploaded notes",
    description="Mode 1: Generate questions from OCR-extracted text content from uploaded notes."
)
@limiter.limit("5/minute")
async def generate_from_notes(
    request: Request,
    body: GenerateFromNotesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    generator: QuestionGeneratorService = Depends(get_generator),
    db: DatabaseManager = Depends(get_database),
) -> GeneratedPaperResponse:
    """
    Generate question paper from uploaded notes content.
    
    This endpoint takes the extracted text from notes (after OCR processing)
    and generates questions based on the provided configuration.
    """
    try:
        verify_user_role(current_user)
        tenant_id, teacher_id = get_user_ids(current_user, body.tenant_id, body.teacher_id)
        
        paper = await generator.generate_from_notes(
            content=body.content,
            generation_config=body.generation_config.to_internal_config(),
            paper_config=body.paper_config.to_internal_config(),
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            store_embeddings=body.store_embeddings,
            source_file_id=body.source_file_id,
        )
        
        # Store the paper in database for download endpoints
        await _store_paper_in_db(db, paper)
        
        return _paper_to_response(paper)
        
    except QuestionGeneratorError as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_from_notes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during question generation"
        )


@router.post(
    "/from-topic",
    response_model=GeneratedPaperResponse,
    summary="Generate questions from topic",
    description="Mode 2: Generate questions by searching the knowledge base for relevant content."
)
@limiter.limit("5/minute")
async def generate_from_topic(
    request: Request,
    body: GenerateFromTopicRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    generator: QuestionGeneratorService = Depends(get_generator),
) -> GeneratedPaperResponse:
    """
    Generate question paper from topic using knowledge base search.
    
    This endpoint searches the knowledge base for content related to the topic
    and uses RAG to generate relevant questions.
    """
    try:
        verify_user_role(current_user)
        tenant_id, teacher_id = get_user_ids(current_user, body.tenant_id, body.teacher_id)
        
        paper = await generator.generate_from_topic(
            topic=body.topic,
            subject=body.subject,
            grade=body.grade,
            generation_config=body.generation_config.to_internal_config(),
            paper_config=body.paper_config.to_internal_config(),
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            chapter=body.chapter,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
        
        return _paper_to_response(paper)
        
    except QuestionGeneratorError as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_from_topic: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during question generation"
        )


@router.post(
    "/preview",
    response_model=QuestionPreviewResponse,
    summary="Preview generated questions",
    description="Generate and preview questions without creating PDF files."
)
@limiter.limit("10/minute")
async def preview_questions(
    request: Request,
    body: QuestionPreviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    generator: QuestionGeneratorService = Depends(get_generator),
) -> QuestionPreviewResponse:
    """
    Preview questions without generating the full paper.
    
    Useful for reviewing questions before finalizing the paper.
    """
    try:
        verify_user_role(current_user)
        
        tenant_id = current_user.get("tenant_id") or body.tenant_id
        teacher_id = current_user.get("user_id") or "preview"
        
        generation_config = body.generation_config.to_internal_config()
        
        # Create a minimal paper config for preview
        paper_config = PaperConfig(
            title="Preview",
            subject=body.subject,
            grade=body.grade,
            chapter=body.chapter,
        )
        
        if body.content:
            # Mode 1: From notes
            paper = await generator.generate_from_notes(
                content=body.content,
                generation_config=generation_config,
                paper_config=paper_config,
                tenant_id=tenant_id,
                teacher_id=teacher_id,
                store_embeddings=False,
            )
        elif body.topic:
            # Mode 2: From topic
            paper = await generator.generate_from_topic(
                topic=body.topic,
                subject=body.subject,
                grade=body.grade,
                generation_config=generation_config,
                paper_config=paper_config,
                tenant_id=tenant_id,
                teacher_id=teacher_id,
                chapter=body.chapter,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'content' or 'topic' must be provided"
            )
        
        questions = paper.get_all_questions()
        preview_data = await generator.preview_questions(
            questions=questions,
            include_answers=body.include_answers,
        )
        
        return QuestionPreviewResponse(
            questions=preview_data,
            total_count=len(questions),
            total_marks=paper.total_marks,
        )
        
    except QuestionGeneratorError as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during preview generation"
        )


# ============================================================================
# Endpoints - Async Generation (Job-based)
# ============================================================================

class AsyncGenerationRequest(BaseModel):
    """Request model for async paper generation."""
    pdf_ids: List[str] = Field(..., description="Source PDF IDs for RAG context")
    subject: str = Field(..., description="Subject name")
    class_grade: str = Field(..., description="Class/grade level")
    blueprint: Dict[str, Any] = Field(..., description="Paper blueprint configuration")
    include_diagrams: bool = Field(default=True, description="Include diagrams in questions")
    exam_style: Optional[str] = Field(default=None, description="Exam style (JEE, NEET, CBSE)")
    tenant_id: Optional[str] = Field(default=None, description="Optional tenant ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pdf_ids": ["pdf_123", "pdf_456"],
                "subject": "Physics",
                "class_grade": "10",
                "blueprint": {
                    "total_questions": 20,
                    "sections": [
                        {"type": "mcq", "count": 10, "marks_each": 1, "difficulty": {"easy": 0.5, "med": 0.4, "hard": 0.1}},
                        {"type": "short", "count": 5, "marks_each": 2, "difficulty": {"easy": 0.3, "med": 0.5, "hard": 0.2}},
                        {"type": "long", "count": 3, "marks_each": 5, "difficulty": {"easy": 0.2, "med": 0.5, "hard": 0.3}}
                    ],
                    "include_diagrams": True,
                    "exam_style": "CBSE"
                },
                "include_diagrams": True,
                "exam_style": "CBSE"
            }
        }


class AsyncGenerationResponse(BaseModel):
    """Response model for async paper generation."""
    job_id: str
    paper_id: str
    status: str = "queued"
    message: str = "Paper generation job created. Poll /api/v1/jobs/{job_id} for status."


@router.post(
    "/generate-async",
    response_model=AsyncGenerationResponse,
    summary="Generate paper asynchronously (job-based)",
    description="""
    Start async paper generation job. Returns immediately with job_id.
    
    This endpoint avoids CloudFront's 30-second timeout by returning immediately
    and processing the heavy work in a background job.
    
    **Workflow:**
    1. POST to this endpoint → receive job_id
    2. Poll GET /api/v1/jobs/{job_id} every 2-5 seconds
    3. When status="succeeded", get paper from GET /api/v1/papers/{paper_id}
    
    **Blueprint format:**
    ```json
    {
        "sections": [
            {"type": "mcq", "count": 10, "marks_each": 1, "difficulty": {"easy": 0.5, "med": 0.4, "hard": 0.1}},
            {"type": "short", "count": 5, "marks_each": 2},
            {"type": "long", "count": 3, "marks_each": 5}
        ]
    }
    ```
    """
)
@limiter.limit("3/minute")
async def generate_paper_async(
    request: Request,
    body: AsyncGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> AsyncGenerationResponse:
    """
    Start async paper generation job.
    
    Returns job_id immediately. Frontend should poll /api/v1/jobs/{job_id}
    for status updates.
    """
    try:
        verify_user_role(current_user)
        
        tenant_id = current_user.get("tenant_id") or body.tenant_id
        user_id = current_user.get("user_id", current_user.get("id", "unknown"))
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tenant_id is required"
            )
        
        if not body.pdf_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one pdf_id is required"
            )
        
        # Create job and paper records
        job, paper_id = await create_paper_generation_job(
            institution_id=tenant_id,
            user_id=user_id,
            pdf_ids=body.pdf_ids,
            subject=body.subject,
            class_grade=body.class_grade,
            blueprint=body.blueprint,
            include_diagrams=body.include_diagrams,
            exam_style=body.exam_style,
        )
        
        # Add background task to process the job
        background_tasks.add_task(run_paper_generation_worker, job.job_id)
        
        logger.info(f"Created paper generation job {job.job_id} for paper {paper_id}")
        
        return AsyncGenerationResponse(
            job_id=job.job_id,
            paper_id=paper_id,
            status="queued",
            message=f"Paper generation job created. Poll /api/v1/jobs/{job.job_id} for status.",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create paper generation job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create paper generation job"
        )


# ============================================================================
# Endpoints - Reference Data
# ============================================================================

@router.get(
    "/templates",
    summary="Get pre-defined paper templates",
    description="Get a list of pre-defined paper templates for common exam patterns."
)
@limiter.limit("60/minute")
async def get_templates(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get pre-defined paper templates."""
    return get_all_templates()


@router.get(
    "/question-types",
    summary="Get supported question types",
    description="Get a list of all supported question types with descriptions."
)
@limiter.limit("60/minute")
async def get_question_types(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get supported question types."""
    return get_question_types_info()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health of the question generation service and its dependencies."
)
@limiter.limit("100/minute")
async def health_check(
    request: Request,
    generator: QuestionGeneratorService = Depends(get_generator),
) -> HealthCheckResponse:
    """Check service health."""
    health = await generator.health_check()
    return HealthCheckResponse(**health)


# ============================================================================
# Helper Functions
# ============================================================================

async def _store_paper_in_db(db: DatabaseManager, paper: GeneratedPaper) -> None:
    """Store generated paper in database for download endpoints."""
    try:
        repository = get_papers_repository(db)
        existing = await repository.get_by_id(paper.tenant_id, paper.paper_id)
        if existing:
            await repository.update(paper)
        else:
            await repository.create(paper)
        logger.info(f"Paper {paper.paper_id} stored in database for download")
    except Exception as store_error:
        logger.error(f"Failed to store paper in database: {store_error}")
        # Don't fail the request, paper was still generated


def _paper_to_response(paper: GeneratedPaper) -> GeneratedPaperResponse:
    """Convert GeneratedPaper to API response model."""
    sections = []
    
    for section in paper.sections:
        questions = []
        for q in section.questions:
            question_resp = GeneratedQuestionResponse(
                question_id=q.question_id,
                question_text=q.question_text,
                question_type=q.question_type,
                marks=q.marks,
                difficulty=q.difficulty,
                has_diagram=q.has_diagram,
                diagram_url=q.diagram_url,
            )
            
            if q.options:
                question_resp.options = [
                    {"label": o.label, "content": o.content, "is_correct": o.is_correct}
                    for o in q.options
                ]
                question_resp.correct_answer = q.correct_answer
            
            if q.solution:
                question_resp.solution = q.solution
            
            questions.append(question_resp)
        
        sections.append(PaperSectionResponse(
            name=section.name,
            instructions=section.instructions,
            question_count=section.question_count,
            total_marks=section.total_marks,
            questions=questions,
        ))
    
    return GeneratedPaperResponse(
        paper_id=paper.paper_id,
        title=paper.title,
        subject=paper.subject,
        grade=paper.grade,
        duration_minutes=paper.duration_minutes,
        total_questions=paper.total_questions,
        total_marks=paper.total_marks,
        status=paper.status.value,
        source_type=paper.source_type,
        sections=sections,
        created_at=paper.created_at.isoformat(),
        completed_at=paper.completed_at.isoformat() if paper.completed_at else None,
        generation_stats=paper.generation_stats,
        question_paper_url=paper.question_paper_url,
        answer_key_url=paper.answer_key_url,
        marking_scheme_url=paper.marking_scheme_url,
        error_message=paper.error_message,
    )
