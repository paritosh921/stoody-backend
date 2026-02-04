"""
Diagram Generation API Endpoints

Provides endpoints for:
- Generating diagrams for questions
- Reviewing and refining instructions
- Managing diagram records

Uses the diagram pipeline service with:
- Kimi 2.5 for LLM tasks (temp=0.6)
- Nano Banan Pro for image generation
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime

from core.database import DatabaseManager
from api.v1.auth_async import get_database
from services.diagram_pipeline import (
    DiagramPipelineService,
    DiagramGenerationRequest,
    DiagramGenerationResponse,
    DiagramInstructions,
    DiagramReview,
)
from services.diagram_pipeline.models import SubjectType, QuestionDiagramRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/diagrams", tags=["diagrams"])


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateDiagramRequest(BaseModel):
    """Request to generate a diagram for a question."""
    question_id: str
    question_text: str
    subject: Optional[str] = None
    diagram_hints: Optional[str] = None
    max_iterations: int = Field(default=3, ge=1, le=5)
    force_regenerate: bool = False


class GenerateInstructionsRequest(BaseModel):
    """Request to generate only instructions (no image)."""
    question_text: str
    subject: Optional[str] = None
    diagram_hints: Optional[str] = None


class GenerateImageRequest(BaseModel):
    """Request to generate image from custom instructions."""
    question_id: str
    instructions: str
    version: int = 1


class ReviewInstructionsRequest(BaseModel):
    """Request to review diagram instructions."""
    question_text: str
    instructions: str
    subject: Optional[str] = None


class DiagramInstructionsResponse(BaseModel):
    """Response containing diagram instructions."""
    instructions: str
    version: int
    subject: Optional[str] = None
    created_at: datetime


class DiagramImageResponse(BaseModel):
    """Response containing diagram image info."""
    id: str
    path: str
    filename: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    base64_data: Optional[str] = None
    instructions_version: int


class DiagramReviewResponse(BaseModel):
    """Response containing diagram review."""
    is_acceptable: bool
    issues: List[dict]
    suggested_instruction_update: Optional[str] = None


class DiagramRecordResponse(BaseModel):
    """Response containing full diagram record."""
    question_id: str
    current_image: Optional[DiagramImageResponse] = None
    instructions_count: int
    review_count: int
    is_finalized: bool
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_subject(subject_str: Optional[str]) -> Optional[SubjectType]:
    """Parse subject string to SubjectType enum."""
    if not subject_str:
        return None
    
    subject_map = {
        "physics": SubjectType.PHYSICS,
        "chemistry": SubjectType.CHEMISTRY,
        "mathematics": SubjectType.MATHEMATICS,
        "maths": SubjectType.MATHEMATICS,
        "math": SubjectType.MATHEMATICS,
        "biology": SubjectType.BIOLOGY,
        "bio": SubjectType.BIOLOGY,
    }
    
    return subject_map.get(subject_str.lower())


def _get_pipeline_service(db=Depends(get_database)) -> DiagramPipelineService:
    """Get the diagram pipeline service instance."""
    return DiagramPipelineService(db=db)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/generate", response_model=DiagramGenerationResponse)
async def generate_diagram(
    request: GenerateDiagramRequest,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Generate a diagram for a question.
    
    This runs the full pipeline:
    1. LLM1 generates instructions
    2. Nano Banan Pro generates image
    3. LLM2 reviews the result
    4. Refine and repeat if needed (up to max_iterations)
    
    Returns the final diagram image and generation history.
    """
    try:
        pipeline_request = DiagramGenerationRequest(
            question_id=request.question_id,
            question_text=request.question_text,
            subject=_parse_subject(request.subject),
            diagram_hints=request.diagram_hints,
            max_iterations=request.max_iterations,
            force_regenerate=request.force_regenerate,
        )
        
        result = await service.generate_diagram(pipeline_request)
        
        return result
        
    except Exception as e:
        logger.error(f"Diagram generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instructions", response_model=DiagramInstructionsResponse)
async def generate_instructions(
    request: GenerateInstructionsRequest,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Generate only the diagram instructions without creating an image.
    
    Useful for:
    - Previewing what instructions will look like
    - Manual editing before image generation
    """
    try:
        instructions = await service.generate_instructions_only(
            question_text=request.question_text,
            subject=_parse_subject(request.subject),
            diagram_hints=request.diagram_hints,
        )
        
        return DiagramInstructionsResponse(
            instructions=instructions.instructions,
            version=instructions.version,
            subject=instructions.subject.value if instructions.subject else None,
            created_at=instructions.created_at,
        )
        
    except Exception as e:
        logger.error(f"Instruction generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=DiagramImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Generate an image from provided instructions.
    
    Useful when instructions have been manually edited.
    """
    try:
        image = await service.generate_image_from_instructions(
            instructions=request.instructions,
            question_id=request.question_id,
            version=request.version,
        )
        
        return DiagramImageResponse(
            id=image.id,
            path=image.path,
            filename=image.filename,
            format=image.format,
            width=image.width,
            height=image.height,
            base64_data=image.base64_data,
            instructions_version=image.instructions_version,
        )
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review", response_model=DiagramReviewResponse)
async def review_instructions(
    request: ReviewInstructionsRequest,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Review diagram instructions without generating an image.
    
    Useful for:
    - Validating manually written instructions
    - Getting feedback before image generation
    """
    try:
        review = await service.review_instructions(
            question_text=request.question_text,
            instructions=request.instructions,
            subject=_parse_subject(request.subject),
        )
        
        return DiagramReviewResponse(
            is_acceptable=review.is_acceptable,
            issues=[{"code": i.code.value, "details": i.details} for i in review.issues],
            suggested_instruction_update=review.suggested_instruction_update,
        )
        
    except Exception as e:
        logger.error(f"Instruction review failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/record/{question_id}", response_model=DiagramRecordResponse)
async def get_diagram_record(
    question_id: str,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Get the diagram record for a question.
    
    Returns the current diagram state including:
    - Current image (if any)
    - Generation history counts
    - Finalization status
    """
    try:
        record = await service.get_diagram_record(question_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Diagram record not found")
        
        image_response = None
        if record.current_image:
            image_response = DiagramImageResponse(
                id=record.current_image.id,
                path=record.current_image.path,
                filename=record.current_image.filename,
                format=record.current_image.format,
                width=record.current_image.width,
                height=record.current_image.height,
                base64_data=record.current_image.base64_data,
                instructions_version=record.current_image.instructions_version,
            )
        
        return DiagramRecordResponse(
            question_id=record.question_id,
            current_image=image_response,
            instructions_count=len(record.instructions_history),
            review_count=len(record.review_history),
            is_finalized=record.is_finalized,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get diagram record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/record/{question_id}")
async def delete_diagram(
    question_id: str,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Delete a diagram and its record.
    
    This removes:
    - The diagram image file
    - The database record
    """
    try:
        deleted = await service.delete_diagram(question_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Diagram record not found")
        
        return {"success": True, "message": "Diagram deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{question_id}/instructions")
async def get_instructions_history(
    question_id: str,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Get the full instructions history for a question's diagram.
    """
    try:
        record = await service.get_diagram_record(question_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Diagram record not found")
        
        return {
            "question_id": question_id,
            "instructions": [
                {
                    "instructions": inst.instructions,
                    "version": inst.version,
                    "created_at": inst.created_at,
                    "refined_from_version": inst.refined_from_version,
                    "refinement_reason": inst.refinement_reason,
                }
                for inst in record.instructions_history
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get instructions history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{question_id}/reviews")
async def get_review_history(
    question_id: str,
    service: DiagramPipelineService = Depends(_get_pipeline_service),
):
    """
    Get the full review history for a question's diagram.
    """
    try:
        record = await service.get_diagram_record(question_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Diagram record not found")
        
        return {
            "question_id": question_id,
            "reviews": [
                {
                    "is_acceptable": review.is_acceptable,
                    "issues": [{"code": i.code.value, "details": i.details} for i in review.issues],
                    "suggested_instruction_update": review.suggested_instruction_update,
                    "reviewed_at": review.reviewed_at,
                    "review_version": review.review_version,
                }
                for review in record.review_history
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
