"""
Question Paper Builder API

Endpoints for:
- Creating and managing question papers
- AI-powered question generation
- Paper preview and PDF export
"""

import logging
import uuid
import json
import traceback
import os
import math
import base64
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from bson import ObjectId as BsonObjectId
import io

from core.database import DatabaseManager
from api.v1.auth_async import get_database, get_current_user
from api.v1.questions_async import get_admin_id_from_user
from services.diagram_pipeline import DiagramPipelineService
from services.diagram_pipeline.models import SubjectType
from utils.s3_storage import upload_file, get_public_url, is_s3_enabled, delete_file as s3_delete_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/paper-builder", tags=["Paper Builder"])


# ============================================================================
# Models
# ============================================================================

class QuestionOption(BaseModel):
    """Option for MCQ questions."""
    label: str  # A, B, C, D
    text: str
    is_correct: bool = False


class PaperQuestion(BaseModel):
    """A question in a paper."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_number: int = 0
    question_text: str
    question_type: str = "mcq"  # mcq, integer, subjective
    options: List[QuestionOption] = []
    correct_answer: str = ""
    marks: int = 4
    negative_marks: int = 1
    difficulty: str = "medium"  # easy, medium, hard
    topic: str = ""
    has_diagram: bool = False
    diagram_instructions: Optional[str] = None
    diagram_image_url: Optional[str] = None
    diagram_base64: Optional[str] = None
    explanation: Optional[str] = None
    source: str = "manual"  # manual, ai_generated, imported


class PaperSection(BaseModel):
    """A section in a paper (e.g., Section A, Physics, etc.)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    instructions: str = ""
    questions: List[PaperQuestion] = []


class PaperConfig(BaseModel):
    """Configuration for a question paper."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Question Paper"
    subtitle: str = ""
    institution_name: str = ""
    exam_name: str = ""
    subject: str
    standard: str  # Class/Grade
    duration_minutes: int = 180
    total_marks: int = 0
    instructions: List[str] = [
        "Read all questions carefully before answering.",
        "All questions are compulsory.",
        "Write legibly and clearly.",
    ]
    sections: List[PaperSection] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "draft"  # draft, finalized, exported
    created_by: Optional[str] = None


class CreatePaperRequest(BaseModel):
    """Request to create a new paper."""
    title: str
    subject: str
    standard: str
    exam_name: str = ""
    institution_name: str = ""
    duration_minutes: int = 180


class AddQuestionRequest(BaseModel):
    """Request to add a question to a paper."""
    paper_id: str
    section_id: Optional[str] = None
    question_text: str
    question_type: str = "mcq"
    options: List[Dict[str, Any]] = []
    correct_answer: str = ""
    marks: int = 4
    negative_marks: int = 1
    difficulty: str = "medium"
    topic: str = ""
    explanation: Optional[str] = None


class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions using AI."""
    topic: str
    question_type: str = "mcq"
    count: int = 5
    difficulty: str = "medium"
    additional_prompt: Optional[str] = None
    # Optional - will be derived from paper if not provided
    subject: Optional[str] = None
    standard: Optional[str] = None
    exam_type: str = "JEE"  # JEE, NEET, CBSE


class GenerateDiagramRequest(BaseModel):
    """Request to generate a diagram for a question."""
    question_text: Optional[str] = None
    subject: Optional[str] = None
    diagram_hints: Optional[str] = None
    additional_instructions: Optional[str] = None


class PaperResponse(BaseModel):
    """Response containing paper data."""
    id: str
    title: str
    subject: str
    standard: str
    status: str
    total_questions: int
    total_marks: int
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Helper Functions
# ============================================================================

def _get_collection(db: DatabaseManager):
    """Get the papers collection."""
    return "question_papers"


async def _get_paper(db: DatabaseManager, paper_id: str) -> Optional[Dict]:
    """Get a paper by ID."""
    try:
        doc = await db.mongo_find_one(_get_collection(db), {"id": paper_id})
        return doc
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        return None


async def _save_paper(db: DatabaseManager, paper: Dict) -> bool:
    """Save a paper to the database."""
    try:
        paper["updated_at"] = datetime.utcnow().isoformat()
        await db.mongo_update_one(
            _get_collection(db),
            {"id": paper["id"]},
            {"$set": paper},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Error saving paper: {e}")
        return False


def _calculate_total_marks(paper: Dict) -> int:
    """Calculate total marks for a paper."""
    total = 0
    for section in paper.get("sections", []):
        for question in section.get("questions", []):
            total += question.get("marks", 0)
    return total


def _count_questions(paper: Dict) -> int:
    """Count total questions in a paper."""
    total = 0
    for section in paper.get("sections", []):
        total += len(section.get("questions", []))
    return total


def _parse_subject_type(subject_name: Optional[str]) -> Optional[SubjectType]:
    """Map subject aliases to diagram pipeline SubjectType."""
    if not subject_name:
        return None
    subject_map = {
        "physics": SubjectType.PHYSICS,
        "chemistry": SubjectType.CHEMISTRY,
        "mathematics": SubjectType.MATHEMATICS,
        "maths": SubjectType.MATHEMATICS,
        "math": SubjectType.MATHEMATICS,
        "biology": SubjectType.BIOLOGY,
    }
    return subject_map.get(subject_name.strip().lower())


def _resolve_document_type(paper: Dict[str, Any]) -> str:
    """Resolve document type for content sync."""
    allowed_types = {"Practice Sets", "Test Series", "Chapter Notes"}
    doc_type = paper.get("document_type") or os.getenv("PAPER_BUILDER_DOCUMENT_TYPE", "Test Series")
    if doc_type not in allowed_types:
        doc_type = "Test Series"
    return doc_type


def _extract_question_text(question: Dict[str, Any]) -> str:
    """Get question text from paper question object."""
    return (question.get("text") or question.get("question_text") or "").strip()


def _extract_question_type(question: Dict[str, Any]) -> str:
    """Get question type from paper question object."""
    return (question.get("type") or question.get("question_type") or "mcq").strip().lower()


def _normalize_options_for_storage(options: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Normalize options to plain text list + enhanced options."""
    normalized_text: List[str] = []
    enhanced: List[Dict[str, Any]] = []
    for idx, opt in enumerate(options or []):
        label = chr(65 + idx)
        text = ""
        if isinstance(opt, dict):
            text = (opt.get("text") or opt.get("option") or opt.get("content") or "").strip()
            label = opt.get("label") or label
        else:
            text = str(opt).strip()
        if not text:
            continue
        normalized_text.append(text)
        enhanced.append({
            "id": f"opt_{idx}_{uuid.uuid4().hex[:6]}",
            "type": "text",
            "content": text,
            "label": label,
            "description": ""
        })
    return normalized_text, enhanced


def _extract_image_id_from_url(url: Optional[str]) -> Optional[str]:
    """Extract image ID from /api/v1/images/{id} URL."""
    if not url:
        return None
    match = re.search(r"/api/v1/images/([A-Za-z0-9\\-]+)$", url)
    if match:
        return match.group(1)
    return None


def _detect_image_format(data: bytes) -> Tuple[str, str]:
    """Detect image format from binary data, return (extension, content_type)."""
    if data.startswith(b"\xFF\xD8\xFF"):
        return "jpeg", "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif", "image/gif"
    if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
        return "webp", "image/webp"
    return "png", "image/png"


async def _store_diagram_image(
    db: DatabaseManager,
    image_base64: str,
    document_id: str,
    current_user: Dict[str, Any],
    is_b2c: bool = False
) -> Optional[Dict[str, Any]]:
    """Persist diagram image to storage + images collection."""
    if not image_base64:
        return None

    # Strip data URI prefix if present
    if image_base64.startswith("data:") and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        logger.warning("Failed to decode diagram base64 data")
        return None

    ext, content_type = _detect_image_format(image_bytes)
    image_id = str(uuid.uuid4())
    filename = f"{image_id}.{ext}"
    local_path = os.path.join("uploads", "diagrams", document_id, filename)

    success, storage_path = await upload_file(
        image_bytes,
        local_path,
        content_type=content_type
    )
    if not success:
        logger.warning("Diagram upload failed; skipping image metadata save")
        return None

    image_metadata = {
        "_id": image_id,
        "filename": filename,
        "original_filename": filename,
        "size": len(image_bytes),
        "content_type": content_type,
        "uploaded_by": current_user.get("user_id"),
        "uploaded_at": datetime.utcnow(),
        "is_processed": True,
        "file_path": storage_path,
        "source_pdf": f"{document_id}.pdf",
        "tags": ["diagram", "paper_builder", "ai_generated"],
        "is_s3": is_s3_enabled()
    }

    success = False
    if is_b2c:
        success = await db.b2c_update_one("images", {"_id": image_id}, {"$set": image_metadata}, upsert=True)
    else:
        success = await db.mongo_update_one("images", {"_id": image_id}, {"$set": image_metadata}, upsert=True)

    if not success:
        logger.error(f"Failed to save image metadata to DB for id: {image_id}")
        return None

    return {
        "id": image_id,
        "filename": filename,
        "path": storage_path,
        "url": f"/api/v1/images/{image_id}",
        "size": len(image_bytes),
        "content_type": content_type,
        "type": "diagram"
    }


async def _process_paper_diagrams(
    db: DatabaseManager,
    paper: Dict[str, Any],
    current_user: Dict[str, Any],
    is_b2c: bool
) -> int:
    """Ensure diagram images are saved to storage/DB and normalize question fields."""
    saved_count = 0
    document_id = paper.get("document_id") or paper.get("id") or str(uuid.uuid4())

    for section in paper.get("sections", []):
        for question in section.get("questions", []):
            diagram_base64 = question.get("diagram_base64")
            diagram_url = question.get("diagram_url") or question.get("diagram_image_url")
            diagram_image_id = question.get("diagram_image_id") or _extract_image_id_from_url(diagram_url)

            if diagram_image_id and not question.get("diagram_image_id"):
                question["diagram_image_id"] = diagram_image_id

            # If a stable image reference already exists, do not upload duplicate base64 data.
            if diagram_base64 and diagram_image_id:
                stable_url = question.get("diagram_url") or question.get("diagram_image_url") or f"/api/v1/images/{diagram_image_id}"
                question["diagram_url"] = stable_url
                question["diagram_image_url"] = stable_url
                question["has_diagram"] = True
                question.pop("diagram_base64", None)
                continue

            if diagram_base64:
                image_ref = await _store_diagram_image(
                    db=db,
                    image_base64=diagram_base64,
                    document_id=document_id,
                    current_user=current_user,
                    is_b2c=is_b2c
                )
                if image_ref:
                    question["diagram_image_id"] = image_ref["id"]
                    question["diagram_url"] = image_ref["url"]
                    question["diagram_image_url"] = image_ref["url"]
                    question["has_diagram"] = True
                    question.pop("diagram_base64", None)
                    saved_count += 1
            elif diagram_image_id and not diagram_url:
                question["diagram_url"] = f"/api/v1/images/{diagram_image_id}"
                question["diagram_image_url"] = question["diagram_url"]
                question["has_diagram"] = True

    return saved_count


async def _sync_paper_to_content(
    db: DatabaseManager,
    paper: Dict[str, Any],
    current_user: Dict[str, Any]
) -> Dict[str, Any]:
    """Sync paper content to documents + questions collections."""
    is_b2c = current_user.get("user_type") == "b2c_admin"
    document_id = paper.get("document_id") or paper.get("id") or str(uuid.uuid4())
    document_type = _resolve_document_type(paper)

    # Flatten questions
    flat_questions: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for section in paper.get("sections", []):
        for question in section.get("questions", []):
            flat_questions.append((section, question))

    if not flat_questions:
        return {"skipped": True, "reason": "no_questions"}

    # Resolve admin ID for tenant isolation
    admin_id = get_admin_id_from_user(current_user)
    try:
        admin_oid = BsonObjectId(admin_id)
    except Exception:
        admin_oid = admin_id

    # Use existing document metadata if present
    if is_b2c:
        existing_doc = await db.b2c_find_one("documents", {"document_id": document_id})
    else:
        existing_doc = await db.mongo_find_one("documents", {"document_id": document_id})

    filename = (existing_doc or {}).get("filename") or f"{document_id}.pdf"
    file_path = (existing_doc or {}).get("file_path") or f"uploads/documents/{document_type}/{filename}"

    total_points = sum((q.get("marks", 0) or 0) for _, q in flat_questions)
    total_images = sum(1 for _, q in flat_questions if q.get("diagram_image_id"))

    document_metadata = {
        "document_id": document_id,
        "title": paper.get("title", "Question Paper"),
        "document_type": document_type,
        "subject": paper.get("subject", "General"),
        "difficulty": paper.get("difficulty", "medium"),
        "course_plan": paper.get("course_plan"),
        "standard": paper.get("standard"),
        "section": None,
        "teacher_ids": paper.get("teacher_ids", []),
        "file_path": file_path,
        "filename": filename,
        "uploaded_by": (existing_doc or {}).get("uploaded_by") or current_user.get("user_id"),
        "admin_id": (existing_doc or {}).get("admin_id") or admin_oid,
        "uploaded_at": (existing_doc or {}).get("uploaded_at") or datetime.utcnow(),
        "ocr_status": "completed",
        "ocr_job_id": (existing_doc or {}).get("ocr_job_id"),
        "extracted_questions_count": len(flat_questions),
        "extracted_images_count": total_images,
        "pages_count": (existing_doc or {}).get("pages_count", 0),
        "total_points": total_points if document_type == "Test Series" else None,
        "total_minutes": paper.get("duration_minutes"),
        "is_validated": (existing_doc or {}).get("is_validated", False),
        "is_active": (existing_doc or {}).get("is_active", True),
        "is_s3": (existing_doc or {}).get("is_s3", False)
    }

    if is_b2c:
        await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": document_metadata}, upsert=True)
    else:
        await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": document_metadata}, upsert=True)

    # Clear existing questions for this document
    if is_b2c:
        await db.b2c_delete_many("questions", {"document_id": document_id})
    else:
        await db.mongo_delete_many("questions", {"document_id": document_id})

    question_docs: List[Dict[str, Any]] = []
    question_number = 0
    for section, question in flat_questions:
        question_number += 1
        question_text = _extract_question_text(question)
        question_type = _extract_question_type(question)
        options_text, enhanced_options = _normalize_options_for_storage(question.get("options", []))

        diagram_image_id = question.get("diagram_image_id") or _extract_image_id_from_url(
            question.get("diagram_url") or question.get("diagram_image_url")
        )
        question_figures = []
        if diagram_image_id:
            question_figures.append({
                "id": diagram_image_id,
                "url": f"/api/v1/images/{diagram_image_id}",
                "type": "diagram"
            })

        question_docs.append({
            "id": question.get("id") or str(uuid.uuid4()),
            "text": question_text,
            "question_text": question_text,
            "subject": paper.get("subject", "General"),
            "difficulty": question.get("difficulty") or paper.get("difficulty", "medium"),
            "document_type": document_type,
            "extracted_at": datetime.utcnow(),
            "pdf_source": filename,
            "document_id": document_id,
            "images": [],
            "question_figures": question_figures,
            "options": options_text if question_type == "mcq" else [],
            "enhanced_options": enhanced_options if question_type == "mcq" else [],
            "correct_answer": question.get("correct_answer"),
            "question_type": question_type,
            "is_image_based_mcq": False,
            "metadata": {
                "paper_id": paper.get("id"),
                "section_id": section.get("id"),
                "section_title": section.get("title") or section.get("name"),
                "question_number": question_number,
                "topic": question.get("topic"),
                "explanation": question.get("explanation"),
                "diagram_instructions": question.get("diagram_instructions"),
                "source": question.get("source", "manual")
            },
            "points": question.get("marks", 4),
            "penalty": question.get("negative_marks", 1),
            "created_by": current_user.get("user_id"),
            "created_at": datetime.utcnow()
        })

    if is_b2c:
        for q_doc in question_docs:
            await db.b2c_insert_one("questions", q_doc)
    else:
        await db.mongo_insert_many("questions", question_docs)

    # Update document counts after insert
    counts_update = {
        "extracted_questions_count": len(question_docs),
        "extracted_images_count": total_images,
        "ocr_status": "completed",
        "ocr_completed_at": datetime.utcnow()
    }
    if is_b2c:
        await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": counts_update})
    else:
        await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": counts_update})

    return {
        "document_id": document_id,
        "questions_saved": len(question_docs),
        "images_saved": total_images
    }


async def _delete_paper_content(
    db: DatabaseManager,
    document_id: str,
    is_b2c: bool
) -> None:
    """Delete synced document/questions/images for a paper."""
    if not document_id:
        return

    if is_b2c:
        questions = await db.b2c_find("questions", {"document_id": document_id})
    else:
        questions = await db.mongo_find("questions", {"document_id": document_id})

    image_ids = set()
    for q in questions:
        for img_ref in (q.get("images") or []):
            if isinstance(img_ref, dict):
                image_ids.add(img_ref.get("id"))
            else:
                image_ids.add(img_ref)
        for fig_ref in (q.get("question_figures") or []):
            if isinstance(fig_ref, dict):
                image_ids.add(fig_ref.get("id"))
            else:
                image_ids.add(fig_ref)

    # Delete questions
    if is_b2c:
        await db.b2c_delete_many("questions", {"document_id": document_id})
    else:
        await db.mongo_delete_many("questions", {"document_id": document_id})

    # Delete images and files
    for image_id in image_ids:
        if not image_id:
            continue
        if is_b2c:
            img_doc = await db.b2c_find_one("images", {"_id": image_id})
        else:
            img_doc = await db.mongo_find_one("images", {"_id": image_id})
        if img_doc:
            file_path = img_doc.get("file_path")
            if file_path:
                await s3_delete_file(file_path)
        if is_b2c:
            await db.b2c_delete_one("images", {"_id": image_id})
        else:
            await db.mongo_delete_one("images", {"_id": image_id})

    # Delete document entry
    if is_b2c:
        await db.b2c_delete_one("documents", {"document_id": document_id})
    else:
        await db.mongo_delete_one("documents", {"document_id": document_id})


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/papers", response_model=PaperResponse)
async def create_paper(
    request: CreatePaperRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Create a new question paper."""
    try:
        paper_id = str(uuid.uuid4())
        
        paper = {
            "id": paper_id,
            "title": request.title,
            "subtitle": "",
            "institution_name": request.institution_name,
            "exam_name": request.exam_name,
            "subject": request.subject,
            "standard": request.standard,
            "duration_minutes": request.duration_minutes,
            "total_marks": 0,
            "instructions": [
                "Read all questions carefully before answering.",
                "All questions are compulsory.",
                "Use of calculator is not permitted.",
            ],
            "sections": [
                {
                    "id": str(uuid.uuid4())[:8],
                    "title": "Section A",
                    "instructions": "",
                    "questions": []
                }
            ],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "draft",
            "created_by": current_user.get("user_id") or current_user.get("username"),
        }
        
        await _save_paper(db, paper)
        
        return PaperResponse(
            id=paper_id,
            title=request.title,
            subject=request.subject,
            standard=request.standard,
            status="draft",
            total_questions=0,
            total_marks=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
    except Exception as e:
        logger.error(f"Error creating paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers")
async def list_papers(
    status: Optional[str] = None,
    subject: Optional[str] = None,
    standard: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    skip: int = Query(default=0, ge=0),
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """List all papers for the current user."""
    try:
        query = {}
        
        if status:
            query["status"] = status
        if subject:
            query["subject"] = subject
        if standard:
            query["standard"] = standard
        
        # Get papers
        docs = await db.mongo_find(
            _get_collection(db),
            query,
            limit=limit,
            skip=skip,
            sort=[("updated_at", -1)]
        )

        # Handle both list and cursor results
        papers = []
        if hasattr(docs, '__aiter__'):
            async for doc in docs:
                doc.pop("_id", None)
                papers.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "subject": doc["subject"],
                    "standard": doc["standard"],
                    "status": doc["status"],
                    "total_questions": _count_questions(doc),
                    "total_marks": _calculate_total_marks(doc),
                    "created_at": doc["created_at"],
                    "updated_at": doc["updated_at"],
                })
        else:
            # It's a list
            for doc in docs:
                doc.pop("_id", None)
                papers.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "subject": doc["subject"],
                    "standard": doc["standard"],
                    "status": doc["status"],
                    "total_questions": _count_questions(doc),
                    "total_marks": _calculate_total_marks(doc),
                    "created_at": doc["created_at"],
                    "updated_at": doc["updated_at"],
                })

        return {"papers": papers, "total": len(papers)}
        
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
async def get_paper(
    paper_id: str,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Get a paper by ID with full details."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        paper.pop("_id", None)
        paper["total_questions"] = _count_questions(paper)
        paper["total_marks"] = _calculate_total_marks(paper)
        
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/papers/{paper_id}")
async def update_paper(
    paper_id: str,
    updates: Dict[str, Any],
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Update paper configuration."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Update allowed fields
        allowed_fields = [
            "title", "subtitle", "institution_name", "exam_name",
            "duration_minutes", "instructions", "sections", "status"
        ]
        
        for field in allowed_fields:
            if field in updates:
                paper[field] = updates[field]
        
        # Ensure content sync identifiers
        if not paper.get("document_id"):
            paper["document_id"] = paper_id
        if not paper.get("document_type"):
            paper["document_type"] = _resolve_document_type(paper)

        # Recalculate totals
        paper["total_marks"] = _calculate_total_marks(paper)

        # Persist diagrams to storage/DB and normalize fields
        try:
            is_b2c = current_user.get("user_type") == "b2c_admin"
            await _process_paper_diagrams(db, paper, current_user, is_b2c)
        except Exception as diag_err:
            logger.warning(f"Diagram processing failed for paper {paper_id}: {diag_err}")

        await _save_paper(db, paper)

        # Sync paper to content system (documents + questions)
        sync_result = None
        try:
            sync_result = await _sync_paper_to_content(db, paper, current_user)
        except Exception as sync_err:
            logger.error(f"Content sync failed for paper {paper_id}: {sync_err}")

        paper.pop("_id", None)
        paper["total_questions"] = _count_questions(paper)
        paper["total_marks"] = _calculate_total_marks(paper)

        response = {"success": True, "message": "Paper updated", "paper": paper}
        if sync_result:
            response["content_sync"] = sync_result
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/papers/{paper_id}")
async def delete_paper(
    paper_id: str,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Delete a paper."""
    try:
        # Clean up content-system artifacts (documents/questions/images)
        try:
            paper = await _get_paper(db, paper_id)
            document_id = (paper or {}).get("document_id") or paper_id
            is_b2c = current_user.get("user_type") == "b2c_admin"
            await _delete_paper_content(db, document_id, is_b2c)
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup content for paper {paper_id}: {cleanup_err}")

        result = await db.mongo_delete_one(_get_collection(db), {"id": paper_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return {"success": True, "message": "Paper deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/questions")
async def add_question(
    paper_id: str,
    request: AddQuestionRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Add a question to a paper."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Find the section
        section_id = request.section_id
        if not section_id and paper["sections"]:
            section_id = paper["sections"][0]["id"]
        
        section_found = False
        for section in paper["sections"]:
            if section["id"] == section_id:
                # Create the question
                question_id = str(uuid.uuid4())[:8]
                question_number = len(section["questions"]) + 1
                
                # Convert options
                options = []
                for i, opt in enumerate(request.options):
                    options.append({
                        "label": chr(65 + i),  # A, B, C, D
                        "text": opt.get("text", ""),
                        "is_correct": opt.get("is_correct", False),
                    })
                
                question = {
                    "id": question_id,
                    "question_number": question_number,
                    "question_text": request.question_text,
                    "question_type": request.question_type,
                    "options": options,
                    "correct_answer": request.correct_answer,
                    "marks": request.marks,
                    "negative_marks": request.negative_marks,
                    "difficulty": request.difficulty,
                    "topic": request.topic,
                    "has_diagram": False,
                    "diagram_instructions": None,
                    "diagram_image_url": None,
                    "diagram_base64": None,
                    "explanation": request.explanation,
                    "source": "manual",
                }
                
                section["questions"].append(question)
                section_found = True
                break
        
        if not section_found:
            raise HTTPException(status_code=404, detail="Section not found")
        
        await _save_paper(db, paper)
        
        return {"success": True, "question_id": question_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/generate-questions")
async def generate_questions(
    paper_id: str,
    request: GenerateQuestionsRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Generate questions using AI (Kimi K2.5 ONLY - temperature 0.6)."""
    try:
        paper = await _get_paper(db, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        # Get subject and standard from paper if not provided
        subject = request.subject or paper.get("subject", "General")
        standard = request.standard or paper.get("standard", "11")
        try:
            min_diagram_ratio = float(os.getenv("DIAGRAM_MIN_RATIO", "0.3"))
        except ValueError:
            min_diagram_ratio = 0.3
        min_diagram_ratio = max(0.0, min(1.0, min_diagram_ratio))

        min_diagram_questions_target = 0
        if request.count > 0 and min_diagram_ratio > 0:
            min_diagram_questions_target = max(1, int(math.ceil(request.count * min_diagram_ratio)))

        def build_system_prompt(
            count: int,
            min_diagram_count: int = 0,
            require_diagram_only: bool = False,
        ) -> str:
            diagram_distribution_block = ""
            if require_diagram_only and count > 0:
                diagram_distribution_block = f"""
DIAGRAM QUOTA (MANDATORY):
- ALL {count} questions must be naturally diagram-based.
- Set "needs_diagram": true for every question.
- Each question must include a concrete visual setup (shape/system/graph/circuit) with specific values and labels."""
            elif min_diagram_count > 0:
                diagram_distribution_block = f"""
DIAGRAM QUOTA (MANDATORY):
- At least {min_diagram_count} out of {count} questions must be naturally diagram-based.
- For diagram-based questions, set "needs_diagram": true and provide detailed "diagram_description".
- Do not force diagrams for purely symbolic/algebraic questions."""

            return f"""You are an expert {request.exam_type} exam question creator for {subject}.
Generate exactly {count} {request.question_type.upper()} questions on: {request.topic}
Difficulty: {request.difficulty} | Class: {standard}

IMPORTANT: Output ONLY valid JSON. Keep explanations SHORT (1-2 sentences max).
IMPORTANT: Every question must be scientifically correct and internally consistent. Avoid ambiguous or misleading setups.
{diagram_distribution_block}

DIAGRAM DECISION - BASED ON QUESTION CONTENT:

SET needs_diagram: true ONLY when the question text contains:
- A NAMED geometric shape (triangle ABC, circle O, rectangle PQRS)
- SPECIFIC measurements (angle = 60°, side = 5cm, radius = 3)
- A curve/graph equation with specific points (parabola y²=4x at point (1,2))
- A physical setup (circuit with 2Ω resistor, lens with f=10cm)

SET needs_diagram: false when the question is:
- Pure calculation/algebra (find x, solve equation, evaluate limit)
- Abstract concepts (if a,b,c are vectors such that...)
- Proofs without specific figures
- Finding parameters (find λ, μ, ν)

DIAGRAM DESCRIPTION MUST contain:
1. The EXACT shape/object name from the question
2. ALL numeric values mentioned (copy angles, lengths, coordinates exactly)
3. Labels for all named points (A, B, C, P, Q, etc.)

EXAMPLE - needs_diagram: true:
Question: "In triangle ABC, angle B = 90°, AB = 3cm, BC = 4cm. Find AC."
diagram_description: "Right triangle ABC with right angle at B, AB=3cm (vertical), BC=4cm (horizontal), find hypotenuse AC"

EXAMPLE - needs_diagram: false:
Question: "If a + b + c = 0, prove that a×b = b×c = c×a"
(No specific shape, just abstract vectors)

JSON format example (mix of with/without diagrams):
[
  {{
    "question_text": "If λ + μ + ν = 0 where λ, μ, ν are scalars...",
    "options": [...],
    "correct_answer": "B",
    "explanation": "Using properties of scalar triple product...",
    "needs_diagram": false,
    "diagram_description": ""
  }},
  {{
    "question_text": "In triangle ABC, if angle A = 60° and AB = AC = 4 cm, find the area.",
    "options": [...],
    "correct_answer": "C",
    "explanation": "Using area formula for isoceles triangle...",
    "needs_diagram": true,
    "diagram_description": "Draw isoceles triangle ABC with vertex A at top, angle 60° marked at A, sides AB and AC labeled 4 cm each"
  }},
  {{
    "question_text": "A parabola y² = 4ax passes through point (2,4). Find the equation of tangent at this point.",
    "options": [...],
    "correct_answer": "A", 
    "explanation": "Using tangent formula for parabola...",
    "needs_diagram": true,
    "diagram_description": "Draw parabola opening rightward with vertex at origin, mark point P(2,4) on curve, draw tangent line at P"
  }}
]

{f"Note: {request.additional_prompt}" if request.additional_prompt else ""}

Rules:
- Output ONLY the JSON array, no markdown, no extra text
- Keep each explanation under 50 words
- Ensure complete, valid JSON
- Only set needs_diagram: true if the question has a SPECIFIC shape/figure with measurements
- Always COPY exact values (angles, lengths, coordinates) from question into diagram_description
- Prefer diagram-friendly question designs first until quota is met; then use needs_diagram: false for non-visual questions
"""

        system_prompt = build_system_prompt(
            request.count,
            min_diagram_count=min_diagram_questions_target,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {request.count} questions on {request.topic} for {subject}"}
        ]

        # Use Kimi K2.5 ONLY (temperature 0.6, no thinking mode)
        from services.diagram_pipeline.kimi_client import KimiClient
        kimi = KimiClient()
        
        if not kimi.api_key:
            raise HTTPException(
                status_code=500, 
                detail="KIMI_API_KEY not configured. Kimi K2.5 is required for question generation."
            )
        
        try:
            # Note: Kimi K2.5 may not support response_format, so we don't use it
            # Use higher max_tokens to avoid truncation
            response = await kimi._call_api(messages, max_tokens=8000)
            logger.info(f"Questions generated using Kimi K2.5 (temp=0.6), response length: {len(response) if response else 0}")
            logger.debug(f"Kimi raw response (first 500 chars): {response[:500] if response else 'EMPTY'}")
        except Exception as kimi_error:
            logger.error(f"Kimi K2.5 API failed: {kimi_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Kimi K2.5 API error: {str(kimi_error)}"
            )

        # Check for empty response
        if not response or not response.strip():
            logger.error("Kimi K2.5 returned empty response")
            raise HTTPException(status_code=500, detail="AI returned empty response")

        def _parse_questions_response(raw_response: str) -> List[Dict[str, Any]]:
            try:
                import re

                # Strip markdown code blocks if present
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith("```"):
                    # Remove opening ```json or ```
                    cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
                    # Remove closing ```
                    cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)

                # First try standard JSON parsing
                json_match = re.search(r'\[[\s\S]*\]', cleaned_response)
                questions_data = None

                if json_match:
                    json_str = json_match.group()
                    try:
                        questions_data = json.loads(json_str, strict=False)
                    except json.JSONDecodeError:
                        # Response might be truncated - try to extract complete question objects
                        logger.warning("JSON truncated, attempting to extract complete questions")

                        # Find all complete JSON objects that look like questions
                        complete_questions = []
                        bracket_count = 0
                        current_obj = ""
                        in_string = False
                        escape_next = False

                        for char in json_str:
                            if escape_next:
                                escape_next = False
                                current_obj += char
                                continue
                            if char == '\\':
                                escape_next = True
                                current_obj += char
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                            if not in_string:
                                if char == '{':
                                    bracket_count += 1
                                elif char == '}':
                                    bracket_count -= 1
                                    if bracket_count == 0 and current_obj.strip():
                                        current_obj += char
                                        try:
                                            obj = json.loads(current_obj.strip().lstrip('[,').rstrip(','), strict=False)
                                            if "question_text" in obj and "options" in obj:
                                                complete_questions.append(obj)
                                        except Exception:
                                            pass
                                        current_obj = ""
                                        continue
                            current_obj += char

                        if complete_questions:
                            questions_data = complete_questions
                            logger.info(f"Recovered {len(complete_questions)} complete questions from truncated response")

                if questions_data is None:
                    questions_data = json.loads(cleaned_response, strict=False)
                    if isinstance(questions_data, dict) and "questions" in questions_data:
                        questions_data = questions_data["questions"]

                return questions_data if isinstance(questions_data, list) else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Kimi K2.5 response: {e}")
                logger.error(f"Response content: {raw_response[:1000] if raw_response else 'NONE'}")
                raise HTTPException(status_code=500, detail="Failed to parse AI response")

        questions_data = _parse_questions_response(response)

        def _local_sanity_filter(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Basic structural checks before expensive validation."""
            valid = []
            for q in questions:
                question_text = q.get("question_text", "")
                options = q.get("options", [])
                correct = str(q.get("correct_answer", "")).strip().upper()
                if not question_text or not options:
                    continue
                labels = []
                for i, opt in enumerate(options):
                    label = opt.get("label") if isinstance(opt, dict) else None
                    label = (label or chr(65 + i)).upper()
                    labels.append(label)
                if correct and correct not in labels:
                    continue
                valid.append(q)
            return valid

        questions_data = _local_sanity_filter(questions_data)

        def _normalize_question_text(text: str) -> str:
            import re
            return re.sub(r'\s+', ' ', (text or "")).strip().lower()

        async def _validate_questions_batch(questions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            """Validate questions for scientific/logical correctness using Kimi."""
            validation_enabled = os.getenv("QUESTION_VALIDATION_ENABLED", "true").lower() not in ["0", "false", "no"]
            fail_open = os.getenv("QUESTION_VALIDATION_FAIL_OPEN", "false").lower() in ["1", "true", "yes"]
            if not validation_enabled or not questions:
                return questions, []

            # Build validation prompt
            payload = []
            for i, q in enumerate(questions):
                payload.append({
                    "index": i,
                    "question_text": q.get("question_text", ""),
                    "options": q.get("options", []),
                    "correct_answer": q.get("correct_answer", ""),
                    "diagram_description": q.get("diagram_description", ""),
                })

            system_prompt = f"""You are a strict {request.exam_type} examiner for {subject} (Class {standard}).
Your task is to VALIDATE each question for scientific correctness and logical consistency.
You MUST attempt to solve the question. If you cannot solve uniquely, mark it INVALID.

Reject questions that have:
- Contradictions or impossible conditions
- Violations of fundamental laws (physics/math)
- Missing data that makes the question unsolvable
- Diagram requirements that conflict with the text
- Units that make no sense
- Correct answer that does not match the solved answer

Output ONLY JSON:
{{"results":[{{"index":0,"is_valid":true,"issues":[],"expected_answer":"B"}}]}}

Rules:
- For MCQ, expected_answer must be the option label (A/B/C/D).
- For integer-type, expected_answer is the numeric value as a string.
- If invalid, set is_valid=false and list short issues (max 2).
- Be strict but fair. If you are uncertain, mark invalid."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}
            ]

            try:
                response_text = await kimi._call_api(messages, max_tokens=1500, response_format={"type": "json_object"})
            except Exception as e:
                logger.error(f"Question validation failed: {e}")
                if fail_open:
                    return questions, []
                return [], [{"error": "validation_call_failed"}]

            # Parse validation response
            try:
                import re
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    match = re.search(r'\{[\s\S]*\}', response_text)
                    data = json.loads(match.group()) if match else {}
                results = data.get("results", [])
            except Exception as e:
                logger.error(f"Failed to parse validation response: {e}. Raw: {response_text[:500]}")
                if fail_open:
                    return questions, []
                return [], [{"error": "validation_parse_failed"}]

            invalid_indices = set()
            invalid_reports = []
            expected_answers = {}
            for item in results:
                idx = item.get("index")
                if idx is None:
                    continue
                if "expected_answer" in item:
                    expected_answers[idx] = str(item.get("expected_answer", "")).strip()
                if not item.get("is_valid", True):
                    invalid_indices.add(idx)
                    invalid_reports.append({
                        "index": idx,
                        "issues": item.get("issues", []),
                    })

            valid_questions = []
            for i, q in enumerate(questions):
                if i in invalid_indices:
                    continue
                # If expected answer provided, enforce match
                if i in expected_answers:
                    expected = expected_answers[i]
                    correct = str(q.get("correct_answer", "")).strip()
                    if expected and correct and expected.upper() != correct.upper():
                        invalid_reports.append({
                            "index": i,
                            "issues": [f"ANSWER_MISMATCH: expected {expected} but correct_answer is {correct}"],
                        })
                        continue
                valid_questions.append(q)

            if invalid_reports:
                logger.info(f"[QUESTION-VALIDATION] Invalid questions: {invalid_reports}")

            return valid_questions, invalid_reports

        # Validate and de-duplicate questions (regenerate if needed)
        seen_questions = set()
        valid_questions, invalid_reports = await _validate_questions_batch(questions_data)
        deduped_questions = []
        for q in valid_questions:
            key = _normalize_question_text(q.get("question_text", ""))
            if not key or key in seen_questions:
                continue
            seen_questions.add(key)
            deduped_questions.append(q)
        questions_data = deduped_questions

        # Top-up if we dropped invalid/duplicate questions
        max_attempts = int(os.getenv("QUESTION_VALIDATION_MAX_ATTEMPTS", "2"))
        attempts = 0
        while len(questions_data) < request.count and attempts < max_attempts:
            attempts += 1
            remaining = request.count - len(questions_data)
            logger.info(f"[QUESTION-VALIDATION] Regenerating {remaining} questions (attempt {attempts}/{max_attempts})")

            current_diagram_count = sum(
                1 for q in questions_data if bool(q.get("needs_diagram"))
            )
            remaining_diagram_target = max(0, min_diagram_questions_target - current_diagram_count)
            remaining_diagram_target = min(remaining, remaining_diagram_target)

            regen_system_prompt = build_system_prompt(
                remaining,
                min_diagram_count=remaining_diagram_target,
            )
            regen_messages = [
                {"role": "system", "content": regen_system_prompt},
                {"role": "user", "content": f"Generate {remaining} NEW questions on {request.topic} for {subject}. Do not repeat previous questions."}
            ]

            try:
                regen_response = await kimi._call_api(regen_messages, max_tokens=6000)
                regen_questions = _parse_questions_response(regen_response)
            except Exception as e:
                logger.error(f"Regeneration failed: {e}")
                break

            regen_valid, _ = await _validate_questions_batch(regen_questions)
            for q in regen_valid:
                key = _normalize_question_text(q.get("question_text", ""))
                if not key or key in seen_questions:
                    continue
                seen_questions.add(key)
                questions_data.append(q)

        if len(questions_data) < request.count:
            logger.warning(f"[QUESTION-VALIDATION] Generated {len(questions_data)} valid questions out of requested {request.count}")
        
        # Generate questions and diagrams
        generated_questions_response = []
        
        # Diagram pipeline for reliable diagram instructions + image generation
        from services.diagram_pipeline.models import DiagramGenerationRequest as DiagramRequest
        
        diagram_service = DiagramPipelineService(db=db)
        image_client = diagram_service.image_generator
        
        # DEBUG: Print to console to ensure we see the output
        print(f"[DEBUG] Diagram image API key present: {bool(image_client.api_key)}")
        print(f"[DEBUG] Diagram image model: {image_client.model}")
        print(f"[DEBUG] Subject: {subject}, is_science: {subject.lower() in ['physics', 'chemistry', 'biology', 'mathematics', 'math']}")
        
        # Log parsed questions to debug diagram fields
        logger.info(f"Parsed {len(questions_data)} questions from Kimi response")
        for i, q in enumerate(questions_data):
            logger.info(f"Q{i+1} diagram fields: needs_diagram={q.get('needs_diagram')}, diagram_desc={q.get('diagram_description', '')[:80] if q.get('diagram_description') else 'None'}")

        def validate_diagram_relevance(question_text: str, diagram_desc: str) -> tuple:
            """
            Validates that diagram description is relevant to the question.
            Returns (is_valid, reason) tuple.
            """
            import re

            def _normalize_visual_text(text: str) -> str:
                subscript_map = str.maketrans({
                    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
                    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                    "−": "-", "–": "-", "—": "-",
                    "°": " deg ",
                })
                return (text or "").translate(subscript_map)

            def _extract_labels(text: str) -> set:
                t = _normalize_visual_text(text)
                labels = set()
                seq_patterns = [
                    r"\b(?:(?i:triangle|line|segment|chord|arc|angle|quadrilateral|polygon|points?|vertices|vertex))\s+([A-Z]{2,6})\b",
                    r"\b(?:(?i:between|join(?:ing)?|from|to))\s+([A-Z]{2,6})\b",
                ]
                for pat in seq_patterns:
                    for seq in re.findall(pat, t):
                        labels.update(list(seq.upper()))

                single_patterns = [
                    r"\b(?:(?i:point|vertex|label(?:ed)?|mass|block|particle|node|terminal|charge))\s+([A-Z])\b",
                    r"\b([A-Z])\s*=",
                    r"\b([A-Z])\s*\(",
                ]
                for pat in single_patterns:
                    for lbl in re.findall(pat, t):
                        labels.add(lbl.upper())

                for seq in re.findall(r'\b([A-Z]{2,6})\b', t):
                    seq = seq.upper()
                    if seq in {"JEE", "NEET", "MCQ", "N", "KG", "CM", "MM", "MS"}:
                        continue
                    labels.update(list(seq))
                return labels

            q_lower = question_text.lower()
            d_lower = diagram_desc.lower()

            # REJECT: Empty or very short descriptions
            if len(diagram_desc.strip()) < 20:
                return False, "Description too short (< 20 chars)"

            # REJECT: Generic/vague descriptions that don't reference anything specific
            vague_patterns = [
                r"^draw\s+a\s+(simple\s+)?diagram",
                r"^illustrate\s+the",
                r"^show\s+the\s+concept",
                r"^represent\s+the\s+problem",
                r"^create\s+a\s+visual",
            ]
            for pattern in vague_patterns:
                if re.match(pattern, d_lower.strip()):
                    return False, "Description too generic"

            # EXTRACT: Named labels/points with context-aware parsing
            q_labels = _extract_labels(question_text)
            d_labels = _extract_labels(diagram_desc)

            # EXTRACT: Find numbers from both
            q_numbers = set(re.findall(r'\d+(?:\.\d+)?', _normalize_visual_text(question_text)))
            d_numbers = set(re.findall(r'\d+(?:\.\d+)?', _normalize_visual_text(diagram_desc)))
            warnings = []

            # VALIDATION: If question has specific labels, diagram should include ALL of them
            if q_labels:
                missing_labels = q_labels - d_labels
                if missing_labels:
                    warnings.append(f"Missing labels {sorted(missing_labels)}")

            # VALIDATION: Numeric coverage should be substantial, but not absolute.
            # Constants like g=10 may be present in text but not always visualized in diagrams.
            if q_numbers:
                matched_numbers = q_numbers & d_numbers
                if len(q_numbers) <= 2:
                    required_matches = len(q_numbers)
                else:
                    # ceil(2n/3): n=3 -> 2, n=4 -> 3, n=5 -> 4
                    required_matches = (2 * len(q_numbers) + 2) // 3
                if len(matched_numbers) < required_matches:
                    missing_numbers = q_numbers - d_numbers
                    warnings.append(
                        f"Low numeric coverage {len(matched_numbers)}/{len(q_numbers)}; "
                        f"missing values {sorted(missing_numbers)}"
                    )

            # VALIDATION: If question mentions a shape, diagram description should include it
            shape_terms = [
                "triangle", "circle", "rectangle", "square", "parabola", "ellipse",
                "hyperbola", "graph", "line", "segment", "chord", "tangent",
                "sector", "cone", "cylinder", "sphere", "prism",
                "circuit", "resistor", "capacitor", "inductor", "lens", "mirror",
            ]
            mentioned_shapes = [s for s in shape_terms if s in q_lower]
            for shape in mentioned_shapes:
                if shape not in d_lower:
                    return False, f"Missing shape term '{shape}'"

            if warnings:
                return True, "Partial coverage: " + "; ".join(warnings)

            # PASSED all checks
            return True, "Valid"

        def is_diagram_eligible(question_text: str) -> bool:
            """
            Decide if a question genuinely needs a diagram.
            Avoid algebraic vector/component questions that are solved symbolically.
            """
            import re

            q = question_text.lower()

            # Exclude pure vector-component algebra (i, j, k) and coplanar/parameter questions
            vector_component_patterns = [
                r"\bi\s*[+\-]\s*\bj\b",
                r"\bj\s*[+\-]\s*\bk\b",
                r"\bposition vectors\b",
                r"\bvector\s+[abc]\b\s*=\s*",
                r"\bai\s*[+\-]\s*bj\s*[+\-]\s*ck\b",
                r"\bcoplanar\b",
                r"\blies in the plane\b",
                r"\bfind\s+x\b",
            ]
            if any(re.search(p, q) for p in vector_component_patterns):
                return False

            # Include explicit geometry or physical setups
            include_patterns = [
                r"\btriangle\b", r"\bcircle\b", r"\brectangle\b", r"\bsquare\b",
                r"\bparabola\b", r"\bellipse\b", r"\bhyperbola\b", r"\bgraph\b",
                r"\bdiagram\b", r"\bfigure\b", r"\bsketch\b",
                r"\bcircuit\b", r"\bresistor\b", r"\bcapacitor\b", r"\binductor\b",
                r"\blens\b", r"\bmirror\b", r"\binclined plane\b", r"\bblock\b",
                r"\bforce\b", r"\bangle\b", r"\bcoordinate\b",
                r"\bpoint\b", r"\bline\b", r"\bsegment\b", r"\bchord\b",
                r"\btangent\b", r"\bsector\b", r"\bcone\b", r"\bcylinder\b",
                r"\bsphere\b", r"\bprism\b",
            ]
            return any(re.search(p, q) for p in include_patterns)

        def _is_diagram_marked(question_obj: Dict[str, Any]) -> bool:
            question_text = (question_obj.get("question_text") or "").strip()
            if not question_text or not is_diagram_eligible(question_text):
                return False
            if not bool(question_obj.get("needs_diagram")):
                return False
            diagram_desc = (question_obj.get("diagram_description") or "").strip()
            return len(diagram_desc) >= 20

        async def _generate_diagram_focused_questions(missing_count: int) -> List[Dict[str, Any]]:
            """Generate additional diagram-first questions when the minimum quota is not met."""
            if missing_count <= 0:
                return []

            generated: List[Dict[str, Any]] = []
            attempts = 0
            max_attempts = 3

            while len(generated) < missing_count and attempts < max_attempts:
                attempts += 1
                remaining = missing_count - len(generated)
                # Ask for a slightly larger batch to absorb validation and duplicate drops.
                batch_size = max(remaining + 2, remaining * 2)

                focused_prompt = build_system_prompt(
                    batch_size,
                    min_diagram_count=batch_size,
                    require_diagram_only=True,
                )
                focused_messages = [
                    {"role": "system", "content": focused_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Generate {batch_size} NEW diagram-first questions on {request.topic} for {subject}. "
                            "Do not repeat previous questions."
                        ),
                    },
                ]

                try:
                    focused_response = await kimi._call_api(focused_messages, max_tokens=7000)
                    focused_questions = _parse_questions_response(focused_response)
                    focused_questions = _local_sanity_filter(focused_questions)
                    focused_valid, _ = await _validate_questions_batch(focused_questions)
                except Exception as focused_err:
                    logger.warning(f"[DIAGRAM-MIN] Focused diagram regeneration failed: {focused_err}")
                    continue

                for question_obj in focused_valid:
                    key = _normalize_question_text(question_obj.get("question_text", ""))
                    if not key or key in seen_questions:
                        continue

                    question_obj["needs_diagram"] = True
                    if not _is_diagram_marked(question_obj):
                        continue

                    seen_questions.add(key)
                    generated.append(question_obj)
                    if len(generated) >= missing_count:
                        break

            return generated

        diagram_min_target = min(len(questions_data), min_diagram_questions_target)
        current_diagram_marked = sum(1 for q in questions_data if _is_diagram_marked(q))
        diagram_shortfall = max(0, diagram_min_target - current_diagram_marked)
        if diagram_shortfall > 0:
            logger.info(
                f"[DIAGRAM-MIN] Detected shortfall={diagram_shortfall} "
                f"(current={current_diagram_marked}, target={diagram_min_target})"
            )

            focused_questions = await _generate_diagram_focused_questions(diagram_shortfall)
            replaceable_indices = [
                i for i, q in enumerate(questions_data)
                if not _is_diagram_marked(q)
            ]

            replaced = 0
            for idx, replacement in zip(replaceable_indices, focused_questions):
                questions_data[idx] = replacement
                replaced += 1
                if replaced >= diagram_shortfall:
                    break

            current_diagram_marked = sum(1 for q in questions_data if _is_diagram_marked(q))
            diagram_shortfall = max(0, diagram_min_target - current_diagram_marked)
            if diagram_shortfall > 0:
                logger.warning(
                    f"[DIAGRAM-MIN] Could not fully satisfy minimum diagram quota. "
                    f"remaining_shortfall={diagram_shortfall}, current={current_diagram_marked}, target={diagram_min_target}"
                )
            else:
                logger.info(
                    f"[DIAGRAM-MIN] Minimum diagram quota satisfied after focused regeneration. "
                    f"current={current_diagram_marked}, target={diagram_min_target}"
                )

        # Limit diagrams to a configurable ratio of total questions
        try:
            diagram_ratio = float(os.getenv("DIAGRAM_MAX_RATIO", "0.3"))
        except ValueError:
            diagram_ratio = 0.3
        diagram_ratio = max(0.0, min(1.0, diagram_ratio))
        total_questions = len(questions_data)
        max_diagrams = int(math.floor(total_questions * diagram_ratio))
        if diagram_ratio > 0 and total_questions > 0 and max_diagrams == 0:
            max_diagrams = 1
        if diagram_min_target > max_diagrams:
            logger.info(
                f"[DIAGRAM-RATIO] Raising max_diagrams from {max_diagrams} to {diagram_min_target} "
                f"to satisfy minimum ratio target"
            )
            max_diagrams = diagram_min_target

        # Pick the best candidates first (valid hints + diagrammatic text)
        diagram_candidates = []
        for idx, q in enumerate(questions_data):
            q_text = q.get("question_text", "")
            desc = q.get("diagram_description", "")
            valid_desc = bool(desc) and validate_diagram_relevance(q_text, desc)[0]
            likely = is_diagram_eligible(q_text)
            needs_flag = q.get("needs_diagram", False)
            score = (2 if valid_desc else 0) + (1 if likely else 0) + (1 if needs_flag else 0)
            if score == 0:
                continue
            diagram_candidates.append((score, idx))

        diagram_candidates.sort(key=lambda x: (-x[0], x[1]))
        candidate_indices = [idx for _, idx in diagram_candidates]

        logger.info(
            f"[DIAGRAM-RATIO] total={total_questions}, ratio={diagram_ratio}, "
            f"max_diagrams={max_diagrams}, candidates={len(diagram_candidates)}"
        )
        accepted_diagrams = 0

        for idx, q_data in enumerate(questions_data):
            question_id = str(uuid.uuid4())[:8]

            # Check if question needs a diagram
            question_text = q_data.get("question_text", "")
            needs_diagram = q_data.get("needs_diagram", False)
            diagram_description = q_data.get("diagram_description", "")
            diagram_hints = diagram_description.strip() if diagram_description else None
            diagram_instructions = diagram_description or None
            diagram_url = None
            diagram_base64 = None
            diagram_image_id = None

            # IMPORTANT: Only generate diagrams when Kimi explicitly provides needs_diagram=true
            # AND a meaningful diagram_description that MATCHES the question content.
            # Many vector algebra/calculation questions don't need diagrams at all.

            # Validate that diagram_description is actually useful AND matches question
            if needs_diagram and diagram_description:
                is_valid, reason = validate_diagram_relevance(question_text, diagram_description)

                if not is_valid:
                    # Description doesn't match question, ignore hints and fall back to question text
                    print(f"[DEBUG] INVALID diagram hints for {question_id}: {reason}")
                    print(f"[DEBUG]   Question: {question_text[:100]}...")
                    print(f"[DEBUG]   Diagram desc: {diagram_description[:100]}...")
                    logger.info(f"[DIAGRAM-HINTS] Ignoring hints: {reason} for {question_id}")
                    diagram_hints = None
                    diagram_instructions = None
                else:
                    print(f"[DEBUG] VALID diagram hints for {question_id}: {reason}")
                    logger.info(f"[DIAGRAM-VALID] Accepted hints for {question_id}")

            # Enforce diagram ratio cap (attempt only while below max)
            allow_by_ratio = idx in candidate_indices and accepted_diagrams < max_diagrams
            if not allow_by_ratio:
                needs_diagram = False
                diagram_hints = None
                diagram_instructions = None
                logger.info(f"[DIAGRAM-SKIP] Skipping {question_id} due to ratio cap")

            # If no usable hints and question does not look diagrammatic, skip
            if needs_diagram and not diagram_hints and not is_diagram_eligible(question_text):
                needs_diagram = False
                logger.info(f"[DIAGRAM-SKIP] Question text not diagrammatic for {question_id}")
                diagram_instructions = None

            # Debug logging for diagram generation
            logger.info(f"Question {question_id}: needs_diagram={needs_diagram}, has_hints={bool(diagram_hints)}, has_api_key={bool(image_client.api_key)}")

            # Generate diagram if needed
            print(f"[DEBUG] Q{idx+1} {question_id}: needs_diagram={needs_diagram}, hints_len={len(diagram_hints) if diagram_hints else 0}, has_key={bool(image_client.api_key)}")

            if needs_diagram and image_client.api_key:
                try:
                    import sys
                    print(f"[DEBUG] >>> GENERATING DIAGRAM (pipeline) for {question_id}", flush=True)
                    sys.stdout.flush()
                    logger.info(f"[DIAGRAM-GEN] Starting diagram pipeline for question {question_id}")
                    logger.info(f"[DIAGRAM-GEN] Model: {image_client.model}, API URL: {image_client.api_url}")
                    if diagram_hints:
                        logger.info(f"[DIAGRAM-GEN] Hints: {diagram_hints[:150]}...")

                    diagram_request = DiagramRequest(
                        question_id=question_id,
                        question_text=question_text,
                        subject=_parse_subject_type(subject),
                        diagram_hints=diagram_hints,
                        max_iterations=3,
                        force_regenerate=True,
                    )

                    diagram_result = await diagram_service.generate_diagram(diagram_request)

                    print(f"[DEBUG] pipeline.generate_diagram() returned for {question_id}, success={diagram_result.success}", flush=True)
                    sys.stdout.flush()

                    if diagram_result and diagram_result.image and diagram_result.image.base64_data:
                        print(f"[DEBUG] Diagram generated! base64 length: {len(diagram_result.image.base64_data)}", flush=True)
                        diagram_base64 = diagram_result.image.base64_data

                        if diagram_result.instructions_history:
                            diagram_instructions = diagram_result.instructions_history[-1].instructions

                        # Persist image to storage AND save metadata to MongoDB
                        is_b2c = current_user.get("user_type") == "b2c_admin"
                        image_ref = await _store_diagram_image(
                            db=db,
                            image_base64=diagram_result.image.base64_data,
                            document_id=paper_id,
                            current_user=current_user,
                            is_b2c=is_b2c,
                        )
                        if image_ref:
                            diagram_url = image_ref.get("url")
                            diagram_image_id = image_ref.get("id")
                            logger.info(f"Diagram persisted: id={diagram_image_id}, url={diagram_url}")
                        elif is_s3_enabled():
                            # Fallback: direct S3 upload if _store_diagram_image failed
                            import base64 as b64mod
                            image_bytes = b64mod.b64decode(diagram_result.image.base64_data)
                            local_path = f"uploads/diagrams/{question_id}_{diagram_result.image.filename}"
                            success, storage_path = await upload_file(
                                image_bytes,
                                local_path,
                                content_type="image/png"
                            )
                            if success:
                                diagram_url = get_public_url(storage_path)
                                logger.warning(f"Diagram uploaded to S3 (no DB record): {diagram_url[:100]}...")
                        else:
                            diagram_url = f"/static/{diagram_result.image.path}"

                        logger.info(f"✅ Diagram generated for question {question_id}")
                        accepted_diagrams += 1
                    else:
                        issues = []
                        if diagram_result and diagram_result.review_history:
                            latest = diagram_result.review_history[-1]
                            issues = [i.code.value for i in latest.issues]
                        logger.warning(
                            f"Diagram generation returned no data for question {question_id}. "
                            f"issues={issues}"
                        )

                except Exception as diagram_error:
                    print(f"[DEBUG] !!! DIAGRAM ERROR for {question_id}: {diagram_error}", flush=True)
                    logger.error(f"[DIAGRAM-ERR] Failed to generate diagram for question {question_id}: {diagram_error}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}", flush=True)
                    logger.error(f"[DIAGRAM-ERR] Traceback: {traceback.format_exc()}")
                    # Continue without diagram - don't fail the entire question
            else:
                # Log why we're not generating a diagram
                skip_reasons = []
                if not needs_diagram:
                    skip_reasons.append("needs_diagram=False")
                if not image_client.api_key:
                    skip_reasons.append("no API key")
                print(f"[DEBUG] SKIPPING diagram for {question_id}: {', '.join(skip_reasons)}")
                logger.info(f"[DIAGRAM-SKIP] Skipping diagram for {question_id}: {', '.join(skip_reasons)}")
            
            generated_questions_response.append({
                "id": question_id,
                "text": q_data.get("question_text", ""),
                "type": request.question_type,
                "marks": 4,
                "options": q_data.get("options", []),
                "correct_answer": q_data.get("correct_answer", ""),
                "explanation": q_data.get("explanation"),
                "difficulty": request.difficulty,
                "topic": request.topic,
                "has_diagram": needs_diagram and diagram_base64 is not None,
                "diagram_instructions": diagram_instructions,
                "diagram_image_id": diagram_image_id,
                "diagram_url": diagram_url,
                "diagram_image_url": diagram_url,
                "diagram_base64": diagram_base64,
                "source": "ai_generated",
            })

        return {
            "success": True,
            "questions_generated": len(generated_questions_response),
            "questions": generated_questions_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/questions/{question_id}/diagram")
async def generate_question_diagram(
    paper_id: str,
    question_id: str,
    request: GenerateDiagramRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Generate a diagram for a specific question."""
    try:
        paper = await _get_paper(db, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Find the question
        question_found = None
        section_idx = None
        question_idx = None
        
        for s_idx, section in enumerate(paper["sections"]):
            for q_idx, question in enumerate(section["questions"]):
                if question["id"] == question_id:
                    question_found = question
                    section_idx = s_idx
                    question_idx = q_idx
                    break
            if question_found:
                break
        
        if not question_found:
            raise HTTPException(status_code=404, detail="Question not found")

        # Generate diagram using the pipeline
        diagram_service = DiagramPipelineService(db=db)

        # Use request data or fall back to stored data
        question_text = request.question_text or question_found.get("question_text", "")
        subject = request.subject or paper.get("subject", "")
        diagram_hints = request.diagram_hints or request.additional_instructions or question_found.get("diagram_instructions")

        from services.diagram_pipeline.models import DiagramGenerationRequest as DiagramRequest

        # Determine subject type
        subject_type = _parse_subject_type(subject)

        diagram_request = DiagramRequest(
            question_id=question_id,
            question_text=question_text,
            subject=subject_type,
            diagram_hints=diagram_hints,
            max_iterations=3,
            force_regenerate=True,
        )
        
        result = await diagram_service.generate_diagram(diagram_request)
        
        if result.success and result.image:
            # Persist the generated image and store a stable image URL.
            stable_diagram_url = None
            stable_diagram_id = None

            if result.image.base64_data:
                is_b2c = current_user.get("user_type") == "b2c_admin"
                image_ref = await _store_diagram_image(
                    db=db,
                    image_base64=result.image.base64_data,
                    document_id=paper.get("document_id") or paper.get("id") or paper_id,
                    current_user=current_user,
                    is_b2c=is_b2c
                )
                if image_ref:
                    stable_diagram_id = image_ref.get("id")
                    stable_diagram_url = image_ref.get("url")

            if not stable_diagram_url and result.image.path:
                path = str(result.image.path)
                stable_diagram_url = path if path.startswith("/") else f"/static/{path}"

            paper["sections"][section_idx]["questions"][question_idx]["has_diagram"] = True
            if stable_diagram_id:
                paper["sections"][section_idx]["questions"][question_idx]["diagram_image_id"] = stable_diagram_id
            paper["sections"][section_idx]["questions"][question_idx]["diagram_url"] = stable_diagram_url
            paper["sections"][section_idx]["questions"][question_idx]["diagram_image_url"] = stable_diagram_url
            # Keep inline base64 in response for immediate rendering, but persist stable URL in DB.
            if stable_diagram_id:
                paper["sections"][section_idx]["questions"][question_idx].pop("diagram_base64", None)
            else:
                paper["sections"][section_idx]["questions"][question_idx]["diagram_base64"] = result.image.base64_data
            
            if result.instructions_history:
                paper["sections"][section_idx]["questions"][question_idx]["diagram_instructions"] = result.instructions_history[-1].instructions
            
            await _save_paper(db, paper)
            
            return {
                "success": True,
                "diagram_url": stable_diagram_url,
                "diagram_image_id": stable_diagram_id,
                "diagram_base64": result.image.base64_data,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.error_message or "Failed to generate diagram"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-imagen")
async def test_imagen_api():
    """Test endpoint to verify Kimi+TikZ diagram rendering is working."""
    from services.diagram_pipeline.tikz_client import TikzImageClient

    client = TikzImageClient()
    
    # Check configuration
    result = {
        "has_api_key": bool(client.api_key),
        "api_key_prefix": client.api_key[:10] + "..." if client.api_key else None,
        "model": client.model,
        "renderer": getattr(client, "pdf_renderer", None),
        "latex_engine": getattr(client, "latex_engine", None),
        "images_dir": str(client.images_dir),
    }
    
    if not client.api_key:
        result["error"] = "KIMI_API_KEY not configured"
        return result
    
    # Try to generate a simple test diagram
    try:
        test_instructions = "Draw a simple black and white circle with label 'Test'"
        diagram = await client.generate_diagram(
            instructions=test_instructions,
            question_id="test_" + str(uuid.uuid4())[:4],
            instructions_version=1,
            aspect_ratio="1:1",
            resolution="1K",
        )
        
        result["success"] = True
        result["diagram_id"] = diagram.id
        result["diagram_path"] = diagram.path
        result["diagram_size"] = diagram.size_bytes
        result["has_base64"] = bool(diagram.base64_data)
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        logger.error(f"TikZ rendering test failed: {e}")
    
    return result


@router.get("/papers/{paper_id}/preview")
async def preview_paper(
    paper_id: str,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Get paper preview data formatted for display."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        paper.pop("_id", None)
        paper["total_questions"] = _count_questions(paper)
        paper["total_marks"] = _calculate_total_marks(paper)
        
        # Number questions sequentially across sections
        question_number = 1
        for section in paper["sections"]:
            for question in section["questions"]:
                question["question_number"] = question_number
                question_number += 1
        
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/sections")
async def add_section(
    paper_id: str,
    title: str,
    instructions: str = "",
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Add a new section to a paper."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        section_id = str(uuid.uuid4())[:8]
        section = {
            "id": section_id,
            "title": title,
            "instructions": instructions,
            "questions": []
        }
        
        paper["sections"].append(section)
        await _save_paper(db, paper)
        
        return {"success": True, "section_id": section_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding section: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/papers/{paper_id}/questions/{question_id}")
async def delete_question(
    paper_id: str,
    question_id: str,
    db: DatabaseManager = Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Delete a question from a paper."""
    try:
        paper = await _get_paper(db, paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Find and remove the question
        question_found = False
        for section in paper["sections"]:
            for i, question in enumerate(section["questions"]):
                if question["id"] == question_id:
                    section["questions"].pop(i)
                    question_found = True
                    break
            if question_found:
                break
        
        if not question_found:
            raise HTTPException(status_code=404, detail="Question not found")
        
        await _save_paper(db, paper)
        
        return {"success": True, "message": "Question deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
