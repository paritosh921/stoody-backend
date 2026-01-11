"""
Pydantic schemas for PDF processing endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MistralOCRImage(BaseModel):
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: Optional[str] = None


class MistralOCRPage(BaseModel):
    index: int
    markdown: str
    images: List[MistralOCRImage]
    dimensions: Dict[str, Any]


class ExtractedQuestion(BaseModel):
    id: str
    text: str
    options: List[str] = []
    correct_answer: Optional[str] = None
    images: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    points: Optional[float] = 4.0  # Default 4 points for Test Series (JEE style)
    penalty: Optional[float] = 1.0  # Default 1 penalty (JEE style)


class PDFProcessingResult(BaseModel):
    job_id: str
    status: str  # 'processing', 'completed', 'error'
    progress: int
    extracted_questions: int = 0
    extracted_images: int = 0
    output_folder: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime
    pages: Optional[List[MistralOCRPage]] = None


class QuestionImage(BaseModel):
    id: str
    filename: str
    path: str
    description: str
    type: str
    base64_data: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class Question(BaseModel):
    id: str
    text: str
    subject: str
    difficulty: str
    extracted_at: datetime
    pdf_source: str
    images: List[QuestionImage] = []
    options: List[str] = []
    correct_answer: Optional[str] = None
    metadata: Dict[str, Any] = {}
    points: Optional[float] = 4.0
    penalty: Optional[float] = 1.0


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    document_type: str
    subject: str
    difficulty: str
    course_plan: Optional[str] = None
    standard: Optional[str] = None
    section: Optional[str] = None  # Section A-F for filtering
    teacher_ids: Optional[List[str]] = None  # Array of teacher IDs for filtering
    file_path: str
    filename: str
    uploaded_by: str
    uploaded_at: datetime
    ocr_status: str
    ocr_job_id: Optional[str] = None
    extracted_questions_count: int = 0
    extracted_images_count: int = 0
    pages_count: int = 0  # Number of pages in the PDF (for Notes display)
    total_points: Optional[float] = None  # Total points for Test Series documents
    total_minutes: Optional[int] = None  # Total minutes for Test Series documents
    file_exists: bool = True  # Whether the physical file exists on disk
    is_active: bool = True  # Whether the document is enabled for students


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int
    page: int
    limit: int
