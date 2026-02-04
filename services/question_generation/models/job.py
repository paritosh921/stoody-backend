"""
Job models for async task processing.

Implements the job system to avoid CloudFront's 30-second timeout.
All heavy tasks run as background jobs with status polling.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


class JobType(str, Enum):
    """Types of async jobs."""
    INGEST_PDF = "ingest_pdf"
    GENERATE_PAPER = "generate_paper"
    RENDER_DIAGRAMS = "render_diagrams"
    EXPORT_PAPER = "export_paper"


class JobStatus(str, Enum):
    """Status of an async job."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobProgress(BaseModel):
    """Progress tracking for a job."""
    current_step: int = 0
    total_steps: int = 1
    step_name: str = ""
    details: Optional[str] = None
    
    @property
    def percentage(self) -> int:
        """Calculate progress percentage (0-100)."""
        if self.total_steps <= 0:
            return 0
        return min(100, int((self.current_step / self.total_steps) * 100))


class Job(BaseModel):
    """
    Async job model for tracking background tasks.
    
    All heavy operations (PDF ingestion, paper generation) run as jobs
    to avoid HTTP timeout issues. Frontend polls job status.
    """
    
    # Identification
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType
    
    # Multi-tenancy
    institution_id: str
    user_id: str  # teacher_id or admin_id
    
    # Status tracking
    status: JobStatus = JobStatus.QUEUED
    progress: JobProgress = Field(default_factory=JobProgress)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None  # Contains pdf_id, paper_id, etc.
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Job-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def update_progress(
        self,
        current_step: int,
        total_steps: Optional[int] = None,
        step_name: str = "",
        details: Optional[str] = None,
    ) -> None:
        """Update job progress."""
        self.progress.current_step = current_step
        if total_steps is not None:
            self.progress.total_steps = total_steps
        self.progress.step_name = step_name
        self.progress.details = details
    
    def succeed(self, result: Dict[str, Any]) -> None:
        """Mark job as succeeded."""
        self.status = JobStatus.SUCCEEDED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress.current_step = self.progress.total_steps
    
    def fail(self, error: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        self.error_details = error_details
    
    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "job_id": self.job_id,
            "type": self.job_type.value,
            "status": self.status.value,
            "progress": self.progress.percentage,
            "progress_details": {
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "step_name": self.progress.step_name,
                "details": self.progress.details,
            },
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document format."""
        return {
            "_id": self.job_id,
            "job_type": self.job_type.value,
            "institution_id": self.institution_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "progress": {
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "step_name": self.progress.step_name,
                "details": self.progress.details,
            },
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "error_details": self.error_details,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_mongo_dict(cls, doc: Dict[str, Any]) -> "Job":
        """Create Job from MongoDB document."""
        progress_data = doc.get("progress", {})
        return cls(
            job_id=doc["_id"],
            job_type=JobType(doc["job_type"]),
            institution_id=doc.get("institution_id", ""),
            user_id=doc.get("user_id", ""),
            status=JobStatus(doc.get("status", "queued")),
            progress=JobProgress(
                current_step=progress_data.get("current_step", 0),
                total_steps=progress_data.get("total_steps", 1),
                step_name=progress_data.get("step_name", ""),
                details=progress_data.get("details"),
            ),
            created_at=doc.get("created_at", datetime.utcnow()),
            started_at=doc.get("started_at"),
            completed_at=doc.get("completed_at"),
            result=doc.get("result"),
            error=doc.get("error"),
            error_details=doc.get("error_details"),
            metadata=doc.get("metadata", {}),
            retry_count=doc.get("retry_count", 0),
            max_retries=doc.get("max_retries", 3),
        )


# ============================================================================
# PDF Ingestion Job Models
# ============================================================================

class PDFIngestionMetadata(BaseModel):
    """Metadata for PDF ingestion job."""
    pdf_id: str
    filename: str
    file_size_bytes: int
    subject: str
    class_grade: str
    chapter: Optional[str] = None
    topics: Optional[List[str]] = None
    
    # Processing options
    use_ocr: bool = False
    ocr_language: str = "en"


class PDFIngestionResult(BaseModel):
    """Result of a completed PDF ingestion job."""
    pdf_id: str
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    total_pages: int
    total_tokens_used: int
    processing_time_ms: int
    ocr_used: bool = False
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# Paper Generation Job Models
# ============================================================================

class PaperGenerationMetadata(BaseModel):
    """Metadata for paper generation job."""
    paper_id: str
    pdf_ids: List[str]  # Source PDFs for RAG
    subject: str
    class_grade: str
    
    # Blueprint configuration
    blueprint: Dict[str, Any]
    include_diagrams: bool = True
    exam_style: Optional[str] = None  # JEE, NEET, CBSE, etc.


class PaperGenerationResult(BaseModel):
    """Result of a completed paper generation job."""
    paper_id: str
    total_questions: int
    questions_approved: int
    questions_needing_review: int
    total_marks: int
    generation_time_ms: int
    total_llm_calls: int
    total_tokens_used: int
    
    # Question breakdown
    questions_by_type: Dict[str, int]
    questions_by_difficulty: Dict[str, int]
    
    # Download URLs
    paper_json_url: Optional[str] = None
    paper_pdf_url: Optional[str] = None


# ============================================================================
# Question Draft and Review Models (for two-LLM loop)
# ============================================================================

class QuestionDraft(BaseModel):
    """
    Output format from LLM1 (Generator).
    Must follow strict JSON schema.
    
    Enhanced for JEE/NEET-style diagram-based questions with:
    - Explicit diagram tool specification (matplotlib, schemdraw, rdkit)
    - Validation checklist for quality assurance
    """
    question_text: str
    question_type: str  # mcq, short, long, numerical
    options: Optional[List[Dict[str, Any]]] = None  # For MCQ
    correct_answer: str
    explanation: str
    marks: int
    difficulty: str  # easy, medium, hard
    source_chunk_ids: Optional[List[str]] = Field(default_factory=list)
    
    # Diagram specification (enhanced)
    diagram_required: bool = False
    diagram_tool: Optional[str] = None  # matplotlib, schemdraw, rdkit, none
    diagram_type: Optional[str] = None  # specific diagram type
    diagram_spec: Optional[Dict[str, Any]] = None
    diagram_rendering_notes: Optional[str] = None  # e.g., "print-friendly, clear labels"
    
    # Validation checklist (from reviewer)
    validation_checks: List[str] = Field(default_factory=list)
    # e.g., ["diagram matches question", "labels unambiguous", "units correct"]
    
    # Metadata for tracking
    topic: Optional[str] = None
    bloom_level: Optional[str] = None
    
    @validator('source_chunk_ids', pre=True, always=True)
    def ensure_source_chunk_ids_list(cls, v):
        """Convert None to empty list for source_chunk_ids."""
        if v is None:
            return []
        return v
    
    @validator('validation_checks', pre=True, always=True)
    def ensure_validation_checks_list(cls, v):
        """Convert None to empty list for validation_checks."""
        if v is None:
            return []
        return v



class ReviewIssue(BaseModel):
    """An issue identified by LLM2 (Reviewer)."""
    issue_type: str  # concept_error, ambiguity, format, diagram, difficulty_mismatch, hallucination
    message: str
    fix_instructions: str


class ReviewResult(BaseModel):
    """
    Output format from LLM2 (Reviewer).
    Must follow strict JSON schema.
    """
    approved: bool
    issues: Optional[List[ReviewIssue]] = Field(default_factory=list)
    required_changes: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None  # 0.0 to 1.0
    
    @validator('issues', pre=True, always=True)
    def ensure_issues_list(cls, v):
        """Convert None to empty list for issues."""
        if v is None:
            return []
        return v


class QuestionGenerationAttempt(BaseModel):
    """Tracks a single attempt to generate a question."""
    attempt_number: int
    draft: QuestionDraft
    review: Optional[ReviewResult] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GeneratedQuestionRecord(BaseModel):
    """
    Final record of a generated question with full history.
    Stored in the database for audit and improvement.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str
    section_index: int
    slot_index: int
    
    # Final question data
    final_draft: QuestionDraft
    
    # Generation history
    attempts: List[QuestionGenerationAttempt] = Field(default_factory=list)
    total_iterations: int = 0
    
    # Status
    approved: bool = False
    needs_manual_review: bool = False
    manual_review_reason: Optional[str] = None
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# ============================================================================
# Paper Generation State (Memory Management)
# ============================================================================

class PaperGenerationState(BaseModel):
    """
    Memory/state for paper generation.
    
    Stored in DB and passed to LLM1 to maintain context and avoid repetition.
    Kept small to control token cost.
    """
    paper_id: str
    
    # Blueprint summary (compact)
    blueprint_summary: str  # e.g., "Physics 10th: 10 MCQ, 5 short, 3 long. Focus: Motion"
    
    # Constraints
    constraints: List[str] = Field(default_factory=list)  # e.g., ["No calculus", "Age-appropriate"]
    
    # Generated questions summary
    generated_questions_summary: List[str] = Field(default_factory=list)
    # e.g., ["MCQ on Newton's 1st law (easy)", "Short on friction types (medium)"]
    
    # Topics already covered (to avoid repetition)
    topics_covered: List[str] = Field(default_factory=list)
    
    # Common mistakes from reviewer (to avoid repeating)
    banned_patterns: List[str] = Field(default_factory=list)
    # e.g., ["Avoid questions about calculus", "Don't use ambiguous wording like 'approximately'"]
    
    # Statistics
    questions_completed: int = 0
    questions_pending: int = 0
    questions_failed: int = 0
    
    # Timestamps
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_completed_question(self, question_summary: str, topics: List[str]) -> None:
        """Add a completed question to the state."""
        self.generated_questions_summary.append(question_summary)
        for topic in topics:
            if topic not in self.topics_covered:
                self.topics_covered.append(topic)
        self.questions_completed += 1
        self.questions_pending -= 1
        self.last_updated_at = datetime.utcnow()
    
    def add_banned_pattern(self, pattern: str) -> None:
        """Add a pattern to avoid based on reviewer feedback."""
        if pattern not in self.banned_patterns and len(self.banned_patterns) < 20:
            self.banned_patterns.append(pattern)
        self.last_updated_at = datetime.utcnow()
    
    def to_prompt_context(self) -> str:
        """Convert to a compact string for inclusion in prompts."""
        lines = [
            f"PAPER BLUEPRINT: {self.blueprint_summary}",
            "",
            "QUESTIONS ALREADY GENERATED:",
        ]
        
        if self.generated_questions_summary:
            for i, q in enumerate(self.generated_questions_summary[-10:], 1):  # Last 10
                lines.append(f"  {i}. {q}")
        else:
            lines.append("  (None yet)")
        
        lines.append("")
        lines.append("TOPICS ALREADY COVERED:")
        if self.topics_covered:
            lines.append(f"  {', '.join(self.topics_covered[-15:])}")  # Last 15
        else:
            lines.append("  (None yet)")
        
        if self.banned_patterns:
            lines.append("")
            lines.append("AVOID THESE PATTERNS (based on previous reviewer feedback):")
            for pattern in self.banned_patterns[-5:]:  # Last 5
                lines.append(f"  - {pattern}")
        
        if self.constraints:
            lines.append("")
            lines.append("CONSTRAINTS:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        
        return "\n".join(lines)
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document format."""
        return {
            "_id": self.paper_id,
            "blueprint_summary": self.blueprint_summary,
            "constraints": self.constraints,
            "generated_questions_summary": self.generated_questions_summary,
            "topics_covered": self.topics_covered,
            "banned_patterns": self.banned_patterns,
            "questions_completed": self.questions_completed,
            "questions_pending": self.questions_pending,
            "questions_failed": self.questions_failed,
            "last_updated_at": self.last_updated_at,
        }
    
    @classmethod
    def from_mongo_dict(cls, doc: Dict[str, Any]) -> "PaperGenerationState":
        """Create from MongoDB document."""
        return cls(
            paper_id=doc["_id"],
            blueprint_summary=doc.get("blueprint_summary", ""),
            constraints=doc.get("constraints", []),
            generated_questions_summary=doc.get("generated_questions_summary", []),
            topics_covered=doc.get("topics_covered", []),
            banned_patterns=doc.get("banned_patterns", []),
            questions_completed=doc.get("questions_completed", 0),
            questions_pending=doc.get("questions_pending", 0),
            questions_failed=doc.get("questions_failed", 0),
            last_updated_at=doc.get("last_updated_at", datetime.utcnow()),
        )


# ============================================================================
# Iteration Logging (for debugging and traceability)
# ============================================================================

class IterationLog(BaseModel):
    """
    Detailed log for each iteration in the LLM1-LLM2 loop.
    
    Stores full input/output for debugging when questions fail.
    """
    iteration_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # LLM1 (Generator) details
    generator_prompt_hash: str = ""  # SHA256 hash of prompt (for brevity)
    generator_prompt_length: int = 0
    generator_output_hash: str = ""  # For repetition detection
    generator_output_length: int = 0
    generator_output_preview: str = ""  # First 500 chars
    
    # Diagram spec snapshot
    diagram_spec_json: Optional[str] = None  # Full diagram_spec as JSON string
    diagram_validation_errors: List[str] = Field(default_factory=list)
    
    # LLM2 (Reviewer) details
    reviewer_prompt_length: int = 0
    reviewer_approved: bool = False
    reviewer_issues: List[str] = Field(default_factory=list)  # Issue types
    reviewer_feedback_summary: str = ""  # First 300 chars of feedback
    
    # Revision mode
    is_revision: bool = False
    is_hard_correction_mode: bool = False
    repetition_detected: bool = False
    
    # Timing
    generator_duration_ms: int = 0
    reviewer_duration_ms: int = 0


class QuestionGenerationDebugLog(BaseModel):
    """
    Complete debug log for a question generation attempt.
    
    Stored separately from the question record for forensic analysis.
    """
    question_id: str
    paper_id: str
    
    # Configuration
    question_type: str
    subject: str
    difficulty: str
    diagram_required: bool
    is_fallback_mode: bool = False
    
    # Iteration logs
    iterations: List[IterationLog] = Field(default_factory=list)
    
    # Final outcome
    final_approved: bool = False
    final_iteration_count: int = 0
    total_duration_ms: int = 0
    failure_reason: Optional[str] = None
    
    # Content hashes for repetition tracking
    output_hashes: List[str] = Field(default_factory=list)
    repetition_count: int = 0
