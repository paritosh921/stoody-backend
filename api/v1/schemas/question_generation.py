"""
Question Generation API Schemas

Pydantic request/response models for question generation endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from services.question_generation import (
    QuestionType,
    BloomLevel,
    QuestionTypeConfig,
    DifficultyDistribution,
    QuestionGenerationConfig,
    PaperConfig,
)


# ============================================================================
# Request Models
# ============================================================================

class QuestionTypeConfigRequest(BaseModel):
    """Configuration for a specific question type."""
    count: int = Field(ge=0, le=50, description="Number of questions of this type")
    marks_per_question: int = Field(ge=1, le=20, default=1, description="Marks for each question")


class DifficultyDistributionRequest(BaseModel):
    """Distribution of difficulty levels."""
    easy: int = Field(default=30, ge=0, le=100)
    medium: int = Field(default=50, ge=0, le=100)
    hard: int = Field(default=20, ge=0, le=100)


class GenerationConfigRequest(BaseModel):
    """Configuration for question generation."""
    
    mcq: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="MCQ configuration"
    )
    short_answer: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="Short answer configuration"
    )
    long_answer: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="Long answer configuration"
    )
    numerical: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="Numerical problems configuration"
    )
    true_false: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="True/False configuration"
    )
    fill_blanks: Optional[QuestionTypeConfigRequest] = Field(
        default=None, description="Fill in blanks configuration"
    )
    
    difficulty_distribution: DifficultyDistributionRequest = Field(
        default_factory=DifficultyDistributionRequest
    )
    
    bloom_levels: List[str] = Field(
        default=["remember", "understand", "apply", "analyze"],
        description="Bloom's taxonomy levels to include"
    )
    
    include_diagrams: bool = Field(default=True, description="Generate diagrams where appropriate")
    include_solutions: bool = Field(default=True, description="Include detailed solutions")
    include_marking_scheme: bool = Field(default=True, description="Include step-by-step marking")
    
    def to_internal_config(self) -> QuestionGenerationConfig:
        """Convert to internal QuestionGenerationConfig."""
        question_types = {}
        
        if self.mcq:
            question_types[QuestionType.MCQ] = QuestionTypeConfig(
                count=self.mcq.count,
                marks_per_question=self.mcq.marks_per_question
            )
        
        if self.short_answer:
            question_types[QuestionType.SHORT_ANSWER] = QuestionTypeConfig(
                count=self.short_answer.count,
                marks_per_question=self.short_answer.marks_per_question
            )
        
        if self.long_answer:
            question_types[QuestionType.LONG_ANSWER] = QuestionTypeConfig(
                count=self.long_answer.count,
                marks_per_question=self.long_answer.marks_per_question
            )
        
        if self.numerical:
            question_types[QuestionType.NUMERICAL] = QuestionTypeConfig(
                count=self.numerical.count,
                marks_per_question=self.numerical.marks_per_question
            )
        
        if self.true_false:
            question_types[QuestionType.TRUE_FALSE] = QuestionTypeConfig(
                count=self.true_false.count,
                marks_per_question=self.true_false.marks_per_question
            )
        
        if self.fill_blanks:
            question_types[QuestionType.FILL_IN_BLANKS] = QuestionTypeConfig(
                count=self.fill_blanks.count,
                marks_per_question=self.fill_blanks.marks_per_question
            )
        
        # Use defaults if no question types specified
        if not question_types:
            question_types = {
                QuestionType.MCQ: QuestionTypeConfig(count=10, marks_per_question=1),
                QuestionType.SHORT_ANSWER: QuestionTypeConfig(count=5, marks_per_question=2),
            }
        
        bloom_level_enums = [BloomLevel(b) for b in self.bloom_levels if b in [e.value for e in BloomLevel]]
        
        return QuestionGenerationConfig(
            question_types=question_types,
            difficulty_distribution=DifficultyDistribution(
                easy=self.difficulty_distribution.easy,
                medium=self.difficulty_distribution.medium,
                hard=self.difficulty_distribution.hard,
            ),
            bloom_levels=bloom_level_enums or [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND],
            include_diagrams=self.include_diagrams,
            include_solutions=self.include_solutions,
            include_marking_scheme=self.include_marking_scheme,
        )


class PaperConfigRequest(BaseModel):
    """Paper layout configuration."""
    title: str = Field(..., description="Paper title")
    subject: str = Field(..., description="Subject name")
    grade: str = Field(..., description="Class/Grade level")
    duration_minutes: int = Field(default=90, ge=15, le=300)
    
    school_name: Optional[str] = None
    exam_name: Optional[str] = None
    exam_date: Optional[str] = None
    chapter: Optional[str] = None
    topics: Optional[List[str]] = None
    
    general_instructions: Optional[List[str]] = Field(default=None)
    
    def to_internal_config(self) -> PaperConfig:
        """Convert to internal PaperConfig."""
        return PaperConfig(
            title=self.title,
            subject=self.subject,
            grade=self.grade,
            duration_minutes=self.duration_minutes,
            school_name=self.school_name,
            exam_name=self.exam_name,
            exam_date=self.exam_date,
            chapter=self.chapter,
            topics=self.topics,
            general_instructions=self.general_instructions,
        )


class GenerateFromNotesRequest(BaseModel):
    """Request to generate questions from notes content."""
    content: str = Field(..., min_length=100, description="The extracted text content from notes")
    generation_config: GenerationConfigRequest
    paper_config: PaperConfigRequest
    tenant_id: str = Field(..., description="Tenant identifier")
    teacher_id: str = Field(..., description="Teacher identifier")
    store_embeddings: bool = Field(default=True, description="Store embeddings for future use")
    source_file_id: Optional[str] = Field(default=None, description="Source file ID for tracking")


class GenerateFromTopicRequest(BaseModel):
    """Request to generate questions from topic search."""
    topic: str = Field(..., min_length=3, description="Topic to generate questions about")
    subject: str = Field(..., description="Subject name")
    grade: str = Field(..., description="Class/Grade level")
    generation_config: GenerationConfigRequest
    paper_config: PaperConfigRequest
    tenant_id: str = Field(..., description="Tenant identifier")
    teacher_id: str = Field(..., description="Teacher identifier")
    chapter: Optional[str] = Field(default=None, description="Optional chapter filter")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of context chunks to retrieve")
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")


class QuestionPreviewRequest(BaseModel):
    """Request to preview questions without generating PDF."""
    content: Optional[str] = Field(default=None, description="Content for notes-based generation")
    topic: Optional[str] = Field(default=None, description="Topic for topic-based generation")
    subject: str = Field(..., description="Subject name")
    grade: str = Field(..., description="Class/Grade level")
    generation_config: GenerationConfigRequest
    tenant_id: str = Field(..., description="Tenant identifier")
    chapter: Optional[str] = None
    include_answers: bool = Field(default=False, description="Include answers in preview")


# ============================================================================
# Response Models
# ============================================================================

class GeneratedQuestionResponse(BaseModel):
    """Response model for a single question."""
    question_id: str
    question_text: str
    question_type: str
    marks: int
    difficulty: str
    options: Optional[List[Dict[str, Any]]] = None
    correct_answer: Optional[str] = None
    solution: Optional[str] = None
    has_diagram: bool = False
    diagram_url: Optional[str] = None


class PaperSectionResponse(BaseModel):
    """Response model for a paper section."""
    name: str
    instructions: str
    question_count: int
    total_marks: int
    questions: List[GeneratedQuestionResponse]


class GeneratedPaperResponse(BaseModel):
    """Response model for a generated paper."""
    paper_id: str
    title: str
    subject: str
    grade: str
    duration_minutes: int
    total_questions: int
    total_marks: int
    status: str
    source_type: str
    sections: List[PaperSectionResponse]
    created_at: str
    completed_at: Optional[str] = None
    generation_stats: Optional[Dict[str, Any]] = None
    question_paper_url: Optional[str] = None
    answer_key_url: Optional[str] = None
    marking_scheme_url: Optional[str] = None
    error_message: Optional[str] = None


class QuestionPreviewResponse(BaseModel):
    """Response model for question preview."""
    questions: List[Dict[str, Any]]
    total_count: int
    total_marks: int


class HealthCheckResponse(BaseModel):
    """Health check response."""
    service: str
    initialized: bool
    dependencies: Dict[str, Any]
