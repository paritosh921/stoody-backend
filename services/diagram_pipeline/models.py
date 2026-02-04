"""
Models for the Diagram Pipeline Service

These models define the data structures for:
- Diagram generation requests/responses
- LLM instructions
- Review feedback
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SubjectType(str, Enum):
    """Subject types for diagram generation."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"


class IssueCode(str, Enum):
    """Standard issue codes for diagram reviews."""
    LABEL_OVERLAP = "LABEL_OVERLAP"
    MISSING_LABEL = "MISSING_LABEL"
    WRONG_ANGLE = "WRONG_ANGLE"
    LOW_CLARITY = "LOW_CLARITY"
    WRONG_CONFIGURATION = "WRONG_CONFIGURATION"
    TOO_CLUTTERED = "TOO_CLUTTERED"
    MISSING_ARROW = "MISSING_ARROW"
    WRONG_PROPORTION = "WRONG_PROPORTION"
    INCOMPLETE_DIAGRAM = "INCOMPLETE_DIAGRAM"
    STYLE_ISSUE = "STYLE_ISSUE"
    OTHER = "OTHER"


class ReviewIssue(BaseModel):
    """A single issue identified in a diagram review."""
    code: IssueCode
    details: str


class DiagramReview(BaseModel):
    """
    LLM2's review of a generated diagram.
    
    The review determines if the diagram is acceptable for exam use
    and provides specific issues with suggested improvements.
    """
    is_acceptable: bool
    issues: List[ReviewIssue] = Field(default_factory=list)
    suggested_instruction_update: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    review_version: int = 1


class DiagramInstructions(BaseModel):
    """
    LLM1's instructions for generating a diagram.
    
    These instructions are sent to Nano Banan Pro for image generation.
    """
    instructions: str
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    subject: Optional[SubjectType] = None
    
    # Track if this was refined from a previous version
    refined_from_version: Optional[int] = None
    refinement_reason: Optional[str] = None


class DiagramImage(BaseModel):
    """
    Metadata for a generated diagram image.
    """
    id: str
    path: str
    filename: str
    format: str = "png"
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Link to the instructions that generated this image
    instructions_version: int = 1
    
    # Base64 data (optional, for inline display)
    base64_data: Optional[str] = None


class DiagramGenerationRequest(BaseModel):
    """
    Request to generate a diagram for a question.
    """
    question_id: str
    question_text: str
    subject: Optional[SubjectType] = None
    
    # Optional hints for the LLM
    diagram_hints: Optional[str] = None
    
    # Maximum refinement iterations (default: 3)
    max_iterations: int = Field(default=3, ge=1, le=5)
    
    # Force regeneration even if a diagram exists
    force_regenerate: bool = False


class DiagramGenerationResponse(BaseModel):
    """
    Response from the diagram generation pipeline.
    """
    question_id: str
    success: bool
    
    # Final image (if successful)
    image: Optional[DiagramImage] = None
    
    # All instruction versions (for history)
    instructions_history: List[DiagramInstructions] = Field(default_factory=list)
    
    # All reviews (for history)
    review_history: List[DiagramReview] = Field(default_factory=list)
    
    # Pipeline metadata
    iterations_used: int = 0
    total_time_seconds: float = 0.0
    
    # Error info (if failed)
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class QuestionDiagramRecord(BaseModel):
    """
    Database record for a question's diagram data.
    
    This is stored alongside the question in MongoDB.
    """
    question_id: str
    
    # Current active image
    current_image: Optional[DiagramImage] = None
    
    # Instruction history (all versions)
    instructions_history: List[DiagramInstructions] = Field(default_factory=list)
    
    # Review history (all reviews)
    review_history: List[DiagramReview] = Field(default_factory=list)
    
    # Generation metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_count: int = 0
    
    # Status
    is_finalized: bool = False
    needs_review: bool = False
    
    def get_current_instructions(self) -> Optional[DiagramInstructions]:
        """Get the most recent instructions."""
        if self.instructions_history:
            return max(self.instructions_history, key=lambda x: x.version)
        return None
    
    def get_latest_review(self) -> Optional[DiagramReview]:
        """Get the most recent review."""
        if self.review_history:
            return max(self.review_history, key=lambda x: x.review_version)
        return None
