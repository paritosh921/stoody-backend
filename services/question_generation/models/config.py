"""
Configuration models for question generation.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    """Types of questions that can be generated."""
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    NUMERICAL = "numerical"
    MATCH_THE_FOLLOWING = "match"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANKS = "fill_blanks"


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BloomLevel(str, Enum):
    """Bloom's taxonomy levels for cognitive skills."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class QuestionTypeConfig(BaseModel):
    """Configuration for a specific question type."""
    count: int = Field(ge=0, le=50, description="Number of questions of this type")
    marks_per_question: int = Field(ge=1, le=20, description="Marks for each question")
    
    @property
    def total_marks(self) -> int:
        return self.count * self.marks_per_question


class DifficultyDistribution(BaseModel):
    """Distribution of difficulty levels (percentages should sum to 100)."""
    easy: int = Field(default=30, ge=0, le=100)
    medium: int = Field(default=50, ge=0, le=100)
    hard: int = Field(default=20, ge=0, le=100)
    
    @field_validator('hard')
    @classmethod
    def validate_distribution(cls, v, info):
        easy = info.data.get('easy', 30)
        medium = info.data.get('medium', 50)
        total = easy + medium + v
        if total != 100:
            raise ValueError(f"Difficulty distribution must sum to 100, got {total}")
        return v


class QuestionGenerationConfig(BaseModel):
    """Complete configuration for generating questions."""
    
    # Question type distribution
    question_types: Dict[QuestionType, QuestionTypeConfig] = Field(
        default_factory=lambda: {
            QuestionType.MCQ: QuestionTypeConfig(count=10, marks_per_question=1),
            QuestionType.SHORT_ANSWER: QuestionTypeConfig(count=5, marks_per_question=2),
            QuestionType.LONG_ANSWER: QuestionTypeConfig(count=3, marks_per_question=5),
            QuestionType.NUMERICAL: QuestionTypeConfig(count=2, marks_per_question=3),
        }
    )
    
    # Difficulty distribution
    difficulty_distribution: DifficultyDistribution = Field(
        default_factory=DifficultyDistribution
    )
    
    # Bloom's taxonomy levels to include
    bloom_levels: List[BloomLevel] = Field(
        default_factory=lambda: [
            BloomLevel.REMEMBER,
            BloomLevel.UNDERSTAND,
            BloomLevel.APPLY,
            BloomLevel.ANALYZE,
        ]
    )
    
    # Content options
    include_diagrams: bool = Field(default=True, description="Generate diagrams where appropriate")
    include_solutions: bool = Field(default=True, description="Include detailed solutions")
    include_marking_scheme: bool = Field(default=True, description="Include step-by-step marking")
    
    # Language and style
    language: str = Field(default="English", description="Language for questions")
    
    @property
    def total_questions(self) -> int:
        return sum(qt.count for qt in self.question_types.values())
    
    @property
    def total_marks(self) -> int:
        return sum(qt.total_marks for qt in self.question_types.values())
    
    def get_questions_per_difficulty(self) -> Dict[DifficultyLevel, int]:
        """Calculate number of questions per difficulty level."""
        total = self.total_questions
        return {
            DifficultyLevel.EASY: int(total * self.difficulty_distribution.easy / 100),
            DifficultyLevel.MEDIUM: int(total * self.difficulty_distribution.medium / 100),
            DifficultyLevel.HARD: int(total * self.difficulty_distribution.hard / 100),
        }


class PaperConfig(BaseModel):
    """Configuration for the paper layout and metadata."""
    
    title: str = Field(..., description="Paper title")
    subject: str = Field(..., description="Subject name")
    grade: str = Field(..., description="Class/Grade level")
    duration_minutes: int = Field(default=90, ge=15, le=300)
    
    # Optional metadata
    school_name: Optional[str] = None
    exam_name: Optional[str] = None
    exam_date: Optional[str] = None
    chapter: Optional[str] = None
    topics: Optional[List[str]] = None
    
    # Instructions
    general_instructions: Optional[List[str]] = Field(
        default_factory=lambda: [
            "All questions are compulsory.",
            "Write your answers clearly and legibly.",
            "Show all working for numerical problems.",
            "Diagrams should be drawn with pencil.",
        ]
    )
    
    # Header configuration
    include_header: bool = True
    include_footer: bool = True
    watermark_text: Optional[str] = None
