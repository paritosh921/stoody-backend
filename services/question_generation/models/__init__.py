"""
Question Generation Models Package
"""

from .config import (
    QuestionType,
    DifficultyLevel,
    BloomLevel,
    QuestionTypeConfig,
    DifficultyDistribution,
    QuestionGenerationConfig,
    PaperConfig,
)

from .question import (
    QuestionOption,
    MarkingStep,
    DiagramSpec,
    GeneratedQuestion,
    GenerationSource,
)

from .paper import (
    PaperSection,
    GeneratedPaper,
    PaperStatus,
)

from .job import (
    Job,
    JobType,
    JobStatus,
    JobProgress,
    QuestionDraft,
    ReviewResult,
    ReviewIssue,
    QuestionGenerationAttempt,
    GeneratedQuestionRecord,
    PaperGenerationState,
    PDFIngestionMetadata,
    PDFIngestionResult,
    PaperGenerationMetadata,
    PaperGenerationResult,
)

__all__ = [
    # Config models
    "QuestionType",
    "DifficultyLevel",
    "BloomLevel",
    "QuestionTypeConfig",
    "DifficultyDistribution",
    "QuestionGenerationConfig",
    "PaperConfig",
    # Question models
    "QuestionOption",
    "MarkingStep",
    "DiagramSpec",
    "GeneratedQuestion",
    "GenerationSource",
    # Paper models
    "PaperSection",
    "GeneratedPaper",
    "PaperStatus",
    # Job models
    "Job",
    "JobType",
    "JobStatus",
    "JobProgress",
    "QuestionDraft",
    "ReviewResult",
    "ReviewIssue",
    "QuestionGenerationAttempt",
    "GeneratedQuestionRecord",
    "PaperGenerationState",
    "PDFIngestionMetadata",
    "PDFIngestionResult",
    "PaperGenerationMetadata",
    "PaperGenerationResult",
]
