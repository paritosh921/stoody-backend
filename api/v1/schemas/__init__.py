"""
API Schemas Package

Contains Pydantic request/response models for API endpoints.
"""

from .question_generation import (
    # Request Models
    QuestionTypeConfigRequest,
    DifficultyDistributionRequest,
    GenerationConfigRequest,
    PaperConfigRequest,
    GenerateFromNotesRequest,
    GenerateFromTopicRequest,
    QuestionPreviewRequest,
    # Response Models
    GeneratedQuestionResponse,
    PaperSectionResponse,
    GeneratedPaperResponse,
    QuestionPreviewResponse,
    HealthCheckResponse,
)

__all__ = [
    # Request Models
    "QuestionTypeConfigRequest",
    "DifficultyDistributionRequest",
    "GenerationConfigRequest",
    "PaperConfigRequest",
    "GenerateFromNotesRequest",
    "GenerateFromTopicRequest",
    "QuestionPreviewRequest",
    # Response Models
    "GeneratedQuestionResponse",
    "PaperSectionResponse",
    "GeneratedPaperResponse",
    "QuestionPreviewResponse",
    "HealthCheckResponse",
]
