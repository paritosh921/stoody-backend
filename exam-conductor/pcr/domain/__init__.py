"""
PCR Domain Layer — Pure business logic for Paginated Character Recognition.

This package contains segmentation, classification, and flagging logic.
No storage, no LLM calls, no API routes. Pure domain models and processing.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md
"""

from .response_models import (
    BoundingBox,
    ContentType,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    PageOCR,
    TextBlock,
)
from .flag_registry import FLAG_REGISTRY, get_flag_definition

__all__ = [
    "BoundingBox",
    "ContentType",
    "DetectedResponse",
    "Flag",
    "FlagSeverity",
    "FlagType",
    "PageOCR",
    "TextBlock",
    "FLAG_REGISTRY",
    "get_flag_definition",
]
