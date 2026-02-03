"""
Paper Templates Package

Contains pre-defined paper templates and question type definitions.
"""

from .paper_templates import (
    PAPER_TEMPLATES,
    QUESTION_TYPES,
    DIFFICULTY_LEVELS,
    BLOOM_LEVELS,
    get_all_templates,
    get_question_types_info,
)

__all__ = [
    "PAPER_TEMPLATES",
    "QUESTION_TYPES",
    "DIFFICULTY_LEVELS",
    "BLOOM_LEVELS",
    "get_all_templates",
    "get_question_types_info",
]
