"""
Practice Module - Utilities
"""

from .question_helpers import (
    load_question_doc,
    options_text_from_question,
    figure_images_base64,
    option_images_base64,
    normalize_choice_text,
    normalize_numeric_text,
)

__all__ = [
    "load_question_doc",
    "options_text_from_question",
    "figure_images_base64",
    "option_images_base64",
    "normalize_choice_text",
    "normalize_numeric_text",
]
