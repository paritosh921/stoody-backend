"""
Paper Templates and Question Type Definitions

Contains pre-defined paper templates for common exam patterns
and definitions for question types, difficulty levels, and Bloom's taxonomy.
"""

from typing import Any, Dict

from services.question_generation import QuestionType


# ============================================================================
# Paper Templates
# ============================================================================

PAPER_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "cbse_class_10_science": {
        "name": "CBSE Class 10 Science (Theory)",
        "subject": "Science",
        "grade": "Class 10",
        "duration_minutes": 180,
        "total_marks": 80,
        "generation_config": {
            "mcq": {"count": 20, "marks_per_question": 1},
            "short_answer": {"count": 10, "marks_per_question": 2},
            "long_answer": {"count": 6, "marks_per_question": 5},
            "numerical": {"count": 4, "marks_per_question": 5},
            "difficulty_distribution": {"easy": 30, "medium": 50, "hard": 20},
        }
    },
    "cbse_class_12_physics": {
        "name": "CBSE Class 12 Physics",
        "subject": "Physics",
        "grade": "Class 12",
        "duration_minutes": 180,
        "total_marks": 70,
        "generation_config": {
            "mcq": {"count": 15, "marks_per_question": 1},
            "short_answer": {"count": 8, "marks_per_question": 2},
            "long_answer": {"count": 5, "marks_per_question": 5},
            "numerical": {"count": 5, "marks_per_question": 3},
            "difficulty_distribution": {"easy": 25, "medium": 50, "hard": 25},
        }
    },
    "cbse_class_12_chemistry": {
        "name": "CBSE Class 12 Chemistry",
        "subject": "Chemistry",
        "grade": "Class 12",
        "duration_minutes": 180,
        "total_marks": 70,
        "generation_config": {
            "mcq": {"count": 16, "marks_per_question": 1},
            "short_answer": {"count": 10, "marks_per_question": 2},
            "long_answer": {"count": 6, "marks_per_question": 5},
            "numerical": {"count": 2, "marks_per_question": 4},
            "difficulty_distribution": {"easy": 30, "medium": 45, "hard": 25},
        }
    },
    "cbse_class_11_maths": {
        "name": "CBSE Class 11 Mathematics",
        "subject": "Mathematics",
        "grade": "Class 11",
        "duration_minutes": 180,
        "total_marks": 80,
        "generation_config": {
            "mcq": {"count": 20, "marks_per_question": 1},
            "short_answer": {"count": 10, "marks_per_question": 2},
            "long_answer": {"count": 4, "marks_per_question": 5},
            "numerical": {"count": 6, "marks_per_question": 5},
            "difficulty_distribution": {"easy": 25, "medium": 50, "hard": 25},
        }
    },
    "unit_test_quick": {
        "name": "Quick Unit Test",
        "subject": "",
        "grade": "",
        "duration_minutes": 45,
        "total_marks": 25,
        "generation_config": {
            "mcq": {"count": 10, "marks_per_question": 1},
            "short_answer": {"count": 5, "marks_per_question": 2},
            "long_answer": {"count": 1, "marks_per_question": 5},
            "difficulty_distribution": {"easy": 40, "medium": 40, "hard": 20},
        }
    },
}


# ============================================================================
# Question Types
# ============================================================================

QUESTION_TYPES = [
    {
        "id": QuestionType.MCQ.value,
        "name": "Multiple Choice Question",
        "description": "Questions with 4 options, one correct answer",
        "typical_marks": [1],
    },
    {
        "id": QuestionType.TRUE_FALSE.value,
        "name": "True/False",
        "description": "Statement-based questions requiring True or False response",
        "typical_marks": [1],
    },
    {
        "id": QuestionType.FILL_IN_BLANKS.value,
        "name": "Fill in the Blanks",
        "description": "Complete sentences with missing words",
        "typical_marks": [1],
    },
    {
        "id": QuestionType.SHORT_ANSWER.value,
        "name": "Short Answer",
        "description": "Questions requiring 2-4 sentence answers",
        "typical_marks": [2, 3],
    },
    {
        "id": QuestionType.LONG_ANSWER.value,
        "name": "Long Answer",
        "description": "Questions requiring detailed explanations (1-2 paragraphs)",
        "typical_marks": [4, 5],
    },
    {
        "id": QuestionType.NUMERICAL.value,
        "name": "Numerical Problem",
        "description": "Problems requiring calculations with step-by-step solutions",
        "typical_marks": [3, 4, 5],
    },
    {
        "id": QuestionType.MATCH_THE_FOLLOWING.value,
        "name": "Match the Following",
        "description": "Match items from two columns",
        "typical_marks": [4, 5],
    },
]


# ============================================================================
# Difficulty Levels
# ============================================================================

DIFFICULTY_LEVELS = [
    {"id": "easy", "name": "Easy", "description": "Basic recall and understanding"},
    {"id": "medium", "name": "Medium", "description": "Application and analysis"},
    {"id": "hard", "name": "Hard", "description": "Complex problem-solving and synthesis"},
]


# ============================================================================
# Bloom's Taxonomy Levels
# ============================================================================

BLOOM_LEVELS = [
    {"id": "remember", "name": "Remember", "description": "Recall facts and basic concepts"},
    {"id": "understand", "name": "Understand", "description": "Explain ideas or concepts"},
    {"id": "apply", "name": "Apply", "description": "Use information in new situations"},
    {"id": "analyze", "name": "Analyze", "description": "Draw connections among ideas"},
    {"id": "evaluate", "name": "Evaluate", "description": "Justify a decision or course of action"},
    {"id": "create", "name": "Create", "description": "Produce new or original work"},
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_templates() -> Dict[str, Any]:
    """Get all paper templates with metadata."""
    return {
        "templates": PAPER_TEMPLATES,
        "total_count": len(PAPER_TEMPLATES),
    }


def get_question_types_info() -> Dict[str, Any]:
    """Get question types, difficulty levels, and Bloom's taxonomy info."""
    return {
        "question_types": QUESTION_TYPES,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "bloom_levels": BLOOM_LEVELS,
    }
