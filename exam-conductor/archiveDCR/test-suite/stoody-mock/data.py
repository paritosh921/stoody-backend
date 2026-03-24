"""Canned response data for the Stoody mock server.

All data is static and deterministic so tests can assert against
known values. Organized by entity type.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

USERS: dict[str, dict[str, Any]] = {
    "tutor-001": {
        "user_id": "tutor-001",
        "name": "Rajesh Kumar",
        "display_name": "Rajesh Kumar",
        "email": "rajesh.kumar@springfield.edu",
        "phone": "+91-9876500001",
        "role": "tutor",
        "institute_name": "Springfield International School",
        "subject_ids": ["math-101", "math-201"],
        "class_ids": ["class-10a", "class-10b"],
    },
    "tutor-002": {
        "user_id": "tutor-002",
        "name": "Priya Sharma",
        "display_name": "Priya Sharma",
        "email": "priya.sharma@springfield.edu",
        "phone": "+91-9876500002",
        "role": "tutor",
        "institute_name": "Springfield International School",
        "subject_ids": ["science-101"],
        "class_ids": ["class-10a"],
    },
    "student-001": {
        "user_id": "student-001",
        "name": "Arjun Mehta",
        "display_name": "Arjun Mehta",
        "email": "arjun.mehta@student.springfield.edu",
        "role": "student",
        "institute_name": "Springfield International School",
        "class_id": "class-10a",
        "section_id": "section-a",
        "roll_number": "10A-01",
    },
    "student-002": {
        "user_id": "student-002",
        "name": "Sneha Patel",
        "display_name": "Sneha Patel",
        "email": "sneha.patel@student.springfield.edu",
        "role": "student",
        "institute_name": "Springfield International School",
        "class_id": "class-10a",
        "section_id": "section-a",
        "roll_number": "10A-02",
    },
    "student-003": {
        "user_id": "student-003",
        "name": "Rohit Gupta",
        "display_name": "Rohit Gupta",
        "email": "rohit.gupta@student.springfield.edu",
        "role": "student",
        "institute_name": "Springfield International School",
        "class_id": "class-10b",
        "section_id": "section-b",
        "roll_number": "10B-01",
    },
    "parent-001": {
        "user_id": "parent-001",
        "name": "Vikram Mehta",
        "display_name": "Vikram Mehta",
        "email": "vikram.mehta@gmail.com",
        "phone": "+91-9876500010",
        "role": "parent",
        "institute_name": "Springfield International School",
    },
    "admin-001": {
        "user_id": "admin-001",
        "name": "Dr. Sunita Reddy",
        "display_name": "Dr. Sunita Reddy",
        "email": "sunita.reddy@springfield.edu",
        "phone": "+91-9876500099",
        "role": "admin",
        "institute_name": "Springfield International School",
    },
}

# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------

PARENT_CHILDREN: dict[str, list[dict[str, str]]] = {
    "parent-001": [
        {"student_id": "student-001", "name": "Arjun Mehta"},
        {"student_id": "student-003", "name": "Rohit Gupta"},
    ],
}

# ---------------------------------------------------------------------------
# Students (by class + section)
# ---------------------------------------------------------------------------

STUDENTS_BY_CLASS: dict[str, list[dict[str, Any]]] = {
    "class-10a:section-a": [
        {"student_id": "student-001", "name": "Arjun Mehta", "roll_number": "10A-01"},
        {"student_id": "student-002", "name": "Sneha Patel", "roll_number": "10A-02"},
    ],
    "class-10b:section-b": [
        {"student_id": "student-003", "name": "Rohit Gupta", "roll_number": "10B-01"},
    ],
}

# ---------------------------------------------------------------------------
# Tutors (by subject)
# ---------------------------------------------------------------------------

TUTORS_BY_SUBJECT: dict[str, list[dict[str, Any]]] = {
    "math-101": [
        {"user_id": "tutor-001", "name": "Rajesh Kumar"},
    ],
    "math-201": [
        {"user_id": "tutor-001", "name": "Rajesh Kumar"},
    ],
    "science-101": [
        {"user_id": "tutor-002", "name": "Priya Sharma"},
    ],
}

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

CLASSES: list[dict[str, Any]] = [
    {
        "class_id": "class-10a",
        "name": "Class 10-A",
        "sections": [
            {"section_id": "section-a", "name": "Section A"},
        ],
    },
    {
        "class_id": "class-10b",
        "name": "Class 10-B",
        "sections": [
            {"section_id": "section-b", "name": "Section B"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

SUBJECTS: list[dict[str, Any]] = [
    {"subject_id": "math-101", "name": "Mathematics (Basic)", "grade": "10"},
    {"subject_id": "math-201", "name": "Mathematics (Advanced)", "grade": "10"},
    {"subject_id": "science-101", "name": "General Science", "grade": "10"},
]
