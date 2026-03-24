"""Generate student, tutor, and exam entities."""

from __future__ import annotations

import random
import uuid

from .constants import (
    CLASSES,
    FIRST_NAMES,
    LAST_NAMES,
    SUBJECTS,
    VARIANTS,
)
from .helpers import distribute_marks, iso_now


def gen_students(rng: random.Random, count: int) -> list[dict]:
    """Generate student records."""
    students = []
    for i in range(count):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        cls = CLASSES[i % len(CLASSES)]
        students.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "stoody_user_id": f"stu_{i+1:04d}",
            "first_name": first,
            "last_name": last,
            "full_name": f"{first} {last}",
            "roll_number": f"{cls}-{i+1:03d}",
            "class_section": cls,
            "email": f"{first.lower()}.{last.lower()}{i}@example.com",
        })
    return students


def gen_tutors(rng: random.Random, count: int = 5) -> list[dict]:
    """Generate tutor records."""
    tutor_names = [
        ("Dr. Anita", "Sharma"), ("Prof. Rajesh", "Kumar"),
        ("Ms. Priya", "Iyer"), ("Mr. Vikram", "Joshi"),
        ("Dr. Sunita", "Patel"),
    ]
    tutors = []
    for i in range(min(count, len(tutor_names))):
        first, last = tutor_names[i]
        tutors.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "stoody_user_id": f"tut_{i+1:04d}",
            "first_name": first,
            "last_name": last,
            "full_name": f"{first} {last}",
            "subject": SUBJECTS[i % len(SUBJECTS)],
            "email": (
                f"{first.lower().replace(' ', '').replace('.', '')}"
                f".{last.lower()}@school.example.com"
            ),
        })
    return tutors


def gen_exams(
    rng: random.Random,
    count: int,
    questions_per_exam: int,
    tutors: list[dict],
) -> list[dict]:
    """Generate exam definitions with rubrics and question regions."""
    exams = []
    for i in range(count):
        subject = SUBJECTS[i % len(SUBJECTS)]
        tutor = tutors[i % len(tutors)]
        exam_id = str(uuid.UUID(int=rng.getrandbits(128)))

        questions = _gen_questions(rng, questions_per_exam, subject)
        total_marks = sum(q["max_marks"] for q in questions)

        exams.append({
            "id": exam_id,
            "title": f"{subject} Exam {i+1}",
            "subject": subject,
            "class_section": CLASSES[i % len(CLASSES)],
            "tutor_id": tutor["id"],
            "tutor_name": tutor["full_name"],
            "duration_minutes": rng.choice([60, 90, 120]),
            "total_marks": total_marks,
            "questions_count": questions_per_exam,
            "variants": VARIANTS[:rng.randint(2, 4)],
            "questions": questions,
            "state": "locked",
            "created_at": iso_now(-30 + i),
            "scheduled_at": iso_now(-20 + i),
        })
    return exams


def _gen_questions(
    rng: random.Random,
    count: int,
    subject: str,
) -> list[dict]:
    """Generate question definitions with rubrics and bounding boxes."""
    questions = []
    for q in range(count):
        max_marks = rng.choice([2, 3, 4, 5, 5, 10])
        steps = rng.randint(1, min(max_marks, 5))
        step_marks = distribute_marks(rng, max_marks, steps)

        row = q // 2
        col = q % 2
        x0 = 100 + col * 1000
        y0 = 200 + row * 500

        questions.append({
            "question_number": q + 1,
            "text": f"Q{q+1}: Sample {subject} question",
            "max_marks": max_marks,
            "steps": steps,
            "step_marks": step_marks,
            "rubric": {
                "criteria": [
                    f"Step {s+1}: {m} mark(s)"
                    for s, m in enumerate(step_marks)
                ],
            },
            "region_bbox": {
                "x": x0, "y": y0,
                "width": 900, "height": 450,
            },
        })
    return questions
