"""
Test data factories for ExamPen E2E pipeline tests.

Generates synthetic event payloads conforming to the NATS event contracts
defined in ``contracts/events/*.schema.json``.
"""

from __future__ import annotations

import base64
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

MINIO_BUCKET = "exampen-pages"


@dataclass
class ExamFactory:
    """Generate exam definitions matching fixture schema."""

    @staticmethod
    def create(
        *,
        exam_id: str | None = None,
        questions_count: int = 10,
        total_marks: int = 42,
    ) -> dict:
        return {
            "id": exam_id or str(uuid.uuid4()),
            "title": "E2E Test Exam",
            "subject": "Mathematics",
            "class_section": "8A",
            "tutor_id": str(uuid.uuid4()),
            "duration_minutes": 60,
            "total_marks": total_marks,
            "questions_count": questions_count,
            "questions": [
                {
                    "question_number": i + 1,
                    "max_marks": 5 if i % 2 == 0 else 2,
                    "steps": 3 if i % 2 == 0 else 1,
                    "step_marks": (
                        [2, 2, 1] if i % 2 == 0 else [2]
                    ),
                    "region_bbox": {
                        "x": 100 + (i % 2) * 1000,
                        "y": 200 + (i // 2) * 500,
                        "width": 900,
                        "height": 450,
                    },
                }
                for i in range(questions_count)
            ],
            "state": "timer_running",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class StudentFactory:
    """Generate student records."""

    @staticmethod
    def create(*, student_id: str | None = None, index: int = 0) -> dict:
        sid = student_id or str(uuid.uuid4())
        return {
            "id": sid,
            "stoody_user_id": f"stu_{index:04d}",
            "first_name": f"Student{index}",
            "last_name": "Test",
            "full_name": f"Student{index} Test",
            "roll_number": f"8A-{index:03d}",
            "class_section": "8A",
            "email": f"student{index}@example.com",
        }

    @staticmethod
    def create_batch(count: int) -> list[dict]:
        return [StudentFactory.create(index=i + 1) for i in range(count)]


@dataclass
class StrokeFactory:
    """Generate synthetic stroke events conforming to contract schemas."""

    @staticmethod
    def create_raw_event(
        *,
        exam_id: str,
        pen_mac: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> dict:
        """Build a ``stroke.raw`` event payload."""
        raw_bytes = bytes(random.getrandbits(8) for _ in range(200))
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "stroke.raw",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "payload_base64": base64.b64encode(raw_bytes).decode(),
            "checksum_crc32": format(
                __import__("binascii").crc32(raw_bytes) & 0xFFFFFFFF,
                "08x",
            ),
            "upload_path": "wifi",
        }

    @staticmethod
    def create_processed_event(
        *,
        exam_id: str,
        pen_mac: str,
        student_id: str,
        page_assignments: list[dict] | None = None,
    ) -> dict:
        """Build a ``stroke.processed`` event payload."""
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "stroke.processed",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "student_id": student_id,
            "normalized_stroke_uri": (
                f"s3://{MINIO_BUCKET}/strokes/{exam_id}/{student_id}.bin"
            ),
            "page_assignments": page_assignments
            or [
                {
                    "page_number": 1,
                    "question_id": "q1",
                    "point_count": 150,
                }
            ],
        }


@dataclass
class AIResultFactory:
    """Generate AI result events."""

    @staticmethod
    def create_event(
        *,
        exam_id: str,
        student_id: str,
        question_results: list[dict] | None = None,
        model_version: str = "hwr-v0.3.1",
    ) -> dict:
        if question_results is None:
            question_results = [
                {
                    "question_id": f"q{i + 1}",
                    "recognized_text": f"[HWR output for Q{i + 1}]",
                    "confidence": round(random.uniform(0.6, 0.98), 3),
                    "step_breakdown": [
                        f"Step {j + 1} text" for j in range(3)
                    ],
                }
                for i in range(10)
            ]
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "ai.result",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "student_id": student_id,
            "model_version": model_version,
            "question_results": question_results,
        }


@dataclass
class ScoreFactory:
    """Generate score.updated events."""

    @staticmethod
    def create_event(
        *,
        exam_id: str,
        student_id: str,
        question_id: str = "q1",
        total_score: float = 3.0,
        lifecycle_state: str = "ai_draft",
        reason: str = "ai_draft_created",
    ) -> dict:
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "score.updated",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
            "lifecycle_state": lifecycle_state,
            "total_score": total_score,
            "reason": reason,
        }
