from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import pytest
from bson import ObjectId

from services.exam_analytics import load_published_exam_attempts


def _snapshot(submission_id: str, exam_id: str, score: float, maximum: float):
    core = {
        "snapshot_version": 1,
        "submission_id": submission_id,
        "exam_id": exam_id,
        "student_id": "STU-1",
        "paper_version_id": "paper-1",
        "score_rows": [
            {
                "question_id": "q-1",
                "score": score,
                "max_score": maximum,
            }
        ],
        "total_score": score,
        "total_max_score": maximum,
        "published_by": "teacher-1",
        "published_at": "2026-08-07T09:00:00+00:00",
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return {**core, "snapshot_hash": digest}


@pytest.mark.asyncio
async def test_published_exam_attempt_combines_snapshot_and_dcr():
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["analytics"]
    student_oid = ObjectId()
    students = [{"_id": student_oid, "student_id": "STU-1"}]

    await db["evalpen_questions"].insert_one(
        {"exam_id": "exam-1", "question_id": "q-1", "subject": "Physics"}
    )
    await db["evalpen_submissions"].insert_many(
        [
            {
                "submission_id": "published-1",
                "exam_id": "exam-1",
                "student_id": "STU-1",
                "publication_status": "published",
                "published_at": datetime(2026, 8, 7, tzinfo=timezone.utc),
                "publication_snapshot": _snapshot("published-1", "exam-1", 3, 5),
            },
            {
                "submission_id": "draft-1",
                "exam_id": "exam-draft",
                "student_id": "STU-1",
                "publication_status": "ready",
            },
        ]
    )
    await db["exampen_dcr_results"].insert_one(
        {
            "exam_id": "exam-1",
            "student_id": str(student_oid),
            "question_id": "dcr-q-1",
            "score": 4,
            "max_score": 5,
        }
    )

    attempts = await load_published_exam_attempts(db, students)

    assert len(attempts) == 1
    assert attempts[0].student_key == str(student_oid)
    assert attempts[0].total_score == 7
    assert attempts[0].max_score == 10
    assert attempts[0].percentage == 70
    assert attempts[0].subject == "Physics"


@pytest.mark.asyncio
async def test_invalid_pcr_snapshot_withholds_entire_exam_attempt():
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["analytics"]
    student_oid = ObjectId()
    students = [{"_id": student_oid, "student_id": "STU-1"}]
    await db["evalpen_questions"].insert_one(
        {"exam_id": "exam-1", "question_id": "q-1"}
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "published-1",
            "exam_id": "exam-1",
            "student_id": "STU-1",
            "publication_status": "published",
            "publication_snapshot": {"total_score": 5, "total_max_score": 5},
        }
    )
    await db["exampen_dcr_results"].insert_one(
        {
            "exam_id": "exam-1",
            "student_id": "STU-1",
            "question_id": "dcr-q-1",
            "score": 5,
            "max_score": 5,
        }
    )

    assert await load_published_exam_attempts(db, students) == []


@pytest.mark.asyncio
async def test_published_dcr_exam_deduplicates_question_revisions():
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["analytics"]
    student_oid = ObjectId()
    students = [{"_id": student_oid, "student_id": "STU-1"}]
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "published-dcr",
            "exam_id": "exam-dcr",
            "student_id": "STU-1",
            "publication_status": "published",
            "published_at": datetime(2026, 8, 7, tzinfo=timezone.utc),
        }
    )
    await db["exampen_dcr_results"].insert_many(
        [
            {
                "exam_id": "exam-dcr",
                "student_id": "STU-1",
                "question_id": "q-1",
                "score": 0,
                "max_score": 2,
                "updated_at": datetime(2026, 8, 6, tzinfo=timezone.utc),
            },
            {
                "exam_id": "exam-dcr",
                "student_id": "STU-1",
                "question_id": "q-1",
                "score": 2,
                "max_score": 2,
                "updated_at": datetime(2026, 8, 7, tzinfo=timezone.utc),
            },
        ]
    )

    attempts = await load_published_exam_attempts(db, students)

    assert len(attempts) == 1
    assert attempts[0].percentage == 100
    assert attempts[0].max_score == 2
