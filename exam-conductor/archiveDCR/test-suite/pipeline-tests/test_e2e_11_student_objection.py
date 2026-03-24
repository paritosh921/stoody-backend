"""
E2E-11: Student BFF objection lifecycle.

Services involved: svc-student-bff, svc-review, svc-score-engine.

What it proves:
    A student files an objection through the student BFF, the teacher
    resolves it via the review service, and the student sees the updated
    score through the student BFF.  Validates the full objection
    round-trip including FSM state transitions visible through the BFF.

Test-ID: E2E-11  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from bff_helpers import (
    review_post,
    seed_student_score,
    student_get,
    student_post,
)

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

# Required fields per OpenAPI ``StudentObjection`` schema.
OBJECTION_REQUIRED = {"objection_id", "exam_id", "question_id", "status"}
# Required fields per ``StudentScoreView`` schema.
SCORE_VIEW_REQUIRED = {
    "exam_id", "total_score", "percentage", "percentile", "questions",
}

LONG_OBJECTION = (
    "I believe my answer deserves more marks because "
    "I showed the correct working steps clearly."
)


class TestStudentObjectionLifecycle:
    """E2E-11 -- student BFF objection lifecycle (real)."""

    async def test_file_objection_returns_201(
        self, http_session, publish_event, nats_client,
        ai_result_factory,
    ):
        """POST /api/v1/student/objections -> 201 with correct shape."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        await seed_student_score(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, student_id=student_id,
        )
        await asyncio.sleep(1)

        status, body = await student_post(
            http_session,
            "/api/v1/student/objections",
            {
                "exam_id": exam_id,
                "question_id": "q3",
                "objection_text": LONG_OBJECTION,
            },
        )
        assert status == 201, f"Expected 201, got {status}"
        assert body is not None
        missing = OBJECTION_REQUIRED - set(body.keys())
        assert not missing, f"StudentObjection missing: {missing}"
        assert body["status"] == "filed"

    async def test_objection_appears_in_student_list(
        self, http_session, publish_event, nats_client,
        ai_result_factory,
    ):
        """GET /api/v1/student/objections returns the filed objection."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        await seed_student_score(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, student_id=student_id,
        )
        await asyncio.sleep(1)

        # File the objection.
        status, filed = await student_post(
            http_session,
            "/api/v1/student/objections",
            {
                "exam_id": exam_id,
                "question_id": "q5",
                "objection_text": (
                    "The AI recognition missed my diagram "
                    "which was clearly within the answer area."
                ),
            },
        )
        assert status == 201

        # Query the objection list.
        ls, body = await student_get(http_session, "/api/v1/student/objections")
        assert ls == 200
        assert "items" in body
        items = body["items"]
        oid = filed["objection_id"]
        found = [o for o in items if o.get("objection_id") == oid]
        assert len(found) == 1, f"Objection {oid} not in list"
        assert found[0]["status"] == "filed"

    async def test_teacher_resolves_student_sees_updated_score(
        self, http_session, publish_event, event_waiter,
        nats_client, ai_result_factory,
    ):
        """Full round-trip: file -> resolve -> score updated via BFF."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        question_id = "q4"
        teacher_id = str(uuid.uuid4())

        # 1) Seed ai_draft score.
        await seed_student_score(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, student_id=student_id,
            question_results=[{
                "question_id": question_id,
                "recognized_text": "answer text",
                "confidence": 0.80,
                "step_breakdown": ["Step 1 text"],
            }],
        )
        await asyncio.sleep(1)

        # 2) File objection via student BFF.
        status, filed = await student_post(
            http_session, "/api/v1/student/objections",
            {
                "exam_id": exam_id,
                "question_id": question_id,
                "objection_text": (
                    "Missed partial credit for showing "
                    "intermediate working steps correctly."
                ),
            },
        )
        assert status == 201

        # 3) Teacher resolves.
        rescore_waiter = event_waiter.wait_for_event(
            "score.updated",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("reason") == "objection_rescored"
            ),
            timeout=20,
        )
        rs, _ = await review_post(
            http_session,
            f"/api/v1/objections/{filed['objection_id']}/resolve",
            {
                "actor_id": teacher_id,
                "resolution": "approved",
                "reason": "Partial credit for correct working",
                "new_score": 4,
            },
        )
        assert rs == 200

        # 4) Wait for re-score event.
        ev = await rescore_waiter
        assert ev["reason"] == "objection_rescored"

        # 5) Student sees updated score.
        await asyncio.sleep(2)
        ss, sb = await student_get(
            http_session,
            f"/api/v1/student/exams/{exam_id}/scores",
        )
        assert ss == 200
        sm = SCORE_VIEW_REQUIRED - set(sb.keys())
        assert not sm, f"StudentScoreView missing: {sm}"
        q_match = [q for q in sb["questions"] if q["question_id"] == question_id]
        assert len(q_match) == 1
        assert q_match[0]["marks_obtained"] == 4

    async def test_objection_fsm_visible_through_bff(
        self, http_session, publish_event, nats_client,
        ai_result_factory,
    ):
        """Rejection status visible via GET /api/v1/student/objections."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        teacher_id = str(uuid.uuid4())

        await seed_student_score(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, student_id=student_id,
        )
        await asyncio.sleep(1)

        status, filed = await student_post(
            http_session, "/api/v1/student/objections",
            {
                "exam_id": exam_id,
                "question_id": "q2",
                "objection_text": (
                    "My handwriting was clearly legible but "
                    "the AI returned very low confidence."
                ),
            },
        )
        assert status == 201

        # Teacher rejects.
        rs, _ = await review_post(
            http_session,
            f"/api/v1/objections/{filed['objection_id']}/resolve",
            {
                "actor_id": teacher_id,
                "resolution": "rejected",
                "reason": "AI recognition correct; no additional marks",
            },
        )
        assert rs == 200

        # Verify updated status.
        await asyncio.sleep(2)
        ls, lb = await student_get(http_session, "/api/v1/student/objections")
        assert ls == 200
        oid = filed["objection_id"]
        found = [o for o in lb["items"] if o.get("objection_id") == oid]
        assert len(found) == 1
        assert found[0]["status"] == "resolved"
