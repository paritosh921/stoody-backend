"""
E2E-03: AI result -> score generation.

Services involved: svc-ai-pipeline, svc-score-engine.

What it proves:
    When a page.ready event is published, svc-ai-pipeline processes the page
    image, produces an ``ai.result`` event, and svc-score-engine consumes it
    to create score records in ``ai_draft`` lifecycle state.  A
    ``score.updated`` event is emitted for each question scored.

Test-ID: E2E-03  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


class TestAIScoring:
    """E2E-03 — page.ready -> ai.result -> score.updated (ai_draft)."""

    async def test_page_ready_produces_ai_result(
        self,
        publish_event,
        event_waiter,
    ):
        """A page.ready event triggers an ai.result event."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        page_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "page.ready",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:00:00Z",
            "exam_id": exam_id,
            "student_id": student_id,
            "page_id": str(uuid.uuid4()),
            "page_number": 1,
            "image_uri": f"s3://exampen-pages/{exam_id}/{student_id}/p1.png",
            "authoritative_source": "strokes",
            "question_ids": ["q1", "q2"],
        }

        waiter = event_waiter.wait_for_event(
            "ai.result",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("student_id") == student_id
            ),
        )

        await publish_event("page.ready", page_event)
        ai_event = await waiter

        assert ai_event["event_type"] == "ai.result"
        assert ai_event["exam_id"] == exam_id
        assert ai_event["student_id"] == student_id
        assert "model_version" in ai_event
        assert isinstance(ai_event["question_results"], list)
        assert len(ai_event["question_results"]) > 0

    async def test_ai_result_triggers_score_updated(
        self,
        publish_event,
        event_waiter,
        ai_result_factory,
    ):
        """An ai.result event triggers a score.updated event in ai_draft."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        ai_event = ai_result_factory.create_event(
            exam_id=exam_id,
            student_id=student_id,
            question_results=[
                {
                    "question_id": "q1",
                    "recognized_text": "x = 5",
                    "confidence": 0.92,
                    "step_breakdown": ["Step 1: x + 3 = 8", "Step 2: x = 5"],
                },
            ],
        )

        waiter = event_waiter.wait_for_event(
            "score.updated",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("student_id") == student_id
                and e.get("lifecycle_state") == "ai_draft"
            ),
        )

        await publish_event("ai.result", ai_event)
        score_event = await waiter

        assert score_event["event_type"] == "score.updated"
        assert score_event["lifecycle_state"] == "ai_draft"
        assert score_event["reason"] == "ai_draft_created"
        assert isinstance(score_event["total_score"], (int, float))

    async def test_ai_result_includes_model_version(
        self,
        publish_event,
        event_waiter,
    ):
        """ai.result event includes model_version for traceability."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        page_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "page.ready",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:00:00Z",
            "exam_id": exam_id,
            "student_id": student_id,
            "page_id": str(uuid.uuid4()),
            "page_number": 1,
            "image_uri": f"s3://exampen-pages/{exam_id}/{student_id}/p1.png",
            "authoritative_source": "strokes",
        }

        waiter = event_waiter.wait_for_event(
            "ai.result",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("page.ready", page_event)
        ai_event = await waiter

        assert "model_version" in ai_event
        assert isinstance(ai_event["model_version"], str)
        assert len(ai_event["model_version"]) > 0

    async def test_ai_result_schema_compliance(
        self,
        ai_result_factory,
    ):
        """Factory-produced ai.result events conform to contract schema."""
        event = ai_result_factory.create_event(
            exam_id=str(uuid.uuid4()),
            student_id=str(uuid.uuid4()),
        )

        required = [
            "event_id",
            "event_type",
            "event_version",
            "occurred_at",
            "exam_id",
            "student_id",
            "model_version",
            "question_results",
        ]
        for f in required:
            assert f in event, f"Missing required field: {f}"

        for qr in event["question_results"]:
            assert "question_id" in qr
            assert "recognized_text" in qr
            assert "confidence" in qr

    async def test_score_updated_schema_compliance(
        self,
        publish_event,
        event_waiter,
        ai_result_factory,
    ):
        """score.updated events conform to contract schema."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        ai_event = ai_result_factory.create_event(
            exam_id=exam_id,
            student_id=student_id,
        )

        waiter = event_waiter.wait_for_event(
            "score.updated",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("ai.result", ai_event)
        score_event = await waiter

        required = [
            "event_id",
            "event_type",
            "event_version",
            "occurred_at",
            "exam_id",
            "student_id",
            "lifecycle_state",
            "total_score",
            "reason",
        ]
        for f in required:
            assert f in score_event, f"Missing required field: {f}"
        assert score_event["lifecycle_state"] in (
            "ai_draft",
            "teacher_reviewed",
            "finalized",
            "published",
            "objection_window",
            "locked",
        )
