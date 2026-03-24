"""
E2E-06: Plagiarism detection end-to-end.

Services involved: svc-ai-pipeline, svc-plagiarism.

What it proves:
    When a ``plagiarism.check`` event is triggered (all AI results ready for
    an exam), svc-plagiarism computes TF-IDF + structural similarity, flags
    known plagiarism pairs, and publishes a ``plagiarism.result`` event.
    Known similar pairs are detected; false positives stay below threshold.

Test-ID: E2E-06  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


class TestPlagiarismDetection:
    """E2E-06 — plagiarism.check -> plagiarism.result with flags."""

    async def test_plagiarism_check_produces_result(
        self,
        publish_event,
        event_waiter,
    ):
        """A plagiarism.check event triggers a plagiarism.result event."""
        exam_id = str(uuid.uuid4())

        check_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "plagiarism.check",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:00:00Z",
            "exam_id": exam_id,
            "student_count": 10,
            "question_count": 10,
            "trigger": "all_ai_results_ready",
        }

        waiter = event_waiter.wait_for_event(
            "plagiarism.result",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("plagiarism.check", check_event)
        result = await waiter

        assert result["event_type"] == "plagiarism.result"
        assert result["exam_id"] == exam_id
        assert isinstance(result["flags"], list)

    async def test_known_similar_pair_detected(
        self,
        publish_event,
        event_waiter,
        ai_result_factory,
    ):
        """Two students with identical answers produce a plagiarism flag."""
        exam_id = str(uuid.uuid4())
        student_a = str(uuid.uuid4())
        student_b = str(uuid.uuid4())

        # Create near-identical AI results for both students.
        shared_results = [
            {
                "question_id": f"q{i + 1}",
                "recognized_text": "The mitochondria is the powerhouse of the cell. "
                "Energy is produced through ATP synthesis.",
                "confidence": 0.95,
            }
            for i in range(5)
        ]

        for sid in (student_a, student_b):
            ai_event = ai_result_factory.create_event(
                exam_id=exam_id,
                student_id=sid,
                question_results=shared_results,
            )
            await publish_event("ai.result", ai_event)

        # Allow AI results to settle, then trigger check.
        await asyncio.sleep(2)

        check_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "plagiarism.check",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:05:00Z",
            "exam_id": exam_id,
            "student_count": 2,
            "question_count": 5,
            "trigger": "all_ai_results_ready",
        }

        waiter = event_waiter.wait_for_event(
            "plagiarism.result",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("plagiarism.check", check_event)
        result = await waiter

        # At least one flag should exist for the identical pair.
        assert len(result["flags"]) >= 1
        flag = result["flags"][0]
        pair_ids = {flag["student_a_id"], flag["student_b_id"]}
        assert pair_ids == {student_a, student_b}
        assert flag["composite_score"] > 0.80
        assert flag["severity"] in ("review_recommended", "strong_match")

    async def test_dissimilar_answers_no_false_positive(
        self,
        publish_event,
        event_waiter,
        ai_result_factory,
    ):
        """Two students with different answers should produce no flags."""
        exam_id = str(uuid.uuid4())
        student_a = str(uuid.uuid4())
        student_b = str(uuid.uuid4())

        results_a = [
            {
                "question_id": f"q{i + 1}",
                "recognized_text": f"Unique answer A for question {i + 1}. "
                f"Photosynthesis occurs in chloroplasts. Plant cells differ from animal cells.",
                "confidence": 0.90,
            }
            for i in range(5)
        ]

        results_b = [
            {
                "question_id": f"q{i + 1}",
                "recognized_text": f"Completely different answer B for question {i + 1}. "
                f"Newton's third law states every action has an equal and opposite reaction.",
                "confidence": 0.88,
            }
            for i in range(5)
        ]

        await publish_event(
            "ai.result",
            ai_result_factory.create_event(
                exam_id=exam_id,
                student_id=student_a,
                question_results=results_a,
            ),
        )
        await publish_event(
            "ai.result",
            ai_result_factory.create_event(
                exam_id=exam_id,
                student_id=student_b,
                question_results=results_b,
            ),
        )

        await asyncio.sleep(2)

        check_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "plagiarism.check",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:10:00Z",
            "exam_id": exam_id,
            "student_count": 2,
            "question_count": 5,
            "trigger": "all_ai_results_ready",
        }

        waiter = event_waiter.wait_for_event(
            "plagiarism.result",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("plagiarism.check", check_event)
        result = await waiter

        # No flags expected for dissimilar answers.
        high_flags = [
            f for f in result["flags"] if f["composite_score"] > 0.80
        ]
        assert len(high_flags) == 0, (
            f"False positive: {len(high_flags)} high-score flag(s) for "
            "dissimilar answers"
        )

    async def test_plagiarism_result_schema_compliance(
        self,
        publish_event,
        event_waiter,
    ):
        """plagiarism.result conforms to contract schema."""
        exam_id = str(uuid.uuid4())

        check_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "plagiarism.check",
            "event_version": "1.0.0",
            "occurred_at": "2026-03-19T10:00:00Z",
            "exam_id": exam_id,
            "student_count": 5,
            "question_count": 10,
            "trigger": "manual_recheck",
        }

        waiter = event_waiter.wait_for_event(
            "plagiarism.result",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("plagiarism.check", check_event)
        result = await waiter

        required = [
            "event_id",
            "event_type",
            "event_version",
            "occurred_at",
            "exam_id",
            "flags",
        ]
        for f in required:
            assert f in result, f"Missing required field: {f}"

        for flag in result["flags"]:
            for ff in [
                "flag_id",
                "student_a_id",
                "student_b_id",
                "question_id",
                "composite_score",
                "severity",
            ]:
                assert ff in flag, f"Flag missing field: {ff}"
            assert flag["severity"] in (
                "review_recommended",
                "strong_match",
            )
