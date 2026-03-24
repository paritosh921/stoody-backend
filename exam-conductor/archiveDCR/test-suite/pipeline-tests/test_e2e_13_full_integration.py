"""
E2E-13: Full happy-path integration smoke test.

Services involved: ALL services -- exam-orch, stroke-ingest, stroke-proc,
doc-assembly, ai-pipeline, score-engine, review, analytics, teacher-bff,
student-bff, notify (webhook), Stoody mock.

What it proves:
    The entire system works end-to-end in a single linear scenario:
    create exam -> arm -> strokes -> AI -> score -> teacher review ->
    publish -> student views -> objection -> resolve -> analytics ->
    webhook delivered.

Test-ID: E2E-13  (bonus, extends TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest

from bff_helpers import (
    analytics_get,
    exam_orch_post,
    review_post,
    score_engine_post,
    student_get,
    student_post,
    teacher_get,
    webhook_get,
)
from conftest import STUDENT_BFF_URL, TEACHER_BFF_URL

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

PHASE_TIMEOUT = 30
STUDENT_COUNT = 5
QUESTION_COUNT = 3


class TestFullHappyPath:
    """E2E-13 -- comprehensive happy-path smoke test."""

    async def test_full_lifecycle(
        self, http_session, publish_event, event_waiter,
        nats_client, stroke_factory, ai_result_factory,
        student_factory, exam_factory,
    ):
        """End-to-end: create -> arm -> strokes -> AI -> score ->
        review -> publish -> student -> objection -> resolve ->
        analytics -> webhook."""

        teacher_id = str(uuid.uuid4())
        students = student_factory.create_batch(STUDENT_COUNT)

        # -- Phase 1: Create exam ----------------------------------------
        s, exam = await exam_orch_post(
            http_session, "/api/v1/exams",
            {
                "title": "E2E-13 Full Integration Exam",
                "subject_id": "math-101",
                "class_id": "class-8",
                "section_id": "section-A",
                "scheduled_at": datetime.now(timezone.utc).isoformat(),
                "duration_min": 60,
                "question_count": QUESTION_COUNT,
                "total_marks": QUESTION_COUNT * 5,
            },
        )
        assert s == 201, f"Phase 1: exam create -> {s}"
        exam_id = exam["exam_id"]

        # -- Phase 2: Arm then start timer -------------------------------
        arm_w = event_waiter.wait_for_event(
            "exam.lifecycle",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("to_state") == "armed"
            ),
            timeout=PHASE_TIMEOUT,
        )
        s, _ = await exam_orch_post(
            http_session,
            f"/api/v1/exams/{exam_id}/transitions",
            {"to_state": "armed", "actor_id": teacher_id},
        )
        assert s == 200, f"Phase 2: arm -> {s}"
        await arm_w

        s, _ = await exam_orch_post(
            http_session,
            f"/api/v1/exams/{exam_id}/transitions",
            {"to_state": "timer_running", "actor_id": teacher_id},
        )
        assert s == 200, f"Phase 2: timer_running -> {s}"

        # -- Phase 3: Raw strokes ---------------------------------------
        for stu in students:
            mac = f"AA:BB:CC:{stu['id'][:2]}:{stu['id'][2:4]}:01"
            for ci in range(QUESTION_COUNT):
                raw = stroke_factory.create_raw_event(
                    exam_id=exam_id, pen_mac=mac,
                    chunk_index=ci, total_chunks=QUESTION_COUNT,
                )
                await publish_event("stroke.raw", raw)
        await asyncio.sleep(5)

        # -- Phase 4: AI results -> score.updated -----------------------
        score_done = asyncio.Event()
        score_events: list[dict] = []

        async def _sh(msg):
            d = json.loads(msg.data.decode())
            if d.get("exam_id") != exam_id:
                return
            score_events.append(d)
            if len(score_events) >= STUDENT_COUNT:
                score_done.set()

        sub = await nats_client.subscribe("score.updated", cb=_sh)
        for stu in students:
            ai = ai_result_factory.create_event(
                exam_id=exam_id, student_id=stu["id"],
                question_results=[
                    {
                        "question_id": f"q{i+1}",
                        "recognized_text": f"A{i+1}",
                        "confidence": 0.85,
                        "step_breakdown": [f"S{j}" for j in range(3)],
                    }
                    for i in range(QUESTION_COUNT)
                ],
            )
            await publish_event("ai.result", ai)
        await asyncio.wait_for(score_done.wait(), timeout=PHASE_TIMEOUT)
        await sub.unsubscribe()
        assert len(score_events) >= STUDENT_COUNT, (
            f"Phase 4: {len(score_events)}/{STUDENT_COUNT} scores"
        )

        # -- Phase 5: Teacher reviews via BFF ---------------------------
        await asyncio.sleep(2)
        s, ov = await teacher_get(
            http_session,
            f"/api/v1/teacher/exams/{exam_id}/scores",
        )
        assert s == 200 and len(ov.get("rows", [])) > 0

        # Override one student's Q1 score.
        target = students[0]["id"]
        s, _ = await score_engine_post(
            http_session,
            f"/api/v1/scores/{exam_id}/students/{target}/questions/q1",
            {
                "teacher_id": teacher_id,
                "new_score": 5,
                "reason": "Full marks -- correct working",
            },
        )
        assert s in (200, 204), f"Phase 5: override -> {s}"

        # -- Phase 6: Finalize and publish -------------------------------
        s, _ = await score_engine_post(
            http_session,
            f"/api/v1/scores/{exam_id}/finalize",
            {"actor_id": teacher_id},
        )
        assert s == 200, f"Phase 6: finalize -> {s}"

        pub_w = event_waiter.wait_for_event(
            "score.updated",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("lifecycle_state") == "published"
            ),
            timeout=PHASE_TIMEOUT,
        )
        s, _ = await score_engine_post(
            http_session,
            f"/api/v1/scores/{exam_id}/publish",
            {"actor_id": teacher_id, "objection_window_days": 7},
        )
        assert s == 200, f"Phase 6: publish -> {s}"
        await pub_w

        # -- Phase 7: Student views score --------------------------------
        await asyncio.sleep(2)
        s, sb = await student_get(
            http_session,
            f"/api/v1/student/exams/{exam_id}/scores",
        )
        assert s == 200
        assert "questions" in sb
        assert len(sb["questions"]) == QUESTION_COUNT

        # -- Phase 8: Student files objection ----------------------------
        s, obj = await student_post(
            http_session, "/api/v1/student/objections",
            {
                "exam_id": exam_id,
                "question_id": "q2",
                "objection_text": (
                    "The AI missed my diagram explanation "
                    "which demonstrates the correct approach."
                ),
            },
        )
        assert s == 201, f"Phase 8: objection -> {s}"

        # -- Phase 9: Teacher resolves -----------------------------------
        rw = event_waiter.wait_for_event(
            "score.updated",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("reason") == "objection_rescored"
            ),
            timeout=PHASE_TIMEOUT,
        )
        s, _ = await review_post(
            http_session,
            f"/api/v1/objections/{obj['objection_id']}/resolve",
            {
                "actor_id": teacher_id,
                "resolution": "approved",
                "reason": "Diagram merits partial credit",
                "new_score": 4,
            },
        )
        assert s == 200, f"Phase 9: resolve -> {s}"
        re = await rw
        assert re["reason"] == "objection_rescored"

        # -- Phase 10: Analytics -----------------------------------------
        await asyncio.sleep(3)
        s, an = await analytics_get(
            http_session,
            f"/api/v1/analytics/exams/{exam_id}/class-stats",
        )
        assert s == 200
        for k in ("mean", "median", "std_dev", "pass_rate"):
            assert k in an, f"ClassStats missing: {k}"

        # -- Phase 11: Webhook -------------------------------------------
        s, wh = await webhook_get(http_session)
        assert s == 200
        wl = wh if isinstance(wh, list) else []
        match = [
            w for w in wl
            if w.get("exam_id") == exam_id
            or (isinstance(w.get("payload"), dict)
                and w["payload"].get("exam_id") == exam_id)
        ]
        assert len(match) >= 1, (
            f"Phase 11: no webhook for exam {exam_id}"
        )

    async def test_rbac_cross_role_isolation(self, http_session):
        """Student tokens blocked on teacher; teacher on student."""
        eid = str(uuid.uuid4())

        # Student on teacher.
        async with http_session.get(
            f"{TEACHER_BFF_URL}/api/v1/teacher/exams/{eid}/scores",
            headers={"Authorization": "Bearer student_test_token"},
        ) as r:
            assert r.status in (401, 403)

        # Teacher on student objections.
        async with http_session.post(
            f"{STUDENT_BFF_URL}/api/v1/student/objections",
            headers={"Authorization": "Bearer teacher_test_token"},
            json={
                "exam_id": eid,
                "question_id": "q1",
                "objection_text": "Teacher should not file this.",
            },
        ) as r:
            assert r.status in (401, 403)
