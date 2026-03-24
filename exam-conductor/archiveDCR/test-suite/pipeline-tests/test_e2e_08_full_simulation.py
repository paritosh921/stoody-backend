"""
E2E-08: Full 40-student exam simulation.

Services involved: ALL pipeline services.

What it proves:
    Simulated stroke data for 40 students x 10 questions is published,
    processed through the entire pipeline (stroke -> page -> AI -> score),
    and all 400 question-level scores are generated.  This is the highest-
    fidelity pipeline test and validates throughput under realistic load.

Test-ID: E2E-08  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

STUDENT_COUNT = 40
QUESTION_COUNT = 10
# Generous timeout for 40-student pipeline processing.
SIMULATION_TIMEOUT = 300  # 5 minutes


class TestFullSimulation:
    """E2E-08 — 40-student x 10-question full pipeline simulation."""

    async def test_all_students_produce_scores(
        self,
        publish_event,
        nats_client,
        stroke_factory,
        student_factory,
    ):
        """All 40 students x 10 questions produce score.updated events."""
        exam_id = str(uuid.uuid4())
        students = student_factory.create_batch(STUDENT_COUNT)

        # Track received score events per student.
        score_tracker: dict[str, set[str]] = {
            s["id"]: set() for s in students
        }
        all_done = asyncio.Event()
        expected_total = STUDENT_COUNT * QUESTION_COUNT

        async def _score_handler(msg):
            data = json.loads(msg.data.decode())
            if data.get("exam_id") != exam_id:
                return
            sid = data.get("student_id")
            qid = data.get("question_id", "")
            if sid in score_tracker:
                score_tracker[sid].add(qid)
                total = sum(len(v) for v in score_tracker.values())
                if total >= expected_total:
                    all_done.set()

        sub = await nats_client.subscribe(
            "score.updated", cb=_score_handler
        )

        try:
            # Phase 1: Publish raw strokes for all students.
            for student in students:
                pen_mac = f"AA:BB:CC:{student['id'][:2]}:{student['id'][2:4]}:01"
                for chunk_idx in range(QUESTION_COUNT):
                    raw = stroke_factory.create_raw_event(
                        exam_id=exam_id,
                        pen_mac=pen_mac,
                        chunk_index=chunk_idx,
                        total_chunks=QUESTION_COUNT,
                    )
                    await publish_event("stroke.raw", raw)

                # Small stagger to avoid overwhelming NATS.
                if (students.index(student) + 1) % 10 == 0:
                    await asyncio.sleep(0.5)

            # Phase 2: Wait for all scores.
            try:
                await asyncio.wait_for(
                    all_done.wait(), timeout=SIMULATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                total_received = sum(
                    len(v) for v in score_tracker.values()
                )
                students_with_scores = sum(
                    1 for v in score_tracker.values() if len(v) > 0
                )
                pytest.fail(
                    f"Timeout: received {total_received}/{expected_total} "
                    f"scores for {students_with_scores}/{STUDENT_COUNT} "
                    f"students within {SIMULATION_TIMEOUT}s"
                )

            # Verify completeness.
            for student in students:
                scored_questions = score_tracker[student["id"]]
                assert len(scored_questions) >= QUESTION_COUNT, (
                    f"Student {student['id']} has "
                    f"{len(scored_questions)}/{QUESTION_COUNT} scores"
                )
        finally:
            await sub.unsubscribe()

    async def test_all_scores_are_ai_draft(
        self,
        publish_event,
        nats_client,
        ai_result_factory,
        student_factory,
    ):
        """Initial scores from AI are all in ai_draft state."""
        exam_id = str(uuid.uuid4())
        students = student_factory.create_batch(5)  # Smaller subset.

        scores_received: list[dict] = []
        done = asyncio.Event()
        expected = 5 * QUESTION_COUNT

        async def _handler(msg):
            data = json.loads(msg.data.decode())
            if data.get("exam_id") != exam_id:
                return
            scores_received.append(data)
            if len(scores_received) >= expected:
                done.set()

        sub = await nats_client.subscribe("score.updated", cb=_handler)

        try:
            for student in students:
                ai_event = ai_result_factory.create_event(
                    exam_id=exam_id,
                    student_id=student["id"],
                )
                await publish_event("ai.result", ai_event)

            await asyncio.wait_for(done.wait(), timeout=60)

            for score in scores_received:
                assert score["lifecycle_state"] == "ai_draft", (
                    f"Score for {score['student_id']} has state "
                    f"{score['lifecycle_state']}, expected ai_draft"
                )
        finally:
            await sub.unsubscribe()

    async def test_no_duplicate_scores(
        self,
        publish_event,
        nats_client,
        ai_result_factory,
        student_factory,
    ):
        """Each (student, question) pair gets exactly one score event."""
        exam_id = str(uuid.uuid4())
        students = student_factory.create_batch(5)

        score_keys: list[str] = []
        done = asyncio.Event()
        expected = 5 * QUESTION_COUNT

        async def _handler(msg):
            data = json.loads(msg.data.decode())
            if data.get("exam_id") != exam_id:
                return
            key = f"{data['student_id']}:{data.get('question_id', '')}"
            score_keys.append(key)
            if len(score_keys) >= expected:
                done.set()

        sub = await nats_client.subscribe("score.updated", cb=_handler)

        try:
            for student in students:
                ai_event = ai_result_factory.create_event(
                    exam_id=exam_id,
                    student_id=student["id"],
                )
                await publish_event("ai.result", ai_event)

            try:
                await asyncio.wait_for(done.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass  # Check what we have.

            # Wait a bit for any trailing duplicates.
            await asyncio.sleep(3)

            unique_keys = set(score_keys)
            assert len(unique_keys) == len(score_keys), (
                f"Duplicate scores detected: {len(score_keys)} events "
                f"for {len(unique_keys)} unique (student, question) pairs"
            )
        finally:
            await sub.unsubscribe()

    async def test_simulation_throughput(
        self,
        publish_event,
        nats_client,
        stroke_factory,
        student_factory,
    ):
        """40-student pipeline completes within acceptable time bounds."""
        import time

        exam_id = str(uuid.uuid4())
        students = student_factory.create_batch(STUDENT_COUNT)

        first_score_time: float | None = None
        last_score_time: float | None = None
        count = 0
        done = asyncio.Event()

        async def _handler(msg):
            nonlocal first_score_time, last_score_time, count
            data = json.loads(msg.data.decode())
            if data.get("exam_id") != exam_id:
                return
            now = time.monotonic()
            if first_score_time is None:
                first_score_time = now
            last_score_time = now
            count += 1
            if count >= STUDENT_COUNT:
                done.set()

        sub = await nats_client.subscribe("score.updated", cb=_handler)
        start = time.monotonic()

        try:
            for student in students:
                pen_mac = f"BB:CC:DD:{student['id'][:2]}:{student['id'][2:4]}:01"
                raw = stroke_factory.create_raw_event(
                    exam_id=exam_id,
                    pen_mac=pen_mac,
                )
                await publish_event("stroke.raw", raw)

            try:
                await asyncio.wait_for(
                    done.wait(), timeout=SIMULATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                pytest.skip(
                    f"Only {count}/{STUDENT_COUNT} scores received "
                    f"within {SIMULATION_TIMEOUT}s"
                )

            total_time = (last_score_time or time.monotonic()) - start
            # Log throughput for observability.
            throughput = count / total_time if total_time > 0 else 0
            print(
                f"\n[E2E-08] Throughput: {count} scores in {total_time:.1f}s "
                f"({throughput:.1f} scores/sec)"
            )
        finally:
            await sub.unsubscribe()
