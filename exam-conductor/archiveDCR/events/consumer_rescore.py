"""NATS consumer: rescore command handler.

Subscribes to ``EXAMPEN.score.rescore_command``, re-evaluates scores
for the specified exam/student/question, and publishes an updated
``EXAMPEN.score.updated`` event.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..domain.rubric_eval import Rubric, RubricStep, evaluate
from ..storage.ai_result_repo import AIResultRepo
from ..storage.score_event_store import ScoreEventStore
from . import subjects
from .publishers import publish_score_updated

logger = logging.getLogger(__name__)

DURABLE = "exampen-rescore"
QUEUE_GROUP = "exampen-rescore-workers"


def _get_rubric_for_question(question_id: str) -> Rubric:
    """Look up the rubric for a question (placeholder — same as score consumer)."""
    return Rubric(
        question_id=question_id,
        version=1,
        steps=[
            RubricStep(label="step_1", max_marks=5.0, keywords=["step_1"]),
            RubricStep(label="step_2", max_marks=5.0, keywords=["step_2"]),
        ],
        negative_marking=False,
    )


async def rescore_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Re-evaluate scores using the latest AI results.

    Steps:
        1. Fetch existing AI results for the student + exam
        2. Re-evaluate each question against its (possibly updated) rubric
        3. Append new score events with reason='rescore'
        4. Publish score.updated
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")
    student_id = payload.get("student_id", "")
    question_ids = payload.get("question_ids", [])  # empty = all questions

    logger.info(
        "Processing rescore_command event_id=%s exam=%s student=%s",
        event_id, exam_id, student_id,
    )

    db = await db_manager.get_tenant_db(tenant_id)
    ai_repo = AIResultRepo(db)
    score_store = ScoreEventStore(db)

    # Fetch latest AI results
    ai_results = await ai_repo.get_results(exam_id, student_id, tenant_id)
    if not ai_results:
        logger.warning(
            "No AI results found for rescore exam=%s student=%s",
            exam_id, student_id,
        )
        return

    # Use the latest AI result document
    latest = ai_results[-1]
    all_qr = latest.get("question_results", [])

    # Filter to requested question_ids if specified
    if question_ids:
        target_ids = set(question_ids)
        all_qr = [qr for qr in all_qr if qr.get("question_id") in target_ids]

    rescored_count = 0
    for qr in all_qr:
        question_id = qr.get("question_id", "")
        rubric = _get_rubric_for_question(question_id)
        score_result = evaluate(qr, rubric)

        score_event = {
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
            "score": score_result.total_marks,
            "max_marks": score_result.max_marks,
            "confidence": score_result.confidence,
            "rubric_version": score_result.rubric_version,
            "state": "rescored",
            "reason": "rescore_command",
            "step_scores": [
                {
                    "label": ss.label,
                    "awarded": ss.awarded,
                    "max": ss.max,
                    "matched": ss.matched,
                }
                for ss in score_result.step_scores
            ],
        }
        await score_store.append_event(tenant_id, score_event)
        rescored_count += 1

    await publish_score_updated(
        nats, exam_id, student_id,
        reason="rescore_command",
        lifecycle_state="rescored",
        tenant_id=tenant_id,
    )

    logger.info(
        "Rescore complete event_id=%s questions_rescored=%d",
        event_id, rescored_count,
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.score.rescore_command with durable consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await rescore_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.RESCORE_COMMAND,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered rescore_consumer")
