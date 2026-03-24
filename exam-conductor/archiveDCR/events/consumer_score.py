"""NATS consumer: score evaluation pipeline.

Subscribes to ``EXAMPEN.ai.result``, evaluates each question against
its rubric, writes score events via ``score_event_store``, and publishes
``EXAMPEN.score.updated``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..domain.rubric_eval import Rubric, RubricStep, evaluate
from ..storage.score_event_store import ScoreEventStore
from . import subjects
from .publishers import publish_score_updated

logger = logging.getLogger(__name__)

DURABLE = "exampen-score-evaluator"
QUEUE_GROUP = "exampen-score-workers"


def _get_rubric_for_question(question_id: str) -> Rubric:
    """Look up the rubric for a question.

    In production this would query a rubric store.  For now returns a
    generic placeholder rubric with two keyword-matched steps.
    """
    return Rubric(
        question_id=question_id,
        version=1,
        steps=[
            RubricStep(label="step_1", max_marks=5.0, keywords=["step_1"]),
            RubricStep(label="step_2", max_marks=5.0, keywords=["step_2"]),
        ],
        negative_marking=False,
    )


async def score_evaluation_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Evaluate AI results against rubrics and produce scores.

    Steps:
        1. Extract question results from the event
        2. For each question, look up rubric and evaluate
        3. Persist score events
        4. Publish score.updated
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")
    student_id = payload.get("student_id", "")
    question_results = payload.get("question_results", [])

    logger.info(
        "Processing ai.result event_id=%s exam=%s student=%s questions=%d",
        event_id, exam_id, student_id, len(question_results),
    )

    db = await db_manager.get_tenant_db(tenant_id)
    store = ScoreEventStore(db)

    for qr in question_results:
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
            "state": "ai_scored",
            "reason": "ai_inference",
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
        await store.append_event(tenant_id, score_event)

    # Publish aggregate notification
    await publish_score_updated(
        nats, exam_id, student_id,
        reason="ai_inference",
        lifecycle_state="scored",
        tenant_id=tenant_id,
    )

    logger.info(
        "Score evaluation complete event_id=%s questions=%d",
        event_id, len(question_results),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.ai.result with durable JetStream consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await score_evaluation_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.AI_RESULT,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered score_evaluation_consumer")
