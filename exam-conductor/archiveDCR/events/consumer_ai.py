"""NATS consumer: AI inference pipeline.

Subscribes to ``EXAMPEN.page.ready``, runs AI inference (mock for now),
writes results to ``ai_result_repo``, and publishes ``EXAMPEN.ai.result``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import uuid4

from ..storage.ai_result_repo import AIResultRepo
from . import subjects
from .publishers import publish_ai_result

logger = logging.getLogger(__name__)

DURABLE = "exampen-ai-inference"
QUEUE_GROUP = "exampen-ai-workers"


def _mock_inference(
    exam_id: str,
    student_id: str,
    page_number: int,
    s3_uri: str,
) -> List[Dict[str, Any]]:
    """Produce mock AI question results.

    In production this would call an ML model endpoint with the page
    image at *s3_uri*.  For now it returns a single placeholder result
    to allow the downstream score pipeline to function end-to-end.
    """
    return [
        {
            "question_id": f"{exam_id}_q{page_number}_1",
            "recognized_text": "[mock recognized text]",
            "confidence": 0.85,
            "step_breakdown": ["step_1_placeholder", "step_2_placeholder"],
            "model_version": "mock-v0.1",
        }
    ]


async def ai_inference_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Run AI inference on a ready page.

    Steps:
        1. Extract page metadata from the event
        2. Run inference (mock)
        3. Persist AI results
        4. Publish ai.result event
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")
    student_id = payload.get("student_id", "")
    page_number = payload.get("page_number", 0)
    s3_uri = payload.get("s3_uri", "")

    logger.info(
        "Processing page.ready event_id=%s exam=%s student=%s page=%d",
        event_id, exam_id, student_id, page_number,
    )

    # 1. Run AI inference (mock)
    question_results = _mock_inference(exam_id, student_id, page_number, s3_uri)

    # 2. Persist to MongoDB
    db = await db_manager.get_tenant_db(tenant_id)
    repo = AIResultRepo(db)

    result_doc = {
        "exam_id": exam_id,
        "student_id": student_id,
        "page_number": page_number,
        "s3_uri": s3_uri,
        "question_results": question_results,
        "inference_id": uuid4().hex,
    }
    await repo.store_result(tenant_id, result_doc)

    # 3. Publish downstream
    await publish_ai_result(
        nats, exam_id, student_id, question_results, tenant_id=tenant_id,
    )

    logger.info(
        "AI inference complete event_id=%s questions=%d",
        event_id, len(question_results),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.page.ready with durable JetStream consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await ai_inference_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.PAGE_READY,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered ai_inference_consumer")
