"""Publish ai.result events to NATS JetStream.

Matches the ai.result.schema.json contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nats

    from src.domain.result_builder import AIResult

logger = logging.getLogger(__name__)

SUBJECT_AI_RESULT = "EXAMPEN.ai.result"


class ResultPublisher:
    """Publishes AIResult events to NATS."""

    def __init__(self, nc: nats.NATS) -> None:
        self._nc = nc

    async def publish(self, result: AIResult) -> None:
        """Serialize and publish the ai.result event.

        The event payload matches ai.result.schema.json:
        - event_id, event_type, event_version, occurred_at
        - exam_id, student_id, model_version
        - source_type
        - question_results[]: question_id, recognized_text, confidence, step_breakdown
        """
        payload = asdict(result)

        # Prune domain-only fields not in the event schema
        for qr in payload.get("question_results", []):
            qr.pop("content_type", None)
            qr.pop("flagged_for_review", None)

        js = self._nc.jetstream()
        await js.publish(
            SUBJECT_AI_RESULT,
            json.dumps(payload).encode(),
        )
        logger.info(
            "Published ai.result event_id=%s exam=%s",
            result.event_id,
            result.exam_id,
        )
