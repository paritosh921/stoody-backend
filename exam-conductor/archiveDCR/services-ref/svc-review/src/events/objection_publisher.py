"""NATS event publisher for objection lifecycle transitions.

Publishes an ``objection`` event on every state transition (matches
``contracts/events/objection.schema.json``).

On resolution with ``approved``, publishes a re-score command to
``svc-score-engine`` via NATS.  svc-review does NOT modify scores
directly (per STATE_OWNERSHIP_MAP.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

_log = get_logger(__name__)

# NATS subjects — EXAMPEN. prefix matches the JetStream stream name.
# Objection events include the action as sub-token: EXAMPEN.objection.{action}
# so that consumers can subscribe to EXAMPEN.objection.* for all actions.
OBJECTION_SUBJECT_PREFIX = "EXAMPEN.objection"
RESCORE_SUBJECT = "EXAMPEN.score.rescore_command"


class ObjectionPublisher:
    """Publishes objection events and re-score commands to NATS."""

    def __init__(self, nats: NatsClient) -> None:
        self._nats = nats

    async def publish_transition(
        self,
        *,
        objection_id: str,
        exam_id: str,
        student_id: str,
        question_id: str,
        action: str,
        state: str,
        actor_id: str | None = None,
    ) -> None:
        """Publish an objection event matching the schema contract."""
        event: dict[str, Any] = {
            "event_id": str(uuid4()),
            "event_type": "objection",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "objection_id": objection_id,
            "student_id": student_id,
            "question_id": question_id,
            "action": action,
            "state": state,
        }
        if actor_id is not None:
            event["actor_id"] = actor_id

        subject = f"{OBJECTION_SUBJECT_PREFIX}.{action}"
        await self._nats.publish(subject, event)
        _log.info(
            "Published objection event id=%s subject=%s state=%s",
            objection_id, subject, state,
        )

    async def publish_rescore_command(
        self,
        *,
        objection_id: str,
        exam_id: str,
        student_id: str,
        question_id: str,
        new_score: float,
        actor_id: str,
    ) -> None:
        """Publish a re-score command for svc-score-engine.

        This is triggered only when an objection is approved.
        svc-review does NOT write scores directly.
        """
        command: dict[str, Any] = {
            "event_id": str(uuid4()),
            "event_type": "rescore_command",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "source": "svc-review",
            "objection_id": objection_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
            "new_score": new_score,
            "actor_id": actor_id,
        }

        await self._nats.publish(RESCORE_SUBJECT, command)
        _log.info(
            "Published rescore command objection=%s exam=%s question=%s new_score=%s",
            objection_id, exam_id, question_id, new_score,
        )
