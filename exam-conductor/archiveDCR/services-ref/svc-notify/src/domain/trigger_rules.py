"""Pure mapping from NATS events to notification actions.

ZERO I/O — this module must never import asyncio, aiohttp, sqlalchemy,
nats, or any I/O library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Channel enum
# ---------------------------------------------------------------------------

class NotificationChannel(str, Enum):
    """Supported notification delivery channels."""

    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NotificationAction:
    """A single notification to dispatch."""

    recipient_id: str
    channel: str  # "email" | "push" | "sms" | "webhook"
    template_name: str
    template_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WebhookAction:
    """Webhook-specific notification action.

    Unlike template-based channels, webhooks carry structured data and a
    *webhook_type* that tells the dispatcher which sender method to call.
    """

    channel: str  # always "webhook"
    webhook_type: str  # "score" | "exam_created" | "exam_completed"
    webhook_data: dict[str, Any] = field(default_factory=dict)

    # Satisfy the duck-type contract the dispatcher uses:
    @property
    def recipient_id(self) -> str:
        return "stoody"

    @property
    def template_name(self) -> str:
        return f"webhook:{self.webhook_type}"

    @property
    def template_data(self) -> dict[str, Any]:
        return self.webhook_data


# ---------------------------------------------------------------------------
# Rule functions — one per event type
# ---------------------------------------------------------------------------

def _rules_score_updated(
    payload: dict[str, Any],
) -> list[NotificationAction | WebhookAction]:
    """score.updated (reason=published) -> notify student + Stoody webhook."""
    if payload.get("reason") != "published":
        return []

    student_id = payload.get("student_id", "")
    if not student_id:
        return []

    data = {
        "exam_id": payload.get("exam_id", ""),
        "student_id": student_id,
        "total_score": payload.get("total_score"),
    }

    actions: list[NotificationAction | WebhookAction] = [
        NotificationAction(
            recipient_id=student_id,
            channel=NotificationChannel.EMAIL,
            template_name="score_published",
            template_data=data,
        ),
        NotificationAction(
            recipient_id=student_id,
            channel=NotificationChannel.PUSH,
            template_name="score_published",
            template_data=data,
        ),
    ]

    # Stoody webhook — aggregate scores list with the single student
    scores = payload.get("scores")
    if scores is None:
        # Build a minimal scores entry from flat fields
        scores = [{
            "student_id": student_id,
            "total": payload.get("total_score", 0),
            "percentage": payload.get("percentage", 0),
            "percentile": payload.get("percentile", 0),
        }]

    actions.append(
        WebhookAction(
            channel=NotificationChannel.WEBHOOK,
            webhook_type="score",
            webhook_data={
                "exam_id": payload.get("exam_id", ""),
                "scores": scores,
            },
        )
    )

    return actions


def _rules_objection(
    payload: dict[str, Any],
) -> list[NotificationAction | WebhookAction]:
    """objection events:
    - state=filed  -> notify assigned evaluator
    - state=resolved -> notify student
    """
    state = payload.get("state", "")
    actions: list[NotificationAction | WebhookAction] = []

    data = {
        "exam_id": payload.get("exam_id", ""),
        "objection_id": payload.get("objection_id", ""),
        "student_id": payload.get("student_id", ""),
        "question_id": payload.get("question_id", ""),
    }

    if state == "filed":
        actor_id = payload.get("actor_id", "")
        # When an objection is filed the evaluator to notify is carried in
        # actor_id (the person who will handle it). If not present, fall back
        # to a generic "evaluator" recipient for upstream resolution.
        evaluator_id = actor_id or "evaluator:unassigned"
        actions.append(
            NotificationAction(
                recipient_id=evaluator_id,
                channel=NotificationChannel.EMAIL,
                template_name="objection_filed",
                template_data=data,
            )
        )
    elif state == "resolved":
        student_id = payload.get("student_id", "")
        if student_id:
            actions.append(
                NotificationAction(
                    recipient_id=student_id,
                    channel=NotificationChannel.EMAIL,
                    template_name="objection_resolved",
                    template_data=data,
                )
            )
            actions.append(
                NotificationAction(
                    recipient_id=student_id,
                    channel=NotificationChannel.PUSH,
                    template_name="objection_resolved",
                    template_data=data,
                )
            )

    return actions


def _rules_exam_lifecycle(
    payload: dict[str, Any],
) -> list[NotificationAction | WebhookAction]:
    """exam.lifecycle:
    - to_state=armed          -> notify students with exam reminder
    - to_state=created        -> Stoody webhook (exam created)
    - to_state=upload_complete -> Stoody webhook (exam completed)
    """
    to_state = payload.get("to_state", "")
    exam_id = payload.get("exam_id", "")
    actions: list[NotificationAction | WebhookAction] = []

    if to_state == "armed":
        data = {
            "exam_id": exam_id,
            "from_state": payload.get("from_state", ""),
            "to_state": "armed",
        }
        actions.extend([
            NotificationAction(
                recipient_id=f"exam:{exam_id}:students",
                channel=NotificationChannel.EMAIL,
                template_name="exam_reminder",
                template_data=data,
            ),
            NotificationAction(
                recipient_id=f"exam:{exam_id}:students",
                channel=NotificationChannel.PUSH,
                template_name="exam_reminder",
                template_data=data,
            ),
        ])

    elif to_state == "created":
        actions.append(
            WebhookAction(
                channel=NotificationChannel.WEBHOOK,
                webhook_type="exam_created",
                webhook_data={
                    "exam_id": exam_id,
                    "subject_id": payload.get("subject_id", ""),
                    "class_id": payload.get("class_id", ""),
                    "date": payload.get("date", ""),
                    "duration": payload.get("duration", 0),
                },
            )
        )

    elif to_state == "upload_complete":
        actions.append(
            WebhookAction(
                channel=NotificationChannel.WEBHOOK,
                webhook_type="exam_completed",
                webhook_data={
                    "exam_id": exam_id,
                    "pens_synced": payload.get("pens_synced", 0),
                    "upload_status": payload.get("upload_status", ""),
                },
            )
        )

    return actions


# ---------------------------------------------------------------------------
# Dispatcher table
# ---------------------------------------------------------------------------

_RULE_TABLE: dict[str, Any] = {
    "score.updated": _rules_score_updated,
    "objection": _rules_objection,
    "exam.lifecycle": _rules_exam_lifecycle,
}


def determine_notifications(
    event_type: str,
    payload: dict[str, Any],
) -> list[NotificationAction | WebhookAction]:
    """Return the list of notifications to send for a given event.

    Unknown event types return an empty list (no crash, no log — the caller
    should never subscribe to events that have no rules).
    """
    rule_fn = _RULE_TABLE.get(event_type)
    if rule_fn is None:
        return []
    return rule_fn(payload)
