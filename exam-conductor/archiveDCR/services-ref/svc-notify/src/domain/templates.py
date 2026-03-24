"""Pure template rendering for notification content.

ZERO I/O — this module must never import asyncio, aiohttp, sqlalchemy,
nats, or any I/O library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderedNotification:
    """Rendered notification ready for dispatch."""

    subject: str
    body_text: str
    body_html: str


# ---------------------------------------------------------------------------
# Template renderers — one per template_name
# ---------------------------------------------------------------------------

def _render_score_published(data: dict[str, Any]) -> RenderedNotification:
    exam_id = data.get("exam_id", "unknown")
    total = data.get("total_score", "N/A")

    subject = "Your exam score has been published"
    body_text = (
        f"Your score for exam {exam_id} has been published.\n"
        f"Total score: {total}\n\n"
        "Log in to the ExamPen student portal to view the full breakdown."
    )
    body_html = (
        f"<h2>Score Published</h2>"
        f"<p>Your score for exam <strong>{exam_id}</strong> has been published.</p>"
        f"<p>Total score: <strong>{total}</strong></p>"
        f"<p>Log in to the ExamPen student portal to view the full breakdown.</p>"
    )
    return RenderedNotification(subject=subject, body_text=body_text, body_html=body_html)


def _render_objection_filed(data: dict[str, Any]) -> RenderedNotification:
    objection_id = data.get("objection_id", "unknown")
    exam_id = data.get("exam_id", "unknown")
    question_id = data.get("question_id", "unknown")

    subject = f"New objection filed — exam {exam_id}"
    body_text = (
        f"A student has filed objection {objection_id} "
        f"for question {question_id} in exam {exam_id}.\n\n"
        "Please review the objection in the teacher dashboard."
    )
    body_html = (
        f"<h2>New Objection Filed</h2>"
        f"<p>A student has filed objection <strong>{objection_id}</strong> "
        f"for question <strong>{question_id}</strong> in exam "
        f"<strong>{exam_id}</strong>.</p>"
        f"<p>Please review the objection in the teacher dashboard.</p>"
    )
    return RenderedNotification(subject=subject, body_text=body_text, body_html=body_html)


def _render_objection_resolved(data: dict[str, Any]) -> RenderedNotification:
    objection_id = data.get("objection_id", "unknown")
    exam_id = data.get("exam_id", "unknown")
    question_id = data.get("question_id", "unknown")

    subject = f"Your objection has been resolved — exam {exam_id}"
    body_text = (
        f"Objection {objection_id} for question {question_id} "
        f"in exam {exam_id} has been resolved.\n\n"
        "Log in to the ExamPen student portal to see the outcome."
    )
    body_html = (
        f"<h2>Objection Resolved</h2>"
        f"<p>Objection <strong>{objection_id}</strong> for question "
        f"<strong>{question_id}</strong> in exam "
        f"<strong>{exam_id}</strong> has been resolved.</p>"
        f"<p>Log in to the ExamPen student portal to see the outcome.</p>"
    )
    return RenderedNotification(subject=subject, body_text=body_text, body_html=body_html)


def _render_exam_reminder(data: dict[str, Any]) -> RenderedNotification:
    exam_id = data.get("exam_id", "unknown")

    subject = "Exam reminder — your exam is about to begin"
    body_text = (
        f"Exam {exam_id} has been armed and is about to begin.\n\n"
        "Please ensure your pen is ready and you are seated at your desk."
    )
    body_html = (
        f"<h2>Exam Reminder</h2>"
        f"<p>Exam <strong>{exam_id}</strong> has been armed and is about to begin.</p>"
        f"<p>Please ensure your pen is ready and you are seated at your desk.</p>"
    )
    return RenderedNotification(subject=subject, body_text=body_text, body_html=body_html)


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------

_TEMPLATE_TABLE: dict[str, Any] = {
    "score_published": _render_score_published,
    "objection_filed": _render_objection_filed,
    "objection_resolved": _render_objection_resolved,
    "exam_reminder": _render_exam_reminder,
}


def render_template(
    template_name: str,
    data: dict[str, Any],
) -> RenderedNotification:
    """Render a notification template.

    Raises ``KeyError`` for unknown template names — this indicates a
    programming error in ``trigger_rules.py``.
    """
    renderer = _TEMPLATE_TABLE.get(template_name)
    if renderer is None:
        raise KeyError(f"Unknown notification template: {template_name}")
    return renderer(data)
