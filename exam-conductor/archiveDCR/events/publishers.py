"""Event publishing helpers for the ExamPen DCR pipeline.

Each helper builds a self-describing event envelope and publishes it
to the appropriate NATS subject.  If the ``nats`` client is ``None``
(e.g. running without NATS in dev), the function logs a warning and
returns gracefully.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from . import subjects

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope builder
# ---------------------------------------------------------------------------

def _envelope(
    event_type: str,
    event_version: int,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap *payload* in a standard event envelope."""
    return {
        "event_id": uuid4().hex,
        "event_type": event_type,
        "event_version": event_version,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


async def _safe_publish(
    nats: Any,
    subject: str,
    data: Dict[str, Any],
) -> None:
    """Publish to *subject* or log a warning when *nats* is unavailable."""
    if nats is None:
        logger.warning(
            "NATS not available — skipping publish to %s (event_id=%s)",
            subject,
            data.get("event_id", "?"),
        )
        return
    await nats.publish(subject, data)


# ---------------------------------------------------------------------------
# Public publishers
# ---------------------------------------------------------------------------

async def publish_exam_lifecycle(
    nats: Any,
    exam_id: str,
    from_state: str,
    to_state: str,
    actor_id: str,
    *,
    tenant_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish an exam lifecycle state transition event."""
    data = _envelope("exam.lifecycle", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "from_state": from_state,
        "to_state": to_state,
        "actor_id": actor_id,
        "metadata": metadata or {},
    })
    await _safe_publish(nats, subjects.EXAM_LIFECYCLE, data)


async def publish_stroke_raw(
    nats: Any,
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
    data: Dict[str, Any],
    *,
    tenant_id: str = "",
) -> None:
    """Publish a raw stroke chunk for processing."""
    event = _envelope("stroke.raw", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "pen_mac": pen_mac,
        "chunk_index": chunk_index,
        "stroke_data": data,
    })
    await _safe_publish(nats, subjects.STROKE_RAW, event)


async def publish_stroke_processed(
    nats: Any,
    exam_id: str,
    pen_mac: str,
    page_assignments: List[Dict[str, Any]],
    *,
    tenant_id: str = "",
) -> None:
    """Publish processed stroke data with page/question assignments."""
    event = _envelope("stroke.processed", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "pen_mac": pen_mac,
        "page_assignments": page_assignments,
    })
    await _safe_publish(nats, subjects.STROKE_PROCESSED, event)


async def publish_page_ready(
    nats: Any,
    exam_id: str,
    student_id: str,
    page_number: int,
    s3_uri: str,
    *,
    tenant_id: str = "",
) -> None:
    """Publish that a page image is assembled and ready for AI inference."""
    event = _envelope("page.ready", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "student_id": student_id,
        "page_number": page_number,
        "s3_uri": s3_uri,
    })
    await _safe_publish(nats, subjects.PAGE_READY, event)


async def publish_ai_result(
    nats: Any,
    exam_id: str,
    student_id: str,
    question_results: List[Dict[str, Any]],
    *,
    tenant_id: str = "",
) -> None:
    """Publish AI inference results for a student's page."""
    event = _envelope("ai.result", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "student_id": student_id,
        "question_results": question_results,
    })
    await _safe_publish(nats, subjects.AI_RESULT, event)


async def publish_score_updated(
    nats: Any,
    exam_id: str,
    student_id: str,
    reason: str,
    lifecycle_state: str,
    *,
    tenant_id: str = "",
) -> None:
    """Publish that a student's score has been updated."""
    event = _envelope("score.updated", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "student_id": student_id,
        "reason": reason,
        "lifecycle_state": lifecycle_state,
    })
    await _safe_publish(nats, subjects.SCORE_UPDATED, event)


async def publish_objection(
    nats: Any,
    objection_id: str,
    exam_id: str,
    action: str,
    state: str,
    *,
    tenant_id: str = "",
) -> None:
    """Publish an objection lifecycle event (filed, reviewed, resolved)."""
    event = _envelope("objection", 1, {
        "objection_id": objection_id,
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "action": action,
        "state": state,
    })
    await _safe_publish(nats, subjects.OBJECTION, event)
