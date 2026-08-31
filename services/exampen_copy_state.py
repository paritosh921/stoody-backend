"""Stable teacher-facing state for ExamPen answer-copy processing.

Provider and worker status strings are implementation details.  Every API that
reports an answer copy must project them through this module so the roster,
queue, counters, and review workspace cannot disagree during a rolling deploy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional


COPY_STATE_VERSION = 1

ACTIVE_COPY_STATES = frozenset({"queued", "checking", "importing"})
TERMINAL_COPY_STATES = frozenset({"ready", "needs_review", "failed"})

_QUEUED_JOB_STATUSES = frozenset(
    {
        "queued",
        "waiting_for_batch",
        "batch_queued",
        "preparing_batch",
        "grading_contract_migration_pending",
        "queued_pipeline_v3",
        "queued_pipeline_v5",
        "queued_pipeline_v6",
        "queued_pipeline_v7",
    }
)
_CHECKING_JOB_STATUSES = frozenset(
    {"processing", "provider_processing", "provider_finalizing"}
)
_IMPORTING_JOB_STATUSES = frozenset({"importing_batch"})
_FAILED_JOB_STATUSES = frozenset(
    {"batch_failed", "failed", "enqueue_failed", "not_enqueued"}
)
_ECONOMY_JOB_STATUSES = frozenset(
    {
        "waiting_for_batch",
        "batch_queued",
        "preparing_batch",
        "provider_processing",
        "provider_finalizing",
        "importing_batch",
        "batch_failed",
    }
)


def canonical_copy_state(job: Optional[Mapping[str, Any]]) -> str:
    """Return the versioned product state for a processing-job document."""

    if not job:
        return "queued"
    status = str(job.get("status") or "").strip().lower()
    if status in _QUEUED_JOB_STATUSES:
        return "queued"
    if status == "retryable_error":
        return "queued" if job.get("next_retry_at") is not None else "failed"
    if status in _CHECKING_JOB_STATUSES:
        return "checking"
    if status in _IMPORTING_JOB_STATUSES:
        return "importing"
    if status == "completed":
        return "ready"
    if status == "blocked_for_review":
        return "needs_review"
    if status in _FAILED_JOB_STATUSES:
        return "failed"
    # Unknown persisted states fail closed.  Treating them as active would leave
    # a copy spinning forever without an owner.
    return "failed"


def checking_mode(job: Optional[Mapping[str, Any]]) -> str:
    """Return the teacher-facing pricing lane, independent of provider state."""

    if not job:
        return "standard"
    status = str(job.get("status") or "").strip().lower()
    mode = str(job.get("processing_mode") or "").strip().lower()
    if mode == "economy" or status in _ECONOMY_JOB_STATUSES:
        return "economy"
    return "standard"


def processing_job_projection(job: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the common public processing contract used by all read APIs."""

    value = job or {}
    state = canonical_copy_state(value)
    retryable = bool(value.get("retryable", True))
    deadline = value.get("provider_expires_at")
    if isinstance(deadline, (int, float)):
        deadline = datetime.fromtimestamp(float(deadline), tz=timezone.utc).isoformat()
    elif hasattr(deadline, "isoformat"):
        deadline = deadline.isoformat()
    elif deadline is not None:
        deadline = str(deadline)
    return {
        "state_version": COPY_STATE_VERSION,
        "copy_state": state,
        "checking_mode": checking_mode(value),
        "provider_status": value.get("provider_batch_status"),
        "provider_phase": value.get("provider_phase"),
        "stage_number": max(0, int(value.get("stage_number") or 0)),
        "stage_count": max(0, int(value.get("stage_count") or 0)),
        "deadline_at": deadline,
        "failure_code": value.get("failure_code"),
        "retryable": retryable,
        "operator_action": value.get("operator_action"),
        "can_retry": state == "failed" and retryable,
    }


def submission_queue_bucket(
    *,
    pcr_ready: bool,
    dcr_complete: bool,
    publication_status: str,
    needs_teacher_review: bool,
    has_unresolved_blocking: bool,
    processing_job: Optional[Mapping[str, Any]],
) -> str:
    """Return the single authoritative queue bucket for one submission."""

    state = canonical_copy_state(processing_job)
    all_evaluated = pcr_ready and dcr_complete
    if has_unresolved_blocking or state == "failed":
        return "blocked"
    if all_evaluated and str(publication_status or "").lower() == "ready":
        return "ready_to_publish"
    if needs_teacher_review or state == "needs_review":
        return "needs_review"
    if state in ACTIVE_COPY_STATES:
        return "pending"
    if not all_evaluated:
        # A worker claiming completion without canonical readiness is a terminal
        # integrity problem, not work that is still running.
        return "blocked" if state == "ready" else "pending"
    return "ready_to_publish"


__all__ = [
    "ACTIVE_COPY_STATES",
    "COPY_STATE_VERSION",
    "TERMINAL_COPY_STATES",
    "canonical_copy_state",
    "checking_mode",
    "processing_job_projection",
    "submission_queue_bucket",
]
