from __future__ import annotations

from datetime import datetime, timezone

from services.exampen_copy_state import (
    canonical_copy_state,
    processing_job_projection,
    submission_queue_bucket,
)


def test_provider_states_share_one_stable_checking_contract():
    for raw_status in ("provider_processing", "provider_finalizing"):
        projection = processing_job_projection(
            {
                "status": raw_status,
                "processing_mode": "economy",
                "provider_batch_status": "finalizing",
                "provider_phase": "grading",
                "stage_number": 2,
                "stage_count": 2,
                "provider_expires_at": 1_788_200_000,
            }
        )
        assert projection["copy_state"] == "checking"
        assert projection["checking_mode"] == "economy"
        assert projection["stage_number"] == 2
        assert projection["stage_count"] == 2
        assert projection["deadline_at"].endswith("+00:00")


def test_retryable_worker_state_is_active_only_while_a_retry_is_scheduled():
    assert canonical_copy_state(
        {"status": "retryable_error", "next_retry_at": datetime.now(timezone.utc)}
    ) == "queued"
    assert canonical_copy_state({"status": "retryable_error"}) == "failed"


def test_batch_failure_is_blocked_but_active_economy_work_is_pending():
    common = {
        "pcr_ready": False,
        "dcr_complete": True,
        "publication_status": "",
        "needs_teacher_review": False,
        "has_unresolved_blocking": False,
    }
    assert submission_queue_bucket(
        **common,
        processing_job={"status": "provider_finalizing"},
    ) == "pending"
    assert submission_queue_bucket(
        **common,
        processing_job={"status": "batch_failed"},
    ) == "blocked"


def test_completed_job_without_canonical_readiness_fails_closed():
    assert submission_queue_bucket(
        pcr_ready=False,
        dcr_complete=True,
        publication_status="",
        needs_teacher_review=False,
        has_unresolved_blocking=False,
        processing_job={"status": "completed"},
    ) == "blocked"
