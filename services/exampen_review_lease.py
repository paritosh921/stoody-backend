"""Fenced submission-level lease for teacher review mutations.

Publishing, reprocessing, and response ownership correction all change the
meaning of one answer copy.  They must never run concurrently even though
their data lives in several Mongo collections.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


REVIEW_MUTATION_LEASE_MINUTES = 10


class SubmissionReviewBusyError(RuntimeError):
    """Raised when another teacher operation owns the answer copy."""


async def acquire_submission_review_lease(
    tenant_db: Any,
    submission_id: str,
    *,
    actor_id: str,
    operation: str,
) -> str:
    """Acquire one recoverable, fenced lease and return its opaque token."""

    token = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    result = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "$or": [
                {"review_mutation_lease_expires_at": {"$exists": False}},
                {"review_mutation_lease_expires_at": None},
                {"review_mutation_lease_expires_at": {"$lte": now}},
            ],
        },
        {
            "$set": {
                "review_mutation_lease_token": token,
                "review_mutation_lease_expires_at": now
                + timedelta(minutes=REVIEW_MUTATION_LEASE_MINUTES),
                "review_mutation_lease_actor_id": str(actor_id or "unknown"),
                "review_mutation_lease_operation": str(operation or "review"),
                "review_mutation_lease_started_at": now,
            }
        },
    )
    if result.matched_count == 1:
        return token

    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id},
        {"_id": 1},
    )
    if submission is None:
        raise ValueError(f"Submission {submission_id} not found")
    raise SubmissionReviewBusyError(
        "Another review, reprocess, or publish operation is already in progress "
        "for this answer copy"
    )


async def release_submission_review_lease(
    tenant_db: Any,
    submission_id: str,
    token: str,
) -> bool:
    """Release only the lease token owned by this caller."""

    result = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "review_mutation_lease_token": token,
        },
        {
            "$unset": {
                "review_mutation_lease_token": "",
                "review_mutation_lease_expires_at": "",
                "review_mutation_lease_actor_id": "",
                "review_mutation_lease_operation": "",
                "review_mutation_lease_started_at": "",
            }
        },
    )
    return result.matched_count == 1
