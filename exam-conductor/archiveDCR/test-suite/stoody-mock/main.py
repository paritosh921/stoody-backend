"""Stoody mock server — lightweight FastAPI app simulating the Stoody platform.

Endpoints served:
  GET  /.well-known/jwks.json             — RSA public keys
  GET  /api/users/{user_id}               — User profiles
  GET  /api/students                      — Student roster (by class + section)
  GET  /api/tutors                        — Tutor list (by subject)
  GET  /api/classes                       — Class list
  GET  /api/subjects                      — Subject list
  GET  /api/parents/{user_id}/children    — Parent-child relationships
  POST /api/webhooks/exampen/scores       — Score webhook (logs + verifies HMAC)
  POST /api/webhooks/exampen/exams        — Exam webhook (logs + verifies HMAC)

Run: ``uvicorn main:app --port 9100``
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from data import (
    CLASSES,
    PARENT_CHILDREN,
    STUDENTS_BY_CLASS,
    SUBJECTS,
    TUTORS_BY_SUBJECT,
    USERS,
)
from keys import get_jwks_dict, make_token

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("stoody-mock")

# HMAC secret for webhook signature verification (optional — skip if unset)
_WEBHOOK_SECRET = os.environ.get("STOODY_WEBHOOK_SECRET", "")
_SIGNATURE_HEADER = "X-ExamPen-Signature"

app = FastAPI(
    title="Stoody Mock Server",
    version="0.1.0",
    description="Simulates Stoody platform APIs for ExamPen development and testing.",
)

# ---------------------------------------------------------------------------
# In-memory webhook log
# ---------------------------------------------------------------------------

_webhook_log: list[dict[str, Any]] = []


def _verify_signature(body: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 signature if a secret is configured."""
    if not _WEBHOOK_SECRET:
        return True  # No secret configured — accept all
    expected = hmac.new(
        _WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# JWKS
# ---------------------------------------------------------------------------


@app.get("/.well-known/jwks.json")
async def jwks() -> dict[str, Any]:
    """Return RSA public keys in JWKS format."""
    return get_jwks_dict()


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------


@app.get("/api/users/{user_id}")
async def get_user(user_id: str) -> dict[str, Any]:
    """Return canned user profile."""
    user = USERS.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------


@app.get("/api/students")
async def list_students(
    class_id: str = Query(..., description="Class ID"),
    section_id: str = Query(..., description="Section ID"),
) -> dict[str, Any]:
    """Return student roster for a class+section."""
    key = f"{class_id}:{section_id}"
    students = STUDENTS_BY_CLASS.get(key, [])
    return {"students": students, "count": len(students)}


# ---------------------------------------------------------------------------
# Tutors
# ---------------------------------------------------------------------------


@app.get("/api/tutors")
async def list_tutors(
    subject_id: str = Query(..., description="Subject ID"),
) -> dict[str, Any]:
    """Return tutors for a subject."""
    tutors = TUTORS_BY_SUBJECT.get(subject_id, [])
    return {"tutors": tutors, "count": len(tutors)}


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


@app.get("/api/classes")
async def list_classes() -> dict[str, Any]:
    """Return all classes."""
    return {"classes": CLASSES, "count": len(CLASSES)}


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


@app.get("/api/subjects")
async def list_subjects() -> dict[str, Any]:
    """Return all subjects."""
    return {"subjects": SUBJECTS, "count": len(SUBJECTS)}


# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------


@app.get("/api/parents/{user_id}/children")
async def get_children(user_id: str) -> dict[str, Any]:
    """Return children linked to a parent."""
    children = PARENT_CHILDREN.get(user_id)
    if children is None:
        raise HTTPException(
            status_code=404, detail=f"No children found for parent {user_id}"
        )
    return {"children": children, "count": len(children)}


# ---------------------------------------------------------------------------
# Webhooks (ExamPen pushes to Stoody)
# ---------------------------------------------------------------------------


@app.post("/api/webhooks/exampen/scores")
async def webhook_scores(request: Request) -> dict[str, str]:
    """Accept score publication webhook. Verifies HMAC and logs payload."""
    body = await request.body()
    signature = request.headers.get(_SIGNATURE_HEADER, "")

    if not _verify_signature(body, signature):
        _log.warning("Webhook scores: HMAC verification FAILED")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    entry = {
        "type": "scores",
        "payload": payload,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "signature_valid": True,
    }
    _log.info("Webhook received: scores — %s", payload)
    _webhook_log.append(entry)
    return {"status": "accepted"}


@app.post("/api/webhooks/exampen/exams")
async def webhook_exams(request: Request) -> dict[str, str]:
    """Accept exam lifecycle webhook. Verifies HMAC and logs payload."""
    body = await request.body()
    signature = request.headers.get(_SIGNATURE_HEADER, "")

    if not _verify_signature(body, signature):
        _log.warning("Webhook exams: HMAC verification FAILED")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    entry = {
        "type": "exams",
        "payload": payload,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "signature_valid": True,
    }
    _log.info("Webhook received: exams — %s", payload)
    _webhook_log.append(entry)
    return {"status": "accepted"}


# ---------------------------------------------------------------------------
# Dev helpers
# ---------------------------------------------------------------------------


@app.get("/debug/webhooks")
async def debug_webhooks() -> dict[str, Any]:
    """Return all received webhooks (dev/debug only)."""
    return {"webhooks": _webhook_log, "count": len(_webhook_log)}


@app.delete("/debug/webhooks")
async def clear_webhooks() -> dict[str, str]:
    """Clear the webhook log (dev/debug only — useful for test isolation)."""
    _webhook_log.clear()
    return {"status": "cleared"}


@app.post("/debug/token")
async def debug_token(
    user_id: str = "tutor-001",
    role: str = "tutor",
    tenant_id: str = "tenant-001",
) -> dict[str, str]:
    """Generate a signed test JWT (dev/debug only)."""
    user = USERS.get(user_id, {})
    token = make_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        name=user.get("name", "Test User"),
        email=user.get("email", "test@stoody.local"),
    )
    return {"token": token}
