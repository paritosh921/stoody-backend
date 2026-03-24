"""Integration tests for objection routes — file -> assign -> resolve cycle.

Test IDs: I-REV-ROUTE-01 through I-REV-ROUTE-09

Mocks DB repo, NATS publisher, and auth dependency so routes can be
tested without a live database or message broker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from src.main import create_app


# -- Fixtures ----------------------------------------------------------------

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()

_EXAM_ID = str(uuid4())
_OBJECTION_ID = str(uuid4())


def _make_student_token() -> str:
    """Create a JWT for a student-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "student-001",
        "tenant_id": "tenant-abc",
        "role": "student",
        "name": "Test Student",
        "jti": f"jti-{uuid4()}",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid-1"})


def _make_evaluator_token() -> str:
    """Create a JWT for an evaluator-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "evaluator-001",
        "tenant_id": "tenant-abc",
        "role": "tutor",
        "name": "Test Evaluator",
        "jti": f"jti-{uuid4()}",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid-1"})


def _objection_detail(
    status: str = "filed",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a mock objection detail dict."""
    base = {
        "objection_id": _OBJECTION_ID,
        "exam_id": _EXAM_ID,
        "student_id": "student-001",
        "question_id": "q-1",
        "objection_text": "I believe my answer was marked incorrectly.",
        "status": status,
        "filed_at": datetime.now(timezone.utc).isoformat(),
        "assigned_to": None,
        "resolution": None,
        "resolution_reason": None,
        "score_delta": None,
    }
    base.update(overrides)
    return base


def _build_client() -> TestClient:
    """Build a TestClient with mocked dependencies."""
    app = create_app()

    # Mock JWKS
    jwks_mock = AsyncMock()
    jwks_mock.get_signing_key = AsyncMock(return_value=_PUBLIC_KEY)
    jwks_mock.warmup = AsyncMock()
    app.state.jwks_manager = jwks_mock
    app.state.db_engine = MagicMock()
    app.state.session_factory = MagicMock()

    # Mock objection repo
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=_objection_detail("filed"))
    repo.get_by_id = AsyncMock(return_value=_objection_detail("filed"))
    repo.list_objections = AsyncMock(return_value=[
        {
            "objection_id": _OBJECTION_ID,
            "exam_id": _EXAM_ID,
            "student_id": "student-001",
            "question_id": "q-1",
            "status": "filed",
            "filed_at": datetime.now(timezone.utc).isoformat(),
        }
    ])
    repo.exists_for_question = AsyncMock(return_value=False)
    repo.transition_state = AsyncMock(
        return_value=_objection_detail("assigned", assigned_to="evaluator-001"),
    )
    app.state.objection_repo = repo

    # Mock NATS publisher
    publisher = AsyncMock()
    publisher.publish_transition = AsyncMock()
    publisher.publish_rescore_command = AsyncMock()
    app.state.objection_publisher = publisher

    # Mock NATS client
    nats_mock = AsyncMock()
    nats_mock.connect = AsyncMock()
    nats_mock.close = AsyncMock()
    app.state.nats_client = nats_mock

    return TestClient(app, raise_server_exceptions=False)


# -- I-REV-ROUTE-01: File an objection ----------------------------------------


def test_file_objection_creates_and_returns_201():
    """I-REV-ROUTE-01: POST /objections returns 201 with objection detail."""
    client = _build_client()
    token = _make_student_token()
    resp = client.post(
        "/api/v1/objections",
        json={
            "exam_id": _EXAM_ID,
            "student_id": "student-001",
            "question_id": "q-1",
            "objection_text": "I believe my answer was marked incorrectly.",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "filed"
    assert body["exam_id"] == _EXAM_ID


# -- I-REV-ROUTE-02: List objections ------------------------------------------


def test_list_objections_returns_items():
    """I-REV-ROUTE-02: GET /objections returns list with items."""
    client = _build_client()
    token = _make_student_token()
    resp = client.get(
        "/api/v1/objections",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert len(body["items"]) == 1


# -- I-REV-ROUTE-03: Get objection detail -------------------------------------


def test_get_objection_detail():
    """I-REV-ROUTE-03: GET /objections/{id} returns full detail."""
    client = _build_client()
    token = _make_student_token()
    resp = client.get(
        f"/api/v1/objections/{_OBJECTION_ID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["objection_id"] == _OBJECTION_ID
    assert "objection_text" in body


# -- I-REV-ROUTE-04: Assign objection -----------------------------------------


def test_assign_objection():
    """I-REV-ROUTE-04: POST /objections/{id}/assign transitions to assigned."""
    client = _build_client()
    token = _make_evaluator_token()
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/assign",
        json={
            "actor_id": "evaluator-001",
            "assigned_to": "evaluator-001",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assigned_to"] == "evaluator-001"


# -- I-REV-ROUTE-05: Resolve objection (approve) ------------------------------


def test_resolve_objection_approve():
    """I-REV-ROUTE-05: POST /objections/{id}/resolve with approval triggers re-score."""
    client = _build_client()
    # Set the repo to return a reviewing-state objection
    reviewing_obj = _objection_detail("reviewing", assigned_to="evaluator-001")
    client.app.state.objection_repo.get_by_id = AsyncMock(return_value=reviewing_obj)
    client.app.state.objection_repo.transition_state = AsyncMock(
        return_value=_objection_detail(
            "resolved",
            resolution="approved",
            resolution_reason="Answer partially correct.",
            score_delta=8.5,
        ),
    )

    token = _make_evaluator_token()
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/resolve",
        json={
            "actor_id": "evaluator-001",
            "resolution": "approved",
            "reason": "Answer partially correct.",
            "new_score": 8.5,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolution"] == "approved"
    assert body["score_delta"] == 8.5

    # Verify re-score command was published
    client.app.state.objection_publisher.publish_rescore_command.assert_called_once()


# -- I-REV-ROUTE-06: Resolve objection (reject with reason) -------------------


def test_resolve_objection_reject():
    """I-REV-ROUTE-06: POST /objections/{id}/resolve with rejection requires reason."""
    client = _build_client()
    reviewing_obj = _objection_detail("reviewing", assigned_to="evaluator-001")
    client.app.state.objection_repo.get_by_id = AsyncMock(return_value=reviewing_obj)
    client.app.state.objection_repo.transition_state = AsyncMock(
        return_value=_objection_detail(
            "resolved",
            resolution="rejected",
            resolution_reason="Scoring is correct per rubric.",
        ),
    )

    token = _make_evaluator_token()
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/resolve",
        json={
            "actor_id": "evaluator-001",
            "resolution": "rejected",
            "reason": "Scoring is correct per rubric.",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolution"] == "rejected"

    # Re-score command should NOT be published for rejection
    client.app.state.objection_publisher.publish_rescore_command.assert_not_called()


# -- I-REV-ROUTE-07: Escalate objection ---------------------------------------


def test_escalate_objection():
    """I-REV-ROUTE-07: POST /objections/{id}/escalate moves to escalated."""
    client = _build_client()
    reviewing_obj = _objection_detail("reviewing", assigned_to="evaluator-001")
    client.app.state.objection_repo.get_by_id = AsyncMock(return_value=reviewing_obj)
    client.app.state.objection_repo.transition_state = AsyncMock(
        return_value=_objection_detail("escalated", assigned_to="hod"),
    )

    token = _make_evaluator_token()
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/escalate",
        json={
            "actor_id": "evaluator-001",
            "escalated_to": "hod",
            "reason": "Need department head review for this edge case.",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "escalated"


# -- I-REV-ROUTE-08: Reject without reason fails ------------------------------


def test_resolve_reject_without_reason_fails():
    """I-REV-ROUTE-08: Reject with reason shorter than 5 chars returns 422."""
    client = _build_client()
    token = _make_evaluator_token()
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/resolve",
        json={
            "actor_id": "evaluator-001",
            "resolution": "rejected",
            "reason": "No",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    # Pydantic validation catches min_length=5 before domain rules
    assert resp.status_code == 422


# -- I-REV-ROUTE-09: Full lifecycle: file -> assign -> resolve -----------------


def test_full_lifecycle():
    """I-REV-ROUTE-09: Complete cycle — file, assign, resolve."""
    client = _build_client()
    student_token = _make_student_token()
    evaluator_token = _make_evaluator_token()

    # Step 1: File
    resp = client.post(
        "/api/v1/objections",
        json={
            "exam_id": _EXAM_ID,
            "student_id": "student-001",
            "question_id": "q-1",
            "objection_text": "I believe my answer was marked incorrectly.",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert resp.status_code == 201

    # Step 2: Assign
    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/assign",
        json={
            "actor_id": "evaluator-001",
            "assigned_to": "evaluator-001",
        },
        headers={"Authorization": f"Bearer {evaluator_token}"},
    )
    assert resp.status_code == 200

    # Step 3: Resolve (set state to reviewing first)
    reviewing_obj = _objection_detail("reviewing", assigned_to="evaluator-001")
    client.app.state.objection_repo.get_by_id = AsyncMock(return_value=reviewing_obj)
    client.app.state.objection_repo.transition_state = AsyncMock(
        return_value=_objection_detail(
            "resolved",
            resolution="approved",
            resolution_reason="Re-evaluated: additional marks awarded.",
            score_delta=9.0,
        ),
    )

    resp = client.post(
        f"/api/v1/objections/{_OBJECTION_ID}/resolve",
        json={
            "actor_id": "evaluator-001",
            "resolution": "approved",
            "reason": "Re-evaluated: additional marks awarded.",
            "new_score": 9.0,
        },
        headers={"Authorization": f"Bearer {evaluator_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["resolution"] == "approved"
    assert body["score_delta"] == 9.0

    # Verify re-score command was published
    client.app.state.objection_publisher.publish_rescore_command.assert_called_once()
