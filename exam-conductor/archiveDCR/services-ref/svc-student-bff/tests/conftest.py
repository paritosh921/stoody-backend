"""Shared fixtures for svc-student-bff tests.

Provides JWT helpers, mock adapter clients, and a pre-configured
TestClient with all dependencies mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from src.main import create_app

# -- Crypto setup ----------------------------------------------------------

PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = PRIVATE_KEY.public_key()

# -- Constants -------------------------------------------------------------

EXAM_ID = str(uuid4())
QUESTION_ID = "q-1"
OBJECTION_ID = str(uuid4())
TEACHER_ID = "teacher-001"
STUDENT_ID = "student-001"
PARENT_ID = "parent-001"
CHILD_STUDENT_ID = "student-child-001"


# -- Token factories -------------------------------------------------------


def make_student_token(
    user_id: str = STUDENT_ID,
    tenant_id: str = "tenant-abc",
) -> str:
    """Create a JWT for a student-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": "student",
        "name": "Test Student",
        "jti": f"jti-{uuid4()}",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(
        payload, PRIVATE_KEY, algorithm="RS256",
        headers={"kid": "test-kid-1"},
    )


def make_parent_token(
    user_id: str = PARENT_ID,
    tenant_id: str = "tenant-abc",
) -> str:
    """Create a JWT for a parent-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": "parent",
        "name": "Test Parent",
        "jti": f"jti-{uuid4()}",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(
        payload, PRIVATE_KEY, algorithm="RS256",
        headers={"kid": "test-kid-1"},
    )


def make_teacher_token(
    user_id: str = TEACHER_ID,
    tenant_id: str = "tenant-abc",
) -> str:
    """Create a JWT for a teacher/tutor-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": "tutor",
        "name": "Test Teacher",
        "jti": f"jti-{uuid4()}",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(
        payload, PRIVATE_KEY, algorithm="RS256",
        headers={"kid": "test-kid-1"},
    )


# -- Mock data factories ---------------------------------------------------


def mock_score_summary(
    exam_id: str = EXAM_ID,
) -> dict[str, Any]:
    """Score summary response from svc-score-engine."""
    return {
        "exam_id": exam_id,
        "total_score": 78.5,
        "percentage": 78.5,
        "percentile": 85.0,
        "pass_fail": "pass",
        "questions": [
            {
                "question_id": "q-1",
                "marks_obtained": 8.5,
                "max_marks": 10,
                "ai_confidence": 0.92,
                "miss_indicator": None,
            },
            {
                "question_id": "q-2",
                "marks_obtained": 7.0,
                "max_marks": 10,
                "ai_confidence": 0.88,
                "miss_indicator": None,
            },
        ],
    }


def mock_objection(
    status: str = "filed",
) -> dict[str, Any]:
    """Single objection response from svc-review."""
    return {
        "objection_id": OBJECTION_ID,
        "exam_id": EXAM_ID,
        "student_id": STUDENT_ID,
        "question_id": QUESTION_ID,
        "status": status,
        "objection_text": "I believe my answer was correct.",
        "resolution_reason": None,
        "new_score": None,
    }


# -- Client builder --------------------------------------------------------


def build_client(
    parent_children: list[str] | None = None,
) -> TestClient:
    """Build a TestClient with all backing-service mocks wired in."""
    app = create_app()

    # Mock JWKS
    jwks_mock = AsyncMock()
    jwks_mock.get_signing_key = AsyncMock(return_value=PUBLIC_KEY)
    jwks_mock.warmup = AsyncMock()
    app.state.jwks_manager = jwks_mock

    # Mock Stoody client (parent-child resolution)
    stoody_mock = AsyncMock()
    stoody_mock.get_parent_children = AsyncMock(
        return_value=parent_children or [CHILD_STUDENT_ID],
    )
    app.state.stoody_client = stoody_mock

    # Mock score client
    score_mock = AsyncMock()
    score_mock.get_score_summary = AsyncMock(
        return_value=mock_score_summary(),
    )
    score_mock.get_question_breakdown = AsyncMock(
        return_value=[
            {"question_id": "q-1", "marks_obtained": 8.5, "max_marks": 10},
            {"question_id": "q-2", "marks_obtained": 7.0, "max_marks": 10},
        ],
    )
    score_mock.get_answer_insight = AsyncMock(
        return_value={
            "question_id": QUESTION_ID,
            "answer_image_uri": "https://cdn.example.com/answers/q1.png",
            "recognized_text": "The answer is 42.",
            "confidence": 0.95,
            "step_breakdown": ["Step 1: Setup", "Step 2: Solve"],
            "feedback": "Well done.",
        },
    )
    app.state.score_client = score_mock

    # Mock review client
    review_mock = AsyncMock()
    review_mock.list_objections = AsyncMock(
        return_value=[mock_objection()],
    )
    review_mock.get_objection = AsyncMock(
        return_value=mock_objection(),
    )
    review_mock.file_objection = AsyncMock(
        return_value=mock_objection(),
    )
    app.state.review_client = review_mock

    # Mock analytics client
    analytics_mock = AsyncMock()
    analytics_mock.get_score_history = AsyncMock(
        return_value=[
            {"exam_id": EXAM_ID, "score": 78.5, "percentile": 85.0},
        ],
    )
    analytics_mock.get_trends = AsyncMock(
        return_value={
            "history": [
                {"exam_id": EXAM_ID, "score": 78.5, "percentile": 85.0},
            ],
        },
    )
    analytics_mock.get_strengths = AsyncMock(
        return_value={
            "strengths": ["Algebra", "Geometry"],
            "weaknesses": ["Trigonometry"],
        },
    )
    app.state.analytics_client = analytics_mock

    # Mock chat client
    chat_mock = AsyncMock()
    chat_mock.get_thread = AsyncMock(
        return_value=[
            {
                "message_id": str(uuid4()),
                "sender_id": STUDENT_ID,
                "content": "Hello teacher",
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "read_at": None,
            },
        ],
    )
    chat_mock.send_message = AsyncMock(
        return_value={
            "message_id": str(uuid4()),
            "sender_id": STUDENT_ID,
            "content": "Question about Q1",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "read_at": None,
        },
    )
    app.state.chat_client = chat_mock

    return TestClient(app, raise_server_exceptions=False)
