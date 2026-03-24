"""RBAC enforcement tests — teacher/admin tokens get 403.

Test IDs: I-SBFF-RBAC-01 through I-SBFF-RBAC-05
"""

from __future__ import annotations

from tests.conftest import (
    EXAM_ID,
    QUESTION_ID,
    TEACHER_ID,
    build_client,
    make_teacher_token,
    make_student_token,
)


# -- I-SBFF-RBAC-01: Teacher token on score route gets 403 ----------------


def test_teacher_score_forbidden():
    """I-SBFF-RBAC-01: Teacher JWT returns 403 on score endpoint."""
    client = build_client()
    token = make_teacher_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert "teacher" in resp.json()["detail"].lower()


# -- I-SBFF-RBAC-02: Teacher token on objection routes gets 403 -----------


def test_teacher_objections_forbidden():
    """I-SBFF-RBAC-02: Teacher JWT returns 403 on objection endpoints."""
    client = build_client()
    token = make_teacher_token()
    resp = client.get(
        "/api/v1/student/objections",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# -- I-SBFF-RBAC-03: Teacher token on performance gets 403 ----------------


def test_teacher_performance_forbidden():
    """I-SBFF-RBAC-03: Teacher JWT returns 403 on performance endpoint."""
    client = build_client()
    token = make_teacher_token()
    resp = client.get(
        "/api/v1/student/performance/history",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# -- I-SBFF-RBAC-04: Teacher token on chat gets 403 -----------------------


def test_teacher_chat_forbidden():
    """I-SBFF-RBAC-04: Teacher JWT returns 403 on chat endpoint."""
    client = build_client()
    token = make_teacher_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/chat/{TEACHER_ID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# -- I-SBFF-RBAC-05: Missing auth returns 401 -----------------------------


def test_missing_auth_returns_401():
    """I-SBFF-RBAC-05: Request without Authorization header returns 401."""
    client = build_client()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
    )
    assert resp.status_code == 401
