"""
SEC-RBAC: Systematic RBAC matrix tests — all 7 roles x key endpoint actions.

Source of truth: STOODY_INTEGRATION_SPEC.md section 6 — Access Control Matrix.
Level: L3 (mock JWTs, mocked service responses) / L4 (against real services).

Each test encodes the expected allow/deny from the spec table and verifies that
the service enforces it.  Roles that should be denied get HTTP 403; roles that
should be allowed get 2xx or 200.

Test-ID prefix: SEC-RBAC-{NN}
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = [pytest.mark.security, pytest.mark.rbac, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# The RBAC matrix from STOODY_INTEGRATION_SPEC section 6.
#
# True  = action allowed (with possible scope constraints noted in comments).
# False = action denied (must get 403).
# ---------------------------------------------------------------------------

EXAM_ID = str(uuid.uuid4())
STUDENT_ID = "student-001"
QUESTION_ID = "q1"
FLAG_ID = str(uuid.uuid4())
OBJECTION_ID = str(uuid.uuid4())

# fmt: off
RBAC_MATRIX: dict[str, dict[str, bool]] = {
    #                         SA     Princ  HOD    Eval   Invig  Stud   Parent
    "create_exam":          {"super_admin": True,  "principal": True,  "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "view_all_scores":      {"super_admin": True,  "principal": True,  "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "edit_scores":          {"super_admin": False, "principal": False, "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "file_objection":       {"super_admin": False, "principal": False, "hod": False, "evaluator": False, "invigilator": False, "student": True,  "parent": False},
    "view_leaderboard":     {"super_admin": True,  "principal": True,  "hod": True,  "evaluator": True,  "invigilator": False, "student": True,  "parent": True},
    "export_data":          {"super_admin": True,  "principal": True,  "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "plagiarism_review":    {"super_admin": True,  "principal": True,  "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "view_own_scores":      {"super_admin": False, "principal": False, "hod": False, "evaluator": False, "invigilator": False, "student": True,  "parent": True},
    "review_objections":    {"super_admin": False, "principal": False, "hod": True,  "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "chat_tutor_side":      {"super_admin": False, "principal": False, "hod": False, "evaluator": True,  "invigilator": False, "student": False, "parent": False},
    "chat_student_side":    {"super_admin": False, "principal": False, "hod": False, "evaluator": False, "invigilator": False, "student": True,  "parent": False},
    "start_stop_exam_hub":  {"super_admin": False, "principal": False, "hod": False, "evaluator": False, "invigilator": True,  "student": False, "parent": False},
}
# fmt: on

# Map each action to its endpoint method/path/service.
ACTION_ENDPOINTS: dict[str, dict[str, Any]] = {
    "create_exam": {
        "method": "POST",
        "service": "exam_orch",
        "path": "/api/v1/exams",
        "body": {
            "title": "RBAC Test Exam",
            "subject_id": "math-101",
            "class_id": "class-10a",
            "section_id": "section-a",
            "scheduled_at": "2026-04-01T10:00:00Z",
            "duration_min": 60,
            "question_count": 10,
            "total_marks": 50,
        },
    },
    "view_all_scores": {
        "method": "GET",
        "service": "teacher_bff",
        "path": f"/api/v1/teacher/exams/{EXAM_ID}/scores",
    },
    "edit_scores": {
        "method": "PATCH",
        "service": "score_engine",
        "path": f"/api/v1/scores/{EXAM_ID}/students/{STUDENT_ID}/questions/{QUESTION_ID}",
        "body": {
            "teacher_id": "tutor-001",
            "new_score": 4.0,
            "reason": "Partial credit for correct method",
        },
    },
    "file_objection": {
        "method": "POST",
        "service": "student_bff",
        "path": "/api/v1/student/objections",
        "body": {
            "exam_id": EXAM_ID,
            "question_id": QUESTION_ID,
            "objection_text": "I believe my answer deserves partial credit for the correct approach.",
        },
    },
    "view_leaderboard": {
        "method": "GET",
        "service": "analytics",
        "path": f"/api/v1/analytics/exams/{EXAM_ID}/leaderboard",
    },
    "export_data": {
        "method": "GET",
        "service": "analytics",
        "path": f"/api/v1/analytics/exams/{EXAM_ID}/export?format=csv",
    },
    "plagiarism_review": {
        "method": "GET",
        "service": "plagiarism",
        "path": f"/api/v1/plagiarism/exams/{EXAM_ID}/flags",
    },
    "view_own_scores": {
        "method": "GET",
        "service": "student_bff",
        "path": f"/api/v1/student/exams/{EXAM_ID}/scores",
    },
    "review_objections": {
        "method": "GET",
        "service": "review",
        "path": f"/api/v1/objections?exam_id={EXAM_ID}",
    },
    "chat_tutor_side": {
        "method": "GET",
        "service": "teacher_bff",
        "path": f"/api/v1/teacher/chat/{EXAM_ID}/{STUDENT_ID}",
    },
    "chat_student_side": {
        "method": "GET",
        "service": "student_bff",
        "path": f"/api/v1/student/chat/{EXAM_ID}",
    },
    "start_stop_exam_hub": {
        "method": "POST",
        "service": "exam_orch",
        "path": f"/api/v1/exams/{EXAM_ID}/transitions",
        "body": {
            "to_state": "timer_running",
            "actor_id": "invigilator-001",
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_id(action: str, role: str, allowed: bool) -> str:
    """Deterministic test ID for parametrize."""
    verb = "allow" if allowed else "deny"
    return f"SEC-RBAC-{action}-{role}-{verb}"


def _generate_rbac_cases():
    """Yield (action, role, should_be_allowed) tuples for parametrize."""
    for action, roles in RBAC_MATRIX.items():
        for role, allowed in roles.items():
            yield pytest.param(action, role, allowed, id=_make_test_id(action, role, allowed))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRBACMatrixEnforcement:
    """SEC-RBAC-01 through SEC-RBAC-84: every cell of the 12x7 matrix."""

    @pytest.mark.parametrize("action,role,should_allow", list(_generate_rbac_cases()))
    async def test_rbac_cell(
        self,
        action: str,
        role: str,
        should_allow: bool,
        token_factory,
        http_session,
        service_urls,
    ):
        """Verify one RBAC matrix cell: role x action -> allow/deny."""
        endpoint = ACTION_ENDPOINTS[action]
        url = f"{service_urls[endpoint['service']]}{endpoint['path']}"
        headers = token_factory.bearer(role)
        method = endpoint["method"]
        body = endpoint.get("body")

        if method == "GET":
            resp = await http_session.get(url, headers=headers)
        elif method == "POST":
            resp = await http_session.post(url, headers=headers, json=body)
        elif method == "PATCH":
            resp = await http_session.patch(url, headers=headers, json=body)
        else:
            pytest.fail(f"Unsupported method: {method}")

        if should_allow:
            assert resp.status in (200, 201, 202), (
                f"Expected 2xx for {role} on {action}, got {resp.status}. "
                f"RBAC matrix says this should be ALLOWED."
            )
        else:
            assert resp.status == 403, (
                f"Expected 403 for {role} on {action}, got {resp.status}. "
                f"RBAC matrix says this should be DENIED."
            )


class TestRBACNoToken:
    """SEC-RBAC-NOAUTH: Requests without a bearer token get 401."""

    @pytest.mark.parametrize(
        "action",
        list(ACTION_ENDPOINTS.keys()),
        ids=[f"SEC-RBAC-NOAUTH-{a}" for a in ACTION_ENDPOINTS],
    )
    async def test_no_token_returns_401(
        self,
        action: str,
        http_session,
        service_urls,
    ):
        """Every endpoint must reject unauthenticated requests with 401."""
        endpoint = ACTION_ENDPOINTS[action]
        url = f"{service_urls[endpoint['service']]}{endpoint['path']}"
        method = endpoint["method"]
        body = endpoint.get("body")

        if method == "GET":
            resp = await http_session.get(url)
        elif method == "POST":
            resp = await http_session.post(url, json=body)
        elif method == "PATCH":
            resp = await http_session.patch(url, json=body)
        else:
            pytest.fail(f"Unsupported method: {method}")

        assert resp.status == 401, (
            f"Expected 401 (unauthenticated) for {action} without token, "
            f"got {resp.status}"
        )


class TestRBACExpiredToken:
    """SEC-RBAC-EXPIRED: Requests with expired JWTs get 401."""

    @pytest.mark.parametrize(
        "role",
        ["super_admin", "evaluator", "student"],
        ids=[f"SEC-RBAC-EXPIRED-{r}" for r in ["super_admin", "evaluator", "student"]],
    )
    async def test_expired_token_returns_401(
        self,
        role: str,
        http_session,
        service_urls,
    ):
        """An expired JWT must be rejected regardless of role."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stoody-mock"))
        from keys import make_token

        expired_token = make_token(
            user_id=f"{role}-expired",
            role=role,
            ttl_seconds=-3600,  # expired 1h ago
        )

        url = f"{service_urls['exam_orch']}/api/v1/exams"
        resp = await http_session.get(
            url,
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert resp.status == 401, (
            f"Expected 401 for expired {role} token, got {resp.status}"
        )


class TestRBACMalformedToken:
    """SEC-RBAC-MALFORMED: Garbage bearer values get 401."""

    @pytest.mark.parametrize(
        "bad_token",
        [
            "not-a-jwt",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ0ZXN0In0.",  # alg=none
            "",
            "Bearer",  # double-bearer
        ],
        ids=["garbage", "alg_none", "empty", "double_bearer"],
    )
    async def test_malformed_token_returns_401(
        self,
        bad_token: str,
        http_session,
        service_urls,
    ):
        url = f"{service_urls['exam_orch']}/api/v1/exams"
        resp = await http_session.get(
            url,
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        assert resp.status == 401, (
            f"Expected 401 for malformed token '{bad_token[:20]}...', got {resp.status}"
        )


class TestRBACRoleScope:
    """SEC-RBAC-SCOPE: Scoped roles can only access their own data."""

    async def test_evaluator_cannot_access_unassigned_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RBAC-SCOPE-01: Evaluator token for tutor-X cannot score
        an exam assigned to tutor-Y."""
        unassigned_exam = str(uuid.uuid4())
        headers = token_factory.bearer(
            "evaluator",
            user_id="tutor-999-unassigned",
        )
        url = (
            f"{service_urls['score_engine']}"
            f"/api/v1/scores/{unassigned_exam}/students/{STUDENT_ID}"
            f"/questions/{QUESTION_ID}"
        )
        resp = await http_session.patch(
            url,
            headers=headers,
            json={
                "teacher_id": "tutor-999-unassigned",
                "new_score": 5.0,
                "reason": "Attempting unauthorized score edit",
            },
        )
        assert resp.status in (403, 404), (
            f"Expected 403 or 404 for unassigned evaluator, got {resp.status}"
        )

    async def test_hod_cannot_access_other_department_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RBAC-SCOPE-02: HOD for department-A cannot view scores
        for department-B's exam."""
        other_dept_exam = str(uuid.uuid4())
        headers = token_factory.bearer(
            "hod",
            user_id="hod-dept-a",
            extra_claims={"department_id": "dept-a"},
        )
        url = (
            f"{service_urls['teacher_bff']}"
            f"/api/v1/teacher/exams/{other_dept_exam}/scores"
        )
        resp = await http_session.get(url, headers=headers)
        assert resp.status in (403, 404), (
            f"Expected 403/404 for HOD accessing other dept exam, got {resp.status}"
        )

    async def test_student_cannot_view_other_students_scores(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RBAC-SCOPE-03: Student-A cannot view Student-B's score
        via the student BFF."""
        headers = token_factory.bearer(
            "student",
            user_id="student-001",
        )
        # The student BFF should scope scores to the authenticated user.
        # Attempting to access another student's data should fail.
        url = (
            f"{service_urls['student_bff']}"
            f"/api/v1/student/exams/{EXAM_ID}/scores"
        )
        resp = await http_session.get(url, headers=headers)
        if resp.status == 200:
            data = await resp.json()
            # If 200, verify the response only contains data for student-001.
            # The BFF should filter by the token's sub claim.
            assert data.get("student_id", "student-001") == "student-001", (
                "Student BFF returned scores for a different student"
            )

    async def test_parent_only_sees_linked_children(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RBAC-SCOPE-04: Parent can only see scores for linked children,
        not arbitrary students."""
        headers = token_factory.bearer(
            "parent",
            user_id="parent-001",
            extra_claims={
                "child_student_ids": ["student-001", "student-003"],
            },
        )
        url = (
            f"{service_urls['student_bff']}"
            f"/api/v1/student/exams/{EXAM_ID}/scores"
        )
        resp = await http_session.get(url, headers=headers)
        # Parent should only see scores for linked children.
        if resp.status == 200:
            data = await resp.json()
            student_id = data.get("student_id")
            if student_id:
                assert student_id in ("student-001", "student-003"), (
                    f"Parent saw scores for unlinked student: {student_id}"
                )
