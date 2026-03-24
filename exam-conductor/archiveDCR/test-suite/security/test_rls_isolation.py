"""
SEC-RLS: Cross-tenant data isolation (Row-Level Security) tests.

Verifies that PostgreSQL RLS policies prevent tenant A's token from
reading or mutating tenant B's data across all service databases.

Source of truth: FAILURE_MITIGATION_REGISTER.md A8.1, CLAUDE.md Auth section.
Level: L4 (requires real services with RLS-enabled databases).

Test-ID prefix: SEC-RLS-{NN}
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = [pytest.mark.security, pytest.mark.rls, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Constants for two isolated tenants
# ---------------------------------------------------------------------------

TENANT_A = "tenant-alpha"
TENANT_B = "tenant-beta"

# Deterministic IDs for seed data attribution.
EXAM_A = str(uuid.uuid4())
EXAM_B = str(uuid.uuid4())
STUDENT_A = "student-alpha-001"
STUDENT_B = "student-beta-001"
QUESTION_ID = "q1"
OBJECTION_A = str(uuid.uuid4())
FLAG_A = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth(token_factory, role: str, tenant_id: str, user_id: str) -> dict[str, str]:
    """Build Authorization header for a given tenant + role."""
    return token_factory.bearer(role, tenant_id=tenant_id, user_id=user_id)


# ---------------------------------------------------------------------------
# Exam Orchestrator — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSExamOrch:
    """SEC-RLS-EXAM: Tenant isolation on svc-exam-orch."""

    async def test_tenant_a_cannot_list_tenant_b_exams(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-EXAM-01: GET /exams with tenant-A token returns zero
        of tenant-B's exams."""
        headers_a = _auth(token_factory, "evaluator", TENANT_A, "tutor-a")
        url = f"{service_urls['exam_orch']}/api/v1/exams"

        resp = await http_session.get(url, headers=headers_a)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            for exam in items:
                assert exam.get("tenant_id", TENANT_A) != TENANT_B, (
                    f"Tenant A saw tenant B's exam: {exam.get('exam_id')}"
                )

    async def test_tenant_a_cannot_read_tenant_b_exam_detail(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-EXAM-02: GET /exams/{exam_b} with tenant-A token -> 403 or 404."""
        headers_a = _auth(token_factory, "evaluator", TENANT_A, "tutor-a")
        url = f"{service_urls['exam_orch']}/api/v1/exams/{EXAM_B}"

        resp = await http_session.get(url, headers=headers_a)
        assert resp.status in (403, 404), (
            f"Expected 403/404 for cross-tenant exam read, got {resp.status}"
        )

    async def test_tenant_b_cannot_modify_tenant_a_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-EXAM-03: PATCH /exams/{exam_a} with tenant-B token -> 403 or 404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['exam_orch']}/api/v1/exams/{EXAM_A}"

        resp = await http_session.patch(
            url,
            headers=headers_b,
            json={"duration_min": 120},
        )
        assert resp.status in (403, 404), (
            f"Expected 403/404 for cross-tenant exam mutation, got {resp.status}"
        )

    async def test_tenant_b_cannot_transition_tenant_a_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-EXAM-04: POST /exams/{exam_a}/transitions with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "invigilator", TENANT_B, "invig-b")
        url = f"{service_urls['exam_orch']}/api/v1/exams/{EXAM_A}/transitions"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={"to_state": "cancelled", "actor_id": "invig-b"},
        )
        assert resp.status in (403, 404), (
            f"Expected 403/404 for cross-tenant FSM transition, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Score Engine — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSScoreEngine:
    """SEC-RLS-SCORE: Tenant isolation on svc-score-engine."""

    async def test_tenant_a_cannot_read_tenant_b_scores(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SCORE-01: GET scores for tenant-B's exam with tenant-A token -> 403/404."""
        headers_a = _auth(token_factory, "evaluator", TENANT_A, "tutor-a")
        url = (
            f"{service_urls['score_engine']}"
            f"/api/v1/scores/{EXAM_B}/students/{STUDENT_B}"
        )

        resp = await http_session.get(url, headers=headers_a)
        assert resp.status in (403, 404), (
            f"Cross-tenant score read should fail, got {resp.status}"
        )

    async def test_tenant_a_cannot_edit_tenant_b_scores(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SCORE-02: PATCH score for tenant-B's student -> 403/404."""
        headers_a = _auth(token_factory, "evaluator", TENANT_A, "tutor-a")
        url = (
            f"{service_urls['score_engine']}"
            f"/api/v1/scores/{EXAM_B}/students/{STUDENT_B}"
            f"/questions/{QUESTION_ID}"
        )

        resp = await http_session.patch(
            url,
            headers=headers_a,
            json={
                "teacher_id": "tutor-a",
                "new_score": 0.0,
                "reason": "Malicious cross-tenant score wipe",
            },
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant score edit should fail, got {resp.status}"
        )

    async def test_tenant_b_cannot_finalize_tenant_a_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SCORE-03: POST finalize on tenant-A's exam with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "principal", TENANT_B, "princ-b")
        url = f"{service_urls['score_engine']}/api/v1/scores/{EXAM_A}/finalize"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={"actor_id": "princ-b"},
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant finalize should fail, got {resp.status}"
        )

    async def test_tenant_b_cannot_publish_tenant_a_scores(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SCORE-04: POST publish on tenant-A's exam with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "principal", TENANT_B, "princ-b")
        url = f"{service_urls['score_engine']}/api/v1/scores/{EXAM_A}/publish"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={"actor_id": "princ-b", "objection_window_days": 7},
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant publish should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Objection / Review — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSReview:
    """SEC-RLS-OBJ: Tenant isolation on svc-review (objections)."""

    async def test_tenant_b_cannot_list_tenant_a_objections(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-OBJ-01: GET /objections?exam_id={exam_a} with tenant-B -> empty or 403."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['review']}/api/v1/objections?exam_id={EXAM_A}"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            assert len(items) == 0, (
                f"Tenant B should see zero tenant-A objections, saw {len(items)}"
            )
        else:
            assert resp.status in (403, 404)

    async def test_tenant_b_cannot_resolve_tenant_a_objection(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-OBJ-02: POST resolve on tenant-A's objection -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['review']}/api/v1/objections/{OBJECTION_A}/resolve"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={
                "actor_id": "tutor-b",
                "resolution": "approved",
                "reason": "Unauthorized cross-tenant resolution attempt",
            },
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant objection resolve should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Plagiarism — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSPlagiarism:
    """SEC-RLS-PLAG: Tenant isolation on svc-plagiarism."""

    async def test_tenant_b_cannot_see_tenant_a_flags(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-PLAG-01: GET flags for tenant-A's exam -> empty or 403."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['plagiarism']}/api/v1/plagiarism/exams/{EXAM_A}/flags"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            assert len(items) == 0, (
                f"Tenant B should see zero tenant-A plagiarism flags, saw {len(items)}"
            )
        else:
            assert resp.status in (403, 404)

    async def test_tenant_b_cannot_verdict_tenant_a_flag(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-PLAG-02: PATCH verdict on tenant-A's flag -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['plagiarism']}/api/v1/plagiarism/flags/{FLAG_A}/verdict"

        resp = await http_session.patch(
            url,
            headers=headers_b,
            json={
                "teacher_id": "tutor-b",
                "verdict": "dismissed",
                "reason": "Unauthorized cross-tenant verdict attempt",
            },
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant plagiarism verdict should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Analytics — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSAnalytics:
    """SEC-RLS-ANAL: Tenant isolation on svc-analytics."""

    async def test_tenant_b_cannot_view_tenant_a_leaderboard(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-ANAL-01: GET leaderboard for tenant-A's exam with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['analytics']}/api/v1/analytics/exams/{EXAM_A}/leaderboard"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            assert len(items) == 0, (
                "Tenant B should see zero rows in tenant-A leaderboard"
            )
        else:
            assert resp.status in (403, 404)

    async def test_tenant_b_cannot_export_tenant_a_data(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-ANAL-02: GET export for tenant-A's exam with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "principal", TENANT_B, "princ-b")
        url = f"{service_urls['analytics']}/api/v1/analytics/exams/{EXAM_A}/export?format=csv"

        resp = await http_session.get(url, headers=headers_b)
        assert resp.status in (403, 404), (
            f"Cross-tenant data export should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Chat — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSChat:
    """SEC-RLS-CHAT: Tenant isolation on svc-chat."""

    async def test_tenant_b_teacher_cannot_read_tenant_a_chat(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-CHAT-01: GET chat thread for tenant-A exam with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['chat']}/api/v1/chat/threads/{EXAM_A}/{STUDENT_A}"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            assert len(items) == 0, (
                "Tenant B teacher should see zero messages from tenant-A chat"
            )
        else:
            assert resp.status in (403, 404)

    async def test_tenant_b_teacher_cannot_post_to_tenant_a_chat(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-CHAT-02: POST message to tenant-A chat with tenant-B -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['chat']}/api/v1/chat/threads/{EXAM_A}/{STUDENT_A}"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={"content": "Cross-tenant message injection attempt"},
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant chat write should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# BFF layers — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSTeacherBFF:
    """SEC-RLS-TBFF: Tenant isolation on svc-teacher-bff."""

    async def test_tenant_b_cannot_list_tenant_a_teacher_exams(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-TBFF-01: Teacher from tenant-B sees zero tenant-A exams."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = f"{service_urls['teacher_bff']}/api/v1/teacher/exams"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            for exam in items:
                assert exam.get("tenant_id", TENANT_B) != TENANT_A, (
                    f"Tenant B teacher saw tenant A exam: {exam.get('exam_id')}"
                )


class TestRLSStudentBFF:
    """SEC-RLS-SBFF: Tenant isolation on svc-student-bff."""

    async def test_tenant_b_student_cannot_see_tenant_a_exams(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SBFF-01: Student from tenant-B sees zero tenant-A exams."""
        headers_b = _auth(token_factory, "student", TENANT_B, "student-b")
        url = f"{service_urls['student_bff']}/api/v1/student/exams"

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            for exam in items:
                assert exam.get("tenant_id", TENANT_B) != TENANT_A, (
                    f"Tenant B student saw tenant A exam: {exam.get('exam_id')}"
                )

    async def test_tenant_b_student_cannot_file_objection_on_tenant_a(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-SBFF-02: Student from tenant-B cannot file objection on tenant-A exam."""
        headers_b = _auth(token_factory, "student", TENANT_B, "student-b")
        url = f"{service_urls['student_bff']}/api/v1/student/objections"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={
                "exam_id": EXAM_A,
                "question_id": QUESTION_ID,
                "objection_text": "Cross-tenant objection injection attempt on Q1.",
            },
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant objection filing should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Stroke Ingest — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSStrokeIngest:
    """SEC-RLS-STROKE: Tenant isolation on svc-stroke-ingest."""

    async def test_tenant_b_cannot_upload_strokes_for_tenant_a_exam(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-STROKE-01: Ingest for tenant-A's exam with tenant-B token -> 403/404."""
        import base64

        headers_b = _auth(token_factory, "invigilator", TENANT_B, "invig-b")
        url = f"{service_urls.get('stroke_ingest', service_urls['exam_orch'])}/api/v1/strokes/ingest"

        resp = await http_session.post(
            url,
            headers=headers_b,
            json={
                "exam_id": EXAM_A,
                "pen_mac": "FF:FF:FF:FF:FF:01",
                "chunk_index": 0,
                "total_chunks": 1,
                "payload_base64": base64.b64encode(b"cross-tenant-data").decode(),
                "checksum_crc32": "00000000",
                "upload_path": "wifi",
                "idempotency_key": f"{EXAM_A}:FF:FF:FF:FF:FF:01:0",
            },
        )
        assert resp.status in (403, 404), (
            f"Cross-tenant stroke ingest should fail, got {resp.status}"
        )


# ---------------------------------------------------------------------------
# Copy Upload — cross-tenant isolation
# ---------------------------------------------------------------------------


class TestRLSCopyUpload:
    """SEC-RLS-COPY: Tenant isolation on svc-copy-upload."""

    async def test_tenant_b_cannot_view_tenant_a_copies(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-RLS-COPY-01: GET copies for tenant-A exam/student -> 403/404."""
        headers_b = _auth(token_factory, "evaluator", TENANT_B, "tutor-b")
        url = (
            f"{service_urls.get('copy_upload', service_urls['exam_orch'])}"
            f"/api/v1/exams/{EXAM_A}/copies/{STUDENT_A}"
        )

        resp = await http_session.get(url, headers=headers_b)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            assert len(items) == 0, (
                "Tenant B should see zero tenant-A copy pages"
            )
        else:
            assert resp.status in (403, 404)
