"""Score viewing route tests — mocked backing services.

Test IDs: I-SBFF-SCORE-01 through I-SBFF-SCORE-06
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from tests.conftest import (
    EXAM_ID,
    QUESTION_ID,
    build_client,
    make_student_token,
    make_parent_token,
    CHILD_STUDENT_ID,
)


# -- I-SBFF-SCORE-01: Student gets own score summary ----------------------


def test_student_score_summary():
    """I-SBFF-SCORE-01: GET /student/exams/{id}/score returns summary."""
    client = build_client()
    token = make_student_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exam_id"] == EXAM_ID
    assert body["total_score"] == 78.5
    assert body["percentage"] == 78.5
    assert body["percentile"] == 85.0
    assert "questions" in body


# -- I-SBFF-SCORE-02: Student gets question breakdown ---------------------


def test_student_question_breakdown():
    """I-SBFF-SCORE-02: GET /student/exams/{id}/questions returns list."""
    client = build_client()
    token = make_student_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/questions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert len(body["items"]) == 2


# -- I-SBFF-SCORE-03: Student gets answer insight -------------------------


def test_student_answer_insight():
    """I-SBFF-SCORE-03: GET /student/exams/{id}/questions/{qid}/answer."""
    client = build_client()
    token = make_student_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/questions/{QUESTION_ID}/answer",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["question_id"] == QUESTION_ID
    assert "answer_image_uri" in body
    assert "recognized_text" in body
    assert body["confidence"] == 0.95


# -- I-SBFF-SCORE-04: Parent gets child's score summary -------------------


def test_parent_score_summary():
    """I-SBFF-SCORE-04: Parent JWT with student_id param gets score."""
    client = build_client(parent_children=[CHILD_STUDENT_ID])
    token = make_parent_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score"
        f"?student_id={CHILD_STUDENT_ID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exam_id"] == EXAM_ID


# -- I-SBFF-SCORE-05: Score not found returns 404 -------------------------


def test_score_not_found():
    """I-SBFF-SCORE-05: Missing score returns 404."""
    client = build_client()
    client.app.state.score_client.get_score_summary = AsyncMock(
        return_value=None,
    )
    token = make_student_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# -- I-SBFF-SCORE-06: Answer insight not found returns 404 ----------------


def test_answer_insight_not_found():
    """I-SBFF-SCORE-06: Missing answer returns 404."""
    client = build_client()
    client.app.state.score_client.get_answer_insight = AsyncMock(
        return_value=None,
    )
    token = make_student_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/questions/q-99/answer",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
