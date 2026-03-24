"""Parent scope tests — parent JWT sees only linked child's data.

Test IDs: I-SBFF-PARENT-01 through I-SBFF-PARENT-06
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from tests.conftest import (
    CHILD_STUDENT_ID,
    EXAM_ID,
    OBJECTION_ID,
    STUDENT_ID,
    build_client,
    make_parent_token,
    mock_objection,
)


# -- I-SBFF-PARENT-01: Parent default child resolution --------------------


def test_parent_single_child_default():
    """I-SBFF-PARENT-01: Parent with one child auto-resolves student_id."""
    client = build_client(parent_children=[CHILD_STUDENT_ID])
    token = make_parent_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    # Score client was called with the child's ID
    client.app.state.score_client.get_score_summary.assert_called_once()
    call_args = client.app.state.score_client.get_score_summary.call_args
    assert call_args.kwargs.get("student_id") == CHILD_STUDENT_ID


# -- I-SBFF-PARENT-02: Parent must specify child when multiple -----------


def test_parent_multiple_children_requires_student_id():
    """I-SBFF-PARENT-02: Parent with 2+ children must provide student_id."""
    client = build_client(parent_children=["child-a", "child-b"])
    token = make_parent_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400
    assert "student_id" in resp.json()["detail"].lower()


# -- I-SBFF-PARENT-03: Parent cannot access unlinked student ---------------


def test_parent_cannot_access_unlinked_student():
    """I-SBFF-PARENT-03: Parent requesting unlinked student gets 403."""
    client = build_client(parent_children=[CHILD_STUDENT_ID])
    token = make_parent_token()
    resp = client.get(
        f"/api/v1/student/exams/{EXAM_ID}/score"
        f"?student_id=some-other-student",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# -- I-SBFF-PARENT-04: Parent cannot file objections -----------------------


def test_parent_cannot_file_objection():
    """I-SBFF-PARENT-04: Parent role blocked from filing objections."""
    client = build_client(parent_children=[CHILD_STUDENT_ID])
    token = make_parent_token()
    resp = client.post(
        f"/api/v1/student/exams/{EXAM_ID}/objections",
        json={
            "exam_id": EXAM_ID,
            "question_id": "q-1",
            "objection_text": "I think my child deserves more marks.",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert "only students" in resp.json()["detail"].lower()


# -- I-SBFF-PARENT-05: Parent cannot send chat messages -------------------


def test_parent_cannot_send_chat():
    """I-SBFF-PARENT-05: Parent role blocked from sending chat messages."""
    client = build_client(parent_children=[CHILD_STUDENT_ID])
    token = make_parent_token()
    resp = client.post(
        f"/api/v1/student/exams/{EXAM_ID}/chat/teacher-001",
        json={"content": "Hello from parent"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# -- I-SBFF-PARENT-06: Parent can view objection of linked child ----------


def test_parent_can_view_child_objection():
    """I-SBFF-PARENT-06: Parent can view objection belonging to linked child."""
    client = build_client(parent_children=[STUDENT_ID])
    # The mock objection has student_id = STUDENT_ID
    obj = mock_objection()
    obj["student_id"] = STUDENT_ID
    client.app.state.review_client.get_objection = AsyncMock(
        return_value=obj,
    )
    token = make_parent_token()
    resp = client.get(
        f"/api/v1/student/objections/{OBJECTION_ID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["objection_id"] == OBJECTION_ID
