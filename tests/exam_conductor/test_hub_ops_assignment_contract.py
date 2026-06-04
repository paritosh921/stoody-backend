"""
ExamPen hub assignment contract tests.

Covers the hub-facing assignment payload used by the edge ExamPen
service. The response must remain backward-compatible while exposing
teacher and pen-binding metadata when the exam document has it.

Run from backend/:
    .\\venv\\Scripts\\python -m pytest tests\\exam_conductor\\test_hub_ops_assignment_contract.py -q
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _hub_user(hub_id: str = "HUB-1") -> Dict[str, Any]:
    return {
        "user_id": hub_id,
        "user_type": "hub",
        "hub_id": hub_id,
        "db_name": "skb_test",
        "scopes": ["hub:heartbeat", "hub:data:upload"],
    }


def _admin_user() -> Dict[str, Any]:
    return {
        "user_id": "admin-1",
        "user_type": "admin",
        "db_name": "skb_test",
        "institution_id": "INST-1",
    }


async def _seed_hub(
    db,
    *,
    hub_id: str = "HUB-1",
    assigned_exam_id: Optional[str] = "EXAM-1",
) -> None:
    doc: Dict[str, Any] = {
        "hub_id": hub_id,
        "hub_name": "Exam hub",
        "status": "online",
        "hub_credentials": {"status": "active"},
    }
    if assigned_exam_id is not None:
        doc["assigned_exam_id"] = assigned_exam_id
    await db["exampen_hubs"].insert_one(doc)


async def _seed_exam(db, **overrides) -> None:
    doc: Dict[str, Any] = {
        "exam_id": overrides.pop("exam_id", "EXAM-1"),
        "exam_type": overrides.pop("exam_type", "pcr"),
        "duration_minutes": overrides.pop("duration_minutes", 75),
        "lifecycle_state": overrides.pop("lifecycle_state", "armed"),
        "roster": overrides.pop("roster", ["S-1", "S-2"]),
        "teacher_ids": overrides.pop("teacher_ids", ["T-1", "T-2"]),
        "pen_bindings": overrides.pop(
            "pen_bindings",
            {
                "AA:BB:CC:DD:EE:01": "S-1",
                "AA:BB:CC:DD:EE:02": "S-2",
            },
        ),
    }
    doc.update(overrides)
    await db["exampen_exams"].insert_one(doc)


async def _call_get_assignment(db, current_user=None, hub_id: str = "HUB-1"):
    from api.v1.hub_ops_async import get_assignment

    with patch("api.v1.hub_ops_async._get_tenant_db", return_value=db):
        return await get_assignment(
            hub_id=hub_id,
            current_user=current_user or _hub_user(hub_id),
            db=None,  # type: ignore[arg-type]
        )


async def _call_assign(db, current_user=None, hub_id: str = "HUB-1"):
    from api.v1.hub_ops_async import AssignRequest, assign_exam_to_hub

    with patch("api.v1.hub_ops_async._get_tenant_db", return_value=db):
        return await assign_exam_to_hub(
            hub_id=hub_id,
            body=AssignRequest(exam_id="EXAM-1"),
            current_user=current_user or _admin_user(),
            db=None,  # type: ignore[arg-type]
        )


async def _call_session_start(db, current_user=None, hub_id: str = "HUB-1"):
    from api.v1.hub_ops_async import SessionEventRequest, session_start

    with patch("api.v1.hub_ops_async._get_tenant_db", return_value=db):
        return await session_start(
            hub_id=hub_id,
            body=SessionEventRequest(exam_id="EXAM-1"),
            current_user=current_user or _hub_user(hub_id),
            db=None,  # type: ignore[arg-type]
        )


async def _call_session_end(db, current_user=None, hub_id: str = "HUB-1"):
    from api.v1.hub_ops_async import SessionEventRequest, session_end

    with patch("api.v1.hub_ops_async._get_tenant_db", return_value=db):
        return await session_end(
            hub_id=hub_id,
            body=SessionEventRequest(exam_id="EXAM-1"),
            current_user=current_user or _hub_user(hub_id),
            db=None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_get_assignment_returns_teacher_ids_and_pen_bindings():
    db = _fresh_db()
    await _seed_hub(db)
    await _seed_exam(db)

    result = await _call_get_assignment(db)

    assert result.hub_id == "HUB-1"
    assert result.assigned_exam_id == "EXAM-1"
    assert result.exam_type == "pcr"
    assert result.duration_minutes == 75
    assert result.lifecycle_state == "armed"
    assert result.roster == ["S-1", "S-2"]
    assert result.teacher_ids == ["T-1", "T-2"]
    assert result.pen_bindings == {
        "AA:BB:CC:DD:EE:01": "S-1",
        "AA:BB:CC:DD:EE:02": "S-2",
    }


@pytest.mark.asyncio
async def test_assign_returns_same_enriched_assignment_payload():
    db = _fresh_db()
    await _seed_hub(db, assigned_exam_id=None)
    await _seed_exam(db, teacher_ids=["T-7"], pen_bindings={"PEN-MAC": "S-7"})

    result = await _call_assign(db)

    assert result.assigned_exam_id == "EXAM-1"
    assert result.teacher_ids == ["T-7"]
    assert result.pen_bindings == {"PEN-MAC": "S-7"}
    hub_doc = await db["exampen_hubs"].find_one({"hub_id": "HUB-1"})
    assert hub_doc["assigned_exam_id"] == "EXAM-1"


@pytest.mark.asyncio
async def test_assignment_duration_falls_back_to_total_minutes():
    db = _fresh_db()
    await _seed_hub(db)
    await _seed_exam(db, duration_minutes=None, total_minutes=42)

    result = await _call_get_assignment(db)

    assert result.duration_minutes == 42


@pytest.mark.asyncio
async def test_missing_teacher_ids_and_pen_bindings_default_empty():
    db = _fresh_db()
    await _seed_hub(db)
    await _seed_exam(db, teacher_ids=None, pen_bindings=None)

    result = await _call_get_assignment(db)

    assert result.teacher_ids == []
    assert result.pen_bindings == {}


@pytest.mark.asyncio
async def test_no_assignment_returns_empty_defaults():
    db = _fresh_db()
    await _seed_hub(db, assigned_exam_id=None)

    result = await _call_get_assignment(db)

    assert result.hub_id == "HUB-1"
    assert result.assigned_exam_id is None
    assert result.teacher_ids == []
    assert result.pen_bindings == {}
    assert result.roster == []


def test_mobile_local_access_scopes_include_exampen_control():
    from api.v1.hub_ops_async import MOBILE_ACCESS_SCOPES

    assert "hub:read" in MOBILE_ACCESS_SCOPES
    assert "hub:pens" in MOBILE_ACCESS_SCOPES
    assert "exampen:control" in MOBILE_ACCESS_SCOPES


@pytest.mark.asyncio
async def test_link_local_hub_preserves_selected_tutors_on_relink():
    """Re-linking a deployed hub must not wipe mobile tutor authorization."""
    db = _fresh_db()
    master_db = _fresh_db()
    await db["exampen_hubs"].insert_one(
        {
            "hub_id": "HUB-1",
            "mobile_access": {
                "access_policy": "selected_only",
                "allowed_tutor_ids": ["TUT-1", "TUT-2"],
            },
        }
    )
    await master_db["exampen_hubs"].insert_one(
        {
            "hub_id": "HUB-1",
            "db_name": "skb_test",
            "institution_id": "INST-1",
        }
    )

    from api.v1.hub_ops_async import LinkLocalHubRequest, link_local_hub

    with (
        patch("api.v1.hub_ops_async._get_tenant_db", return_value=db),
        patch("api.v1.hub_ops_async._get_master_db", return_value=master_db),
    ):
        result = await link_local_hub(
            body=LinkLocalHubRequest(
                hub_id="HUB-1",
                hub_name="Exam hub",
                capabilities=["smartboard", "pen_uplink"],
            ),
            current_user=_admin_user(),
            db=None,  # type: ignore[arg-type]
        )

    hub_doc = await db["exampen_hubs"].find_one({"hub_id": "HUB-1"})
    assert result.hub_id == "HUB-1"
    assert hub_doc["mobile_access"]["allowed_tutor_ids"] == ["TUT-1", "TUT-2"]
    assert hub_doc["mobile_access"]["access_policy"] == "selected_only"


@pytest.mark.asyncio
async def test_hub_session_events_move_exam_lifecycle():
    db = _fresh_db()
    await _seed_hub(db)
    await _seed_exam(
        db,
        lifecycle_state="armed",
        hub_assignments=[
            {
                "hub_id": "HUB-1",
                "hub_name": "Exam hub",
                "assigned_at": None,
                "session_started_at": None,
                "session_ended_at": None,
            }
        ],
    )

    started = await _call_session_start(db)
    after_start = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    ended = await _call_session_end(db)
    after_end = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})

    assert started.event == "session_started"
    assert after_start["lifecycle_state"] == "in_progress"
    assert after_start["hub_assignments"][0]["session_started_at"] is not None
    assert ended.event == "session_ended"
    assert after_end["lifecycle_state"] == "collection_closed"
    assert after_end["hub_assignments"][0]["session_ended_at"] is not None


@pytest.mark.asyncio
async def test_hub_session_start_rejects_unassigned_exam():
    db = _fresh_db()
    await _seed_hub(db, assigned_exam_id="OTHER-EXAM")
    await _seed_exam(
        db,
        lifecycle_state="armed",
        hub_assignments=[
            {
                "hub_id": "HUB-1",
                "hub_name": "Exam hub",
                "assigned_at": None,
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_session_start(db)

    assert exc_info.value.status_code == 409
    assert "not assigned" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_hub_session_start_rejects_exam_missing_hub_assignment_row():
    db = _fresh_db()
    await _seed_hub(db)
    await _seed_exam(db, lifecycle_state="armed", hub_assignments=[])

    with pytest.raises(HTTPException) as exc_info:
        await _call_session_start(db)

    assert exc_info.value.status_code == 409
    assert "does not include hub assignment" in str(exc_info.value.detail)
