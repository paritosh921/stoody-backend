"""Tests for student-facing pen-binding endpoints in `api/v1/student_pens_async.py`.

Mobile/web students authenticate with a student JWT and need a way to:
  * Read their own bindings + per-student policy (`GET /student/pens`).
  * Auto-bind a fresh pen on first connect (`POST /student/pens`).

Deletion is intentionally NOT exposed here — admin-only via the existing
`DELETE /admin/pens/{mac}` (slice #5/#8).

The student endpoints share the same tenant `pens` collection that the
agent server (`stoody-ble-agent/server`) and the admin endpoints
(`admin_pens_async`) read/write, so a binding made through any path
shows up everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _student_user(username: str = "alice", db_name: str = "skb_test") -> dict:
    return {
        "user_id": f"oid-{username}",
        "username": username,
        "user_type": "student",
        "db_name": db_name,
    }


async def _seed_student(db, *, username: str, student_id: str = "STU-X", allowed: int | None = None) -> None:
    doc = {"username": username, "student_id": student_id, "full_name": username}
    if allowed is not None:
        doc["allowed_pen_count"] = allowed
    await db["students"].insert_one(doc)


@pytest.fixture
def db():
    from mongomock_motor import AsyncMongoMockClient
    return AsyncMongoMockClient()["skb_test"]


async def _call_list(db_obj, *, user):
    from api.v1.student_pens_async import student_list_my_pens

    with patch("api.v1.student_pens_async.get_tenant_db_or_403", return_value=db_obj):
        return await student_list_my_pens(
            db=None,  # type: ignore[arg-type]
            current_user=user,
        )


async def _call_register(db_obj, *, user, mac: str, name: str):
    from api.v1.student_pens_async import (
        student_register_my_pen,
        StudentRegisterPenRequest,
    )
    with patch("api.v1.student_pens_async.get_tenant_db_or_403", return_value=db_obj):
        return await student_register_my_pen(
            payload=StudentRegisterPenRequest(pen_mac=mac, pen_name=name),
            db=None,  # type: ignore[arg-type]
            current_user=user,
        )


@pytest.mark.asyncio
async def test_list_returns_empty_when_no_bindings(db):
    await _seed_student(db, username="alice")
    out = await _call_list(db, user=_student_user("alice"))
    assert out["pens"] == []
    assert out["allowed_pen_count"] == 1
    assert out["current_count"] == 0


@pytest.mark.asyncio
async def test_list_returns_only_my_active_bindings(db):
    await _seed_student(db, username="alice")
    await _seed_student(db, username="bob", student_id="STU-B")
    # Alice's pens
    await db["pens"].insert_many([
        {"user_id": "alice", "pen_mac": "AA:BB:CC:DD:EE:01", "pen_name": "Alice 1", "status": "active"},
        {"user_id": "alice", "pen_mac": "11:22:33:44:55:66", "pen_name": "Old", "status": "deregistered"},
    ])
    # Bob's pen — must NOT leak into alice's listing.
    await db["pens"].insert_one(
        {"user_id": "bob", "pen_mac": "DD:EE:FF:00:11:22", "pen_name": "Bob 1", "status": "active"}
    )

    out = await _call_list(db, user=_student_user("alice"))
    macs = sorted(p["pen_mac"] for p in out["pens"])
    assert macs == ["AA:BB:CC:DD:EE:01"]
    assert out["current_count"] == 1


@pytest.mark.asyncio
async def test_list_uses_per_student_limit(db):
    await _seed_student(db, username="alice", allowed=4)
    out = await _call_list(db, user=_student_user("alice"))
    assert out["allowed_pen_count"] == 4


@pytest.mark.asyncio
async def test_register_first_pen_succeeds(db):
    await _seed_student(db, username="alice")
    out = await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="My pen")
    assert out["pen_mac"] == "AA:BB:CC:DD:EE:01"
    assert out["pen_name"] == "My pen"
    assert out["status"] == "registered"


@pytest.mark.asyncio
async def test_register_same_mac_is_rename(db):
    """Idempotent rename — must NOT count toward the limit."""
    await _seed_student(db, username="alice")
    await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="Old")
    out = await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="New name")
    assert out["pen_name"] == "New name"


@pytest.mark.asyncio
async def test_register_blocked_at_limit(db):
    from fastapi import HTTPException
    await _seed_student(db, username="alice")  # default limit 1
    await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="First")
    with pytest.raises(HTTPException) as exc:
        await _call_register(db, user=_student_user("alice"), mac="11:22:33:44:55:66", name="Second")
    assert exc.value.status_code == 409
    detail = (exc.value.detail or "").lower()
    assert "limit" in detail or "admin" in detail


@pytest.mark.asyncio
async def test_register_allowed_under_admin_raised_limit(db):
    await _seed_student(db, username="alice", allowed=2)
    await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="First")
    out = await _call_register(db, user=_student_user("alice"), mac="11:22:33:44:55:66", name="Second")
    assert out["pen_mac"] == "11:22:33:44:55:66"


@pytest.mark.asyncio
async def test_register_409_when_pen_owned_by_other_student(db):
    from fastapi import HTTPException
    await _seed_student(db, username="alice")
    await _seed_student(db, username="bob", student_id="STU-B")
    await db["pens"].insert_one(
        {"user_id": "bob", "pen_mac": "AA:BB:CC:DD:EE:01", "pen_name": "Bob's", "status": "active"}
    )
    with pytest.raises(HTTPException) as exc:
        await _call_register(db, user=_student_user("alice"), mac="AA:BB:CC:DD:EE:01", name="Trying to steal")
    assert exc.value.status_code == 409
    assert "another" in (exc.value.detail or "").lower() or "registered" in (exc.value.detail or "").lower()


@pytest.mark.asyncio
async def test_register_normalizes_mac_case(db):
    await _seed_student(db, username="alice")
    out = await _call_register(db, user=_student_user("alice"), mac="aa:bb:cc:dd:ee:01", name="x")
    assert out["pen_mac"] == "AA:BB:CC:DD:EE:01"


@pytest.mark.asyncio
async def test_register_rejects_invalid_mac(db):
    from fastapi import HTTPException
    from api.v1.student_pens_async import StudentRegisterPenRequest
    with pytest.raises(HTTPException) as exc:
        StudentRegisterPenRequest(pen_mac="not-a-mac", pen_name="x")
    assert exc.value.status_code == 400


def test_router_exposes_expected_routes():
    from api.v1.student_pens_async import router
    paths = {(route.path, tuple(sorted(route.methods))) for route in router.routes}
    assert ("/pens", ("GET",)) in paths
    assert ("/pens", ("POST",)) in paths
    # No DELETE — students cannot self-unbind.
    assert all(method != ("DELETE",) for _, method in paths)
