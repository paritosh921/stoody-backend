"""Tests for admin pen-binding helpers in `api/v1/admin_pens_async.py`.

Run from `stoody-backend/`:
    pytest tests/test_admin_pens.py -v

We test the data-shaping helpers directly with `mongomock-motor` so the
suite stays hermetic — the route bodies themselves are thin wrappers over
these helpers + standard FastAPI dependencies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the backend package root is importable regardless of pytest's CWD.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def mongo_db():
    from mongomock_motor import AsyncMongoMockClient

    client = AsyncMongoMockClient()
    return client["skb_test"]


@pytest.mark.asyncio
async def test_normalize_mac_accepts_lowercase_and_uppercases():
    from api.v1.admin_pens_async import _normalize_mac

    assert _normalize_mac("aa:bb:cc:dd:ee:01") == "AA:BB:CC:DD:EE:01"


@pytest.mark.asyncio
async def test_normalize_mac_rejects_garbage():
    from fastapi import HTTPException
    from api.v1.admin_pens_async import _normalize_mac

    with pytest.raises(HTTPException) as exc:
        _normalize_mac("not-a-mac")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_attach_student_metadata_joins_username():
    from api.v1.admin_pens_async import _attach_student_metadata

    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await db["students"].insert_many(
        [
            {"username": "alice", "student_id": "STU-001", "full_name": "Alice Doe"},
            {"username": "bob", "student_id": "STU-002", "name": "Bob Roy"},
        ]
    )
    pens = [
        {"pen_mac": "AA:BB:CC", "user_id": "alice", "status": "active"},
        {"pen_mac": "DD:EE:FF", "user_id": "bob", "status": "active"},
        {"pen_mac": "11:22:33", "user_id": "ghost", "status": "active"},  # no student row
    ]
    out = await _attach_student_metadata(db, pens)
    by_mac = {row["pen_mac"]: row for row in out}
    assert by_mac["AA:BB:CC"]["student_id"] == "STU-001"
    assert by_mac["AA:BB:CC"]["student_name"] == "Alice Doe"
    assert by_mac["DD:EE:FF"]["student_name"] == "Bob Roy"
    # Orphan binding is not dropped — admin needs to see and clean it up.
    assert by_mac["11:22:33"]["student_id"] is None
    assert by_mac["11:22:33"]["student_name"] is None


@pytest.mark.asyncio
async def test_resolve_student_username_404_for_unknown():
    from fastapi import HTTPException
    from api.v1.admin_pens_async import _resolve_student_username

    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    with pytest.raises(HTTPException) as exc:
        await _resolve_student_username(db, "STU-NOPE")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_resolve_student_username_returns_username():
    from api.v1.admin_pens_async import _resolve_student_username

    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await db["students"].insert_one({"username": "alice", "student_id": "STU-001"})
    assert await _resolve_student_username(db, "STU-001") == "alice"


async def _seed_student(db, *, username: str, student_id: str = "STU-X") -> None:
    await db["students"].insert_one({"username": username, "student_id": student_id, "full_name": username})


async def _call_admin_assign(student_id: str, pen_mac: str, pen_name: str, *, db):
    """Bypass FastAPI deps: call the route function directly with stubbed deps."""
    from api.v1.admin_pens_async import admin_assign_pen, AdminAssignPenRequest

    # admin_assign_pen calls get_tenant_db_or_403; we stub the resolver via patch.
    from unittest.mock import patch

    with patch("api.v1.admin_pens_async.get_tenant_db_or_403", return_value=db):
        return await admin_assign_pen(
            student_id=student_id,
            payload=AdminAssignPenRequest(pen_mac=pen_mac, pen_name=pen_name),
            db=None,  # type: ignore[arg-type]
            current_user={"db_name": "skb_test", "user_id": "admin-1"},
        )


@pytest.mark.asyncio
async def test_admin_assign_blocks_when_student_already_has_active_binding():
    """Admins also obey the one-active-pen-per-student rule. They must
    explicitly unbind the old MAC first."""
    from fastapi import HTTPException
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await _seed_student(db, username="alice", student_id="STU-1")
    # Pre-existing active binding for Alice.
    await db["pens"].insert_one(
        {
            "user_id": "alice",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "pen_name": "Old pen",
            "status": "active",
        }
    )

    with pytest.raises(HTTPException) as exc:
        await _call_admin_assign("STU-1", "11:22:33:44:55:66", "New pen", db=db)
    assert exc.value.status_code == 409
    detail = (exc.value.detail or "").lower()
    # Message must point the admin at the resolution path. Either wording
    # ("limit reached" / "raise the limit" / "unbind") is fine.
    assert "limit" in detail or "unbind" in detail


@pytest.mark.asyncio
async def test_admin_assign_idempotent_for_same_pen():
    """Re-assigning the same MAC to the same student is a rename, not a conflict."""
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await _seed_student(db, username="alice", student_id="STU-1")
    await db["pens"].insert_one(
        {
            "user_id": "alice",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "pen_name": "Old name",
            "status": "active",
        }
    )
    result = await _call_admin_assign("STU-1", "AA:BB:CC:DD:EE:01", "New label", db=db)
    assert result.pen_name == "New label"
    assert result.pen_mac == "AA:BB:CC:DD:EE:01"


@pytest.mark.asyncio
async def test_admin_assign_after_unbind_succeeds():
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await _seed_student(db, username="alice", student_id="STU-1")
    await db["pens"].insert_one(
        {
            "user_id": "alice",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "pen_name": "Old pen",
            "status": "deregistered",  # admin already unbound
        }
    )
    result = await _call_admin_assign("STU-1", "11:22:33:44:55:66", "New pen", db=db)
    assert result.pen_mac == "11:22:33:44:55:66"
    assert result.status == "active"


async def _call_admin_set_pen_limit(student_id: str, allowed: int, *, db):
    from unittest.mock import patch
    from api.v1.admin_pens_async import (
        admin_set_pen_limit,
        AdminSetPenLimitRequest,
    )

    with patch("api.v1.admin_pens_async.get_tenant_db_or_403", return_value=db):
        return await admin_set_pen_limit(
            student_id=student_id,
            payload=AdminSetPenLimitRequest(allowed_pen_count=allowed),
            db=None,  # type: ignore[arg-type]
            current_user={"db_name": "skb_test", "user_id": "admin-1"},
        )


@pytest.mark.asyncio
async def test_admin_set_pen_limit_persists():
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await db["students"].insert_one({"username": "alice", "student_id": "STU-1"})
    result = await _call_admin_set_pen_limit("STU-1", 3, db=db)
    assert result == {"student_id": "STU-1", "allowed_pen_count": 3}

    fresh = await db["students"].find_one({"student_id": "STU-1"})
    assert fresh["allowed_pen_count"] == 3


@pytest.mark.asyncio
async def test_admin_set_pen_limit_rejects_zero():
    from fastapi import HTTPException
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await db["students"].insert_one({"username": "alice", "student_id": "STU-1"})
    with pytest.raises(HTTPException) as exc:
        await _call_admin_set_pen_limit("STU-1", 0, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admin_set_pen_limit_rejects_negative():
    from fastapi import HTTPException
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    await db["students"].insert_one({"username": "alice", "student_id": "STU-1"})
    with pytest.raises(HTTPException) as exc:
        await _call_admin_set_pen_limit("STU-1", -2, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admin_set_pen_limit_404_for_unknown_student():
    from fastapi import HTTPException
    db = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["skb_test"]
    with pytest.raises(HTTPException) as exc:
        await _call_admin_set_pen_limit("STU-NONE", 2, db=db)
    assert exc.value.status_code == 404


def test_router_exports_expected_routes():
    """Smoke check that all four admin pen endpoints are wired."""
    from api.v1.admin_pens_async import router

    paths = {(route.path, tuple(sorted(route.methods))) for route in router.routes}
    assert ("/pens", ("GET",)) in paths
    assert ("/students/{student_id}/pens", ("GET",)) in paths
    assert ("/pens/{pen_mac}", ("DELETE",)) in paths
    assert ("/students/{student_id}/pens", ("POST",)) in paths
    assert ("/students/{student_id}/pen-limit", ("PATCH",)) in paths
