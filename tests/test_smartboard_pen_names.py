"""Tests for the SmartBoard pen-name display map endpoint helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def mongo_db():
    client = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()
    return client["skb_test"]


@pytest.mark.asyncio
async def test_smartboard_pen_names_prefer_student_name_over_pen_label(mongo_db):
    from api.v1.smartboard_async import _load_smartboard_pen_names

    await mongo_db["students"].insert_one(
        {"username": "alice", "student_id": "STU-1", "full_name": "Alice Doe"}
    )
    await mongo_db["pens"].insert_one(
        {
            "user_id": "alice",
            "pen_mac": "aa:bb:cc:dd:ee:01",
            "pen_id": "pen-alice",
            "pen_name": "Alice's pen",
            "status": "active",
        }
    )

    rows = await _load_smartboard_pen_names(
        mongo_db,
        {
            "user_type": "admin",
            "db_name": "skb_test",
            "enabled_features": {"smartboard_cloud_access": True},
        },
    )

    assert len(rows) == 1
    assert rows[0].pen_mac == "AA:BB:CC:DD:EE:01"
    assert rows[0].name == "Alice Doe"
    assert rows[0].pen_name == "Alice's pen"
    assert rows[0].student_name == "Alice Doe"


@pytest.mark.asyncio
async def test_smartboard_pen_names_fall_back_to_pen_name_when_student_name_missing(mongo_db):
    from api.v1.smartboard_async import _load_smartboard_pen_names

    await mongo_db["students"].insert_one({"username": "bob", "student_id": "STU-2"})
    await mongo_db["pens"].insert_one(
        {
            "user_id": "bob",
            "pen_mac": "AA:BB:CC:DD:EE:02",
            "pen_id": "pen-bob",
            "pen_name": "Bob practice pen",
            "status": "active",
        }
    )

    rows = await _load_smartboard_pen_names(
        mongo_db,
        {
            "user_type": "admin",
            "db_name": "skb_test",
            "enabled_features": {"smartboard_cloud_access": True},
        },
    )

    assert len(rows) == 1
    assert rows[0].name == "Bob practice pen"
    assert rows[0].student_name is None


@pytest.mark.asyncio
async def test_smartboard_pen_names_scope_tutor_without_edit_permission(mongo_db):
    from api.v1.smartboard_async import _load_smartboard_pen_names

    await mongo_db["tutors"].insert_one(
        {"tutor_id": "TUT-1", "assigned_student_ids": ["STU-1"]}
    )
    await mongo_db["students"].insert_many(
        [
            {"username": "alice", "student_id": "STU-1", "full_name": "Alice Doe"},
            {"username": "bob", "student_id": "STU-2", "full_name": "Bob Roy"},
        ]
    )
    await mongo_db["pens"].insert_many(
        [
            {
                "user_id": "alice",
                "pen_mac": "AA:BB:CC:DD:EE:01",
                "pen_id": "pen-alice",
                "pen_name": "Alice's pen",
                "status": "active",
            },
            {
                "user_id": "bob",
                "pen_mac": "AA:BB:CC:DD:EE:02",
                "pen_id": "pen-bob",
                "pen_name": "Bob's pen",
                "status": "active",
            },
        ]
    )

    rows = await _load_smartboard_pen_names(
        mongo_db,
        {
            "user_type": "tutor",
            "tutor_id": "TUT-1",
            "admin_id": "507f1f77bcf86cd799439011",
            "db_name": "skb_test",
            "enabled_features": {"smartboard_cloud_access": True},
        },
    )

    assert [row.pen_mac for row in rows] == ["AA:BB:CC:DD:EE:01"]
    assert rows[0].name == "Alice Doe"


@pytest.mark.asyncio
async def test_smartboard_pen_names_do_not_include_requested_connected_macs_outside_tutor_scope(mongo_db):
    from api.v1.smartboard_async import _load_smartboard_pen_names

    await mongo_db["tutors"].insert_one(
        {"tutor_id": "TUT-1", "assigned_student_ids": ["STU-1"]}
    )
    await mongo_db["students"].insert_many(
        [
            {"username": "alice", "student_id": "STU-1", "full_name": "Alice Doe"},
            {"username": "lavyansh", "student_id": "STU-2", "full_name": "Lavyansh Mendiratta"},
        ]
    )
    await mongo_db["pens"].insert_many(
        [
            {
                "user_id": "alice",
                "pen_mac": "AA:BB:CC:DD:EE:01",
                "pen_id": "pen-alice",
                "pen_name": "Alice's pen",
                "status": "active",
            },
            {
                "user_id": "lavyansh",
                "pen_mac": "FB:A8:E7:D2:75:61",
                "pen_id": "pen-live",
                "pen_name": "XZY-A8FB(BLE)",
                "status": "active",
            },
        ]
    )

    rows = await _load_smartboard_pen_names(
        mongo_db,
        {
            "user_type": "tutor",
            "tutor_id": "TUT-1",
            "admin_id": "507f1f77bcf86cd799439011",
            "db_name": "skb_test",
            "enabled_features": {"smartboard_cloud_access": True},
        },
        requested_pen_macs={"FB:A8:E7:D2:75:61"},
    )

    assert rows == []


@pytest.mark.asyncio
async def test_smartboard_pen_names_filter_requested_macs_within_tutor_scope(mongo_db):
    from api.v1.smartboard_async import _load_smartboard_pen_names

    await mongo_db["tutors"].insert_one(
        {"tutor_id": "TUT-1", "assigned_student_ids": ["STU-1", "STU-2"]}
    )
    await mongo_db["students"].insert_many(
        [
            {"username": "alice", "student_id": "STU-1", "full_name": "Alice Doe"},
            {"username": "lavyansh", "student_id": "STU-2", "full_name": "Lavyansh Mendiratta"},
        ]
    )
    await mongo_db["pens"].insert_many(
        [
            {
                "user_id": "alice",
                "pen_mac": "AA:BB:CC:DD:EE:01",
                "pen_id": "pen-alice",
                "pen_name": "Alice's pen",
                "status": "active",
            },
            {
                "user_id": "lavyansh",
                "pen_mac": "FB:A8:E7:D2:75:61",
                "pen_id": "pen-live",
                "pen_name": "XZY-A8FB(BLE)",
                "status": "active",
            },
        ]
    )

    rows = await _load_smartboard_pen_names(
        mongo_db,
        {
            "user_type": "tutor",
            "tutor_id": "TUT-1",
            "admin_id": "507f1f77bcf86cd799439011",
            "db_name": "skb_test",
            "enabled_features": {"smartboard_cloud_access": True},
        },
        requested_pen_macs={"FB:A8:E7:D2:75:61"},
    )

    assert [row.pen_mac for row in rows] == ["FB:A8:E7:D2:75:61"]
    assert rows[0].name == "Lavyansh Mendiratta"


def test_smartboard_pen_names_requires_cloud_feature():
    from fastapi import HTTPException
    from api.v1.smartboard_async import require_smartboard_cloud_user

    with pytest.raises(HTTPException) as exc:
        require_smartboard_cloud_user({"user_type": "tutor", "enabled_features": {}})

    assert exc.value.status_code == 403
