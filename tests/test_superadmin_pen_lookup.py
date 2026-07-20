from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest
from bson import ObjectId

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class _DbManager:
    def __init__(self, client, *, unavailable: set[str] | None = None):
        self.client = client
        self.unavailable = unavailable or set()

    async def get_master_db(self):
        return self.client["master"]

    async def get_tenant_db(self, db_name: str):
        if db_name in self.unavailable:
            return None
        return self.client[db_name]


@pytest.mark.asyncio
async def test_superadmin_pen_lookup_scans_assigned_tenants_only():
    from mongomock_motor import AsyncMongoMockClient
    from api.v1 import superadmin_async

    client = AsyncMongoMockClient()
    master_db = client["master"]
    admin_id = ObjectId()
    other_admin_id = ObjectId()
    selected_tenant_id = ObjectId()
    other_tenant_id = ObjectId()
    unavailable_tenant_id = ObjectId()
    foreign_tenant_id = ObjectId()
    now = datetime.utcnow()

    await master_db["tenants"].insert_many(
        [
            {
                "_id": selected_tenant_id,
                "assigned_superadmin_id": admin_id,
                "institution_id": "ABCD-0001",
                "institution_name": "Selected School",
                "db_name": "skb_abcd_0001",
                "status": "active",
            },
            {
                "_id": other_tenant_id,
                "assigned_superadmin_id": admin_id,
                "institution_id": "WXYZ-0002",
                "institution_name": "Other School",
                "db_name": "skb_wxyz_0002",
                "status": "active",
            },
            {
                "_id": unavailable_tenant_id,
                "assigned_superadmin_id": admin_id,
                "institution_id": "MISS-0003",
                "institution_name": "Unavailable School",
                "db_name": "skb_missing_0003",
                "status": "active",
            },
            {
                "_id": foreign_tenant_id,
                "assigned_superadmin_id": other_admin_id,
                "institution_id": "OTHR-0004",
                "institution_name": "Foreign School",
                "db_name": "skb_other_0004",
                "status": "active",
            },
        ]
    )

    selected_db = client["skb_abcd_0001"]
    await selected_db["students"].insert_one(
        {"username": "alice", "student_id": "STU-001", "full_name": "Alice Doe"}
    )
    await selected_db["pens"].insert_one(
        {
            "pen_id": "pen_selected",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "pen_name": "Alice Pen",
            "status": "active",
            "user_id": "alice",
            "last_registered_at": now,
            "created_at": now - timedelta(days=1),
        }
    )

    other_db = client["skb_wxyz_0002"]
    await other_db["students"].insert_one(
        {"username": "bob", "student_id": "STU-002", "name": "Bob Roy"}
    )
    await other_db["pens"].insert_one(
        {
            "pen_id": "pen_old",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "pen_name": "Old Bob Pen",
            "status": "deregistered",
            "user_id": "bob",
            "last_registered_at": now - timedelta(days=5),
            "created_at": now - timedelta(days=10),
            "deregistered_at": now - timedelta(days=2),
        }
    )

    foreign_db = client["skb_other_0004"]
    await foreign_db["pens"].insert_one(
        {
            "pen_id": "pen_foreign",
            "pen_mac": "AA:BB:CC:DD:EE:01",
            "status": "active",
            "user_id": "mallory",
            "last_registered_at": now,
        }
    )

    response = await superadmin_async.lookup_pen_globally_for_tenant(
        str(selected_tenant_id),
        pen_id="aa:bb:cc:dd:ee:01",
        include_deregistered=True,
        db=_DbManager(client, unavailable={"skb_missing_0003"}),
        admin={"admin_id": str(admin_id), "email": "owner@example.com"},
    )

    assert response["normalized_query"] == "AA:BB:CC:DD:EE:01"
    assert response["total_matches"] == 2
    assert response["scanned_tenants"] == 2
    assert response["skipped_tenants"] == 1
    assert [row["institution_name"] for row in response["results"]] == [
        "Selected School",
        "Other School",
    ]
    assert response["results"][0]["is_selected_tenant"] is True
    assert response["results"][0]["student_name"] == "Alice Doe"
    assert response["results"][1]["status"] == "deregistered"
    assert response["results"][1]["student_id"] == "STU-002"


@pytest.mark.asyncio
async def test_superadmin_pen_lookup_can_hide_deregistered_bindings():
    from mongomock_motor import AsyncMongoMockClient
    from api.v1 import superadmin_async

    client = AsyncMongoMockClient()
    master_db = client["master"]
    admin_id = ObjectId()
    tenant_id = ObjectId()
    await master_db["tenants"].insert_one(
        {
            "_id": tenant_id,
            "assigned_superadmin_id": admin_id,
            "institution_name": "Selected School",
            "db_name": "skb_abcd_0001",
        }
    )
    tenant_db = client["skb_abcd_0001"]
    await tenant_db["pens"].insert_many(
        [
            {"pen_id": "pen_shared", "pen_mac": "AA:BB:CC:DD:EE:01", "status": "active"},
            {"pen_id": "pen_shared", "pen_mac": "AA:BB:CC:DD:EE:02", "status": "deregistered"},
        ]
    )

    response = await superadmin_async.lookup_pen_globally_for_tenant(
        str(tenant_id),
        pen_id="pen_shared",
        include_deregistered=False,
        db=_DbManager(client),
        admin={"admin_id": str(admin_id), "email": "owner@example.com"},
    )

    assert response["total_matches"] == 1
    assert response["results"][0]["status"] == "active"


def test_superadmin_pen_lookup_route_is_registered_before_dynamic_tenant_route():
    from api.v1 import superadmin_async

    paths = [route.path for route in superadmin_async.router.routes]
    assert "/tenants/{tenant_id}/pen-lookup" in paths
    assert paths.index("/tenants/{tenant_id}/pen-lookup") < paths.index("/tenants/{tenant_id}")
