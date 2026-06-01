import asyncio

import pytest
from fastapi import HTTPException

from api.v1.smartboard_sessions_async import (
    COLLECTION_NAME,
    _get_collection,
    get_tutor_id_from_user,
)


class _TenantDb:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        self.collections.setdefault(name, object())
        return self.collections[name]


class _Db:
    def __init__(self, tenant_db):
        self.tenant_db = tenant_db
        self.requested_db_names = []

    async def get_tenant_db(self, db_name):
        self.requested_db_names.append(db_name)
        return self.tenant_db


def test_get_tutor_id_from_user_prefers_tutor_id():
    assert get_tutor_id_from_user({"tutor_id": "tutor-1", "user_id": "user-1"}) == "tutor-1"


def test_get_tutor_id_from_user_falls_back_to_user_id():
    assert get_tutor_id_from_user({"user_id": "user-1"}) == "user-1"


def test_get_collection_uses_authenticated_users_tenant_db():
    tenant_db = _TenantDb()
    db = _Db(tenant_db)

    collection = asyncio.run(_get_collection(db, {"db_name": "skb_demo"}))

    assert collection is tenant_db.collections[COLLECTION_NAME]
    assert db.requested_db_names == ["skb_demo"]


def test_get_collection_rejects_users_without_tenant_db():
    db = _Db(_TenantDb())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_get_collection(db, {"user_type": "tutor"}))

    assert exc.value.status_code == 401
    assert db.requested_db_names == []


def test_get_collection_returns_503_when_tenant_db_unavailable():
    db = _Db(None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_get_collection(db, {"db_name": "skb_demo"}))

    assert exc.value.status_code == 503
    assert db.requested_db_names == ["skb_demo"]
