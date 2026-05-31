from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bson import ObjectId

from api.v1 import superadmin_async
from core.auth import AuthManager


class _Collection:
    def __init__(self, document: dict | None = None):
        self.document = document
        self.find_one = AsyncMock(return_value=document)
        self.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1, matched_count=1))


class _MasterDb(dict):
    def __init__(self, tenant: dict):
        super().__init__()
        self["tenants"] = _Collection(tenant)


class _Db:
    def __init__(self, *, tenant: dict, tenant_db):
        self.master_db = _MasterDb(tenant)
        self.tenant_db = tenant_db

    async def get_master_db(self):
        return self.master_db

    async def get_tenant_db(self, db_name: str):
        assert db_name == self.tenant_db["db_name"]
        return self.tenant_db


class _TenantDb(dict):
    def __init__(self, *, db_name: str, admin_doc: dict):
        super().__init__()
        self["db_name"] = db_name
        self["admins"] = _Collection(admin_doc)


def test_superadmin_password_reset_generates_copyable_admin_password(monkeypatch):
    asyncio.run(_superadmin_password_reset_generates_copyable_admin_password(monkeypatch))


async def _superadmin_password_reset_generates_copyable_admin_password(monkeypatch):
    tenant_id = ObjectId()
    admin_id = ObjectId()
    tenant = {
        "_id": tenant_id,
        "status": "active",
        "db_name": "skb_abcd_1234",
        "admin_email": "school-admin@example.com",
        "assigned_superadmin_id": ObjectId(),
    }
    tenant_db = _TenantDb(
        db_name="skb_abcd_1234",
        admin_doc={
            "_id": admin_id,
            "email": "school-admin@example.com",
            "role": "master_admin",
            "password_hash": "old-hash",
        },
    )
    db = _Db(tenant=tenant, tenant_db=tenant_db)
    superadmin = {"admin_id": str(tenant["assigned_superadmin_id"]), "email": "owner@example.com"}
    cache_manager = object()
    auth_manager = SimpleNamespace(
        get_password_hash=AuthManager().get_password_hash,
        invalidate_user_session=AsyncMock(return_value=True),
        cache_manager=cache_manager,
    )
    revoke_user_session = AsyncMock()
    monkeypatch.setattr(superadmin_async, "revoke_user_session", revoke_user_session)

    response = await superadmin_async.reset_tenant_admin_password(
        str(tenant_id),
        superadmin_async.ResetPasswordRequest(),
        db=db,
        admin=superadmin,
        auth_manager=auth_manager,
    )

    generated_password = response["generated_password"]
    assert response["success"] is True
    assert len(generated_password) >= 16

    update = tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert AuthManager().verify_password(generated_password, update["password_hash"])
    assert "two_fa.enabled" not in update
    assert "two_fa.secret_enc" not in update
    auth_manager.invalidate_user_session.assert_awaited_once_with(str(admin_id))
    revoke_user_session.assert_awaited_once_with(cache_manager, str(admin_id))


def test_superadmin_password_reset_succeeds_without_auth_manager():
    asyncio.run(_superadmin_password_reset_succeeds_without_auth_manager())


async def _superadmin_password_reset_succeeds_without_auth_manager():
    tenant_id = ObjectId()
    admin_id = ObjectId()
    tenant = {
        "_id": tenant_id,
        "status": "active",
        "db_name": "skb_abcd_1234",
        "admin_email": "school-admin@example.com",
        "assigned_superadmin_id": ObjectId(),
    }
    tenant_db = _TenantDb(
        db_name="skb_abcd_1234",
        admin_doc={
            "_id": admin_id,
            "email": "school-admin@example.com",
            "role": "master_admin",
            "password_hash": "old-hash",
        },
    )
    db = _Db(tenant=tenant, tenant_db=tenant_db)
    superadmin = {"admin_id": str(tenant["assigned_superadmin_id"]), "email": "owner@example.com"}

    response = await superadmin_async.reset_tenant_admin_password(
        str(tenant_id),
        superadmin_async.ResetPasswordRequest(),
        db=db,
        admin=superadmin,
        auth_manager=None,
    )

    assert response["success"] is True
    update = tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert AuthManager().verify_password(response["generated_password"], update["password_hash"])


def test_superadmin_2fa_reset_is_separate_from_password_reset(monkeypatch):
    asyncio.run(_superadmin_2fa_reset_is_separate_from_password_reset(monkeypatch))


async def _superadmin_2fa_reset_is_separate_from_password_reset(monkeypatch):
    tenant_id = ObjectId()
    admin_id = ObjectId()
    tenant = {
        "_id": tenant_id,
        "status": "active",
        "db_name": "skb_abcd_1234",
        "admin_email": "school-admin@example.com",
        "assigned_superadmin_id": ObjectId(),
    }
    tenant_db = _TenantDb(
        db_name="skb_abcd_1234",
        admin_doc={
            "_id": admin_id,
            "email": "school-admin@example.com",
            "role": "master_admin",
            "password_hash": "existing-hash",
            "two_fa": {"enabled": True, "required": True, "secret_enc": "encrypted-secret"},
        },
    )
    db = _Db(tenant=tenant, tenant_db=tenant_db)
    superadmin = {"admin_id": str(tenant["assigned_superadmin_id"]), "email": "owner@example.com"}
    cache_manager = object()
    auth_manager = SimpleNamespace(
        invalidate_user_session=AsyncMock(return_value=True),
        cache_manager=cache_manager,
    )
    revoke_user_session = AsyncMock()
    monkeypatch.setattr(superadmin_async, "revoke_user_session", revoke_user_session)

    response = await superadmin_async.reset_tenant_admin_2fa(
        str(tenant_id),
        db=db,
        admin=superadmin,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    update = tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert update["two_fa.enabled"] is False
    assert update["two_fa.required"] is False
    assert update["two_fa.secret_enc"] is None
    assert update["two_fa.temp_secret_enc"] is None
    assert "password_hash" not in update
    auth_manager.invalidate_user_session.assert_awaited_once_with(str(admin_id))
    revoke_user_session.assert_awaited_once_with(cache_manager, str(admin_id))


def test_superadmin_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch):
    asyncio.run(_superadmin_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch))


async def _superadmin_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch):
    tenant_id = ObjectId()
    admin_id = ObjectId()
    tenant = {
        "_id": tenant_id,
        "status": "active",
        "db_name": "skb_abcd_1234",
        "admin_email": "school-admin@example.com",
        "assigned_superadmin_id": ObjectId(),
    }
    tenant_db = _TenantDb(
        db_name="skb_abcd_1234",
        admin_doc={
            "_id": admin_id,
            "email": "school-admin@example.com",
            "role": "master_admin",
            "password_hash": "existing-hash",
            "two_fa": {"enabled": False, "required": False, "secret_enc": None},
        },
    )
    db = _Db(tenant=tenant, tenant_db=tenant_db)
    superadmin = {"admin_id": str(tenant["assigned_superadmin_id"]), "email": "owner@example.com"}
    auth_manager = SimpleNamespace(
        invalidate_user_session=AsyncMock(side_effect=RuntimeError("cache unavailable")),
        cache_manager=object(),
    )
    revoke_user_session = AsyncMock(side_effect=RuntimeError("redis unavailable"))
    monkeypatch.setattr(superadmin_async, "revoke_user_session", revoke_user_session)

    response = await superadmin_async.reset_tenant_admin_2fa(
        str(tenant_id),
        db=db,
        admin=superadmin,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    update = tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert update["two_fa.enabled"] is False
    assert update["two_fa.required"] is False


def test_superadmin_2fa_reset_succeeds_without_auth_manager():
    asyncio.run(_superadmin_2fa_reset_succeeds_without_auth_manager())


async def _superadmin_2fa_reset_succeeds_without_auth_manager():
    tenant_id = ObjectId()
    admin_id = ObjectId()
    tenant = {
        "_id": tenant_id,
        "status": "active",
        "db_name": "skb_abcd_1234",
        "admin_email": "school-admin@example.com",
        "assigned_superadmin_id": ObjectId(),
    }
    tenant_db = _TenantDb(
        db_name="skb_abcd_1234",
        admin_doc={
            "_id": admin_id,
            "email": "school-admin@example.com",
            "role": "master_admin",
            "password_hash": "existing-hash",
            "two_fa": {"enabled": True, "required": True, "secret_enc": "encrypted-secret"},
        },
    )
    db = _Db(tenant=tenant, tenant_db=tenant_db)
    superadmin = {"admin_id": str(tenant["assigned_superadmin_id"]), "email": "owner@example.com"}

    response = await superadmin_async.reset_tenant_admin_2fa(
        str(tenant_id),
        db=db,
        admin=superadmin,
        auth_manager=None,
    )

    assert response["success"] is True
    update = tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert update["two_fa.enabled"] is False
    assert update["two_fa.required"] is False
