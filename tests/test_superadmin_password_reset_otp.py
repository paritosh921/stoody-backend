import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bson import ObjectId


class _Collection:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.find_one_calls = []
        self.inserted = []
        self.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1, matched_count=1))

    async def find_one(self, query, *args, **kwargs):
        self.find_one_calls.append(query)
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    async def insert_one(self, doc):
        self.inserted.append(doc)

    async def count_documents(self, query):
        return len([
            doc for doc in self.docs + self.inserted
            if all(
                doc.get(k) == v
                for k, v in query.items()
                if not isinstance(v, dict)
            )
        ])


class _MasterDb(dict):
    def __init__(self, superadmin):
        super().__init__(
            super_admins=_Collection([superadmin]),
            password_reset_otps=_Collection([]),
        )


class _Db:
    def __init__(self, superadmin):
        self.master_db = _MasterDb(superadmin)

    async def get_master_db(self):
        return self.master_db


def test_superadmin_otp_request_searches_superadmins_only_and_sends_stored_email(monkeypatch):
    from api.v1 import superadmin_async

    asyncio.run(_superadmin_otp_request_searches_superadmins_only(monkeypatch, superadmin_async))


async def _superadmin_otp_request_searches_superadmins_only(monkeypatch, superadmin_async):
    superadmin = {
        "_id": ObjectId(),
        "email": "owner@example.com",
        "username": "owner",
        "name": "Owner",
        "is_active": True,
        "status": "active",
    }
    db = _Db(superadmin)
    sent = []

    async def fake_send_password_reset_otp(*, to_email, otp, username, role, expire_minutes):
        sent.append({"to_email": to_email, "role": role, "username": username})
        return True

    monkeypatch.setattr(superadmin_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    response = await superadmin_async.request_superadmin_password_reset_otp(
        superadmin_async.SuperAdminPasswordResetRequest(username="owner", email="owner@example.com"),
        db=db,
    )

    assert response.success is True
    assert db.master_db["super_admins"].find_one_calls == [{"email": "owner@example.com"}]
    assert db.master_db["password_reset_otps"].inserted[0]["role"] == "superadmin"
    assert db.master_db["password_reset_otps"].inserted[0]["tenant_id"] is None
    assert sent == [{"to_email": "owner@example.com", "role": "superadmin", "username": "Owner"}]


def test_superadmin_otp_request_respects_existing_cooldown(monkeypatch):
    from api.v1 import superadmin_async
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_superadmin_otp_request_respects_existing_cooldown(monkeypatch, superadmin_async, PasswordResetOtpManager))


async def _superadmin_otp_request_respects_existing_cooldown(monkeypatch, superadmin_async, PasswordResetOtpManager):
    admin_id = ObjectId()
    superadmin = {
        "_id": admin_id,
        "email": "owner@example.com",
        "username": "owner",
        "name": "Owner",
        "is_active": True,
        "status": "active",
    }
    db = _Db(superadmin)
    existing = PasswordResetOtpManager().create_otp_record(
        user_id=str(admin_id),
        email="owner@example.com",
        role="superadmin",
        tenant_id=None,
        otp="123456",
    )["record"]
    db.master_db["password_reset_otps"].docs.append(existing)
    sent = []

    async def fake_send_password_reset_otp(*, to_email, otp, username, role, expire_minutes):
        sent.append(to_email)
        return True

    monkeypatch.setattr(superadmin_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    response = await superadmin_async.request_superadmin_password_reset_otp(
        superadmin_async.SuperAdminPasswordResetRequest(username="owner", email="owner@example.com"),
        db=db,
    )

    assert response.success is True
    assert sent == []
    assert db.master_db["password_reset_otps"].inserted == []


def test_superadmin_otp_request_no_records_found_does_not_send_email(monkeypatch):
    from fastapi import HTTPException
    from api.v1 import superadmin_async

    asyncio.run(_superadmin_otp_request_no_records_found_does_not_send_email(monkeypatch, superadmin_async, HTTPException))


async def _superadmin_otp_request_no_records_found_does_not_send_email(monkeypatch, superadmin_async, HTTPException):
    superadmin = {
        "_id": ObjectId(),
        "email": "owner@example.com",
        "username": "owner",
        "name": "Owner",
        "is_active": True,
        "status": "active",
    }
    db = _Db(superadmin)
    sent = []

    async def fake_send_password_reset_otp(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(superadmin_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    try:
        await superadmin_async.request_superadmin_password_reset_otp(
            superadmin_async.SuperAdminPasswordResetRequest(username="wrong-owner", email="owner@example.com"),
            db=db,
        )
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "No records found"

    assert sent == []
    assert db.master_db["password_reset_otps"].inserted == []


def test_superadmin_router_uses_otp_paths_and_removes_old_request_flag_path():
    from api.v1 import superadmin_async

    paths = {route.path for route in superadmin_async.router.routes}

    assert "/password-reset/request" in paths
    assert "/password-reset/complete" in paths
    assert "/password/request-reset" not in paths


def test_superadmin_complete_consumes_otp_and_clears_reset_flags():
    from api.v1 import superadmin_async
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_superadmin_complete_consumes_otp(superadmin_async, PasswordResetOtpManager))


async def _superadmin_complete_consumes_otp(superadmin_async, PasswordResetOtpManager):
    admin_id = ObjectId()
    superadmin = {
        "_id": admin_id,
        "email": "owner@example.com",
        "username": "owner",
        "name": "Owner",
        "status": "active",
        "is_active": True,
        "password_hash": "old",
        "password_reset_requested": True,
    }
    db = _Db(superadmin)
    otp_record = PasswordResetOtpManager().create_otp_record(
        user_id=str(admin_id),
        email="owner@example.com",
        role="superadmin",
        tenant_id=None,
        otp="123456",
    )["record"]
    otp_record["_id"] = ObjectId()
    db.master_db["password_reset_otps"].docs.append(otp_record)

    response = await superadmin_async.complete_superadmin_password_reset_otp(
        superadmin_async.SuperAdminPasswordResetCompleteRequest(
            username="owner",
            email="owner@example.com",
            otp="123456",
            new_password="new-password-123",
        ),
        db=db,
    )

    assert response.success is True
    admin_update = db.master_db["super_admins"].update_one.await_args.args[1]
    assert admin_update["$set"]["requires_password_change"] is False
    assert admin_update["$set"]["password_reset_requested"] is False
    otp_update = db.master_db["password_reset_otps"].update_one.await_args.args[1]
    assert otp_update["$set"]["used"] is True
