from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bson import ObjectId

from api.v1 import auth_async, totp_2fa
from core.auth import AuthManager


MOBILE_SESSION_DELTA = timedelta(days=14)
MOBILE_SESSION_SECONDS = int(MOBILE_SESSION_DELTA.total_seconds())
WEB_2FA_SECONDS = 6 * 60 * 60


class _Collection:
    def __init__(self, document: dict | None = None):
        self.document = document
        self.find_one = AsyncMock(return_value=document)
        self.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
        self.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))


class _TenantDb(dict):
    def __init__(self, *, student_doc: dict | None = None, tutor_doc: dict | None = None):
        super().__init__()
        self["students"] = _Collection(student_doc)
        self["tutors"] = _Collection(tutor_doc)
        self["pen_tokens"] = _Collection()
        self["student_activity_log"] = _Collection()


class _AuthManager:
    def __init__(self, *, tutor_data: dict | None = None):
        self.cache_manager = None
        self.authenticate_tutor = AsyncMock(return_value=tutor_data)
        self.create_user_session = AsyncMock(
            return_value={"access_token": "session-token", "user": {"user_id": "user-1"}}
        )

    def verify_password(self, _plain_password: str, _hashed_password: str) -> bool:
        return True


def _request(headers: dict[str, str] | None = None):
    return SimpleNamespace(
        headers=headers or {},
        state=SimpleNamespace(),
        client=SimpleNamespace(host="127.0.0.1"),
    )


async def _call(endpoint, *args, **kwargs):
    handler = getattr(endpoint, "__wrapped__", endpoint)
    return await handler(*args, **kwargs)


def _jwt_ttl_seconds(token: str) -> int:
    payload = AuthManager().decode_access_token(token)
    assert payload is not None
    return int(payload["exp"]) - int(payload["iat"])


def test_auth_manager_can_create_two_week_mobile_session_token():
    async def run():
        session = await AuthManager().create_user_session(
            {
                "user_id": "student-1",
                "user_type": "student",
                "username": "student1",
                "tenant_id": "ABCD-1234",
                "db_name": "skb_abcd_1234",
            },
            expires_delta=MOBILE_SESSION_DELTA,
        )

        assert session["expires_in"] == MOBILE_SESSION_SECONDS
        assert MOBILE_SESSION_SECONDS - 2 <= _jwt_ttl_seconds(session["access_token"]) <= MOBILE_SESSION_SECONDS + 2

    asyncio.run(run())


def test_student_mobile_login_requests_two_week_backend_session(monkeypatch):
    asyncio.run(_student_mobile_login_requests_two_week_backend_session(monkeypatch))


async def _student_mobile_login_requests_two_week_backend_session(monkeypatch):
    student_id = ObjectId()
    tenant = {
        "tenant_id": "ABCD-1234",
        "db_name": "skb_abcd_1234",
        "institution_id": "inst-1",
        "subdomain": "school",
        "enabled_features": {},
        "status": "active",
    }
    tenant_db = _TenantDb(
        student_doc={
            "_id": student_id,
            "username": "student1",
            "username_lower": "student1",
            "password_hash": "hash",
            "is_active": True,
            "admin_id": ObjectId(),
        }
    )
    auth_manager = _AuthManager()
    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", AsyncMock(return_value=tenant))
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", AsyncMock(return_value=tenant_db))
    monkeypatch.setattr(auth_async, "record_auth_login", lambda *_args, **_kwargs: None)

    await _call(
        auth_async.student_login,
        _request({"x-app-source": "stoody-mobile"}),
        auth_async.StudentLoginRequest(
            username="student1",
            password="secret1",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert auth_manager.create_user_session.await_args.kwargs["expires_delta"] == MOBILE_SESSION_DELTA


def test_2fa_mobile_direct_login_requests_two_week_backend_session(monkeypatch):
    asyncio.run(_2fa_mobile_direct_login_requests_two_week_backend_session(monkeypatch))


async def _2fa_mobile_direct_login_requests_two_week_backend_session(monkeypatch):
    tutor_id = ObjectId()
    tenant = {
        "tenant_id": "ABCD-1234",
        "db_name": "skb_abcd_1234",
        "institution_id": "inst-1",
        "subdomain": "school",
        "enabled_features": {"tutor_portal_access": True},
        "status": "active",
    }
    tutor_doc = {
        "_id": tutor_id,
        "username": "teacher1",
        "username_lower": "teacher1",
        "password_hash": "hash",
        "is_active": True,
        "two_fa": {"required": False, "enabled": False},
    }
    tenant_db = _TenantDb(tutor_doc=tutor_doc)
    auth_manager = _AuthManager(
        tutor_data={
            "user_id": str(tutor_id),
            "username": "teacher1",
            "user_type": "tutor",
        }
    )
    monkeypatch.setattr(totp_2fa, "_resolve_tenant_for_auth", AsyncMock(return_value=tenant))
    monkeypatch.setattr(totp_2fa, "_get_tenant_db_or_503", AsyncMock(return_value=tenant_db))
    monkeypatch.setattr(
        "api.v1.totp_2fa.clear_user_session_revocation",
        AsyncMock(),
        raising=False,
    )

    await _call(
        totp_2fa.login_with_2fa,
        _request({"x-app-source": "stoody-mobile"}),
        totp_2fa.LoginRequest(
            username="teacher1",
            password="secret1",
            user_type="tutor",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert auth_manager.create_user_session.await_args.kwargs["expires_delta"] == MOBILE_SESSION_DELTA


def test_2fa_mobile_otp_token_uses_two_week_expiry():
    token = totp_2fa.create_access_token_for_auth_request(
        AuthManager(),
        {
            "user_id": "teacher-1",
            "user_type": "tutor",
            "username": "teacher1",
            "tenant_id": "ABCD-1234",
            "db_name": "skb_abcd_1234",
        },
        _request({"x-app-source": "stoody-mobile"}),
    )

    assert MOBILE_SESSION_SECONDS - 2 <= _jwt_ttl_seconds(token) <= MOBILE_SESSION_SECONDS + 2


def test_2fa_web_otp_token_keeps_six_hour_expiry():
    token = totp_2fa.create_access_token_for_auth_request(
        AuthManager(),
        {
            "user_id": "teacher-1",
            "user_type": "tutor",
            "username": "teacher1",
            "tenant_id": "ABCD-1234",
            "db_name": "skb_abcd_1234",
        },
        _request(),
    )

    assert WEB_2FA_SECONDS - 2 <= _jwt_ttl_seconds(token) <= WEB_2FA_SECONDS + 2
