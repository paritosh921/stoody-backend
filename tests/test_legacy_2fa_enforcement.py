from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from api.v1 import auth_async, auth_cookie, totp_2fa
from middleware.tenant_middleware import TenantMiddleware


class _Collection:
    def __init__(self, document: dict | None = None):
        self.document = document
        self.find_one = AsyncMock(return_value=document)
        self.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))


class _TenantDb(dict):
    def __init__(self, *, admin_doc: dict | None = None, tutor_doc: dict | None = None):
        super().__init__()
        self["admins"] = _Collection(admin_doc)
        self["tutors"] = _Collection(tutor_doc)


class _AuthManager:
    def __init__(
        self,
        *,
        admin_data: dict | None = None,
        tutor_data: dict | None = None,
        token_user: dict | None = None,
    ):
        self.cache_manager = None
        self.authenticate_admin = AsyncMock(return_value=admin_data)
        self.authenticate_tutor = AsyncMock(return_value=tutor_data)
        self.verify_token_and_get_user = AsyncMock(return_value=token_user)
        self.create_user_session = AsyncMock(
            return_value={"access_token": "legacy-token", "user": {"user_id": "user-1"}}
        )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return True

    def get_password_hash(self, password: str) -> str:
        return f"hashed:{password}"

    async def invalidate_user_session(self, user_id: str) -> bool:
        return True


@pytest.fixture
def auth_dependencies(monkeypatch):
    tenant = {
        "tenant_id": "ABCD-1234",
        "db_name": "skb_abcd_1234",
        "institution_id": "inst-1",
        "subdomain": "school",
        "enabled_features": {"tutor_portal_access": True},
        "status": "active",
    }
    tenant_db = _TenantDb()
    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", AsyncMock(return_value=tenant))
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", AsyncMock(return_value=tenant_db))
    monkeypatch.setattr(auth_async, "record_auth_login", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auth_cookie, "_resolve_tenant_for_auth", AsyncMock(return_value=tenant))
    monkeypatch.setattr(auth_cookie, "_get_tenant_db_or_503", AsyncMock(return_value=tenant_db))
    monkeypatch.setattr(totp_2fa, "_resolve_tenant_for_auth", AsyncMock(return_value=tenant))
    monkeypatch.setattr(totp_2fa, "_get_tenant_db_or_503", AsyncMock(return_value=tenant_db))
    monkeypatch.setattr(
        "api.v1.totp_2fa.clear_user_session_revocation",
        AsyncMock(),
        raising=False,
    )
    return SimpleNamespace(tenant=tenant, tenant_db=tenant_db)


def _request():
    return SimpleNamespace(state=SimpleNamespace())


async def _call(endpoint, *args, **kwargs):
    handler = getattr(endpoint, "__wrapped__", endpoint)
    return await handler(*args, **kwargs)


def _route_paths(router) -> set[str]:
    return {getattr(route, "path", "") for route in router.routes}


def test_legacy_admin_tutor_login_routes_are_not_registered():
    assert "/admin/login" not in _route_paths(auth_async.router)
    assert "/tutor/login" not in _route_paths(auth_async.router)
    assert "/admin/cookie-login" not in _route_paths(auth_cookie.router)
    assert "/tutor/cookie-login" not in _route_paths(auth_cookie.router)


def test_legacy_admin_tutor_login_paths_are_not_tenant_middleware_exempt():
    assert "/api/v1/auth/admin/login" not in TenantMiddleware.EXEMPT_PATHS
    assert "/api/v1/auth/tutor/login" not in TenantMiddleware.EXEMPT_PATHS


def test_2fa_login_has_no_legacy_admin_email_exemption(auth_dependencies):
    asyncio.run(_2fa_login_has_no_legacy_admin_email_exemption(auth_dependencies))


async def _2fa_login_has_no_legacy_admin_email_exemption(auth_dependencies):
    admin_doc = {
        "_id": ObjectId(),
        "email": "cielknowledge@gmail.com",
        "password_hash": "hash",
        "is_active": True,
        "two_fa": {"required": True, "enabled": False},
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    auth_manager = _AuthManager(
        admin_data={
            "user_id": str(admin_doc["_id"]),
            "email": "cielknowledge@gmail.com",
            "user_type": "admin",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="cielknowledge@gmail.com",
            password="secret1",
            user_type="admin",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "SETUP_2FA"
    assert response.temp_token
    auth_manager.create_user_session.assert_not_awaited()


def test_2fa_login_defaults_existing_user_without_2fa_doc_to_setup(auth_dependencies):
    asyncio.run(_2fa_login_defaults_existing_user_without_2fa_doc_to_setup(auth_dependencies))


async def _2fa_login_defaults_existing_user_without_2fa_doc_to_setup(auth_dependencies):
    admin_doc = {
        "_id": ObjectId(),
        "email": "existing-admin@example.com",
        "password_hash": "hash",
        "is_active": True,
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    auth_manager = _AuthManager(
        admin_data={
            "user_id": str(admin_doc["_id"]),
            "email": "existing-admin@example.com",
            "user_type": "admin",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="existing-admin@example.com",
            password="secret1",
            user_type="admin",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "SETUP_2FA"
    assert response.temp_token
    auth_manager.create_user_session.assert_not_awaited()


def test_2fa_login_allows_done_when_requirement_is_disabled(auth_dependencies):
    asyncio.run(_2fa_login_allows_done_when_requirement_is_disabled(auth_dependencies))


async def _2fa_login_allows_done_when_requirement_is_disabled(auth_dependencies):
    tutor_doc = {
        "_id": ObjectId(),
        "username": "teacher1",
        "username_lower": "teacher1",
        "password_hash": "hash",
        "is_active": True,
        "two_fa": {"required": False, "enabled": True, "secret_enc": "secret"},
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    auth_manager = _AuthManager(
        tutor_data={
            "user_id": str(tutor_doc["_id"]),
            "username": "teacher1",
            "user_type": "tutor",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="teacher1",
            password="secret1",
            user_type="tutor",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "DONE"
    assert response.access_token == "legacy-token"
    auth_manager.create_user_session.assert_awaited_once()


def test_2fa_admin_login_returns_password_change_requirement(auth_dependencies):
    asyncio.run(_2fa_admin_login_returns_password_change_requirement(auth_dependencies))


async def _2fa_admin_login_returns_password_change_requirement(auth_dependencies):
    admin_doc = {
        "_id": ObjectId(),
        "email": "admin@example.com",
        "password_hash": "hash",
        "is_active": True,
        "requires_password_change": True,
        "two_fa": {"required": False, "enabled": False},
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    auth_manager = _AuthManager(
        admin_data={
            "user_id": str(admin_doc["_id"]),
            "email": "admin@example.com",
            "user_type": "admin",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="admin@example.com",
            password="secret1",
            user_type="admin",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "DONE"
    assert response.user["requires_password_change"] is True
    session_user_data = auth_manager.create_user_session.await_args.args[0]
    assert session_user_data["requires_password_change"] is True


def test_admin_change_password_clears_password_change_requirement(auth_dependencies):
    asyncio.run(_admin_change_password_clears_password_change_requirement(auth_dependencies))


async def _admin_change_password_clears_password_change_requirement(auth_dependencies):
    admin_id = ObjectId()
    admin_doc = {
        "_id": admin_id,
        "email": "admin@example.com",
        "password_hash": "old-hash",
        "is_active": True,
        "requires_password_change": True,
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    db = SimpleNamespace(get_tenant_db=AsyncMock(return_value=auth_dependencies.tenant_db))
    auth_manager = _AuthManager()

    response = await _call(
        auth_async.admin_change_password,
        _request(),
        auth_async.StudentChangePasswordRequest(
            current_password="generated-password",
            new_password="NewStrong1!",
        ),
        current_user={
            "user_id": str(admin_id),
            "user_type": "admin",
            "db_name": "skb_abcd_1234",
        },
        db=db,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    update = auth_dependencies.tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert update["password_hash"] == "hashed:NewStrong1!"
    assert update["requires_password_change"] is False
    assert "password_changed_at" in update


def test_2fa_login_hard_bypasses_playstoreteacher(auth_dependencies):
    asyncio.run(_2fa_login_hard_bypasses_playstoreteacher(auth_dependencies))


async def _2fa_login_hard_bypasses_playstoreteacher(auth_dependencies):
    tutor_doc = {
        "_id": ObjectId(),
        "username": "playstoreteacher",
        "username_lower": "playstoreteacher",
        "password_hash": "hash",
        "is_active": True,
        "two_fa": {"required": True, "enabled": False},
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    auth_manager = _AuthManager(
        tutor_data={
            "user_id": str(tutor_doc["_id"]),
            "username": "playstoreteacher",
            "user_type": "tutor",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="playstoreteacher",
            password="secret1",
            user_type="tutor",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "DONE"
    assert response.temp_token is None
    assert response.access_token == "legacy-token"
    auth_manager.create_user_session.assert_awaited_once()


def test_2fa_status_hard_bypasses_playstoreteacher(auth_dependencies):
    asyncio.run(_2fa_status_hard_bypasses_playstoreteacher(auth_dependencies))


async def _2fa_status_hard_bypasses_playstoreteacher(auth_dependencies):
    tutor_id = ObjectId()
    tutor_doc = {
        "_id": tutor_id,
        "username": "playstoreteacher",
        "username_lower": "playstoreteacher",
        "two_fa": {"required": True, "enabled": True, "secret_enc": "secret"},
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    db = SimpleNamespace(get_tenant_db=AsyncMock(return_value=auth_dependencies.tenant_db))
    auth_manager = _AuthManager(
        token_user={
            "user_id": str(tutor_id),
            "user_type": "tutor",
            "db_name": "skb_abcd_1234",
        }
    )

    response = await totp_2fa.get_2fa_status(
        _request(),
        credentials=SimpleNamespace(credentials="token"),
        db=db,
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.two_fa_enabled is False
    assert response.two_fa_required is False


def test_playstoreteacher_cannot_enable_2fa_requirement(auth_dependencies):
    asyncio.run(_playstoreteacher_cannot_enable_2fa_requirement(auth_dependencies))


async def _playstoreteacher_cannot_enable_2fa_requirement(auth_dependencies):
    tutor_id = ObjectId()
    tutor_doc = {
        "_id": tutor_id,
        "username": "playstoreteacher",
        "username_lower": "playstoreteacher",
        "two_fa": {"required": False, "enabled": False},
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    db = SimpleNamespace(get_tenant_db=AsyncMock(return_value=auth_dependencies.tenant_db))
    auth_manager = _AuthManager(
        token_user={
            "user_id": str(tutor_id),
            "user_type": "tutor",
            "db_name": "skb_abcd_1234",
        }
    )

    response = await totp_2fa.set_2fa_requirement(
        _request(),
        totp_2fa.RequirementUpdateRequest(required=True),
        credentials=SimpleNamespace(credentials="token"),
        db=db,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    assert response["two_fa_required"] is False
    assert response["two_fa_enabled"] is False
    update = auth_dependencies.tenant_db["tutors"].update_one.await_args.args[1]["$set"]
    assert update["two_fa.required"] is False
    assert update["two_fa.enabled"] is False
    assert update["two_fa.secret_enc"] is None


def test_current_user_can_toggle_2fa_requirement_without_deleting_secret(auth_dependencies):
    asyncio.run(_current_user_can_toggle_2fa_requirement_without_deleting_secret(auth_dependencies))


async def _current_user_can_toggle_2fa_requirement_without_deleting_secret(auth_dependencies):
    admin_id = ObjectId()
    admin_doc = {
        "_id": admin_id,
        "email": "admin@example.com",
        "two_fa": {"required": True, "enabled": True, "secret_enc": "encrypted-secret"},
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    db = SimpleNamespace(get_tenant_db=AsyncMock(return_value=auth_dependencies.tenant_db))
    auth_manager = _AuthManager(
        token_user={
            "user_id": str(admin_id),
            "user_type": "admin",
            "db_name": "skb_abcd_1234",
        }
    )

    response = await totp_2fa.set_2fa_requirement(
        _request(),
        totp_2fa.RequirementUpdateRequest(required=False),
        credentials=SimpleNamespace(credentials="token"),
        db=db,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    assert response["two_fa_required"] is False
    update = auth_dependencies.tenant_db["admins"].update_one.await_args.args[1]["$set"]
    assert update["two_fa.required"] is False
    assert "two_fa.secret_enc" not in update
    assert "two_fa.enabled" not in update


def test_password_verification_does_not_update_admin_last_login(auth_dependencies):
    asyncio.run(_password_verification_does_not_update_admin_last_login(auth_dependencies))


async def _password_verification_does_not_update_admin_last_login(auth_dependencies):
    admin_doc = {
        "_id": ObjectId(),
        "email": "admin@example.com",
        "password_hash": "hash",
        "is_active": True,
    }
    auth_dependencies.tenant_db["admins"] = _Collection(admin_doc)
    auth_manager = auth_async.AuthManager()
    auth_manager.check_auth_rate_limit = AsyncMock(return_value=(True, 9))
    auth_manager.verify_password = lambda _plain, _hashed: True

    result = await auth_manager.authenticate_admin(
        "admin@example.com",
        "secret1",
        db_manager=object(),
        db_override=auth_dependencies.tenant_db,
    )

    assert result is not None
    auth_dependencies.tenant_db["admins"].update_one.assert_not_awaited()


def test_password_verification_does_not_update_tutor_last_login(auth_dependencies):
    asyncio.run(_password_verification_does_not_update_tutor_last_login(auth_dependencies))


async def _password_verification_does_not_update_tutor_last_login(auth_dependencies):
    tutor_doc = {
        "_id": ObjectId(),
        "username": "teacher1",
        "username_lower": "teacher1",
        "password_hash": "hash",
        "is_active": True,
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    auth_manager = auth_async.AuthManager()
    auth_manager.check_auth_rate_limit = AsyncMock(return_value=(True, 9))
    auth_manager.verify_password = lambda _plain, _hashed: True

    result = await auth_manager.authenticate_tutor(
        "teacher1",
        "secret1",
        db_manager=object(),
        db_override=auth_dependencies.tenant_db,
    )

    assert result is not None
    auth_dependencies.tenant_db["tutors"].update_one.assert_not_awaited()


def test_2fa_direct_done_updates_last_login_when_requirement_disabled(auth_dependencies):
    asyncio.run(_2fa_direct_done_updates_last_login_when_requirement_disabled(auth_dependencies))


async def _2fa_direct_done_updates_last_login_when_requirement_disabled(auth_dependencies):
    tutor_doc = {
        "_id": ObjectId(),
        "username": "teacher2",
        "username_lower": "teacher2",
        "password_hash": "hash",
        "is_active": True,
        "two_fa": {"required": False, "enabled": True, "secret_enc": "secret"},
    }
    auth_dependencies.tenant_db["tutors"] = _Collection(tutor_doc)
    auth_manager = _AuthManager(
        tutor_data={
            "user_id": str(tutor_doc["_id"]),
            "username": "teacher2",
            "user_type": "tutor",
        }
    )

    response = await _call(
        totp_2fa.login_with_2fa,
        _request(),
        totp_2fa.LoginRequest(
            username="teacher2",
            password="secret1",
            user_type="tutor",
            tenant_id="ABCD-1234",
        ),
        db=object(),
        auth_manager=auth_manager,
    )

    assert response.success is True
    assert response.next == "DONE"
    update = auth_dependencies.tenant_db["tutors"].update_one.await_args.args[1]["$set"]
    assert "last_login" in update
