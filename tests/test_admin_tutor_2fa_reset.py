from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi import HTTPException

from api.v1 import tutor_async


class _Db:
    def __init__(self, tutor_doc: dict | None):
        self.tutor_doc = tutor_doc
        self.mongo_find_one = AsyncMock(return_value=tutor_doc)
        self.mongo_update_one = AsyncMock(return_value=True)


def _admin_user() -> dict:
    return {
        "user_type": "admin",
        "user_id": str(ObjectId()),
        "email": "admin@example.com",
        "permissions": ["manage_tutors"],
    }


async def _call(endpoint, *args, **kwargs):
    handler = getattr(endpoint, "__wrapped__", endpoint)
    return await handler(*args, **kwargs)


def test_admin_can_reset_tutor_2fa_without_resetting_password(monkeypatch):
    asyncio.run(_admin_can_reset_tutor_2fa_without_resetting_password(monkeypatch))


async def _admin_can_reset_tutor_2fa_without_resetting_password(monkeypatch):
    tutor_id = "tutor-123"
    tutor_object_id = ObjectId()
    db = _Db(
        {
            "_id": tutor_object_id,
            "tutor_id": tutor_id,
            "password_hash": "existing-hash",
            "two_fa": {
                "enabled": True,
                "required": True,
                "secret_enc": "encrypted-secret",
                "temp_secret_enc": "encrypted-temp",
            },
        }
    )
    auth_manager = SimpleNamespace(
        invalidate_user_session=AsyncMock(return_value=True),
        cache_manager=object(),
    )
    revoke_user_session = AsyncMock()
    monkeypatch.setattr(tutor_async, "revoke_user_session", revoke_user_session)

    response = await _call(
        tutor_async.reset_tutor_2fa,
        request=None,
        tutor_id=tutor_id,
        current_user=_admin_user(),
        db=db,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    db.mongo_find_one.assert_awaited_once_with("tutors", {"tutor_id": tutor_id})
    update = db.mongo_update_one.await_args.args[2]["$set"]
    assert update["two_fa.enabled"] is False
    assert update["two_fa.required"] is False
    assert update["two_fa.secret_enc"] is None
    assert update["two_fa.temp_secret_enc"] is None
    assert update["two_fa.reset_reason"] == "admin_reset"
    assert "password_hash" not in update
    assert "requires_password_change" not in update
    auth_manager.invalidate_user_session.assert_awaited_once_with(str(tutor_object_id))
    revoke_user_session.assert_awaited_once_with(auth_manager.cache_manager, str(tutor_object_id))


def test_admin_tutor_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch):
    asyncio.run(_admin_tutor_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch))


async def _admin_tutor_2fa_reset_succeeds_if_session_revocation_fails(monkeypatch):
    tutor_id = "tutor-disabled"
    db = _Db(
        {
            "_id": ObjectId(),
            "tutor_id": tutor_id,
            "two_fa": {"enabled": False, "required": False, "secret_enc": None},
        }
    )
    auth_manager = SimpleNamespace(
        invalidate_user_session=AsyncMock(side_effect=RuntimeError("cache unavailable")),
        cache_manager=object(),
    )
    monkeypatch.setattr(
        tutor_async,
        "revoke_user_session",
        AsyncMock(side_effect=RuntimeError("redis unavailable")),
    )

    response = await _call(
        tutor_async.reset_tutor_2fa,
        request=None,
        tutor_id=tutor_id,
        current_user=_admin_user(),
        db=db,
        auth_manager=auth_manager,
    )

    assert response["success"] is True
    update = db.mongo_update_one.await_args.args[2]["$set"]
    assert update["two_fa.enabled"] is False
    assert update["two_fa.required"] is False


def test_admin_tutor_2fa_reset_returns_404_for_unknown_tutor():
    asyncio.run(_admin_tutor_2fa_reset_returns_404_for_unknown_tutor())


async def _admin_tutor_2fa_reset_returns_404_for_unknown_tutor():
    db = _Db(None)

    with pytest.raises(HTTPException) as exc_info:
        await _call(
            tutor_async.reset_tutor_2fa,
            request=None,
            tutor_id="missing",
            current_user=_admin_user(),
            db=db,
            auth_manager=None,
        )

    assert exc_info.value.status_code == 404
