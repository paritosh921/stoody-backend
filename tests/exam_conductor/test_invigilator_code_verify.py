from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _hub_user(hub_id: str = "HUB-1") -> Dict[str, Any]:
    return {"user_id": hub_id, "user_type": "hub", "hub_id": hub_id, "db_name": "skb_test"}


def _tutor_user(tutor_id: str = "TUT-1") -> Dict[str, Any]:
    return {
        "user_id": f"user-{tutor_id}",
        "user_type": "tutor",
        "tutor_id": tutor_id,
        "db_name": "skb_test",
    }


async def _seed(db) -> None:
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-1",
            "teacher_ids": ["TUT-1"],
            "hub_assignments": [{"hub_id": "HUB-1", "status": "active"}],
        }
    )
    await db["exampen_invigilator_codes"].insert_one(
        {
            "exam_id": "EXAM-1",
            "code": "ABC123",
            "used": False,
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        }
    )


async def _verify(db, current_user, code: str = "ABC123"):
    from api.v1.evalpen_invigilator_async import VerifyCodeRequest, verify_invigilator_code

    with patch("api.v1.evalpen_invigilator_async._get_tenant_db", return_value=db):
        return await verify_invigilator_code(
            body=VerifyCodeRequest(exam_id="EXAM-1", code=code),
            current_user=current_user,
            db=None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_hub_can_verify_code_for_assigned_exam():
    db = _fresh_db()
    await _seed(db)

    result = await _verify(db, _hub_user("HUB-1"))

    assert result.valid is True
    assert result.exam_id == "EXAM-1"


@pytest.mark.asyncio
async def test_hub_cannot_verify_code_for_unassigned_exam():
    db = _fresh_db()
    await _seed(db)

    with pytest.raises(HTTPException) as exc:
        await _verify(db, _hub_user("HUB-2"))

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_tutor_code_visibility_and_code_value_are_enforced():
    db = _fresh_db()
    await _seed(db)

    with pytest.raises(HTTPException) as exc:
        await _verify(db, _tutor_user("TUT-2"))
    assert exc.value.status_code == 403

    with pytest.raises(HTTPException) as exc:
        await _verify(db, _tutor_user("TUT-1"), code="BAD999")
    assert exc.value.status_code == 403
