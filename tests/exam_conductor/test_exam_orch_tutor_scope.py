from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _admin_user() -> Dict[str, Any]:
    return {"user_id": "admin-1", "user_type": "admin", "db_name": "skb_test"}


def _tutor_user(tutor_id: str = "tut-1") -> Dict[str, Any]:
    return {
        "user_id": f"user-{tutor_id}",
        "user_type": "tutor",
        "tutor_id": tutor_id,
        "admin_id": "admin-1",
        "db_name": "skb_test",
    }


async def _seed_document(db, **overrides) -> None:
    doc = {
        "document_id": "doc-1",
        "title": "PCR Paper",
        "exam_mode": "pcr",
        "exam_finalized": True,
        "admin_id": "admin-1",
        "teacher_ids": ["tut-1"],
        "is_active": True,
    }
    doc.update(overrides)
    await db["documents"].insert_one(doc)


async def _create_exam(db, current_user, **body_overrides):
    from api.v1.exam_orch_async import ExamCreateRequest, create_exam

    body = ExamCreateRequest(
        exam_id=body_overrides.pop("exam_id", "exam-1"),
        exam_type=body_overrides.pop("exam_type", ""),
        prepared_document_id=body_overrides.pop("prepared_document_id", "doc-1"),
        **body_overrides,
    )
    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await create_exam(body=body, current_user=current_user, db=None)


async def _list_exams(db, current_user):
    from api.v1.exam_orch_async import list_exams

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await list_exams(current_user=current_user, db=None)


async def _get_exam(db, exam_id: str, current_user):
    from api.v1.exam_orch_async import get_exam

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await get_exam(exam_id=exam_id, current_user=current_user, db=None)


@pytest.mark.asyncio
async def test_create_exam_maps_prepared_document_to_tutor_owner():
    db = _fresh_db()
    await _seed_document(
        db,
        document_id="doc-1",
        exam_mode="dcr",
        admin_id="admin-doc",
        teacher_ids=["tut-1", "tut-2"],
    )

    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        exam_type="pcr",
        prepared_document_id="doc-1",
    )

    assert result.exam_type == "dcr"
    assert result.admin_id == "admin-doc"
    assert result.teacher_ids == ["tut-1", "tut-2"]
    assert result.created_by_tutor_id == "tut-1"


@pytest.mark.asyncio
async def test_tutor_cannot_create_exam_from_other_tutor_paper():
    db = _fresh_db()
    await _seed_document(db, teacher_ids=["tut-other"])

    with pytest.raises(HTTPException) as exc:
        await _create_exam(db, _tutor_user("tut-1"))

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_unfinalized_or_untyped_document_is_not_exam_ready():
    db = _fresh_db()
    await _seed_document(db, document_id="draft", exam_finalized=False)
    await _seed_document(db, document_id="untyped", exam_mode=None)

    for document_id in ("draft", "untyped"):
        with pytest.raises(HTTPException) as exc:
            await _create_exam(db, _admin_user(), prepared_document_id=document_id)
        assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_tutor_list_and_get_are_scoped_to_visible_exams():
    db = _fresh_db()
    await db["exampen_exams"].insert_many(
        [
            {"exam_id": "created", "created_by_tutor_id": "tut-1", "teacher_ids": []},
            {"exam_id": "assigned", "teacher_ids": ["tut-1"]},
            {"exam_id": "open", "teacher_ids": []},
            {"exam_id": "hidden", "teacher_ids": ["tut-2"]},
        ]
    )

    visible = await _list_exams(db, _tutor_user("tut-1"))

    assert [exam.exam_id for exam in visible.items] == ["created", "assigned", "open"]
    assert (await _get_exam(db, "assigned", _tutor_user("tut-1"))).exam_id == "assigned"
    with pytest.raises(HTTPException) as exc:
        await _get_exam(db, "hidden", _tutor_user("tut-1"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_see_exam_hidden_from_tutor():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {"exam_id": "hidden", "teacher_ids": ["tut-2"], "admin_id": "admin-1"}
    )

    result = await _get_exam(db, "hidden", _admin_user())

    assert result.exam_id == "hidden"
