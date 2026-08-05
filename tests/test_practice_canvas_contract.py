from __future__ import annotations

from copy import deepcopy

import pytest
from bson import ObjectId
from fastapi import HTTPException
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from api.v1.copy_sets_async import ensure_practice_copy_set_for_user
from api.v1.practice_async import QuestionPageRefsModel
from api.v1.strokes_async import _raise_sanitized_canvas_write_error
from api.v1.tutor_async import _question_page_identity_clauses


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = document.get(key)
        if isinstance(expected, dict) and "$exists" in expected:
            if (key in document) is not bool(expected["$exists"]):
                return False
        elif actual != expected:
            return False
    return True


class _FakeCopySets:
    def __init__(self, documents: list[dict] | None = None):
        self.documents = [deepcopy(document) for document in (documents or [])]

    async def find_one(self, query: dict, *args, **kwargs):
        return next(
            (
                deepcopy(document)
                for document in self.documents
                if _matches(document, query)
            ),
            None,
        )

    async def update_one(self, query: dict, update: dict):
        for document in self.documents:
            if not _matches(document, query):
                continue
            document.update(update.get("$set") or {})
            for key in update.get("$unset") or {}:
                document.pop(key, None)
            return

    async def find_one_and_update(
        self,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
        **kwargs,
    ):
        for document in self.documents:
            if not _matches(document, query):
                continue
            document.update(update.get("$set") or {})
            return deepcopy(document)
        if not upsert:
            return None
        document = {**query, **(update.get("$setOnInsert") or {})}
        document["_id"] = ObjectId()
        self.documents.append(document)
        return deepcopy(document)


class _FakeTenantDb:
    def __init__(self, copy_sets: _FakeCopySets):
        self.copy_sets = copy_sets

    def __getitem__(self, name: str):
        assert name == "copy_sets"
        return self.copy_sets


class _FakeDatabaseManager:
    def __init__(self, copy_sets: _FakeCopySets):
        self.tenant_db = _FakeTenantDb(copy_sets)

    async def get_tenant_db(self, db_name: str):
        assert db_name == "tenant_test"
        return self.tenant_db


@pytest.mark.asyncio
async def test_practice_copy_resolver_claims_legacy_copy_and_is_idempotent():
    legacy_id = ObjectId()
    copy_sets = _FakeCopySets(
        [
            {
                "_id": legacy_id,
                "user_id": "student-1",
                "title": "Practice",
                "is_archived": True,
            }
        ]
    )
    database = _FakeDatabaseManager(copy_sets)
    current_user = {
        "user_id": "student-1",
        "db_name": "tenant_test",
        "user_type": "student",
    }

    first = await ensure_practice_copy_set_for_user(current_user, database)
    second = await ensure_practice_copy_set_for_user(current_user, database)

    assert first["_id"] == legacy_id
    assert second["_id"] == legacy_id
    assert second["purpose"] == "practice"
    assert second["is_archived"] is False
    assert len(copy_sets.documents) == 1


def test_question_page_identity_clauses_preserve_mixed_book_types():
    clauses = _question_page_identity_clauses(
        active_pages=[7, 8],
        book_type="MS",
        virtual_pages=[
            {"physicalPageNo": 7, "bookType": "MS"},
            {"physicalPageNo": 7, "bookType": "LS"},
            {"physicalPageNo": 7, "bookType": "MS"},
        ],
    )

    assert clauses == [
        {"book_type": "MS", "page_number": 7},
        {"book_type": "LS", "page_number": 7},
    ]


def test_question_page_identity_clauses_support_legacy_attempts():
    assert _question_page_identity_clauses(
        active_pages=[2, "3"],
        book_type="ls",
        virtual_pages=None,
    ) == [
        {"book_type": "LS", "page_number": 2},
        {"book_type": "LS", "page_number": 3},
    ]


def test_question_page_references_are_bounded_at_the_api_contract():
    with pytest.raises(ValidationError):
        QuestionPageRefsModel(activePages=list(range(51)))


def test_question_page_identity_clauses_cap_legacy_fanout():
    clauses = _question_page_identity_clauses(
        active_pages=list(range(100)),
        book_type="MS",
        virtual_pages=None,
    )

    assert len(clauses) == 50
    assert clauses[-1] == {"book_type": "MS", "page_number": 49}


def test_legacy_canvas_index_conflict_is_reported_as_schema_unavailable():
    duplicate = DuplicateKeyError(
        "duplicate key",
        details={
            "errmsg": (
                "E11000 duplicate key error collection: tenant.canvas_pages "
                "index: uniq_canvas_page dup key: { user_id: 'student-1' }"
            ),
            "keyValue": {"user_id": "student-1", "book_type": "MS", "page_number": 7},
        },
    )

    with pytest.raises(HTTPException) as raised:
        _raise_sanitized_canvas_write_error(duplicate)

    assert raised.value.status_code == 503
    assert "Run the copy-set migration" in str(raised.value.detail)
