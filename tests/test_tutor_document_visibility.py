from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi import HTTPException
from mongomock_motor import AsyncMongoMockClient

from api.v1 import pdf_async, tutor_async
from utils.tutor_scoping import (
    build_tutor_document_candidate_filter,
    get_tutor_document_access_context,
    student_matches_document_scope,
    tutor_can_access_document,
)


ADMIN_ID = ObjectId()
TUTOR_ID = "TUT-CLASS-6"
TUTOR_OBJECT_ID = ObjectId()


def _tutor_doc() -> dict:
    return {
        "_id": TUTOR_OBJECT_ID,
        "tutor_id": TUTOR_ID,
        "created_by": str(ADMIN_ID),
        "teaching_assignments": [
            {
                "standard": "6",
                "subject": "Physics",
                "sections": ["A"],
            }
        ],
    }


def _document(
    document_id: str,
    *,
    standard: str,
    subject: str = "Physics",
    section: str | None = None,
    teacher_ids: list[str] | None = None,
) -> dict:
    return {
        "_id": ObjectId(),
        "document_id": document_id,
        "title": document_id,
        "document_type": "Practice Sets",
        "standard": standard,
        "section": section,
        "subject": subject,
        "teacher_ids": teacher_ids or [],
        "admin_id": ADMIN_ID,
        "extracted_questions_count": 10,
        "is_active": True,
    }


class _AnalyticsDb:
    def __init__(self, documents: list[dict]):
        self.documents = documents

    async def mongo_find_one(self, collection: str, query: dict, **_kwargs):
        if collection == "tutors":
            return _tutor_doc()
        if collection == "documents":
            document_id = _query_document_id(query)
            return next(
                (doc for doc in self.documents if doc["document_id"] == document_id),
                None,
            )
        return None

    async def mongo_find(self, collection: str, _query: dict, **_kwargs):
        if collection == "documents":
            return self.documents
        if collection == "questions":
            return []
        return []

    async def mongo_aggregate(self, _collection: str, _pipeline: list[dict]):
        return []


class _MetadataDb:
    def __init__(self, document: dict):
        self.document = document
        self.mongo_update_one = AsyncMock(return_value=True)

    async def mongo_find_one(self, collection: str, _query: dict, **_kwargs):
        if collection == "documents":
            return self.document
        if collection == "tutors":
            return _tutor_doc()
        return None

    async def mongo_find(self, _collection: str, _query: dict, **_kwargs):
        return []


def _query_document_id(query: dict) -> str | None:
    if "document_id" in query:
        return query["document_id"]
    for condition in query.get("$and", []):
        if "document_id" in condition:
            return condition["document_id"]
    return None


def _current_user() -> dict:
    return {
        "user_type": "tutor",
        "tutor_id": TUTOR_ID,
        "user_id": str(TUTOR_OBJECT_ID),
        "admin_id": str(ADMIN_ID),
    }


@pytest.mark.asyncio
async def test_document_context_preserves_string_and_object_id_tenant_variants():
    context = await get_tutor_document_access_context(
        _current_user(),
        _AnalyticsDb([]),
    )

    assert context is not None
    assert str(ADMIN_ID) in context["admin_match_values"]
    assert ADMIN_ID in context["admin_match_values"]
    assert str(TUTOR_OBJECT_ID) in context["actor_ids"]


@pytest.mark.asyncio
async def test_database_candidate_filter_bounds_results_to_teacher_scope():
    context = await get_tutor_document_access_context(
        _current_user(),
        _AnalyticsDb([]),
    )
    assert context is not None

    collection = AsyncMongoMockClient()["scope_test"]["documents"]
    await collection.insert_many(
        [
            _document("CLASS6", standard="Class 6", section="A"),
            _document("CLASS10", standard="10", section="A"),
            _document("WRONGSUBJECT", standard="6", subject="Chemistry"),
        ]
    )
    cursor = collection.find(
        {
            "$and": [
                {"admin_id": {"$in": context["admin_match_values"]}},
                build_tutor_document_candidate_filter(context),
            ]
        }
    )
    results = await cursor.to_list(length=20)

    assert [document["document_id"] for document in results] == ["CLASS6"]


def test_unassigned_paper_requires_matching_class_subject_and_section():
    tutor = _tutor_doc()
    access = lambda doc: tutor_can_access_document(
        tutor,
        doc,
        tutor_id=TUTOR_ID,
        actor_ids=[str(TUTOR_OBJECT_ID)],
        admin_ids=[str(ADMIN_ID)],
    )

    assert access(_document("CLASS6", standard="Class 6", section="A")) is True
    assert access(_document("CLASS10", standard="10", section="A")) is False
    assert access(_document("WRONGSUBJECT", standard="6", subject="Chemistry")) is False
    assert access(_document("WRONGSECTION", standard="6", section="B")) is False


def test_explicit_assignment_still_cannot_cross_tenants():
    document = _document(
        "EXPLICIT",
        standard="10",
        teacher_ids=[TUTOR_ID],
    )
    document["admin_id"] = ObjectId()

    assert tutor_can_access_document(
        _tutor_doc(),
        document,
        tutor_id=TUTOR_ID,
        admin_ids=[str(ADMIN_ID)],
    ) is False


def test_student_roster_is_scoped_to_document_class_and_section():
    class_six_a = {"grade": "Class 6", "section": "a"}

    assert student_matches_document_scope(
        class_six_a,
        _document("ALL6", standard="6"),
    ) is True
    assert student_matches_document_scope(
        class_six_a,
        _document("6A", standard="6", section="A"),
    ) is True
    assert student_matches_document_scope(
        class_six_a,
        _document("6B", standard="6", section="B"),
    ) is False
    assert student_matches_document_scope(
        class_six_a,
        _document("10A", standard="10", section="A"),
    ) is False


@pytest.mark.asyncio
async def test_teacher_documents_feed_excludes_other_classes_and_uses_paper_roster(
    monkeypatch,
):
    documents = [
        _document("CLASS6", standard="6", section="A"),
        _document("CLASS10", standard="10", section="A"),
        _document("CLASS6B", standard="6", section="B"),
    ]
    students = [
        {"_id": ObjectId(), "student_id": "S6A", "grade": "6", "section": "A"},
        {"_id": ObjectId(), "student_id": "S6B", "grade": "6", "section": "B"},
        {"_id": ObjectId(), "student_id": "S10", "grade": "10", "section": "A"},
    ]
    monkeypatch.setattr(
        tutor_async,
        "_get_tutor_visible_students",
        AsyncMock(return_value=students),
    )

    response = await tutor_async.get_tutor_document_analytics.__wrapped__(
        request=None,
        current_user=_current_user(),
        db=_AnalyticsDb(documents),
    )

    assert [
        item["document_id"] for item in response["data"]["documents"]
    ] == ["CLASS6"]
    assert response["data"]["documents"][0]["total_visible_students"] == 1


@pytest.mark.asyncio
async def test_tutor_cannot_toggle_document_outside_teaching_scope():
    db = _MetadataDb(_document("CLASS10", standard="10", section="A"))

    with pytest.raises(HTTPException) as exc:
        await pdf_async.update_document_metadata.__wrapped__(
            request=None,
            document_id="CLASS10",
            metadata={"is_active": False},
            current_user=_current_user(),
            db=db,
        )

    assert exc.value.status_code == 403
    db.mongo_update_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_tutor_can_deactivate_document_inside_teaching_scope():
    db = _MetadataDb(_document("CLASS6", standard="6", section="A"))

    response = await pdf_async.update_document_metadata.__wrapped__(
        request=None,
        document_id="CLASS6",
        metadata={"is_active": False},
        current_user=_current_user(),
        db=db,
    )

    assert response["updated_fields"]["is_active"] is False
    db.mongo_update_one.assert_awaited_once()
