import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock

import api.v1.pdf_async as pdf_async
from api.v1.pdf_async import (
    _build_test_series_activation_errors,
    _can_upload_answer_sheet,
    _missing_correct_answer_question_numbers,
    update_document_metadata,
)


class FakeContentDb:
    def __init__(self, document, questions=None):
        self.document = document
        self.questions = questions or []
        self.update = None

    async def mongo_find_one(self, collection, query):
        if collection == "documents" and query.get("document_id") == self.document.get("document_id"):
            return self.document
        return None

    async def mongo_find(self, collection, query):
        if collection == "questions" and query.get("document_id") == self.document.get("document_id"):
            return self.questions
        if collection == "answer_question_mappings":
            return []
        return []

    async def mongo_update_one(self, collection, query, update):
        self.update = {"collection": collection, "query": query, "update": update}
        return True


@pytest.mark.parametrize(
    ("document_type", "exam_mode", "expected"),
    [
        ("Test Series", None, True),
        ("Test Series", "pcr", True),
        ("Test Series", "PCR", True),
        ("Test Series", "dcr", False),
        ("Test Series", "tally", False),
        ("Practice Sets", "pcr", False),
    ],
)
def test_answer_sheet_upload_policy_allows_pcr_but_not_other_offline_modes(
    document_type,
    exam_mode,
    expected,
):
    assert _can_upload_answer_sheet(document_type, exam_mode) is expected


def test_mixed_content_activation_requires_question_categories_and_objective_answers_only():
    document = {
        "document_type": "Test Series",
        "question_type": "mixed",
        "total_minutes": 60,
    }
    questions = [
        {"question_number": 1, "question_type": "mcq", "correct_answer": ""},
        {"question_number": 2, "question_type": "subjective", "correct_answer": ""},
        {"question_number": 3, "question_type": "unclassified", "correct_answer": ""},
        {"question_number": 4, "correct_answer": ""},
    ]

    assert _missing_correct_answer_question_numbers(questions, document) == [1]

    errors = _build_test_series_activation_errors(
        document=document,
        questions=questions,
    )

    assert "1" in errors
    assert "Question category is not selected for: 3, 4" in errors


@pytest.mark.asyncio
async def test_update_document_metadata_persists_question_category_for_practice_and_test_docs():
    db = FakeContentDb(
        {
            "document_id": "DOC1",
            "document_type": "Practice Sets",
            "question_type": "mcq",
            "is_active": False,
        }
    )

    response = await update_document_metadata.__wrapped__(
        request=None,
        document_id="DOC1",
        metadata={"question_type": "mixed"},
        current_user={"user_id": "admin1"},
        db=db,
    )

    assert response["updated_fields"]["question_type"] == "mixed"
    assert db.update["update"]["$set"]["question_type"] == "mixed"


@pytest.mark.asyncio
async def test_update_document_metadata_rejects_mixed_category_for_chapter_notes():
    db = FakeContentDb(
        {
            "document_id": "DOC2",
            "document_type": "Chapter Notes",
            "question_type": "mcq",
            "is_active": False,
        }
    )

    with pytest.raises(HTTPException) as exc:
        await update_document_metadata.__wrapped__(
            request=None,
            document_id="DOC2",
            metadata={"question_type": "mixed"},
            current_user={"user_id": "admin1"},
            db=db,
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_finalized_pcr_document_allows_operational_activation_only(monkeypatch):
    db = FakeContentDb(
        {
            "document_id": "PCR1",
            "document_type": "Test Series",
            "exam_mode": "pcr",
            "question_type": "subjective",
            "total_minutes": 60,
            "is_active": False,
            "exam_finalized": True,
        },
        questions=[{"question_number": 1, "question_type": "subjective"}],
    )
    monkeypatch.setattr(pdf_async, "_notify_students_content_activated", AsyncMock())

    response = await update_document_metadata.__wrapped__(
        request=None,
        document_id="PCR1",
        metadata={"is_active": True},
        current_user={"user_id": "admin1"},
        db=db,
    )

    assert response["updated_fields"]["is_active"] is True
    assert db.update["update"]["$set"]["is_active"] is True

    with pytest.raises(HTTPException) as exc:
        await update_document_metadata.__wrapped__(
            request=None,
            document_id="PCR1",
            metadata={"title": "Changed after finalization"},
            current_user={"user_id": "admin1"},
            db=db,
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_document_metadata_rechecks_activation_when_active_category_changes():
    db = FakeContentDb(
        {
            "document_id": "DOC3",
            "document_type": "Test Series",
            "question_type": "subjective",
            "total_minutes": 60,
            "is_active": True,
        },
        questions=[
            {"question_number": 1, "question_type": "unclassified", "correct_answer": ""},
        ],
    )

    with pytest.raises(HTTPException) as exc:
        await update_document_metadata.__wrapped__(
            request=None,
            document_id="DOC3",
            metadata={"question_type": "mixed"},
            current_user={"user_id": "admin1"},
            db=db,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail["missing_question_category_numbers"] == [1]
