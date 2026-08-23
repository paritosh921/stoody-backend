from copy import deepcopy

import pytest

from api.v1 import pdf_async
from services.answer_mapping_lifecycle import (
    completed_answer_mapping_status,
    effective_answer_mapping_status,
)
from services.answer_solution_coverage_service import AnswerSolutionCoverageService


class _LifecycleCollection:
    def __init__(self, document):
        self.document = deepcopy(document)

    async def find_one_and_update(self, query, update, return_document=None):
        if (
            self.document.get("answer_mapping_status") == "mapping"
            and self.document.get("answer_mapping_lease_expires_at") is not None
        ):
            return None
        self.document.update(deepcopy(update.get("$set") or {}))
        for key in (update.get("$unset") or {}):
            self.document.pop(key, None)
        return deepcopy(self.document)

    async def update_one(self, query, update):
        self.document.update(deepcopy(update.get("$set") or {}))
        for key in (update.get("$unset") or {}):
            self.document.pop(key, None)
        return True


class _LifecycleDb:
    def __init__(self, document, questions=None, mappings=None):
        self.collection = _LifecycleCollection(document)
        self.questions = deepcopy(questions or [])
        self.mappings = deepcopy(mappings or [])

    async def get_context_db(self):
        return {"documents": self.collection}

    async def mongo_find(self, collection_name, query):
        if collection_name == "questions":
            return deepcopy(self.questions)
        if collection_name == "answer_question_mappings":
            return deepcopy(self.mappings)
        return []


def test_explicit_answer_mapping_lifecycle_is_authoritative():
    document = {
        "answer_sheet_path": "s3://bucket/answer.pdf",
        "answer_sheet_ocr_status": "completed",
        "ocr_status": "completed",
        "answer_sheet_mapped_answers_count": 5,
        "answer_mapping_status": "mapping",
    }

    assert effective_answer_mapping_status(document, question_count=5) == "mapping"


def test_legacy_answer_mapping_lifecycle_is_derived_without_migration():
    base = {
        "answer_sheet_path": "s3://bucket/answer.pdf",
        "answer_solution_mode": "upload",
    }

    assert effective_answer_mapping_status({**base, "answer_sheet_ocr_status": "processing"}) == "extracting"
    assert effective_answer_mapping_status({
        **base,
        "answer_sheet_ocr_status": "completed",
        "ocr_status": "processing",
    }) == "waiting_for_questions"
    assert effective_answer_mapping_status({
        **base,
        "answer_sheet_ocr_status": "completed",
        "ocr_status": "completed",
        "answer_sheet_mapped_answers_count": 5,
    }, question_count=5) == "completed"
    assert effective_answer_mapping_status({
        **base,
        "answer_sheet_ocr_status": "completed",
        "ocr_status": "completed",
        "answer_sheet_mapped_answers_count": 0,
    }, question_count=5) == "needs_review"


def test_completed_mapping_requires_full_coverage_without_manual_review():
    assert completed_answer_mapping_status(mapped_count=5, question_count=5) == "completed"
    assert completed_answer_mapping_status(
        mapped_count=5,
        question_count=5,
        manual_review_count=1,
    ) == "needs_review"
    assert completed_answer_mapping_status(mapped_count=4, question_count=5) == "needs_review"


def test_active_mapping_keeps_coverage_pending_even_with_partial_rows():
    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_sheet_path": "s3://bucket/answer.pdf",
            "answer_solution_mode": "upload",
            "answer_sheet_ocr_status": "completed",
            "answer_mapping_status": "mapping",
        },
        questions=[{"id": "q1"}, {"id": "q2"}],
        mappings=[{
            "question_id": "q1",
            "answer_text": "Teacher answer",
            "source": "answer_sheet_full_ocr",
            "review_status": "accepted",
        }],
    )

    assert result["answer_solution_coverage_status"] == "pending"
    assert "answer_mapping_pending" in result["answer_solution_coverage_summary"]["reasons"]


@pytest.mark.asyncio
async def test_mapping_claim_allows_only_one_credit_spending_worker():
    db = _LifecycleDb({
        "document_id": "paper-1",
        "answer_mapping_status": "waiting_for_questions",
    })

    first_run = await pdf_async._claim_answer_mapping_run(
        db=db,
        is_b2c=False,
        document_id="paper-1",
    )
    second_run = await pdf_async._claim_answer_mapping_run(
        db=db,
        is_b2c=False,
        document_id="paper-1",
    )

    assert first_run
    assert second_run is None
    assert db.collection.document["answer_mapping_status"] == "mapping"


@pytest.mark.asyncio
async def test_completed_answer_ocr_waits_visibly_for_question_extraction():
    db = _LifecycleDb({
        "document_id": "paper-1",
        "answer_sheet_path": "s3://bucket/answer.pdf",
        "answer_sheet_ocr_status": "completed",
    })

    result = await pdf_async.map_completed_answer_sheet_after_question_ocr(
        document=deepcopy(db.collection.document),
        current_user={},
        db=db,
    )

    assert result["status"] == "waiting_for_questions"
    assert db.collection.document["answer_mapping_status"] == "waiting_for_questions"
    assert db.collection.document["answer_mapping_progress"] == 55


@pytest.mark.asyncio
async def test_completed_stale_mappings_are_reused_without_another_model_call(monkeypatch):
    questions = [
        {"id": f"new-q{index}", "question_number": index}
        for index in range(1, 6)
    ]
    mappings = [
        {
            "mapping_id": f"paper-1:old-q{index}:a{index}",
            "question_id": f"old-q{index}",
            "answer_number": str(index),
            "answer_text": f"Teacher answer {index}",
            "source": "answer_sheet_full_ocr",
            "review_status": "needs_review" if index == 5 else "accepted",
            "manual_review_required": index == 5,
        }
        for index in range(1, 6)
    ]
    db = _LifecycleDb(
        {
            "document_id": "paper-1",
            "exam_mode": "pcr",
            "ocr_status": "completed",
            "answer_sheet_path": "s3://bucket/answer.pdf",
            "answer_solution_mode": "auto",
            "answer_sheet_ocr_status": "completed",
            "answer_mapping_status": "needs_review",
            "answer_sheet_mapped_answers_count": 4,
        },
        questions=questions,
        mappings=mappings,
    )

    class _MustNotRunMapper:
        def __init__(self, *args, **kwargs):
            raise AssertionError("existing mapped answers must not spend another model call")

    monkeypatch.setattr(pdf_async, "AnswerSheetMappingService", _MustNotRunMapper)

    result = await pdf_async.map_completed_answer_sheet_after_question_ocr(
        document=deepcopy(db.collection.document),
        current_user={},
        db=db,
    )

    assert result["status"] == "needs_review"
    assert result["mapped_count"] == 4
    assert result["solution_count"] == 5
    assert result["already_current"] is True
    assert result["summary"]["rebound_mapping_count"] == 5
    assert db.collection.document["answer_solution_mode"] == "upload"
    assert db.collection.document["answer_mapping_status"] == "needs_review"
