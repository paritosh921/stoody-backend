from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.v1 import pdf_async


class _MappingReviewDb:
    def __init__(self, mappings):
        self.document = {
            "document_id": "paper-1",
            "admin_id": "admin-1",
            "exam_mode": "pcr",
            "answer_sheet_path": "s3://bucket/answers.pdf",
        }
        self.questions = [
            {"id": "q1", "document_id": "paper-1", "question_type": "subjective"},
            {"id": "q2", "document_id": "paper-1", "question_type": "subjective"},
            {"id": "q3", "document_id": "paper-1", "question_type": "subjective"},
        ]
        self.mappings = deepcopy(mappings)
        self.document_updates = []
        self.mapping_updates = []
        self.question_updates = []

    async def mongo_find_one(self, collection_name, query):
        if collection_name == "documents":
            return deepcopy(self.document)
        if collection_name == "answer_question_mappings":
            mapping_id = query.get("mapping_id")
            return deepcopy(next((item for item in self.mappings if item["mapping_id"] == mapping_id), None))
        if collection_name == "questions":
            question_id = query.get("id")
            return deepcopy(next((item for item in self.questions if item["id"] == question_id), None))
        return None

    async def mongo_find(self, collection_name, query):
        if collection_name == "questions":
            return deepcopy(self.questions)
        if collection_name == "answer_question_mappings":
            requested_ids = query.get("mapping_id", {}).get("$in")
            rows = self.mappings
            if requested_ids is not None:
                rows = [item for item in rows if item["mapping_id"] in requested_ids]
            return deepcopy(rows)
        return []

    async def mongo_update_many(self, collection_name, query, update):
        requested_ids = set(query["mapping_id"]["$in"])
        modified = 0
        for mapping in self.mappings:
            if mapping["mapping_id"] in requested_ids:
                mapping.update(deepcopy(update["$set"]))
                modified += 1
        return modified

    async def mongo_update_one(self, collection_name, query, update, upsert=False):
        if collection_name == "documents":
            self.document.update(deepcopy(update["$set"]))
            self.document_updates.append(deepcopy(update["$set"]))
            return True
        if collection_name == "answer_question_mappings":
            mapping = next(item for item in self.mappings if item["mapping_id"] == query["mapping_id"])
            mapping.update(deepcopy(update["$set"]))
            self.mapping_updates.append(deepcopy(update["$set"]))
            return True
        if collection_name == "questions":
            question = next(item for item in self.questions if item["id"] == query["id"])
            question.update(deepcopy(update["$set"]))
            self.question_updates.append(deepcopy(update["$set"]))
            return True
        return False


def _admin():
    return {"user_type": "admin", "user_id": "admin-1"}


def _request():
    return Request({"type": "http", "method": "PATCH", "path": "/", "headers": []})


@pytest.mark.asyncio
async def test_bulk_mapping_review_only_accepts_complete_non_rejected_solutions():
    db = _MappingReviewDb(
        [
            {
                "document_id": "paper-1",
                "mapping_id": "m1",
                "question_id": "q1",
                "answer_text": "A complete worked solution",
                "review_status": "needs_review",
                "manual_review_required": True,
            },
            {
                "document_id": "paper-1",
                "mapping_id": "m2",
                "question_id": "q2",
                "answer_text": "Teacher rejected this mapping",
                "review_status": "rejected",
                "manual_review_required": True,
            },
            {
                "document_id": "paper-1",
                "mapping_id": "m3",
                "question_id": "q3",
                "answer_text": "",
                "review_status": "needs_review",
                "manual_review_required": True,
            },
        ]
    )

    result = await pdf_async.bulk_update_document_answer_mapping_review(
        request=_request(),
        document_id="paper-1",
        review_request=pdf_async.BulkAnswerMappingReviewRequest(
            mappingIds=["m1", "m2", "m3"],
            confirmReviewed=True,
        ),
        current_user=_admin(),
        db=db,
    )

    assert result["eligibleCount"] == 1
    assert result["updatedCount"] == 1
    assert result["skippedCount"] == 2
    assert db.mappings[0]["review_status"] == "accepted"
    assert db.mappings[0]["manual_review_required"] is False
    assert db.mappings[1]["review_status"] == "rejected"
    assert db.mappings[2]["review_status"] == "needs_review"
    assert db.document_updates


@pytest.mark.asyncio
async def test_single_mapping_cannot_accept_an_empty_worked_solution():
    db = _MappingReviewDb(
        [
            {
                "document_id": "paper-1",
                "mapping_id": "m1",
                "question_id": "q1",
                "answer_text": "",
                "review_status": "needs_review",
                "manual_review_required": True,
            }
        ]
    )

    with pytest.raises(HTTPException) as error:
        await pdf_async.update_document_answer_mapping_review(
            document_id="paper-1",
            mapping_id="m1",
            review_request=pdf_async.AnswerMappingReviewRequest(reviewStatus="accepted"),
            current_user=_admin(),
            db=db,
        )

    assert error.value.status_code == 422
    assert db.mapping_updates == []


@pytest.mark.asyncio
async def test_editing_a_mapping_updates_the_marking_source_and_keeps_the_original():
    db = _MappingReviewDb(
        [
            {
                "document_id": "paper-1",
                "mapping_id": "m1",
                "question_id": "q1",
                "answer_text": "OCR text",
                "final_answer_text": "Old final",
                "review_status": "needs_review",
                "manual_review_required": True,
            }
        ]
    )

    await pdf_async.update_document_answer_mapping_review(
        document_id="paper-1",
        mapping_id="m1",
        review_request=pdf_async.AnswerMappingReviewRequest(
            reviewStatus="accepted",
            answerText="Teacher corrected worked solution",
        ),
        current_user=_admin(),
        db=db,
    )

    mapping = db.mappings[0]
    assert mapping["answer_text"] == "Teacher corrected worked solution"
    assert mapping["final_answer_text"] == "Teacher corrected worked solution"
    assert mapping["original_answer_text"] == "OCR text"
    assert mapping["original_final_answer_text"] == "Old final"
    assert mapping["review_status"] == "accepted"
    assert mapping["manual_review_required"] is False


@pytest.mark.asyncio
async def test_subjective_mapping_candidate_is_not_applied_as_an_objective_key():
    db = _MappingReviewDb(
        [
            {
                "document_id": "paper-1",
                "mapping_id": "m1",
                "question_id": "q1",
                "answer_text": "A complete worked solution",
                "correct_answer_candidate": "B",
                "review_status": "needs_review",
                "manual_review_required": True,
            }
        ]
    )

    result = await pdf_async.update_document_answer_mapping_review(
        document_id="paper-1",
        mapping_id="m1",
        review_request=pdf_async.AnswerMappingReviewRequest(
            reviewStatus="accepted",
            correctAnswer="B",
        ),
        current_user=_admin(),
        db=db,
    )

    assert result["appliedCorrectAnswer"] is None
    assert "correct_answer" not in db.questions[0]
    assert db.question_updates == []


@pytest.mark.asyncio
async def test_objective_mapping_candidate_is_applied_to_the_saved_answer_key():
    db = _MappingReviewDb(
        [
            {
                "document_id": "paper-1",
                "mapping_id": "m1",
                "question_id": "q1",
                "answer_text": "B",
                "correct_answer_candidate": "B",
                "review_status": "needs_review",
                "manual_review_required": True,
            }
        ]
    )
    db.questions[0]["question_type"] = "mcq"

    result = await pdf_async.update_document_answer_mapping_review(
        document_id="paper-1",
        mapping_id="m1",
        review_request=pdf_async.AnswerMappingReviewRequest(
            reviewStatus="accepted",
            correctAnswer="B",
        ),
        current_user=_admin(),
        db=db,
    )

    assert result["appliedCorrectAnswer"] == "B"
    assert db.questions[0]["correct_answer"] == "B"
    assert db.question_updates[0]["correct_answer_source"] == "answer_sheet_mapping_review"
