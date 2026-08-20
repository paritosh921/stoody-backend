from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

from api.v1 import pdf_async


class _ConcurrentOcrDb:
    """Model the answer job attempting to write after question OCR completes."""

    def __init__(self) -> None:
        self.document = {
            "document_id": "paper-1",
            "exam_mode": "pcr",
            "ocr_status": "processing",
            "answer_sheet_ocr_status": "completed",
            "answer_sheet_ocr_completed_at": datetime(2026, 8, 20, 10, 0, 0),
            "answer_solution_mode": "upload",
            "answer_sheet_path": "s3://bucket/french-answer-key.pdf",
        }
        self.questions = []
        self.mappings = [
            {
                "document_id": "paper-1",
                "mapping_id": f"m{index}",
                "question_id": f"q{index}",
                "answer_text": f"Solution {index}",
                "review_status": "accepted",
                "manual_review_required": False,
            }
            for index in range(1, 6)
        ]
        self.attempted_statuses = []

    async def mongo_find_one(self, collection_name, query):
        if collection_name == "documents":
            return deepcopy(self.document)
        return None

    async def mongo_find(self, collection_name, query):
        if collection_name == "questions":
            return deepcopy(self.questions)
        if collection_name == "answer_question_mappings":
            return deepcopy(self.mappings)
        return []

    async def mongo_update_one(self, collection_name, query, update, upsert=False):
        assert collection_name == "documents"
        desired = deepcopy(update["$set"])
        self.attempted_statuses.append(desired["answer_solution_coverage_status"])

        if len(self.attempted_statuses) == 1:
            # Question OCR commits between the answer job's calculation and its
            # cache write.  MongoDB's guarded update must reject the stale write.
            self.questions = [
                {
                    "id": f"q{index}",
                    "document_id": "paper-1",
                    "question_type": "subjective",
                }
                for index in range(1, 6)
            ]
            self.document.update(
                {
                    "ocr_status": "completed",
                    "ocr_completed_at": datetime(2026, 8, 20, 10, 0, 1),
                }
            )
            return False

        self.document.update(desired)
        return True


@pytest.mark.asyncio
async def test_stale_pending_coverage_cannot_overwrite_completed_question_ocr():
    db = _ConcurrentOcrDb()

    result = await pdf_async.refresh_answer_solution_coverage(
        db=db,
        is_b2c=False,
        document_id="paper-1",
        document=deepcopy(db.document),
        questions=[],
        mappings=deepcopy(db.mappings),
    )

    assert db.attempted_statuses == ["pending", "ready"]
    assert result["answer_solution_coverage_status"] == "ready"
    assert result["answer_solution_coverage_summary"]["question_count"] == 5
    assert result["answer_solution_coverage_summary"]["mapped_answer_count"] == 5
    assert db.document["answer_solution_coverage_status"] == "ready"

