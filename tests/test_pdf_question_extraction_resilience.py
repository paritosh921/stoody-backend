import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.v1 import pdf_async


def _completion(payload, *, finish_reason="stop"):
    content = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False)
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.mark.asyncio
async def test_question_extraction_splits_truncated_batches_and_preserves_unicode(
    monkeypatch,
):
    calls = []

    async def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        calls.append(prompt)
        if "starts on pages [0, 1]" in prompt:
            return _completion('{"questions":[', finish_reason="length")
        if "starts on pages [0]" in prompt:
            return _completion(
                {
                    "questions": [
                        {
                            "number": "1",
                            "text": "Find √x",
                            "options": ["1", "2", "3", "4"],
                            "page": 0,
                            "has_figure": False,
                        }
                    ]
                }
            )
        if "starts on pages [1]" in prompt:
            return _completion(
                {
                    "questions": [
                        {
                            "number": "2",
                            "text": "Evaluate 10⁻²",
                            "options": ["0.1", "0.01", "1", "10"],
                            "page": 1,
                            "has_figure": False,
                        }
                    ]
                }
            )
        return _completion(
            {
                "questions": [
                    {
                        "number": "3",
                        "text": "Select C",
                        "options": ["A", "B", "C", "D"],
                        "page": 2,
                        "has_figure": False,
                    }
                ]
            }
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    monkeypatch.setattr(pdf_async, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(pdf_async, "GROQ_MODEL", "test-model")
    monkeypatch.setenv("QUESTION_EXTRACTION_PAGE_BATCH_SIZE", "2")
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **_kwargs: fake_client,
    )

    questions = await pdf_async.extract_questions_with_gpt(
        {
            "pages": [
                {"index": 0, "markdown": "1. Find √x", "images": []},
                {"index": 1, "markdown": "2. Evaluate 10⁻²", "images": []},
                {"index": 2, "markdown": "3. Select C", "images": []},
            ]
        },
        "Mathematics",
        "medium",
    )

    assert [question.metadata["question_number"] for question in questions] == [
        "1",
        "2",
        "3",
    ]
    assert questions[0].text == "Find √x"
    assert questions[1].text == "Evaluate 10⁻²"
    assert sum("starts on pages [0, 1]" in prompt for prompt in calls) == 2


def test_console_safe_text_escapes_math_for_cp1252(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", SimpleNamespace(encoding="cp1252"))
        safe = pdf_async._console_safe_text("√ and ⁻")

    assert safe == r"\u221a and \u207b"


@pytest.mark.asyncio
async def test_question_replacement_rolls_back_when_fallback_insert_fails():
    old_question = {"_id": "old-id", "document_id": "paper-1", "id": "old"}

    class Cursor:
        async def to_list(self, length=None):
            return [dict(item) for item in collection.items]

    class Collection:
        def __init__(self):
            self.items = [dict(old_question)]
            self.insert_attempt = 0

        def find(self, _filter):
            return Cursor()

        async def delete_many(self, query, **_kwargs):
            self.items = [
                item
                for item in self.items
                if item.get("document_id") != query.get("document_id")
            ]

        async def insert_many(self, docs, **_kwargs):
            self.insert_attempt += 1
            if self.insert_attempt == 1:
                self.items.append(dict(docs[0]))
                raise RuntimeError("simulated insert failure")
            self.items.extend(dict(doc) for doc in docs)
            return SimpleNamespace(inserted_ids=[doc["_id"] for doc in docs])

    class Client:
        async def start_session(self):
            raise RuntimeError("transactions are not supported")

    class TargetDB:
        client = Client()

        def __getitem__(self, name):
            assert name == "questions"
            return collection

    class DB:
        async def get_context_db(self):
            return TargetDB()

    collection = Collection()
    with pytest.raises(RuntimeError, match="simulated insert failure"):
        await pdf_async._replace_document_questions(
            db=DB(),
            is_b2c=False,
            document_id="paper-1",
            question_docs=[
                {
                    "_id": "new-id",
                    "document_id": "paper-1",
                    "id": "new",
                }
            ],
        )

    assert collection.items == [old_question]


@pytest.mark.asyncio
async def test_answer_sheet_worker_refreshes_question_ocr_state_before_mapping(
    monkeypatch,
):
    mapping = AsyncMock(
        return_value={
            "mapped_count": 1,
            "manual_review_count": 0,
            "mappings": [{"question_id": "q1"}],
            "summary": {"mapping_deferred": False},
        }
    )
    reconcile = AsyncMock(
        return_value={
            "summary": {
                "auto_applied_count": 1,
                "review_required_count": 0,
            },
            "candidates": [],
        }
    )

    class LayoutProvider:
        async def analyze(self, **_kwargs):
            return {"page_count": 1}

    class BlockNormalizer:
        def normalize(self, **_kwargs):
            return {"answers": [{"question_number": 1, "text": "C"}], "answer_count": 1}

    class MappingService:
        pass

    MappingService.map_full_document_blocks = mapping

    class ReconciliationService:
        pass

    ReconciliationService.reconcile = reconcile

    class Validator:
        def validate_answer_sheet(self, **_kwargs):
            return {
                "status": "trusted_draft",
                "score": 1.0,
                "manual_segmentation_recommended": False,
            }

    class DB:
        async def mongo_find_one(self, collection, query):
            assert collection == "documents"
            return {
                "document_id": query["document_id"],
                "ocr_status": "completed",
            }

        async def mongo_find(self, collection, query):
            assert collection == "questions"
            return [{"id": "q1", "document_id": query["document_id"]}]

        async def mongo_update_one(self, *_args, **_kwargs):
            return True

    class Cache:
        async def set(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(pdf_async, "DocumentLayoutProvider", LayoutProvider)
    monkeypatch.setattr(pdf_async, "AnswerSheetBlockNormalizer", BlockNormalizer)
    monkeypatch.setattr(pdf_async, "AnswerSheetMappingService", MappingService)
    monkeypatch.setattr(
        pdf_async,
        "AnswerKeyReconciliationService",
        ReconciliationService,
    )
    monkeypatch.setattr(
        pdf_async,
        "FullDocumentExtractionValidator",
        Validator,
    )
    monkeypatch.setattr(
        pdf_async,
        "call_sarvam_ocr",
        AsyncMock(
            return_value={
                "pages": [
                    {
                        "index": 0,
                        "markdown": "1. C",
                        "images": [],
                        "dimensions": {},
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(
        pdf_async,
        "refresh_answer_solution_coverage",
        AsyncMock(),
    )

    result = await pdf_async.run_answer_sheet_ocr_pipeline(
        document={
            "document_id": "paper-1",
            "answer_sheet_path": "answer.pdf",
            # This is the stale request-time state. The DB has since completed.
            "ocr_status": "processing",
        },
        file_content=b"%PDF",
        job_id="answer-job",
        processing_result={
            "job_id": "answer-job",
            "status": "processing",
            "progress": 20,
            "timestamp": pdf_async.datetime.utcnow(),
        },
        current_user={"user_id": "admin-1"},
        db=DB(),
        cache=Cache(),
    )

    assert result.status == "completed"
    mapping.assert_awaited_once()
    reconcile.assert_awaited_once()
