from __future__ import annotations

import json

import pytest

from core import ocr_service
from api.v1 import question_attempts


def _attempt(attempt_id: str) -> question_attempts.QuestionAttempt:
    return question_attempts.QuestionAttempt(
        id=attempt_id,
        question_text="What is 2 + 2?",
        lock_ts=1.0,
        status="completed",
        ai_final_answer="4",
        tenant_id="tenant-1",
    )


async def _authorize(_request):
    return {"tenant_id": "tenant-1"}


async def _tenant_db(_user):
    return "tenant-db"


@pytest.mark.asyncio
async def test_evaluate_answers_uses_one_labeled_multimodal_gate_call(monkeypatch):
    calls = []

    async def fake_gate(tenant_db, images, prompt, **kwargs):
        calls.append({"tenant_db": tenant_db, "images": images, "prompt": prompt, **kwargs})
        return json.dumps({
            "results": [
                {
                    "pen_id": "PEN-1",
                    "score": "correct",
                    "extracted_answer": "4",
                    "correct_answer": "4",
                    "feedback": "Correct",
                },
                {
                    "pen_id": "PEN-2",
                    "score": "incorrect",
                    "extracted_answer": "5",
                    "correct_answer": "4",
                    "feedback": "Check the addition",
                },
            ],
        })

    monkeypatch.setattr(ocr_service, "_gate_vision_images_call", fake_gate)
    service = ocr_service.OCRService()
    try:
        results = await service.evaluate_answers(
            "What is 2 + 2?",
            [
                {"label": "PEN-1", "image_b64": "image-one"},
                {"label": "PEN-2", "image_b64": "image-two"},
            ],
            tenant_db="tenant-db",
            correct_answer="4",
        )
    finally:
        await service.close()

    assert len(calls) == 1
    assert [image["label"] for image in calls[0]["images"]] == ["PEN-1", "PEN-2"]
    assert set(results) == {"PEN-1", "PEN-2"}
    assert results["PEN-1"]["score"] == "correct"
    assert results["PEN-2"]["score"] == "incorrect"


@pytest.mark.asyncio
async def test_evaluate_answers_returns_only_valid_unique_expected_labels(monkeypatch):
    async def fake_gate(*_args, **_kwargs):
        return json.dumps({
            "results": [
                {"pen_id": "PEN-1", "score": "unexpected"},
                {"pen_id": "UNKNOWN", "score": "correct"},
                {"pen_id": "PEN-1", "score": "incorrect"},
            ],
        })

    monkeypatch.setattr(ocr_service, "_gate_vision_images_call", fake_gate)
    service = ocr_service.OCRService()
    try:
        results = await service.evaluate_answers(
            "Question",
            [
                {"label": "PEN-1", "image_b64": "image-one"},
                {"label": "PEN-2", "image_b64": "image-two"},
            ],
        )
    finally:
        await service.close()

    assert set(results) == {"PEN-1"}
    assert results["PEN-1"]["score"] == "inconclusive"


@pytest.mark.asyncio
async def test_evaluate_all_chunks_responses_into_groups_of_six(monkeypatch):
    grouped_labels = []

    class FakeOCRService:
        async def evaluate_answers(self, question_text, answer_images, **_kwargs):
            assert question_text
            labels = [image["label"] for image in answer_images]
            grouped_labels.append(labels)
            return {
                label: {"success": True, "score": "correct"}
                for label in labels
            }

        async def evaluate_answer(self, **_kwargs):
            raise AssertionError("complete grouped results must not use individual fallback")

    attempt_id = "grouped-attempt"
    question_attempts._active_attempts[attempt_id] = _attempt(attempt_id)
    monkeypatch.setattr(question_attempts, "_require_smartboard_auth", _authorize)
    monkeypatch.setattr(question_attempts, "_resolve_tenant_db", _tenant_db)
    monkeypatch.setattr(ocr_service, "get_ocr_service", lambda: FakeOCRService())
    payload = question_attempts.EvaluateAllRequest(
        pen_images={f"PEN-{index}": f"image-{index}" for index in range(13)}
    )

    try:
        response = await question_attempts.evaluate_all_responses(
            object(), attempt_id, payload
        )
    finally:
        question_attempts._active_attempts.pop(attempt_id, None)

    assert [len(labels) for labels in grouped_labels] == [6, 6, 1]
    assert len(response.results) == 13
    assert response.success is True


@pytest.mark.asyncio
async def test_evaluate_all_falls_back_only_for_missing_grouped_labels(monkeypatch):
    fallback_images = []

    class FakeOCRService:
        async def evaluate_answers(self, question_text, answer_images, **_kwargs):
            assert question_text
            first = answer_images[0]["label"]
            return {first: {"success": True, "score": "correct"}}

        async def evaluate_answer(self, question_text, answer_image_b64, **_kwargs):
            assert question_text
            fallback_images.append(answer_image_b64)
            return {"success": True, "score": "partial"}

    attempt_id = "fallback-attempt"
    question_attempts._active_attempts[attempt_id] = _attempt(attempt_id)
    monkeypatch.setattr(question_attempts, "_require_smartboard_auth", _authorize)
    monkeypatch.setattr(question_attempts, "_resolve_tenant_db", _tenant_db)
    monkeypatch.setattr(ocr_service, "get_ocr_service", lambda: FakeOCRService())
    payload = question_attempts.EvaluateAllRequest(
        pen_images={"PEN-1": "image-1", "PEN-2": "image-2"}
    )

    try:
        response = await question_attempts.evaluate_all_responses(
            object(), attempt_id, payload
        )
    finally:
        question_attempts._active_attempts.pop(attempt_id, None)

    assert fallback_images == ["image-2"]
    assert response.results["PEN-1"].score == "correct"
    assert response.results["PEN-2"].score == "partial"
