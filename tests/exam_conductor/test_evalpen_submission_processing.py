from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


def _tutor_user() -> Dict[str, Any]:
    return {
        "user_id": "user-TUT-1",
        "user_type": "tutor",
        "tutor_id": "TUT-1",
        "db_name": "skb_test",
    }


def _admin_user() -> Dict[str, Any]:
    return {
        "user_id": "ADMIN-1",
        "user_type": "admin",
        "db_name": "skb_test",
    }


class _FakeProcessor:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[str] = []

    async def process_submission(self, submission_id: str) -> Any:
        self.calls.append(submission_id)
        return self.result


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


async def _seed_exam_submission_visibility(db):
    await db["exampen_exams"].insert_many(
        [
            {
                "exam_id": "VISIBLE-1",
                "teacher_ids": ["TUT-1"],
                "created_by_tutor_id": "TUT-1",
                "admin_id": "ADMIN-1",
            },
            {
                "exam_id": "OPEN-1",
                "teacher_ids": [],
                "admin_id": "ADMIN-1",
            },
            {
                "exam_id": "HIDDEN-1",
                "teacher_ids": ["TUT-2"],
                "admin_id": "ADMIN-1",
            },
        ]
    )
    await db["evalpen_submissions"].insert_many(
        [
            {
                "submission_id": "SUB-VISIBLE",
                "exam_id": "VISIBLE-1",
                "student_id": "STU-1",
                "admin_id": "ADMIN-1",
                "source": "ble_pen",
                "segmentation_status": "pending",
                "submitted_at": 3,
            },
            {
                "submission_id": "SUB-OPEN",
                "exam_id": "OPEN-1",
                "student_id": "STU-2",
                "admin_id": "ADMIN-1",
                "source": "ble_pen",
                "segmentation_status": "pending",
                "submitted_at": 2,
            },
            {
                "submission_id": "SUB-HIDDEN",
                "exam_id": "HIDDEN-1",
                "student_id": "STU-3",
                "admin_id": "ADMIN-1",
                "source": "ble_pen",
                "segmentation_status": "pending",
                "submitted_at": 1,
            },
        ]
    )


class _FakeOCRAdapter:
    async def recognize_pages(self, _pages_data, *, source: str = "pen"):
        from api.v1._exampen_imports import load_exampen

        models = load_exampen("pcr.domain.response_models")
        ocr_service = load_exampen("pcr.services.ocr_service")

        return ocr_service.OCRResult(
            pages=[
                models.PageOCR(
                    page_number=1,
                    page_width_mm=210.0,
                    page_height_mm=297.0,
                    text_blocks=[
                        models.TextBlock(
                            text="Q.No 1.Ans The answer is forty two.",
                            bbox=models.BoundingBox(
                                x_min=10.0,
                                y_min=20.0,
                                x_max=180.0,
                                y_max=45.0,
                            ),
                            confidence=0.95,
                            source="pen",
                        ),
                    ],
                    source="pen",
                    mean_ocr_confidence=0.95,
                )
            ],
            source=source,
            metadata={"adapter": "fake"},
        )


class _UnmarkedProjectileOCRAdapter:
    """OCR fixture for a handwritten answer that omitted a Q.No marker."""

    async def recognize_pages(self, _pages_data, *, source: str = "camera"):
        from api.v1._exampen_imports import load_exampen

        models = load_exampen("pcr.domain.response_models")
        ocr_service = load_exampen("pcr.services.ocr_service")

        return ocr_service.OCRResult(
            pages=[
                models.PageOCR(
                    page_number=1,
                    page_width_mm=210.0,
                    page_height_mm=297.0,
                    text_blocks=[
                        models.TextBlock(
                            text=(
                                "Initial speed u = 20 m/s at theta = 37 degrees. "
                                "Time of flight, horizontal range, and maximum "
                                "height are calculated using g = 10."
                            ),
                            bbox=models.BoundingBox(
                                x_min=10.0,
                                y_min=20.0,
                                x_max=180.0,
                                y_max=70.0,
                            ),
                            confidence=0.95,
                            source="camera",
                        ),
                    ],
                    source="camera",
                    mean_ocr_confidence=0.95,
                )
            ],
            source=source,
            metadata={"adapter": "unmarked-projectile"},
        )


class _EmptyTextOCRAdapter:
    async def recognize_pages(self, _pages_data, *, source: str = "pen"):
        from api.v1._exampen_imports import load_exampen

        models = load_exampen("pcr.domain.response_models")
        ocr_service = load_exampen("pcr.services.ocr_service")

        return ocr_service.OCRResult(
            pages=[
                models.PageOCR(
                    page_number=1,
                    page_width_mm=210.0,
                    page_height_mm=297.0,
                    text_blocks=[],
                    source="pen",
                    mean_ocr_confidence=0.0,
                )
            ],
            source=source,
            metadata={"adapter": "empty-text"},
        )


class _FailingVisionGate:
    async def call(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


class _RecordingVisionGate:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(
            content='[{"text":"Q.No 1.Ans Visible answer","confidence":0.91}]'
        )


def test_ocr_model_resolver_uses_explicit_ocr_override(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("OCR_VISION_MODEL", "gpt-4o")

    assert ocr_service._get_ocr_vision_model() == "gpt-4o"


def test_ocr_model_resolver_uses_gate_provider_default(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    monkeypatch.delenv("OCR_VISION_MODEL", raising=False)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")

    assert ocr_service._get_ocr_vision_model() == "gemini-2.5-pro"


def test_ocr_model_resolver_ignores_blank_ocr_override(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    monkeypatch.setenv("OCR_VISION_MODEL", "   ")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    assert ocr_service._get_ocr_vision_model() == "gemini-2.5-flash"


def test_ocr_model_resolver_defaults_to_openai_when_provider_unset(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    monkeypatch.delenv("OCR_VISION_MODEL", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert ocr_service._get_ocr_vision_model() == "gpt-4o"


def test_ocr_model_resolver_falls_back_when_provider_import_fails(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    monkeypatch.delenv("OCR_VISION_MODEL", raising=False)
    with patch.object(
        ocr_service.importlib,
        "import_module",
        side_effect=ImportError("missing provider"),
    ):
        assert ocr_service._get_ocr_vision_model() == "gpt-4o"


@pytest.mark.asyncio
async def test_pen_ocr_adapter_raises_when_gate_call_fails():
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")
    adapter = ocr_service.LLMVisionPenAdapter(gate=_FailingVisionGate())

    with pytest.raises(RuntimeError, match="LLM Vision OCR failed"):
        await adapter.recognize_pages(
            [
                {
                    "page_number": 1,
                    "raw_strokes": [
                        {
                            "points": [
                                {"x": 10, "y": 10, "t": 1},
                                {"x": 20, "y": 20, "t": 2},
                            ]
                        }
                    ],
                }
            ],
            source="pen",
        )


@pytest.mark.asyncio
async def test_pen_ocr_adapter_logs_prompt_version_metadata(monkeypatch):
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")
    monkeypatch.setattr(ocr_service, "_get_ocr_vision_model", lambda: "gpt-4o")
    gate = _RecordingVisionGate()
    adapter = ocr_service.LLMVisionPenAdapter(gate=gate)

    result = await adapter.recognize_pages(
        [
            {
                "page_number": 1,
                "raw_strokes": [
                    {
                        "points": [
                            {"x": 10, "y": 10, "t": 1},
                            {"x": 20, "y": 20, "t": 2},
                        ]
                    }
                ],
            }
        ],
        source="pen",
    )

    assert result.pages[0].text_blocks[0].text == "Q.No 1.Ans Visible answer"
    metadata = gate.calls[0]["kwargs"]["metadata"]
    assert metadata["pcr_stage"] == "ocr_pen"
    assert metadata["stroke_count"] == 1
    assert metadata["ocr_prompt_version"] == "exampen-qno-v2"


@pytest.mark.asyncio
async def test_camera_ocr_resolves_private_student_copy_from_s3(monkeypatch):
    """PCR workers must read private S3 pages, not an API-worker disk path."""
    import base64

    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")
    download = AsyncMock(return_value=b"private-png-bytes")
    monkeypatch.setattr(ocr_service, "download_private_object", download)

    result = await ocr_service._resolve_image_base64(
        "s3://stoody-test/private/exampen/student-answer-copies/tenant/exam/attempt/page-1.png"
    )

    assert result == base64.b64encode(b"private-png-bytes").decode("ascii")
    assert download.await_args.kwargs["allowed_key_prefix"] == "private/exampen/"


def test_pen_stroke_renderer_crops_and_upscales_content():
    import base64
    import io

    from PIL import Image
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")
    image_b64 = ocr_service._render_strokes_to_base64(
        [
            {
                "points": [
                    {"x": 10, "y": 10, "t": 1},
                    {"x": 30, "y": 11, "t": 2},
                ],
                "strokeWidth": 2,
            }
        ],
        210.0,
        297.0,
    )

    assert image_b64
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
        assert img.width < 1240
        assert img.height < 1754
        assert img.width > 300
        assert img.height > 200


def test_vision_ocr_prompt_preserves_exampen_qno_markers():
    from api.v1._exampen_imports import load_exampen

    ocr_service = load_exampen("pcr.services.ocr_service")

    messages = ocr_service._build_vision_messages("aW1hZ2U=")
    prompt = messages[0]["content"][0]["text"]

    assert "camera photo, scanned PDF page" in prompt
    assert "'Q1', 'Q. 1', 'Question 1'" in prompt
    assert "preserve them" in prompt


@pytest.mark.asyncio
async def test_tutor_lists_submissions_for_visible_admin_owned_exams():
    from api.v1.evalpen_submissions_async import list_submissions

    db = _fresh_db()
    await _seed_exam_submission_visibility(db)

    with patch(
        "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
        new=AsyncMock(return_value=db),
    ):
        response = await list_submissions(
            current_user=_tutor_user(),
            db=object(),
        )

    submission_ids = {item["submission_id"] for item in response["items"]}
    assert submission_ids == {"SUB-VISIBLE", "SUB-OPEN"}


@pytest.mark.asyncio
async def test_admin_lists_admin_owned_submissions_only():
    from api.v1.evalpen_submissions_async import list_submissions

    db = _fresh_db()
    await db["evalpen_submissions"].insert_many(
        [
            {
                "submission_id": "SUB-ADMIN",
                "exam_id": "EXAM-1",
                "student_id": "STU-1",
                "admin_id": "ADMIN-1",
                "source": "ble_pen",
                "segmentation_status": "pending",
                "submitted_at": 2,
            },
            {
                "submission_id": "SUB-OTHER",
                "exam_id": "EXAM-2",
                "student_id": "STU-2",
                "admin_id": "ADMIN-2",
                "source": "ble_pen",
                "segmentation_status": "pending",
                "submitted_at": 1,
            },
        ]
    )

    with patch(
        "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
        new=AsyncMock(return_value=db),
    ):
        response = await list_submissions(
            current_user=_admin_user(),
            db=object(),
        )

    assert [item["submission_id"] for item in response["items"]] == ["SUB-ADMIN"]


@pytest.mark.asyncio
async def test_submission_repository_filters_multiple_exam_ids():
    from api.v1._exampen_imports import load_exampen

    db = _fresh_db()
    await _seed_exam_submission_visibility(db)
    repo = load_exampen("pcr.storage").SubmissionRepository(db)

    docs = await repo.list_submissions(exam_ids=["VISIBLE-1", "OPEN-1"])
    submission_ids = {doc["submission_id"] for doc in docs}

    assert submission_ids == {"SUB-VISIBLE", "SUB-OPEN"}


@pytest.mark.asyncio
async def test_tutor_cannot_process_hidden_submission_id():
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    await _seed_exam_submission_visibility(db)
    processor = _FakeProcessor(
        SimpleNamespace(
            submission_id="SUB-HIDDEN",
            page_count=1,
            response_count=1,
            inserted_count=1,
            duplicate_count=0,
            blocked_count=0,
            warning_count=0,
            error=None,
        )
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(return_value=processor),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await process_submission(
                "SUB-HIDDEN",
                current_user=_tutor_user(),
                db=object(),
            )

    assert exc_info.value.status_code == 403
    assert processor.calls == []


@pytest.mark.asyncio
async def test_tutor_can_process_visible_submission_id():
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    await _seed_exam_submission_visibility(db)
    processor = _FakeProcessor(
        SimpleNamespace(
            submission_id="SUB-VISIBLE",
            page_count=1,
            response_count=1,
            inserted_count=1,
            duplicate_count=0,
            blocked_count=0,
            warning_count=0,
            error=None,
        )
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(return_value=processor),
        ),
    ):
        response = await process_submission(
            "SUB-VISIBLE",
            current_user=_tutor_user(),
            db=object(),
        )

    assert processor.calls == ["SUB-VISIBLE"]
    assert response.submission_id == "SUB-VISIBLE"
    assert response.response_count == 1


@pytest.mark.asyncio
async def test_tutor_cannot_read_hidden_submission_responses():
    from api.v1.evalpen_submissions_async import get_submission_responses

    db = _fresh_db()
    await _seed_exam_submission_visibility(db)
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-HIDDEN",
            "submission_id": "SUB-HIDDEN",
            "exam_id": "HIDDEN-1",
            "student_id": "STU-3",
            "content_type": "TEXT_ONLY",
            "eval_status": "pending",
        }
    )

    with patch(
        "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
        new=AsyncMock(return_value=db),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_submission_responses(
                "SUB-HIDDEN",
                current_user=_tutor_user(),
                db=object(),
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_process_submission_route_runs_pcr_processor_without_client_answer_text():
    from api.v1.evalpen_submissions_async import process_submission

    result = SimpleNamespace(
        submission_id="SUB-1",
        page_count=2,
        response_count=3,
        inserted_count=3,
        duplicate_count=0,
        blocked_count=1,
        warning_count=2,
        error=None,
    )
    processor = _FakeProcessor(result)
    tenant_db = _fresh_db()
    await tenant_db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "STU-1",
            "admin_id": "ADMIN-1",
            "source": "ble_pen",
            "segmentation_status": "pending",
        }
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=tenant_db),
        ),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(return_value=processor),
        ),
    ):
        response = await process_submission(
            "SUB-1",
            current_user=_admin_user(),
            db=object(),
        )

    assert processor.calls == ["SUB-1"]
    assert response.submission_id == "SUB-1"
    assert response.segmentation_status == "complete"
    assert response.page_count == 2
    assert response.response_count == 3
    assert response.inserted_count == 3
    assert response.blocked_count == 1
    assert response.warning_count == 2


@pytest.mark.asyncio
async def test_process_submission_route_reports_processor_errors_as_bad_request():
    from api.v1.evalpen_submissions_async import process_submission

    processor = _FakeProcessor(
        SimpleNamespace(
            submission_id="SUB-1",
            page_count=0,
            response_count=0,
            inserted_count=0,
            duplicate_count=0,
            blocked_count=0,
            warning_count=0,
            error="No answer pages found for submission",
        )
    )
    tenant_db = _fresh_db()
    await tenant_db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "STU-1",
            "admin_id": "ADMIN-1",
            "source": "ble_pen",
            "segmentation_status": "pending",
        }
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=tenant_db),
        ),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(return_value=processor),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await process_submission(
                "SUB-1",
                current_user=_admin_user(),
                db=object(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No answer pages found for submission"


@pytest.mark.asyncio
async def test_eval_route_builds_eval_core_with_solution_cache():
    from api.v1.evalpen_evaluate_async import _build_eval_core

    db = _fresh_db()

    core = await _build_eval_core(db)

    assert hasattr(core, "evaluate_response")


@pytest.mark.asyncio
async def test_process_submission_route_reads_ingested_artifact_and_writes_detected_response():
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")

    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-1",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [
                    {
                        "points": [
                            {"x": 10, "y": 10, "t": 1},
                            {"x": 20, "y": 20, "t": 2},
                        ]
                    }
                ],
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_FakeOCRAdapter(),
        ),
    ):
        response = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    assert response.submission_id == ingest_result.submission_id
    assert response.segmentation_status == "complete"
    assert response.page_count == 1
    assert response.response_count >= 1
    assert response.inserted_count >= 1

    stored_submission = await db["evalpen_submissions"].find_one(
        {"submission_id": ingest_result.submission_id}
    )
    assert stored_submission is not None
    assert stored_submission["admin_id"] == "ADMIN-1"
    assert stored_submission["source"] == "ble_pen"

    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": ingest_result.submission_id}
    ).to_list(length=10)
    assert len(responses) == response.response_count
    assert responses[0]["exam_id"] == "EXAM-1"
    assert responses[0]["student_id"] == "STU-1"
    assert "forty two" in responses[0]["detected_text"]


@pytest.mark.asyncio
async def test_process_submission_maps_marker_to_canonical_session_question_id():
    """Q1 OCR markers must resolve to a session metadata ID, never exam_Q1."""
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-CANONICAL::question-1",
            "exam_id": "EXAM-CANONICAL",
            "question_number": 1,
            "question_text": "What is the answer?",
            "question_type": "subjective",
            "max_marks": 1,
        }
    )

    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-CANONICAL",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [{"points": [{"x": 10, "y": 10, "t": 1}]}],
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_FakeOCRAdapter(),
        ),
    ):
        result = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    assert result.blocked_count == 0
    stored = await db["evalpen_detected_responses"].find_one(
        {"submission_id": ingest_result.submission_id}
    )
    assert stored["question_id"] == "EXAM-CANONICAL::question-1"
    assert stored["question_number"] == 1
    assert stored["question_assignment"]["method"] == "marker"


@pytest.mark.asyncio
async def test_process_submission_creates_zero_answer_slots_for_unanswered_questions():
    """One answered question on a 3-question paper must still be scored out of all 3."""
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-SLOTS::question-{number}",
                "exam_id": "EXAM-SLOTS",
                "question_number": number,
                "question_text": f"Question {number}",
                "question_type": "subjective",
                "max_marks": 4,
            }
            for number in (1, 2, 3)
        ]
    )
    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-SLOTS",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [{"points": [{"x": 10, "y": 10, "t": 1}]}],
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_FakeOCRAdapter(),
        ),
    ):
        result = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    slots = await db["evalpen_detected_responses"].find(
        {"submission_id": ingest_result.submission_id}
    ).sort("question_number", 1).to_list(length=10)

    assert result.response_count == 3
    assert [slot["question_number"] for slot in slots] == [1, 2, 3]
    assert slots[0]["is_missing_response"] is False
    assert slots[0]["eval_status"] == "ready"
    assert [slot["is_missing_response"] for slot in slots[1:]] == [True, True]
    assert [slot["answer_state"] for slot in slots[1:]] == ["not_attempted", "not_attempted"]
    assert [slot["eval_status"] for slot in slots[1:]] == ["ready", "ready"]


@pytest.mark.asyncio
async def test_process_submission_content_matches_unmarked_handwritten_answer():
    """A strong unmarked answer is associated before automatic PCR scoring."""
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": "EXAM-AUTO-MAP::projectile",
                "exam_id": "EXAM-AUTO-MAP",
                "question_number": 1,
                "question_text": (
                    "A projectile is launched with speed 20 m/s at 37 degrees. "
                    "Find time of flight, horizontal range, and maximum height."
                ),
                "question_type": "subjective",
                "max_marks": 4,
            },
            {
                "question_id": "EXAM-AUTO-MAP::wedge",
                "exam_id": "EXAM-AUTO-MAP",
                "question_number": 2,
                "question_text": "A block moves on a smooth wedge. Find the normal reaction.",
                "question_type": "subjective",
                "max_marks": 4,
            },
        ]
    )

    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-AUTO-MAP",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="camera",
        pages=[
            {
                "page_number": 1,
                "image_url": "data:image/png;base64,ZmFrZQ==",
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_UnmarkedProjectileOCRAdapter(),
        ),
    ):
        result = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    assert result.blocked_count == 0
    stored = await db["evalpen_detected_responses"].find_one(
        {"submission_id": ingest_result.submission_id}
    )
    assert stored["question_id"] == "EXAM-AUTO-MAP::projectile"
    assert stored["question_number"] == 1
    assert stored["question_assignment"]["method"] == "content_match"
    assert stored["question_assignment"]["score"] >= 8


@pytest.mark.asyncio
async def test_process_submission_blocks_unmapped_response_for_teacher_review():
    """Missing metadata must be reviewable, not auto-scored generically."""
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")
    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-NO-METADATA",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [{"points": [{"x": 10, "y": 10, "t": 1}]}],
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_FakeOCRAdapter(),
        ),
    ):
        result = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    assert result.blocked_count == 1
    stored = await db["evalpen_detected_responses"].find_one(
        {"submission_id": ingest_result.submission_id}
    )
    assert stored["question_id"] is None
    assert stored["eval_status"] == "blocked"


@pytest.mark.asyncio
async def test_reprocessing_submission_supersedes_previous_detected_responses():
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import (
        get_submission_responses,
        process_submission,
    )

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")

    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-REPROCESS",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [
                    {
                        "points": [
                            {"x": 10, "y": 10, "t": 1},
                            {"x": 20, "y": 20, "t": 2},
                        ]
                    }
                ],
            }
        ],
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-STALE-BLOCKED",
            "submission_id": ingest_result.submission_id,
            "question_id": "EXAM-REPROCESS_Q1",
            "exam_id": "EXAM-REPROCESS",
            "student_id": "STU-1",
            "detected_text": "Q.No 1.Ans old Q.No 2.Ans stale",
            "source_pages": [{"page_number": 1}],
            "content_type": "TEXT_ONLY",
            "eval_status": "blocked",
            "flags": [
                {
                    "flag_id": "FLG-OLD",
                    "source": "clubbed_detector",
                    "flag_type": "clubbed_multiple_markers",
                    "severity": "blocking",
                    "reason": "old segmentation",
                }
            ],
            "_immutable": True,
        }
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_FakeOCRAdapter(),
        ),
    ):
        response = await process_submission(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )
        visible = await get_submission_responses(
            ingest_result.submission_id,
            current_user=_admin_user(),
            db=object(),
        )

    assert response.response_count == 1
    stale = await db["evalpen_detected_responses"].find_one(
        {"response_id": "RESP-STALE-BLOCKED"}
    )
    assert stale is not None
    assert stale["detected_text"] == "Q.No 1.Ans old Q.No 2.Ans stale"
    assert stale["eval_status"] == "superseded"
    assert stale["audit_trail"][0]["action"] == "detected_response_superseded"

    visible_ids = {item["response_id"] for item in visible["items"]}
    assert "RESP-STALE-BLOCKED" not in visible_ids
    assert len(visible_ids) == 1


@pytest.mark.asyncio
async def test_process_submission_route_fails_when_pen_ocr_produces_no_text_blocks():
    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_submissions_async import process_submission

    db = _fresh_db()
    ingest_mod = load_exampen("ingest.service")
    submission_service_mod = load_exampen("pcr.services.submission_service")

    ingest = ingest_mod.IngestService(db)
    await ingest.initialize()
    ingest_result = await ingest.ingest_submission(
        exam_id="EXAM-EMPTY-OCR",
        student_id="STU-1",
        admin_id="ADMIN-1",
        source="ble_pen",
        pen_mac="AA:BB:CC:DD:EE:FF",
        hub_id="HUB-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": [
                    {
                        "points": [
                            {"x": 10, "y": 10, "t": 1},
                            {"x": 20, "y": 20, "t": 2},
                        ]
                    }
                ],
            }
        ],
    )

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=db),
        ),
        patch.object(
            submission_service_mod,
            "create_ocr_adapter",
            return_value=_EmptyTextOCRAdapter(),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await process_submission(
                ingest_result.submission_id,
                current_user=_admin_user(),
                db=object(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "OCR produced no text blocks"

    stored_submission = await db["evalpen_submissions"].find_one(
        {"submission_id": ingest_result.submission_id}
    )
    assert stored_submission is not None
    assert stored_submission["segmentation_status"] == "failed"

    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": ingest_result.submission_id}
    ).to_list(length=10)
    assert responses == []
