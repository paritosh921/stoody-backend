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
    tenant_db = object()

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
            current_user=_tutor_user(),
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

    with (
        patch(
            "api.v1.evalpen_submissions_async._get_tenant_db_for_user",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(return_value=processor),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await process_submission(
                "SUB-1",
                current_user=_tutor_user(),
                db=object(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No answer pages found for submission"


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
            current_user=_tutor_user(),
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
