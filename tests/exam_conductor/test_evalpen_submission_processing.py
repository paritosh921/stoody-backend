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
