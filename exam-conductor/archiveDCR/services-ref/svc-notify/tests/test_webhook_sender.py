"""Unit tests for adapters/webhook_sender.py — Stoody webhook delivery.

Tests use mocked HTTP via aioresponses. No real network calls.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from aioresponses import aioresponses
from yarl import URL

from src.adapters.webhook_sender import StoodyWebhookSender, _compute_hmac

BASE_URL = "http://stoody-test:9100"
SECRET = "test-webhook-secret"
SCORES_URL = f"{BASE_URL}/api/webhooks/exampen/scores"
EXAMS_URL = f"{BASE_URL}/api/webhooks/exampen/exams"


def _make_sender(
    *,
    max_retries: int = 3,
    base_delay: float = 0.01,
) -> StoodyWebhookSender:
    """Create a sender with fast retries for testing."""
    return StoodyWebhookSender(
        base_url=BASE_URL,
        secret=SECRET,
        max_retries=max_retries,
        base_delay=base_delay,
    )


def _get_calls(mock: aioresponses, url: str) -> list:
    """Extract call records for a given URL from aioresponses."""
    key = ("POST", URL(url))
    return mock.requests.get(key, [])


# ---------------------------------------------------------------------------
# HMAC signature
# ---------------------------------------------------------------------------


class TestHmacSignature:
    """Verify HMAC-SHA256 computation."""

    def test_compute_hmac_deterministic(self) -> None:
        body = b'{"exam_id":"e-1","scores":[]}'
        sig = _compute_hmac(body, SECRET)
        expected = hmac.new(
            SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        assert sig == expected

    def test_compute_hmac_different_secrets(self) -> None:
        body = b'{"key":"value"}'
        sig_a = _compute_hmac(body, "secret-a")
        sig_b = _compute_hmac(body, "secret-b")
        assert sig_a != sig_b

    def test_empty_body(self) -> None:
        sig = _compute_hmac(b"", SECRET)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Score webhook
# ---------------------------------------------------------------------------


class TestSendScoreWebhook:
    """Tests for send_score_webhook."""

    @pytest.mark.asyncio
    async def test_success_on_200(self) -> None:
        sender = _make_sender()
        with aioresponses() as m:
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            ok = await sender.send_score_webhook(
                exam_id="e-100",
                scores=[
                    {
                        "student_id": "s-1",
                        "total": 85,
                        "percentage": 85.0,
                        "percentile": 92,
                    }
                ],
            )
        assert ok is True

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self) -> None:
        sender = _make_sender(max_retries=1)
        with aioresponses() as m:
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            await sender.send_score_webhook(
                exam_id="e-200",
                scores=[{"student_id": "s-2", "total": 70, "percentage": 70.0, "percentile": 55}],
            )

            calls = _get_calls(m, SCORES_URL)
            assert len(calls) == 1
            req_body = calls[0].kwargs["data"]
            parsed = json.loads(req_body)
            assert parsed["exam_id"] == "e-200"
            assert len(parsed["scores"]) == 1
            assert parsed["scores"][0]["student_id"] == "s-2"

    @pytest.mark.asyncio
    async def test_sends_hmac_header(self) -> None:
        sender = _make_sender(max_retries=1)
        with aioresponses() as m:
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            await sender.send_score_webhook(exam_id="e-1", scores=[])

            calls = _get_calls(m, SCORES_URL)
            headers = calls[0].kwargs["headers"]
            assert "X-ExamPen-Signature" in headers
            body = calls[0].kwargs["data"]
            expected_sig = _compute_hmac(body, SECRET)
            assert headers["X-ExamPen-Signature"] == expected_sig

    @pytest.mark.asyncio
    async def test_returns_false_on_server_error(self) -> None:
        sender = _make_sender(max_retries=1, base_delay=0.001)
        with aioresponses() as m:
            m.post(SCORES_URL, status=500)
            ok = await sender.send_score_webhook(exam_id="e-1", scores=[])
        assert ok is False

    @pytest.mark.asyncio
    async def test_no_signature_when_secret_empty(self) -> None:
        sender = StoodyWebhookSender(
            base_url=BASE_URL,
            secret="",
            max_retries=1,
            base_delay=0.001,
        )
        with aioresponses() as m:
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            await sender.send_score_webhook(exam_id="e-1", scores=[])

            calls = _get_calls(m, SCORES_URL)
            headers = calls[0].kwargs["headers"]
            assert "X-ExamPen-Signature" not in headers


# ---------------------------------------------------------------------------
# Exam webhook
# ---------------------------------------------------------------------------


class TestSendExamWebhook:
    """Tests for send_exam_webhook."""

    @pytest.mark.asyncio
    async def test_created_payload(self) -> None:
        sender = _make_sender(max_retries=1)
        with aioresponses() as m:
            m.post(EXAMS_URL, status=200, payload={"status": "accepted"})
            ok = await sender.send_exam_webhook(
                exam_id="e-300",
                status="created",
                data={
                    "subject_id": "subj-math",
                    "class_id": "cls-10a",
                    "date": "2026-04-01",
                    "duration": 120,
                },
            )
        assert ok is True

    @pytest.mark.asyncio
    async def test_created_payload_contents(self) -> None:
        sender = _make_sender(max_retries=1)
        with aioresponses() as m:
            m.post(EXAMS_URL, status=200, payload={"status": "accepted"})
            await sender.send_exam_webhook(
                exam_id="e-300",
                status="created",
                data={
                    "subject_id": "subj-math",
                    "class_id": "cls-10a",
                    "date": "2026-04-01",
                    "duration": 120,
                },
            )

            calls = _get_calls(m, EXAMS_URL)
            parsed = json.loads(calls[0].kwargs["data"])
            assert parsed["exam_id"] == "e-300"
            assert parsed["subject_id"] == "subj-math"
            assert parsed["class_id"] == "cls-10a"
            assert parsed["date"] == "2026-04-01"
            assert parsed["duration"] == 120
            assert "status" not in parsed

    @pytest.mark.asyncio
    async def test_completed_payload_contents(self) -> None:
        sender = _make_sender(max_retries=1)
        with aioresponses() as m:
            m.post(EXAMS_URL, status=200, payload={"status": "accepted"})
            await sender.send_exam_webhook(
                exam_id="e-400",
                status="completed",
                data={
                    "pens_synced": 38,
                    "upload_status": "complete",
                },
            )

            calls = _get_calls(m, EXAMS_URL)
            parsed = json.loads(calls[0].kwargs["data"])
            assert parsed["exam_id"] == "e-400"
            assert parsed["status"] == "completed"
            assert parsed["pens_synced"] == 38
            assert parsed["upload_status"] == "complete"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    """Verify exponential backoff and retry counts."""

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self) -> None:
        sender = _make_sender(max_retries=3, base_delay=0.001)
        with aioresponses() as m:
            m.post(SCORES_URL, status=500)
            m.post(SCORES_URL, status=502)
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            ok = await sender.send_score_webhook(exam_id="e-1", scores=[])
        assert ok is True

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_false(self) -> None:
        sender = _make_sender(max_retries=3, base_delay=0.001)
        with aioresponses() as m:
            m.post(SCORES_URL, status=500)
            m.post(SCORES_URL, status=500)
            m.post(SCORES_URL, status=500)
            ok = await sender.send_score_webhook(exam_id="e-1", scores=[])
        assert ok is False

    @pytest.mark.asyncio
    async def test_retries_on_network_error(self) -> None:
        sender = _make_sender(max_retries=2, base_delay=0.001)
        with aioresponses() as m:
            m.post(SCORES_URL, exception=ConnectionError("refused"))
            m.post(SCORES_URL, status=200, payload={"status": "accepted"})
            ok = await sender.send_score_webhook(exam_id="e-1", scores=[])
        assert ok is True

    @pytest.mark.asyncio
    async def test_all_network_errors_returns_false(self) -> None:
        sender = _make_sender(max_retries=2, base_delay=0.001)
        with aioresponses() as m:
            m.post(SCORES_URL, exception=ConnectionError("refused"))
            m.post(SCORES_URL, exception=TimeoutError("timed out"))
            ok = await sender.send_score_webhook(exam_id="e-1", scores=[])
        assert ok is False
