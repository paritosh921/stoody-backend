"""
ExamPen Test Harness — LLM Gate tests.

Test IDs covered:
    U-GATE-01  Caller validation — unregistered caller_id is rejected
    U-GATE-02  Per-call input/output limits enforced
    U-GATE-03  Daily/weekly/monthly budget enforcement
    U-GATE-04  Token log entry shape matches spec
    I-GATE-01  DCR caller goes through gate
    I-GATE-02  PCR caller goes through gate
    I-GATE-03  Usage API returns correct shape

Spec authority: new-docs/architecture/LLM_GATE_SPEC.md
Failure modes:  GATE-01 (budget exhaustion), GATE-02 (caller bypass),
                GATE-03 (log consistency)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — exam-conductor uses a hyphen in its directory name
# ---------------------------------------------------------------------------
_EC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

from llm_gate.models import (
    ALLOWED_CALLER_IDS,
    BudgetExhaustedError,
    CallerID,
    GateConfig,
    GateResponse,
    TokenLimitExceededError,
    TokenUsage,
    TokenUsageLogEntry,
    UnregisteredCallerError,
)
from llm_gate.budget import BudgetChecker
from llm_gate.gate import LLMGate
from llm_gate.provider import (
    ProviderResponse,
    _call_openai_responses,
    estimate_tokens_for_messages,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gate_config_strict():
    """GateConfig with tight limits for testing enforcement."""
    return GateConfig(
        max_input_tokens=4000,
        max_output_tokens=2000,
        daily_token_limit=50_000,
        weekly_token_limit=200_000,
        monthly_token_limit=500_000,
    )


@pytest.fixture
def gate_config_unlimited():
    """GateConfig with no limits (all None)."""
    return GateConfig(
        max_input_tokens=None,
        max_output_tokens=None,
        daily_token_limit=None,
        weekly_token_limit=None,
        monthly_token_limit=None,
    )


def _make_mock_repo(
    config: Optional[GateConfig] = None,
    tokens_since: int = 0,
    rollups: Optional[list] = None,
):
    """Build a mock GateRepository."""
    repo = AsyncMock()
    repo.get_config = AsyncMock(
        return_value=config or GateConfig()
    )
    repo.upsert_config = AsyncMock()
    repo.sum_tokens_since = AsyncMock(return_value=tokens_since)
    repo.list_rollups = AsyncMock(return_value=rollups or [])
    repo.append_log = AsyncMock()
    repo.ensure_indexes = AsyncMock()
    return repo


def _make_provider_response(content="test response", input_tok=100, output_tok=50):
    return ProviderResponse(
        content=content,
        input_tokens=input_tok,
        output_tokens=output_tok,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        model="test-model",
    )


# ===========================================================================
# U-GATE-01: Caller validation — unregistered caller_id is rejected
# ===========================================================================


class TestUGate01:
    """U-GATE-01: Allowed caller validation (LLM_GATE_SPEC section 5)."""

    def test_u_gate_01_allowed_callers_match_spec(self):
        """All spec-defined caller IDs are present in the allow-list."""
        expected = {
            "pcr_eval_core",
            "pcr_objective_extraction",
            "pcr_cache_warmup",
            "pcr_clubbed_h4",
            "pcr_practice",
            "dcr_ai",
            "dcr_devanagari",
        }
        assert ALLOWED_CALLER_IDS == expected

    def test_u_gate_01_caller_enum_values(self):
        """CallerID enum has exactly the registered identities."""
        values = {c.value for c in CallerID}
        assert len(values) == 7
        assert "pcr_eval_core" in values
        assert "pcr_objective_extraction" in values
        assert "dcr_ai" in values

    def test_u_gate_01_unregistered_caller_rejected(self):
        """Gate.call() raises UnregisteredCallerError for unknown caller_id."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo()

            with pytest.raises(UnregisteredCallerError) as exc_info:
                await gate.call(
                    model_id="test-model",
                    prompt="hello",
                    caller_id="rogue_caller",
                )
            assert "rogue_caller" in str(exc_info.value)
        asyncio.run(_run())

    def test_u_gate_01_empty_caller_rejected(self):
        """Empty string is not a valid caller_id."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo()

            with pytest.raises(UnregisteredCallerError):
                await gate.call(
                    model_id="test-model",
                    prompt="hello",
                    caller_id="",
                )
        asyncio.run(_run())


# ===========================================================================
# U-GATE-02: Per-call input/output limits enforced
# ===========================================================================


class TestUGate02:
    """U-GATE-02: Per-call input/output limits (LLM_GATE_SPEC section 6, 9.2)."""

    def test_u_gate_02_input_limit_exceeded(self, gate_config_strict):
        """TokenLimitExceededError raised when estimated input exceeds ceiling."""
        with pytest.raises(TokenLimitExceededError) as exc_info:
            BudgetChecker.check_per_call_input(gate_config_strict, 5000)
        err = exc_info.value
        assert err.limit_type == "per_call_input"
        assert err.estimated_tokens == 5000
        assert err.allowed_tokens == 4000

    def test_u_gate_02_input_within_limit(self, gate_config_strict):
        """No error when estimated input is within ceiling."""
        BudgetChecker.check_per_call_input(gate_config_strict, 3000)

    def test_u_gate_02_output_limit_exceeded(self, gate_config_strict):
        """TokenLimitExceededError raised when requested output exceeds ceiling."""
        with pytest.raises(TokenLimitExceededError) as exc_info:
            BudgetChecker.check_per_call_output(gate_config_strict, 3000)
        err = exc_info.value
        assert err.limit_type == "per_call_output"
        assert err.estimated_tokens == 3000
        assert err.allowed_tokens == 2000

    def test_u_gate_02_output_within_limit(self, gate_config_strict):
        """No error when requested output is within ceiling."""
        BudgetChecker.check_per_call_output(gate_config_strict, 1500)

    def test_u_gate_02_output_none_not_checked(self, gate_config_strict):
        """No error when requested output is None (gate clamps later)."""
        BudgetChecker.check_per_call_output(gate_config_strict, None)

    def test_u_gate_02_unlimited_config_passes(self, gate_config_unlimited):
        """No error when config limits are None (unlimited)."""
        BudgetChecker.check_per_call_input(gate_config_unlimited, 999_999)
        BudgetChecker.check_per_call_output(gate_config_unlimited, 999_999)

    def test_u_gate_02_error_to_dict_shape(self, gate_config_strict):
        """TokenLimitExceededError.to_dict() has the correct spec shape."""
        try:
            BudgetChecker.check_per_call_input(gate_config_strict, 6200)
        except TokenLimitExceededError as e:
            d = e.to_dict()
            assert d["error"] == "token_limit_exceeded"
            assert d["limit_type"] == "per_call_input"
            assert d["estimated_tokens"] == 6200
            assert d["allowed_tokens"] == 4000


# ===========================================================================
# U-GATE-03: Daily/weekly/monthly budget enforcement
# ===========================================================================


class TestUGate03:
    """U-GATE-03: Budget period enforcement (LLM_GATE_SPEC section 6, 9.1)."""

    def test_u_gate_03_daily_budget_exhausted(self, gate_config_strict):
        """BudgetExhaustedError raised when daily usage meets limit."""
        async def _run():
            repo = _make_mock_repo(tokens_since=50_000)
            checker = BudgetChecker(repo)

            with pytest.raises(BudgetExhaustedError) as exc_info:
                await checker.check_budgets(gate_config_strict)
            err = exc_info.value
            assert err.period == "daily"
            assert err.used_tokens == 50_000
            assert err.limit_tokens == 50_000
        asyncio.run(_run())

    def test_u_gate_03_daily_under_limit_passes(self, gate_config_strict):
        """No error when daily usage is below limit."""
        async def _run():
            repo = _make_mock_repo(tokens_since=30_000)
            checker = BudgetChecker(repo)
            await checker.check_budgets(gate_config_strict)
        asyncio.run(_run())

    def test_u_gate_03_weekly_budget_exhausted(self):
        """BudgetExhaustedError raised when weekly usage meets limit."""
        async def _run():
            config = GateConfig(
                daily_token_limit=None,
                weekly_token_limit=100_000,
                monthly_token_limit=None,
            )
            repo = _make_mock_repo(tokens_since=100_000)
            checker = BudgetChecker(repo)

            with pytest.raises(BudgetExhaustedError) as exc_info:
                await checker.check_budgets(config)
            assert exc_info.value.period == "weekly"
        asyncio.run(_run())

    def test_u_gate_03_monthly_budget_exhausted(self):
        """BudgetExhaustedError raised when monthly usage meets limit."""
        async def _run():
            config = GateConfig(
                daily_token_limit=None,
                weekly_token_limit=None,
                monthly_token_limit=200_000,
            )
            repo = _make_mock_repo(tokens_since=200_000)
            checker = BudgetChecker(repo)

            with pytest.raises(BudgetExhaustedError) as exc_info:
                await checker.check_budgets(config)
            assert exc_info.value.period == "monthly"
        asyncio.run(_run())

    def test_u_gate_03_unlimited_config_never_raises(
        self, gate_config_unlimited
    ):
        """No error when all budget limits are None."""
        async def _run():
            repo = _make_mock_repo(tokens_since=999_999_999)
            checker = BudgetChecker(repo)
            await checker.check_budgets(gate_config_unlimited)
        asyncio.run(_run())

    def test_u_gate_03_budget_error_to_dict_shape(self, gate_config_strict):
        """BudgetExhaustedError.to_dict() has the correct spec shape."""
        async def _run():
            repo = _make_mock_repo(tokens_since=50_000)
            checker = BudgetChecker(repo)

            try:
                await checker.check_budgets(gate_config_strict)
            except BudgetExhaustedError as e:
                d = e.to_dict()
                assert d["error"] == "budget_exhausted"
                assert d["period"] == "daily"
                assert "used_tokens" in d
                assert "limit_tokens" in d
                assert "resets_at" in d
        asyncio.run(_run())


# ===========================================================================
# U-GATE-04: Token log entry shape matches spec
# ===========================================================================


class TestUGate04:
    """U-GATE-04: Append-only token log shape (LLM_GATE_SPEC section 7.2)."""

    def test_u_gate_04_log_entry_shape(self):
        """TokenUsageLogEntry has all fields from LLM_GATE_SPEC 7.2."""
        entry = TokenUsageLogEntry(
            model="gpt-4o",
            caller="pcr_eval_core",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens=0,
            total_tokens=1500,
            estimated_cost_usd=0.015,
            metadata={"response_id": "RESP-abc123"},
        )
        assert entry.model == "gpt-4o"
        assert entry.caller == "pcr_eval_core"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.cache_read_tokens == 200
        assert entry.cache_creation_tokens == 0
        assert entry.total_tokens == 1500
        assert entry.estimated_cost_usd == 0.015
        assert isinstance(entry.called_at, datetime)
        assert entry.metadata == {"response_id": "RESP-abc123"}

    def test_u_gate_04_token_usage_shape(self):
        """TokenUsage model matches GateResponse.usage shape (section 4.2)."""
        usage = TokenUsage(
            model="claude-sonnet-4-20250514",
            caller="pcr_eval_core",
            input_tokens=800,
            output_tokens=400,
            cache_read_tokens=100,
            cache_creation_tokens=50,
            total_tokens=1200,
            estimated_cost_usd=0.012,
        )
        assert usage.model == "claude-sonnet-4-20250514"
        assert usage.total_tokens == 1200
        assert isinstance(usage.timestamp, datetime)

    def test_u_gate_04_gate_response_shape(self):
        """GateResponse contains content and usage with correct fields."""
        usage = TokenUsage(
            model="test-model",
            caller="dcr_ai",
            total_tokens=100,
        )
        resp = GateResponse(content="test output", usage=usage)
        assert resp.content == "test output"
        assert resp.usage.caller == "dcr_ai"
        assert resp.usage.total_tokens == 100


# ===========================================================================
# I-GATE-01: DCR caller goes through gate
# ===========================================================================


class TestIGate01:
    """I-GATE-01: DCR fallback call flows through gate."""

    def test_i_gate_01_dcr_caller_accepted(self):
        """dcr_ai caller_id is accepted by the gate and produces a response."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo(config=GateConfig())

            provider_resp = _make_provider_response("DCR fallback result")

            with patch("llm_gate.gate.call_provider", new_callable=AsyncMock, return_value=provider_resp), \
                 patch("llm_gate.gate.estimate_tokens", return_value=100), \
                 patch("llm_gate.gate.estimate_tokens_for_messages", return_value=100), \
                 patch("llm_gate.gate.estimate_cost", return_value=0.001):
                response = await gate.call(
                    model_id="gpt-4o",
                    prompt="Recognize this handwriting",
                    caller_id="dcr_ai",
                )

            assert isinstance(response, GateResponse)
            assert response.content == "DCR fallback result"
            assert response.usage.caller == "dcr_ai"
        asyncio.run(_run())

    def test_i_gate_01_dcr_devanagari_accepted(self):
        """dcr_devanagari caller_id is accepted by the gate."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo(config=GateConfig())

            provider_resp = _make_provider_response("Devanagari result")

            with patch("llm_gate.gate.call_provider", new_callable=AsyncMock, return_value=provider_resp), \
                 patch("llm_gate.gate.estimate_tokens_for_messages", return_value=100), \
                 patch("llm_gate.gate.estimate_cost", return_value=0.001):
                response = await gate.call(
                    model_id="gpt-4o",
                    prompt="Recognize Devanagari",
                    caller_id="dcr_devanagari",
                )

            assert response.usage.caller == "dcr_devanagari"
        asyncio.run(_run())


# ===========================================================================
# I-GATE-02: PCR caller goes through gate
# ===========================================================================


class TestIGate02:
    """I-GATE-02: PCR evaluation call flows through gate."""

    def test_i_gate_02_pcr_eval_core_accepted(self):
        """pcr_eval_core caller_id is accepted by the gate."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo(config=GateConfig())

            provider_resp = _make_provider_response("PCR eval result")

            with patch("llm_gate.gate.call_provider", new_callable=AsyncMock, return_value=provider_resp), \
                 patch("llm_gate.gate.estimate_tokens_for_messages", return_value=100), \
                 patch("llm_gate.gate.estimate_cost", return_value=0.001):
                response = await gate.call(
                    model_id="claude-sonnet-4-20250514",
                    prompt="Evaluate student response",
                    caller_id="pcr_eval_core",
                )

            assert response.usage.caller == "pcr_eval_core"
        asyncio.run(_run())

    def test_i_gate_02_pcr_practice_accepted(self):
        """pcr_practice caller_id is accepted by the gate."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            gate._repo = _make_mock_repo(config=GateConfig())

            provider_resp = _make_provider_response("Practice result")

            with patch("llm_gate.gate.call_provider", new_callable=AsyncMock, return_value=provider_resp), \
                 patch("llm_gate.gate.estimate_tokens_for_messages", return_value=100), \
                 patch("llm_gate.gate.estimate_cost", return_value=0.001):
                response = await gate.call(
                    model_id="claude-haiku-4-20250514",
                    prompt="Practice eval",
                    caller_id="pcr_practice",
                )

            assert response.usage.caller == "pcr_practice"
        asyncio.run(_run())


class TestOpenAIResponsesDocumentInput:
    """Native file inputs remain inside the shared, usage-metered gate."""

    def test_document_input_estimator_counts_files_and_images_without_base64_text(self):
        estimated = estimate_tokens_for_messages(
            "",
            responses_input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "grade this paper"},
                        {
                            "type": "input_file",
                            "filename": "paper.pdf",
                            "file_data": "data:application/pdf;base64," + "A" * 100_000,
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64," + "B" * 100_000,
                        },
                    ],
                }
            ],
        )
        assert 11_000 <= estimated < 12_000

    def test_openai_responses_payload_uses_private_structured_file_call(self):
        class _HTTPResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "model": "gpt-5.1-2025-11-13",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": '{"questions":[]}'},
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 2000,
                        "output_tokens": 50,
                        "input_tokens_details": {"cached_tokens": 1500},
                    },
                }

        class _HTTPClient:
            def __init__(self):
                self.payload = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, _url, *, headers, json):
                assert headers["Authorization"] == "Bearer secret"
                self.payload = json
                return _HTTPResponse()

        async def _run():
            client = _HTTPClient()
            with patch("llm_gate.provider.httpx.AsyncClient", return_value=client):
                result = await _call_openai_responses(
                    "gpt-5.1-2025-11-13",
                    responses_input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "filename": "paper.pdf",
                                    "file_data": "data:application/pdf;base64,AAAA",
                                }
                            ],
                        }
                    ],
                    json_schema={"type": "object", "properties": {}},
                    prompt_cache_key="pcr-paper-static",
                    reasoning_effort="medium",
                    temperature=0.10,
                    max_output_tokens=8000,
                    api_key="secret",
                )
            assert client.payload["store"] is False
            assert client.payload["prompt_cache_key"] == "pcr-paper-static"
            assert client.payload["text"]["format"]["strict"] is True
            assert client.payload["input"][0]["content"][0]["type"] == "input_file"
            assert "temperature" not in client.payload
            assert result.content == '{"questions":[]}'
            assert result.cache_read_tokens == 1500
            assert result.completion_status == "completed"
            assert result.incomplete_reason == ""

        asyncio.run(_run())

    def test_openai_responses_surfaces_incomplete_generation_status(self):
        class _HTTPResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "model": "gpt-5.1-2025-11-13",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": '{"questions": ['},
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                }

        class _HTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, _url, *, headers, json):
                return _HTTPResponse()

        async def _run():
            with patch("llm_gate.provider.httpx.AsyncClient", return_value=_HTTPClient()):
                result = await _call_openai_responses(
                    "gpt-5.1-2025-11-13",
                    responses_input=[{"role": "user", "content": [{"type": "input_text", "text": "x"}]}],
                    api_key="secret",
                )
            assert result.completion_status == "incomplete"
            assert result.incomplete_reason == "max_output_tokens"

        asyncio.run(_run())

    def test_openai_responses_retains_temperature_for_supported_model(self):
        class _HTTPResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "model": "gpt-4.1",
                    "output": [],
                    "usage": {},
                }

        class _HTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, _url, *, headers, json):
                assert json["temperature"] == 0.10
                return _HTTPResponse()

        async def _run():
            with patch("llm_gate.provider.httpx.AsyncClient", return_value=_HTTPClient()):
                await _call_openai_responses(
                    "gpt-4.1",
                    responses_input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "grade"}],
                        }
                    ],
                    temperature=0.10,
                    api_key="secret",
                )

        asyncio.run(_run())


class TestIGate02UsageLogging:
    """I-GATE-02: successful PCR calls append one usage record."""

    def test_i_gate_02_log_appended_on_success(self):
        """Gate appends a token log entry after a successful call."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            repo = _make_mock_repo(config=GateConfig())
            gate._repo = repo

            provider_resp = _make_provider_response("result", input_tok=200, output_tok=100)

            with patch("llm_gate.gate.call_provider", new_callable=AsyncMock, return_value=provider_resp), \
                 patch("llm_gate.gate.estimate_tokens_for_messages", return_value=200), \
                 patch("llm_gate.gate.estimate_cost", return_value=0.002):
                await gate.call(
                    model_id="gpt-4o",
                    prompt="test",
                    caller_id="pcr_eval_core",
                )

            repo.append_log.assert_called_once()
            log_entry = repo.append_log.call_args[0][0]
            assert isinstance(log_entry, TokenUsageLogEntry)
            assert log_entry.caller == "pcr_eval_core"
            assert log_entry.input_tokens == 200
            assert log_entry.output_tokens == 100
        asyncio.run(_run())


# ===========================================================================
# I-GATE-03: Usage API returns correct shape
# ===========================================================================


class TestIGate03:
    """I-GATE-03: Usage API reflects stored config and usage."""

    def test_i_gate_03_current_usage_shape(self, gate_config_strict):
        """current_usage() returns dict with daily/weekly/monthly windows."""
        async def _run():
            repo = _make_mock_repo(config=gate_config_strict, tokens_since=10_000)
            checker = BudgetChecker(repo)
            usage = await checker.current_usage(gate_config_strict)

            assert "daily" in usage
            assert "weekly" in usage
            assert "monthly" in usage

            for window in [usage["daily"], usage["weekly"], usage["monthly"]]:
                assert "used_tokens" in window
                assert "limit_tokens" in window
                assert "remaining_tokens" in window
        asyncio.run(_run())

    def test_i_gate_03_remaining_tokens_calculated(self, gate_config_strict):
        """remaining_tokens = limit - used (clamped to 0)."""
        async def _run():
            repo = _make_mock_repo(config=gate_config_strict, tokens_since=10_000)
            checker = BudgetChecker(repo)
            usage = await checker.current_usage(gate_config_strict)

            daily = usage["daily"]
            assert daily["used_tokens"] == 10_000
            assert daily["limit_tokens"] == 50_000
            assert daily["remaining_tokens"] == 40_000
        asyncio.run(_run())

    def test_i_gate_03_unlimited_shows_none(self, gate_config_unlimited):
        """When limits are None, remaining_tokens is None."""
        async def _run():
            repo = _make_mock_repo(config=gate_config_unlimited, tokens_since=5000)
            checker = BudgetChecker(repo)
            usage = await checker.current_usage(gate_config_unlimited)

            assert usage["daily"]["limit_tokens"] is None
            assert usage["daily"]["remaining_tokens"] is None
        asyncio.run(_run())

    def test_i_gate_03_gate_current_usage_delegates(self):
        """LLMGate.current_usage() delegates to BudgetChecker."""
        async def _run():
            db = MagicMock()
            gate = LLMGate(db)
            mock_repo = _make_mock_repo(config=GateConfig(), tokens_since=100)
            gate._repo = mock_repo
            gate._budget = BudgetChecker(mock_repo)
            usage = await gate.current_usage()

            assert "daily" in usage
            assert "weekly" in usage
            assert "monthly" in usage
        asyncio.run(_run())
