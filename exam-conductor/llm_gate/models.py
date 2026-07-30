"""
LLM Gate — Pydantic models for gate contract, config, logging, and rollups.

Spec authority: new-docs/architecture/LLM_GATE_SPEC.md
Test IDs covered: U-GATE-01 (caller validation), U-GATE-04 (log shape)
Failure modes referenced: GATE-02 (caller bypass), GATE-03 (log consistency)
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Allowed caller identities (LLM_GATE_SPEC.md §5)
# ---------------------------------------------------------------------------

class CallerID(str, enum.Enum):
    """
    Registered caller identities.  Any value outside this enum MUST be
    rejected by the gate (GATE-02 mitigation).
    """

    PCR_EVAL_CORE = "pcr_eval_core"
    PCR_CACHE_WARMUP = "pcr_cache_warmup"
    PCR_CLUBBED_H4 = "pcr_clubbed_h4"
    PCR_PRACTICE = "pcr_practice"
    DCR_AI = "dcr_ai"
    DCR_DEVANAGARI = "dcr_devanagari"


ALLOWED_CALLER_IDS: frozenset[str] = frozenset(c.value for c in CallerID)


# ---------------------------------------------------------------------------
# Gate call request
# ---------------------------------------------------------------------------

class GateRequest(BaseModel):
    """Input payload for a single gate invocation."""

    model_id: str = Field(..., description="Provider model identifier, e.g. 'gpt-4o', 'claude-sonnet-4-20250514'")
    prompt: str = Field(..., description="The prompt text (system+user combined or user-only).  May be empty for pure vision calls when messages is provided.")
    caller_id: str = Field(..., description="Registered caller identity")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Pre-built messages array for multimodal/vision calls.  When provided, prompt is ignored for the LLM call.")
    responses_input: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "OpenAI Responses API input. Supports native input_file and "
            "input_image parts. Mutually exclusive with messages."
        ),
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Strict structured-output JSON schema for Responses API calls.",
    )
    prompt_cache_key: Optional[str] = Field(
        None,
        description="Stable non-PII cache-routing key for repeated prompt prefixes.",
    )
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high"]] = None
    max_output_tokens: Optional[int] = Field(None, ge=1, description="Override for max output tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override (where allowed)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Opaque metadata forwarded to logs")


# ---------------------------------------------------------------------------
# Usage sub-model (embedded in GateResponse and persisted to log)
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Token accounting for a single gate call — matches LLM_GATE_SPEC.md §4.2."""

    model: str
    caller: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Gate response (§4.2)
# ---------------------------------------------------------------------------

class GateResponse(BaseModel):
    """
    Return value of ``gate.call()``.

    ``content`` contains the LLM's text output; ``usage`` contains full
    token accounting.  Provider completion metadata is normalized separately
    from the response body so callers can distinguish malformed output from a
    provider response that was deliberately stopped at its output ceiling.
    """

    content: str
    usage: TokenUsage
    provider_status: Optional[str] = None
    incomplete_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Error payloads (§9)
# ---------------------------------------------------------------------------

class BudgetExhaustedError(Exception):
    """Raised when a budget period is exhausted (GATE-01 mitigation)."""

    def __init__(
        self,
        period: str,
        used_tokens: int,
        limit_tokens: int,
        resets_at: datetime,
    ):
        self.period = period
        self.used_tokens = used_tokens
        self.limit_tokens = limit_tokens
        self.resets_at = resets_at
        super().__init__(
            f"budget_exhausted: {period} — used {used_tokens}/{limit_tokens}, "
            f"resets at {resets_at.isoformat()}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "budget_exhausted",
            "period": self.period,
            "used_tokens": self.used_tokens,
            "limit_tokens": self.limit_tokens,
            "resets_at": self.resets_at.isoformat() + "Z",
        }


class TokenLimitExceededError(Exception):
    """Raised when a per-call token limit is exceeded (§9.2)."""

    def __init__(
        self,
        limit_type: Literal["per_call_input", "per_call_output"],
        estimated_tokens: int,
        allowed_tokens: int,
    ):
        self.limit_type = limit_type
        self.estimated_tokens = estimated_tokens
        self.allowed_tokens = allowed_tokens
        super().__init__(
            f"token_limit_exceeded: {limit_type} — "
            f"estimated {estimated_tokens}, allowed {allowed_tokens}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "token_limit_exceeded",
            "limit_type": self.limit_type,
            "estimated_tokens": self.estimated_tokens,
            "allowed_tokens": self.allowed_tokens,
        }


class UnregisteredCallerError(Exception):
    """Raised when a caller_id is not in the allow-list (GATE-02)."""

    def __init__(self, caller_id: str):
        self.caller_id = caller_id
        super().__init__(f"unregistered_caller: {caller_id}")


# ---------------------------------------------------------------------------
# Configuration models (§6 + §7.1)
# ---------------------------------------------------------------------------

class GateConfig(BaseModel):
    """
    Persisted as a single document per tenant in ``llm_gate_config``
    with ``_id = "gate_config"`` (§7.1).
    """

    max_input_tokens: Optional[int] = Field(None, ge=1, description="Per-call input token ceiling; null = unlimited")
    max_output_tokens: Optional[int] = Field(None, ge=1, description="Per-call output token ceiling; null = unlimited")
    daily_token_limit: Optional[int] = Field(None, ge=1, description="Daily budget; null = unlimited")
    weekly_token_limit: Optional[int] = Field(None, ge=1, description="Weekly budget; null = unlimited")
    monthly_token_limit: Optional[int] = Field(None, ge=1, description="Monthly budget; null = unlimited")
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Token usage log document (§7.2)
# ---------------------------------------------------------------------------

class TokenUsageLogEntry(BaseModel):
    """Shape of a single append-only document in ``llm_token_usage_log``."""

    model: str
    caller: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    called_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Token usage rollup document (§7.3)
# ---------------------------------------------------------------------------

class PeriodType(str, enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BreakdownEntry(BaseModel):
    key: str
    total_tokens: int = 0
    total_input: int = 0
    total_output: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0


class TokenUsageRollup(BaseModel):
    """Shape of a single rollup document in ``llm_token_usage_rollup``."""

    period_type: PeriodType
    period_start: datetime
    period_end: datetime
    total_tokens: int = 0
    total_input: int = 0
    total_output: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    breakdown_by_model: List[BreakdownEntry] = Field(default_factory=list)
    breakdown_by_caller: List[BreakdownEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Usage API response models (eval-usage.openapi.yaml)
# ---------------------------------------------------------------------------

class UsageWindow(BaseModel):
    """One budget window as returned by ``GET /usage/current``."""

    used_tokens: int = 0
    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None


class CurrentUsage(BaseModel):
    daily: UsageWindow
    weekly: UsageWindow
    monthly: UsageWindow
