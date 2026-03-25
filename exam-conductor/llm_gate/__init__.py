"""
LLM Gate — Shared gate module for ExamPen.

Public surface:

    from exam_conductor.llm_gate import LLMGate, GateResponse, CallerID

    gate = LLMGate(db)
    await gate.initialize()
    resp = await gate.call("gpt-4o", prompt, "pcr_eval_core")

Spec authority : new-docs/architecture/LLM_GATE_SPEC.md
Hard constraint: C4 — All LLM calls through the gate.
"""

from .gate import LLMGate
from .models import (
    ALLOWED_CALLER_IDS,
    BudgetExhaustedError,
    CallerID,
    CurrentUsage,
    GateConfig,
    GateRequest,
    GateResponse,
    PeriodType,
    TokenLimitExceededError,
    TokenUsage,
    TokenUsageLogEntry,
    TokenUsageRollup,
    UnregisteredCallerError,
    UsageWindow,
)
from .provider import estimate_cost, estimate_tokens, estimate_tokens_for_messages
from .repository import GateRepository
from .rollup import run_all_rollups, run_daily_rollup, run_monthly_rollup, run_weekly_rollup

__all__ = [
    # Core
    "LLMGate",
    "GateRepository",
    # Models — request / response
    "GateRequest",
    "GateResponse",
    "TokenUsage",
    # Models — config
    "GateConfig",
    # Models — storage
    "TokenUsageLogEntry",
    "TokenUsageRollup",
    "PeriodType",
    # Models — API
    "CurrentUsage",
    "UsageWindow",
    # Models — caller
    "CallerID",
    "ALLOWED_CALLER_IDS",
    # Errors
    "BudgetExhaustedError",
    "TokenLimitExceededError",
    "UnregisteredCallerError",
    # Utilities
    "estimate_cost",
    "estimate_tokens",
    "estimate_tokens_for_messages",
    # Rollup functions
    "run_daily_rollup",
    "run_weekly_rollup",
    "run_monthly_rollup",
    "run_all_rollups",
]
