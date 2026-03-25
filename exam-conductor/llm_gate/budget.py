"""
LLM Gate — Budget enforcement logic.

Handles:
- Per-call input / output token limit checks (§6, §9.2)
- Daily / weekly / monthly budget enforcement against rollup-aware usage (§6, §9.1)

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §4 step 3–4, §6, §9
Test IDs        : U-GATE-02 (per-call limits), U-GATE-03 (budget checks)
Failure modes   : GATE-01 (budget exhaustion → explicit refusal)
Hard constraint : C5 — Gate owns token budget state only.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from .models import (
    BudgetExhaustedError,
    GateConfig,
    PeriodType,
    TokenLimitExceededError,
)
from .repository import GateRepository

# Raw log rows have a 7-day TTL (repository.py §7.2 / §8).
# Any period longer than this must combine rollups with raw logs.
_RAW_LOG_RETENTION_DAYS = 7

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _day_start(now: datetime) -> datetime:
    """Return midnight UTC of the current day."""
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _week_start(now: datetime) -> datetime:
    """Return midnight UTC of the current ISO week's Monday."""
    day_start = _day_start(now)
    return day_start - timedelta(days=day_start.weekday())  # Monday=0


def _month_start(now: datetime) -> datetime:
    """Return midnight UTC of the first day of the current month."""
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_day(now: datetime) -> datetime:
    return _day_start(now) + timedelta(days=1)


def _next_week(now: datetime) -> datetime:
    return _week_start(now) + timedelta(weeks=1)


def _next_month(now: datetime) -> datetime:
    ms = _month_start(now)
    # Advance to first of next month
    if ms.month == 12:
        return ms.replace(year=ms.year + 1, month=1)
    return ms.replace(month=ms.month + 1)


# ---------------------------------------------------------------------------
# Budget checker
# ---------------------------------------------------------------------------

class BudgetChecker:
    """
    Stateless helper that reads config + current usage from the repository
    and raises on violations.

    The budget checker does NOT mutate any state — it only reads.  Logging
    of actual usage happens after the provider call succeeds (gate.py).
    """

    def __init__(self, repo: GateRepository) -> None:
        self._repo = repo

    # ------------------------------------------------------------------
    # Hybrid rollup + raw-log token sum (handles 7-day TTL gap)
    # ------------------------------------------------------------------

    async def _sum_tokens_for_period(
        self,
        period_start: datetime,
        now: datetime,
    ) -> int:
        """
        Sum tokens for a period that may extend beyond the 7-day raw-log
        retention window.

        Strategy:
        - If the entire period falls within the raw-log window, just query
          raw logs (fast, no rollup dependency).
        - Otherwise, sum *daily rollups* for the portion older than the
          raw-log cutoff and add the raw-log total for the recent portion.

        This ensures weekly/monthly budgets remain accurate even after
        MongoDB's TTL index deletes old log rows.
        """
        raw_log_cutoff = now - timedelta(days=_RAW_LOG_RETENTION_DAYS)

        if period_start >= raw_log_cutoff:
            # Entire period is within the raw-log window — simple path.
            return await self._repo.sum_tokens_since(period_start)

        # Fetch daily rollups that cover the pre-cutoff portion.
        rollups = await self._repo.list_rollups(
            period_type=PeriodType.DAILY,
            since=period_start,
            limit=31,  # At most ~31 days for a monthly window
        )
        rollup_total = sum(
            r.total_tokens
            for r in rollups
            if r.period_start < raw_log_cutoff
        )

        # Add raw-log total for the recent portion still in MongoDB.
        raw_total = await self._repo.sum_tokens_since(raw_log_cutoff)

        return rollup_total + raw_total

    # ------------------------------------------------------------------
    # Per-call limit enforcement (§4 step 3, §9.2)
    # ------------------------------------------------------------------

    @staticmethod
    def check_per_call_input(
        config: GateConfig,
        estimated_input_tokens: int,
    ) -> None:
        """
        Raise ``TokenLimitExceededError`` if the estimated input tokens
        exceed the configured per-call input ceiling.
        """
        if config.max_input_tokens is not None and estimated_input_tokens > config.max_input_tokens:
            raise TokenLimitExceededError(
                limit_type="per_call_input",
                estimated_tokens=estimated_input_tokens,
                allowed_tokens=config.max_input_tokens,
            )

    @staticmethod
    def check_per_call_output(
        config: GateConfig,
        requested_output_tokens: Optional[int],
    ) -> None:
        """
        Raise ``TokenLimitExceededError`` if the requested max-output
        exceeds the configured per-call output ceiling.

        If the caller did not request a specific output limit, the gate
        will clamp it to the config ceiling later — no error raised here.
        """
        if (
            config.max_output_tokens is not None
            and requested_output_tokens is not None
            and requested_output_tokens > config.max_output_tokens
        ):
            raise TokenLimitExceededError(
                limit_type="per_call_output",
                estimated_tokens=requested_output_tokens,
                allowed_tokens=config.max_output_tokens,
            )

    # ------------------------------------------------------------------
    # Period budget enforcement (§4 step 4, §9.1)
    # ------------------------------------------------------------------

    async def check_budgets(self, config: GateConfig, now: Optional[datetime] = None) -> None:
        """
        Check daily, weekly, and monthly token budgets against current usage.

        Raises ``BudgetExhaustedError`` with the tightest violated period.
        """
        now = now or datetime.utcnow()

        # Daily
        if config.daily_token_limit is not None:
            used = await self._repo.sum_tokens_since(_day_start(now))
            if used >= config.daily_token_limit:
                raise BudgetExhaustedError(
                    period="daily",
                    used_tokens=used,
                    limit_tokens=config.daily_token_limit,
                    resets_at=_next_day(now),
                )

        # Weekly — may span beyond 7-day raw-log retention window
        if config.weekly_token_limit is not None:
            used = await self._sum_tokens_for_period(_week_start(now), now)
            if used >= config.weekly_token_limit:
                raise BudgetExhaustedError(
                    period="weekly",
                    used_tokens=used,
                    limit_tokens=config.weekly_token_limit,
                    resets_at=_next_week(now),
                )

        # Monthly — always spans beyond 7-day raw-log retention window
        if config.monthly_token_limit is not None:
            used = await self._sum_tokens_for_period(_month_start(now), now)
            if used >= config.monthly_token_limit:
                raise BudgetExhaustedError(
                    period="monthly",
                    used_tokens=used,
                    limit_tokens=config.monthly_token_limit,
                    resets_at=_next_month(now),
                )

    # ------------------------------------------------------------------
    # Current usage query (used by usage API — I-GATE-03)
    # ------------------------------------------------------------------

    async def current_usage(
        self,
        config: GateConfig,
        now: Optional[datetime] = None,
    ) -> dict:
        """
        Return a dict matching ``CurrentUsage`` shape from the OpenAPI spec:
        ``{ daily: UsageWindow, weekly: UsageWindow, monthly: UsageWindow }``.
        """
        now = now or datetime.utcnow()

        daily_used = await self._repo.sum_tokens_since(_day_start(now))
        weekly_used = await self._sum_tokens_for_period(_week_start(now), now)
        monthly_used = await self._sum_tokens_for_period(_month_start(now), now)

        def _window(used: int, limit: Optional[int]) -> dict:
            if limit is None:
                return {"used_tokens": used, "limit_tokens": None, "remaining_tokens": None}
            return {
                "used_tokens": used,
                "limit_tokens": limit,
                "remaining_tokens": max(0, limit - used),
            }

        return {
            "daily": _window(daily_used, config.daily_token_limit),
            "weekly": _window(weekly_used, config.weekly_token_limit),
            "monthly": _window(monthly_used, config.monthly_token_limit),
        }
