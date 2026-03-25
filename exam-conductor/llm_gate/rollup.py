"""
LLM Gate — Token usage rollup aggregation.

Provides async functions to aggregate raw token usage logs into daily,
weekly, and monthly rollup documents.  These functions are framework-
agnostic: they can be called from a Celery task, a cron job, or manually
from a REPL / test harness.

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §7.3 / §8
Ownership       : LLM gate (STATE_OWNERSHIP_MAP.md)
Test IDs        : U-GATE-04 (log shape), I-GATE-03 (usage API)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from .budget import _day_start, _month_start, _week_start
from .models import BreakdownEntry, PeriodType, TokenUsageRollup
from .repository import GateRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prev_day_range(ref: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) for the calendar day *before* ``ref``."""
    end = _day_start(ref)
    start = end - timedelta(days=1)
    return start, end


def _prev_week_range(ref: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) for the ISO week ending before ``ref``.

    ``ref`` is expected to fall on a Monday, so the previous week runs
    from the Monday 7 days earlier to ``ref`` midnight.
    """
    end = _week_start(ref)
    start = end - timedelta(weeks=1)
    return start, end


def _prev_month_range(ref: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) for the calendar month before ``ref``.

    ``ref`` is expected to fall on the 1st of a month, so the previous
    month runs from the 1st of the prior month to ``ref`` midnight.
    """
    end = _month_start(ref)
    # Go back one day to land in the previous month, then get its start.
    start = _month_start(end - timedelta(days=1))
    return start, end


def _build_breakdown(raw: List[Dict]) -> List[BreakdownEntry]:
    """Convert raw aggregation dicts into ``BreakdownEntry`` models."""
    return [
        BreakdownEntry(
            key=r["key"],
            total_tokens=r["total_tokens"],
            total_input=r["total_input"],
            total_output=r["total_output"],
            total_cost_usd=r["total_cost_usd"],
            call_count=r["call_count"],
        )
        for r in raw
    ]


async def _aggregate_and_upsert(
    repo: GateRepository,
    period_type: PeriodType,
    start: datetime,
    end: datetime,
) -> TokenUsageRollup:
    """Run the aggregation pipeline and persist a rollup document."""
    summary = await repo.sum_tokens_in_range(start, end)
    by_model = await repo.breakdown_in_range(start, end, "model")
    by_caller = await repo.breakdown_in_range(start, end, "caller")

    rollup = TokenUsageRollup(
        period_type=period_type,
        period_start=start,
        period_end=end,
        total_tokens=summary["total_tokens"],
        total_input=summary["total_input"],
        total_output=summary["total_output"],
        total_cost_usd=summary["total_cost_usd"],
        call_count=summary["call_count"],
        breakdown_by_model=_build_breakdown(by_model),
        breakdown_by_caller=_build_breakdown(by_caller),
    )

    await repo.upsert_rollup(rollup)
    logger.info(
        "Upserted %s rollup for [%s, %s): %d tokens, %d calls",
        period_type.value,
        start.isoformat(),
        end.isoformat(),
        rollup.total_tokens,
        rollup.call_count,
    )
    return rollup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_daily_rollup(
    db: AsyncIOMotorDatabase,
    target_date: Optional[datetime] = None,
) -> TokenUsageRollup:
    """Aggregate yesterday's raw log into a daily rollup.

    Parameters
    ----------
    db:
        An async Motor database handle for the tenant.
    target_date:
        Reference point; the rollup covers the calendar day *before*
        this timestamp.  Defaults to ``datetime.utcnow()``.
    """
    ref = target_date or datetime.utcnow()
    start, end = _prev_day_range(ref)
    repo = GateRepository(db)
    return await _aggregate_and_upsert(repo, PeriodType.DAILY, start, end)


async def run_weekly_rollup(
    db: AsyncIOMotorDatabase,
    target_date: Optional[datetime] = None,
) -> TokenUsageRollup:
    """Aggregate last week's dailies into a weekly rollup.

    Intended to be called on Mondays.  The rollup covers the 7-day
    window from the previous Monday to the current Monday midnight.

    Parameters
    ----------
    db:
        An async Motor database handle for the tenant.
    target_date:
        Reference point (should be a Monday).  Defaults to
        ``datetime.utcnow()``.
    """
    ref = target_date or datetime.utcnow()
    start, end = _prev_week_range(ref)
    repo = GateRepository(db)
    return await _aggregate_and_upsert(repo, PeriodType.WEEKLY, start, end)


async def run_monthly_rollup(
    db: AsyncIOMotorDatabase,
    target_date: Optional[datetime] = None,
) -> TokenUsageRollup:
    """Aggregate last month's weeklies into a monthly rollup.

    Intended to be called on the 1st of the month.  The rollup covers
    the entire previous calendar month.

    Parameters
    ----------
    db:
        An async Motor database handle for the tenant.
    target_date:
        Reference point (should be the 1st of a month).  Defaults to
        ``datetime.utcnow()``.
    """
    ref = target_date or datetime.utcnow()
    start, end = _prev_month_range(ref)
    repo = GateRepository(db)
    return await _aggregate_and_upsert(repo, PeriodType.MONTHLY, start, end)


async def run_all_rollups(db: AsyncIOMotorDatabase) -> Dict[str, Optional[TokenUsageRollup]]:
    """Run daily rollup, and optionally weekly (Monday) and monthly (1st).

    Returns a dict with keys ``"daily"``, ``"weekly"``, ``"monthly"``
    mapped to the created rollup (or ``None`` if skipped).
    """
    now = datetime.utcnow()
    results: Dict[str, Optional[TokenUsageRollup]] = {
        "daily": None,
        "weekly": None,
        "monthly": None,
    }

    # Daily — always runs
    results["daily"] = await run_daily_rollup(db, target_date=now)

    # Weekly — only on Monday (weekday 0)
    if now.weekday() == 0:
        results["weekly"] = await run_weekly_rollup(db, target_date=now)
    else:
        logger.debug("Skipping weekly rollup (today is weekday %d, not Monday)", now.weekday())

    # Monthly — only on 1st of the month
    if now.day == 1:
        results["monthly"] = await run_monthly_rollup(db, target_date=now)
    else:
        logger.debug("Skipping monthly rollup (today is day %d, not 1st)", now.day)

    return results
