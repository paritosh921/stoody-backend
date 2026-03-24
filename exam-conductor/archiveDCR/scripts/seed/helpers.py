"""Shared helper functions for seed data generation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone


def distribute_marks(rng: random.Random, total: int, steps: int) -> list[int]:
    """Distribute total marks across steps."""
    if steps == 1:
        return [total]
    parts = sorted(rng.sample(range(1, total), min(steps - 1, total - 1)))
    result = [parts[0]]
    for i in range(1, len(parts)):
        result.append(parts[i] - parts[i - 1])
    result.append(total - parts[-1])
    return [max(1, m) for m in result[:steps]]


def iso_now(offset_days: int = 0, offset_hours: int = 0) -> str:
    """ISO timestamp relative to now."""
    dt = datetime.now(timezone.utc) + timedelta(
        days=offset_days, hours=offset_hours,
    )
    return dt.isoformat(timespec="seconds")
