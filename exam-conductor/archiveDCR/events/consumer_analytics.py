"""NATS consumer: analytics recomputation pipeline.

Subscribes to ``EXAMPEN.score.updated``, recomputes percentiles,
leaderboard, and class stats, then writes results to
``analytics_cache_repo``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

from ..domain.class_stats import compute_class_stats
from ..domain.leaderboard import (
    LeaderboardScope,
    ScoreEntry,
    generate_leaderboard,
)
from ..domain.percentile import StudentScore, compute_percentiles
from ..storage.analytics_cache_repo import AnalyticsCacheRepo
from ..storage.score_event_store import ScoreEventStore
from . import subjects

logger = logging.getLogger(__name__)

DURABLE = "exampen-analytics"
QUEUE_GROUP = "exampen-analytics-workers"


async def analytics_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Recompute percentiles, leaderboard, and class stats for an exam.

    Triggered whenever a score is updated.  Fetches the full exam
    overview from the score store and recomputes all analytics
    atomically.
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")

    logger.info(
        "Processing score.updated event_id=%s exam=%s — recomputing analytics",
        event_id, exam_id,
    )

    db = await db_manager.get_tenant_db(tenant_id)
    score_store = ScoreEventStore(db)
    cache_repo = AnalyticsCacheRepo(db)

    # 1. Fetch aggregated scores per student
    overview = await score_store.get_exam_overview(exam_id, tenant_id)
    if not overview:
        logger.debug("No scores yet for exam=%s — skipping analytics", exam_id)
        return

    # 2. Compute percentiles
    student_scores = [
        StudentScore(student_id=row["student_id"], score=row["total_score"])
        for row in overview
    ]
    percentile_map = compute_percentiles(student_scores)

    percentile_entries = [
        {"student_id": sid, "percentile": pct}
        for sid, pct in percentile_map.items()
    ]
    await cache_repo.upsert_percentiles(exam_id, tenant_id, percentile_entries)

    # 3. Generate leaderboard
    score_entries = [
        ScoreEntry(
            student_id=row["student_id"],
            student_name=row.get("student_name", row["student_id"]),
            score=row["total_score"],
            percentile=percentile_map.get(row["student_id"], 0.0),
        )
        for row in overview
    ]
    leaderboard = generate_leaderboard(score_entries, LeaderboardScope.INSTITUTE)
    leaderboard_dicts = [asdict(entry) for entry in leaderboard]
    await cache_repo.upsert_leaderboard(exam_id, tenant_id, leaderboard_dicts)

    # 4. Compute and cache class stats
    all_scores = [row["total_score"] for row in overview]
    stats = compute_class_stats(all_scores)
    stats_doc = {
        "exam_id": exam_id,
        "type": "class_stats",
        **asdict(stats),
    }
    # Upsert class stats as a single document in the analytics cache
    await cache_repo._coll.update_one(
        {"exam_id": exam_id, "tenant_id": tenant_id, "type": "class_stats"},
        {"$set": stats_doc},
        upsert=True,
    )

    logger.info(
        "Analytics recomputed event_id=%s exam=%s students=%d",
        event_id, exam_id, len(overview),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.score.updated with durable JetStream consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await analytics_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.SCORE_UPDATED,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered analytics_consumer")
