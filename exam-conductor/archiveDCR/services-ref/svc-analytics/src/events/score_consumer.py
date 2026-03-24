"""NATS JetStream consumer for score.updated events.

Subscribes to ``score.updated`` events published by svc-score-engine.
On each event:
1. Update local score cache (exam_score_cache table)
2. Recompute percentiles for the affected exam (idempotent)
3. Recompute leaderboard for the affected exam (idempotent)

svc-analytics is the ONLY writer of percentile data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import nats
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext

from src.domain.percentile import StudentScore, compute_percentiles
from src.domain.leaderboard import (
    LeaderboardScope,
    ScoreEntry,
    generate_leaderboard,
)
from src.storage.analytics_repo import AnalyticsRepo

_log = logging.getLogger(__name__)


class ScoreConsumer:
    """Consumes score.updated events and recomputes analytics."""

    def __init__(
        self,
        repo: AnalyticsRepo,
        nats_url: str,
        stream: str,
        subject: str,
        durable_name: str,
    ) -> None:
        self._repo = repo
        self._nats_url = nats_url
        self._stream = stream
        self._subject = subject
        self._durable_name = durable_name
        self._nc: NATSClient | None = None
        self._js: JetStreamContext | None = None
        self._sub: Any = None

    async def start(self) -> None:
        """Connect to NATS and subscribe to score.updated events."""
        self._nc = await nats.connect(self._nats_url)
        self._js = self._nc.jetstream()

        self._sub = await self._js.subscribe(
            self._subject,
            stream=self._stream,
            durable=self._durable_name,
            cb=self._handle_message,
        )
        _log.info(
            "Score consumer started subject=%s durable=%s",
            self._subject, self._durable_name,
        )

    async def stop(self) -> None:
        """Unsubscribe and close NATS connection."""
        if self._sub:
            await self._sub.unsubscribe()
        if self._nc:
            await self._nc.close()
        _log.info("Score consumer stopped")

    async def _handle_message(self, msg: Any) -> None:
        """Process a single score.updated event.

        Idempotent: reprocessing the same event produces the same
        materialized state.
        """
        try:
            payload = json.loads(msg.data.decode())
            await self._process_score_event(payload)
            await msg.ack()
        except Exception:
            _log.exception(
                "Failed to process score.updated event: %s",
                msg.data,
            )
            # NAK so NATS redelivers
            await msg.nak()

    async def _process_score_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Core processing logic for a score.updated event.

        Steps:
        1. Upsert score into local cache
        2. Fetch all scores for the exam
        3. Recompute percentiles
        4. Recompute leaderboard
        """
        exam_id = event["exam_id"]
        student_id = event["student_id"]
        total_score = event["total_score"]
        # tenant_id is derived from the exam context; for now we
        # store it alongside score events or derive from DB.
        tenant_id = event.get("tenant_id", "default")

        _log.info(
            "Processing score.updated exam=%s student=%s score=%s",
            exam_id, student_id, total_score,
        )

        # Step 1: Update local score cache
        await self._repo.upsert_score_cache(
            exam_id=exam_id,
            student_id=student_id,
            total_score=total_score,
            tenant_id=tenant_id,
        )

        # Step 2: Fetch all scores for this exam
        all_scores = await self._repo.get_exam_scores(
            exam_id=exam_id,
            tenant_id=tenant_id,
        )

        if not all_scores:
            return

        # Step 3: Recompute percentiles (idempotent)
        student_scores = [
            StudentScore(
                student_id=s["student_id"],
                score=s["total_score"],
            )
            for s in all_scores
        ]
        percentiles = compute_percentiles(student_scores)

        await self._repo.upsert_percentiles(
            exam_id=exam_id,
            percentiles=percentiles,
            tenant_id=tenant_id,
        )

        # Step 4: Recompute leaderboard (idempotent)
        score_entries = [
            ScoreEntry(
                student_id=s["student_id"],
                student_name=s.get("student_name", ""),
                score=s["total_score"],
                percentile=percentiles.get(s["student_id"], 0.0),
            )
            for s in all_scores
        ]
        leaderboard = generate_leaderboard(
            score_entries,
            scope=LeaderboardScope.INSTITUTE,
        )

        await self._repo.upsert_leaderboard(
            exam_id=exam_id,
            rows=[
                {
                    "student_id": entry.student_id,
                    "student_name": entry.student_name,
                    "rank": entry.rank,
                    "score": entry.score,
                    "percentile": entry.percentile,
                }
                for entry in leaderboard
            ],
            tenant_id=tenant_id,
        )

        _log.info(
            "Analytics recomputed exam=%s students=%d",
            exam_id, len(all_scores),
        )
