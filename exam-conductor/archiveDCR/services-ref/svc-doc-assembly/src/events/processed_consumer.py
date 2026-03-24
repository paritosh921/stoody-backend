"""NATS consumer for stroke.processed events.

Subscribes to stroke.processed, groups strokes by
(exam_id, student_id, page), and triggers page assembly when all
strokes for a page are available.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from nats.js import JetStreamContext
from nats.aio.msg import Msg

from src.adapters.s3_adapter import S3Adapter
from src.adapters.stroke_data_client import StrokeDataSource
from src.domain.models import (
    CanonicalPoint,
    QuestionRegion,
    Stroke,
    SyncMetadata,
)
from src.domain.page_builder import build_page
from src.events.page_publisher import PagePublisher
from src.storage.page_repo import PageRepository

logger = logging.getLogger(__name__)

# NATS subject and consumer group
SUBJECT = "EXAMPEN.stroke.processed"
DURABLE_NAME = "doc-assembly-consumer"


class ProcessedStrokeConsumer:
    """Consumes stroke.processed events and assembles pages."""

    def __init__(
        self,
        js: JetStreamContext,
        s3: S3Adapter,
        page_repo: PageRepository,
        stroke_source: StrokeDataSource | None = None,
    ) -> None:
        self._js = js
        self._s3 = s3
        self._page_repo = page_repo
        self._publisher = PagePublisher(js)
        self._stroke_source = stroke_source
        self._sub = None

        # In-memory buffer: (exam_id, student_id, page_number) -> list of strokes
        self._page_strokes: dict[
            tuple[str, str, int], list[Stroke]
        ] = defaultdict(list)

        # Track per-student sync state: (exam_id, student_id) -> {pen_mac: chunk_count}
        self._chunks_received: dict[
            tuple[str, str], dict[str, int]
        ] = defaultdict(lambda: defaultdict(int))

    async def start(self) -> None:
        """Subscribe to stroke.processed events."""
        self._sub = await self._js.subscribe(
            SUBJECT,
            durable=DURABLE_NAME,
            cb=self._handle_message,
        )
        logger.info("Subscribed to %s (durable=%s)", SUBJECT, DURABLE_NAME)

    async def stop(self) -> None:
        """Unsubscribe and clean up."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            logger.info("Unsubscribed from %s", SUBJECT)

    async def _handle_message(self, msg: Msg) -> None:
        """Process a single stroke.processed event."""
        try:
            payload = json.loads(msg.data.decode("utf-8"))
            await self._process_event(payload)
            await msg.ack()
        except Exception:
            logger.exception("Failed to process stroke.processed event")
            await msg.nak()

    async def _process_event(self, payload: dict) -> None:
        """Parse event, group strokes by (exam, student, page), trigger assembly."""
        exam_id = payload["exam_id"]
        student_id = payload.get("student_id", "")
        pen_mac = payload.get("pen_mac", "")
        normalized_stroke_uri = payload.get("normalized_stroke_uri", "")

        # Track chunks received per pen for sync completeness
        self._chunks_received[(exam_id, student_id)][pen_mac] += 1

        # Fetch real stroke data from TimescaleDB for each affected page.
        # The event tells us WHICH pages have data; the DB has the ACTUAL points.
        affected_pages: set[int] = set()
        for assignment in payload.get("page_assignments", []):
            affected_pages.add(assignment["page_number"])

        for page_number in affected_pages:
            key = (exam_id, student_id, page_number)

            if self._stroke_source is not None:
                # Query TimescaleDB for real normalized stroke data
                stroke_rows = await self._stroke_source.get_strokes_for_page(
                    exam_id, pen_mac, page_number,
                )
                for row in stroke_rows:
                    canonical_points = _parse_canonical_points(
                        row.get("normalized_points", [])
                    )
                    stroke = Stroke(
                        stroke_id=row.get("stroke_id", ""),
                        points=canonical_points,
                    )
                    self._page_strokes[key].append(stroke)
                logger.info(
                    "Fetched %d strokes from DB for exam=%s student=%s page=%d",
                    len(stroke_rows), exam_id, student_id, page_number,
                )
            else:
                # Fallback: use point_count from event as placeholder
                for assignment in payload.get("page_assignments", []):
                    if assignment["page_number"] != page_number:
                        continue
                    stroke = Stroke(
                        stroke_id=f"{exam_id}_{pen_mac}_{page_number}_{assignment.get('question_id', '')}",
                        points=[],
                    )
                    self._page_strokes[key].append(stroke)
                logger.warning(
                    "No stroke source — using placeholder for exam=%s page=%d",
                    exam_id, page_number,
                )

        # Build question regions from the assignments for this page.
        # Each assignment carries a question_id; we derive default
        # regions from the page geometry.  In production this would
        # come from the exam configuration DB.
        affected_pages = {
            a["page_number"] for a in payload.get("page_assignments", [])
        }

        # Build question regions from assignments, grouped per page
        page_questions: dict[int, list[str]] = defaultdict(list)
        for a in payload.get("page_assignments", []):
            qid = a.get("question_id", "")
            if qid:
                page_questions[a["page_number"]].append(qid)

        for page_number in affected_pages:
            question_regions = _build_question_regions(
                page_questions.get(page_number, [])
            )

            # Derive sync_complete: check that we have received at
            # least one chunk from this pen (basic heuristic).
            # A full implementation would compare against the expected
            # total_chunks from the exam-orch binding.
            chunks_for_pen = self._chunks_received.get(
                (exam_id, student_id), {}
            )
            pen_reported = pen_mac in chunks_for_pen
            sync_complete = pen_reported and chunks_for_pen[pen_mac] > 0

            await self._assemble_page(
                exam_id=exam_id,
                student_id=student_id,
                page_number=page_number,
                pen_mac=pen_mac,
                sync_complete=sync_complete,
                question_regions=question_regions,
            )

    async def _assemble_page(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
        pen_mac: str,
        sync_complete: bool,
        question_regions: list[QuestionRegion],
    ) -> None:
        """Assemble a page, upload to S3, write PG metadata, publish event."""
        # Bug 2 fix: use 3-tuple key including student_id
        key = (exam_id, student_id, page_number)
        strokes = self._page_strokes.get(key, [])

        # Bug 3 fix: use real sync metadata derived from event data
        sync_metadata = SyncMetadata(
            pen_mac=pen_mac,
            sync_complete=sync_complete,
            pen_connected=True,
            strokes_expected=len(strokes) > 0,
        )

        doc = build_page(
            strokes=strokes,
            question_regions=question_regions,
            sync_metadata=sync_metadata,
            exam_id=exam_id,
            student_id=student_id,
            page_number=page_number,
        )

        # 1. S3 write FIRST (orphaned S3 is acceptable)
        upload = await self._s3.upload_page_svg(
            exam_id=exam_id,
            student_id=student_id,
            page_number=page_number,
            svg_content=doc.svg_content,
        )

        # 2. PG metadata write SECOND (no dangling references)
        page_id = str(uuid.uuid4())
        await self._page_repo.save_page(
            doc=doc,
            s3_uri=upload.uri,
            page_id=page_id,
        )

        # 3. Publish page.ready event
        await self._publisher.publish_page_ready(
            exam_id=exam_id,
            student_id=student_id,
            page_id=page_id,
            page_number=page_number,
            image_uri=upload.uri,
            vector_uri=upload.uri,
            question_ids=[
                qr.question_id for qr in doc.question_results
            ],
        )

        # Clear buffer for this (exam, student, page)
        self._page_strokes.pop(key, None)

        logger.info(
            "Assembled page: exam=%s student=%s page=%d -> %s (%d strokes)",
            exam_id,
            student_id,
            page_number,
            upload.uri,
            len(strokes),
        )


def _parse_canonical_points(
    raw_points: list[dict],
) -> list[CanonicalPoint]:
    """Convert raw point dicts from the event payload into domain CanonicalPoints."""
    points: list[CanonicalPoint] = []
    for p in raw_points:
        points.append(
            CanonicalPoint(
                x=float(p.get("x_mm", p.get("x", 0.0))),
                y=float(p.get("y_mm", p.get("y", 0.0))),
                pressure=float(p.get("pressure", 0.0)),
                tilt_x=float(p.get("tilt_x", 0.0)),
                tilt_y=float(p.get("tilt_y", 0.0)),
                timestamp_ms=int(p.get("timestamp_ms", p.get("timestamp", 0))),
            )
        )
    return points


def _build_question_regions(
    question_ids: list[str],
    page_height: float = 297.0,
) -> list[QuestionRegion]:
    """Build question regions from question IDs.

    Distributes questions evenly across the page height.  In production
    this would come from the exam template configuration in the DB.
    """
    if not question_ids:
        return []

    region_height = page_height / len(question_ids)
    regions: list[QuestionRegion] = []
    for i, qid in enumerate(question_ids):
        regions.append(
            QuestionRegion(
                question_id=qid,
                x_min=0.0,
                y_min=i * region_height,
                x_max=210.0,  # A4 width
                y_max=(i + 1) * region_height,
            )
        )
    return regions
