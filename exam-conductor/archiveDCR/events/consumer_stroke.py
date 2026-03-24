"""NATS consumer: stroke processing pipeline.

Subscribes to ``EXAMPEN.stroke.raw``, runs dedup + normalize + page
assignment from domain modules, writes results via ``stroke_processed_repo``,
and publishes ``EXAMPEN.stroke.processed``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..domain.dedup import filter_duplicates, make_idempotency_key
from ..domain.normalizer import normalize_coordinates
from ..domain.page_assigner import build_page_assignments
from ..storage.stroke_processed_repo import StrokeProcessedRepo
from . import subjects
from .publishers import publish_stroke_processed

logger = logging.getLogger(__name__)

DURABLE = "exampen-stroke-processor"
QUEUE_GROUP = "exampen-stroke-workers"


async def stroke_processor_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Process a single raw stroke event.

    Steps:
        1. Extract exam_id, pen_mac, chunk_index, tenant_id
        2. Build idempotency key and check for duplicates
        3. Normalize raw coordinates to mm
        4. Assign strokes to pages/questions
        5. Persist processed strokes
        6. Publish stroke.processed event
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    pen_mac = payload.get("pen_mac", "")
    chunk_index = payload.get("chunk_index", 0)
    tenant_id = payload.get("tenant_id", "")
    stroke_data = payload.get("stroke_data", {})

    logger.info(
        "Processing stroke.raw event_id=%s exam=%s pen=%s chunk=%d",
        event_id, exam_id, pen_mac, chunk_index,
    )

    # 1. Dedup check
    idem_key = make_idempotency_key(exam_id, pen_mac, chunk_index)
    new_indices, _ = filter_duplicates([idem_key])
    if not new_indices:
        logger.debug("Duplicate stroke chunk ignored: %s", idem_key)
        return

    # 2. Normalize coordinates
    raw_points = stroke_data.get("points", [])
    book_type = stroke_data.get("book_type", "LS")
    normalized = normalize_coordinates(raw_points, book_type)

    # 3. Build processed stroke documents
    strokes = [{
        "exam_id": exam_id,
        "pen_mac": pen_mac,
        "chunk_index": chunk_index,
        "book_type": book_type,
        "page_number": stroke_data.get("page_number", 0),
        "normalized_points": normalized,
        "raw_point_count": len(raw_points),
    }]

    # 4. Build page assignments summary
    page_assignments = build_page_assignments(strokes)

    # 5. Persist to MongoDB
    db = await db_manager.get_tenant_db(tenant_id)
    repo = StrokeProcessedRepo(db)
    committed = await repo.commit_strokes(idem_key, strokes, tenant_id)
    if not committed:
        logger.debug("Stroke batch already committed: %s", idem_key)
        return

    # 6. Publish downstream event
    await publish_stroke_processed(
        nats, exam_id, pen_mac, page_assignments, tenant_id=tenant_id,
    )

    logger.info(
        "Stroke processed event_id=%s key=%s points=%d",
        event_id, idem_key, len(normalized),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.stroke.raw with durable JetStream consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await stroke_processor_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.STROKE_RAW,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered stroke_processor_consumer")
