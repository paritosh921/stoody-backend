"""NATS consumer: page assembly pipeline.

Subscribes to ``EXAMPEN.stroke.processed``, assembles page images
from processed strokes, writes to ``page_repo``, and publishes
``EXAMPEN.page.ready``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..storage.page_repo import PageRepo
from . import subjects
from .publishers import publish_page_ready

logger = logging.getLogger(__name__)

DURABLE = "exampen-page-assembler"
QUEUE_GROUP = "exampen-page-workers"


async def page_assembly_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Assemble a page from processed stroke data.

    Steps:
        1. Extract exam_id, tenant_id, pen_mac, page_assignments
        2. For each page referenced in assignments, upsert the page doc
        3. Publish page.ready for each page that has enough data
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")
    pen_mac = payload.get("pen_mac", "")
    page_assignments = payload.get("page_assignments", [])

    logger.info(
        "Processing stroke.processed event_id=%s exam=%s pages=%d",
        event_id, exam_id, len(page_assignments),
    )

    db = await db_manager.get_tenant_db(tenant_id)
    repo = PageRepo(db)

    # Determine student_id from binding (pen_mac -> student_id).
    # For now, use pen_mac as a proxy; binding_repo lookup would go here.
    student_id = payload.get("student_id", pen_mac)

    seen_pages: set[int] = set()
    for assignment in page_assignments:
        page_number = assignment.get("page_number", 0)
        if page_number in seen_pages:
            continue
        seen_pages.add(page_number)

        page_data = {
            "pen_mac": pen_mac,
            "stroke_count": assignment.get("point_count", 0),
            "status": "assembled",
        }
        await repo.upsert(exam_id, student_id, page_number, tenant_id, page_data)

        # S3 URI would be set by a real image renderer; placeholder for now.
        s3_uri = (
            f"s3://exampen-pages/{tenant_id}/{exam_id}"
            f"/{student_id}/page_{page_number}.png"
        )

        await publish_page_ready(
            nats, exam_id, student_id, page_number, s3_uri,
            tenant_id=tenant_id,
        )

    logger.info(
        "Page assembly complete event_id=%s pages=%s",
        event_id, sorted(seen_pages),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.stroke.processed with durable JetStream consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await page_assembly_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.STROKE_PROCESSED,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered page_assembly_consumer")
