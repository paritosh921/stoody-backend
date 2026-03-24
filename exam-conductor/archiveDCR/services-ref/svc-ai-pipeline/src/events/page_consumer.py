"""NATS consumer for page.ready events.

Downloads page images from S3, runs the AI pipeline
(HWR -> step detection -> classification -> result assembly),
stores results in PostgreSQL, and publishes ai.result events.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import nats
from nats.aio.msg import Msg

from src.adapters.s3_adapter import download_page_image
from src.domain.classifier import classify_content
from src.domain.hwr_engine import recognize_text
from src.domain.result_builder import (
    build_question_result,
    build_result,
)
from src.domain.step_detector import detect_steps
from src.events.result_publisher import ResultPublisher
from src.storage.result_repo import ResultRepo

if TYPE_CHECKING:
    from src.adapters.model_adapter import ModelRegistry
    from src.config import Settings

logger = logging.getLogger(__name__)

# NATS subject for page-ready events.
SUBJECT_PAGE_READY = "EXAMPEN.page.ready"


class PageConsumer:
    """Subscribes to page.ready, orchestrates the AI pipeline."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._nc: nats.NATS | None = None
        self._sub = None

    async def start(self) -> None:
        """Connect to NATS and subscribe to page.ready."""
        self._nc = await nats.connect(self._settings.nats_url)
        js = self._nc.jetstream()
        self._sub = await js.subscribe(
            SUBJECT_PAGE_READY,
            durable="svc-ai-pipeline",
            manual_ack=True,
        )
        logger.info("Subscribed to %s", SUBJECT_PAGE_READY)
        # Start processing in the background
        import asyncio
        asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Unsubscribe and close NATS connection."""
        if self._sub:
            await self._sub.unsubscribe()
        if self._nc:
            await self._nc.close()

    async def _consume_loop(self) -> None:
        """Pull messages and process them."""
        async for msg in self._sub.messages:
            try:
                await self._handle_message(msg)
                await msg.ack()
            except Exception:
                logger.exception("Failed to process page.ready message")

    async def _handle_message(self, msg: Msg) -> None:
        """Process a single page.ready event."""
        payload = json.loads(msg.data)
        exam_id = payload["exam_id"]
        student_id = payload["student_id"]
        image_uri = payload["image_uri"]
        question_ids: list[str] = payload.get("question_ids", [])
        source_type = payload.get("authoritative_source", "strokes")

        logger.info(
            "Processing page.ready exam=%s student=%s questions=%d",
            exam_id, student_id, len(question_ids),
        )

        # 1. Download page image from S3
        image_data = await download_page_image(
            self._settings.minio_url,
            self._settings.minio_access_key,
            self._settings.minio_secret_key,
            self._settings.minio_bucket,
            image_uri,
        )

        # 2. Run AI pipeline per question
        model_version = self._registry.current_version()
        inference_fn = self._registry.get_inference_fn("hwr")
        question_results = []

        for q_id in question_ids:
            hwr_result = recognize_text(
                image_data=image_data,
                language="en",
                run_inference_fn=inference_fn,
                threshold=self._settings.confidence_threshold,
            )

            features = self._registry.get_inference_fn("classifier")(image_data)
            content_type_result = classify_content(features)

            steps = detect_steps(
                hwr_result.recognized_text,
                content_type_result.content_type.value,
            )

            qr = build_question_result(
                hwr=hwr_result,
                steps=steps,
                content_type=content_type_result.content_type,
                question_id=q_id,
            )
            question_results.append(qr)

        ai_result = build_result(
            question_results=question_results,
            exam_id=exam_id,
            student_id=student_id,
            model_version=model_version,
            source_type=source_type if source_type != "both" else "strokes",
        )

        # 3. Store result in PostgreSQL (INSERT, never UPDATE)
        repo = ResultRepo(self._settings.database_url)
        await repo.store_result(ai_result)

        # 4. Publish ai.result event AFTER DB commit
        publisher = ResultPublisher(self._nc)
        await publisher.publish(ai_result)

        logger.info(
            "Published ai.result exam=%s student=%s model=%s",
            exam_id, student_id, model_version,
        )
