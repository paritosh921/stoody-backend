"""Development-safe automatic processor for durable PCR jobs.

Production uses the supervised Celery worker in :mod:`celery_app`.  Local
development should still behave like the product: after a student submits an
answer copy, the job is picked up automatically instead of sitting in Redis
until somebody opens a second terminal.  This processor only runs when the
explicit development setting enables it, and it claims the same durable job
record as Celery so the two execution modes cannot process a copy twice.
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any

from services.exampen_workflow import (
    DISPATCHABLE_JOB_STATUSES,
    PROCESSING_JOBS_COLLECTION,
    mark_processing_job_retryable_error,
    process_pcr_processing_job,
)


logger = logging.getLogger(__name__)


class InlinePCRProcessor:
    """Poll durable PCR jobs and process them inside one local API process.

    The processor is intentionally a development fallback, not a replacement
    for a production worker service.  Jobs are atomically claimed by
    ``process_pcr_processing_job``; if a real Celery worker takes one first,
    this processor harmlessly skips it.
    """

    def __init__(
        self,
        db_manager: Any,
        *,
        poll_seconds: float = 3.0,
        concurrency: int = 1,
        retry_delay_seconds: float = 60.0,
    ) -> None:
        self._db_manager = db_manager
        self._poll_seconds = max(1.0, float(poll_seconds))
        self._retry_delay_seconds = max(1.0, float(retry_delay_seconds))
        self._concurrency = max(1, int(concurrency))
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._active_jobs: dict[str, asyncio.Task[None]] = {}
        self._retry_not_before: dict[str, float] = {}
        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None

    @property
    def running_job_count(self) -> int:
        return len(self._active_jobs)

    async def start(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name="exampen-inline-pcr-processor",
        )
        logger.info(
            "ExamPen inline PCR processor started (poll=%ss, concurrency=%s)",
            self._poll_seconds,
            self._concurrency,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        tasks: list[asyncio.Task[Any]] = []
        if self._loop_task is not None:
            self._loop_task.cancel()
            tasks.append(self._loop_task)
        tasks.extend(self._active_jobs.values())
        for task in self._active_jobs.values():
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_jobs.clear()
        self._loop_task = None
        logger.info("ExamPen inline PCR processor stopped")

    async def run_once(self) -> int:
        """Schedule currently dispatchable PCR jobs and return how many started."""
        master_db = await self._db_manager.get_master_db()
        if master_db is None:
            logger.warning("Inline PCR processor skipped: master database is unavailable")
            return 0

        tenants = await master_db["tenants"].find(
            {"status": "active"},
            {"db_name": 1, "_id": 0},
        ).to_list(length=1000)

        scheduled = 0
        # Keep capacity bookkeeping independent of asyncio's private semaphore
        # state.  A task may be scheduled before it has acquired the semaphore,
        # so ``_value`` is neither public API nor a reliable job count here.
        available_slots = max(0, self._concurrency - len(self._active_jobs))
        if available_slots == 0:
            return 0

        for tenant in tenants:
            if available_slots <= 0:
                break
            db_name = str(tenant.get("db_name") or "").strip()
            if not db_name:
                continue
            try:
                tenant_db = await self._db_manager.get_tenant_db(db_name)
            except Exception:
                logger.exception("Inline PCR processor could not resolve tenant %s", db_name)
                continue
            if tenant_db is None:
                continue

            cursor = tenant_db[PROCESSING_JOBS_COLLECTION].find(
                {"status": {"$in": list(DISPATCHABLE_JOB_STATUSES)}},
                {"job_id": 1},
            ).sort("updated_at", 1).limit(available_slots)
            jobs = await cursor.to_list(length=available_slots)
            for job in jobs:
                job_id = str(job.get("job_id") or "").strip()
                if not job_id:
                    continue
                key = f"{db_name}:{job_id}"
                if key in self._active_jobs or monotonic() < self._retry_not_before.get(key, 0):
                    continue
                task = asyncio.create_task(
                    self._process_job(key, tenant_db, job_id),
                    name=f"exampen-inline-{job_id}",
                )
                self._active_jobs[key] = task
                task.add_done_callback(lambda _task, active_key=key: self._active_jobs.pop(active_key, None))
                scheduled += 1
                available_slots -= 1
                if available_slots <= 0:
                    break

        return scheduled

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Inline PCR processor polling pass failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_seconds)
            except asyncio.TimeoutError:
                continue

    async def _process_job(self, key: str, tenant_db: Any, job_id: str) -> None:
        async with self._semaphore:
            try:
                await process_pcr_processing_job(tenant_db, job_id)
                self._retry_not_before.pop(key, None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Inline PCR processing failed for job %s", job_id)
                try:
                    await mark_processing_job_retryable_error(tenant_db, job_id, exc)
                finally:
                    self._retry_not_before[key] = monotonic() + self._retry_delay_seconds
