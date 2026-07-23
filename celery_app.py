"""Celery application and beat schedule for Stoody backend.

Provides a single Celery app instance with a beat schedule for periodic
tasks such as the LLM gate token usage rollup.

Usage::

    celery -A celery_app worker --loglevel=info
    celery -A celery_app beat --loglevel=info
    # Or combined:
    celery -A celery_app worker -B --loglevel=info
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

from config_async import settings

logger = logging.getLogger(__name__)

# Uvicorn adds this directory in main_async.py, but Celery is started through
# this module directly.  Keep the conducted-exam packages importable in both
# runtimes.
_exam_conductor_dir = str(Path(__file__).resolve().parent / "exam-conductor")
if _exam_conductor_dir not in sys.path:
    sys.path.insert(0, _exam_conductor_dir)

app = Celery(
    "stoody",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

app.conf.beat_schedule = {
    "llm-gate-daily-rollup": {
        "task": "celery_app.run_daily_gate_rollup",
        "schedule": crontab(hour=1, minute=0),
    },
    "exampen-processing-reconciler": {
        "task": "celery_app.reconcile_exampen_processing_jobs",
        "schedule": 60.0,
    },
}


@app.task(name="celery_app.run_daily_gate_rollup", bind=True, max_retries=2)
def run_daily_gate_rollup(self) -> dict:
    """Run daily token usage rollup for all tenants.

    Iterates all tenants in ``skb_master.tenants`` and runs
    ``run_all_rollups()`` against each tenant database.
    """
    from core.database import DatabaseManager

    async def _run() -> dict:
        db_mgr = DatabaseManager()
        await db_mgr.initialize()

        master_db = await db_mgr.get_master_db()
        cursor = master_db["tenants"].find(
            {"status": "active"},
            {"db_name": 1, "_id": 0},
        )
        tenants = await cursor.to_list(length=1000)

        results: dict = {"tenants_processed": 0, "errors": []}

        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            try:
                tenant_db = await db_mgr.get_tenant_db(db_name)
                if tenant_db is None:
                    continue

                from llm_gate.rollup import run_all_rollups

                await run_all_rollups(tenant_db)
                results["tenants_processed"] += 1
            except Exception as exc:
                logger.exception("Rollup failed for tenant %s", db_name)
                results["errors"].append({"tenant": db_name, "error": str(exc)})

        return results

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.exception("Gate rollup task failed")
        raise self.retry(exc=exc, countdown=60)


@app.task(name="celery_app.reconcile_exampen_processing_jobs", bind=True, max_retries=2)
def reconcile_exampen_processing_jobs(self) -> dict:
    """Replay durable PCR jobs after queue/worker interruptions for all tenants."""
    from core.database import DatabaseManager

    async def _run() -> dict:
        db_mgr = DatabaseManager()
        await db_mgr.initialize()
        master_db = await db_mgr.get_master_db()
        cursor = master_db["tenants"].find(
            {"status": "active"},
            {"db_name": 1, "_id": 0},
        )
        tenants = await cursor.to_list(length=1000)
        results: dict = {
            "tenants_processed": 0,
            "dispatched": 0,
            "stale_recovered": 0,
            "errors": [],
        }

        from services.exampen_workflow import reconcile_processing_jobs

        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            try:
                tenant_db = await db_mgr.get_tenant_db(db_name)
                if tenant_db is None:
                    continue
                result = await reconcile_processing_jobs(tenant_db, db_name=db_name)
                results["tenants_processed"] += 1
                results["dispatched"] += int(result.get("dispatched") or 0)
                results["stale_recovered"] += int(result.get("stale_recovered") or 0)
            except Exception as exc:
                logger.exception("ExamPen job reconciliation failed for tenant %s", db_name)
                results["errors"].append({"tenant": db_name, "error": str(exc)})
        return results

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.exception("ExamPen processing reconciliation task failed")
        raise self.retry(exc=exc, countdown=60)


@app.task(name="celery_app.process_exampen_pcr_submission", bind=True, max_retries=3)
def process_exampen_pcr_submission(
    self,
    db_name: str,
    job_id: str,
    required_pipeline_version: int,
) -> dict:
    """Run the durable OCR/segmentation/evaluation workflow for one PCR copy."""
    from core.database import DatabaseManager

    execution_token = f"celery:{self.request.id}"

    async def _run() -> dict:
        db_mgr = DatabaseManager()
        await db_mgr.initialize()
        tenant_db = await db_mgr.get_tenant_db(db_name)
        if tenant_db is None:
            raise RuntimeError(f"Tenant database {db_name} is not available")

        from services.exampen_workflow import process_pcr_processing_job

        return await process_pcr_processing_job(
            tenant_db,
            job_id,
            execution_token=execution_token,
            required_pipeline_version=required_pipeline_version,
        )

    async def _record_error(exc: Exception, terminal: bool) -> None:
        try:
            db_mgr = DatabaseManager()
            await db_mgr.initialize()
            tenant_db = await db_mgr.get_tenant_db(db_name)
            if tenant_db is None:
                return
            from services.exampen_workflow import (
                mark_processing_job_failed,
                mark_processing_job_retryable_error,
            )
            if terminal:
                await mark_processing_job_failed(
                    tenant_db,
                    job_id,
                    exc,
                    expected_lease_token=execution_token,
                )
            else:
                await mark_processing_job_retryable_error(
                    tenant_db,
                    job_id,
                    exc,
                    expected_lease_token=execution_token,
                )
        except Exception:
            logger.exception("Unable to record PCR processing failure for job %s", job_id)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        terminal = self.request.retries >= self.max_retries
        logger.exception("PCR processing task failed for job %s (terminal=%s)", job_id, terminal)
        asyncio.run(_record_error(exc, terminal))
        if terminal:
            return {"job_id": job_id, "status": "failed", "error": str(exc)}
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
