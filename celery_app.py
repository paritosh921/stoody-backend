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

from celery import Celery
from celery.schedules import crontab

from config_async import settings

logger = logging.getLogger(__name__)

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
