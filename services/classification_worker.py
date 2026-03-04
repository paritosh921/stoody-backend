"""
Classification Worker for Stoody Notes

Background worker that polls classification_queue every 10s and processes
pending jobs through the OCR → classify pipeline.

Spawned as an asyncio.create_task in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Max jobs to process per poll cycle (per tenant DB)
BATCH_SIZE = 20

# Poll interval in seconds
POLL_INTERVAL = 10

# Max retry attempts per job
MAX_ATTEMPTS = 3


async def classification_worker_loop(db_manager) -> None:
    """Main worker loop — runs until cancelled."""
    import os
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    logger.info("Classification worker started")

    while True:
        try:
            await _process_pending_classifications(db_manager, openai_client)
        except asyncio.CancelledError:
            logger.info("Classification worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Classification worker error: {e}")
        await asyncio.sleep(POLL_INTERVAL)


async def _process_pending_classifications(
    db_manager,
    openai_client: Optional[AsyncOpenAI],
) -> None:
    """Poll all tenant DBs for pending classification jobs."""
    from services.note_classification_service import process_page

    now = datetime.utcnow()

    # Get all active tenant DB names from master
    tenant_db_names = await _get_active_tenant_dbs(db_manager)

    for db_name in tenant_db_names:
        try:
            tenant_db = await db_manager.get_tenant_db(db_name)
            if tenant_db is None:
                continue

            jobs = await tenant_db["classification_queue"].find(
                {
                    "status": "pending",
                    "process_after": {"$lte": now},
                    "attempts": {"$lt": MAX_ATTEMPTS},
                }
            ).limit(BATCH_SIZE).to_list(BATCH_SIZE)

            for job in jobs:
                await _process_single_job(tenant_db, job, openai_client)

        except Exception as e:
            logger.error(f"Error processing tenant {db_name}: {e}")


async def _process_single_job(
    tenant_db,
    job: Dict[str, Any],
    openai_client: Optional[AsyncOpenAI],
) -> None:
    """Process a single classification job with error handling."""
    from services.note_classification_service import process_page

    job_id = job["_id"]
    user_id = job.get("user_id", "")
    attempts = job.get("attempts", 0)

    # Mark as processing
    await tenant_db["classification_queue"].update_one(
        {"_id": job_id},
        {"$set": {"status": "processing"}, "$inc": {"attempts": 1}},
    )

    try:
        await process_page(tenant_db, user_id, job, openai_client)

        # Mark completed
        await tenant_db["classification_queue"].update_one(
            {"_id": job_id},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}},
        )
        logger.debug(
            f"Classified page {job.get('page_number')} for user {user_id}"
        )
    except Exception as e:
        new_status = "failed" if attempts + 1 >= MAX_ATTEMPTS else "pending"
        await tenant_db["classification_queue"].update_one(
            {"_id": job_id},
            {"$set": {"status": new_status, "error": str(e)[:500]}},
        )
        logger.warning(
            f"Classification job {job_id} {'failed permanently' if new_status == 'failed' else 'will retry'}: {e}"
        )


async def _get_active_tenant_dbs(db_manager) -> list:
    """Get list of tenant DB names from master tenants collection + B2C DB."""
    try:
        master_db = await db_manager.get_master_db()
        if master_db is None:
            return []

        cursor = master_db["tenants"].find(
            {"status": "active"},
            {"db_name": 1},
        )
        db_names = []
        async for doc in cursor:
            db_name = doc.get("db_name")
            if db_name:
                db_names.append(db_name)

        # Include B2C database
        try:
            from config_async import MONGODB_DB_STOODY
            if MONGODB_DB_STOODY and MONGODB_DB_STOODY not in db_names:
                db_names.append(MONGODB_DB_STOODY)
        except ImportError:
            pass

        return db_names
    except Exception as e:
        logger.error(f"Failed to get tenant DBs: {e}")
        return []
