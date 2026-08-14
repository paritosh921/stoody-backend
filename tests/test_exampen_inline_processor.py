from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


def _database(name: str):
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()[name]


class _DbManager:
    def __init__(self, master_db, tenant_dbs):
        self._master_db = master_db
        self._tenant_dbs = tenant_dbs

    async def get_master_db(self):
        return self._master_db

    async def get_tenant_db(self, db_name: str):
        return self._tenant_dbs.get(db_name)


@pytest.mark.asyncio
async def test_inline_processor_picks_up_durable_queued_job_from_active_tenant():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await master_db["tenants"].insert_one({"status": "active", "db_name": "skb_test"})
    await tenant_db["exampen_processing_jobs"].insert_one(
        {"job_id": "JOB-1", "submission_id": "SUB-1", "status": "queued"}
    )

    called: list[str] = []

    async def fake_process(
        db,
        job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert db is tenant_db
        assert execution_token.startswith("inline:")
        assert required_pipeline_version == 3
        called.append(job_id)
        return {"job_id": job_id, "status": "completed"}

    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=fake_process,
    ):
        assert await processor.run_once() == 1
        await asyncio.sleep(0)

    assert called == ["JOB-1"]


@pytest.mark.asyncio
async def test_inline_processor_does_not_schedule_same_job_twice_while_active():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await master_db["tenants"].insert_one({"status": "active", "db_name": "skb_test"})
    await tenant_db["exampen_processing_jobs"].insert_one(
        {"job_id": "JOB-1", "submission_id": "SUB-1", "status": "queued"}
    )

    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def slow_process(
        _db,
        job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert execution_token.startswith("inline:")
        assert required_pipeline_version == 3
        calls.append(job_id)
        started.set()
        await release.wait()
        return {"job_id": job_id, "status": "completed"}

    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=slow_process,
    ):
        assert await processor.run_once() == 1
        await started.wait()
        assert await processor.run_once() == 0
        release.set()
        await asyncio.sleep(0)

    assert calls == ["JOB-1"]


@pytest.mark.asyncio
async def test_inline_processor_recovers_stalled_processing_job_and_retries_it():
    from services.exampen_inline_processor import InlinePCRProcessor
    from services.exampen_workflow import PROCESSING_HEARTBEAT_STALE_SECONDS

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await master_db["tenants"].insert_one({"status": "active", "db_name": "skb_test"})
    now = datetime.now(timezone.utc)
    await tenant_db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-stalled",
            "submission_id": "SUB-stalled",
            "status": "processing",
            "lease_token": "dead-worker",
            "lease_expires_at": now + timedelta(minutes=20),
            "updated_at": now
            - timedelta(seconds=PROCESSING_HEARTBEAT_STALE_SECONDS + 1),
        }
    )

    called: list[str] = []

    async def fake_process(
        _db,
        job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert execution_token.startswith("inline:")
        assert required_pipeline_version == 3
        called.append(job_id)
        return {"job_id": job_id, "status": "completed"}

    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=fake_process,
    ):
        # Recovery persists a short backoff first; it must not immediately
        # reclaim the same crashed job in the recovery pass.
        assert await processor.run_once() == 0
        await asyncio.sleep(0)

    assert called == []
    recovered = await tenant_db["exampen_processing_jobs"].find_one(
        {"job_id": "JOB-stalled"}
    )
    assert recovered["status"] == "retryable_error"
    assert "lease_token" not in recovered
    retry_at = recovered["next_retry_at"]
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    assert retry_at > now


@pytest.mark.asyncio
async def test_inline_processor_honors_durable_retry_schedule():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await master_db["tenants"].insert_one({"status": "active", "db_name": "skb_test"})
    await tenant_db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-delayed",
            "submission_id": "SUB-delayed",
            "status": "retryable_error",
            "next_retry_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
    )
    called: list[str] = []

    async def fake_process(
        _db,
        job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert execution_token.startswith("inline:")
        assert required_pipeline_version == 3
        called.append(job_id)
        return {"job_id": job_id, "status": "completed"}

    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=fake_process,
    ):
        assert await processor.run_once() == 0
        assert called == []
        await tenant_db["exampen_processing_jobs"].update_one(
            {"job_id": "JOB-delayed"},
            {
                "$set": {
                    "next_retry_at": datetime.now(timezone.utc)
                    - timedelta(seconds=1)
                }
            },
        )
        assert await processor.run_once() == 1
        await asyncio.sleep(0)

    assert called == ["JOB-delayed"]
