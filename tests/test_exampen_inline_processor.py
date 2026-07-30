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
        assert required_pipeline_version == 4
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
        assert required_pipeline_version == 4
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
        assert required_pipeline_version == 4
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

    assert called == ["JOB-stalled"]
    recovered = await tenant_db["exampen_processing_jobs"].find_one(
        {"job_id": "JOB-stalled"}
    )
    assert recovered["status"] == "retryable_error"
    assert "lease_token" not in recovered


@pytest.mark.asyncio
async def test_inline_processor_releases_owned_job_when_hot_reload_cancels_it():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await tenant_db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-reload",
            "submission_id": "SUB-reload",
            "status": "processing",
            "attempts": 1,
            "lease_token": "placeholder",
        }
    )
    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    async def cancelled_process(
        _db,
        _job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert required_pipeline_version == 4
        await tenant_db["exampen_processing_jobs"].update_one(
            {"job_id": "JOB-reload"},
            {"$set": {"lease_token": execution_token}},
        )
        raise asyncio.CancelledError

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=cancelled_process,
    ):
        with pytest.raises(asyncio.CancelledError):
            await processor._process_job(
                "skb_test:JOB-reload",
                tenant_db,
                "JOB-reload",
            )

    stored = await tenant_db["exampen_processing_jobs"].find_one(
        {"job_id": "JOB-reload"}
    )
    assert stored["status"] == "retryable_error"
    assert "restarted" in stored["last_error"]
    assert "lease_token" not in stored


@pytest.mark.asyncio
async def test_inline_processor_stops_retrying_after_attempt_limit():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await tenant_db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-failing",
            "submission_id": "SUB-failing",
            "status": "processing",
            "attempts": 3,
            "lease_token": "placeholder",
        }
    )
    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    async def failed_process(
        _db,
        _job_id: str,
        *,
        execution_token: str,
        required_pipeline_version: int,
    ):
        assert required_pipeline_version == 4
        await tenant_db["exampen_processing_jobs"].update_one(
            {"job_id": "JOB-failing"},
            {"$set": {"lease_token": execution_token}},
        )
        raise RuntimeError("model returned invalid structured output")

    with patch(
        "services.exampen_inline_processor.process_pcr_processing_job",
        new=failed_process,
    ):
        await processor._process_job(
            "skb_test:JOB-failing",
            tenant_db,
            "JOB-failing",
        )

    stored = await tenant_db["exampen_processing_jobs"].find_one(
        {"job_id": "JOB-failing"}
    )
    assert stored["status"] == "failed"
    assert "invalid structured output" in stored["last_error"]
    assert stored["failure_history"][-1]["attempt"] == 3
    assert "lease_token" not in stored


@pytest.mark.asyncio
async def test_inline_processor_respects_database_retry_time_across_instances():
    from services.exampen_inline_processor import InlinePCRProcessor

    master_db = _database("skb_master")
    tenant_db = _database("skb_test")
    await master_db["tenants"].insert_one(
        {"status": "active", "db_name": "skb_test"}
    )
    await tenant_db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-backoff",
            "submission_id": "SUB-backoff",
            "status": "retryable_error",
            "next_attempt_at": datetime.now(timezone.utc)
            + timedelta(minutes=5),
        }
    )
    processor = InlinePCRProcessor(
        _DbManager(master_db, {"skb_test": tenant_db}),
        poll_seconds=1,
        concurrency=1,
    )

    assert await processor.run_once() == 0
