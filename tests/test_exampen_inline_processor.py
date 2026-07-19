from __future__ import annotations

import asyncio
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

    async def fake_process(db, job_id: str, *, execution_token: str):
        assert db is tenant_db
        assert execution_token.startswith("inline:")
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

    async def slow_process(_db, job_id: str, *, execution_token: str):
        assert execution_token.startswith("inline:")
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
