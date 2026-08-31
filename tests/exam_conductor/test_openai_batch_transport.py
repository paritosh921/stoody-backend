from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import httpx

_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
_EC_DIR = os.path.join(_BACKEND_DIR, "exam-conductor")
for path in (_BACKEND_DIR, _EC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from llm_gate.batch import BatchReplayGate, DeferredBatchCall
from llm_gate.budget import BudgetChecker
from llm_gate.models import BudgetExhaustedError, GateConfig, GateResponse, TokenUsage
from llm_gate.provider import estimate_cost
from services.exampen_openai_batch import (
    BATCH_ITEMS_COLLECTION,
    BATCH_PARTS_COLLECTION,
    OpenAIBatchClient,
    PROCESSING_JOBS_COLLECTION,
    StaleBatchGenerationError,
    _batch_max_requests_per_part,
    _create_provider_parts,
    _import_item,
    _recover_interrupted_part_creation,
    cancel_economy_batch_group,
    classify_economy_batch_failure,
    describe_provider_batch_progress,
    parse_batch_jsonl,
    partition_batch_requests,
    prepare_economy_batch_group,
    reconcile_economy_batches,
)


def test_provider_progress_exposes_delayed_finalizing_without_inventing_completion():
    observed_at = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)

    progress = describe_provider_batch_progress(
        {
            "status": "finalizing",
            "expires_at": int(observed_at.timestamp()) - 90,
            "finalizing_at": int(observed_at.timestamp()) - 120,
            "request_counts": {"total": 2, "completed": 1, "failed": 0},
        },
        now=observed_at,
    )

    assert progress["job_status"] == "provider_finalizing"
    assert progress["stage"] == "provider_finalizing"
    assert progress["delayed"] is True
    assert progress["delay_seconds"] == 90
    assert progress["completed"] == 1
    assert "1/2 requests complete" in progress["message"]


def test_batch_file_access_failure_requires_provider_configuration_change():
    result = classify_economy_batch_failure(
        "Cannot find file file-123, or organization org-123 does not have access to it."
    )

    assert result["failure_code"] == "provider_batch_file_access"
    assert result["retryable"] is False
    assert "Batch-enabled OpenAI project" in result["operator_action"]


def test_provider_timeout_has_specific_retryable_failure_contract():
    result = classify_economy_batch_failure("Temporary provider timeout")

    assert result["failure_code"] == "provider_batch_expired"
    assert result["retryable"] is True
    assert "Retry only this copy" in result["operator_action"]


def test_economy_batch_defaults_to_one_copy_per_provider_failure_domain(monkeypatch):
    monkeypatch.delenv("EXAMPEN_BATCH_MAX_REQUESTS_PER_PART", raising=False)
    assert _batch_max_requests_per_part() == 1

    monkeypatch.setenv("EXAMPEN_BATCH_MAX_REQUESTS_PER_PART", "7")
    assert _batch_max_requests_per_part() == 7


@pytest.mark.asyncio
async def test_superseded_batch_generation_never_replays_or_overwrites_marks():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-new-run",
        "status": "provider_processing",
        "grading_generation": 3,
    })
    item = {
        "custom_id": "copy-old-run",
        "job_id": "job-new-run",
        "submission_id": "submission-1",
        "job_generation": 2,
    }

    with patch("services.exampen_openai_batch._run_grader", new=AsyncMock()) as grader:
        with pytest.raises(StaleBatchGenerationError):
            await _import_item(
                db,
                item=item,
                provider_line={
                    "response": {"status_code": 200, "body": {"id": "response-old"}}
                },
            )

    grader.assert_not_awaited()
    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-new-run"})
    assert job["status"] == "provider_processing"
    assert job["grading_generation"] == 3


@pytest.mark.asyncio
async def test_legacy_batch_item_is_generation_zero_not_a_wildcard():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-retried-after-deploy",
        "status": "provider_processing",
        "grading_generation": 1,
    })
    legacy_item = {
        "custom_id": "copy-from-generation-zero",
        "job_id": "job-retried-after-deploy",
        "submission_id": "submission-1",
    }

    with patch("services.exampen_openai_batch._run_grader", new=AsyncMock()) as grader:
        with pytest.raises(StaleBatchGenerationError):
            await _import_item(
                db,
                item=legacy_item,
                provider_line={
                    "response": {"status_code": 200, "body": {"id": "legacy-response"}}
                },
            )

    grader.assert_not_awaited()
    job = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "job-retried-after-deploy"}
    )
    assert job["status"] == "provider_processing"
    assert job["grading_generation"] == 1


@pytest.mark.asyncio
async def test_batch_client_keeps_project_key_scope_and_waits_for_file_processing():
    calls = []
    file_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal file_reads
        calls.append(request)
        headers = {
            "x-request-id": f"req-{len(calls)}",
            # Response metadata is deliberately not suitable for request
            # authority: OpenAI can return an organization label here.
            "openai-organization": "school-label",
            "openai-project": "proj_exampen",
        }
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(
                200,
                headers=headers,
                json={"id": "file-1", "purpose": "batch", "status": "uploaded"},
            )
        if request.method == "GET" and request.url.path == "/v1/files/file-1":
            file_reads += 1
            return httpx.Response(
                200,
                headers=headers,
                json={
                    "id": "file-1",
                    "purpose": "batch",
                    "status": "uploaded" if file_reads == 1 else "processed",
                },
            )
        if request.method == "POST" and request.url.path == "/v1/batches":
            return httpx.Response(
                200,
                headers=headers,
                json={"id": "batch-1", "status": "validating"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    client = OpenAIBatchClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with patch("services.exampen_openai_batch.asyncio.sleep", new=AsyncMock()):
        uploaded = await client.upload_jsonl("copies.jsonl", b"{}\n")
        ready = await client.wait_for_file_ready(uploaded["id"])
        batch = await client.create_batch(input_file_id=uploaded["id"], metadata={})

    assert ready["status"] == "processed"
    assert batch["id"] == "batch-1"
    assert uploaded["_request_metadata"]["organization"] == "school-label"
    assert uploaded["_request_metadata"]["project"] == "proj_exampen"
    assert client.scope == {}
    assert all("openai-organization" not in call.headers for call in calls)
    assert all("openai-project" not in call.headers for call in calls)


def test_batch_client_uses_only_explicit_valid_scope_ids():
    client = OpenAIBatchClient(
        api_key="test-key",
        organization="org-school",
        project="proj_exampen",
    )

    assert client.scope == {
        "organization": "org-school",
        "project": "proj_exampen",
    }
    assert client._headers["OpenAI-Organization"] == "org-school"
    assert client._headers["OpenAI-Project"] == "proj_exampen"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"organization": "school-label"}, "organization ID"),
        ({"project": "project-label"}, "project ID"),
    ],
)
def test_batch_client_rejects_scope_labels(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        OpenAIBatchClient(api_key="test-key", **kwargs)


@pytest.mark.asyncio
async def test_batch_client_rejects_provider_file_processing_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "file-bad",
                "purpose": "batch",
                "status": "error",
                "status_details": "invalid JSONL",
            },
        )

    client = OpenAIBatchClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="invalid JSONL"):
        await client.wait_for_file_ready("file-bad")


def test_batch_gate_defers_exact_authorized_responses_body():
    async def _run():
        gate = AsyncMock()
        gate.prepare_batch_responses_call.return_value = {
            "model": "gpt-5.1",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "copy"}]}],
            "store": False,
        }
        wrapper = BatchReplayGate(gate)
        with pytest.raises(DeferredBatchCall) as captured:
            await wrapper.call(
                "gpt-5.1",
                "",
                "pcr_eval_core",
                responses_input=[{"role": "user", "content": []}],
                metadata={"submission_id": "must-not-be-provider-metadata"},
            )
        assert captured.value.call_index == 0
        assert captured.value.request_body["store"] is False
        assert captured.value.request_body["model"] == "gpt-5.1"

    asyncio.run(_run())


def test_batch_gate_replays_without_double_logging_recorded_calls():
    async def _run():
        gate = AsyncMock()
        gate.prepare_batch_responses_call.return_value = {"model": "gpt-5.1", "input": [], "store": False}
        gate.record_batch_response.return_value = GateResponse(
            content='{"ok":true}',
            usage=TokenUsage(model="gpt-5.1", caller="pcr_eval_core"),
        )
        wrapper = BatchReplayGate(
            gate,
            response_bodies=[{
                "model": "gpt-5.1",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }],
            recorded_call_indexes=[0],
        )
        response = await wrapper.call(
            "gpt-5.1",
            "",
            "pcr_eval_core",
            responses_input=[{"role": "user", "content": []}],
        )
        assert response.content == '{"ok":true}'
        assert gate.record_batch_response.await_args.kwargs["persist_log"] is False

    asyncio.run(_run())


def test_partition_is_size_bounded_and_never_mixes_models():
    entries = []
    for index, model in enumerate(("gpt-5.1", "gpt-5.1", "gpt-4o")):
        entries.append({
            "custom_id": f"copy-{index}",
            "model": model,
            "request_body": {"model": model, "input": "x" * 20},
        })
    partitions, oversized = partition_batch_requests(entries, max_bytes=140, max_requests=10)
    assert not oversized
    assert sum(len(partition) for partition in partitions) == 3
    assert all(len({entry["model"] for entry in partition}) == 1 for partition in partitions)


def test_parse_batch_jsonl_maps_out_of_order_results_by_custom_id():
    content = (
        '{"custom_id":"copy-2","response":{"status_code":200,"body":{"id":"two"}}}\n'
        '{"custom_id":"copy-1","response":{"status_code":200,"body":{"id":"one"}}}\n'
    )
    parsed = parse_batch_jsonl(content)
    assert parsed["copy-1"]["response"]["body"]["id"] == "one"
    assert parsed["copy-2"]["response"]["body"]["id"] == "two"


def test_batch_cost_uses_discount_and_cached_input_rate():
    standard = estimate_cost("gpt-5.1", 1_000_000, 1_000_000)
    batch = estimate_cost(
        "gpt-5.1",
        1_000_000,
        1_000_000,
        cache_read_tokens=500_000,
        batch=True,
    )
    assert standard == pytest.approx(11.25)
    assert batch == pytest.approx(5.34375)


@pytest.mark.asyncio
async def test_batch_reservation_fails_closed_when_group_exceeds_daily_headroom():
    repo = AsyncMock()
    repo.sum_tokens_since.return_value = 80
    checker = BudgetChecker(repo)

    with pytest.raises(BudgetExhaustedError) as captured:
        await checker.check_reservation(
            GateConfig(daily_token_limit=100),
            reserved_tokens=21,
        )

    assert captured.value.period == "daily"
    assert captured.value.used_tokens == 101


@pytest.mark.asyncio
async def test_interrupted_creation_recovers_existing_provider_batch_without_resubmitting():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-safe-recovery",
        "batch_group_id": "econ-1",
        "status": "creating",
        "input_file_id": "file-input",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_one({
        "custom_id": "copy-1",
        "local_part_id": "part-safe-recovery",
        "job_id": "job-1",
        "import_status": "pending",
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-1",
        "status": "preparing_batch",
    })
    client = AsyncMock()
    client.list_batches.return_value = {
        "data": [{
            "id": "batch-provider-1",
            "status": "in_progress",
            "metadata": {"stoody_part": "part-safe-recovery"},
        }],
        "has_more": False,
    }

    await _recover_interrupted_part_creation(
        db,
        group={"batch_group_id": "econ-1"},
        client=client,
    )

    part = await db[BATCH_PARTS_COLLECTION].find_one({"local_part_id": "part-safe-recovery"})
    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-1"})
    assert part["provider_batch_id"] == "batch-provider-1"
    assert part["recovered_after_interruption"] is True
    assert job["status"] == "provider_processing"
    client.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_before_provider_creation_returns_jobs_to_waiting():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-cancel",
        "status": "queued",
        "job_ids": ["job-cancel"],
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-cancel",
        "status": "batch_queued",
        "provider_batch_group_id": "econ-cancel",
    })

    result = await cancel_economy_batch_group(
        db,
        batch_group_id="econ-cancel",
    )

    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-cancel"})
    assert result["status"] == "cancelled"
    assert job["status"] == "waiting_for_batch"
    assert "provider_batch_group_id" not in job


@pytest.mark.asyncio
async def test_recovery_request_can_create_a_followup_provider_part():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    group = {
        "batch_group_id": "econ-recovery",
        "exam_id": "exam-1",
        "status": "provider_processing",
    }
    await db["exampen_provider_batches"].insert_one(group)
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-recovery",
        "status": "provider_processing",
    })
    client = AsyncMock()
    client.scope = {}
    client.upload_jsonl.return_value = {"id": "file-recovery"}
    client.wait_for_file_ready.return_value = {
        "id": "file-recovery",
        "purpose": "batch",
        "status": "processed",
    }
    client.create_batch.return_value = {
        "id": "batch-recovery",
        "status": "validating",
    }
    request_body = {"model": "gpt-5.1", "input": [], "store": False}
    entry = {
        "custom_id": "recovery-1",
        "job_id": "job-recovery",
        "submission_id": "submission-1",
        "grader_kind": "full_document",
        "call_index": 1,
        "stage": "recovery",
        "model": "gpt-5.1",
        "request_body": request_body,
        "jsonl_line": (
            '{"custom_id":"recovery-1","method":"POST","url":"/v1/responses",'
            '"body":{"model":"gpt-5.1","input":[],"store":false}}\n'
        ).encode(),
    }

    created = await _create_provider_parts(
        db,
        group=group,
        entries=[entry],
        stage="recovery",
        client=client,
    )

    assert created == 1
    part = await db[BATCH_PARTS_COLLECTION].find_one({"provider_batch_id": "batch-recovery"})
    assert part["stage"] == "recovery"


@pytest.mark.asyncio
async def test_terminal_provider_failure_is_preserved_on_job_part_and_group():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-failed",
        "status": "provider_processing",
        "job_ids": ["job-failed"],
    })
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-failed",
        "batch_group_id": "econ-failed",
        "provider_batch_id": "batch-failed",
        "status": "in_progress",
        "input_file_id": "file-missing",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_one({
        "custom_id": "copy-failed",
        "local_part_id": "part-failed",
        "batch_group_id": "econ-failed",
        "job_id": "job-failed",
        "import_status": "pending",
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-failed",
        "status": "provider_processing",
    })
    client = AsyncMock()
    client.scope = {"project": "proj-exampen"}
    client.retrieve_batch.return_value = {
        "id": "batch-failed",
        "status": "failed",
        "input_file_id": "file-missing",
        "errors": {
            "data": [{"message": "Cannot find file file-missing, or organization org-test does not have access to it"}],
        },
    }

    with patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=client):
        summary = await reconcile_economy_batches(db)

    part = await db[BATCH_PARTS_COLLECTION].find_one({"local_part_id": "part-failed"})
    item = await db[BATCH_ITEMS_COLLECTION].find_one({"custom_id": "copy-failed"})
    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-failed"})
    group = await db["exampen_provider_batches"].find_one({"batch_group_id": "econ-failed"})
    assert summary["items_failed"] == 1
    assert part["status"] == "imported"
    assert "Cannot find file file-missing" in part["last_error"]
    assert "Cannot find file file-missing" in item["last_error"]
    assert job["status"] == "batch_failed"
    assert "Cannot find file file-missing" in job["last_error"]
    assert group["status"] == "completed_with_errors"
    assert "Cannot find file file-missing" in group["last_error"]
    assert group["failure_code"] == "provider_batch_file_access"
    assert group["retryable"] is False


@pytest.mark.asyncio
async def test_reconcile_exposes_finalizing_progress_and_clears_stale_job_error():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-finalizing",
        "status": "provider_processing",
        "job_ids": ["job-finalizing"],
    })
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-finalizing",
        "batch_group_id": "econ-finalizing",
        "provider_batch_id": "batch-finalizing",
        "status": "in_progress",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_one({
        "custom_id": "copy-finalizing",
        "local_part_id": "part-finalizing",
        "batch_group_id": "econ-finalizing",
        "job_id": "job-finalizing",
        "import_status": "pending",
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-finalizing",
        "status": "provider_processing",
        "last_error": "obsolete error from an earlier attempt",
    })
    client = AsyncMock()
    client.scope = {"project": "proj-exampen"}
    client.retrieve_batch.return_value = {
        "id": "batch-finalizing",
        "status": "finalizing",
        "expires_at": 1,
        "finalizing_at": 2,
        "request_counts": {"total": 2, "completed": 1, "failed": 0},
    }

    with patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=client):
        summary = await reconcile_economy_batches(db)

    part = await db[BATCH_PARTS_COLLECTION].find_one({"local_part_id": "part-finalizing"})
    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-finalizing"})
    group = await db["exampen_provider_batches"].find_one({"batch_group_id": "econ-finalizing"})

    assert summary["parts_polled"] == 1
    assert part["status"] == "finalizing"
    assert job["status"] == "provider_finalizing"
    assert job["provider_batch_status"] == "finalizing"
    assert job["provider_request_counts"] == {"total": 2, "completed": 1, "failed": 0}
    assert job["provider_delayed"] is True
    assert job["progress"]["stage"] == "provider_finalizing"
    assert "last_error" not in job
    assert group["status"] == "provider_processing"


@pytest.mark.asyncio
async def test_economy_batch_response_recovers_legacy_provider_failure_safely():
    from mongomock_motor import AsyncMongoMockClient
    from api.v1.exam_orch_async import _economy_batch_to_response

    db = AsyncMongoMockClient()["skb_test"]
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-legacy-failed",
        "batch_group_id": "econ-legacy-failed",
        "provider_batch_id": "batch-legacy-failed",
        "status": "imported",
        "provider_state": {
            "status": "failed",
            "errors": {
                "data": [{"message": "Cannot find file file-legacy, or organization org-test does not have access to it"}],
            },
            "request_counts": {"completed": 0, "failed": 0, "total": 2},
        },
    })

    response = await _economy_batch_to_response(db, {
        "batch_group_id": "econ-legacy-failed",
        "exam_id": "exam-legacy-failed",
        "status": "completed_with_errors",
        "requested_count": 2,
        "failed_count": 2,
    })

    assert "Cannot find file file-legacy" in response.last_error
    assert response.failure_code == "provider_batch_file_access"
    assert response.retryable is False
    assert "Batch-enabled OpenAI project" in response.operator_action
    assert "provider_state" not in response.parts[0]


@pytest.mark.asyncio
async def test_economy_batch_response_reports_real_provider_finalizing_phase():
    from mongomock_motor import AsyncMongoMockClient
    from api.v1.exam_orch_async import _economy_batch_to_response

    db = AsyncMongoMockClient()["skb_test"]
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-finalizing-response",
        "batch_group_id": "econ-finalizing-response",
        "provider_batch_id": "batch-finalizing-response",
        "status": "finalizing",
        "updated_at": datetime(2026, 8, 31, 13, 0, tzinfo=timezone.utc),
        "provider_state": {
            "status": "finalizing",
            "expires_at": 1,
            "finalizing_at": 2,
            "request_counts": {"total": 2, "completed": 1, "failed": 0},
        },
    })

    response = await _economy_batch_to_response(db, {
        "batch_group_id": "econ-finalizing-response",
        "exam_id": "exam-finalizing-response",
        "status": "provider_processing",
        "requested_count": 2,
    })

    assert response.status == "provider_processing"
    assert response.provider_status == "finalizing"
    assert response.provider_total_count == 2
    assert response.provider_completed_count == 1
    assert response.provider_failed_count == 0
    assert response.provider_delayed is True
    assert response.provider_delay_seconds > 0
    assert response.provider_expires_at is not None
    assert response.provider_finalizing_at is not None
    assert response.provider_last_polled_at is not None


@pytest.mark.asyncio
async def test_cancelled_provider_batch_returns_unfinished_copy_to_economy_waiting():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-cancelled-provider",
        "status": "cancelling",
        "job_ids": ["job-cancelled-provider"],
    })
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-cancelled-provider",
        "batch_group_id": "econ-cancelled-provider",
        "provider_batch_id": "batch-cancelled-provider",
        "status": "cancelling",
        "input_file_id": "file-cancelled-provider",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_one({
        "custom_id": "copy-cancelled-provider",
        "local_part_id": "part-cancelled-provider",
        "batch_group_id": "econ-cancelled-provider",
        "job_id": "job-cancelled-provider",
        "import_status": "pending",
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-cancelled-provider",
        "status": "provider_processing",
        "provider_batch_group_id": "econ-cancelled-provider",
    })
    client = AsyncMock()
    client.scope = {}
    client.retrieve_batch.return_value = {
        "id": "batch-cancelled-provider",
        "status": "cancelled",
        "input_file_id": "file-cancelled-provider",
    }

    with patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=client):
        await reconcile_economy_batches(db)

    item = await db[BATCH_ITEMS_COLLECTION].find_one({"custom_id": "copy-cancelled-provider"})
    job = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "job-cancelled-provider"})
    group = await db["exampen_provider_batches"].find_one({"batch_group_id": "econ-cancelled-provider"})
    assert item["import_status"] == "superseded"
    assert job["status"] == "waiting_for_batch"
    assert "provider_batch_group_id" not in job
    assert group["status"] == "cancelled"
    assert group["failed_count"] == 0


@pytest.mark.asyncio
async def test_terminal_provider_part_is_claimed_imported_and_cleaned_once():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-import",
        "status": "provider_processing",
        "job_ids": ["job-import"],
    })
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-import",
        "batch_group_id": "econ-import",
        "provider_batch_id": "batch-import",
        "status": "completed",
        "input_file_id": "file-input",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_one({
        "custom_id": "copy-import",
        "local_part_id": "part-import",
        "batch_group_id": "econ-import",
        "job_id": "job-import",
        "import_status": "pending",
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_one({
        "job_id": "job-import",
        "status": "provider_processing",
    })
    client = AsyncMock()
    client.retrieve_batch.return_value = {
        "id": "batch-import",
        "status": "completed",
        "input_file_id": "file-input",
        "output_file_id": "file-output",
    }
    client.file_content.return_value = (
        b'{"custom_id":"copy-import","response":{"status_code":200,"body":{"id":"response-1"}}}\n'
    )

    with (
        patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=client),
        patch("services.exampen_openai_batch._import_item", new=AsyncMock(return_value=None)),
    ):
        summary = await reconcile_economy_batches(db)

    part = await db[BATCH_PARTS_COLLECTION].find_one({"local_part_id": "part-import"})
    item = await db[BATCH_ITEMS_COLLECTION].find_one({"custom_id": "copy-import"})
    group = await db["exampen_provider_batches"].find_one({"batch_group_id": "econ-import"})
    assert summary["items_imported"] == 1
    assert part["status"] == "imported"
    assert item["import_status"] == "completed"
    assert group["status"] == "completed"
    assert client.delete_file.await_count == 2


@pytest.mark.asyncio
async def test_expired_batch_preserves_completed_copy_and_fails_only_unfinished_copy():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-partial-expiry",
        "status": "provider_processing",
        "job_ids": ["job-completed-copy", "job-expired-copy"],
    })
    await db[BATCH_PARTS_COLLECTION].insert_one({
        "local_part_id": "part-partial-expiry",
        "batch_group_id": "econ-partial-expiry",
        "provider_batch_id": "batch-partial-expiry",
        "status": "in_progress",
        "input_file_id": "file-input",
    })
    await db[BATCH_ITEMS_COLLECTION].insert_many([
        {
            "custom_id": "copy-completed",
            "local_part_id": "part-partial-expiry",
            "batch_group_id": "econ-partial-expiry",
            "job_id": "job-completed-copy",
            "import_status": "pending",
        },
        {
            "custom_id": "copy-expired",
            "local_part_id": "part-partial-expiry",
            "batch_group_id": "econ-partial-expiry",
            "job_id": "job-expired-copy",
            "import_status": "pending",
        },
    ])
    await db[PROCESSING_JOBS_COLLECTION].insert_many([
        {"job_id": "job-completed-copy", "status": "provider_processing"},
        {"job_id": "job-expired-copy", "status": "provider_processing"},
    ])

    client = AsyncMock()
    client.scope = {}
    client.retrieve_batch.return_value = {
        "id": "batch-partial-expiry",
        "status": "expired",
        "input_file_id": "file-input",
        "output_file_id": "file-output",
        "request_counts": {"total": 2, "completed": 1, "failed": 0},
    }
    client.file_content.return_value = (
        b'{"custom_id":"copy-completed","response":{"status_code":200,'
        b'"body":{"id":"response-completed"}}}\n'
    )

    async def import_completed_only(tenant_db, *, item, provider_line, missing_result_error):
        if provider_line is None:
            raise RuntimeError(missing_result_error)
        await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
            {"job_id": item["job_id"]},
            {"$set": {"status": "completed"}},
        )
        return None

    with (
        patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=client),
        patch("services.exampen_openai_batch._import_item", side_effect=import_completed_only),
    ):
        summary = await reconcile_economy_batches(db)

    completed_item = await db[BATCH_ITEMS_COLLECTION].find_one(
        {"custom_id": "copy-completed"}
    )
    expired_item = await db[BATCH_ITEMS_COLLECTION].find_one(
        {"custom_id": "copy-expired"}
    )
    completed_job = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "job-completed-copy"}
    )
    expired_job = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "job-expired-copy"}
    )
    group = await db["exampen_provider_batches"].find_one(
        {"batch_group_id": "econ-partial-expiry"}
    )

    assert summary["items_imported"] == 1
    assert summary["items_failed"] == 1
    assert completed_item["import_status"] == "completed"
    assert completed_job["status"] == "completed"
    assert expired_item["import_status"] == "failed"
    assert expired_job["status"] == "batch_failed"
    assert expired_job["failure_code"] == "provider_batch_expired"
    assert expired_job["retryable"] is True
    assert group["status"] == "completed_with_errors"


@pytest.mark.asyncio
async def test_preparation_flushes_large_copy_requests_in_bounded_chunks(monkeypatch):
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    job_ids = [f"job-{index}" for index in range(3)]
    await db["exampen_provider_batches"].insert_one({
        "batch_group_id": "econ-chunks",
        "exam_id": "exam-chunks",
        "status": "queued",
        "job_ids": job_ids,
    })
    await db[PROCESSING_JOBS_COLLECTION].insert_many([
        {
            "job_id": job_id,
            "submission_id": f"submission-{index}",
            "status": "batch_queued",
            "created_at": index,
        }
        for index, job_id in enumerate(job_ids)
    ])
    monkeypatch.setenv("EXAMPEN_BATCH_PREPARE_FLUSH_BYTES", str(8 * 1024 * 1024))
    deferred = {
        "request_body": {"model": "gpt-5.1", "input": [], "store": False},
        "call_index": 0,
        "model": "gpt-5.1",
        "grader_kind": "full_document",
    }
    create_parts = AsyncMock(return_value=1)

    with (
        patch("services.exampen_openai_batch._run_grader", new=AsyncMock(return_value=(None, deferred))),
        patch("services.exampen_openai_batch._jsonl_line", return_value=b"x" * (5 * 1024 * 1024)),
        patch("services.exampen_openai_batch.OpenAIBatchClient", return_value=AsyncMock()),
        patch("services.exampen_openai_batch._create_provider_parts", new=create_parts),
    ):
        result = await prepare_economy_batch_group(
            db,
            batch_group_id="econ-chunks",
        )

    assert result["status"] == "provider_processing"
    assert create_parts.await_count == 2
    assert all(
        len(call.kwargs["entries"]) <= 2 for call in create_parts.await_args_list
    )
