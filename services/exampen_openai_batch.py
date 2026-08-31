"""Durable, discounted OpenAI Batch orchestration for ExamPen grading.

One provider request always represents one student copy. Requests are grouped
only for transport/pricing, never for semantic grading. The existing objective
and whole-document graders remain the scoring authorities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
from pymongo import ReturnDocument

from api.v1._exampen_imports import load_exampen

logger = logging.getLogger(__name__)

BATCH_GROUPS_COLLECTION = "exampen_provider_batches"
BATCH_PARTS_COLLECTION = "exampen_provider_batch_parts"
BATCH_ITEMS_COLLECTION = "exampen_provider_batch_items"
PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
ECONOMY_WAITING_JOB_STATUS = "waiting_for_batch"

ACTIVE_GROUP_STATUSES = {"queued", "preparing", "provider_processing", "importing", "cancelling"}
ACTIVE_PROVIDER_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
TERMINAL_PROVIDER_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_BATCH_JOB_STATUSES = {
    "batch_queued",
    "preparing_batch",
    "provider_processing",
    "provider_finalizing",
    "importing_batch",
}
PART_IMPORTING_STATUS = "importing_results"
DEFAULT_JSONL_LIMIT = 180 * 1024 * 1024
MAX_BATCH_REQUESTS = 50_000


class StaleBatchGenerationError(RuntimeError):
    """Raised when delayed provider output belongs to a superseded grading run."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _short_error(value: Any, limit: int = 600) -> str:
    if isinstance(value, Mapping):
        try:
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            pass
    return str(value or "Unknown error").replace("\n", " ").strip()[:limit]


def _batch_max_requests_per_part() -> int:
    """Bound the failure domain of one provider Batch.

    One copy per Batch is the safe default: a slow or expired copy cannot hold
    every other student's completed output.  Operators may deliberately widen
    the shard after validating their provider limits and latency objectives.
    """

    try:
        configured = int(os.getenv("EXAMPEN_BATCH_MAX_REQUESTS_PER_PART", "1"))
    except (TypeError, ValueError):
        configured = 1
    return max(1, min(MAX_BATCH_REQUESTS, configured))


def _entry_stage(entry: Mapping[str, Any]) -> Dict[str, Any]:
    call_index = max(1, int(entry.get("call_index") or 1))
    grader_kind = str(entry.get("grader_kind") or "")
    stage_count = 2 if grader_kind == "full_document" else 1
    stage_number = min(call_index, stage_count)
    phase = (
        "mapping"
        if grader_kind == "full_document" and stage_number == 1
        else "grading"
    )
    return {
        "provider_phase": phase,
        "stage_number": stage_number,
        "stage_count": stage_count,
    }


def _entry_generation(entry: Mapping[str, Any]) -> int:
    # Items created before grading generations were introduced belong to
    # generation zero. Never interpret a missing legacy field as "match any
    # generation", because delayed legacy output could otherwise claim a newer
    # retry after deployment.
    return int(entry.get("job_generation") or 0)


def _provider_request_metadata(payload: Mapping[str, Any]) -> Dict[str, str]:
    metadata = payload.get("_request_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    return {
        key: str(metadata.get(key) or "").strip()
        for key in ("request_id", "organization", "project")
        if str(metadata.get(key) or "").strip()
    }


def _client_provider_scope(client: Any) -> Dict[str, str]:
    scope = getattr(client, "scope", {})
    if not isinstance(scope, Mapping):
        return {}
    return {
        key: str(scope.get(key) or "").strip()
        for key in ("organization", "project")
        if str(scope.get(key) or "").strip()
    }


def _safe_file_state(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("id", "purpose", "status", "bytes", "created_at", "expires_at")
        if payload.get(key) is not None
    } | ({"request": _provider_request_metadata(payload)} if _provider_request_metadata(payload) else {})


def provider_batch_failure(provider_state: Mapping[str, Any]) -> Optional[str]:
    """Return the provider's terminal Batch reason without losing its message."""

    status_value = str(provider_state.get("status") or "").strip().lower()
    messages: List[str] = []
    errors = provider_state.get("errors")
    error_data = errors.get("data") if isinstance(errors, Mapping) else None
    if isinstance(error_data, list):
        for item in error_data:
            if not isinstance(item, Mapping):
                continue
            message = str(item.get("message") or item.get("code") or "").strip()
            if message and message not in messages:
                messages.append(message)
    if messages:
        return _short_error(
            f"OpenAI {status_value or 'rejected'} the economy check: "
            + "; ".join(messages[:5])
        )
    if status_value == "expired":
        return "OpenAI economy checking expired before this copy completed"
    if status_value == "cancelled":
        return "Economy checking was cancelled before this copy completed"
    if status_value == "failed":
        return "OpenAI economy checking failed before returning a result"
    return None


def describe_provider_batch_progress(
    provider_state: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the stable active-provider progress contract used by jobs and APIs.

    OpenAI can remain in ``finalizing`` after the nominal completion window.
    That is not a completed result and must not be converted into marks or a
    failure locally. It is materially different from model work still running,
    so expose it explicitly while reconciliation continues.
    """

    observed_at = now or _now()
    status_value = str(provider_state.get("status") or "").strip().lower()
    counts = provider_state.get("request_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    total = max(0, int(counts.get("total") or 0))
    completed = max(0, int(counts.get("completed") or 0))
    failed = max(0, int(counts.get("failed") or 0))

    expires_at = provider_state.get("expires_at")
    try:
        expires_epoch = float(expires_at) if expires_at is not None else None
    except (TypeError, ValueError):
        expires_epoch = None
    delay_seconds = (
        max(0, int(observed_at.timestamp() - expires_epoch))
        if expires_epoch is not None and observed_at.timestamp() >= expires_epoch
        else 0
    )
    delayed = bool(
        delay_seconds
        and status_value in {"validating", "in_progress", "finalizing"}
    )

    if status_value == "validating":
        stage = "provider_validating"
        message = "OpenAI is validating the economy Batch"
    elif status_value == "finalizing":
        stage = "provider_finalizing"
        message = (
            "OpenAI is delayed while finalizing the economy Batch; Stoody is still polling"
            if delayed
            else "OpenAI is finalizing the economy Batch results"
        )
    elif status_value == "cancelling":
        stage = "provider_cancelling"
        message = "OpenAI is cancelling the economy Batch"
    else:
        stage = "provider_processing"
        message = "OpenAI economy checking is running"
    if total:
        message = f"{message} ({completed}/{total} requests complete)"

    return {
        "status": status_value or "in_progress",
        "total": total,
        "completed": completed,
        "failed": failed,
        "expires_at": expires_at,
        "finalizing_at": provider_state.get("finalizing_at"),
        "delayed": delayed,
        "delay_seconds": delay_seconds,
        "job_status": (
            "provider_finalizing"
            if status_value == "finalizing"
            else "provider_processing"
        ),
        "stage": stage,
        "message": message,
    }


def classify_economy_batch_failure(error: Any) -> Dict[str, Any]:
    """Turn provider text into a stable retry/operations contract.

    A Batch input file that is successfully uploaded and retrieved but rejected
    by the asynchronous Batch validator is not repaired by resubmitting the same
    copies. Treat that provider/project fault differently from transient Batch
    failures so the UI cannot invite an endless discounted retry loop.
    """

    message = _short_error(error)
    normalized = message.casefold()
    file_access_failure = (
        "cannot find file" in normalized
        and "does not have access to it" in normalized
    )
    if file_access_failure:
        return {
            "failure_code": "provider_batch_file_access",
            "retryable": False,
            "operator_action": (
                "Configure a Batch-enabled OpenAI project and project API key, "
                "verify it with a no-cost Batch preflight, then restart the backend."
            ),
        }
    if any(
        marker in normalized
        for marker in (
            "http 401",
            "http 403",
            "invalid api key",
            "incorrect api key",
            "organization must be an organization id",
            "project must be a project id",
        )
    ):
        return {
            "failure_code": "provider_authentication",
            "retryable": False,
            "operator_action": (
                "Correct the OpenAI project credentials and restart the backend "
                "before starting Economy checking again."
            ),
        }
    if any(marker in normalized for marker in ("server_is_overloaded", "server overloaded", "overloaded")):
        return {
            "failure_code": "provider_overloaded",
            "retryable": True,
            "operator_action": "Retry this copy in Economy, or use Standard checking if it is urgent.",
        }
    if any(marker in normalized for marker in ("rate_limit", "rate limit", "http 429")):
        return {
            "failure_code": "provider_rate_limited",
            "retryable": True,
            "operator_action": "Wait for provider capacity, then retry only this copy.",
        }
    if any(marker in normalized for marker in ("expired", "timed out", "timeout")):
        return {
            "failure_code": "provider_batch_expired",
            "retryable": True,
            "operator_action": "Retry only this copy; completed copies remain unchanged.",
        }
    if "cancelled" in normalized or "canceled" in normalized:
        return {
            "failure_code": "provider_batch_cancelled",
            "retryable": True,
            "operator_action": "Start Economy checking again for the unfinished copy.",
        }
    if any(
        marker in normalized
        for marker in (
            "invalid_request_error",
            "invalid request",
            "unsupported parameter",
            "model_not_found",
            "content_policy_violation",
        )
    ):
        return {
            "failure_code": "provider_invalid_request",
            "retryable": False,
            "operator_action": "Correct the grading request or model configuration before retrying.",
        }
    return {
        "failure_code": "provider_batch_failed",
        "retryable": True,
        "operator_action": None,
    }


def _opaque_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _jsonl_line(custom_id: str, body: Mapping[str, Any]) -> bytes:
    payload = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": dict(body),
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def partition_batch_requests(
    entries: Sequence[Dict[str, Any]],
    *,
    max_bytes: int = DEFAULT_JSONL_LIMIT,
    max_requests: int = MAX_BATCH_REQUESTS,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Partition exact JSONL bytes by model and provider upload limits."""

    bounded_bytes = max(1, int(max_bytes))
    bounded_requests = max(1, min(MAX_BATCH_REQUESTS, int(max_requests)))
    partitions: List[List[Dict[str, Any]]] = []
    oversized: List[Dict[str, Any]] = []
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        by_model.setdefault(str(entry.get("model") or ""), []).append(entry)
    for model_entries in by_model.values():
        current: List[Dict[str, Any]] = []
        current_bytes = 0
        for entry in model_entries:
            line = entry.get("jsonl_line")
            if not isinstance(line, (bytes, bytearray)):
                line = _jsonl_line(str(entry["custom_id"]), entry["request_body"])
                entry["jsonl_line"] = line
            line_size = len(line)
            if line_size > bounded_bytes:
                oversized.append(entry)
                continue
            if current and (
                len(current) >= bounded_requests
                or current_bytes + line_size > bounded_bytes
            ):
                partitions.append(current)
                current = []
                current_bytes = 0
            current.append(entry)
            current_bytes += line_size
        if current:
            partitions.append(current)
    return partitions, oversized


def parse_batch_jsonl(content: bytes | str) -> Dict[str, Dict[str, Any]]:
    """Parse provider output/error JSONL by `custom_id`, never by line order."""

    text = content.decode("utf-8") if isinstance(content, bytes) else str(content or "")
    parsed: Dict[str, Dict[str, Any]] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed OpenAI Batch output line")
            continue
        custom_id = str(item.get("custom_id") or "").strip()
        if custom_id:
            parsed[custom_id] = item
    return parsed


class OpenAIBatchClient:
    """Small HTTP adapter for Files + Batches; no SDK version dependency.

    Files and Batches must be created in the same OpenAI organization/project
    scope.  Project API keys provide that scope themselves.  Legacy or
    multi-organization keys may use explicitly configured organization/project
    IDs, but response metadata must never be promoted into request headers:
    ``openai-organization`` is diagnostic metadata and can be an organization
    label rather than the ``org-...`` ID required by ``OpenAI-Organization``.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = str(api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self._organization = str(
            organization
            or os.getenv("OPENAI_ORGANIZATION")
            or os.getenv("OPENAI_ORG_ID")
            or ""
        ).strip()
        self._project = str(
            project
            or os.getenv("OPENAI_PROJECT")
            or os.getenv("OPENAI_PROJECT_ID")
            or ""
        ).strip()
        if self._organization and not self._organization.startswith("org-"):
            raise RuntimeError(
                "OPENAI_ORGANIZATION must be an organization ID starting with 'org-'"
            )
        if self._project and not self._project.startswith("proj_"):
            raise RuntimeError(
                "OPENAI_PROJECT must be a project ID starting with 'proj_'"
            )
        self._transport = transport

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        if self._project:
            headers["OpenAI-Project"] = self._project
        return headers

    @property
    def scope(self) -> Dict[str, str]:
        """Return safe provider scope metadata (never the API key)."""

        return {
            key: value
            for key, value in {
                "organization": self._organization,
                "project": self._project,
            }.items()
            if value
        }

    def _capture_response_metadata(self, response: httpx.Response) -> Dict[str, str]:
        """Capture provider diagnostics without changing request authority."""

        metadata = {
            key: str(value).strip()
            for key, value in {
                "request_id": response.headers.get("x-request-id"),
                "organization": response.headers.get("openai-organization"),
                "project": response.headers.get("openai-project"),
            }.items()
            if str(value or "").strip()
        }
        return metadata

    def _decode_json_response(
        self,
        response: httpx.Response,
        *,
        fallback_error: str,
    ) -> Dict[str, Any]:
        metadata = self._capture_response_metadata(response)
        try:
            body = dict(response.json() or {})
        except Exception:
            body = {}
        if response.status_code >= 400:
            message = fallback_error
            error = body.get("error") if isinstance(body.get("error"), Mapping) else {}
            message = str(error.get("message") or message)
            request_suffix = (
                f"; OpenAI request id {metadata['request_id']}"
                if metadata.get("request_id")
                else ""
            )
            raise RuntimeError(
                f"{message} (HTTP {response.status_code}{request_suffix})"
            )
        if metadata:
            body["_request_metadata"] = metadata
        return body

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0), transport=self._transport
        ) as client:
            response = await client.request(
                method,
                f"{self._base_url}{path}",
                headers={**self._headers, "Content-Type": "application/json"},
                json=payload,
            )
        return self._decode_json_response(
            response,
            fallback_error="OpenAI Batch request failed",
        )

    async def upload_jsonl(self, filename: str, content: bytes) -> Dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(180.0), transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base_url}/files",
                headers=self._headers,
                data={"purpose": "batch"},
                files={"file": (filename, content, "application/jsonl")},
            )
        return self._decode_json_response(
            response,
            fallback_error="OpenAI Batch input upload failed",
        )

    async def retrieve_file(self, file_id: str) -> Dict[str, Any]:
        return await self._json_request("GET", f"/files/{file_id}")

    async def wait_for_file_ready(self, file_id: str) -> Dict[str, Any]:
        """Verify a Batch input file is visible and processed in this scope.

        OpenAI file creation and provider-side processing are asynchronous.  A
        Batch is never created from an id that has not first been retrieved in
        the exact organization/project scope used by this client.
        """

        try:
            timeout_seconds = float(
                os.getenv("EXAMPEN_BATCH_FILE_READY_TIMEOUT_SECONDS", "120")
            )
        except (TypeError, ValueError):
            timeout_seconds = 120.0
        try:
            poll_seconds = float(
                os.getenv("EXAMPEN_BATCH_FILE_READY_POLL_SECONDS", "1")
            )
        except (TypeError, ValueError):
            poll_seconds = 1.0
        timeout_seconds = max(5.0, min(600.0, timeout_seconds))
        poll_seconds = max(0.05, min(5.0, poll_seconds))
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_status = "unknown"

        while True:
            file_state = await self.retrieve_file(file_id)
            returned_id = str(file_state.get("id") or "")
            if returned_id and returned_id != file_id:
                raise RuntimeError("OpenAI returned the wrong Batch input file")
            purpose = str(file_state.get("purpose") or "").strip().lower()
            if purpose and purpose != "batch":
                raise RuntimeError(
                    f"OpenAI file {file_id} has purpose {purpose!r}, not 'batch'"
                )
            last_status = str(file_state.get("status") or "").strip().lower()
            if last_status == "error":
                details = file_state.get("status_details") or "provider processing failed"
                raise RuntimeError(
                    f"OpenAI could not process Batch input file {file_id}: "
                    f"{_short_error(details)}"
                )
            # File.status is deprecated by OpenAI.  A successful scoped GET
            # with no status is therefore the modern readiness signal.
            if not last_status or last_status == "processed":
                return file_state
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"OpenAI Batch input file {file_id} was still {last_status!r} "
                    f"after {timeout_seconds:.0f} seconds"
                )
            await asyncio.sleep(poll_seconds)

    async def create_batch(
        self,
        *,
        input_file_id: str,
        metadata: Dict[str, str],
        endpoint: str = "/v1/responses",
    ) -> Dict[str, Any]:
        supported_endpoints = {"/v1/responses", "/v1/moderations"}
        if endpoint not in supported_endpoints:
            raise ValueError(f"Unsupported Stoody Batch endpoint: {endpoint}")
        try:
            output_seconds = int(os.getenv("EXAMPEN_BATCH_OUTPUT_RETENTION_SECONDS", "86400"))
        except (TypeError, ValueError):
            output_seconds = 86400
        output_seconds = max(3600, min(2_592_000, output_seconds))
        return await self._json_request(
            "POST",
            "/batches",
            payload={
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": "24h",
                "output_expires_after": {
                    "anchor": "created_at",
                    "seconds": output_seconds,
                },
                "metadata": metadata,
            },
        )

    async def retrieve_batch(self, provider_batch_id: str) -> Dict[str, Any]:
        return await self._json_request("GET", f"/batches/{provider_batch_id}")

    async def list_batches(self, *, after: Optional[str] = None) -> Dict[str, Any]:
        path = "/batches?limit=100"
        if after:
            path += f"&after={after}"
        return await self._json_request("GET", path)

    async def cancel_batch(self, provider_batch_id: str) -> Dict[str, Any]:
        return await self._json_request("POST", f"/batches/{provider_batch_id}/cancel")

    async def file_content(self, file_id: str) -> bytes:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(180.0), transport=self._transport
        ) as client:
            response = await client.get(
                f"{self._base_url}/files/{file_id}/content",
                headers=self._headers,
            )
        if response.status_code >= 400:
            self._decode_json_response(
                response,
                fallback_error="OpenAI Batch result download failed",
            )
        self._capture_response_metadata(response)
        return response.content

    async def delete_file(self, file_id: str) -> None:
        if not file_id:
            return
        try:
            await self._json_request("DELETE", f"/files/{file_id}")
        except Exception:
            logger.warning("Could not delete OpenAI Batch file %s", file_id, exc_info=True)


async def ensure_batch_indexes(tenant_db: Any) -> None:
    # Legacy jobs predate the run-generation fence.  Backfill once so delayed
    # provider output can be compared atomically during rolling deployment.
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
        {"grading_generation": {"$exists": False}},
        {"$set": {"grading_generation": 0}},
    )
    await tenant_db[BATCH_GROUPS_COLLECTION].create_index(
        "batch_group_id", unique=True, name="uniq_exampen_batch_group"
    )
    await tenant_db[BATCH_GROUPS_COLLECTION].create_index(
        [("exam_id", 1), ("status", 1)], name="idx_exampen_batch_exam_status"
    )
    await tenant_db[BATCH_PARTS_COLLECTION].create_index(
        "local_part_id", unique=True, name="uniq_exampen_batch_part"
    )
    await tenant_db[BATCH_PARTS_COLLECTION].create_index(
        "provider_batch_id", unique=True, sparse=True, name="uniq_provider_batch"
    )
    await tenant_db[BATCH_ITEMS_COLLECTION].create_index(
        "custom_id", unique=True, name="uniq_exampen_batch_custom_id"
    )
    await tenant_db[BATCH_ITEMS_COLLECTION].create_index(
        "parent_custom_id",
        unique=True,
        sparse=True,
        name="uniq_exampen_batch_recovery_parent",
    )
    await tenant_db[BATCH_ITEMS_COLLECTION].create_index(
        [("batch_group_id", 1), ("import_status", 1)],
        name="idx_exampen_batch_item_import",
    )
    await tenant_db[BATCH_ITEMS_COLLECTION].create_index(
        [("job_id", 1), ("job_generation", 1), ("import_status", 1)],
        name="idx_exampen_batch_item_generation",
    )


async def create_economy_batch_group(
    tenant_db: Any,
    *,
    exam_id: str,
    db_name: str,
    requested_by: str,
    submission_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Reserve eligible waiting copies without reprocessing completed copies."""

    await ensure_batch_indexes(tenant_db)
    active = await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
        {"exam_id": exam_id, "status": {"$in": sorted(ACTIVE_GROUP_STATUSES)}}
    )
    if active:
        return active
    requested = {str(value).strip() for value in (submission_ids or []) if str(value).strip()}
    job_filter: Dict[str, Any] = {
        "exam_id": exam_id,
        "status": {"$in": ["waiting_for_batch", "batch_failed"]},
    }
    if requested:
        job_filter["submission_id"] = {"$in": sorted(requested)}
    jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(job_filter).sort(
        "created_at", 1
    ).to_list(length=5000)
    if not jobs:
        raise ValueError("No submitted answer copies are waiting for economy checking")
    if requested:
        found = {str(job.get("submission_id") or "") for job in jobs}
        if requested - found:
            raise ValueError(
                "One or more selected copies are unavailable, already checked, or published"
            )
    group_id = _opaque_id("econ")
    now = _now()
    group = {
        "batch_group_id": group_id,
        "exam_id": exam_id,
        "db_name": db_name,
        "status": "queued",
        "requested_by": requested_by,
        "requested_at": now,
        "job_ids": [str(job.get("job_id") or "") for job in jobs],
        "requested_count": len(jobs),
        "provider_request_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    await tenant_db[BATCH_GROUPS_COLLECTION].insert_one(group)
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
        {"job_id": {"$in": group["job_ids"]}, "status": {"$in": ["waiting_for_batch", "batch_failed"]}},
        {
            "$set": {
                "status": "batch_queued",
                "processing_mode": "economy",
                "provider_batch_group_id": group_id,
                "progress": {
                    "stage": "batch_queued",
                    "message": "Queued for economy checking",
                },
                "updated_at": now,
            },
            "$inc": {"grading_generation": 1},
            "$unset": {
                "last_error": "",
                "failure_code": "",
                "retryable": "",
                "operator_action": "",
                "finished_at": "",
                "provider_batch_id": "",
                "provider_batch_status": "",
                "provider_phase": "",
                "provider_expires_at": "",
                "provider_request_counts": "",
                "provider_delayed": "",
                "stage_number": "",
                "stage_count": "",
            },
        },
    )
    return group


async def _run_grader(
    tenant_db: Any,
    *,
    submission_id: str,
    grader_kind: Optional[str] = None,
    response_bodies: Optional[Sequence[Mapping[str, Any]]] = None,
    recorded_call_indexes: Optional[Sequence[int]] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Run existing graders until complete or until the next provider call."""

    llm_gate_module = load_exampen("llm_gate")
    pcr_services = load_exampen("pcr.services")
    gate = llm_gate_module.LLMGate(tenant_db)
    await gate.initialize()
    replay_gate = llm_gate_module.BatchReplayGate(
        gate,
        response_bodies=response_bodies,
        recorded_call_indexes=recorded_call_indexes,
    )

    if grader_kind in {None, "objective"}:
        objective = pcr_services.ObjectiveAnswerSheetGradingService(tenant_db, replay_gate)
        try:
            result = await objective.grade_submission(submission_id)
        except llm_gate_module.DeferredBatchCall as exc:
            return None, {
                "request_body": exc.request_body,
                "call_index": exc.call_index,
                "model": exc.model_id,
                "grader_kind": "objective",
            }
        if result.handled or grader_kind == "objective":
            return result, None

    full_document = pcr_services.FullDocumentGradingService(tenant_db, replay_gate)
    result = await full_document.grade_submission(submission_id)
    if result.status == "waiting_for_batch" and result.deferred_provider_request:
        return None, {
            "request_body": result.deferred_provider_request,
            "call_index": int(result.deferred_call_index or 0),
            "model": str(result.deferred_provider_request.get("model") or ""),
            "grader_kind": "full_document",
            "run_id": result.run_id,
        }
    return result, None


async def _commit_grading_result(
    tenant_db: Any,
    *,
    job: Mapping[str, Any],
    result: Any,
    expected_generation: Optional[int] = None,
) -> None:
    """Commit the same public job projection used by the immediate worker."""

    now = _now()
    processing_path = str(
        getattr(result, "processing_path", "") or "full_document_visual"
    )
    final_status = str(getattr(result, "status", "") or "blocked_for_review")
    job_filter: Dict[str, Any] = {
        "job_id": job.get("job_id"),
        "status": {"$in": sorted(ACTIVE_BATCH_JOB_STATUSES)},
    }
    if expected_generation is not None:
        job_filter["grading_generation"] = expected_generation
    committed = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        job_filter,
        {
            "$set": {
                "status": final_status,
                "processing_path": processing_path,
                "document_grading_run_id": getattr(result, "run_id", None),
                "segmentation": {
                    "path": processing_path,
                    "page_count": int(getattr(result, "page_count", 0) or 0),
                    "response_count": int(getattr(result, "response_count", 0) or 0),
                    "blocked_count": int(getattr(result, "blocked_count", 0) or 0),
                    "warning_count": int(getattr(result, "warning_count", 0) or 0),
                },
                "evaluation": {
                    "path": processing_path,
                    "evaluated_count": int(getattr(result, "evaluated_count", 0) or 0),
                    "blocked_count": int(getattr(result, "blocked_count", 0) or 0),
                    "error_count": len(getattr(result, "errors", []) or []),
                    "remaining_ready": 0,
                    "scored_questions": int(getattr(result, "evaluated_count", 0) or 0),
                    "missing_question_count": int(getattr(result, "blocked_count", 0) or 0),
                },
                "review": {
                    "state": str(getattr(result, "review_state", "not_applicable")),
                    "document_review_required": bool(
                        getattr(result, "document_review_required", False)
                    ),
                    "reasons": list(getattr(result, "review_reasons", []) or [])[:20],
                },
                "last_error": "; ".join(list(getattr(result, "errors", []) or [])[:10]) or None,
                "finished_at": now,
                "updated_at": now,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "failure_code": "",
                "retryable": "",
                "operator_action": "",
            },
        },
    )
    if committed.matched_count != 1:
        raise StaleBatchGenerationError(
            "Provider output belongs to a superseded grading run"
        )
    try:
        from services.exampen_workflow import _maybe_mark_exam_ready_for_review

        await _maybe_mark_exam_ready_for_review(
            tenant_db, str(job.get("exam_id") or "")
        )
    except Exception:
        logger.warning("Could not refresh exam readiness after Batch import", exc_info=True)


async def _mark_job_failed(
    tenant_db: Any,
    job_id: str,
    error: Any,
    *,
    expected_generation: Optional[int] = None,
) -> None:
    failure = classify_economy_batch_failure(error)
    job_filter: Dict[str, Any] = {
        "job_id": job_id,
        "status": {"$nin": ["completed", "blocked_for_review"]},
    }
    if expected_generation is not None:
        job_filter["grading_generation"] = expected_generation
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        job_filter,
        {
            "$set": {
                "status": "batch_failed",
                "last_error": _short_error(error),
                **failure,
                "progress": {
                    "stage": "batch_failed",
                    "message": "Economy checking stopped for this copy; no marks were changed",
                },
                "finished_at": _now(),
                "updated_at": _now(),
            }
        },
    )


async def _authorize_group_token_reservation(
    tenant_db: Any,
    entries: Sequence[Dict[str, Any]],
) -> int:
    """Reserve a conservative token ceiling before purchasing delayed work."""

    provider_module = load_exampen("llm_gate.provider")
    new_reservation = 0
    for entry in entries:
        body = entry.get("request_body") if isinstance(entry.get("request_body"), Mapping) else {}
        responses_input = body.get("input") if isinstance(body.get("input"), list) else []
        input_tokens = int(
            provider_module.estimate_tokens_for_messages(
                "",
                responses_input=responses_input,
            )
        )
        output_tokens = max(0, int(body.get("max_output_tokens") or 0))
        reserved_tokens = input_tokens + output_tokens
        entry["reserved_tokens"] = reserved_tokens
        new_reservation += reserved_tokens

    existing_items = await tenant_db[BATCH_ITEMS_COLLECTION].find(
        {
            "import_status": {
                "$in": ["pending", "retry", "importing", "waiting_recovery"]
            },
            "reserved_tokens": {"$gt": 0},
        },
        {"reserved_tokens": 1},
    ).to_list(length=200_000)
    existing_reservation = sum(
        max(0, int(item.get("reserved_tokens") or 0)) for item in existing_items
    )
    llm_gate_module = load_exampen("llm_gate")
    gate = llm_gate_module.LLMGate(tenant_db)
    await gate.initialize()
    await gate.check_batch_reservation(existing_reservation + new_reservation)
    return new_reservation


async def _create_provider_parts(
    tenant_db: Any,
    *,
    group: Mapping[str, Any],
    entries: Sequence[Dict[str, Any]],
    stage: str,
    client: OpenAIBatchClient,
) -> int:
    async def preparation_is_current() -> bool:
        lease_token = str(group.get("preparation_lease_token") or "")
        current = await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
            {"batch_group_id": group["batch_group_id"]},
            {"status": 1, "preparation_lease_token": 1},
        )
        if stage == "recovery":
            return bool(
                current
                and str(current.get("status") or "")
                in {"provider_processing", "importing"}
            )
        return bool(
            current
            and str(current.get("status") or "") == "preparing"
            and lease_token
            and str(current.get("preparation_lease_token") or "") == lease_token
        )

    pending_entries = list(entries)
    if stage == "recovery":
        deduplicated: List[Dict[str, Any]] = []
        for entry in pending_entries:
            parent_custom_id = str(entry.get("parent_custom_id") or "")
            existing = (
                await tenant_db[BATCH_ITEMS_COLLECTION].find_one(
                    {"parent_custom_id": parent_custom_id}, {"custom_id": 1}
                )
                if parent_custom_id
                else None
            )
            if existing:
                continue
            deduplicated.append(entry)
        pending_entries = deduplicated
    if not pending_entries:
        return 0
    await _authorize_group_token_reservation(tenant_db, pending_entries)
    try:
        max_bytes = int(os.getenv("EXAMPEN_BATCH_JSONL_MAX_BYTES", str(DEFAULT_JSONL_LIMIT)))
    except (TypeError, ValueError):
        max_bytes = DEFAULT_JSONL_LIMIT
    max_bytes = max(1, min(199 * 1024 * 1024, max_bytes))
    partitions, oversized = partition_batch_requests(
        pending_entries,
        max_bytes=max_bytes,
        max_requests=_batch_max_requests_per_part(),
    )
    for entry in oversized:
        await _mark_job_failed(
            tenant_db,
            str(entry.get("job_id") or ""),
            "One copy is too large for the 200 MB OpenAI Batch input limit",
            expected_generation=_entry_generation(entry),
        )
    created = 0
    for partition_index, partition in enumerate(partitions):
        if not await preparation_is_current():
            break
        local_part_id = _opaque_id("part")
        now = _now()
        content = b"".join(bytes(entry["jsonl_line"]) for entry in partition)
        part_doc = {
            "local_part_id": local_part_id,
            "batch_group_id": group["batch_group_id"],
            "exam_id": group["exam_id"],
            "stage": stage,
            "model": str(partition[0].get("model") or ""),
            "status": "uploading",
            "item_count": len(partition),
            "input_bytes": len(content),
            "created_at": now,
            "updated_at": now,
        }
        await tenant_db[BATCH_PARTS_COLLECTION].insert_one(part_doc)
        item_docs = []
        for entry in partition:
            stage_projection = _entry_stage(entry)
            item_doc = {
                    "custom_id": entry["custom_id"],
                    "batch_group_id": group["batch_group_id"],
                    "local_part_id": local_part_id,
                    "exam_id": group["exam_id"],
                    "job_id": entry["job_id"],
                    "submission_id": entry["submission_id"],
                    "grader_kind": entry["grader_kind"],
                    "call_index": int(entry.get("call_index") or 0),
                    "job_generation": _entry_generation(entry),
                    **stage_projection,
                    "stage": stage,
                    "model": entry["model"],
                    "prior_response_bodies": list(entry.get("prior_response_bodies") or []),
                    "recorded_call_indexes": list(entry.get("recorded_call_indexes") or []),
                    "reserved_tokens": max(0, int(entry.get("reserved_tokens") or 0)),
                    "import_status": "pending",
                    "created_at": now,
                    "updated_at": now,
                }
            if entry.get("parent_custom_id"):
                item_doc["parent_custom_id"] = str(entry["parent_custom_id"])
            item_docs.append(item_doc)
        if item_docs:
            await tenant_db[BATCH_ITEMS_COLLECTION].insert_many(item_docs, ordered=True)
        input_file_id = ""
        try:
            uploaded = await client.upload_jsonl(f"{local_part_id}.jsonl", content)
            input_file_id = str(uploaded.get("id") or "")
            if not input_file_id:
                raise RuntimeError("OpenAI did not return a Batch input file id")
            await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {"local_part_id": local_part_id},
                {
                    "$set": {
                        "input_file_id": input_file_id,
                        "input_file_state": _safe_file_state(uploaded),
                        "provider_scope": _client_provider_scope(client),
                        "status": "awaiting_input_file",
                        "updated_at": _now(),
                    }
                },
            )
            ready_file = await client.wait_for_file_ready(input_file_id)
            await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {"local_part_id": local_part_id, "status": "awaiting_input_file"},
                {
                    "$set": {
                        "input_file_state": _safe_file_state(ready_file),
                        "provider_scope": _client_provider_scope(client),
                        "status": "creating",
                        "updated_at": _now(),
                    }
                },
            )
            if not await preparation_is_current():
                await client.delete_file(input_file_id)
                await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                    {"local_part_id": local_part_id},
                    {
                        "$set": {
                            "status": "imported",
                            "cancelled_before_provider_creation": True,
                            "updated_at": _now(),
                        }
                    },
                )
                await tenant_db[BATCH_ITEMS_COLLECTION].update_many(
                    {"local_part_id": local_part_id, "import_status": "pending"},
                    {"$set": {"import_status": "superseded", "updated_at": _now()}},
                )
                continue
            provider_batch = await client.create_batch(
                input_file_id=input_file_id,
                metadata={
                    "stoody_batch_group": str(group["batch_group_id"])[:64],
                    "stoody_part": local_part_id[:64],
                    "stage": stage[:64],
                },
            )
            provider_batch_id = str(provider_batch.get("id") or "")
            if not provider_batch_id:
                raise RuntimeError("OpenAI did not return a Batch id")
            if not await preparation_is_current():
                # Cancellation may win while the provider is accepting the
                # request.  Persist the provider id before requesting cancel so
                # reconciliation can never lose or duplicate paid work.
                cancelled_state = await client.cancel_batch(provider_batch_id)
                await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                    {"local_part_id": local_part_id},
                    {
                        "$set": {
                            "provider_batch_id": provider_batch_id,
                            "status": str(cancelled_state.get("status") or "cancelling"),
                            "provider_state": cancelled_state,
                            "provider_scope": _client_provider_scope(client),
                            "updated_at": _now(),
                        }
                    },
                )
                created += 1
                continue
            await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {"local_part_id": local_part_id},
                {
                    "$set": {
                        "provider_batch_id": provider_batch_id,
                        "status": str(provider_batch.get("status") or "validating"),
                        "provider_state": provider_batch,
                        "provider_scope": _client_provider_scope(client),
                        "updated_at": _now(),
                    }
                },
            )
            for entry in partition:
                stage_projection = _entry_stage(entry)
                stage_number = int(stage_projection["stage_number"])
                stage_count = int(stage_projection["stage_count"])
                entry_job_filter: Dict[str, Any] = {
                    "job_id": entry["job_id"],
                    "status": {"$in": sorted(ACTIVE_BATCH_JOB_STATUSES)},
                }
                entry_generation = _entry_generation(entry)
                if entry_generation is not None:
                    entry_job_filter["grading_generation"] = entry_generation
                await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                    entry_job_filter,
                    {
                        "$set": {
                            "status": "provider_processing",
                            "provider_batch_id": provider_batch_id,
                            "provider_batch_status": str(
                                provider_batch.get("status") or "validating"
                            ),
                            "provider_expires_at": provider_batch.get("expires_at"),
                            **stage_projection,
                            "progress": {
                                "stage": "provider_processing",
                                "message": (
                                    f"Economy checking stage {stage_number} of {stage_count}: "
                                    f"{stage_projection['provider_phase']}"
                                ),
                            },
                            "updated_at": _now(),
                        },
                        "$unset": {
                            "last_error": "",
                            "failure_code": "",
                            "retryable": "",
                            "operator_action": "",
                        },
                    },
                )
            created += 1
        except Exception as exc:
            failure_contract = classify_economy_batch_failure(exc)
            await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {"local_part_id": local_part_id},
                {
                    "$set": {
                        "status": "imported",
                        "provider_creation_failed": True,
                        "last_error": _short_error(exc),
                        **failure_contract,
                        "updated_at": _now(),
                    }
                },
            )
            await tenant_db[BATCH_ITEMS_COLLECTION].update_many(
                {"local_part_id": local_part_id, "import_status": "pending"},
                {
                    "$set": {
                        "import_status": "failed",
                        "reserved_tokens": 0,
                        "last_error": _short_error(exc),
                        **failure_contract,
                        "updated_at": _now(),
                    }
                },
            )
            for entry in partition:
                await _mark_job_failed(
                    tenant_db,
                    entry["job_id"],
                    exc,
                    expected_generation=_entry_generation(entry),
                )
            if input_file_id:
                await client.delete_file(input_file_id)
    return created


async def _recover_interrupted_part_creation(
    tenant_db: Any,
    *,
    group: Mapping[str, Any],
    client: OpenAIBatchClient,
) -> None:
    """Resolve the upload/create crash boundary without buying duplicate work."""

    incomplete_parts = await tenant_db[BATCH_PARTS_COLLECTION].find(
        {
            "batch_group_id": group["batch_group_id"],
            "provider_batch_id": {"$exists": False},
            "status": {"$in": ["uploading", "awaiting_input_file", "creating"]},
        }
    ).to_list(length=1000)
    if not incomplete_parts:
        return

    provider_by_local_part: Dict[str, Dict[str, Any]] = {}
    after: Optional[str] = None
    # The normal case is the first page. A few bounded pages also cover a busy
    # account without turning a recovery pass into an unbounded provider scan.
    for _ in range(5):
        page = await client.list_batches(after=after)
        values = page.get("data") if isinstance(page.get("data"), list) else []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
            local_part_id = str(metadata.get("stoody_part") or "")
            if local_part_id:
                provider_by_local_part[local_part_id] = dict(value)
        if not page.get("has_more") or not values:
            break
        after = str(values[-1].get("id") or "")
        if not after:
            break

    for part in incomplete_parts:
        local_part_id = str(part.get("local_part_id") or "")
        item_docs = await tenant_db[BATCH_ITEMS_COLLECTION].find(
            {"local_part_id": local_part_id}, {"job_id": 1}
        ).to_list(length=MAX_BATCH_REQUESTS)
        job_ids = [str(item.get("job_id") or "") for item in item_docs if item.get("job_id")]
        recovered = provider_by_local_part.get(local_part_id)
        provider_batch_id = str((recovered or {}).get("id") or "")
        if provider_batch_id:
            await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {"local_part_id": local_part_id, "provider_batch_id": {"$exists": False}},
                {
                    "$set": {
                        "provider_batch_id": provider_batch_id,
                        "status": str(recovered.get("status") or "validating"),
                        "provider_state": recovered,
                        "recovered_after_interruption": True,
                        "updated_at": _now(),
                    }
                },
            )
            if job_ids:
                await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
                    {"job_id": {"$in": job_ids}},
                    {
                        "$set": {
                            "status": "provider_processing",
                            "progress": {
                                "stage": "provider_processing",
                                "message": "Economy checking is running; each provider stage has its own window",
                            },
                            "updated_at": _now(),
                        },
                        "$unset": {"last_error": ""},
                    },
                )
            continue

        # No provider Batch exists for this opaque part id. The interrupted
        # local attempt can be retired and safely rebuilt with fresh ids.
        input_file_id = str(part.get("input_file_id") or "")
        if input_file_id:
            await client.delete_file(input_file_id)
        await tenant_db[BATCH_PARTS_COLLECTION].update_one(
            {"local_part_id": local_part_id, "provider_batch_id": {"$exists": False}},
            {
                "$set": {
                    "status": "imported",
                    "provider_creation_interrupted": True,
                    "last_error": "Recovered an interrupted provider Batch creation",
                    "updated_at": _now(),
                }
            },
        )
        await tenant_db[BATCH_ITEMS_COLLECTION].update_many(
            {"local_part_id": local_part_id, "import_status": "pending"},
            {
                "$set": {
                    "import_status": "superseded",
                    "reserved_tokens": 0,
                    "updated_at": _now(),
                }
            },
        )
        if job_ids:
            await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
                {
                    "job_id": {"$in": job_ids},
                    "status": {"$in": ["batch_queued", "preparing_batch"]},
                },
                {"$set": {"status": "batch_queued", "updated_at": _now()}},
            )


async def prepare_economy_batch_group(
    tenant_db: Any,
    *,
    batch_group_id: str,
) -> Dict[str, Any]:
    """Prepare canonical requests and submit size-bounded provider batches."""

    await ensure_batch_indexes(tenant_db)
    lease_token = uuid.uuid4().hex
    now = _now()
    group = await tenant_db[BATCH_GROUPS_COLLECTION].find_one_and_update(
        {
            "batch_group_id": batch_group_id,
            "$or": [
                {"status": "queued"},
                {
                    "status": "preparing",
                    "preparation_lease_expires_at": {"$lte": now},
                },
            ],
        },
        {
            "$set": {
                "status": "preparing",
                "preparation_lease_token": lease_token,
                "preparation_lease_expires_at": now + timedelta(minutes=30),
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not group:
        current = await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
            {"batch_group_id": batch_group_id}
        )
        return current or {"batch_group_id": batch_group_id, "status": "not_found"}

    existing_client: Optional[OpenAIBatchClient] = None
    incomplete_part_count = await tenant_db[BATCH_PARTS_COLLECTION].count_documents(
        {
            "batch_group_id": batch_group_id,
            "provider_batch_id": {"$exists": False},
            "status": {"$in": ["uploading", "creating"]},
        }
    )
    if incomplete_part_count:
        existing_client = OpenAIBatchClient()
        await _recover_interrupted_part_creation(
            tenant_db,
            group=group,
            client=existing_client,
        )

    confirmed_parts = await tenant_db[BATCH_PARTS_COLLECTION].find(
        {
            "batch_group_id": batch_group_id,
            "provider_batch_id": {"$exists": True},
        },
        {"local_part_id": 1},
    ).to_list(length=1000)
    confirmed_part_ids = [str(part.get("local_part_id") or "") for part in confirmed_parts]
    assigned_job_ids: set[str] = set()
    if confirmed_part_ids:
        assigned_items = await tenant_db[BATCH_ITEMS_COLLECTION].find(
            {"local_part_id": {"$in": confirmed_part_ids}}, {"job_id": 1}
        ).to_list(length=MAX_BATCH_REQUESTS)
        assigned_job_ids = {
            str(item.get("job_id") or "")
            for item in assigned_items
            if item.get("job_id")
        }

    entries: List[Dict[str, Any]] = []
    entry_bytes = 0
    local_completed = 0
    failures = 0
    created_parts = len(confirmed_parts)
    provider_client = existing_client
    preparation_error: Optional[Exception] = None
    try:
        flush_bytes = int(
            os.getenv("EXAMPEN_BATCH_PREPARE_FLUSH_BYTES", str(64 * 1024 * 1024))
        )
    except (TypeError, ValueError):
        flush_bytes = 64 * 1024 * 1024
    flush_bytes = max(8 * 1024 * 1024, min(170 * 1024 * 1024, flush_bytes))

    async def flush_entries() -> Optional[Exception]:
        nonlocal entries, entry_bytes, created_parts, provider_client, failures
        if not entries:
            return None
        pending = entries
        entries = []
        entry_bytes = 0
        try:
            provider_client = provider_client or OpenAIBatchClient()
            created_parts += await _create_provider_parts(
                tenant_db,
                group=group,
                entries=pending,
                stage="primary",
                client=provider_client,
            )
            return None
        except Exception as exc:
            failures += len(pending)
            logger.exception("Could not submit an economy-check preparation chunk")
            for pending_entry in pending:
                await _mark_job_failed(
                    tenant_db,
                    str(pending_entry.get("job_id") or ""),
                    exc,
                    expected_generation=_entry_generation(pending_entry),
                )
            return exc

    jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
        {"job_id": {"$in": list(group.get("job_ids") or [])}}
    ).sort("created_at", 1).to_list(length=5000)
    for job in jobs:
        renewed = await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
            {
                "batch_group_id": batch_group_id,
                "status": "preparing",
                "preparation_lease_token": lease_token,
            },
            {
                "$set": {
                    "preparation_lease_expires_at": _now() + timedelta(minutes=30),
                    "updated_at": _now(),
                }
            },
        )
        if renewed.matched_count != 1:
            return await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
                {"batch_group_id": batch_group_id}
            )
        job_id = str(job.get("job_id") or "")
        submission_id = str(job.get("submission_id") or "")
        if job_id in assigned_job_ids:
            continue
        if str(job.get("status") or "") in {"completed", "blocked_for_review"}:
            continue
        await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
            {"job_id": job_id, "status": {"$in": ["batch_queued", "batch_failed"]}},
            {"$set": {"status": "preparing_batch", "updated_at": _now()}},
        )
        try:
            result, deferred = await _run_grader(
                tenant_db,
                submission_id=submission_id,
            )
            if result is not None:
                if not getattr(result, "handled", False):
                    raise RuntimeError(
                        getattr(result, "skipped_reason", None)
                        or "The grading contract declined this copy"
                    )
                await _commit_grading_result(
                    tenant_db,
                    job=job,
                    result=result,
                    expected_generation=int(job.get("grading_generation") or 0),
                )
                local_completed += 1
                continue
            if not deferred or not deferred.get("request_body"):
                raise RuntimeError("The grader did not produce a Batch request")
            custom_id = _opaque_id("copy")
            entry = {
                "custom_id": custom_id,
                "job_id": job_id,
                "submission_id": submission_id,
                "job_generation": int(job.get("grading_generation") or 0),
                **deferred,
            }
            entry["jsonl_line"] = _jsonl_line(custom_id, entry["request_body"])
            entries.append(entry)
            entry_bytes += len(entry["jsonl_line"])
            if entry_bytes >= flush_bytes or len(entries) >= 250:
                preparation_error = await flush_entries()
                if preparation_error:
                    break
        except Exception as exc:
            failures += 1
            logger.exception("Could not prepare economy checking for job %s", job_id)
            await _mark_job_failed(
                tenant_db,
                job_id,
                exc,
                expected_generation=int(job.get("grading_generation") or 0),
            )

    try:
        if not preparation_error:
            preparation_error = await flush_entries()
        if preparation_error:
            for pending_job in jobs:
                pending_job_id = str(pending_job.get("job_id") or "")
                if not pending_job_id or pending_job_id in assigned_job_ids:
                    continue
                await _mark_job_failed(
                    tenant_db,
                    pending_job_id,
                    preparation_error,
                    expected_generation=int(
                        pending_job.get("grading_generation") or 0
                    ),
                )
        provider_request_count = await tenant_db[BATCH_ITEMS_COLLECTION].count_documents(
            {
                "batch_group_id": batch_group_id,
                "import_status": {"$ne": "superseded"},
            }
        )
        failed_count = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {"job_id": {"$in": list(group.get("job_ids") or [])}, "status": "batch_failed"}
        )
        next_status = (
            "provider_processing"
            if created_parts
            else "completed_with_errors"
            if provider_request_count or failed_count or failures
            else "completed"
        )
        first_failed_job = (
            await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
                {
                    "job_id": {"$in": list(group.get("job_ids") or [])},
                    "status": "batch_failed",
                },
                {"last_error": 1},
            )
            if failed_count
            else None
        )
        group_error = _short_error(
            preparation_error
            or (first_failed_job or {}).get("last_error")
        ) if (preparation_error or first_failed_job) else None
        group_update: Dict[str, Any] = {
            "$set": {
                "status": next_status,
                "provider_request_count": provider_request_count,
                "local_completed_count": int(group.get("local_completed_count") or 0) + local_completed,
                "failed_count": failed_count,
                "updated_at": _now(),
            },
            "$unset": {
                "preparation_lease_token": "",
                "preparation_lease_expires_at": "",
            },
        }
        if group_error:
            group_update["$set"]["last_error"] = group_error
        else:
            group_update["$unset"]["last_error"] = ""
        await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
            {
                "batch_group_id": batch_group_id,
                "status": "preparing",
                "preparation_lease_token": lease_token,
            },
            group_update,
        )
    except Exception as exc:
        await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
            {"batch_group_id": batch_group_id, "preparation_lease_token": lease_token},
            {
                "$set": {"status": "failed", "last_error": _short_error(exc), "updated_at": _now()},
                "$unset": {"preparation_lease_token": "", "preparation_lease_expires_at": ""},
            },
        )
        raise
    return await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
        {"batch_group_id": batch_group_id}
    )


async def _read_part_results(
    client: OpenAIBatchClient,
    provider_state: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for field in ("output_file_id", "error_file_id"):
        file_id = str(provider_state.get(field) or "")
        if not file_id:
            continue
        results.update(parse_batch_jsonl(await client.file_content(file_id)))
    return results


async def _import_item(
    tenant_db: Any,
    *,
    item: Dict[str, Any],
    provider_line: Optional[Mapping[str, Any]],
    missing_result_error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    expected_generation = _entry_generation(item)
    job_filter: Dict[str, Any] = {
        "job_id": item.get("job_id"),
        "status": {"$in": sorted(ACTIVE_BATCH_JOB_STATUSES)},
    }
    if expected_generation is not None:
        job_filter["grading_generation"] = expected_generation
    claimed = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        job_filter,
        {
            "$set": {
                "status": "importing_batch",
                "provider_phase": "importing",
                "progress": {
                    "stage": "importing_batch",
                    "message": "Validating and importing the returned marks",
                },
                "updated_at": _now(),
            }
        },
    )
    if claimed.matched_count != 1:
        raise StaleBatchGenerationError(
            "Provider output belongs to a superseded grading run"
        )
    job = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": item.get("job_id")}
    )
    if not job:
        raise RuntimeError("Processing job no longer exists")
    line = dict(provider_line or {})
    response = line.get("response") if isinstance(line.get("response"), dict) else {}
    response_body = response.get("body") if isinstance(response.get("body"), dict) else None
    status_code = int(response.get("status_code") or 0) if response else 0
    if status_code != 200 or response_body is None:
        error = (
            line.get("error")
            or (response.get("body") if response else None)
            or missing_result_error
            or "OpenAI completed the Batch without returning this copy's result"
        )
        raise RuntimeError(_short_error(error))

    previous = [
        dict(value)
        for value in (item.get("prior_response_bodies") or [])
        if isinstance(value, Mapping)
    ]
    response_bodies = previous + [dict(response_body)]
    recorded = [int(value) for value in (item.get("recorded_call_indexes") or [])]
    result, deferred = await _run_grader(
        tenant_db,
        submission_id=str(item.get("submission_id") or ""),
        grader_kind=str(item.get("grader_kind") or "") or None,
        response_bodies=response_bodies,
        recorded_call_indexes=recorded,
    )
    current_call_index = int(item.get("call_index") or 0)
    if deferred:
        if int(deferred.get("call_index") or 0) <= current_call_index:
            raise RuntimeError("Grader requested an already completed provider call")
        custom_id = _opaque_id("recovery")
        recovery = {
            "custom_id": custom_id,
            "job_id": str(item.get("job_id") or ""),
            "submission_id": str(item.get("submission_id") or ""),
            "job_generation": expected_generation,
            "prior_response_bodies": response_bodies,
            "recorded_call_indexes": sorted(set(recorded + [current_call_index])),
            "parent_custom_id": str(item.get("custom_id") or ""),
            **deferred,
        }
        recovery["jsonl_line"] = _jsonl_line(custom_id, recovery["request_body"])
        return recovery
    if result is None or not getattr(result, "handled", False):
        raise RuntimeError("Imported provider output did not complete grading")
    await _commit_grading_result(
        tenant_db,
        job=job,
        result=result,
        expected_generation=expected_generation,
    )
    return None


async def reconcile_economy_batches(tenant_db: Any) -> Dict[str, int]:
    """Poll provider parts, import outputs idempotently, and submit recovery parts."""

    await ensure_batch_indexes(tenant_db)
    summary = {"parts_polled": 0, "items_imported": 0, "items_failed": 0, "recovery_parts": 0}
    client: Optional[OpenAIBatchClient] = None
    parts = await tenant_db[BATCH_PARTS_COLLECTION].find(
        {
            "status": {
                "$in": sorted(
                    ACTIVE_PROVIDER_STATUSES
                    | TERMINAL_PROVIDER_STATUSES
                    | {PART_IMPORTING_STATUS}
                )
            }
        }
    ).sort("created_at", 1).to_list(length=1000)
    for part in parts:
        if str(part.get("status") or "") == "imported":
            continue
        provider_batch_id = str(part.get("provider_batch_id") or "")
        if not provider_batch_id:
            continue
        if client is None:
            client = OpenAIBatchClient()
        import_lease_token = uuid.uuid4().hex
        if str(part.get("status") or "") == PART_IMPORTING_STATUS:
            claimed_part = await tenant_db[BATCH_PARTS_COLLECTION].find_one_and_update(
                {
                    "local_part_id": part.get("local_part_id"),
                    "status": PART_IMPORTING_STATUS,
                    "import_lease_expires_at": {"$lte": _now()},
                },
                {
                    "$set": {
                        "import_lease_token": import_lease_token,
                        "import_lease_expires_at": _now() + timedelta(hours=2),
                        "updated_at": _now(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if not claimed_part:
                continue
            provider_state = dict(claimed_part.get("provider_state") or {})
            provider_status = str(provider_state.get("status") or "")
            if provider_status not in TERMINAL_PROVIDER_STATUSES:
                # A stale malformed claim should be polled normally next pass.
                await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                    {
                        "local_part_id": part.get("local_part_id"),
                        "import_lease_token": import_lease_token,
                    },
                    {
                        "$set": {"status": provider_status or "in_progress", "updated_at": _now()},
                        "$unset": {"import_lease_token": "", "import_lease_expires_at": ""},
                    },
                )
                continue
            part = claimed_part
        else:
            provider_state = await client.retrieve_batch(provider_batch_id)
            observed_status = str(part.get("status") or "")
            provider_status = str(provider_state.get("status") or observed_status or "in_progress")
            provider_failure = provider_batch_failure(provider_state)
            summary["parts_polled"] += 1
            part_set: Dict[str, Any] = {
                "status": provider_status,
                "provider_state": provider_state,
                "provider_scope": _client_provider_scope(client),
                "updated_at": _now(),
            }
            if provider_failure:
                part_set["last_error"] = provider_failure
                part_set.update(classify_economy_batch_failure(provider_failure))
            observed = await tenant_db[BATCH_PARTS_COLLECTION].update_one(
                {
                    "local_part_id": part.get("local_part_id"),
                    "status": observed_status,
                },
                {"$set": part_set},
            )
            if observed.matched_count and provider_status in ACTIVE_PROVIDER_STATUSES:
                progress = describe_provider_batch_progress(provider_state)
                item_jobs = await tenant_db[BATCH_ITEMS_COLLECTION].find(
                    {"local_part_id": part.get("local_part_id")},
                    {"job_id": 1, "job_generation": 1},
                ).to_list(length=MAX_BATCH_REQUESTS)
                for item_job in item_jobs:
                    item_job_id = str(item_job.get("job_id") or "")
                    if not item_job_id:
                        continue
                    progress_filter: Dict[str, Any] = {
                        "job_id": item_job_id,
                        "status": {"$in": sorted(ACTIVE_BATCH_JOB_STATUSES)},
                    }
                    item_generation = _entry_generation(item_job)
                    if item_generation is not None:
                        progress_filter["grading_generation"] = item_generation
                    await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                        progress_filter,
                        {
                            "$set": {
                                "status": progress["job_status"],
                                "provider_batch_status": progress["status"],
                                "provider_batch_id": provider_batch_id,
                                "provider_last_polled_at": _now(),
                                "provider_request_counts": {
                                    "total": progress["total"],
                                    "completed": progress["completed"],
                                    "failed": progress["failed"],
                                },
                                "provider_expires_at": progress["expires_at"],
                                "provider_delayed": progress["delayed"],
                                "progress": {
                                    "stage": progress["stage"],
                                    "message": progress["message"],
                                },
                                "updated_at": _now(),
                            },
                            "$unset": {"last_error": ""},
                        },
                    )
            if not observed.matched_count or provider_status not in TERMINAL_PROVIDER_STATUSES:
                continue
            claimed_part = await tenant_db[BATCH_PARTS_COLLECTION].find_one_and_update(
                {
                    "local_part_id": part.get("local_part_id"),
                    "status": provider_status,
                },
                {
                    "$set": {
                        "status": PART_IMPORTING_STATUS,
                        "import_lease_token": import_lease_token,
                        "import_lease_expires_at": _now() + timedelta(hours=2),
                        "updated_at": _now(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if not claimed_part:
                continue
            part = claimed_part

        results = await _read_part_results(client, provider_state)
        provider_failure = provider_batch_failure(provider_state)
        group = await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
            {"batch_group_id": part.get("batch_group_id")}
        )
        cancellation_requested = bool(
            group
            and str(group.get("status") or "") == "cancelling"
            and provider_status == "cancelled"
        )
        recovery_entries: List[Dict[str, Any]] = []
        claimable_item_filter: Dict[str, Any] = {
            "$or": [
                {"import_status": {"$in": ["pending", "retry"]}},
                {
                    "import_status": "importing",
                    "import_lease_expires_at": {"$lte": _now()},
                },
            ]
        }
        items = await tenant_db[BATCH_ITEMS_COLLECTION].find(
            {"local_part_id": part.get("local_part_id"), **claimable_item_filter}
        ).to_list(length=MAX_BATCH_REQUESTS)
        for raw_item in items:
            item = await tenant_db[BATCH_ITEMS_COLLECTION].find_one_and_update(
                {"custom_id": raw_item.get("custom_id"), **claimable_item_filter},
                {
                    "$set": {
                        "import_status": "importing",
                        "import_lease_expires_at": _now() + timedelta(hours=2),
                        "updated_at": _now(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if not item:
                continue
            provider_line = results.get(str(item.get("custom_id") or ""))
            if cancellation_requested and provider_line is None:
                await tenant_db[BATCH_ITEMS_COLLECTION].update_one(
                    {"custom_id": item["custom_id"], "import_status": "importing"},
                    {
                        "$set": {
                            "import_status": "superseded",
                            "reserved_tokens": 0,
                            "last_error": "Cancelled before OpenAI checked this copy",
                            "updated_at": _now(),
                        },
                        "$unset": {"import_lease_expires_at": ""},
                    },
                )
                cancel_job_filter: Dict[str, Any] = {
                    "job_id": str(item.get("job_id") or ""),
                    "status": {"$in": sorted(ACTIVE_BATCH_JOB_STATUSES)},
                }
                item_generation = _entry_generation(item)
                if item_generation is not None:
                    cancel_job_filter["grading_generation"] = item_generation
                await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                    cancel_job_filter,
                    {
                        "$set": {
                            "status": ECONOMY_WAITING_JOB_STATUS,
                            "progress": {
                                "stage": ECONOMY_WAITING_JOB_STATUS,
                                "message": "Waiting for economy checking",
                            },
                            "updated_at": _now(),
                        },
                        "$unset": {"provider_batch_group_id": "", "last_error": ""},
                    },
                )
                continue
            try:
                recovery = await _import_item(
                    tenant_db,
                    item=item,
                    provider_line=provider_line,
                    missing_result_error=provider_failure,
                )
                next_status = "waiting_recovery" if recovery else "completed"
                await tenant_db[BATCH_ITEMS_COLLECTION].update_one(
                    {"custom_id": item["custom_id"], "import_status": "importing"},
                    {
                        "$set": {
                            "import_status": next_status,
                            "reserved_tokens": 0,
                            "imported_at": _now(),
                            "updated_at": _now(),
                        },
                        "$unset": {"import_lease_expires_at": ""},
                    },
                )
                if recovery:
                    recovery_entries.append(recovery)
                else:
                    summary["items_imported"] += 1
            except StaleBatchGenerationError as exc:
                logger.info(
                    "Ignoring superseded OpenAI Batch item %s: %s",
                    item.get("custom_id"),
                    exc,
                )
                await tenant_db[BATCH_ITEMS_COLLECTION].update_one(
                    {"custom_id": item["custom_id"], "import_status": "importing"},
                    {
                        "$set": {
                            "import_status": "superseded",
                            "reserved_tokens": 0,
                            "last_error": _short_error(exc),
                            "updated_at": _now(),
                        },
                        "$unset": {"import_lease_expires_at": ""},
                    },
                )
            except Exception as exc:
                summary["items_failed"] += 1
                logger.exception("Could not import OpenAI Batch item %s", item.get("custom_id"))
                failure_contract = classify_economy_batch_failure(exc)
                await tenant_db[BATCH_ITEMS_COLLECTION].update_one(
                    {"custom_id": item["custom_id"], "import_status": "importing"},
                    {
                        "$set": {
                            "import_status": "failed",
                            "reserved_tokens": 0,
                            "last_error": _short_error(exc),
                            **failure_contract,
                            "updated_at": _now(),
                        },
                        "$unset": {"import_lease_expires_at": ""},
                    },
                )
                await _mark_job_failed(
                    tenant_db,
                    str(item.get("job_id") or ""),
                    exc,
                    expected_generation=_entry_generation(item),
                )
        if recovery_entries and group:
            summary["recovery_parts"] += await _create_provider_parts(
                tenant_db,
                group=group,
                entries=recovery_entries,
                stage="recovery",
                client=client,
            )
        finalized_part = await tenant_db[BATCH_PARTS_COLLECTION].update_one(
            {
                "local_part_id": part.get("local_part_id"),
                "import_lease_token": import_lease_token,
            },
            {
                "$set": {"status": "imported", "imported_at": _now(), "updated_at": _now()},
                "$unset": {"import_lease_token": "", "import_lease_expires_at": ""},
            },
        )
        if finalized_part.matched_count:
            for file_field in ("input_file_id", "output_file_id", "error_file_id"):
                file_id = str(provider_state.get(file_field) or part.get(file_field) or "")
                if file_id:
                    await client.delete_file(file_id)

    groups = await tenant_db[BATCH_GROUPS_COLLECTION].find(
        {"status": {"$in": ["provider_processing", "importing", "cancelling"]}}
    ).to_list(length=1000)
    for group in groups:
        group_was_cancelling = str(group.get("status") or "") == "cancelling"
        active_parts = await tenant_db[BATCH_PARTS_COLLECTION].count_documents(
            {
                "batch_group_id": group.get("batch_group_id"),
                "status": {"$ne": "imported"},
            }
        )
        if active_parts:
            continue
        failed_items = await tenant_db[BATCH_ITEMS_COLLECTION].count_documents(
            {"batch_group_id": group.get("batch_group_id"), "import_status": "failed"}
        )
        completed_items = await tenant_db[BATCH_ITEMS_COLLECTION].count_documents(
            {"batch_group_id": group.get("batch_group_id"), "import_status": "completed"}
        )
        failed_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {
                "job_id": {"$in": list(group.get("job_ids") or [])},
                "status": "batch_failed",
            }
        )
        total_failures = max(failed_items, failed_jobs)
        first_failed_item = (
            await tenant_db[BATCH_ITEMS_COLLECTION].find_one(
                {
                    "batch_group_id": group.get("batch_group_id"),
                    "import_status": "failed",
                },
                {"last_error": 1},
            )
            if total_failures
            else None
        )
        first_failed_job = (
            await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
                {
                    "job_id": {"$in": list(group.get("job_ids") or [])},
                    "status": "batch_failed",
                },
                {"last_error": 1},
            )
            if total_failures
            else None
        )
        group_error = str(
            (first_failed_item or {}).get("last_error")
            or (first_failed_job or {}).get("last_error")
            or ""
        ).strip()
        failure_contract = classify_economy_batch_failure(group_error) if group_error else {}
        final_group_update: Dict[str, Any] = {
            "$set": {
                "status": (
                    "cancelled"
                    if group_was_cancelling
                    else "completed_with_errors"
                    if total_failures
                    else "completed"
                ),
                "completed_count": completed_items,
                "failed_count": total_failures,
                "completed_at": _now(),
                "updated_at": _now(),
                **failure_contract,
            }
        }
        if group_error:
            final_group_update["$set"]["last_error"] = _short_error(group_error)
        else:
            final_group_update["$unset"] = {
                "last_error": "",
                "failure_code": "",
                "retryable": "",
                "operator_action": "",
            }
        await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
            {"batch_group_id": group.get("batch_group_id")},
            final_group_update,
        )
    return summary


async def reconcile_economy_batch_groups(tenant_db: Any) -> Dict[str, int]:
    """Resume preparation leases and poll all active provider work for a tenant."""

    queued = await tenant_db[BATCH_GROUPS_COLLECTION].find(
        {
            "$or": [
                {"status": "queued"},
                {"status": "preparing", "preparation_lease_expires_at": {"$lte": _now()}},
            ]
        }
    ).to_list(length=100)
    prepared = 0
    for group in queued:
        try:
            await prepare_economy_batch_group(
                tenant_db,
                batch_group_id=str(group.get("batch_group_id") or ""),
            )
            prepared += 1
        except Exception:
            logger.exception(
                "Economy Batch preparation failed for %s", group.get("batch_group_id")
            )
    polled = await reconcile_economy_batches(tenant_db)
    return {"groups_prepared": prepared, **polled}


async def cancel_economy_batch_group(
    tenant_db: Any,
    *,
    batch_group_id: str,
) -> Dict[str, Any]:
    group = await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
        {"batch_group_id": batch_group_id}
    )
    if not group:
        raise ValueError("Economy batch was not found")
    current_status = str(group.get("status") or "")
    if current_status not in ACTIVE_GROUP_STATUSES:
        return group
    await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
        {"batch_group_id": batch_group_id, "status": current_status},
        {
            "$set": {"status": "cancelling", "updated_at": _now()},
            "$unset": {
                "preparation_lease_token": "",
                "preparation_lease_expires_at": "",
            },
        },
    )
    parts = await tenant_db[BATCH_PARTS_COLLECTION].find(
        {"batch_group_id": batch_group_id, "status": {"$in": sorted(ACTIVE_PROVIDER_STATUSES)}}
    ).to_list(length=1000)
    provider_parts = [part for part in parts if str(part.get("provider_batch_id") or "")]
    if provider_parts:
        client = OpenAIBatchClient()
        for part in provider_parts:
            provider_batch_id = str(part.get("provider_batch_id") or "")
            await client.cancel_batch(provider_batch_id)
    else:
        now = _now()
        await tenant_db[BATCH_GROUPS_COLLECTION].update_one(
            {"batch_group_id": batch_group_id, "status": "cancelling"},
            {"$set": {"status": "cancelled", "completed_at": now, "updated_at": now}},
        )
        await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
            {
                "job_id": {"$in": list(group.get("job_ids") or [])},
                "status": {"$in": ["batch_queued", "preparing_batch"]},
            },
            {
                "$set": {
                    "status": ECONOMY_WAITING_JOB_STATUS,
                    "progress": {
                        "stage": ECONOMY_WAITING_JOB_STATUS,
                        "message": "Waiting for economy checking",
                    },
                    "updated_at": now,
                },
                "$unset": {"provider_batch_group_id": ""},
            },
        )
    return await tenant_db[BATCH_GROUPS_COLLECTION].find_one(
        {"batch_group_id": batch_group_id}
    )


__all__ = [
    "OpenAIBatchClient",
    "cancel_economy_batch_group",
    "classify_economy_batch_failure",
    "create_economy_batch_group",
    "parse_batch_jsonl",
    "partition_batch_requests",
    "prepare_economy_batch_group",
    "reconcile_economy_batch_groups",
]
