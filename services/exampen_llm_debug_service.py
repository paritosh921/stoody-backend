"""Staff-authorized, redacted LLM request/response traces for PCR review.

The provider receives inline base64 images.  Persisting those bodies again in
MongoDB would create an unbounded second copy of every answer sheet, so grading
stores request text plus image hashes and metadata.  This service regenerates
the exact provider image bytes from immutable canonical pages and verifies
persisted hashes before an authorized preview is returned.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from api.v1._exampen_imports import load_exampen
from services.objective_answer_ledger_contract import (
    OBJECTIVE_PROMPT_VERSION,
    objective_extraction_catalog,
)


class LlmDebugTraceError(RuntimeError):
    """The requested trace is unavailable or fails an integrity check."""


async def _objective_context(
    tenant_db: Any,
    submission_id: str,
) -> Dict[str, Any]:
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if not submission:
        raise LlmDebugTraceError("Answer-copy submission not found")

    run_id = str(submission.get("document_grading_run_id") or "")
    run: Optional[Dict[str, Any]] = None
    if run_id:
        run = await tenant_db["evalpen_document_grading_runs"].find_one(
            {"run_id": run_id, "submission_id": submission_id}
        )
    if not run:
        run = await tenant_db["evalpen_document_grading_runs"].find_one(
            {"submission_id": submission_id},
            sort=[("updated_at", -1)],
        )
    if not run:
        raise LlmDebugTraceError("No LLM grading run exists for this answer copy")
    run_id = str(run.get("run_id") or "")
    prompt_version = str(run.get("prompt_version") or "")
    if prompt_version != OBJECTIVE_PROMPT_VERSION:
        raise LlmDebugTraceError(
            "Exact request reconstruction is currently available for the "
            "objective answer-ledger lane only"
        )

    exam_id = str(submission.get("exam_id") or run.get("exam_id") or "")
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if not exam:
        raise LlmDebugTraceError("The grading run's exam is missing")
    paper_version_id = str(exam.get("paper_version_id") or "")
    paper = await tenant_db["exampen_paper_versions"].find_one(
        {"paper_version_id": paper_version_id}
    )
    if not paper:
        raise LlmDebugTraceError("The immutable paper snapshot is missing")

    questions = await tenant_db["evalpen_questions"].find(
        {"exam_id": exam_id}
    ).sort("question_number", 1).to_list(length=2000)
    if not questions:
        raise LlmDebugTraceError("The immutable question catalog is empty")
    answer_pages = await tenant_db["evalpen_answer_pages"].find(
        {"submission_id": submission_id}
    ).sort("page_number", 1).to_list(length=100)
    if not answer_pages:
        raise LlmDebugTraceError("The canonical answer-copy pages are missing")

    grading = load_exampen("pcr.services.full_document_grading")
    page_assets, _ = await grading._student_page_assets(answer_pages)
    catalog = objective_extraction_catalog(questions)
    paper_context = dict(paper.get("paper_context") or {})
    paper_hash = str(
        paper_context.get("question_paper_sha256")
        or exam.get("paper_content_hash")
        or paper.get("content_hash")
        or ""
    )
    cache_key = (
        "pcr-objective-ledger-"
        + hashlib.sha256(
            (
                paper_hash
                + "|"
                + OBJECTIVE_PROMPT_VERSION
                + "|"
                + json.dumps(catalog, sort_keys=True, separators=(",", ":"))
            ).encode("utf-8")
        ).hexdigest()[:32]
    )
    contract = dict(exam.get("pcr_grading_contract") or {})
    model_id = str(
        run.get("model_used")
        or run.get("requested_model_id")
        or contract.get("model_id")
        or "unknown"
    )
    try:
        temperature = float(contract.get("temperature", 0.1))
    except (TypeError, ValueError):
        temperature = 0.1
    reasoning_effort = str(contract.get("reasoning_effort") or "medium")

    call_specs: Dict[int, Dict[str, Any]] = {}
    manifests: Dict[int, Dict[str, Any]] = {}
    asset_metadata: Dict[int, List[Dict[str, Any]]] = {}
    asset_blobs: Dict[str, Tuple[bytes, str]] = {}
    for asset in page_assets:
        call_spec, _ = grading.build_objective_page_call_spec(
            asset=asset,
            catalog=catalog,
            model_id=model_id,
            prompt_cache_key=cache_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            submission_id=submission_id,
            exam_id=exam_id,
            run_id=run_id,
            question_count=len(questions),
        )
        manifest, image_assets, blobs = (
            grading.build_llm_debug_request_manifest(call_spec)
        )
        serialized_manifest = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if '"correct_answer"' in serialized_manifest:
            raise LlmDebugTraceError(
                "Security invariant failed: answer key entered the LLM request"
            )
        call_specs[asset.page_number] = call_spec
        manifests[asset.page_number] = manifest
        asset_metadata[asset.page_number] = image_assets
        asset_blobs.update(blobs)

    return {
        "submission": submission,
        "exam": exam,
        "paper": paper,
        "run": run,
        "run_id": run_id,
        "exam_id": exam_id,
        "questions": questions,
        "manifests": manifests,
        "asset_metadata": asset_metadata,
        "asset_blobs": asset_blobs,
        "call_specs": call_specs,
    }


async def build_submission_llm_debug_bundle(
    tenant_db: Any,
    submission_id: str,
) -> Dict[str, Any]:
    """Return the staff debugger payload without duplicating image bodies."""

    context = await _objective_context(tenant_db, submission_id)
    run = context["run"]
    run_id = context["run_id"]
    persisted_traces = await tenant_db["evalpen_llm_debug_traces"].find(
        {"submission_id": submission_id, "run_id": run_id}
    ).sort("page_number", 1).to_list(length=100)
    effective_model = str(
        run.get("model_used")
        or run.get("requested_model_id")
        or next(
            (
                (item.get("usage") or {}).get("model")
                for item in persisted_traces
                if isinstance(item.get("usage"), dict)
                and (item.get("usage") or {}).get("model")
            ),
            "",
        )
        or next(
            (
                (item.get("request") or {}).get("model_id")
                for item in persisted_traces
                if isinstance(item.get("request"), dict)
                and (item.get("request") or {}).get("model_id")
            ),
            "",
        )
        or "unknown"
    )
    persisted_by_page = {
        int(item.get("page_number") or 0): item
        for item in persisted_traces
        if int(item.get("page_number") or 0) > 0
    }
    saved_ledgers = dict(run.get("objective_page_ledgers") or {})
    saved_usages = dict(run.get("objective_page_ledger_usages") or {})
    pages: List[Dict[str, Any]] = []

    for page_number in sorted(context["manifests"]):
        trace = persisted_by_page.get(page_number)
        reconstructed_assets = context["asset_metadata"][page_number]
        if trace:
            persisted_assets = list(trace.get("image_assets") or [])
            reconstructed_by_id = {
                item["asset_id"]: item for item in reconstructed_assets
            }
            for persisted_asset in persisted_assets:
                current = reconstructed_by_id.get(
                    str(persisted_asset.get("asset_id") or "")
                )
                if (
                    not current
                    or current.get("sha256") != persisted_asset.get("sha256")
                ):
                    raise LlmDebugTraceError(
                        f"Provider image integrity check failed for page {page_number}"
                    )
            request = dict(trace.get("request") or {})
            image_assets = persisted_assets
            raw_response = str(trace.get("raw_response") or "")
            parsed_response = trace.get("parsed_response")
            usage = dict(trace.get("usage") or {})
            trace_origin = "persisted_provider_trace"
            response_source = "verbatim_provider_response"
            trace_status = str(trace.get("status") or "unknown")
            response_error = trace.get("response_error")
            provider_status = trace.get("provider_status")
            incomplete_reason = trace.get("incomplete_reason")
            requested_at = trace.get("requested_at")
            completed_at = trace.get("completed_at")
        else:
            request = context["manifests"][page_number]
            image_assets = reconstructed_assets
            parsed_response = saved_ledgers.get(str(page_number))
            raw_response = (
                json.dumps(
                    parsed_response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if isinstance(parsed_response, dict)
                else ""
            )
            usage = dict(saved_usages.get(str(page_number)) or {})
            trace_origin = "reconstructed_from_immutable_input"
            response_source = "parsed_run_checkpoint"
            trace_status = "historical_trace_reconstructed"
            response_error = None
            provider_status = None
            incomplete_reason = None
            requested_at = run.get("created_at")
            completed_at = run.get("completed_at")

        browser_assets = []
        for item in image_assets:
            asset = dict(item)
            asset_id = str(asset.get("asset_id") or "")
            asset["preview_path"] = (
                "/api/v1/evalpen/review/submissions/"
                + quote(submission_id, safe="")
                + "/llm-debug/assets/"
                + quote(asset_id, safe="")
            )
            browser_assets.append(asset)
        pages.append(
            {
                "page_number": page_number,
                "status": trace_status,
                "trace_origin": trace_origin,
                "requested_at": requested_at,
                "completed_at": completed_at,
                "request": request,
                "images": browser_assets,
                "response": {
                    "source": response_source,
                    "raw": raw_response,
                    "parsed": parsed_response,
                    "usage": usage,
                    "error": response_error,
                    "provider_status": provider_status,
                    "incomplete_reason": incomplete_reason,
                },
            }
        )

    return {
        "submission_id": submission_id,
        "exam_id": context["exam_id"],
        "run": {
            "run_id": run_id,
            "status": run.get("status"),
            "prompt_version": run.get("prompt_version"),
            "model_used": effective_model,
            "input_fingerprint": run.get("input_fingerprint"),
            "created_at": run.get("created_at"),
            "completed_at": run.get("completed_at"),
            "token_usage": run.get("token_usage") or {},
        },
        "security": {
            "answer_key_sent_to_llm": False,
            "api_credentials_exposed": False,
            "image_access": "staff-authorized, no-store",
        },
        "trace_quality": (
            "persisted"
            if pages and all(
                page["trace_origin"] == "persisted_provider_trace"
                for page in pages
            )
            else "reconstructed_historical"
        ),
        "pages": pages,
        "validation": {
            "errors": list(run.get("validation_errors") or []),
            "document_review": dict(run.get("document_review") or {}),
            "result": dict(run.get("result") or {}),
            "merged_ledger": run.get("validated_payload"),
        },
    }


async def get_submission_llm_debug_asset(
    tenant_db: Any,
    submission_id: str,
    asset_id: str,
) -> Tuple[bytes, str, str]:
    """Return one verified image body that was part of the provider request."""

    context = await _objective_context(tenant_db, submission_id)
    blob = context["asset_blobs"].get(asset_id)
    if not blob:
        raise LlmDebugTraceError("LLM request image was not found")
    image_bytes, media_type = blob
    digest = hashlib.sha256(image_bytes).hexdigest()

    persisted = await tenant_db["evalpen_llm_debug_traces"].find_one(
        {
            "submission_id": submission_id,
            "run_id": context["run_id"],
            "image_assets.asset_id": asset_id,
        },
        {"image_assets": 1},
    )
    if persisted:
        expected = next(
            (
                str(item.get("sha256") or "")
                for item in persisted.get("image_assets") or []
                if str(item.get("asset_id") or "") == asset_id
            ),
            "",
        )
        if not expected or expected != digest:
            raise LlmDebugTraceError(
                "The reconstructed provider image does not match its saved hash"
            )
    return image_bytes, media_type, digest
