"""Shared policy for immutable PCR contracts and selected-copy upgrades.

An exam contract remains the historical cohort default.  A teacher may
explicitly reprocess one unpublished legacy Subjective copy through the current
worker without mutating or regrading the rest of the cohort.  That exception is
stored on the durable processing job and copied into its grading-run audit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4


V16_PROMPT_VERSION = "pcr-full-document-visual-v16"
V16_PIPELINE_VERSION = 7
V16_MAPPING_PIPELINE_VERSION = "whole-copy-rubric-v7"
V16_REQUIRED_PROCESSING_PATH = "full_document_visual"
SELECTED_COPY_CONTRACT_SCOPE = "selected_submission_reprocess"

LEGACY_SELECTED_COPY_SOURCE_VERSIONS = frozenset(
    {
        "pcr-full-document-visual-v4",
        "pcr-full-document-visual-v5",
        "pcr-full-document-visual-v6",
        "pcr-full-document-visual-v11",
        "pcr-full-document-visual-v12",
        "pcr-full-document-visual-v13",
        "pcr-full-document-visual-v14",
        "pcr-full-document-visual-v15",
    }
)
SUPPORTED_WORKER_PIPELINES = frozenset({4, 5, 6, V16_PIPELINE_VERSION})


def is_supported_worker_contract(contract: Any) -> bool:
    payload = contract if isinstance(contract, Mapping) else {}
    candidate = payload.get("pipeline_version")
    if candidate is None:
        prompt_version = str(payload.get("prompt_version") or "").strip()
        if prompt_version.endswith("-v16"):
            candidate = V16_PIPELINE_VERSION
        elif prompt_version.endswith("-v15"):
            candidate = 6
        elif prompt_version.endswith("-v14"):
            candidate = 5
        elif prompt_version.endswith("-v13") or not prompt_version:
            candidate = 4
    try:
        return int(candidate) in SUPPORTED_WORKER_PIPELINES
    except (TypeError, ValueError):
        return False


def is_legacy_selected_copy_source(contract: Any) -> bool:
    payload = contract if isinstance(contract, Mapping) else {}
    return (
        str(payload.get("prompt_version") or "").strip()
        in LEGACY_SELECTED_COPY_SOURCE_VERSIONS
    )


def selected_copy_contract_override(job: Any) -> dict[str, Any]:
    payload = job if isinstance(job, Mapping) else {}
    override = payload.get("grading_contract_override")
    if not isinstance(override, Mapping):
        return {}
    candidate = dict(override)
    try:
        pipeline_version = int(candidate.get("pipeline_version") or 0)
    except (TypeError, ValueError):
        return {}
    if (
        candidate.get("scope") != SELECTED_COPY_CONTRACT_SCOPE
        or not candidate.get("override_id")
        or str(candidate.get("target_submission_id") or "")
        != str(payload.get("submission_id") or "")
        or candidate.get("prompt_version") != V16_PROMPT_VERSION
        or pipeline_version != V16_PIPELINE_VERSION
        or candidate.get("mapping_pipeline_version") != V16_MAPPING_PIPELINE_VERSION
        or candidate.get("required_processing_path") != V16_REQUIRED_PROCESSING_PATH
    ):
        return {}
    return candidate


def effective_grading_contract(
    exam_contract: Any,
    job: Any = None,
) -> tuple[dict[str, Any], str]:
    override = selected_copy_contract_override(job)
    if override:
        return override, SELECTED_COPY_CONTRACT_SCOPE
    return (
        dict(exam_contract) if isinstance(exam_contract, Mapping) else {},
        "exam",
    )


def build_selected_copy_v16_override(
    source_contract: Mapping[str, Any],
    *,
    submission_id: str,
    requested_by: str,
    requested_at: datetime,
) -> dict[str, Any]:
    if not is_legacy_selected_copy_source(source_contract):
        raise ValueError(
            "The source contract is not eligible for selected-copy upgrade"
        )
    override: dict[str, Any] = {
        "override_id": f"PCR-COPY-UPGRADE-{uuid4().hex}",
        "scope": SELECTED_COPY_CONTRACT_SCOPE,
        "target_submission_id": str(submission_id),
        "source_prompt_version": str(source_contract.get("prompt_version") or ""),
        "prompt_version": V16_PROMPT_VERSION,
        "pipeline_version": V16_PIPELINE_VERSION,
        "mapping_pipeline_version": V16_MAPPING_PIPELINE_VERSION,
        "required_processing_path": V16_REQUIRED_PROCESSING_PATH,
        "requested_by": requested_by or "unknown",
        "requested_at": requested_at,
    }
    for field in ("model_id", "temperature", "reasoning_effort"):
        if source_contract.get(field) is not None:
            override[field] = source_contract[field]
    return override
