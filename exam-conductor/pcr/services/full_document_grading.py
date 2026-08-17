"""Submission-level visual grading for PCR answer copies.

This is the primary camera/PDF path for papers where handwriting, diagrams,
tables, and answer ownership cannot safely be reduced to OCR text first. An
answer-key-free whole-copy request maps exact student-owned evidence regions;
one cached grading request then scores that immutable evidence against the
frozen solution and criterion plan.

Deterministic code does not decide what the handwriting means.  It validates
the model's evidence ledger against immutable question IDs, page bounds, and
locked mark limits.  Missing or ambiguous evidence becomes ``unresolved`` and
blocks publication; it never becomes an inferred zero.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from pymongo.errors import DuplicateKeyError
from services.pcr_grading_contract_policy import (
    SELECTED_COPY_CONTRACT_SCOPE,
    effective_grading_contract,
    selected_copy_contract_override,
)
from services.objective_scoring_service import (
    ObjectiveScoringContractError,
    score_objective_response,
)
from ..domain.response_models import ContentType
from ..marking_policy import (
    method_policy_instruction,
    flatten_assessment_unit_criteria,
    normalize_assessment_units,
    normalize_marking_criteria,
    normalize_marking_policy,
    normalize_method_policy,
    strictness_instruction,
    validate_assessment_units,
)
from ..storage.evaluation_repo import EvaluationRepository
from ..storage.response_repo import DetectedResponseRepository
from .ocr_service import AssetIntegrityError, _resolve_image_base64
from .visual_evidence_graph import (
    EVIDENCE_GRAPH_VERSION,
    PROMPT_VERSION as _EVIDENCE_GRAPH_PROMPT_VERSION,
    V15_PROMPT_VERSION as _EVIDENCE_GRAPH_V15_PROMPT_VERSION,
    evidence_mapping_schema,
    compact_mapping_schema,
    compact_mapping_system_instructions,
    merge_compact_mapping_payloads,
    reconcile_compact_mapping_recovery,
    grading_schema as evidence_grading_schema,
    grading_system_instructions,
    mapping_system_instructions,
    merge_mapping_and_grading,
    verification_schema,
    verification_system_instructions,
    validate_mapping_payload,
)
from .orientation_views import (
    OrientationViewError,
    build_orientation_views,
    view_region_to_original,
)
from .whole_copy_grading import (
    PROMPT_VERSION as _V16_PROMPT_VERSION,
    merge_recovery_payload as _merge_whole_copy_recovery_payload,
    normalize_payload as _normalize_whole_copy_payload,
    output_limit as _whole_copy_output_limit,
    system_instructions as _whole_copy_system_instructions,
    whole_copy_schema as _whole_copy_schema,
)

logger = logging.getLogger(__name__)

_PROMPT_VERSION = _EVIDENCE_GRAPH_PROMPT_VERSION
_V14_PROMPT_VERSION = "pcr-full-document-visual-v14"
_V15_PROMPT_VERSION = _EVIDENCE_GRAPH_V15_PROMPT_VERSION
_SUPPORTED_PROMPT_VERSIONS = {
    _PROMPT_VERSION,
    _V14_PROMPT_VERSION,
    _V15_PROMPT_VERSION,
    _V16_PROMPT_VERSION,
}
_V14_MAX_OUTPUT_TOKENS = 4_000
_V14_MAX_QUESTIONS_PER_BATCH = 4
_BOUNDED_GRADE_MAX_OUTPUT_TOKENS = 16_000
_RUNS_COLLECTION = "evalpen_document_grading_runs"
_PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
_CALLER_ID = "pcr_eval_core"
_DEFAULT_REASONING_EFFORT = "medium"
_MAX_PAGE_COUNT = 50
_MAX_STATIC_PDF_BYTES = 45 * 1024 * 1024
_MAX_REQUEST_PAYLOAD_BYTES = 45 * 1024 * 1024
_A4_WIDTH_MM = 210.0
_A4_HEIGHT_MM = 297.0


def _visual_method(prompt_version: str) -> str:
    if prompt_version == _V16_PROMPT_VERSION:
        return "whole_copy_visual"
    if prompt_version == _V15_PROMPT_VERSION:
        return "evidence_first_visual_v15"
    if prompt_version == _V14_PROMPT_VERSION:
        return "evidence_first_visual_v14"
    return "evidence_first_visual_v13"


def _bounded_mapping_reasoning_effort(value: str) -> str:
    """Keep compact ownership calls cheap and predictable."""

    normalized = str(value or "").strip().lower()
    return "minimal" if normalized not in {"none", "minimal"} else normalized


def _mapping_reasoning_effort(value: str, *, prompt_version: str) -> str:
    """Use enough reasoning for multilingual visual ownership in v15.

    v14 remains reproducible.  v15 deliberately sets a medium floor because
    mapping jumbled handwriting is the semantic decision that gates every
    later mark; saving tokens here caused entire copies to become ungradeable.
    """

    if prompt_version != _V15_PROMPT_VERSION:
        return _bounded_mapping_reasoning_effort(value)
    normalized = str(value or "").strip().lower()
    return "high" if normalized == "high" else "medium"


def _bounded_grading_output_limit(
    questions: Sequence[Mapping[str, Any]],
    *,
    reasoning_effort: str,
    verification: bool = False,
) -> int:
    """Size the response budget from the locked structured-output contract.

    ``max_output_tokens`` includes reasoning tokens for reasoning models. A
    fixed 4k ceiling therefore truncates legitimate five-criterion answers
    even when their visible JSON is compact. The allowance is a ceiling, not
    prepaid usage; billing remains based on tokens actually produced.
    """

    question_count = max(1, len(questions))
    criterion_count = sum(
        len(_criteria(dict(question)))
        for question in questions
        if isinstance(question, Mapping)
    )
    normalized_effort = str(reasoning_effort or "").strip().lower()
    reasoning_reserve = {
        "none": 500,
        "minimal": 1_000,
        "low": 2_000,
        "medium": 4_000,
        "high": 6_000,
    }.get(normalized_effort, 4_000)
    per_criterion = 350 if verification else 550
    visible_contract_budget = (
        1_000
        + 400 * question_count
        + per_criterion * max(1, criterion_count)
    )
    required = reasoning_reserve + visible_contract_budget
    rounded = int(math.ceil(required / 1_000.0) * 1_000)
    return min(
        _BOUNDED_GRADE_MAX_OUTPUT_TOKENS,
        max(_V14_MAX_OUTPUT_TOKENS, rounded),
    )


class FullDocumentGateProtocol(Protocol):
    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        **kwargs: Any,
    ) -> Any: ...


class FullDocumentGradingError(RuntimeError):
    """Raised when the primary document request cannot be completed safely."""


class UnsupportedGradingContractError(FullDocumentGradingError):
    """Raised when an exam contract requires an explicit cohort migration."""

    retryable = False


class GradingRunIdentityError(FullDocumentGradingError):
    """Raised for a deterministic grading-run ownership or identity conflict."""

    retryable = False


class StructuredGradingOutputError(FullDocumentGradingError):
    """Terminal provider-output failure that must not burn an identical retry."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        completion_status: str = "",
        incomplete_reason: str = "",
        max_output_tokens: int = 0,
        token_usage: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.structured_output_failure = {
            "completion_status": completion_status or "unknown",
            "incomplete_reason": incomplete_reason,
            "max_output_tokens": max(0, int(max_output_tokens or 0)),
        }
        self.token_usage = dict(token_usage or {})


@dataclass
class FullDocumentGradingResult:
    handled: bool
    submission_id: str
    status: str = "not_applicable"
    skipped_reason: Optional[str] = None
    page_count: int = 0
    response_count: int = 0
    evaluated_count: int = 0
    blocked_count: int = 0
    warning_count: int = 0
    run_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    document_review_required: bool = False
    review_state: str = "not_applicable"
    review_reasons: List[str] = field(default_factory=list)


@dataclass
class _DocumentReview:
    all_student_work_accounted: bool
    confidence: float
    warnings: List[str]
    required: bool

    def as_dict(self, *, run_id: str, prompt_version: str) -> Dict[str, Any]:
        return {
            "status": "pending_review" if self.required else "verified",
            "required": self.required,
            "all_student_work_accounted": self.all_student_work_accounted,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "grading_run_id": run_id,
            "prompt_version": prompt_version,
            "updated_at": datetime.now(timezone.utc),
        }


@dataclass
class _ValidatedGrade:
    question: Dict[str, Any]
    question_number: int
    attempt_status: str
    confidence: float
    student_answer: str
    content_type: str
    source_pages: List[Dict[str, Any]]
    method_analysis: Dict[str, Any]
    criterion_marks: List[Dict[str, Any]]
    total_score: Optional[float]
    overall_feedback: str
    manual_review_required: bool
    review_reason: str
    validation_errors: List[str] = field(default_factory=list)


class FullDocumentGradingService:
    """Grade one immutable camera/PDF submission as a complete visual document."""

    def __init__(
        self,
        tenant_db: Any,
        gate: FullDocumentGateProtocol,
        *,
        model_id: Optional[str] = None,
        response_repo: Optional[DetectedResponseRepository] = None,
        evaluation_repo: Optional[EvaluationRepository] = None,
    ) -> None:
        self._db = tenant_db
        self._gate = gate
        self._model_id = (
            model_id
            or os.getenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "gpt-5.1").strip()
        )
        self._responses = response_repo or DetectedResponseRepository(tenant_db)
        self._evaluations = evaluation_repo or EvaluationRepository(tenant_db)

    async def grade_submission(
        self,
        submission_id: str,
    ) -> FullDocumentGradingResult:
        """Run or resume the full-document grading materialization."""
        submission = await self._db["evalpen_submissions"].find_one(
            {"submission_id": submission_id}
        )
        if submission is None:
            raise FullDocumentGradingError("Canonical submission was not found")

        source = str(submission.get("source") or "camera").lower()
        if source not in {"camera", "pdf", "scan"}:
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason=f"Submission source {source or 'unknown'} is not visual",
            )
        exam_id = str(submission.get("exam_id") or "")
        student_id = str(submission.get("student_id") or "")
        exam = await self._db["exampen_exams"].find_one({"exam_id": exam_id})
        if not exam or str(exam.get("exam_type") or "") != "pcr":
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason="Submission is not attached to a PCR exam",
            )
        paper_version = await self._db["exampen_paper_versions"].find_one(
            {"paper_version_id": exam.get("paper_version_id")}
        )
        canonical_visual_required = _paper_requires_canonical_visual(paper_version)
        if not _feature_enabled():
            if canonical_visual_required:
                raise FullDocumentGradingError(
                    "This exam is locked to canonical full-document visual grading, "
                    "but that worker capability is disabled"
                )
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason="Full-document visual grading is disabled for a legacy exam",
            )
        processing_job = await self._db["exampen_processing_jobs"].find_one(
            {"submission_id": submission_id, "exam_id": exam_id}
        )
        grading_contract, contract_scope = effective_grading_contract(
            exam.get("pcr_grading_contract"),
            processing_job,
        )
        contract_override = selected_copy_contract_override(processing_job)
        contract_override_id = str(contract_override.get("override_id") or "")
        source_prompt_version = str(
            contract_override.get("source_prompt_version") or ""
        )
        contract_version = str(grading_contract.get("prompt_version") or "").strip()
        # A frozen cohort selects its pipeline explicitly.  Existing exams with
        # no contract remain on v13; v14 is opt-in through migration and can
        # therefore never silently change a live v13 cohort.
        prompt_version = contract_version or _PROMPT_VERSION
        if contract_version and contract_version not in _SUPPORTED_PROMPT_VERSIONS:
            raise UnsupportedGradingContractError(
                "This exam is locked to grading contract "
                f"{contract_version}, which this worker does not support. "
                "Do not mix grading contracts within one exam; migrate and reprocess "
                "the complete exam together."
            )
        model_id = str(
            grading_contract.get("model_id") or self._model_id
        ).strip()
        temperature = _contract_temperature(grading_contract)
        reasoning_effort = str(
            grading_contract.get("reasoning_effort") or _DEFAULT_REASONING_EFFORT
        ).strip().lower()
        if not _is_openai_visual_model(model_id):
            if canonical_visual_required:
                raise FullDocumentGradingError(
                    "This exam is locked to canonical full-document visual grading, "
                    f"but worker model {model_id or 'unknown'} is not compatible"
                )
            logger.info(
                "Full-document grading skipped for non-OpenAI model %s",
                model_id,
            )
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason=f"Worker model {model_id or 'unknown'} is not visual",
            )

        questions = await self._db["evalpen_questions"].find(
            {"exam_id": exam_id}
        ).sort("question_number", 1).to_list(length=2000)
        questions = [q for q in questions if str(q.get("question_id") or "")]
        if not questions:
            raise FullDocumentGradingError("Immutable PCR question catalog is empty")
        catalog_errors = _validate_question_catalog(questions)
        if catalog_errors:
            raise FullDocumentGradingError(
                "Immutable PCR question catalog is invalid: "
                + "; ".join(catalog_errors[:10])
            )
        if temperature is None:
            temperature = _grading_temperature(questions)
        if reasoning_effort not in {"none", "minimal", "low", "medium", "high"}:
            raise FullDocumentGradingError(
                "Immutable PCR grading contract has an unsupported reasoning effort"
            )
        answer_pages = await self._db["evalpen_answer_pages"].find(
            {"submission_id": submission_id}
        ).sort("page_number", 1).to_list(length=_MAX_PAGE_COUNT + 1)
        if not answer_pages:
            raise FullDocumentGradingError("Canonical student answer pages are missing")
        if len(answer_pages) > _MAX_PAGE_COUNT:
            raise FullDocumentGradingError(
                f"Student copy has {len(answer_pages)} pages; maximum is {_MAX_PAGE_COUNT}"
            )

        paper_assets = dict((paper_version or {}).get("paper_assets") or {})
        paper_asset = dict(paper_assets.get("question_paper") or {})
        solution_asset = dict(paper_assets.get("teacher_solution") or {})
        document_id = str(
            exam.get("prepared_document_id")
            or (paper_version or {}).get("document_id")
            or ""
        )

        if paper_asset:
            # A modern PCR session must be independent from the mutable
            # authoring document and from the API worker's filesystem. The
            # snapshot pins a content-addressed private object and its hash.
            from services.exampen_paper_service import load_canonical_paper_asset

            paper_bytes = await load_canonical_paper_asset(paper_asset)
            solution_bytes = (
                await load_canonical_paper_asset(solution_asset)
                if solution_asset
                else None
            )
            document = {
                "document_id": document_id,
                "filename": paper_asset.get("filename") or "question-paper.pdf",
                "answer_sheet_filename": (
                    solution_asset.get("filename") or "teacher-solution.pdf"
                ),
            }
        else:
            if canonical_visual_required:
                from services.exampen_paper_service import CanonicalPaperAssetError

                raise CanonicalPaperAssetError(
                    "The exam requires canonical visual grading, but its immutable "
                    "paper asset manifest is unavailable"
                )

            # Legacy sessions predate a frozen object-store asset. Keep their
            # old review-safe behaviour for compatibility only; new finalised
            # PCR papers never take this branch.
            document = await self._db["documents"].find_one(
                {"document_id": document_id}
            )
            if not document:
                return FullDocumentGradingResult(
                    handled=False,
                    submission_id=submission_id,
                    skipped_reason="Legacy exam has no immutable question-paper record",
                )
            paper_bytes = await _read_canonical_file(
                str(document.get("file_path") or ""),
                expected_sha256=document.get("sha256"),
            )
            if not paper_bytes:
                return FullDocumentGradingResult(
                    handled=False,
                    submission_id=submission_id,
                    skipped_reason="Legacy question-paper asset could not be loaded",
                )
            solution_bytes = await _read_canonical_file(
                str(document.get("answer_sheet_path") or ""),
                expected_sha256=document.get("answer_sheet_sha256"),
            )
        if len(paper_bytes) + len(solution_bytes or b"") > _MAX_STATIC_PDF_BYTES:
            raise FullDocumentGradingError(
                "Question paper and teacher solution exceed the document-input size limit"
            )
        paper_file_hash = hashlib.sha256(paper_bytes).hexdigest()
        solution_file_hash = (
            hashlib.sha256(solution_bytes).hexdigest() if solution_bytes else None
        )

        generation_revision = await _materialization_revision(
            self._db,
            submission_id,
        )
        prior_revision_run = await self._db[_RUNS_COLLECTION].find_one(
            {
                "submission_id": submission_id,
                "prompt_version": prompt_version,
                "$or": [
                    {"generation_revision": generation_revision},
                    {"grading_revision": generation_revision},
                ],
            },
            sort=[("updated_at", -1), ("created_at", -1)],
        )
        if prior_revision_run:
            # Resume the exact technical run that already owns this revision.
            # This also remains stable when the first provider response froze a
            # dated model snapshot for subsequent students in the cohort.
            run_id = str(prior_revision_run.get("run_id") or "")
            model_id = str(
                prior_revision_run.get("requested_model_id") or model_id
            )
            if not run_id:
                raise GradingRunIdentityError(
                    "Saved submission grading run is missing its immutable identity"
                )
        input_fingerprint = _input_fingerprint(
            submission_id=submission_id,
            exam=exam,
            answer_pages=answer_pages,
            questions=questions,
            model_id=model_id,
            paper_hash=paper_file_hash,
            solution_hash=solution_file_hash,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
        )
        generation_fingerprint = _generation_fingerprint(
            submission_id=submission_id,
            input_fingerprint=input_fingerprint,
            generation_revision=generation_revision,
        )
        if prior_revision_run:
            _assert_run_identity(
                prior_revision_run,
                submission_id=submission_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                generation_revision=generation_revision,
                allow_legacy_generation_fingerprint=True,
            )
        else:
            run_id = f"DOCGR-{generation_fingerprint[:24]}"
        materialization_id = f"{run_id}:r{generation_revision}"
        await self._db[_RUNS_COLLECTION].create_index(
            "run_id", unique=True, name="uniq_document_grading_run"
        )
        existing_run = await self._db[_RUNS_COLLECTION].find_one({"run_id": run_id})
        if existing_run:
            _assert_run_identity(
                existing_run,
                submission_id=submission_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                generation_revision=generation_revision,
                allow_legacy_generation_fingerprint=True,
            )
        resumed_grading_run = False
        if existing_run and existing_run.get("status") == "completed":
            active_count = await self._db["evalpen_detected_responses"].count_documents(
                {
                    "submission_id": submission_id,
                    "mapping_version_id": materialization_id,
                    "superseded_at": {"$exists": False},
                }
            )
            if active_count == len(questions):
                return _result_from_run(existing_run, submission_id)

        generation_lease_token: Optional[str] = None
        if not existing_run or existing_run.get("status") not in {
            "validated",
            "materializing",
            "completed",
        }:
            existing_run, generation_lease_token = await _claim_or_wait_for_run(
                self._db,
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                submission_id=submission_id,
                student_id=student_id,
                exam_id=exam_id,
                generation_revision=generation_revision,
                requested_model_id=model_id,
                page_count=len(answer_pages),
                prompt_version=prompt_version,
                contract_scope=contract_scope,
                contract_override_id=contract_override_id or None,
                source_prompt_version=source_prompt_version or None,
            )

        if existing_run and existing_run.get("status") in {
            "validated",
            "materializing",
            "completed",
        }:
            resumed_grading_run = True
            raw_payload = existing_run.get("validated_payload")
            if not isinstance(raw_payload, dict):
                raise FullDocumentGradingError("Saved document grading ledger is invalid")
            usage = dict(existing_run.get("token_usage") or {})
            raw_llm = str(existing_run.get("raw_llm_response") or "")
        else:
            if not generation_lease_token:
                raise FullDocumentGradingError(
                    "The submission grading run could not acquire generation ownership"
                )
            if prompt_version in {_V15_PROMPT_VERSION, _V16_PROMPT_VERSION}:
                student_content, student_image_bytes = await _student_copy_content(
                    answer_pages,
                    orientation_recovery=True,
                    coordinate_evidence=(prompt_version != _V16_PROMPT_VERSION),
                )
            else:
                student_content, student_image_bytes = await _student_copy_content(
                    answer_pages
                )
            if (
                len(paper_bytes)
                + len(solution_bytes or b"")
                + student_image_bytes
                > _MAX_REQUEST_PAYLOAD_BYTES
            ):
                raise FullDocumentGradingError(
                    "Paper, solution, and original student pages exceed the visual "
                    "request size limit"
                )
            try:
                common_generation_args = {
                    "gate": self._gate,
                    "run_id": run_id,
                    "submission_id": submission_id,
                    "exam_id": exam_id,
                    "questions": questions,
                    "page_count": len(answer_pages),
                    "paper_bytes": paper_bytes,
                    "solution_bytes": solution_bytes,
                    "student_content": student_content,
                    "paper_filename": str(
                        document.get("filename") or "question-paper.pdf"
                    ),
                    "solution_filename": str(
                        document.get("answer_sheet_filename")
                        or "teacher-solution.pdf"
                    ),
                    "model_id": model_id,
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature,
                    "paper_hash": paper_file_hash,
                    "solution_hash": solution_file_hash,
                }
                if prompt_version == _V16_PROMPT_VERSION:
                    raw_payload, raw_llm, usage = await _run_whole_copy_grading(
                        **common_generation_args,
                    )
                else:
                    raw_payload, raw_llm, usage = await _run_evidence_first_grading(
                        db=self._db,
                        existing_run=existing_run,
                        generation_lease_token=generation_lease_token,
                        pipeline_version=prompt_version,
                        **common_generation_args,
                    )
            except Exception as exc:
                failure_update: Dict[str, Any] = {
                    "status": "failed",
                    "generation_error": str(exc)[:500],
                    "updated_at": datetime.now(timezone.utc),
                }
                structured_failure = getattr(exc, "structured_output_failure", None)
                if isinstance(structured_failure, dict):
                    failure_update["structured_output_failure"] = structured_failure
                failure_usage = getattr(exc, "token_usage", None)
                if isinstance(failure_usage, Mapping) and failure_usage:
                    failure_update["token_usage"] = dict(failure_usage)
                await self._db[_RUNS_COLLECTION].update_one(
                    {
                        "run_id": run_id,
                        "generation_lease_token": generation_lease_token,
                    },
                    {
                        "$set": failure_update,
                        "$unset": {
                            "generation_lease_token": "",
                            "generation_lease_expires_at": "",
                        },
                    },
                )
                if getattr(exc, "retryable", True) is False:
                    raise
                raise FullDocumentGradingError(
                    f"Full-document model request failed: {str(exc)[:400]}"
                ) from exc

            if contract_scope != SELECTED_COPY_CONTRACT_SCOPE:
                try:
                    await _freeze_exam_grading_contract(
                        self._db,
                        exam_id=exam_id,
                        model_id=str(usage.get("model") or model_id),
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        prompt_version=prompt_version,
                    )
                except Exception as exc:
                    await self._db[_RUNS_COLLECTION].update_one(
                        {
                            "run_id": run_id,
                            "generation_lease_token": generation_lease_token,
                        },
                        {
                            "$set": {
                                "status": "failed",
                                "generation_error": str(exc)[:500],
                                "updated_at": datetime.now(timezone.utc),
                            },
                            "$unset": {
                                "generation_lease_token": "",
                                "generation_lease_expires_at": "",
                            },
                        },
                    )
                    raise
            now = datetime.now(timezone.utc)
            saved_run = await self._db[_RUNS_COLLECTION].update_one(
                {
                    "run_id": run_id,
                    "generation_lease_token": generation_lease_token,
                },
                {
                    "$set": {
                        "status": "validated",
                        "prompt_version": prompt_version,
                        "contract_scope": contract_scope,
                        "contract_override_id": contract_override_id or None,
                        "source_prompt_version": source_prompt_version or None,
                        "model_used": usage.get("model") or model_id,
                        "validated_payload": raw_payload,
                        "raw_llm_response": raw_llm,
                        "token_usage": usage,
                        "updated_at": now,
                    },
                    "$unset": {
                        "generation_lease_token": "",
                        "generation_lease_expires_at": "",
                        "generation_error": "",
                    },
                },
            )
            if saved_run.matched_count != 1:
                raise FullDocumentGradingError(
                    "Submission grading ownership expired before the ledger was saved"
                )

        grades, document_errors, document_review = _validate_ledger(
            raw_payload,
            questions=questions,
            page_count=len(answer_pages),
        )
        await self._db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "materializing",
                    "validation_errors": document_errors,
                    "document_review": document_review.as_dict(
                        run_id=run_id,
                        prompt_version=prompt_version,
                    ),
                    "updated_at": datetime.now(timezone.utc),
                },
            },
        )

        result = await self._materialize(
            run_id=run_id,
            materialization_id=materialization_id,
            submission=submission,
            questions=questions,
            grades=grades,
            raw_payload=raw_payload,
            usage=usage,
            page_count=len(answer_pages),
            document_errors=document_errors,
            document_review=document_review,
            resumed_grading_run=resumed_grading_run,
            grading_input_hash=input_fingerprint,
            prompt_version=prompt_version,
        )
        await self._db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "result": {
                        "response_count": result.response_count,
                        "evaluated_count": result.evaluated_count,
                        "blocked_count": result.blocked_count,
                        "warning_count": result.warning_count,
                        "errors": result.errors,
                        "document_review_required": result.document_review_required,
                        "review_state": result.review_state,
                        "review_reasons": result.review_reasons,
                    },
                    "completed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        return result

    async def _materialize(
        self,
        *,
        run_id: str,
        materialization_id: str,
        submission: Dict[str, Any],
        questions: List[Dict[str, Any]],
        grades: List[_ValidatedGrade],
        raw_payload: Dict[str, Any],
        usage: Dict[str, Any],
        page_count: int,
        document_errors: List[str],
        document_review: _DocumentReview,
        resumed_grading_run: bool,
        grading_input_hash: str,
        prompt_version: str,
    ) -> FullDocumentGradingResult:
        submission_id = str(submission.get("submission_id") or "")
        exam_id = str(submission.get("exam_id") or "")
        student_id = str(submission.get("student_id") or "")
        model_used = str(usage.get("model") or self._model_id)
        response_docs: List[Dict[str, Any]] = []
        evaluation_docs: List[Dict[str, Any]] = []

        raw_by_number = {
            int(item.get("question_number")): item
            for item in (raw_payload.get("questions") or [])
            if isinstance(item, dict) and _positive_int(item.get("question_number"))
        }
        for grade in grades:
            question_id = str(grade.question.get("question_id") or "")
            raw_question_result = raw_by_number.get(grade.question_number, {})
            visual_evidence = {
                "method": _visual_method(prompt_version),
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "source_page_numbers": sorted(
                    {
                        int(region.get("page_number") or 0)
                        for region in grade.source_pages
                        if int(region.get("page_number") or 0) > 0
                    }
                ),
                "region_ids": [
                    str(region.get("region_id") or "")
                    for region in grade.source_pages
                    if str(region.get("region_id") or "")
                ],
            }
            semantic_evidence_signature = _semantic_evidence_signature(
                question_id=question_id,
                student_answer=grade.student_answer,
                source_pages=grade.source_pages,
                visual_evidence=visual_evidence,
                prompt_version=prompt_version,
            )
            grading_consistency_key = _grading_consistency_key(
                question_id=question_id,
                student_answer=grade.student_answer,
                method_analysis=grade.method_analysis,
                prompt_version=prompt_version,
                model_used=model_used,
            )
            # The key remains an audit/calibration dimension only. A different
            # student's score must never overwrite this student's evidence-based
            # criterion decisions, even when model transcriptions happen to match.
            consistency_calibration: Optional[Dict[str, Any]] = None
            response_id = _stable_id(
                "RESP-DOC", submission_id, materialization_id, question_id
            )
            unresolved = grade.attempt_status == "unresolved"
            is_missing = grade.attempt_status == "not_attempted"
            objective_result: Optional[Dict[str, Any]] = None
            if _is_objective_question(grade.question) and not unresolved:
                try:
                    objective_result = score_objective_response(
                        grade.question,
                        grade.student_answer,
                    )
                except ObjectiveScoringContractError:
                    # Readiness and grade validation already guard this path.
                    # Keep persistence fail-closed if an immutable record is
                    # nevertheless inconsistent.
                    unresolved = True
            flags: List[Dict[str, Any]] = []
            if unresolved:
                flags.append(
                    _review_flag(
                        response_id,
                        severity="blocking",
                        reason=grade.review_reason,
                        prompt_version=prompt_version,
                    )
                )
            elif grade.manual_review_required:
                flags.append(
                    _review_flag(
                        response_id,
                        severity="warning",
                        reason=grade.review_reason,
                        prompt_version=prompt_version,
                    )
                )

            response_doc = {
                "response_id": response_id,
                "submission_id": submission_id,
                "question_id": question_id,
                "question_number": grade.question_number,
                "sub_part": None,
                "question_assignment": {
                    "method": (
                        "full_document_visual"
                        if prompt_version == _V16_PROMPT_VERSION
                        else _visual_method(prompt_version)
                    ),
                    "confidence": grade.confidence,
                    "prompt_version": prompt_version,
                    "model_used": model_used,
                    "grading_run_id": run_id,
                    "materialization_id": materialization_id,
                    "manual_review_required": grade.manual_review_required or unresolved,
                    "reason": grade.review_reason or None,
                    "method_analysis": grade.method_analysis,
                    "absence_proof": (
                        {
                            "verified": True,
                            "method": "full_document_visual_coverage",
                            "confidence": document_review.confidence,
                            "grading_run_id": run_id,
                        }
                        if is_missing
                        else None
                    ),
                },
                "exam_id": exam_id,
                "student_id": student_id,
                "detected_text": grade.student_answer,
                "source_pages": grade.source_pages,
                "visual_evidence": visual_evidence,
                "semantic_evidence_signature": semantic_evidence_signature,
                "grading_consistency_key": grading_consistency_key or None,
                "consistency_calibration": consistency_calibration,
                "evidence_version": 4,
                "evidence_atom_ids": [
                    str(item.get("region_id"))
                    or _stable_id(
                        "region",
                        submission_id,
                        str(item["page_number"]),
                        str(item.get("x_start", "")),
                        str(item["y_start"]),
                        str(item.get("x_end", "")),
                        str(item["y_end"]),
                    )
                    for item in grade.source_pages
                ],
                "content_type": grade.content_type,
                "text_coverage_ratio": _text_coverage_for_type(grade.content_type),
                "segmentation_confidence": grade.confidence,
                "ocr_confidence": None,
                "flags": flags,
                "word_count": len(grade.student_answer.split()),
                "is_continuation": any(
                    str(item.get("continuation_group") or "")
                    for item in grade.source_pages
                ),
                "is_missing_response": is_missing,
                "absence_proven": is_missing,
                "manual_review_required": grade.manual_review_required or unresolved,
                "manual_review_reason": grade.review_reason or None,
                "answer_state": (
                    "unresolved" if unresolved else "not_attempted" if is_missing else "detected"
                ),
                "grading_mode": (
                    "objective"
                    if _is_objective_question(grade.question)
                    else "subjective"
                ),
                "objective_result": objective_result,
                "eval_status": "blocked" if unresolved else "pending",
                "mapping_version_id": materialization_id,
                "_immutable": True,
                "created_at": datetime.now(timezone.utc),
            }
            response_docs.append(response_doc)

            if unresolved or grade.total_score is None:
                continue
            max_marks = _max_marks(grade.question)
            eval_id = _stable_id(
                "EVAL-DOC", submission_id, materialization_id, question_id
            )
            evaluation_docs.append(
                {
                    "evaluation_id": eval_id,
                    "evaluation_input_version": 2,
                    "mapping_version_id": materialization_id,
                    "response_id": response_id,
                    "question_id": question_id,
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "prompt_version": prompt_version,
                    "visual_evidence": visual_evidence,
                    "semantic_evidence_signature": semantic_evidence_signature,
                    "grading_consistency_key": grading_consistency_key or None,
                    "consistency_calibration": consistency_calibration,
                    "eval_path": (
                        "full_document_visual_not_attempted"
                        if is_missing
                        else (
                            "full_document_visual_objective"
                            if objective_result is not None
                            else "full_document_visual"
                        )
                    ),
                    "model_used": model_used,
                    "total_score": grade.total_score,
                    "max_score": max_marks,
                    "scoreable_max": max_marks,
                    "marking_policy": dict(grade.question.get("marking_policy") or {}),
                    "method_policy": _question_method_policy(grade.question),
                    "method_analysis": grade.method_analysis,
                    "manual_review_required": grade.manual_review_required,
                    "step_marks": [
                        {
                            "step": item["description"],
                            "marks_awarded": item["marks_awarded"],
                            "max_marks": item["max_marks"],
                            "rationale": item["rationale"],
                        }
                        for item in grade.criterion_marks
                    ],
                    "criterion_marks": grade.criterion_marks,
                    "overall_feedback": grade.overall_feedback,
                    "grading_mode": (
                        "objective"
                        if objective_result is not None
                        else "subjective"
                    ),
                    "objective_result": objective_result,
                    "reference_solution": _reference_solution(grade.question),
                    "token_usage": {
                        "document_call_id": run_id,
                        "model": model_used,
                        "caller": usage.get("caller") or _CALLER_ID,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "document_call_total_tokens": usage.get("total_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_tokens", 0),
                    },
                    "raw_llm_response": json.dumps(
                        raw_question_result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "eval_flags": (
                        [
                            {
                                "flag_type": "llm_score_divergence",
                                "severity": "warning",
                                "reason": grade.review_reason,
                            }
                        ]
                        if grade.manual_review_required
                        else []
                    ),
                    "audit_trail": [
                        {
                            "actor_id": "system",
                            "timestamp": datetime.now(timezone.utc),
                            "action": "evaluation_created",
                            "before": None,
                            "after": {
                                "total_score": grade.total_score,
                                "max_score": max_marks,
                                "eval_path": "full_document_visual",
                                "model_used": model_used,
                                "grading_run_id": run_id,
                                "manual_review_required": grade.manual_review_required,
                            },
                            "reason": (
                                "Full-document visual evaluation against immutable paper "
                                "and teacher solution"
                            ),
                        }
                    ],
                    "created_at": datetime.now(timezone.utc),
                }
            )

        await self._responses.insert_responses_bulk(response_docs)
        for evaluation_doc in evaluation_docs:
            await self._evaluations.insert_evaluation(evaluation_doc)

        status_by_response = {
            doc["response_id"]: (
                "blocked"
                if doc.get("answer_state") == "unresolved"
                else "manual_review"
                if doc.get("manual_review_required")
                else "evaluated"
            )
            for doc in response_docs
        }
        for response_id, eval_status in status_by_response.items():
            await self._responses.update_eval_status(response_id, eval_status)
        await self._responses.supersede_responses_for_submission(
            submission_id,
            keep_response_ids=[doc["response_id"] for doc in response_docs],
            reason="full_document_visual_grading",
        )
        blocked = sum(1 for grade in grades if grade.attempt_status == "unresolved")
        question_warnings = sum(
            1
            for grade in grades
            if grade.attempt_status != "unresolved" and grade.manual_review_required
        )
        warnings = question_warnings + int(document_review.required)
        review_state = (
            "blocked"
            if blocked
            else "needs_review"
            if document_review.required
            else "ready"
        )
        await self._db["evalpen_submissions"].update_one(
            {"submission_id": submission_id},
            {
                "$set": {
                    "segmentation_status": "complete",
                    "processing_path": "full_document_visual",
                    "document_grading_run_id": run_id,
                    "document_grading_materialization_id": materialization_id,
                    "grading_input_hash": grading_input_hash,
                    "resumed_grading_run": resumed_grading_run,
                    "document_review": document_review.as_dict(
                        run_id=run_id,
                        prompt_version=prompt_version,
                    ),
                    "review_state": review_state,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$unset": {"reused_grading_input": ""},
            },
        )

        evaluated = len(evaluation_docs)
        errors = list(document_errors)
        review_reasons = list(document_review.warnings)
        review_reasons.extend(
            f"Q{grade.question_number}: {grade.review_reason}"
            for grade in grades
            if grade.review_reason
            and (grade.attempt_status == "unresolved" or grade.manual_review_required)
        )
        return FullDocumentGradingResult(
            handled=True,
            submission_id=submission_id,
            status=(
                "blocked_for_review"
                if review_state == "blocked"
                else "completed"
            ),
            page_count=page_count,
            response_count=len(response_docs),
            evaluated_count=evaluated,
            blocked_count=blocked,
            warning_count=warnings,
            run_id=run_id,
            errors=errors,
            document_review_required=document_review.required,
            review_state=review_state,
            review_reasons=list(dict.fromkeys(review_reasons)),
        )


async def _claim_or_wait_for_run(
    tenant_db: Any,
    *,
    run_id: str,
    input_fingerprint: str,
    generation_fingerprint: str,
    submission_id: str,
    student_id: str,
    exam_id: str,
    generation_revision: int,
    requested_model_id: str,
    page_count: int,
    prompt_version: str,
    contract_scope: str = "exam",
    contract_override_id: Optional[str] = None,
    source_prompt_version: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Single-flight technical retries for one submission grading generation.

    ``run_id`` is submission- and generation-scoped. Another student's upload,
    even when its bytes are identical, therefore cannot join or reuse this
    run. Automatic worker retries keep the same generation; an explicit
    operator reprocess increments it and intentionally creates a fresh model
    interpretation. The lease prevents duplicate paid calls when workers race
    on the same immutable generation.
    """

    now = datetime.now(timezone.utc)
    lease_token = uuid.uuid4().hex
    lease_expires_at = now + timedelta(minutes=15)
    collection = tenant_db[_RUNS_COLLECTION]
    existing = await collection.find_one({"run_id": run_id})
    if existing is not None:
        _assert_run_identity(
            existing,
            submission_id=submission_id,
            input_fingerprint=input_fingerprint,
            generation_fingerprint=generation_fingerprint,
            generation_revision=generation_revision,
            allow_legacy_generation_fingerprint=True,
        )

    if existing is None:
        try:
            claimed = await collection.update_one(
                {"run_id": run_id},
                {
                    "$setOnInsert": {
                        "run_id": run_id,
                        "submission_id": submission_id,
                        "student_id": student_id,
                        "exam_id": exam_id,
                        "grading_revision": generation_revision,
                        "generation_revision": generation_revision,
                        "prompt_version": prompt_version,
                        "contract_scope": contract_scope,
                        "contract_override_id": contract_override_id,
                        "source_prompt_version": source_prompt_version,
                        "requested_model_id": requested_model_id,
                        "input_fingerprint": input_fingerprint,
                        "generation_fingerprint": generation_fingerprint,
                        "page_count": page_count,
                        "status": "generating",
                        "generation_lease_token": lease_token,
                        "generation_lease_expires_at": lease_expires_at,
                        "created_at": now,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )
            if claimed.upserted_id is not None:
                return None, lease_token
        except DuplicateKeyError:
            # Another worker won the unique run reservation after our initial
            # read. Join its single-flight wait instead of failing the copy.
            pass
    else:
        reclaimed = await collection.update_one(
            {
                "run_id": run_id,
                "$or": [
                    {"status": "failed"},
                    {
                        "status": "generating",
                        "generation_lease_expires_at": {"$lte": now},
                    },
                ],
            },
            {
                "$set": {
                    "status": "generating",
                    "grading_revision": generation_revision,
                    "generation_revision": generation_revision,
                    "generation_fingerprint": generation_fingerprint,
                    "generation_lease_token": lease_token,
                    "generation_lease_expires_at": lease_expires_at,
                    "generation_error": None,
                    "updated_at": now,
                },
            },
        )
        if reclaimed.matched_count == 1:
            return None, lease_token

    try:
        configured_wait = float(
            os.getenv("PCR_GRADING_SINGLEFLIGHT_WAIT_SECONDS", "120") or 120
        )
    except (TypeError, ValueError):
        configured_wait = 120.0
    wait_seconds = max(5.0, min(180.0, configured_wait))
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        existing = await collection.find_one({"run_id": run_id})
        if existing is not None:
            _assert_run_identity(
                existing,
                submission_id=submission_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                generation_revision=generation_revision,
                allow_legacy_generation_fingerprint=True,
            )
        if existing and existing.get("status") in {
            "validated",
            "materializing",
            "completed",
        }:
            return existing, None
        if existing and existing.get("status") == "failed":
            # Retry through the normal claim path instead of starting an
            # uncoordinated second model request.
            return await _claim_or_wait_for_run(
                tenant_db,
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                submission_id=submission_id,
                student_id=student_id,
                exam_id=exam_id,
                generation_revision=generation_revision,
                requested_model_id=requested_model_id,
                page_count=page_count,
                prompt_version=prompt_version,
            )
        if asyncio.get_running_loop().time() >= deadline:
            raise FullDocumentGradingError(
                "This submission revision is already being graded; retry after its "
                "current run finishes"
            )
        await asyncio.sleep(0.5)


def _run_generation_revision(run: Mapping[str, Any]) -> int:
    raw_revision = run.get("generation_revision")
    if raw_revision is None:
        raw_revision = run.get("grading_revision")
    try:
        return max(0, int(raw_revision or 0))
    except (TypeError, ValueError) as exc:
        raise GradingRunIdentityError(
            "Saved submission grading run has an invalid generation revision"
        ) from exc


def _assert_run_identity(
    run: Mapping[str, Any],
    *,
    submission_id: str,
    input_fingerprint: str,
    generation_fingerprint: str,
    generation_revision: int,
    allow_legacy_generation_fingerprint: bool = False,
) -> None:
    """Fail closed before joining, reclaiming, or replaying a grading run."""

    if str(run.get("submission_id") or "") != submission_id:
        raise GradingRunIdentityError(
            "Submission grading run ownership does not match the requested generation"
        )
    saved_input_fingerprint = str(run.get("input_fingerprint") or "")
    if not saved_input_fingerprint or saved_input_fingerprint != input_fingerprint:
        raise GradingRunIdentityError(
            "Submission grading run input does not match the requested generation"
        )
    if _run_generation_revision(run) != generation_revision:
        raise GradingRunIdentityError(
            "Submission grading run revision does not match the requested generation"
        )
    saved_generation_fingerprint = str(
        run.get("generation_fingerprint") or ""
    )
    if saved_generation_fingerprint:
        if saved_generation_fingerprint != generation_fingerprint:
            raise GradingRunIdentityError(
                "Submission grading run identity does not match the requested generation"
            )
    elif not allow_legacy_generation_fingerprint:
        raise GradingRunIdentityError(
            "Saved submission grading run is missing its generation identity"
        )


async def _materialization_revision(tenant_db: Any, submission_id: str) -> int:
    """Return a retry-stable grading generation for this submission job.

    Technical retries keep the same revision. An explicit reprocess increments
    the generation and creates both a fresh model ledger and new immutable
    response/evaluation rows. Previous completed generations remain untouched.
    """

    jobs = await tenant_db[_PROCESSING_JOBS_COLLECTION].find(
        {"submission_id": submission_id}
    ).sort([("created_at", -1), ("updated_at", -1)]).to_list(length=1)
    if not jobs:
        return 0
    try:
        return max(0, int(jobs[0].get("reprocess_count") or 0))
    except (TypeError, ValueError):
        return 0


async def _freeze_exam_grading_contract(
    tenant_db: Any,
    *,
    exam_id: str,
    model_id: str,
    temperature: float,
    reasoning_effort: str,
    prompt_version: str,
) -> None:
    """Freeze one prompt/model contract for every submission in an exam.

    Provider aliases may resolve to a dated snapshot.  The first completed
    provider response records that resolved model, and later submissions use
    the same identifier even if deployment defaults change.  Concurrent first
    submissions may race, so the winner is re-read and any disagreement fails
    closed instead of silently mixing graders within one cohort.
    """

    now = datetime.now(timezone.utc)
    contract = {
        "prompt_version": prompt_version,
        "model_id": model_id,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "locked_at": now,
    }
    await tenant_db["exampen_exams"].update_one(
        {
            "exam_id": exam_id,
            "$or": [
                {"pcr_grading_contract": {"$exists": False}},
                {"pcr_grading_contract": None},
                {"pcr_grading_contract.model_id": {"$exists": False}},
            ],
        },
        {
            "$set": {
                "pcr_grading_contract": contract,
                "updated_at": now,
            }
        },
    )
    # Older finalized exams predate the sampling controls in the frozen
    # contract.  Fill only absent fields; never overwrite an established
    # cohort setting.
    await tenant_db["exampen_exams"].update_one(
        {
            "exam_id": exam_id,
            "pcr_grading_contract.prompt_version": prompt_version,
            "pcr_grading_contract.model_id": model_id,
            "pcr_grading_contract.temperature": {"$exists": False},
        },
        {
            "$set": {
                "pcr_grading_contract.temperature": temperature,
                "updated_at": now,
            }
        },
    )
    await tenant_db["exampen_exams"].update_one(
        {
            "exam_id": exam_id,
            "pcr_grading_contract.prompt_version": prompt_version,
            "pcr_grading_contract.model_id": model_id,
            "pcr_grading_contract.reasoning_effort": {"$exists": False},
        },
        {
            "$set": {
                "pcr_grading_contract.reasoning_effort": reasoning_effort,
                "updated_at": now,
            }
        },
    )
    frozen_exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"pcr_grading_contract": 1},
    )
    frozen = dict((frozen_exam or {}).get("pcr_grading_contract") or {})
    if (
        str(frozen.get("prompt_version") or "") != prompt_version
        or str(frozen.get("model_id") or "") != model_id
        or abs(_temperature(frozen.get("temperature")) - temperature) > 0.0001
        or str(frozen.get("reasoning_effort") or "") != reasoning_effort
    ):
        raise FullDocumentGradingError(
            "The exam grading contract changed while this submission was being "
            "processed. The result was not materialized; reprocess the cohort under "
            "one locked model and prompt version."
        )


def _feature_enabled() -> bool:
    return os.getenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _paper_requires_canonical_visual(
    paper_version: Optional[Dict[str, Any]],
) -> bool:
    """Return whether this immutable paper forbids the legacy OCR grader.

    Finalization records a typed capability contract on modern PCR papers.
    Once that contract exists and is ready, every camera/PDF submission in the
    cohort must use the same full-document visual path. A temporary storage,
    provider, or worker problem is retryable infrastructure failure, never
    permission to switch one student onto a different marking engine.
    """

    context = dict((paper_version or {}).get("paper_context") or {})
    return bool(
        context.get("ready")
        and str(context.get("version") or "")
        in {
            "canonical-full-document-visual-v1",
            "canonical-full-document-visual-v2",
        }
    )


def _is_openai_visual_model(model_id: str) -> bool:
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider and provider != "openai":
        return False
    normalized = model_id.strip().lower()
    return normalized.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))


async def _read_canonical_file(
    storage_path: str,
    *,
    expected_sha256: Any = None,
) -> Optional[bytes]:
    if not storage_path:
        return None
    data: Optional[bytes]
    if storage_path.startswith("s3://"):
        from utils.s3_storage import download_file

        data = await download_file(storage_path)
    else:
        backend_root = Path(__file__).resolve().parents[3]
        candidate = Path(storage_path)
        if not candidate.is_absolute():
            candidate = backend_root / candidate
        candidate = candidate.resolve(strict=False)
        allowed_roots = [(backend_root / "uploads").resolve(strict=False)]
        try:
            from config_async import settings

            allowed_roots.append(
                Path(settings.UPLOAD_PRIVATE_LOCAL_DIR).resolve(strict=False)
            )
        except Exception:
            pass
        if not any(root == candidate or root in candidate.parents for root in allowed_roots):
            logger.error("Refusing canonical PDF outside approved upload roots: %s", candidate)
            return None
        if not candidate.is_file():
            return None
        data = await asyncio.to_thread(candidate.read_bytes)
    if not data:
        return None
    expected = str(expected_sha256 or "").strip().lower()
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise AssetIntegrityError("Canonical paper asset integrity verification failed")
    return data


async def _run_whole_copy_grading(
    *,
    gate: FullDocumentGateProtocol,
    run_id: str,
    submission_id: str,
    exam_id: str,
    questions: List[Dict[str, Any]],
    page_count: int,
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
    model_id: str,
    reasoning_effort: str,
    temperature: float,
    paper_hash: str,
    solution_hash: Optional[str],
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Grade a complete copy in one call and recover at most once.

    There is deliberately no mapper, cropper, per-page grading batch, verifier,
    or output-exhaustion split in this contract.  The model owns semantic
    association and marking while the server validates identifiers and sums.
    """

    catalog = [_catalog_question(question) for question in questions]
    primary_limit = _whole_copy_output_limit(
        catalog,
        reasoning_effort=reasoning_effort,
    )
    cache_key = _stage_cache_key(
        "whole-copy-static",
        paper_hash=paper_hash,
        solution_hash=solution_hash,
        prompt_version=_V16_PROMPT_VERSION,
    )
    response: Any = None
    try:
        response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_whole_copy_responses_input(
                catalog=catalog,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=student_content,
                paper_filename=paper_filename,
                solution_filename=solution_filename,
            ),
            json_schema=_whole_copy_schema(catalog),
            prompt_cache_key=cache_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=primary_limit,
            metadata={
                "pcr_stage": "whole_copy_visual_grading",
                "prompt_version": _V16_PROMPT_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_count": len(questions),
                "page_count": page_count,
                "run_id": run_id,
                "provider_call_number": 1,
                "provider_call_limit": 2,
                "recursive_splitting": False,
            },
        )
    except Exception:
        raise

    primary_usage = _usage_dict(response, fallback_model=model_id)
    completion_failure = _response_completion_failure(response)
    if completion_failure:
        raise StructuredGradingOutputError(
            "Whole-copy grading exhausted its output budget before completing",
            completion_status=completion_failure["completion_status"],
            incomplete_reason=completion_failure["incomplete_reason"],
            max_output_tokens=primary_limit,
            token_usage=primary_usage,
        )
    primary_raw = str(getattr(response, "content", "") or "")
    primary_payload = _parse_json_object(primary_raw)
    if primary_payload is None:
        raise StructuredGradingOutputError(
            "Whole-copy grading returned invalid structured JSON",
            completion_status=str(
                getattr(response, "completion_status", "completed") or "completed"
            ),
            max_output_tokens=primary_limit,
            token_usage=primary_usage,
        )

    normalized_primary = _normalize_whole_copy_payload(primary_payload)
    primary_grades, _, primary_review = _validate_ledger(
        normalized_primary,
        questions=questions,
        page_count=page_count,
    )
    retry_numbers = {
        grade.question_number
        for grade in primary_grades
        if grade.attempt_status == "unresolved" or grade.manual_review_required
    }
    if primary_review.required:
        # The primary call saw unexplained work.  The one permitted recovery
        # must reassess the whole catalog so it cannot make absence claims from
        # a question subset.
        retry_numbers = {
            _positive_int(question.get("question_number")) or index
            for index, question in enumerate(questions, start=1)
        }
    if not retry_numbers:
        return normalized_primary, primary_raw, primary_usage

    retry_questions = [
        question
        for index, question in enumerate(questions, start=1)
        if (_positive_int(question.get("question_number")) or index) in retry_numbers
    ]
    retry_catalog = [_catalog_question(question) for question in retry_questions]
    recovery_limit = _whole_copy_output_limit(
        retry_catalog,
        reasoning_effort=reasoning_effort,
        recovery=True,
    )
    recovery_response: Any = None
    try:
        recovery_response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_whole_copy_responses_input(
                # Keep the immutable full catalog and PDFs as the same cached
                # request prefix. The final recovery instruction and schema
                # restrict output to the requested question numbers.
                catalog=catalog,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=student_content,
                paper_filename=paper_filename,
                solution_filename=solution_filename,
                recovery_question_numbers=sorted(retry_numbers),
            ),
            json_schema=_whole_copy_schema(retry_catalog),
            prompt_cache_key=cache_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=recovery_limit,
            metadata={
                "pcr_stage": "whole_copy_visual_recovery",
                "prompt_version": _V16_PROMPT_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_count": len(retry_questions),
                "page_count": page_count,
                "run_id": run_id,
                "provider_call_number": 2,
                "provider_call_limit": 2,
                "recursive_splitting": False,
            },
        )
    except Exception:
        logger.exception("Bounded whole-copy recovery failed for run %s", run_id)
        return normalized_primary, primary_raw, primary_usage

    recovery_usage = _usage_dict(recovery_response, fallback_model=model_id)
    combined_usage = _aggregate_usages(
        [primary_usage, recovery_usage], fallback_model=model_id
    )
    if _response_completion_failure(recovery_response):
        logger.warning("Bounded whole-copy recovery was incomplete for run %s", run_id)
        return normalized_primary, primary_raw, combined_usage
    recovery_raw = str(getattr(recovery_response, "content", "") or "")
    recovery_payload = _parse_json_object(recovery_raw)
    if recovery_payload is None:
        logger.warning("Bounded whole-copy recovery returned invalid JSON for run %s", run_id)
        return normalized_primary, primary_raw, combined_usage

    normalized_recovery = _normalize_whole_copy_payload(recovery_payload)
    retry_grades, _, _ = _validate_ledger(
        normalized_recovery,
        questions=retry_questions,
        page_count=page_count,
    )
    resolved_numbers = [
        grade.question_number
        for grade in retry_grades
        if grade.attempt_status != "unresolved" and not grade.manual_review_required
    ]
    if not resolved_numbers:
        return normalized_primary, primary_raw, combined_usage
    merged_payload = _merge_whole_copy_recovery_payload(
        primary_payload,
        recovery_payload,
        recovered_question_numbers=resolved_numbers,
    )
    normalized_merged = _normalize_whole_copy_payload(merged_payload)
    return (
        normalized_merged,
        json.dumps(merged_payload, ensure_ascii=False, separators=(",", ":")),
        combined_usage,
    )


def _build_whole_copy_responses_input(
    *,
    catalog: Sequence[Mapping[str, Any]],
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: Sequence[Mapping[str, Any]],
    paper_filename: str,
    solution_filename: str,
    recovery_question_numbers: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE MARKING CATALOG. Question IDs, ordering, maximum marks, "
                "reference solutions, criteria, assessment units, and marking policies "
                "are authoritative.\n"
                + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": "data:application/pdf;base64," + base64.b64encode(paper_bytes).decode("ascii"),
        },
    ]
    if solution_bytes:
        static_content.extend([
            {
                "type": "input_text",
                "text": "TEACHER-UPLOADED SOLUTION / MARKING-SCHEME PDF:",
            },
            {
                "type": "input_file",
                "filename": _safe_pdf_filename(
                    solution_filename, "teacher-solution.pdf"
                ),
                "file_data": "data:application/pdf;base64,"
                + base64.b64encode(solution_bytes).decode("ascii"),
            },
        ])
    variable_content = [dict(item) for item in student_content]
    if recovery_question_numbers:
        variable_content.insert(
            0,
            {
                "type": "input_text",
                "text": (
                    "ONE AND ONLY RECOVERY PASS. Re-check the complete copy, but return "
                    "only these question numbers exactly once: "
                    + ", ".join(str(number) for number in recovery_question_numbers)
                    + ". Resolve readable Hindi, sideways, diagram, continuation, and "
                    "ownership cases directly. Do not revisit any other score."
                ),
            },
        )
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": _whole_copy_system_instructions()}],
        },
        {"role": "user", "content": static_content},
        {"role": "user", "content": variable_content},
    ]


async def _run_evidence_first_grading(
    *,
    db: Any,
    gate: FullDocumentGateProtocol,
    existing_run: Optional[Mapping[str, Any]],
    generation_lease_token: str,
    run_id: str,
    submission_id: str,
    exam_id: str,
    questions: List[Dict[str, Any]],
    page_count: int,
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
    model_id: str,
    reasoning_effort: str,
    temperature: float,
    paper_hash: str,
    solution_hash: Optional[str],
    pipeline_version: str = _PROMPT_VERSION,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Map student-owned evidence once, then grade the fixed map once.

    The mapping checkpoint is deliberately persisted before grading. A provider
    failure during grading can therefore resume without purchasing completed
    visual association work again. V13 remains the two-call legacy contract;
    bounded contracts checkpoint page units, and v15 adds one absence/ownership
    recovery pass only when the first map leaves any question without attempted
    evidence.
    """

    if pipeline_version in {_V14_PROMPT_VERSION, _V15_PROMPT_VERSION}:
        return await _run_bounded_evidence_pipeline(
            db=db,
            gate=gate,
            existing_run=existing_run,
            generation_lease_token=generation_lease_token,
            run_id=run_id,
            submission_id=submission_id,
            exam_id=exam_id,
            questions=questions,
            page_count=page_count,
            paper_bytes=paper_bytes,
            solution_bytes=solution_bytes,
            student_content=student_content,
            paper_filename=paper_filename,
            solution_filename=solution_filename,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            paper_hash=paper_hash,
            solution_hash=solution_hash,
            prompt_version=pipeline_version,
        )

    current_run = dict(existing_run or {})
    mapping_payload = current_run.get("evidence_mapping_payload")
    mapping_raw = str(current_run.get("evidence_mapping_raw") or "")
    mapping_usage = dict(current_run.get("evidence_mapping_usage") or {})
    resolved_model = str(current_run.get("model_used") or model_id)
    mapping_catalog = [_mapping_catalog_question(question) for question in questions]
    mapping_cache_key = _stage_cache_key(
        "mapping",
        paper_hash=paper_hash,
        solution_hash=None,
    )

    if not isinstance(mapping_payload, dict):
        mapping_response = await gate.call(
            model_id=resolved_model,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_mapping_responses_input(
                catalog=mapping_catalog,
                paper_bytes=paper_bytes,
                student_content=student_content,
                paper_filename=paper_filename,
            ),
            json_schema=evidence_mapping_schema(mapping_catalog),
            prompt_cache_key=mapping_cache_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=_evidence_mapping_output_limit(len(questions)),
            metadata={
                "pcr_stage": "student_evidence_mapping",
                "prompt_version": _PROMPT_VERSION,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_count": len(questions),
                "page_count": page_count,
                "run_id": run_id,
                "provider_call_number": 1,
                "provider_call_limit": 2,
            },
        )
        mapping_raw = str(getattr(mapping_response, "content", "") or "")
        mapping_payload = _require_complete_structured_payload(
            mapping_response,
            raw=mapping_raw,
            stage="Student evidence mapping",
            output_limit=_evidence_mapping_output_limit(len(questions)),
        )
        mapping_usage = _usage_dict(mapping_response, fallback_model=resolved_model)
        resolved_model = str(mapping_usage.get("model") or resolved_model)
        checkpoint = await db[_RUNS_COLLECTION].update_one(
            {
                "run_id": run_id,
                "generation_lease_token": generation_lease_token,
            },
            {
                "$set": {
                    "evidence_mapping_payload": mapping_payload,
                    "evidence_mapping_raw": mapping_raw,
                    "evidence_mapping_usage": mapping_usage,
                    "model_used": resolved_model,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        if checkpoint.matched_count != 1:
            raise FullDocumentGradingError(
                "Submission grading ownership expired while saving evidence mapping"
            )

    question_numbers = [
        _positive_int(question.get("question_number")) or index
        for index, question in enumerate(questions, start=1)
    ]
    mapping = validate_mapping_payload(
        mapping_payload,
        question_numbers=question_numbers,
        page_count=page_count,
    )
    attempted_numbers = {
        number
        for number, item in mapping.questions.items()
        if item.get("attempt_status") == "attempted"
    }
    attempted_questions = [
        question
        for index, question in enumerate(questions, start=1)
        if (_positive_int(question.get("question_number")) or index)
        in attempted_numbers
    ]

    if attempted_questions:
        static_grading_catalog = [_catalog_question(question) for question in questions]
        attempted_grading_contracts = [
            _catalog_question(question) for question in attempted_questions
        ]
        grading_page_numbers = {
            int(region.get("page_number") or 0)
            for number in attempted_numbers
            for region in (
                (mapping.questions.get(number) or {}).get("evidence_regions") or []
            )
            if int(region.get("page_number") or 0) > 0
        }
        grading_response = await gate.call(
            model_id=resolved_model,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_evidence_grading_responses_input(
                catalog=static_grading_catalog,
                mapping=mapping,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=_student_content_for_pages(
                    student_content,
                    grading_page_numbers,
                ),
                paper_filename=paper_filename,
                solution_filename=solution_filename,
            ),
            json_schema=evidence_grading_schema(
                attempted_grading_contracts,
                mapping,
            ),
            prompt_cache_key=_stage_cache_key(
                "grading",
                paper_hash=paper_hash,
                solution_hash=solution_hash,
            ),
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=_evidence_grading_output_limit(len(attempted_questions)),
            metadata={
                "pcr_stage": "mapped_evidence_grading",
                "prompt_version": _PROMPT_VERSION,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_count": len(attempted_questions),
                "page_count": page_count,
                "run_id": run_id,
                "provider_call_number": 2,
                "provider_call_limit": 2,
            },
        )
        grading_raw = str(getattr(grading_response, "content", "") or "")
        grading_payload = _require_complete_structured_payload(
            grading_response,
            raw=grading_raw,
            stage="Mapped evidence grading",
            output_limit=_evidence_grading_output_limit(len(attempted_questions)),
        )
        if grading_payload.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
            raise StructuredGradingOutputError(
                "Mapped evidence grader returned the wrong evidence contract"
            )
        grading_usage = _usage_dict(grading_response, fallback_model=resolved_model)
    else:
        grading_payload = {
            "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
            "questions": [],
        }
        grading_raw = json.dumps(grading_payload, separators=(",", ":"))
        grading_usage = {}

    merged = merge_mapping_and_grading(mapping, grading_payload)
    raw_combined = json.dumps(
        {"mapping": mapping_payload, "grading": grading_payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    usage = _aggregate_usages(
        [mapping_usage, grading_usage],
        fallback_model=resolved_model,
    )
    return merged, raw_combined, usage


async def _run_bounded_evidence_pipeline(
    *,
    db: Any,
    gate: FullDocumentGateProtocol,
    existing_run: Optional[Mapping[str, Any]],
    generation_lease_token: str,
    run_id: str,
    submission_id: str,
    exam_id: str,
    questions: List[Dict[str, Any]],
    page_count: int,
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
    model_id: str,
    reasoning_effort: str,
    temperature: float,
    paper_hash: str,
    solution_hash: Optional[str],
    prompt_version: str = _V14_PROMPT_VERSION,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Execute bounded evidence mapping and grading work units.

    Every provider request is bounded.  A max-output failure recursively splits
    the *work unit* (pages first, then question catalog) and never repeats an
    identical oversized request.  Completed unit payloads are checkpointed so a
    worker restart only purchases missing units.
    """

    current_run = dict(existing_run or {})
    question_numbers = [
        _positive_int(question.get("question_number")) or index
        for index, question in enumerate(questions, start=1)
    ]
    mapping_catalog = [_mapping_catalog_question(question) for question in questions]
    raw_units = current_run.get("evidence_mapping_units")
    unit_records = [dict(item) for item in raw_units if isinstance(item, Mapping)] if isinstance(raw_units, list) else []
    split_records = [dict(item) for item in (current_run.get("evidence_mapping_split_manifests") or []) if isinstance(item, Mapping)]
    completed_units = {str(item.get("unit_id") or ""): item for item in unit_records if item.get("payload")}
    unit_usages: List[Dict[str, Any]] = [
        dict(item.get("usage") or {}) for item in unit_records if isinstance(item.get("usage"), Mapping)
    ]
    page_numbers = list(range(1, max(1, page_count) + 1))
    page_groups = [page_numbers[index : index + 2] for index in range(0, len(page_numbers), 2)]
    mapping_payloads: List[Mapping[str, Any]] = []

    async def persist_units() -> None:
        checkpoint = await db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "generation_lease_token": generation_lease_token},
            {"$set": {
                "evidence_mapping_units": unit_records,
                "evidence_mapping_split_manifests": split_records,
                "evidence_mapping_usage": _aggregate_usages(unit_usages, fallback_model=model_id),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        if checkpoint.matched_count != 1:
            raise FullDocumentGradingError("Submission grading ownership expired while saving mapping unit")

    async def map_unit(pages: List[int], catalog: List[Dict[str, Any]], unit_id: str) -> None:
        existing = next((item for item in unit_records if str(item.get("unit_id") or "") == unit_id), None)
        existing_split = next((item for item in split_records if str(item.get("unit_id") or "") == unit_id), None)
        existing = existing or existing_split
        if existing and existing.get("payload"):
            mapping_payloads.append(existing["payload"])
            return
        if existing and existing.get("status") == "split":
            for child in existing.get("children") or []:
                child_pages = [int(value) for value in child.get("pages") or []]
                child_numbers = {int(value) for value in child.get("question_numbers") or []}
                child_catalog = [item for item in mapping_catalog if int(item.get("question_number") or 0) in child_numbers]
                await map_unit(child_pages, child_catalog, str(child.get("unit_id") or ""))
            return
        content = _student_content_for_pages(student_content, set(pages))
        response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_compact_mapping_responses_input(
                catalog=catalog,
                paper_bytes=paper_bytes,
                student_content=content,
                paper_filename=paper_filename,
                page_numbers=pages,
                prompt_version=prompt_version,
            ),
            json_schema=compact_mapping_schema(
                catalog,
                prompt_version=prompt_version,
            ),
            prompt_cache_key=_stage_cache_key(
                "mapping-static",
                paper_hash=paper_hash,
                solution_hash=None,
                prompt_version=prompt_version,
            ),
            reasoning_effort=_mapping_reasoning_effort(
                reasoning_effort,
                prompt_version=prompt_version,
            ),
            temperature=temperature,
            max_output_tokens=_V14_MAX_OUTPUT_TOKENS,
            metadata={
                "pcr_stage": "bounded_student_evidence_mapping",
                "prompt_version": prompt_version,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "run_id": run_id,
                "unit_id": unit_id,
                "unit_pages": pages,
                "question_count": len(catalog),
                "max_output_tokens": _V14_MAX_OUTPUT_TOKENS,
            },
        )
        raw = str(getattr(response, "content", "") or "")
        try:
            payload = _require_complete_structured_payload(
                response, raw=raw, stage=f"Student evidence mapping unit {unit_id}", output_limit=_V14_MAX_OUTPUT_TOKENS
            )
        except StructuredGradingOutputError as exc:
            # Split before retrying.  This condition is deliberately checked
            # against both dimensions because a single page can contain a very
            # large question catalog and vice versa.
            if not _is_output_exhaustion(exc):
                raise
            if len(pages) > 1:
                midpoint = max(1, len(pages) // 2)
                children = [
                    {"unit_id": unit_id + "-a", "pages": pages[:midpoint], "question_numbers": [int(item.get("question_number")) for item in catalog]},
                    {"unit_id": unit_id + "-b", "pages": pages[midpoint:], "question_numbers": [int(item.get("question_number")) for item in catalog]},
                ]
                _record_split(split_records, unit_id, pages, catalog, children)
                await persist_units()
                await map_unit(pages[:midpoint], catalog, unit_id + "-a")
                await map_unit(pages[midpoint:], catalog, unit_id + "-b")
                return
            if len(catalog) > 1:
                midpoint = max(1, len(catalog) // 2)
                children = [
                    {"unit_id": unit_id + "-a", "pages": pages, "question_numbers": [int(item.get("question_number")) for item in catalog[:midpoint]]},
                    {"unit_id": unit_id + "-b", "pages": pages, "question_numbers": [int(item.get("question_number")) for item in catalog[midpoint:]]},
                ]
                _record_split(split_records, unit_id, pages, catalog, children)
                await persist_units()
                await map_unit(pages, catalog[:midpoint], unit_id + "-a")
                await map_unit(pages, catalog[midpoint:], unit_id + "-b")
                return
            raise
        _validate_compact_unit_pages(payload, pages)
        usage = _usage_dict(response, fallback_model=model_id)
        payload = _compact_payload_from_response(
            payload,
            prompt_version=prompt_version,
            student_content=content,
        )
        record = {"unit_id": unit_id, "pages": pages, "question_numbers": [
            int(item.get("question_number")) for item in catalog if item.get("question_number")
        ], "payload": payload, "usage": usage}
        unit_records.append(record)
        completed_units[unit_id] = record
        unit_usages.append(usage)
        mapping_payloads.append(payload)
        await persist_units()

    for index, pages in enumerate(page_groups, start=1):
        await map_unit(pages, mapping_catalog, f"pages-{index}")

    mapping = merge_compact_mapping_payloads(
        mapping_payloads, question_numbers=question_numbers, page_count=page_count
    )
    recovery_payloads: List[Mapping[str, Any]] = []
    recovery_usages: List[Dict[str, Any]] = []
    if prompt_version == _V15_PROMPT_VERSION:
        recovery_question_numbers = {
            number
            for number, item in mapping.questions.items()
            if item.get("attempt_status") != "attempted"
        }
        recovery_pages = {
            int(region.get("page_number") or 0)
            for region in mapping.unassigned_regions
            if int(region.get("page_number") or 0) > 0
        }
        recovery_pages.update(
            int(region.get("page_number") or 0)
            for number in recovery_question_numbers
            for region in mapping.questions[number].get("evidence_regions") or []
            if int(region.get("page_number") or 0) > 0
        )
        # A zero for a supposedly absent answer is a high-impact decision.
        # Reinspect the full copy once whenever any catalog question lacks
        # attempted evidence, even when the first mapper claimed completeness.
        # This also catches ownership collapse (for example, all Hindi work
        # being assigned to one long-answer question).
        if recovery_question_numbers:
            recovery_pages.update(page_numbers)

        if recovery_question_numbers and recovery_pages:
            raw_recovery_units = current_run.get("evidence_mapping_recovery_units")
            recovery_records = (
                [dict(item) for item in raw_recovery_units if isinstance(item, Mapping)]
                if isinstance(raw_recovery_units, list)
                else []
            )
            recovery_splits = [
                dict(item)
                for item in (
                    current_run.get("evidence_mapping_recovery_split_manifests") or []
                )
                if isinstance(item, Mapping)
            ]
            recovery_usages.extend(
                dict(item.get("usage") or {})
                for item in recovery_records
                if isinstance(item.get("usage"), Mapping)
            )

            async def persist_recovery_units() -> None:
                checkpoint = await db[_RUNS_COLLECTION].update_one(
                    {
                        "run_id": run_id,
                        "generation_lease_token": generation_lease_token,
                    },
                    {"$set": {
                        "evidence_mapping_recovery_units": recovery_records,
                        "evidence_mapping_recovery_split_manifests": recovery_splits,
                        "evidence_mapping_recovery_usage": _aggregate_usages(
                            recovery_usages,
                            fallback_model=model_id,
                        ),
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
                if checkpoint.matched_count != 1:
                    raise FullDocumentGradingError(
                        "Submission grading ownership expired while saving recovery unit"
                    )

            def recovery_context(pages: Sequence[int]) -> Dict[str, Any]:
                allowed_pages = set(pages)
                assigned = [
                    {
                        "question_number": number,
                        "prior_attempt_status": mapped.get("attempt_status"),
                        "region_id": region.get("region_id"),
                        "page_number": region.get("page_number"),
                        "x_start": region.get("x_start"),
                        "y_start": region.get("y_start"),
                        "x_end": region.get("x_end"),
                        "y_end": region.get("y_end"),
                    }
                    for number, mapped in mapping.questions.items()
                    for region in mapped.get("evidence_regions") or []
                    if int(region.get("page_number") or 0) in allowed_pages
                ]
                unassigned = [
                    {
                        key: region.get(key)
                        for key in (
                            "region_id", "page_number", "x_start", "y_start",
                            "x_end", "y_end", "evidence_kind",
                        )
                    }
                    for region in mapping.unassigned_regions
                    if int(region.get("page_number") or 0) in allowed_pages
                ]
                return {
                    "coordinate_space": "normalized_1000_original_page_frame",
                    "questions_without_attempted_evidence": sorted(
                        recovery_question_numbers
                    ),
                    "immutable_assigned_regions": assigned,
                    "regions_requiring_reassociation": unassigned,
                    "prior_all_student_work_accounted": bool(
                        mapping.document_review.get("all_student_work_accounted")
                    ),
                }

            async def recover_unit(
                pages: List[int],
                catalog: List[Dict[str, Any]],
                unit_id: str,
            ) -> None:
                existing = next(
                    (
                        item for item in recovery_records
                        if str(item.get("unit_id") or "") == unit_id
                    ),
                    None,
                )
                existing_split = next(
                    (
                        item for item in recovery_splits
                        if str(item.get("unit_id") or "") == unit_id
                    ),
                    None,
                )
                existing = existing or existing_split
                if existing and existing.get("payload"):
                    recovery_payloads.append(existing["payload"])
                    return
                if existing and existing.get("status") == "split":
                    for child in existing.get("children") or []:
                        child_pages = [int(value) for value in child.get("pages") or []]
                        child_numbers = {
                            int(value) for value in child.get("question_numbers") or []
                        }
                        child_catalog = [
                            item for item in mapping_catalog
                            if int(item.get("question_number") or 0) in child_numbers
                        ]
                        await recover_unit(
                            child_pages,
                            child_catalog,
                            str(child.get("unit_id") or ""),
                        )
                    return

                content = _student_content_for_pages(student_content, set(pages))
                response = await gate.call(
                    model_id=model_id,
                    prompt="",
                    caller_id=_CALLER_ID,
                    responses_input=_build_compact_mapping_responses_input(
                        catalog=catalog,
                        paper_bytes=paper_bytes,
                        student_content=content,
                        paper_filename=paper_filename,
                        page_numbers=pages,
                        prompt_version=prompt_version,
                        recovery_context=recovery_context(pages),
                    ),
                    json_schema=compact_mapping_schema(
                        catalog,
                        prompt_version=prompt_version,
                        recovery_pass=True,
                    ),
                    prompt_cache_key=_stage_cache_key(
                        "mapping-recovery-static",
                        paper_hash=paper_hash,
                        solution_hash=None,
                        prompt_version=prompt_version,
                    ),
                    reasoning_effort=_mapping_reasoning_effort(
                        reasoning_effort,
                        prompt_version=prompt_version,
                    ),
                    temperature=temperature,
                    max_output_tokens=_V14_MAX_OUTPUT_TOKENS,
                    metadata={
                        "pcr_stage": "bounded_student_evidence_mapping_recovery",
                        "prompt_version": prompt_version,
                        "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                        "submission_id": submission_id,
                        "exam_id": exam_id,
                        "run_id": run_id,
                        "unit_id": unit_id,
                        "unit_pages": pages,
                        "question_count": len(catalog),
                        "max_output_tokens": _V14_MAX_OUTPUT_TOKENS,
                    },
                )
                raw = str(getattr(response, "content", "") or "")
                try:
                    payload = _require_complete_structured_payload(
                        response,
                        raw=raw,
                        stage=f"Student evidence mapping recovery unit {unit_id}",
                        output_limit=_V14_MAX_OUTPUT_TOKENS,
                    )
                except StructuredGradingOutputError as exc:
                    if not _is_output_exhaustion(exc):
                        raise
                    if len(pages) > 1:
                        midpoint = max(1, len(pages) // 2)
                        children = [
                            {
                                "unit_id": unit_id + "-a",
                                "pages": pages[:midpoint],
                                "question_numbers": [
                                    int(item.get("question_number")) for item in catalog
                                ],
                            },
                            {
                                "unit_id": unit_id + "-b",
                                "pages": pages[midpoint:],
                                "question_numbers": [
                                    int(item.get("question_number")) for item in catalog
                                ],
                            },
                        ]
                        _record_split(recovery_splits, unit_id, pages, catalog, children)
                        await persist_recovery_units()
                        await recover_unit(pages[:midpoint], catalog, unit_id + "-a")
                        await recover_unit(pages[midpoint:], catalog, unit_id + "-b")
                        return
                    if len(catalog) > 1:
                        midpoint = max(1, len(catalog) // 2)
                        children = [
                            {
                                "unit_id": unit_id + "-a",
                                "pages": pages,
                                "question_numbers": [
                                    int(item.get("question_number"))
                                    for item in catalog[:midpoint]
                                ],
                            },
                            {
                                "unit_id": unit_id + "-b",
                                "pages": pages,
                                "question_numbers": [
                                    int(item.get("question_number"))
                                    for item in catalog[midpoint:]
                                ],
                            },
                        ]
                        _record_split(recovery_splits, unit_id, pages, catalog, children)
                        await persist_recovery_units()
                        await recover_unit(pages, catalog[:midpoint], unit_id + "-a")
                        await recover_unit(pages, catalog[midpoint:], unit_id + "-b")
                        return
                    raise
                _validate_compact_unit_pages(payload, pages)
                payload = _compact_payload_from_response(
                    payload,
                    prompt_version=prompt_version,
                    student_content=content,
                )
                usage = _usage_dict(response, fallback_model=model_id)
                record = {
                    "unit_id": unit_id,
                    "pages": pages,
                    "question_numbers": [
                        int(item.get("question_number"))
                        for item in catalog if item.get("question_number")
                    ],
                    "payload": payload,
                    "usage": usage,
                }
                recovery_records.append(record)
                recovery_usages.append(usage)
                recovery_payloads.append(payload)
                await persist_recovery_units()

            recovery_page_list = sorted(recovery_pages)
            for index in range(0, len(recovery_page_list), 2):
                await recover_unit(
                    recovery_page_list[index : index + 2],
                    mapping_catalog,
                    f"recovery-pages-{index // 2 + 1}",
                )
            mapping = reconcile_compact_mapping_recovery(
                mapping_payloads,
                recovery_payloads,
                question_numbers=question_numbers,
                page_count=page_count,
                recovered_page_numbers=recovery_page_list,
            )
    attempted_numbers = {
        number for number, item in mapping.questions.items()
        if item.get("attempt_status") == "attempted"
    }
    attempted_questions = [
        question for index, question in enumerate(questions, start=1)
        if (_positive_int(question.get("question_number")) or index) in attempted_numbers
    ]

    raw_batches = current_run.get("evidence_grading_batches")
    batch_records = [dict(item) for item in raw_batches if isinstance(item, Mapping)] if isinstance(raw_batches, list) else []
    split_batch_records = [dict(item) for item in (current_run.get("evidence_grading_split_manifests") or []) if isinstance(item, Mapping)]
    completed_batches = {str(item.get("batch_id") or ""): item for item in batch_records if isinstance(item.get("payload"), Mapping)}
    grade_usages: List[Dict[str, Any]] = [
        dict(item.get("usage") or {}) for item in batch_records if isinstance(item.get("usage"), Mapping)
    ]
    grading_payloads: List[Mapping[str, Any]] = []
    grading_catalog = [_catalog_question(question) for question in questions]
    batches = [attempted_questions[i : i + _V14_MAX_QUESTIONS_PER_BATCH] for i in range(0, len(attempted_questions), _V14_MAX_QUESTIONS_PER_BATCH)]

    async def persist_batches() -> None:
        checkpoint = await db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "generation_lease_token": generation_lease_token},
            {"$set": {
                "evidence_grading_batches": batch_records,
                "evidence_grading_split_manifests": split_batch_records,
                "evidence_grading_usage": _aggregate_usages(grade_usages, fallback_model=model_id),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        if checkpoint.matched_count != 1:
            raise FullDocumentGradingError("Submission grading ownership expired while saving grading batch")

    async def grade_batch(items: List[Dict[str, Any]], batch_id: str) -> None:
        existing = next((item for item in batch_records if str(item.get("batch_id") or "") == batch_id), None)
        existing_split = next((item for item in split_batch_records if str(item.get("batch_id") or "") == batch_id), None)
        existing = existing or existing_split
        if existing and existing.get("payload"):
            grading_payloads.append(existing["payload"])
            return
        if existing and existing.get("status") == "split":
            by_number = {_positive_int(item.get("question_number")): item for item in attempted_questions}
            for child in existing.get("children") or []:
                child_items = [by_number[number] for number in child.get("question_numbers") or [] if number in by_number]
                await grade_batch(child_items, str(child.get("batch_id") or ""))
            return
        numbers = {_positive_int(item.get("question_number")) for item in items}
        subset = _subset_mapping(mapping, {number for number in numbers if number})
        page_set = {
            int(region.get("page_number") or 0)
            for number in subset.questions
            for region in subset.questions[number].get("evidence_regions") or []
            if int(region.get("page_number") or 0) > 0
        }
        output_limit = _bounded_grading_output_limit(
            items,
            reasoning_effort=reasoning_effort,
        )
        response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_evidence_grading_responses_input(
                catalog=grading_catalog,
                requested_question_numbers=sorted(number for number in numbers if number),
                mapping=subset,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=_student_content_for_pages(
                    student_content,
                    page_set,
                    source_rotations=_source_rotations_from_mapping(subset),
                ),
                paper_filename=paper_filename,
                solution_filename=solution_filename,
            ),
            json_schema=evidence_grading_schema(
                [item for item in grading_catalog if _positive_int(item.get("question_number")) in numbers], subset
            ),
            prompt_cache_key=_stage_cache_key(
                "grading-static",
                paper_hash=paper_hash,
                solution_hash=solution_hash,
                prompt_version=prompt_version,
            ),
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=output_limit,
            metadata={
                "pcr_stage": "bounded_mapped_evidence_grading",
                "prompt_version": prompt_version,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "run_id": run_id,
                "batch_id": batch_id,
                "question_count": len(items),
                "max_output_tokens": output_limit,
                "static_prompt_cache": True,
            },
        )
        raw = str(getattr(response, "content", "") or "")
        try:
            payload = _require_complete_structured_payload(
                response,
                raw=raw,
                stage=f"Mapped evidence grading batch {batch_id}",
                output_limit=output_limit,
            )
        except StructuredGradingOutputError as exc:
            if not _is_output_exhaustion(exc):
                raise
            if len(items) > 1:
                midpoint = max(1, len(items) // 2)
                children = [
                    {"batch_id": batch_id + "-a", "question_numbers": [int(item.get("question_number")) for item in items[:midpoint]]},
                    {"batch_id": batch_id + "-b", "question_numbers": [int(item.get("question_number")) for item in items[midpoint:]]},
                ]
                _record_batch_split(split_batch_records, batch_id, items, children)
                await persist_batches()
                await grade_batch(items[:midpoint], batch_id + "-a")
                await grade_batch(items[midpoint:], batch_id + "-b")
                return
            raise
        if payload.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
            raise StructuredGradingOutputError("Mapped evidence grader returned the wrong evidence contract")
        usage = _usage_dict(response, fallback_model=model_id)
        record = {"batch_id": batch_id, "question_numbers": sorted(numbers), "payload": payload, "usage": usage}
        batch_records.append(record)
        completed_batches[batch_id] = record
        grade_usages.append(usage)
        grading_payloads.append(payload)
        await persist_batches()

    for index, batch in enumerate(batches, start=1):
        await grade_batch(batch, f"batch-{index}")

    all_grades = []
    for payload in grading_payloads:
        all_grades.extend(item for item in payload.get("questions", []) if isinstance(item, Mapping))
    grading_payload = {"evidence_graph_version": EVIDENCE_GRAPH_VERSION, "questions": all_grades}
    merged = merge_mapping_and_grading(mapping, grading_payload)
    verification_raw, verification_usage = await _run_full_score_verification(
        db=db,
        gate=gate,
        existing_run=current_run,
        generation_lease_token=generation_lease_token,
        run_id=run_id,
        submission_id=submission_id,
        exam_id=exam_id,
        questions=questions,
        mapping=mapping,
        merged=merged,
        paper_bytes=paper_bytes,
        solution_bytes=solution_bytes,
        student_content=student_content,
        paper_filename=paper_filename,
        solution_filename=solution_filename,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        paper_hash=paper_hash,
        solution_hash=solution_hash,
        prompt_version=prompt_version,
    )
    raw_combined = json.dumps(
        {
            "mapping_units": mapping_payloads,
            "mapping_recovery_units": recovery_payloads,
            "grading_batches": grading_payloads,
            "verification": verification_raw,
        },
        ensure_ascii=False, separators=(",", ":"),
    )
    usage = _aggregate_usages(
        unit_usages + recovery_usages + grade_usages + verification_usage,
        fallback_model=model_id,
    )
    return merged, raw_combined, usage


async def _run_full_score_verification(
    *,
    db: Any,
    gate: FullDocumentGateProtocol,
    existing_run: Mapping[str, Any],
    generation_lease_token: str,
    run_id: str,
    submission_id: str,
    exam_id: str,
    questions: List[Dict[str, Any]],
    mapping: Any,
    merged: Dict[str, Any],
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
    model_id: str,
    reasoning_effort: str,
    temperature: float,
    paper_hash: str,
    solution_hash: Optional[str],
    prompt_version: str = _V14_PROMPT_VERSION,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Audit only subjective provisional full marks, without changing scores."""

    by_number = {
        int(item.get("question_number")): item
        for item in (merged.get("questions") or [])
        if isinstance(item, Mapping) and _positive_int(item.get("question_number"))
    }
    candidates: List[Dict[str, Any]] = []
    for position, question in enumerate(questions, start=1):
        number = _positive_int(question.get("question_number")) or position
        result = by_number.get(number) or {}
        if _is_objective_question(question):
            continue
        if result.get("attempt_status") != "attempted" or result.get("needs_review"):
            continue
        criteria = list(question.get("marking_criteria") or [])
        marks = list(result.get("criterion_marks") or [])
        max_total = sum(max(0.0, float(item.get("max_marks") or 0)) for item in criteria)
        by_criterion = {
            str(item.get("criterion_id") or ""): item
            for item in marks if isinstance(item, Mapping)
        }
        if not criteria or max_total <= 0 or len(by_criterion) != len(criteria):
            continue
        if abs(sum(float(item.get("marks_awarded") or 0) for item in marks) - max_total) > 0.001:
            continue
        if any(
            abs(float(by_criterion.get(str(item.get("criterion_id") or ""), {}).get("marks_awarded") or 0) - max(0.0, float(item.get("max_marks") or 0))) > 0.001
            for item in criteria
        ):
            continue
        candidates.append(question)
    if not candidates:
        return {"batches": [], "skipped": True}, []

    current_run = dict(existing_run or {})
    raw_batches = current_run.get("evidence_verification_batches")
    batch_records = [dict(item) for item in raw_batches if isinstance(item, Mapping)] if isinstance(raw_batches, list) else []
    split_records = [dict(item) for item in (current_run.get("evidence_verification_split_manifests") or []) if isinstance(item, Mapping)]
    usage_records: List[Dict[str, Any]] = [
        dict(item.get("usage") or {}) for item in batch_records if isinstance(item.get("usage"), Mapping)
    ]
    audit_payloads: List[Mapping[str, Any]] = []
    audit_flags: Dict[int, str] = {}
    candidate_by_number = {
        _positive_int(item.get("question_number")): item for item in candidates
    }

    async def persist() -> None:
        checkpoint = await db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "generation_lease_token": generation_lease_token},
            {"$set": {
                "evidence_verification_batches": batch_records,
                "evidence_verification_split_manifests": split_records,
                "evidence_verification_usage": _aggregate_usages(usage_records, fallback_model=model_id),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        if checkpoint.matched_count != 1:
            raise FullDocumentGradingError("Submission grading ownership expired while saving full-score audit")

    async def audit_batch(items: List[Dict[str, Any]], batch_id: str) -> None:
        existing = next((item for item in batch_records if str(item.get("batch_id") or "") == batch_id), None)
        existing = existing or next((item for item in split_records if str(item.get("batch_id") or "") == batch_id), None)
        if existing and isinstance(existing.get("payload"), Mapping):
            audit_payloads.append(existing["payload"])
            return
        if existing and existing.get("status") == "invalid":
            for number in existing.get("question_numbers") or []:
                audit_flags[_positive_int(number) or 0] = str(
                    existing.get("error") or "Independent full-score verification did not complete"
                )[:500]
            return
        if existing and existing.get("status") == "split":
            for child in existing.get("children") or []:
                child_items = [candidate_by_number[number] for number in child.get("question_numbers") or [] if number in candidate_by_number]
                await audit_batch(child_items, str(child.get("batch_id") or ""))
            return
        numbers = {_positive_int(item.get("question_number")) for item in items}
        subset = _subset_mapping(mapping, {number for number in numbers if number})
        page_set = {
            int(region.get("page_number") or 0)
            for number in subset.questions
            for region in subset.questions[number].get("evidence_regions") or []
            if int(region.get("page_number") or 0) > 0
        }
        contracts = [
            _catalog_question(item)
            for item in items
        ]
        audit_reasoning_effort = _mapping_reasoning_effort(
            reasoning_effort,
            prompt_version=prompt_version,
        )
        output_limit = _bounded_grading_output_limit(
            items,
            reasoning_effort=audit_reasoning_effort,
            verification=True,
        )
        response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=_build_verification_responses_input(
                catalog=[_catalog_question(item) for item in questions],
                requested_question_numbers=sorted(number for number in numbers if number),
                mapping=subset,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=_student_content_for_pages(
                    student_content,
                    page_set,
                    source_rotations=_source_rotations_from_mapping(subset),
                ),
                paper_filename=paper_filename,
                solution_filename=solution_filename,
            ),
            json_schema=verification_schema(contracts, subset),
            prompt_cache_key=_stage_cache_key(
                "verification-static",
                paper_hash=paper_hash,
                solution_hash=solution_hash,
                prompt_version=prompt_version,
            ),
            reasoning_effort=audit_reasoning_effort,
            temperature=temperature,
            max_output_tokens=output_limit,
            metadata={
                "pcr_stage": "bounded_full_score_verification",
                "prompt_version": prompt_version,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "run_id": run_id,
                "batch_id": batch_id,
                "question_count": len(items),
                "max_output_tokens": output_limit,
                "static_prompt_cache": True,
                "primary_marks_in_input": False,
            },
        )
        raw = str(getattr(response, "content", "") or "")
        try:
            payload = _require_complete_structured_payload(
                response,
                raw=raw,
                stage=f"Full-score verification batch {batch_id}",
                output_limit=output_limit,
            )
        except StructuredGradingOutputError as exc:
            if _is_output_exhaustion(exc) and len(items) > 1:
                midpoint = max(1, len(items) // 2)
                children = [
                    {"batch_id": batch_id + "-a", "question_numbers": [int(item.get("question_number")) for item in items[:midpoint]]},
                    {"batch_id": batch_id + "-b", "question_numbers": [int(item.get("question_number")) for item in items[midpoint:]]},
                ]
                _record_batch_split(split_records, batch_id, items, children)
                await persist()
                await audit_batch(items[:midpoint], batch_id + "-a")
                await audit_batch(items[midpoint:], batch_id + "-b")
                return
            for item in items:
                audit_flags[_positive_int(item.get("question_number")) or 0] = "Independent full-score verification did not complete"
            batch_records.append({
                "batch_id": batch_id,
                "status": "invalid",
                "question_numbers": sorted(number for number in numbers if number),
                "error": str(exc)[:500],
            })
            await persist()
            return
        if payload.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
            for item in items:
                audit_flags[_positive_int(item.get("question_number")) or 0] = "Independent full-score verification returned an invalid contract"
            batch_records.append({"batch_id": batch_id, "status": "invalid", "question_numbers": sorted(number for number in numbers if number), "error": "wrong evidence contract"})
            await persist()
            return
        usage = _usage_dict(response, fallback_model=model_id)
        batch_records.append({"batch_id": batch_id, "question_numbers": sorted(number for number in numbers if number), "payload": payload, "usage": usage})
        usage_records.append(usage)
        audit_payloads.append(payload)
        await persist()

    batches = [candidates[index : index + _V14_MAX_QUESTIONS_PER_BATCH] for index in range(0, len(candidates), _V14_MAX_QUESTIONS_PER_BATCH)]
    for index, batch in enumerate(batches, start=1):
        await audit_batch(batch, f"audit-{index}")

    # Compare independently supported marks to the immutable provisional rows.
    audited_numbers: set[int] = set()
    for payload in audit_payloads:
        for audit_question in payload.get("questions") or []:
            if not isinstance(audit_question, Mapping):
                continue
            number = _positive_int(audit_question.get("question_number"))
            if not number or number not in candidate_by_number or number in audited_numbers:
                if number:
                    audit_flags[number] = "Independent full-score verification returned duplicate or unknown question"
                continue
            audited_numbers.add(number)
            question = candidate_by_number[number]
            mapped_regions = {
                str(region.get("region_id") or "")
                for region in (mapping.questions.get(number) or {}).get("evidence_regions") or []
            }
            primary = by_number.get(number) or {}
            primary_rows = {
                str(row.get("criterion_id") or ""): row
                for row in primary.get("criterion_marks") or [] if isinstance(row, Mapping)
            }
            criteria = list(question.get("marking_criteria") or [])
            audit_rows = [row for row in audit_question.get("criterion_marks") or [] if isinstance(row, Mapping)]
            seen_ids: set[str] = set()
            if len(audit_rows) != len(criteria):
                audit_flags[number] = "Independent full-score verification omitted a criterion"
                continue
            for row in audit_rows:
                criterion_id = str(row.get("criterion_id") or "")
                seen_ids.add(criterion_id)
                primary_row = primary_rows.get(criterion_id)
                cited = [str(value) for value in row.get("evidence_region_ids") or [] if str(value)]
                marks_supported = _finite_float(row.get("marks_supported"))
                max_marks = next((float(item.get("max_marks") or 0) for item in criteria if str(item.get("criterion_id") or "") == criterion_id), None)
                if not primary_row or marks_supported is None or max_marks is None or marks_supported < -0.001 or marks_supported > max_marks + 0.001 or not cited or not set(cited).issubset(mapped_regions):
                    audit_flags[number] = "Independent full-score verification returned invalid criterion evidence"
                    continue
                if abs(marks_supported - float(primary_row.get("marks_awarded") or 0)) > 0.001:
                    audit_flags[number] = "Independent full-score verification disagreed with a full-score criterion"
            if seen_ids != set(primary_rows):
                audit_flags[number] = "Independent full-score verification returned an incomplete criterion set"
    for number in set(candidate_by_number) - audited_numbers:
        audit_flags[number] = audit_flags.get(number) or "Independent full-score verification did not return this question"
    for number, reason in audit_flags.items():
        result = by_number.get(number)
        if not result:
            continue
        result["needs_review"] = True
        result["review_reason"] = str(result.get("review_reason") or "")
        result["review_reason"] = "; ".join(filter(None, [result["review_reason"], reason]))[:800]
    return {"batches": audit_payloads, "flagged_questions": sorted(audit_flags)}, usage_records


_ORIENTATION_PAGE_LABEL_RE = re.compile(
    r"^Student answer-copy page (?P<page>\d+),.*?"
    r"source_rotation_degrees_clockwise=(?P<rotation>0|90|270);.*?"
    r"original_width_px=(?P<width>\d+);\s*"
    r"original_height_px=(?P<height>\d+)\."
)


def _orientation_manifest_from_student_content(
    student_content: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Return the exact orientation views supplied to the mapper."""

    manifest: Dict[int, Dict[int, Dict[str, int]]] = {}
    for item in student_content or []:
        if not isinstance(item, Mapping) or item.get("type") != "input_text":
            continue
        match = _ORIENTATION_PAGE_LABEL_RE.match(str(item.get("text") or ""))
        if not match:
            continue
        page = int(match.group("page"))
        rotation = int(match.group("rotation"))
        manifest.setdefault(page, {})[rotation] = {
            "width_px": int(match.group("width")),
            "height_px": int(match.group("height")),
        }
    return manifest


def _compact_region_in_original_frame(
    region: Mapping[str, Any],
    *,
    prompt_version: str,
    orientation_manifest: Mapping[int, Mapping[int, Mapping[str, int]]],
) -> Dict[str, Any]:
    """Validate a provider region and materialize immutable original coordinates."""

    allowed_keys = (
        "region_id", "page_number", "x_start", "y_start", "x_end", "y_end",
        "evidence_kind", "authorship", "continuation_group", "sequence",
        "mapping_confidence", "supersedes_region_ids",
    )
    compact = {key: region.get(key) for key in allowed_keys if key in region}
    if prompt_version != _V15_PROMPT_VERSION:
        if region.get("coordinate_frame") is not None:
            compact["coordinate_frame"] = region.get("coordinate_frame")
        return compact

    page_number = _positive_int(region.get("page_number"))
    try:
        rotation = int(region.get("source_rotation_degrees_clockwise"))
    except (TypeError, ValueError):
        rotation = -1
    if page_number is None or rotation not in {0, 90, 270}:
        raise StructuredGradingOutputError(
            "Orientation-aware mapping returned an invalid page or rotation"
        )
    available_views = orientation_manifest.get(page_number) or {}
    if available_views and rotation not in available_views:
        raise StructuredGradingOutputError(
            "Orientation-aware mapping cited a view that was not supplied"
        )
    try:
        compact = view_region_to_original(
            compact,
            rotation_degrees_clockwise=rotation,
        )
    except OrientationViewError as exc:
        raise StructuredGradingOutputError(
            "Orientation-aware mapping returned invalid region coordinates"
        ) from exc
    dimensions = available_views.get(rotation) or {}
    compact["coordinate_frame"] = {
        "id": f"physical-page-{page_number}-original-frame",
        "kind": "original_stored_page",
        "coordinate_space": "normalized_1000",
        "width_px": dimensions.get("width_px"),
        "height_px": dimensions.get("height_px"),
        "source_rotation_degrees_clockwise": rotation,
        "invertible": True,
    }
    return compact


def _compact_payload_from_response(
    payload: Mapping[str, Any],
    *,
    prompt_version: str = _V14_PROMPT_VERSION,
    student_content: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Persist compact mapper output in the immutable original-page frame."""

    orientation_manifest = _orientation_manifest_from_student_content(student_content)

    compact: Dict[str, Any] = {
        "mapping_version": (
            "pcr-compact-evidence-map-v2"
            if prompt_version == _V15_PROMPT_VERSION
            else "pcr-compact-evidence-map-v1"
        ),
        "evidence_graph_version": payload.get("evidence_graph_version", EVIDENCE_GRAPH_VERSION),
        "all_student_work_accounted": bool(
            payload.get("all_student_work_accounted")
            or (payload.get("document_review") or {}).get("all_student_work_accounted")
        ),
        "questions": [],
        "unassigned_student_regions": [],
    }
    for question in payload.get("questions") or []:
        if not isinstance(question, Mapping):
            continue
        compact_question = {
            "question_number": question.get("question_number"),
            "attempt_status": question.get("attempt_status", "attempted"),
            "content_type": question.get("content_type", "MIXED"),
            "evidence_regions": [
                _compact_region_in_original_frame(
                    region,
                    prompt_version=prompt_version,
                    orientation_manifest=orientation_manifest,
                )
                for region in (question.get("evidence_regions") or []) if isinstance(region, Mapping)
            ],
        }
        if prompt_version == _V15_PROMPT_VERSION:
            compact_question["association_basis"] = str(
                question.get("association_basis") or ""
            )[:240]
        compact["questions"].append(compact_question)
    for region in payload.get("unassigned_student_regions") or []:
        if isinstance(region, Mapping):
            compact["unassigned_student_regions"].append(
                _compact_region_in_original_frame(
                    region,
                    prompt_version=prompt_version,
                    orientation_manifest=orientation_manifest,
                )
            )
    return compact


def _validate_compact_unit_pages(payload: Mapping[str, Any], pages: Sequence[int]) -> None:
    allowed = {int(page) for page in pages}
    for question in payload.get("questions") or []:
        if not isinstance(question, Mapping):
            continue
        regions = question.get("evidence_regions") or []
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            page = _positive_int(region.get("page_number"))
            if page not in allowed:
                raise StructuredGradingOutputError(
                    "Compact mapping unit returned evidence outside its supplied pages"
                )
    for region in payload.get("unassigned_student_regions") or []:
        if isinstance(region, Mapping) and _positive_int(region.get("page_number")) not in allowed:
            raise StructuredGradingOutputError(
                "Compact mapping unit returned unassigned evidence outside its supplied pages"
            )


def _is_output_exhaustion(exc: BaseException) -> bool:
    failure = getattr(exc, "structured_output_failure", None)
    if not isinstance(failure, Mapping):
        return False
    reason = str(failure.get("incomplete_reason") or "").strip().lower()
    status = str(failure.get("completion_status") or "").strip().lower()
    return reason == "max_output_tokens" or status == "incomplete"


def _record_split(
    records: List[Dict[str, Any]],
    unit_id: str,
    pages: Sequence[int],
    catalog: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
) -> None:
    record = {
        "unit_id": unit_id,
        "status": "split",
        "pages": list(pages),
        "question_numbers": [
            int(item.get("question_number")) for item in catalog if item.get("question_number")
        ],
        "children": [dict(child) for child in children],
    }
    for index, existing in enumerate(records):
        if str(existing.get("unit_id") or "") == unit_id:
            records[index] = record
            return
    records.append(record)


def _record_batch_split(
    records: List[Dict[str, Any]],
    batch_id: str,
    items: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
) -> None:
    record = {
        "batch_id": batch_id,
        "status": "split",
        "question_numbers": [
            int(item.get("question_number")) for item in items if item.get("question_number")
        ],
        "children": [dict(child) for child in children],
    }
    for index, existing in enumerate(records):
        if str(existing.get("batch_id") or "") == batch_id:
            records[index] = record
            return
    records.append(record)


def _subset_mapping(mapping: Any, question_numbers: set[int]) -> Any:
    selected = {number: mapping.questions[number] for number in sorted(question_numbers) if number in mapping.questions}
    return type(mapping)(
        document_review=dict(mapping.document_review),
        questions=selected,
        unassigned_regions=list(mapping.unassigned_regions),
        errors=list(mapping.errors),
    )


def _mapping_catalog_question(question: Mapping[str, Any]) -> Dict[str, Any]:
    """Return association context with no answer or marking leakage."""

    objective = _is_objective_question(dict(question))
    return {
        "question_number": _positive_int(question.get("question_number")),
        "question_id": str(question.get("question_id") or ""),
        "question_text": str(question.get("question_text") or "")[:4000],
        "response_kind": "objective_selection" if objective else "worked_response",
        "options": _objective_options(dict(question)) if objective else [],
        "expects_diagram": bool(question.get("expects_diagram")),
    }


def _student_content_for_pages(
    student_content: Sequence[Mapping[str, Any]],
    page_numbers: set[int],
    *,
    source_rotations: Optional[Mapping[int, set[int]]] = None,
) -> List[Dict[str, Any]]:
    """Keep only the physical pages and orientation views owned by the evidence."""

    selected: List[Dict[str, Any]] = []
    include_current_page = True
    saw_page_label = False
    prefix = "Student answer-copy page "
    for raw_item in student_content:
        item = dict(raw_item)
        if item.get("type") == "input_text":
            text = str(item.get("text") or "")
            if text.startswith(prefix):
                saw_page_label = True
                suffix = text[len(prefix) :].split(",", 1)[0].strip()
                page_number = _positive_int(suffix)
                include_current_page = bool(
                    page_number is not None and page_number in page_numbers
                )
                if include_current_page and source_rotations and page_number in source_rotations:
                    match = _ORIENTATION_PAGE_LABEL_RE.match(text)
                    if match:
                        include_current_page = (
                            int(match.group("rotation"))
                            in source_rotations[page_number]
                        )
                if include_current_page:
                    selected.append(item)
                continue
        if not saw_page_label or include_current_page:
            selected.append(item)
    return selected


def _source_rotations_from_mapping(mapping: Any) -> Dict[int, set[int]]:
    """Collect provider-selected views from validated original-frame regions."""

    selected: Dict[int, set[int]] = {}
    for item in getattr(mapping, "questions", {}).values():
        for region in item.get("evidence_regions") or []:
            if not isinstance(region, Mapping):
                continue
            page = _positive_int(region.get("page_number"))
            frame = region.get("coordinate_frame")
            if not page or not isinstance(frame, Mapping):
                continue
            try:
                rotation = int(frame.get("source_rotation_degrees_clockwise"))
            except (TypeError, ValueError):
                continue
            if rotation in {0, 90, 270}:
                selected.setdefault(page, set()).add(rotation)
    return selected


def _build_mapping_responses_input(
    *,
    catalog: Sequence[Mapping[str, Any]],
    paper_bytes: bytes,
    student_content: List[Dict[str, Any]],
    paper_filename: str,
) -> List[Dict[str, Any]]:
    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE MAPPING-ONLY QUESTION CATALOG. It contains no answer key, "
                "solution, rubric, acceptable evidence, or marking criteria:\n"
                + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": "data:application/pdf;base64,"
            + base64.b64encode(paper_bytes).decode("ascii"),
        },
    ]
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": mapping_system_instructions()}],
        },
        {"role": "user", "content": static_content},
        {"role": "user", "content": list(student_content)},
    ]


def _build_compact_mapping_responses_input(
    *,
    catalog: Sequence[Mapping[str, Any]],
    paper_bytes: bytes,
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    page_numbers: Sequence[int],
    prompt_version: str = _V14_PROMPT_VERSION,
    recovery_context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build one bounded mapping request with only ownership context."""

    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "COMPACT MAPPING CATALOG (no answers, solutions, or marks):\n"
                + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": "data:application/pdf;base64,"
            + base64.b64encode(paper_bytes).decode("ascii"),
        },
    ]
    unit_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "MAPPING UNIT SOURCE PAGES: "
            + ",".join(str(item) for item in page_numbers),
        },
    ]
    if recovery_context:
        unit_content.append({
            "type": "input_text",
            "text": (
                "ONE BOUNDED OWNERSHIP-RECOVERY PASS. Reinspect only the supplied "
                "physical pages. Keep correct prior ownership unchanged. If a prior "
                "assigned region actually belongs to another catalog question, or an "
                "unresolved prior region can now be confirmed, return a tight replacement "
                "region and list the exact old region_id in supersedes_region_ids. Use "
                "supersedes only for a visible, overlapping ownership correction or "
                "confirmation; otherwise return an empty list. Reassociate the "
                "listed unassigned regions, find missed student work, and independently "
                "verify questions previously treated as absent. This is mapping only, "
                "not grading:\n"
                + json.dumps(
                    dict(recovery_context),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        })
    unit_content.extend(dict(item) for item in student_content)
    return [
        {
            "role": "developer",
            "content": [{
                "type": "input_text",
                "text": compact_mapping_system_instructions(prompt_version),
            }],
        },
        {"role": "user", "content": static_content},
        {
            "role": "user",
            "content": unit_content,
        },
    ]


def _build_evidence_grading_responses_input(
    *,
    catalog: Sequence[Mapping[str, Any]],
    requested_question_numbers: Optional[Sequence[int]] = None,
    mapping: Any,
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
) -> List[Dict[str, Any]]:
    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE GRADING BLUEPRINT. Question identity, solution, criterion "
                "IDs, acceptable evidence, policies, and maximum marks are locked:\n"
                + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": "data:application/pdf;base64,"
            + base64.b64encode(paper_bytes).decode("ascii"),
        },
    ]
    if solution_bytes:
        static_content.extend(
            [
                {"type": "input_text", "text": "FROZEN TEACHER SOLUTION PDF:"},
                {
                    "type": "input_file",
                    "filename": _safe_pdf_filename(
                        solution_filename, "teacher-solution.pdf"
                    ),
                    "file_data": "data:application/pdf;base64,"
                    + base64.b64encode(solution_bytes).decode("ascii"),
                },
            ]
        )
    dynamic_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "GRADE ONLY QUESTION NUMBERS: "
            + ",".join(
                str(item)
                for item in (
                    requested_question_numbers
                    if requested_question_numbers is not None
                    else sorted(mapping.questions)
                )
            ),
        },
        {
            "type": "input_text",
            "text": (
                "VALIDATED IMMUTABLE STUDENT EVIDENCE MAP. Region ownership and the "
                "mapper-owned student_answer cannot be changed during grading:\n"
                + json.dumps(
                    mapping.as_payload(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        },
        *student_content,
    ]
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": grading_system_instructions()}],
        },
        {"role": "user", "content": static_content},
        {"role": "user", "content": dynamic_content},
    ]


def _build_verification_responses_input(
    *,
    catalog: Sequence[Mapping[str, Any]],
    requested_question_numbers: Sequence[int],
    mapping: Any,
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
) -> List[Dict[str, Any]]:
    """Build audit input without primary marks or primary rationale."""

    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "IMMUTABLE AUDIT QUESTION/RUBRIC:\n" + json.dumps(
                list(catalog), ensure_ascii=False, separators=(",", ":")
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": "data:application/pdf;base64," + base64.b64encode(paper_bytes).decode("ascii"),
        },
    ]
    if solution_bytes:
        static_content.extend([
            {"type": "input_text", "text": "FROZEN TEACHER SOLUTION PDF:"},
            {
                "type": "input_file",
                "filename": _safe_pdf_filename(solution_filename, "teacher-solution.pdf"),
                "file_data": "data:application/pdf;base64" + "," + base64.b64encode(solution_bytes).decode("ascii"),
            },
        ])
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": verification_system_instructions()}],
        },
        {"role": "user", "content": static_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "AUDIT ONLY QUESTION NUMBERS: "
                    + ",".join(str(item) for item in requested_question_numbers),
                },
                {
                    "type": "input_text",
                    "text": "FIXED STUDENT EVIDENCE MAP (no primary marks included):\n"
                    + json.dumps(mapping.as_payload(), ensure_ascii=False, separators=(",", ":")),
                },
                *student_content,
            ],
        },
    ]


def _require_complete_structured_payload(
    response: Any,
    *,
    raw: str,
    stage: str,
    output_limit: int,
) -> Dict[str, Any]:
    completion_failure = _response_completion_failure(response)
    if completion_failure:
        raise StructuredGradingOutputError(
            f"{stage} exhausted its output budget before completing",
            completion_status=completion_failure["completion_status"],
            incomplete_reason=completion_failure["incomplete_reason"],
            max_output_tokens=output_limit,
        )
    payload = _parse_json_object(raw)
    if payload is None:
        raise StructuredGradingOutputError(
            f"{stage} returned invalid structured JSON",
            completion_status=str(
                getattr(response, "completion_status", "completed") or "completed"
            ),
            max_output_tokens=output_limit,
        )
    return payload


def _stage_cache_key(
    stage: str,
    *,
    paper_hash: str,
    solution_hash: Optional[str],
    prompt_version: str = _PROMPT_VERSION,
) -> str:
    material = ":".join(
        [
            prompt_version,
            EVIDENCE_GRAPH_VERSION,
            stage,
            paper_hash,
            solution_hash or "",
        ]
    )
    if prompt_version == _V16_PROMPT_VERSION:
        prefix = "pcr-v16-"
    elif prompt_version == _V15_PROMPT_VERSION:
        prefix = "pcr-v15-"
    elif prompt_version == _V14_PROMPT_VERSION:
        prefix = "pcr-v14-"
    else:
        prefix = "pcr-v13-"
    return prefix + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _evidence_mapping_output_limit(question_count: int) -> int:
    return min(24_000, max(10_000, 1_100 * max(1, int(question_count or 0))))


def _evidence_grading_output_limit(question_count: int) -> int:
    return min(28_000, max(12_000, 1_300 * max(1, int(question_count or 0))))


async def _student_copy_content(
    answer_pages: List[Dict[str, Any]],
    *,
    orientation_recovery: bool = False,
    coordinate_evidence: bool = True,
) -> tuple[List[Dict[str, Any]], int]:
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "STUDENT ANSWER COPY. Inspect every page visually. Page labels below "
                "are authoritative source-page numbers, not question numbers."
            ),
        }
    ]
    total_bytes = 0
    for page in answer_pages:
        page_number = int(page.get("page_number") or 0)
        raw_ref = page.get("raw_image_ref")
        if page_number <= 0 or not isinstance(raw_ref, str) or not raw_ref.strip():
            raise FullDocumentGradingError(
                f"Canonical student page {page_number or '?'} has no image asset"
            )
        image_b64 = await _resolve_image_base64(
            raw_ref,
            expected_sha256=page.get("asset_sha256"),
        )
        if not image_b64:
            raise FullDocumentGradingError(
                f"Canonical student page {page_number} could not be loaded"
            )
        try:
            original = base64.b64decode(image_b64, validate=True)
        except Exception as exc:
            raise FullDocumentGradingError(
                f"Canonical student page {page_number} is not a valid image"
            ) from exc
        media_type = _image_media_type(original)
        if not media_type:
            raise FullDocumentGradingError(
                f"Canonical student page {page_number} has an unsupported image format"
            )
        if not orientation_recovery:
            total_bytes += len(original)
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Student answer-copy page {page_number}, unaltered original image. "
                        "Read it in its natural orientation; do not require pre-cropping."
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{media_type};base64,"
                        + base64.b64encode(original).decode("ascii")
                    ),
                    "detail": "high",
                }
            )
            continue

        try:
            views = build_orientation_views(
                original,
                physical_page_number=page_number,
                width_px=_positive_int(page.get("image_width_px")),
                height_px=_positive_int(page.get("image_height_px")),
            )
        except OrientationViewError as exc:
            raise FullDocumentGradingError(
                f"Canonical student page {page_number} could not be prepared safely"
            ) from exc
        # A strong ruled-line signal proves the original is sideways but cannot
        # prove direction. Send only the two readable candidates, not a third
        # known-sideways copy. The immutable original remains the coordinate
        # frame and is never overwritten.
        model_views = (
            [view for view in views if not view.is_original]
            if len(views) > 1
            else list(views)
        )
        for view in model_views:
            rotation = int(view.rotation_degrees_clockwise)
            frame = view.coordinate_frame
            view_instruction = (
                "Use one readable view only and return coordinates relative to that "
                "displayed view."
                if coordinate_evidence
                else (
                    "Use the readable duplicate to understand the work, but cite only "
                    f"physical source page {page_number}; do not return coordinates."
                )
            )
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Student answer-copy page {page_number}, physical-page orientation "
                        f"view; source_rotation_degrees_clockwise={rotation}; "
                        f"view_id={view.view_id}; alternate_of={view.alternate_of}; "
                        f"original_width_px={frame['original_width_px']}; "
                        f"original_height_px={frame['original_height_px']}. "
                        "Alternate views with the same alternate_of value are the same "
                        f"physical page. {view_instruction}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        "data:image/png;base64,"
                        + base64.b64encode(view.image_bytes).decode("ascii")
                        if not view.is_original
                        else f"data:{media_type};base64,"
                        + base64.b64encode(view.image_bytes).decode("ascii")
                    ),
                    "detail": "high",
                }
            )
            total_bytes += len(view.image_bytes)
    return content, total_bytes


def _image_media_type(image_bytes: bytes) -> Optional[str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None




def _usage_dict(response: Any, *, fallback_model: str) -> Dict[str, Any]:
    usage_obj = getattr(response, "usage", None)
    return {
        "model": str(getattr(usage_obj, "model", fallback_model) or fallback_model),
        "caller": str(getattr(usage_obj, "caller", _CALLER_ID) or _CALLER_ID),
        "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
        "cache_read_tokens": int(
            getattr(usage_obj, "cache_read_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        "estimated_cost_usd": float(
            getattr(usage_obj, "estimated_cost_usd", 0.0) or 0.0
        ),
    }


def _aggregate_usages(
    usages: Iterable[Mapping[str, Any]],
    *,
    fallback_model: str,
) -> Dict[str, Any]:
    items = [dict(item) for item in usages if isinstance(item, Mapping) and item]
    model = next(
        (str(item.get("model")) for item in reversed(items) if item.get("model")),
        fallback_model,
    )
    return {
        "model": model,
        "caller": _CALLER_ID,
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in items),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in items),
        "cache_read_tokens": sum(
            int(item.get("cache_read_tokens") or 0) for item in items
        ),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in items),
        "estimated_cost_usd": round(
            sum(float(item.get("estimated_cost_usd") or 0.0) for item in items),
            8,
        ),
        "stage_count": len(items),
    }




def _response_completion_failure(response: Any) -> Optional[Dict[str, str]]:
    status = str(getattr(response, "completion_status", "completed") or "completed")
    if status == "completed":
        return None
    return {
        "completion_status": status,
        "incomplete_reason": str(getattr(response, "incomplete_reason", "") or ""),
    }


def _validate_ledger(
    payload: Dict[str, Any],
    *,
    questions: List[Dict[str, Any]],
    page_count: int,
) -> tuple[List[_ValidatedGrade], List[str], _DocumentReview]:
    structural_errors = [
        str(error).strip()[:500]
        for error in (payload.get("validation_errors") or [])
        if str(error).strip()
    ]
    raw_document_review = payload.get("document_review")
    if not isinstance(raw_document_review, Mapping):
        raw_document_review = {}
    document_warnings: List[str] = [
        str(warning).strip()[:500]
        for warning in (raw_document_review.get("warnings") or [])
        if str(warning).strip()
    ]
    all_student_work_accounted = bool(
        raw_document_review.get("all_student_work_accounted")
    )
    if not all_student_work_accounted:
        document_warnings.append(
            "Not all visible student work was assigned to a catalog question"
        )
    document_review = _DocumentReview(
        all_student_work_accounted=(
            all_student_work_accounted and not structural_errors
        ),
        confidence=(1.0 if all_student_work_accounted and not structural_errors else 0.0),
        warnings=document_warnings,
        required=bool(structural_errors) or not all_student_work_accounted,
    )
    if structural_errors:
        document_review.required = True
        document_warnings.append(
            "The whole-copy grading result has structural validation errors"
        )
    document_warnings[:] = list(dict.fromkeys(document_warnings))
    candidates: Dict[int, List[Dict[str, Any]]] = {}
    for item in payload.get("questions") or []:
        if not isinstance(item, dict):
            continue
        number = _positive_int(item.get("question_number"))
        if number:
            candidates.setdefault(number, []).append(item)

    grades: List[_ValidatedGrade] = []
    for position, question in enumerate(questions, start=1):
        number = _positive_int(question.get("question_number")) or position
        matches = candidates.get(number, [])
        if len(matches) != 1:
            reason = (
                "Model returned no result for this question"
                if not matches
                else "Model returned duplicate results for this question"
            )
            grades.append(_unresolved_grade(question, number, reason))
            continue
        grade = _validate_question_grade(
            matches[0],
            question=question,
            question_number=number,
            page_count=page_count,
        )
        grades.append(grade)

    expected_numbers = {
        _positive_int(question.get("question_number")) or position
        for position, question in enumerate(questions, start=1)
    }
    unexpected = sorted(set(candidates) - expected_numbers)
    if unexpected:
        document_warnings.append(
            "Model returned non-catalog question numbers: "
            + ", ".join(str(value) for value in unexpected)
        )
        document_review.required = True
    return grades, structural_errors, document_review


def _not_applicable_method_analysis() -> Dict[str, Any]:
    return {
        "detected_method": "",
        "method_classification": "not_applicable",
        "method_validity": "not_applicable",
        "method_requirement_satisfied": True,
        "confidence": 1.0,
        "explanation": "No student method needs to be assessed for this answer state.",
        "error_carried_forward": "not_applicable",
        "error_carried_forward_reason": "",
    }


def _validate_question_grade(
    item: Dict[str, Any],
    *,
    question: Dict[str, Any],
    question_number: int,
    page_count: int,
) -> _ValidatedGrade:
    status = str(item.get("attempt_status") or "unresolved").strip().lower()
    if status not in {"attempted", "not_attempted", "unresolved"}:
        status = "unresolved"
    confidence = _bounded_confidence(item.get("confidence"))
    student_answer = str(item.get("student_answer") or "").strip()
    content_type = str(item.get("content_type") or ContentType.MIXED.value).upper()
    if content_type not in {value.value for value in ContentType}:
        content_type = ContentType.MIXED.value
    source_pages, region_errors = _validate_question_source_pages(
        item,
        question_number=question_number,
        page_count=page_count,
    )
    validation_errors = list(region_errors)
    evidence_region_ids = {
        str(region.get("region_id") or "")
        for region in source_pages
        if str(region.get("region_id") or "")
    }
    max_marks = _max_marks(question)
    criteria = _criteria(question)
    method_analysis = _not_applicable_method_analysis()
    criterion_marks: List[Dict[str, Any]] = []
    total_score: Optional[float] = None
    manual_review = bool(item.get("needs_review"))
    review_reason = str(item.get("review_reason") or "").strip()
    objective_question = _is_objective_question(question)

    if status == "unresolved":
        return _unresolved_grade(
            question,
            question_number,
            review_reason or "The model could not verify this answer state",
            confidence=confidence,
            source_pages=source_pages,
            student_answer=student_answer,
            content_type=content_type,
        )

    if status == "not_attempted":
        raw_total = _finite_float(item.get("total_score"))
        raw_marks = item.get("criterion_marks")
        if (
            student_answer
            or source_pages
            or (isinstance(raw_marks, list) and bool(raw_marks))
            or raw_total is None
            or abs(raw_total) > 0.01
            or manual_review
        ):
            return _unresolved_grade(
                question,
                question_number,
                "The model returned contradictory or uncertain evidence for a "
                "not-attempted decision",
                confidence=confidence,
                source_pages=source_pages,
                student_answer=student_answer,
                content_type=content_type,
            )
        criterion_marks = [
            {
                "criterion_id": criterion["criterion_id"],
                "description": criterion["description"],
                "marks_awarded": 0.0,
                "max_marks": criterion["max_marks"],
                "decision": "not_met",
                "confidence": 1.0,
                "rationale": "No student attempt was found after reviewing the full copy.",
                "evidence": "No student evidence located on any submitted page.",
                "missing_evidence": criterion.get("acceptable_evidence") or "",
                "credit_basis": "no_credit",
            }
            for criterion in criteria
        ]
        return _ValidatedGrade(
            question=question,
            question_number=question_number,
            attempt_status="not_attempted",
            confidence=confidence,
            student_answer="",
            content_type=ContentType.TEXT_ONLY.value,
            source_pages=[],
            method_analysis=method_analysis,
            criterion_marks=criterion_marks,
            total_score=0.0,
            overall_feedback=(
                str(item.get("overall_feedback") or "Question not attempted.").strip()
            ),
            manual_review_required=False,
            review_reason="",
        )

    if not student_answer and objective_question:
        validation_errors.append("Attempted answer has no student transcription")
    if not source_pages:
        validation_errors.append("Attempted answer has no visual evidence region")
    if objective_question:
        if validation_errors:
            return _unresolved_grade(
                question,
                question_number,
                "; ".join(dict.fromkeys(validation_errors)),
                confidence=confidence,
                source_pages=source_pages,
                student_answer=student_answer,
                content_type=content_type,
            )
        try:
            objective_result = score_objective_response(question, student_answer)
        except ObjectiveScoringContractError as exc:
            return _unresolved_grade(
                question,
                question_number,
                str(exc),
                confidence=confidence,
                source_pages=source_pages,
                student_answer=student_answer,
                content_type=content_type,
            )
        selected = str(objective_result["selected_answer"])
        correct = str(objective_result["correct_answer"])
        points_earned = float(objective_result["points_earned"])
        return _ValidatedGrade(
            question=question,
            question_number=question_number,
            attempt_status="attempted",
            confidence=confidence,
            student_answer=selected,
            content_type=content_type,
            source_pages=source_pages,
            method_analysis=_not_applicable_method_analysis(),
            criterion_marks=[],
            total_score=points_earned,
            overall_feedback=(
                f"Selected {selected}. Correct answer: {correct}."
                if objective_result["is_correct"]
                else (
                    f"Selected {selected}. Correct answer: {correct}. "
                    f"{objective_result['penalty_marks']:g} mark(s) deducted."
                )
            ),
            manual_review_required=manual_review,
            review_reason=review_reason,
        )

    raw_marks_value = item.get("criterion_marks")
    if isinstance(raw_marks_value, Mapping):
        raw_marks = [
            {"criterion_id": str(criterion_id), **dict(score)}
            for criterion_id, score in raw_marks_value.items()
            if isinstance(score, Mapping)
        ]
    else:
        raw_marks = (
            [dict(raw) for raw in raw_marks_value if isinstance(raw, Mapping)]
            if isinstance(raw_marks_value, list)
            else []
        )
    if not student_answer:
        evidence_fragments: List[str] = []
        for region in source_pages:
            fragment = str(region.get("evidence") or "").strip()
            if fragment and fragment not in evidence_fragments:
                evidence_fragments.append(fragment)
        if evidence_fragments:
            student_answer = " ".join(evidence_fragments)[:4000]
        else:
            student_answer = "Visible work is present on the cited answer page."
            manual_review = True
            review_reason = review_reason or (
                "The work was graded visually, but its text transcription is incomplete"
            )

    expected_ids = [str(criterion["criterion_id"]) for criterion in criteria]
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for position, raw in enumerate(raw_marks):
        fallback_id = expected_ids[position] if position < len(expected_ids) else ""
        criterion_id = str(raw.get("criterion_id") or fallback_id).strip()
        if not criterion_id:
            continue
        if criterion_id in raw_by_id:
            duplicate_ids.add(criterion_id)
            continue
        raw_by_id[criterion_id] = raw
    returned_ids = set(raw_by_id)
    if criteria and (
        duplicate_ids
        or returned_ids != set(expected_ids)
    ):
        validation_errors.append(
            "Criterion results do not match the locked marking plan"
        )
    for criterion in criteria:
        criterion_id = str(criterion["criterion_id"])
        raw = raw_by_id.get(criterion_id)
        if raw is None:
            continue
        criterion_confidence = _bounded_confidence(raw.get("confidence"))
        awarded = _finite_float(raw.get("marks_awarded"))
        if awarded is None or awarded < 0 or awarded > criterion["max_marks"]:
            validation_errors.append(
                f"Criterion {criterion_id} award is outside its locked range"
            )
            continue
        maximum = criterion["max_marks"]
        if abs(awarded - maximum) <= 0.01:
            decision = "met"
        elif awarded <= 0.01:
            decision = "not_met"
        else:
            decision = "partially_met"
        rationale = str(raw.get("rationale") or "").strip()
        evidence = str(raw.get("evidence") or "").strip()
        if not rationale:
            rationale = {
                "met": "Correct.",
                "partially_met": "Part of the required step is correct.",
                "not_met": "The required step is not shown correctly.",
            }[decision]
        evidence = evidence or student_answer[:500]
        raw_cited_ids = raw.get("evidence_region_ids")
        cited_region_ids = list(
            dict.fromkeys(
                str(region_id).strip()
                for region_id in (
                    raw_cited_ids if isinstance(raw_cited_ids, list) else []
                )
                if str(region_id).strip()
            )
        )
        if not cited_region_ids:
            validation_errors.append(
                f"Criterion {criterion_id} does not cite mapped student evidence"
            )
        elif not set(cited_region_ids).issubset(evidence_region_ids):
            validation_errors.append(
                f"Criterion {criterion_id} cites evidence owned by another question"
            )
        missing_evidence = ""
        if decision != "met":
            missing_evidence = str(
                criterion.get("acceptable_evidence")
                or "The remaining required work was not demonstrated."
            ).strip()
        credit_basis = str(raw.get("credit_basis") or "").strip().lower()
        if awarded <= 0.01:
            credit_basis = "no_credit"
        elif credit_basis == "error_carried_forward" and _question_method_policy(
            question
        ).get("allow_error_carried_forward", True):
            credit_basis = "error_carried_forward"
        else:
            credit_basis = "direct_evidence"
        criterion_marks.append(
            {
                "criterion_id": criterion_id,
                "description": criterion["description"],
                "marks_awarded": round(awarded, 2),
                "max_marks": maximum,
                "decision": decision,
                "confidence": criterion_confidence,
                "rationale": rationale,
                "evidence": evidence,
                "evidence_region_ids": cited_region_ids,
                "missing_evidence": missing_evidence,
                "credit_basis": credit_basis,
            }
        )
    if criteria and len(criterion_marks) == len(criteria):
        total_score = round(sum(mark["marks_awarded"] for mark in criterion_marks), 2)
        # The model-reported total is advisory. The server-owned criterion sum
        # is the only authoritative question total.
    elif not criteria:
        raw_total = _finite_float(item.get("total_score"))
        if raw_total is None or raw_total < 0 or raw_total > max_marks:
            validation_errors.append("Question total is outside its locked range")
        else:
            total_score = round(raw_total, 2)
            criterion_marks = [
                {
                    "criterion_id": "overall",
                    "description": "Overall response",
                    "marks_awarded": total_score,
                    "max_marks": max_marks,
                    "rationale": str(item.get("overall_feedback") or "").strip(),
                    "evidence": student_answer[:500],
                    "evidence_region_ids": sorted(evidence_region_ids),
                    "credit_basis": "direct_evidence" if total_score > 0 else "no_credit",
                }
            ]

    # Method compliance is enforced by the locked criterion rows that award
    # marks for demonstrating a method.  It must not globally invalidate an
    # otherwise complete score: many questions name an operation while their
    # rubric awards marks only for correct, independently verifiable results.

    if validation_errors:
        return _unresolved_grade(
            question,
            question_number,
            "; ".join(dict.fromkeys(validation_errors)),
            confidence=confidence,
            source_pages=source_pages,
            student_answer=student_answer,
            content_type=content_type,
        )

    return _ValidatedGrade(
        question=question,
        question_number=question_number,
        attempt_status="attempted",
        confidence=confidence,
        student_answer=student_answer,
        content_type=content_type,
        source_pages=source_pages,
        method_analysis=method_analysis,
        criterion_marks=criterion_marks,
        total_score=total_score,
        overall_feedback=str(item.get("overall_feedback") or "").strip(),
        manual_review_required=manual_review,
        review_reason=review_reason,
    )


def _validate_question_source_pages(
    item: Mapping[str, Any],
    *,
    question_number: int,
    page_count: int,
) -> tuple[List[Dict[str, Any]], List[str]]:
    raw_pages = item.get("source_pages")
    if not isinstance(raw_pages, list):
        return [], ["Source pages must be an array"]
    regions: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_page_numbers: set[int] = set()
    seen_region_ids: set[str] = set()
    for value in raw_pages:
        if isinstance(value, Mapping):
            page_number = _positive_int(value.get("page_number"))
        else:
            page_number = _positive_int(value)
        if not page_number or page_number > page_count:
            errors.append("Answer refers to a non-submitted page")
            continue
        if not isinstance(value, Mapping):
            # Compatibility for persisted v12 ledgers only. New v13 output is
            # required to provide stable two-dimensional evidence regions.
            if page_number in seen_page_numbers:
                continue
            seen_page_numbers.add(page_number)
            regions.append(
                {
                    "region_id": f"q{question_number}-legacy-page-{page_number}",
                    "page_number": page_number,
                    "x_start": 0.0,
                    "y_start": 0.0,
                    "x_end": _A4_WIDTH_MM,
                    "y_end": _A4_HEIGHT_MM,
                    "coordinate_space": "original_page_mm",
                    "evidence": "Legacy complete-page evidence reference.",
                }
            )
            continue

        coordinate_space = str(
            value.get("coordinate_space") or "normalized_1000"
        ).strip()
        if coordinate_space == "normalized_1000":
            x_max, y_max = 1000.0, 1000.0
        elif coordinate_space == "original_page_mm":
            x_max, y_max = _A4_WIDTH_MM, _A4_HEIGHT_MM
        else:
            errors.append("Answer evidence uses an unsupported coordinate space")
            continue
        coordinates = {
            key: _finite_float(value.get(key))
            for key in ("x_start", "y_start", "x_end", "y_end")
        }
        if any(coordinate is None for coordinate in coordinates.values()):
            errors.append("Answer evidence is missing two-dimensional coordinates")
            continue
        x_start = float(coordinates["x_start"])
        y_start = float(coordinates["y_start"])
        x_end = float(coordinates["x_end"])
        y_end = float(coordinates["y_end"])
        if (
            x_start < 0
            or y_start < 0
            or x_end > x_max
            or y_end > y_max
            or x_end <= x_start
            or y_end <= y_start
        ):
            errors.append("Answer evidence has an invalid two-dimensional region")
            continue
        region_id = str(value.get("region_id") or "").strip()
        if not region_id:
            errors.append("Answer evidence has no stable region ID")
            continue
        if region_id in seen_region_ids:
            errors.append("Answer evidence repeats a region ID")
            continue
        seen_region_ids.add(region_id)
        regions.append(
            {
                "region_id": region_id[:120],
                "page_number": page_number,
                "x_start": round(x_start, 3),
                "y_start": round(y_start, 3),
                "x_end": round(x_end, 3),
                "y_end": round(y_end, 3),
                "coordinate_space": coordinate_space,
                "coordinate_frame": dict(value.get("coordinate_frame"))
                if isinstance(value.get("coordinate_frame"), Mapping)
                else value.get("coordinate_frame"),
                "evidence_kind": str(value.get("evidence_kind") or "mixed")[:40],
                "authorship": str(value.get("authorship") or "student")[:40],
                "continuation_group": str(
                    value.get("continuation_group") or ""
                )[:120],
                "sequence": max(1, _positive_int(value.get("sequence")) or 1),
                "evidence": str(value.get("evidence") or "").strip()[:1000],
                "diagram_components": [
                    str(item).strip()[:300]
                    for item in (value.get("diagram_components") or [])
                    if str(item).strip()
                ][:30],
                "mapping_confidence": _bounded_confidence(
                    value.get("mapping_confidence")
                ),
            }
        )
    return regions, errors


def _unresolved_grade(
    question: Dict[str, Any],
    question_number: int,
    reason: str,
    *,
    confidence: float = 0.0,
    source_pages: Optional[List[Dict[str, Any]]] = None,
    student_answer: str = "",
    content_type: str = ContentType.MIXED.value,
) -> _ValidatedGrade:
    return _ValidatedGrade(
        question=question,
        question_number=question_number,
        attempt_status="unresolved",
        confidence=confidence,
        student_answer=student_answer,
        content_type=content_type,
        source_pages=source_pages or [],
        method_analysis={
            **_not_applicable_method_analysis(),
            "method_classification": "unresolved",
            "method_validity": "unresolved",
            "method_requirement_satisfied": False,
            "confidence": confidence,
            "explanation": reason[:800],
            "error_carried_forward": "unresolved",
        },
        criterion_marks=[],
        total_score=None,
        overall_feedback="No verified answer state exists for this question.",
        manual_review_required=True,
        review_reason=reason[:800],
        validation_errors=[reason[:800]],
    )


def _catalog_question(question: Dict[str, Any]) -> Dict[str, Any]:
    policy = _question_marking_policy(question)
    method_policy = _question_method_policy(question)
    objective = _is_objective_question(question)
    assessment_units: List[Dict[str, Any]] = []
    assessment_units_invalid = False
    if not objective:
        try:
            assessment_units = normalize_assessment_units(
                question.get("assessment_units"),
                assign_missing_ids=False,
            )
        except (TypeError, ValueError):
            assessment_units_invalid = True
    return {
        "question_number": _positive_int(question.get("question_number")),
        "question_id": str(question.get("question_id") or ""),
        "question_text": str(question.get("question_text") or "")[:4000],
        "max_marks": _max_marks(question),
        "grading_mode": "objective" if objective else "subjective",
        "answer_format": "option_label" if objective else "worked_response",
        "options": _objective_options(question) if objective else [],
        "reference_solution": (
            "" if objective else _reference_solution(question)[:5000]
        ),
        "marking_criteria": [] if objective else _criteria(question),
        "assessment_units": assessment_units,
        "assessment_units_invalid": assessment_units_invalid,
        "marking_policy": policy,
        "method_policy": method_policy,
        "method_standard": method_policy_instruction(method_policy),
        "marking_standard": strictness_instruction(
            str(policy.get("strictness") or "balanced")
        ),
        "expects_diagram": bool(question.get("expects_diagram")),
    }


def _is_objective_question(question: Dict[str, Any]) -> bool:
    return str(
        question.get("grading_mode")
        or question.get("question_type")
        or ""
    ).strip().lower() in {"objective", "mcq", "integer"}


def _objective_options(question: Dict[str, Any]) -> List[Dict[str, str]]:
    options = question.get("options")
    if not isinstance(options, list):
        return []
    normalized: List[Dict[str, str]] = []
    for index, option in enumerate(options):
        if isinstance(option, dict):
            label = str(option.get("label") or chr(ord("A") + index)).strip().upper()
            text = str(
                option.get("text")
                or option.get("content")
                or option.get("value")
                or ""
            ).strip()
        else:
            label = chr(ord("A") + index)
            text = str(option or "").strip()
        if text:
            normalized.append({"label": label, "text": text[:2000]})
    return normalized


def _validate_question_catalog(questions: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen_numbers: set[int] = set()
    seen_ids: set[str] = set()
    for position, question in enumerate(questions, start=1):
        question_id = str(question.get("question_id") or "").strip()
        number = _positive_int(question.get("question_number")) or position
        if question_id in seen_ids:
            errors.append(f"duplicate question_id {question_id}")
        seen_ids.add(question_id)
        if number in seen_numbers:
            errors.append(f"duplicate question number Q{number}")
        seen_numbers.add(number)
        max_marks = _max_marks(question)
        if max_marks <= 0:
            errors.append(f"Q{number} has no positive maximum mark")
        criteria = _criteria(question)
        assessment_units = _assessment_units(question)
        if bool(question.get("assessment_units_invalid")):
            errors.append(f"Q{number} has invalid assessment-unit metadata")
        if assessment_units:
            unit_errors = validate_assessment_units(
                assessment_units,
                max_marks,
                require_reference_solution=True,
            )
            errors.extend(f"Q{number} {error}" for error in unit_errors)
            projected_criteria = normalize_marking_criteria(
                flatten_assessment_unit_criteria(assessment_units),
                assign_missing_ids=False,
            )
            saved_criteria = normalize_marking_criteria(
                criteria,
                assign_missing_ids=False,
            )
            if projected_criteria != saved_criteria:
                errors.append(
                    f"Q{number} assessment-unit criteria projection is out of sync"
                )
        criterion_ids = [item["criterion_id"] for item in criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            errors.append(f"Q{number} has duplicate locked criterion IDs")
        if criteria:
            for criterion in criteria:
                if criterion["max_marks"] <= 0:
                    errors.append(
                        f"Q{number} criterion {criterion['criterion_id']} has no positive mark"
                    )
                if not criterion["description"]:
                    errors.append(
                        f"Q{number} criterion {criterion['criterion_id']} has no description"
                    )
                if not criterion["acceptable_evidence"]:
                    errors.append(
                        f"Q{number} criterion {criterion['criterion_id']} has no acceptable evidence"
                    )
            criterion_total = round(sum(item["max_marks"] for item in criteria), 2)
            if abs(criterion_total - max_marks) > 0.01:
                errors.append(
                    f"Q{number} criterion maximums total {criterion_total:g}, "
                    f"question maximum is {max_marks:g}"
                )
    return errors


def _criteria(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        normalized = normalize_marking_criteria(
            question.get("marking_criteria"),
            assign_missing_ids=False,
        )
    except (TypeError, ValueError):
        return []
    criteria: List[Dict[str, Any]] = []
    for item in normalized:
        criterion_id = str(item.get("criterion_id") or "").strip()
        max_marks = _finite_float(item.get("max_marks"))
        if not criterion_id or max_marks is None or max_marks < 0:
            continue
        description = str(item.get("description") or "").strip()
        acceptable_evidence = str(
            item.get("acceptable_evidence")
            or item.get("expected_evidence")
            or item.get("evidence")
            or description
        ).strip()
        criteria.append(
            {
                "criterion_id": criterion_id,
                "description": description,
                "max_marks": round(max_marks, 2),
                "acceptable_evidence": acceptable_evidence,
            }
        )
    return criteria


def _assessment_units(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return normalize_assessment_units(
            question.get("assessment_units"),
            assign_missing_ids=False,
        )
    except (TypeError, ValueError):
        return []


def _question_marking_policy(question: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return normalize_marking_policy(question.get("marking_policy"))
    except (TypeError, ValueError):
        return normalize_marking_policy(None)


def _question_method_policy(question: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return normalize_method_policy(question.get("method_policy"))
    except (TypeError, ValueError):
        return normalize_method_policy(None)


def _max_marks(question: Dict[str, Any]) -> float:
    value = _finite_float(question.get("max_marks"))
    return round(max(0.0, value or 0.0), 2)


def _reference_solution(question: Dict[str, Any]) -> str:
    return str(
        question.get("reference_solution")
        or question.get("teacher_reference_solution")
        or ""
    ).strip()


def _review_flag(
    response_id: str,
    *,
    severity: str,
    reason: str,
    prompt_version: str,
) -> Dict[str, Any]:
    return {
        "flag_id": _stable_id("FLG-DOC", response_id, reason),
        "response_id": response_id,
        "source": "full_document_visual",
        "flag_type": "llm_score_divergence",
        "severity": severity,
        "reason": reason,
        "suggested_action": "Review the cited pages against the original answer copy",
        "metadata": {"prompt_version": prompt_version},
    }


def _text_coverage_for_type(content_type: str) -> float:
    return {
        ContentType.TEXT_ONLY.value: 1.0,
        ContentType.MIXED.value: 0.6,
        ContentType.DIAGRAM_HEAVY.value: 0.2,
        ContentType.TABLE_PRESENT.value: 0.5,
    }.get(content_type, 0.5)


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        parsed = json.loads(cleaned)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _input_fingerprint(
    *,
    submission_id: str,
    exam: Dict[str, Any],
    answer_pages: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model_id: str,
    paper_hash: str,
    solution_hash: Optional[str],
    temperature: float,
    reasoning_effort: str,
    prompt_version: str,
) -> str:
    payload = {
        "version": prompt_version,
        "model": model_id,
        # Student grading output is never content-addressed across people. The
        # immutable submission remains the ownership boundary. The separate
        # generation fingerprint adds explicit operator reprocess intent.
        "submission_id": submission_id,
        "exam_id": exam.get("exam_id"),
        "paper_version_id": exam.get("paper_version_id"),
        "paper_hash": paper_hash,
        "solution_hash": solution_hash,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "question_catalog": [_catalog_question(question) for question in questions],
        "pages": [
            [
                page.get("page_number"),
                page.get("asset_sha256")
                or page.get("content_hash")
                or page.get("page_id")
                or page.get("raw_image_ref"),
            ]
            for page in answer_pages
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _generation_fingerprint(
    *,
    submission_id: str,
    input_fingerprint: str,
    generation_revision: int,
) -> str:
    """Derive one paid-call identity from immutable input plus operator intent."""

    payload = {
        "version": "pcr-grading-generation-v1",
        "submission_id": submission_id,
        "input_fingerprint": input_fingerprint,
        "generation_revision": max(0, int(generation_revision)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _semantic_evidence_signature(
    *,
    question_id: str,
    student_answer: str,
    source_pages: Sequence[Mapping[str, Any]],
    visual_evidence: Mapping[str, Any],
    prompt_version: str,
) -> str:
    """Fingerprint model-interpreted evidence without including awarded marks.

    This is an audit/calibration key, not plagiarism detection and not a source
    of marks. Equivalent ledgers can be compared across a cohort without joining
    or reusing student-owned response rows.
    """

    payload = {
        "question_id": question_id,
        "prompt_version": prompt_version,
        "student_answer": " ".join(student_answer.lower().split()),
        "regions": [
            {
                key: region.get(key)
                for key in (
                    "page_number",
                    "x_start",
                    "y_start",
                    "x_end",
                    "y_end",
                    "evidence_kind",
                )
            }
            for region in source_pages
        ],
        "interpretation_hypotheses": visual_evidence.get(
            "interpretation_hypotheses"
        )
        or [],
        "visual_semantics": visual_evidence.get("visual_semantics") or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _grading_consistency_key(
    *,
    question_id: str,
    student_answer: str,
    method_analysis: Mapping[str, Any],
    prompt_version: str,
    model_used: str,
) -> str:
    """Key equivalent normalized work within one immutable grading contract.

    Coordinates, page numbers, handwriting style, confidence, and awarded marks
    are deliberately excluded. Exact normalized work can therefore share a
    cohort precedent, while different steps or methods remain independent.
    """

    normalized_answer = " ".join(str(student_answer or "").casefold().split())
    if len(normalized_answer) < 2:
        return ""
    payload = {
        "version": "pcr-cohort-consistency-v1",
        "question_id": str(question_id or ""),
        "student_answer": normalized_answer,
        "method_analysis": {
            key: method_analysis.get(key)
            for key in (
                "detected_method",
                "method_classification",
                "method_validity",
                "method_requirement_satisfied",
                "error_carried_forward",
            )
        },
        "prompt_version": str(prompt_version or ""),
        "model_used": str(model_used or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _safe_pdf_filename(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name
    if not name.lower().endswith(".pdf"):
        name = fallback
    return name[:160]


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bounded_confidence(value: Any) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))


def _temperature(value: Any) -> float:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0.0 or parsed > 2.0:
        raise FullDocumentGradingError(
            "Immutable PCR grading contract has an invalid sampling temperature"
        )
    return round(parsed, 2)


def _contract_temperature(contract: Dict[str, Any]) -> Optional[float]:
    if "temperature" not in contract:
        return None
    return _temperature(contract.get("temperature"))


def _grading_temperature(questions: List[Dict[str, Any]]) -> float:
    values = {
        _temperature(_question_marking_policy(question).get("temperature", 0.10))
        for question in questions
    }
    if not values:
        return 0.10
    if len(values) != 1:
        raise FullDocumentGradingError(
            "One full-document grading request cannot mix question temperatures"
        )
    return next(iter(values))


def _result_from_run(
    run: Dict[str, Any],
    submission_id: str,
) -> FullDocumentGradingResult:
    result = dict(run.get("result") or {})
    blocked = int(result.get("blocked_count") or 0)
    warnings = int(result.get("warning_count") or 0)
    return FullDocumentGradingResult(
        handled=True,
        submission_id=submission_id,
        status="completed",
        page_count=int(run.get("page_count") or 0),
        response_count=int(result.get("response_count") or 0),
        evaluated_count=int(result.get("evaluated_count") or 0),
        blocked_count=blocked,
        warning_count=warnings,
        run_id=str(run.get("run_id") or "") or None,
        errors=[str(value) for value in (result.get("errors") or [])],
        document_review_required=bool(
            result.get("document_review_required")
        ),
        review_state=str(result.get("review_state") or "ready"),
        review_reasons=[
            str(value) for value in (result.get("review_reasons") or [])
        ],
    )
