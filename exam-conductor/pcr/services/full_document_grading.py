"""Submission-level visual grading for PCR answer copies.

This is the primary camera/PDF path for papers where handwriting, diagrams,
tables, and answer ownership cannot safely be reduced to OCR text first. One
whole-copy request receives the immutable question paper, teacher solution
(when present), and every unaltered student page. At most one bounded recovery
request rechecks only genuinely unresolved questions.

Deterministic code does not decide what the handwriting means.  It validates
the model's evidence ledger against immutable question IDs, page bounds, and
locked mark limits.  Missing or ambiguous evidence becomes ``unresolved`` and
blocks publication; it never becomes an inferred zero.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from pymongo.errors import DuplicateKeyError
from services.objective_scoring_service import (
    ObjectiveScoringContractError,
    score_objective_response,
)
from services.page_orientation import detect_sideways_page

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

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "pcr-full-document-visual-v12"
_SUPPORTED_PROMPT_VERSIONS = {_PROMPT_VERSION}
_RUNS_COLLECTION = "evalpen_document_grading_runs"
_PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
_CALLER_ID = "pcr_eval_core"
_DEFAULT_REASONING_EFFORT = "medium"
_MAX_PAGE_COUNT = 50
_MAX_STATIC_PDF_BYTES = 45 * 1024 * 1024
_MAX_REQUEST_PAYLOAD_BYTES = 45 * 1024 * 1024
_A4_WIDTH_MM = 210.0
_A4_HEIGHT_MM = 297.0


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
    ) -> None:
        super().__init__(message)
        self.structured_output_failure = {
            "completion_status": completion_status or "unknown",
            "incomplete_reason": incomplete_reason,
            "max_output_tokens": max(0, int(max_output_tokens or 0)),
        }


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
        grading_contract = dict(exam.get("pcr_grading_contract") or {})
        contract_version = str(grading_contract.get("prompt_version") or "").strip()
        prompt_version = _PROMPT_VERSION
        if contract_version and contract_version not in _SUPPORTED_PROMPT_VERSIONS:
            raise UnsupportedGradingContractError(
                "This exam is locked to grading contract "
                f"{contract_version}, which this worker does not support. "
                "Do not mix grading contracts within one exam; migrate and reprocess "
                "the complete exam together."
            )
        if contract_version and contract_version != prompt_version:
            raise UnsupportedGradingContractError(
                "The immutable paper requires grading contract "
                f"{prompt_version}, but this cohort is locked to {contract_version}. "
                "Migrate and reprocess the complete cohort; never mix grading "
                "contracts student by student."
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
            primary_output_limit = _whole_copy_output_limit(len(questions))
            gate_response: Any = None
            try:
                request_input = _build_responses_input(
                    questions=questions,
                    paper_bytes=paper_bytes,
                    solution_bytes=solution_bytes,
                    student_content=student_content,
                    paper_filename=str(
                        document.get("filename") or "question-paper.pdf"
                    ),
                    solution_filename=str(
                        document.get("answer_sheet_filename")
                        or "teacher-solution.pdf"
                    ),
                )
                gate_response = await self._gate.call(
                    model_id=model_id,
                    prompt="",
                    caller_id=_CALLER_ID,
                    responses_input=request_input,
                    json_schema=_whole_copy_schema(questions),
                    prompt_cache_key=(
                        "pcr-paper-"
                        + _static_context_hash(
                            exam,
                            paper_hash=paper_file_hash,
                            solution_hash=solution_file_hash,
                            prompt_version=prompt_version,
                        )[:32]
                    ),
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_output_tokens=primary_output_limit,
                    metadata={
                        "pcr_stage": "full_document_visual_grading",
                        "prompt_version": prompt_version,
                        "submission_id": submission_id,
                        "exam_id": exam_id,
                        "question_count": len(questions),
                        "page_count": len(answer_pages),
                        "run_id": run_id,
                        "provider_call_number": 1,
                        "provider_call_limit": 2,
                    },
                )
                completion_failure = _response_completion_failure(gate_response)
                if completion_failure:
                    raise StructuredGradingOutputError(
                        "The model exhausted its output budget before returning the "
                        "complete grading ledger",
                        completion_status=completion_failure["completion_status"],
                        incomplete_reason=completion_failure["incomplete_reason"],
                        max_output_tokens=primary_output_limit,
                    )
                raw_llm = str(getattr(gate_response, "content", "") or "")
                raw_payload = _parse_json_object(raw_llm)
                if raw_payload is None:
                    raise StructuredGradingOutputError(
                        "The model returned invalid structured grading JSON",
                        completion_status=str(
                            getattr(gate_response, "completion_status", "completed")
                            or "completed"
                        ),
                        max_output_tokens=primary_output_limit,
                    )
            except Exception as exc:
                failure_update: Dict[str, Any] = {
                    "status": "failed",
                    "generation_error": str(exc)[:500],
                    "updated_at": datetime.now(timezone.utc),
                }
                if gate_response is not None:
                    failure_update["token_usage"] = _usage_dict(
                        gate_response,
                        fallback_model=model_id,
                    )
                structured_failure = getattr(exc, "structured_output_failure", None)
                if isinstance(structured_failure, dict):
                    failure_update["structured_output_failure"] = structured_failure
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

            usage = _usage_dict(gate_response, fallback_model=model_id)
            raw_payload, raw_llm, usage = await _recover_unresolved_once(
                gate=self._gate,
                primary_payload=raw_payload,
                primary_raw=raw_llm,
                primary_usage=usage,
                questions=questions,
                answer_pages=answer_pages,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                document=document,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                submission_id=submission_id,
                exam_id=exam_id,
                run_id=run_id,
            )
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
                "method": "whole_copy_visual",
                "source_page_numbers": list(
                    raw_question_result.get("source_pages") or []
                ),
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
            consistency_calibration: Optional[Dict[str, Any]] = None
            if (
                grading_consistency_key
                and grade.attempt_status == "attempted"
                and grade.total_score is not None
                and not grade.manual_review_required
                and not _is_objective_question(grade.question)
            ):
                peer_cursor = self._db["evalpen_evaluations"].find(
                    {
                        "exam_id": exam_id,
                        "question_id": question_id,
                        "student_id": {"$ne": student_id},
                        "prompt_version": prompt_version,
                        "model_used": model_used,
                        "grading_consistency_key": grading_consistency_key,
                        "manual_review_required": False,
                    },
                    {
                        "evaluation_id": 1,
                        "student_id": 1,
                        "total_score": 1,
                        "max_score": 1,
                        "criterion_marks": 1,
                        "created_at": 1,
                    },
                ).sort("created_at", -1).limit(100)
                peer_docs = await peer_cursor.to_list(length=100)
                latest_by_student: Dict[str, Dict[str, Any]] = {}
                current_max = _max_marks(grade.question)
                current_ids = {
                    str(item.get("criterion_id") or "")
                    for item in grade.criterion_marks
                }
                for peer in peer_docs:
                    peer_student = str(peer.get("student_id") or "")
                    peer_ids = {
                        str(item.get("criterion_id") or "")
                        for item in (peer.get("criterion_marks") or [])
                        if isinstance(item, dict)
                    }
                    peer_max = _finite_float(peer.get("max_score"))
                    if (
                        not peer_student
                        or peer_student in latest_by_student
                        or peer_max is None
                        or abs(peer_max - current_max) > 0.01
                        or peer_ids != current_ids
                    ):
                        continue
                    latest_by_student[peer_student] = peer

                variants: Dict[str, List[Dict[str, Any]]] = {}
                for peer in latest_by_student.values():
                    variants.setdefault(
                        _criterion_award_signature(peer),
                        [],
                    ).append(peer)
                if len(variants) == 1 and latest_by_student:
                    canonical = next(iter(variants.values()))[0]
                    canonical_marks = {
                        str(item.get("criterion_id") or ""): item
                        for item in (canonical.get("criterion_marks") or [])
                        if isinstance(item, dict)
                    }
                    calibrated_marks: List[Dict[str, Any]] = []
                    for current_mark in grade.criterion_marks:
                        calibrated = dict(current_mark)
                        canonical_mark = canonical_marks.get(
                            str(current_mark.get("criterion_id") or "")
                        )
                        if canonical_mark:
                            for key in (
                                "marks_awarded",
                                "decision",
                                "credit_basis",
                                "rationale",
                                "missing_evidence",
                            ):
                                if key in canonical_mark:
                                    calibrated[key] = canonical_mark[key]
                        calibrated_marks.append(calibrated)
                    grade.criterion_marks = calibrated_marks
                    grade.total_score = float(canonical.get("total_score") or 0.0)
                    consistency_calibration = {
                        "status": "reused_unanimous_peer_award",
                        "source_evaluation_id": canonical.get("evaluation_id"),
                        "peer_student_count": len(latest_by_student),
                    }
                elif len(variants) > 1:
                    grade.manual_review_required = True
                    conflict_reason = (
                        "Equivalent normalized work has conflicting prior awards "
                        "within this exam and requires consistency review"
                    )
                    grade.review_reason = (
                        f"{grade.review_reason}; {conflict_reason}"
                        if grade.review_reason
                        else conflict_reason
                    )
                    consistency_calibration = {
                        "status": "conflicting_peer_awards",
                        "peer_student_count": len(latest_by_student),
                        "award_variant_count": len(variants),
                    }
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
                    "method": "full_document_visual",
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
                    _stable_id(
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
                "is_continuation": len(grade.source_pages) > 1,
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


async def _student_copy_content(
    answer_pages: List[Dict[str, Any]],
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


async def _recover_unresolved_once(
    *,
    gate: FullDocumentGateProtocol,
    primary_payload: Dict[str, Any],
    primary_raw: str,
    primary_usage: Dict[str, Any],
    questions: List[Dict[str, Any]],
    answer_pages: List[Dict[str, Any]],
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    document: Dict[str, Any],
    model_id: str,
    reasoning_effort: str,
    temperature: float,
    submission_id: str,
    exam_id: str,
    run_id: str,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Use at most one extra provider call for questions that truly need recovery."""

    grades, _, _ = _validate_ledger(
        primary_payload,
        questions=questions,
        page_count=len(answer_pages),
    )
    retry_numbers = {
        grade.question_number
        for grade in grades
        if grade.attempt_status == "unresolved" or grade.manual_review_required
    }
    if not retry_numbers:
        return primary_payload, primary_raw, primary_usage

    retry_questions = [
        question
        for index, question in enumerate(questions, start=1)
        if (_positive_int(question.get("question_number")) or index) in retry_numbers
    ]
    recovery_content, recovery_bytes = await _student_recovery_content(answer_pages)
    if len(paper_bytes) + len(solution_bytes or b"") + recovery_bytes > _MAX_REQUEST_PAYLOAD_BYTES:
        recovery_content, recovery_bytes = await _student_copy_content(answer_pages)
    if len(paper_bytes) + len(solution_bytes or b"") + recovery_bytes > _MAX_REQUEST_PAYLOAD_BYTES:
        logger.warning(
            "Skipping bounded PCR recovery call for run %s because the original "
            "whole-copy payload exceeds the request limit",
            run_id,
        )
        return primary_payload, primary_raw, primary_usage

    recovery_content.insert(
        0,
        {
            "type": "input_text",
            "text": (
                "ONE BOUNDED RECOVERY PASS. Re-check only the requested questions. "
                "Use the complete pages and any alternate upright views below. A dark, "
                "sideways, faint, or Hindi answer is not unresolved if its meaning can "
                "reasonably be read. Do not revisit questions outside this catalog."
            ),
        },
    )
    retry_input = _build_responses_input(
        questions=retry_questions,
        paper_bytes=paper_bytes,
        solution_bytes=solution_bytes,
        student_content=recovery_content,
        paper_filename=str(document.get("filename") or "question-paper.pdf"),
        solution_filename=str(
            document.get("answer_sheet_filename") or "teacher-solution.pdf"
        ),
    )
    try:
        recovery_output_limit = _recovery_output_limit(len(retry_questions))
        response = await gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=retry_input,
            json_schema=_whole_copy_schema(retry_questions),
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=recovery_output_limit,
            metadata={
                "pcr_stage": "full_document_visual_recovery",
                "prompt_version": _PROMPT_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_count": len(retry_questions),
                "page_count": len(answer_pages),
                "run_id": run_id,
                "provider_call_number": 2,
                "provider_call_limit": 2,
            },
        )
    except Exception:
        logger.exception("Bounded whole-copy recovery call failed for run %s", run_id)
        return primary_payload, primary_raw, primary_usage

    recovery_usage = _usage_dict(response, fallback_model=model_id)
    merged_usage = _aggregate_usages(
        [primary_usage, recovery_usage],
        fallback_model=model_id,
    )
    completion_failure = _response_completion_failure(response)
    if completion_failure:
        logger.warning(
            "Bounded whole-copy recovery was incomplete for run %s: %s",
            run_id,
            completion_failure["incomplete_reason"] or "unknown reason",
        )
        return primary_payload, primary_raw, merged_usage

    retry_raw = str(getattr(response, "content", "") or "")
    retry_payload = _parse_json_object(retry_raw)
    if retry_payload is None:
        logger.warning("Bounded whole-copy recovery returned invalid JSON for run %s", run_id)
        return primary_payload, primary_raw, merged_usage

    retry_grades, _, _ = _validate_ledger(
        retry_payload,
        questions=retry_questions,
        page_count=len(answer_pages),
    )
    retry_grade_by_number = {
        grade.question_number: grade for grade in retry_grades
    }

    primary_items = [
        dict(item)
        for item in (primary_payload.get("questions") or [])
        if isinstance(item, Mapping)
    ]
    primary_by_number = {
        _positive_int(item.get("question_number")): item
        for item in primary_items
        if _positive_int(item.get("question_number"))
    }
    for item in retry_payload.get("questions") or []:
        if not isinstance(item, Mapping):
            continue
        number = _positive_int(item.get("question_number"))
        if number not in retry_numbers:
            continue
        replacement = dict(item)
        validated_retry = retry_grade_by_number.get(number)
        retry_resolved = bool(
            validated_retry
            and validated_retry.attempt_status != "unresolved"
            and not validated_retry.manual_review_required
        )
        if retry_resolved:
            primary_by_number[number] = replacement

    merged_questions = []
    for index, question in enumerate(questions, start=1):
        number = _positive_int(question.get("question_number")) or index
        if number in primary_by_number:
            merged_questions.append(primary_by_number[number])
    merged_payload = {"questions": merged_questions}
    merged_raw = json.dumps(merged_payload, ensure_ascii=False, separators=(",", ":"))
    return merged_payload, merged_raw, merged_usage


async def _student_recovery_content(
    answer_pages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    """Return originals plus clean upright candidates only for sideways pages."""

    content, total_bytes = await _student_copy_content(answer_pages)
    for page in answer_pages:
        page_number = int(page.get("page_number") or 0)
        raw_ref = page.get("raw_image_ref")
        if page_number <= 0 or not isinstance(raw_ref, str) or not raw_ref.strip():
            continue
        image_b64 = await _resolve_image_base64(
            raw_ref,
            expected_sha256=page.get("asset_sha256"),
        )
        if not image_b64:
            continue
        try:
            original = base64.b64decode(image_b64, validate=True)
        except Exception:
            continue
        sideways, _ = await asyncio.to_thread(detect_sideways_page, original)
        if not sideways:
            continue
        for rotation in (90, 270):
            rotated = await asyncio.to_thread(
                _rotate_image_clockwise,
                original,
                rotation,
            )
            total_bytes += len(rotated)
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"Page {page_number} alternate clean reading view, rotated "
                            f"{rotation} degrees clockwise. It is the same physical page, "
                            "not additional student work."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(rotated).decode("ascii"),
                        "detail": "high",
                    },
                ]
            )
    return content, total_bytes


def _rotate_image_clockwise(image_bytes: bytes, rotation_degrees: int) -> bytes:
    rotation = int(rotation_degrees or 0) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("Page rotation must be 0, 90, 180, or 270 degrees")
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if rotation:
            image = image.rotate(-rotation, expand=True, fillcolor="white")
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=96,
            subsampling=0,
            optimize=True,
        )
        value = output.getvalue()
        if not value:
            raise ValueError("Recovery orientation image is empty")
        return value


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


def _build_responses_input(
    *,
    questions: List[Dict[str, Any]],
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    student_content: List[Dict[str, Any]],
    paper_filename: str,
    solution_filename: str,
) -> List[Dict[str, Any]]:
    catalog = [_catalog_question(q) for q in questions]
    static_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE MARKING CATALOG. Question IDs, ordering, maximum marks, "
                "and criterion maximums are authoritative. The attached paper and "
                "teacher solution are visual semantic evidence and may contain "
                "handwriting, formulae, diagrams, tables, or graphs.\n"
                + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
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
                {
                    "type": "input_text",
                    "text": "TEACHER-UPLOADED SOLUTION / MARKING-SCHEME PDF:",
                },
                {
                    "type": "input_file",
                    "filename": _safe_pdf_filename(
                        solution_filename,
                        "teacher-solution.pdf",
                    ),
                    "file_data": "data:application/pdf;base64,"
                    + base64.b64encode(solution_bytes).decode("ascii"),
                },
            ]
        )
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": _system_instructions()}],
        },
        {"role": "user", "content": static_content},
        {"role": "user", "content": student_content},
    ]


def _system_instructions() -> str:
    return (
        "You are grading one complete handwritten answer copy. Read the original "
        "question paper, teacher solution or marking scheme, and every student page "
        "directly. The pages may be dark, sideways, photographed at an angle, marked "
        "by a teacher, or written in Hindi or another language. Preserve the student's "
        "meaning and script. Do not depend on OCR, crops, coordinates, confidence "
        "thresholds, or exact wording.\n\n"
        "First locate each answer across the complete copy, then grade it against the "
        "matching catalog question and locked marking criteria. Several answers may "
        "share one page and an answer may continue on another page. Use visible question "
        "numbers, wording, context, and page continuity. Return every catalog question "
        "exactly once.\n\n"
        "Use attempted when relevant work is visible. Use not_attempted only after "
        "checking every page and finding no work for that question. Use unresolved only "
        "when the answer or its ownership is genuinely unreadable; never use it merely "
        "because the photograph is imperfect or confidence is low. For readable work, "
        "make the best evidence-supported award.\n\n"
        "The catalog grading_mode is authoritative; never infer a different mode from "
        "the wording or from the presence of options. A catalog item marked subjective "
        "must return every locked criterion exactly once, even when it contains one or "
        "more multiple-choice subparts. Keep marks within each criterion maximum. "
        "Equivalent correct wording or methods receive credit. Award correct visible "
        "steps even when the final answer is wrong. Only a catalog item explicitly "
        "marked objective may return no criterion rows: transcribe its selected option "
        "and set total_score 0 so the server can apply the answer key.\n\n"
        "source_pages contains only physical answer-copy page numbers. student_answer is "
        "a concise faithful transcription of visible work. overall_feedback is one or "
        "two direct teacher-style sentences. needs_review is true only when unreadability "
        "or ownership ambiguity materially prevents a reliable score. Do not mention AI, "
        "OCR, confidence, image processing, schemas, or evidence mapping."
    )


def _whole_copy_schema(questions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    question_variants = [
        _question_output_schema(question, fallback_number=index)
        for index, question in enumerate(questions, start=1)
    ]
    if not question_variants:
        raise FullDocumentGradingError("Cannot build a grading schema without questions")
    question_items: Dict[str, Any]
    if len(question_variants) == 1:
        question_items = question_variants[0]
    else:
        question_items = {"anyOf": question_variants}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "questions": {
                "type": "array",
                "items": question_items,
                "minItems": len(question_variants),
                "maxItems": len(question_variants),
            },
        },
        "required": ["questions"],
    }


def _question_output_schema(
    question: Mapping[str, Any],
    *,
    fallback_number: int,
) -> Dict[str, Any]:
    """Bind one output row to its immutable grading mode and rubric shape."""

    number = _positive_int(question.get("question_number")) or fallback_number
    objective = _is_objective_question(dict(question))
    criteria = [] if objective else _criteria(dict(question))
    criterion_variants = [
        _criterion_output_schema(criterion) for criterion in criteria
    ]
    if criterion_variants:
        criterion_items: Dict[str, Any]
        if len(criterion_variants) == 1:
            criterion_items = criterion_variants[0]
        else:
            criterion_items = {"anyOf": criterion_variants}
    else:
        # ``items`` remains a valid schema even though objective rows are
        # structurally constrained to an empty criterion array.
        criterion_items = _criterion_output_schema(None)
    required_criterion_count = len(criteria)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "integer", "enum": [number]},
            "attempt_status": {
                "type": "string",
                "enum": ["attempted", "not_attempted", "unresolved"],
            },
            "student_answer": {"type": "string"},
            "source_pages": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
            "criterion_marks": {
                "type": "array",
                "items": criterion_items,
                "minItems": required_criterion_count,
                "maxItems": required_criterion_count,
            },
            "total_score": {
                "type": "number",
                "minimum": 0,
                "maximum": _max_marks(dict(question)),
            },
            "overall_feedback": {"type": "string"},
            "needs_review": {"type": "boolean"},
            "review_reason": {"type": "string"},
        },
        "required": [
            "question_number",
            "attempt_status",
            "student_answer",
            "source_pages",
            "criterion_marks",
            "total_score",
            "overall_feedback",
            "needs_review",
            "review_reason",
        ],
    }


def _criterion_output_schema(
    criterion: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "criterion_id": {"type": "string"},
        "marks_awarded": {"type": "number", "minimum": 0},
        "rationale": {"type": "string"},
        "evidence": {"type": "string"},
        "credit_basis": {
            "type": "string",
            "enum": [
                "direct_evidence",
                "error_carried_forward",
                "no_credit",
            ],
        },
    }
    if criterion is not None:
        criterion_id = str(criterion.get("criterion_id") or "").strip()
        if criterion_id:
            properties["criterion_id"] = {"type": "string", "enum": [criterion_id]}
        properties["marks_awarded"]["maximum"] = max(
            0.0,
            float(criterion.get("max_marks") or 0.0),
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": [
            "criterion_id",
            "marks_awarded",
            "rationale",
            "evidence",
            "credit_basis",
        ],
    }


def _whole_copy_output_limit(question_count: int) -> int:
    """Budget reasoning plus JSON without forcing the model to spend the cap."""

    count = max(1, int(question_count or 0))
    return min(32_000, max(24_000, 1_600 * count))


def _recovery_output_limit(question_count: int) -> int:
    """Allow one smaller recovery ledger while keeping the second call bounded."""

    count = max(1, int(question_count or 0))
    return min(20_000, max(12_000, 1_600 * count))


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
    document_warnings: List[str] = []
    document_review = _DocumentReview(
        all_student_work_accounted=not structural_errors,
        confidence=1.0 if not structural_errors else 0.0,
        warnings=document_warnings,
        required=bool(structural_errors),
    )
    if structural_errors:
        document_review.required = True
        document_warnings.append(
            "The whole-copy grading result has structural validation errors"
        )
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
    confidence = 1.0
    student_answer = str(item.get("student_answer") or "").strip()
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
        for raw in raw_marks:
            fragment = str(raw.get("evidence") or "").strip()
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
        criterion_confidence = 1.0
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
        cited_region_ids = sorted(evidence_region_ids)
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
    seen: set[int] = set()
    for value in raw_pages:
        page_number = _positive_int(value)
        if not page_number or page_number > page_count:
            errors.append("Answer refers to a non-submitted page")
            continue
        if page_number in seen:
            continue
        seen.add(page_number)
        regions.append(
            {
                "region_id": f"q{question_number}-page-{page_number}",
                "page_number": page_number,
                "x_start": 0.0,
                "y_start": 0.0,
                "x_end": _A4_WIDTH_MM,
                "y_end": _A4_HEIGHT_MM,
                "coordinate_space": "original_page_mm",
                "evidence": "Complete source page cited by the whole-copy grader.",
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


def _criterion_award_signature(evaluation: Mapping[str, Any]) -> str:
    """Return the score-only identity used to detect cohort disagreement."""

    marks = [
        {
            "criterion_id": str(item.get("criterion_id") or ""),
            "marks_awarded": round(
                float(_finite_float(item.get("marks_awarded")) or 0.0),
                2,
            ),
            "max_marks": round(
                float(_finite_float(item.get("max_marks")) or 0.0),
                2,
            ),
            "decision": str(item.get("decision") or ""),
            "credit_basis": str(item.get("credit_basis") or ""),
        }
        for item in (evaluation.get("criterion_marks") or [])
        if isinstance(item, Mapping)
    ]
    payload = {
        "total_score": round(
            float(_finite_float(evaluation.get("total_score")) or 0.0),
            2,
        ),
        "criterion_marks": sorted(marks, key=lambda item: item["criterion_id"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _static_context_hash(
    exam: Dict[str, Any],
    *,
    paper_hash: str,
    solution_hash: Optional[str],
    prompt_version: str,
) -> str:
    value = "\x1f".join(
        [
            prompt_version,
            str(exam.get("paper_version_id") or ""),
            paper_hash,
            solution_hash or "",
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
