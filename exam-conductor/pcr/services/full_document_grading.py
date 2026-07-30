"""Submission-level visual grading for PCR answer copies.

This is the primary camera/PDF path for papers where handwriting, diagrams,
tables, and answer ownership cannot safely be reduced to OCR text first.
Subjective papers use the full visual evidence graph. Objective papers use one
compact Responses request per canonical student page and are scored
deterministically after answer transcription.

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
from services.objective_answer_ledger_contract import (
    OBJECTIVE_LEDGER_VERSION,
    OBJECTIVE_PAPER_CONTEXT_VERSION,
    OBJECTIVE_PROMPT_VERSION,
    all_questions_are_objective,
    merge_objective_page_ledgers,
    objective_extraction_catalog,
    objective_page_observation_schema,
    objective_reader_instructions,
)
from services.objective_scoring_service import (
    ObjectiveScoringContractError,
    score_objective_response,
)
from services.canonical_asset_storage import read_canonical_asset

from ..domain.response_models import ContentType
from ..marking_policy import (
    ANY_VALID_METHOD,
    NO_METHOD_REQUIRED,
    SPECIFIED_METHOD_REQUIRED,
    method_policy_instruction,
    normalize_marking_criteria,
    normalize_marking_policy,
    normalize_method_policy,
    strictness_instruction,
)
from ..storage.evaluation_repo import EvaluationRepository
from ..storage.response_repo import DetectedResponseRepository
from .ocr_service import AssetIntegrityError, _resolve_image_base64
from .visual_evidence_graph import (
    EVIDENCE_GRAPH_VERSION,
    PROMPT_VERSION as _EVIDENCE_GRAPH_PROMPT_VERSION,
    evidence_mapping_schema,
    grading_system_instructions,
    mapping_system_instructions,
    merge_mapping_and_grading,
    question_grading_schema,
    validate_mapping_payload,
)

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "pcr-full-document-visual-v4"
_SUPPORTED_PROMPT_VERSIONS = {
    _PROMPT_VERSION,
    _EVIDENCE_GRAPH_PROMPT_VERSION,
    OBJECTIVE_PROMPT_VERSION,
}
_RUNS_COLLECTION = "evalpen_document_grading_runs"
_LLM_DEBUG_TRACES_COLLECTION = "evalpen_llm_debug_traces"
_PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
_CALLER_ID = "pcr_eval_core"
_OBJECTIVE_OUTPUT_BUDGET_POLICY = "objective-ledger-cardinality-v1"
_OBJECTIVE_OUTPUT_TOKEN_FLOOR = 6_000
_OBJECTIVE_OUTPUT_TOKEN_CEILING = 20_000
_OBJECTIVE_OUTPUT_BASE_TOKENS = 2_500
_OBJECTIVE_OUTPUT_TOKENS_PER_QUESTION = 100
_OBJECTIVE_REASONING_TOKEN_RESERVE = {
    "none": 0,
    "minimal": 1_000,
    "low": 2_000,
    "medium": 4_000,
    "high": 6_000,
}
_AUTO_ACCEPT_CONFIDENCE = 0.80
_ABSENCE_CONFIDENCE = 0.85
_CRITERION_AUTO_ACCEPT_CONFIDENCE = 0.85
_CRITERION_MIN_SCORE_CONFIDENCE = 0.65
_DEFAULT_REASONING_EFFORT = "medium"
_MAX_PAGE_COUNT = 50
_MAX_STATIC_PDF_BYTES = 45 * 1024 * 1024
_MAX_REQUEST_PAYLOAD_BYTES = 45 * 1024 * 1024
# Subjective visual batches: smaller groups reduce truncated/invalid JSON.
_DEFAULT_VISUAL_QUESTIONS_PER_BATCH = 3
_VISUAL_GRADE_BATCH_ATTEMPTS = 2
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


class CanonicalAssetUnavailableError(FullDocumentGradingError):
    """Raised when a required immutable grading asset cannot be loaded."""


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
    processing_path: str = "full_document_visual"
    run_id: Optional[str] = None
    materialization_id: Optional[str] = None
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


@dataclass(frozen=True)
class _StudentPageAsset:
    page_number: int
    original_bytes: bytes
    global_bytes: bytes
    global_media_type: str
    original_media_type: str = "image/jpeg"


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
        evidence_graph_required = _paper_requires_evidence_graph(paper_version)
        objective_ledger_required = _paper_requires_objective_ledger(paper_version)
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
        prompt_version = (
            OBJECTIVE_PROMPT_VERSION
            if objective_ledger_required
            else (
                _EVIDENCE_GRAPH_PROMPT_VERSION
                if evidence_graph_required
                else _PROMPT_VERSION
            )
        )
        if contract_version and contract_version not in _SUPPORTED_PROMPT_VERSIONS:
            raise FullDocumentGradingError(
                "This exam is locked to grading contract "
                f"{contract_version}, which this worker does not support. "
                "Do not mix grading contracts within one exam; migrate and reprocess "
                "the complete exam together."
            )
        if contract_version and contract_version != prompt_version:
            raise FullDocumentGradingError(
                "The immutable paper requires grading contract "
                f"{prompt_version}, but this cohort is locked to {contract_version}. "
                "Migrate and reprocess the complete cohort; never mix grading "
                "contracts student by student."
            )
        model_id = str(
            grading_contract.get("model_id")
            or (
                os.getenv(
                    "PCR_OBJECTIVE_GRADING_MODEL",
                    "gpt-5.6-sol",
                ).strip()
                if objective_ledger_required
                else self._model_id
            )
        ).strip()
        temperature = _contract_temperature(grading_contract)
        reasoning_effort = str(
            grading_contract.get("reasoning_effort")
            or (
                os.getenv("PCR_OBJECTIVE_REASONING_EFFORT", "medium")
                if objective_ledger_required
                else _DEFAULT_REASONING_EFFORT
            )
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
        if objective_ledger_required and not all_questions_are_objective(questions):
            raise FullDocumentGradingError(
                "The immutable paper is locked to objective answer-ledger grading, "
                "but its question catalog contains a non-objective question"
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

        document_id = str(
            exam.get("prepared_document_id")
            or (paper_version or {}).get("document_id")
            or ""
        )
        document = await self._db["documents"].find_one(
            {"document_id": document_id}
        )
        if not document:
            if canonical_visual_required:
                raise CanonicalAssetUnavailableError(
                    "The exam requires canonical visual grading, but its immutable "
                    "question-paper record is unavailable"
                )
            # Legacy sessions without the original PDF remain on the existing
            # review-safe pipeline. Do not accept a client-provided substitute.
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason="Legacy exam has no immutable question-paper record",
            )

        try:
            paper_bytes = await _read_canonical_file(
                str(document.get("file_path") or ""),
                expected_sha256=document.get("sha256"),
            )
        except AssetIntegrityError:
            raise
        except Exception as exc:
            raise CanonicalAssetUnavailableError(
                "The immutable question-paper asset could not be loaded from storage"
            ) from exc
        if not paper_bytes:
            if canonical_visual_required:
                raise CanonicalAssetUnavailableError(
                    "The exam requires canonical visual grading, but its immutable "
                    "question-paper asset could not be loaded"
                )
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
                skipped_reason="Legacy question-paper asset could not be loaded",
            )
        try:
            solution_bytes = await _read_canonical_file(
                str(document.get("answer_sheet_path") or ""),
                expected_sha256=document.get("answer_sheet_sha256"),
            )
        except AssetIntegrityError:
            raise
        except Exception as exc:
            raise CanonicalAssetUnavailableError(
                "The immutable teacher-solution asset could not be loaded from storage"
            ) from exc
        if (
            not objective_ledger_required
            and len(paper_bytes) + len(solution_bytes or b"")
            > _MAX_STATIC_PDF_BYTES
        ):
            raise FullDocumentGradingError(
                "Question paper and teacher solution exceed the document-input size limit"
            )
        paper_file_hash = hashlib.sha256(paper_bytes).hexdigest()
        solution_file_hash = (
            hashlib.sha256(solution_bytes).hexdigest() if solution_bytes else None
        )

        generation_revision = await _generation_revision(
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
            }
        )
        if prior_revision_run:
            # Resume the exact technical run that already owns this generation.
            # This also remains stable when the first provider response froze a
            # dated model snapshot for subsequent students in the cohort.
            input_fingerprint = str(
                prior_revision_run.get("input_fingerprint") or ""
            )
            generation_fingerprint = str(
                prior_revision_run.get("generation_fingerprint")
                or prior_revision_run.get("input_fingerprint")
                or ""
            )
            run_id = str(prior_revision_run.get("run_id") or "")
            model_id = str(
                prior_revision_run.get("requested_model_id") or model_id
            )
            if not input_fingerprint or not generation_fingerprint or not run_id:
                raise FullDocumentGradingError(
                    "Saved submission grading run is missing its immutable identity"
                )
        else:
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
            run_id = f"DOCGR-{generation_fingerprint[:24]}"
        materialization_id = f"{run_id}:g{generation_revision}"
        await self._db[_RUNS_COLLECTION].create_index(
            "run_id", unique=True, name="uniq_document_grading_run"
        )
        await self._db[_RUNS_COLLECTION].create_index(
            [
                ("submission_id", 1),
                ("prompt_version", 1),
                ("generation_revision", 1),
            ],
            name="submission_grading_generation",
        )
        await self._db[_LLM_DEBUG_TRACES_COLLECTION].create_index(
            "trace_id", unique=True, name="uniq_llm_debug_trace"
        )
        await self._db[_LLM_DEBUG_TRACES_COLLECTION].create_index(
            [("submission_id", 1), ("run_id", 1), ("page_number", 1)],
            name="submission_llm_debug_trace",
        )
        existing_run = await self._db[_RUNS_COLLECTION].find_one({"run_id": run_id})
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
            student_assets: Optional[List[_StudentPageAsset]] = None
            if prompt_version in {
                _EVIDENCE_GRAPH_PROMPT_VERSION,
                OBJECTIVE_PROMPT_VERSION,
            }:
                student_assets, student_image_bytes = await _student_page_assets(
                    answer_pages
                )
                student_content = (
                    _student_content_from_assets(student_assets)
                    if prompt_version == _EVIDENCE_GRAPH_PROMPT_VERSION
                    else []
                )
            else:
                student_content, student_image_bytes = await _student_copy_content(
                    answer_pages
                )
            if (
                (
                    0
                    if prompt_version == OBJECTIVE_PROMPT_VERSION
                    else len(paper_bytes) + len(solution_bytes or b"")
                )
                + student_image_bytes
                > _MAX_REQUEST_PAYLOAD_BYTES
            ):
                raise FullDocumentGradingError(
                    "Paper, solution, and optimized student pages exceed the visual "
                    "request size limit"
                )
            try:
                if prompt_version == OBJECTIVE_PROMPT_VERSION:
                    raw_payload, raw_llm, usage = (
                        await self._run_objective_answer_ledger(
                            run_id=run_id,
                            generation_lease_token=generation_lease_token,
                            existing_run=await self._db[_RUNS_COLLECTION].find_one(
                                {"run_id": run_id}
                            ),
                            submission_id=submission_id,
                            exam_id=exam_id,
                            questions=questions,
                            student_assets=student_assets or [],
                            model_id=model_id,
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            paper_file_hash=paper_file_hash,
                        )
                    )
                    gate_response = None
                elif prompt_version == _EVIDENCE_GRAPH_PROMPT_VERSION:
                    raw_payload, raw_llm, usage = await self._run_evidence_graph(
                        run_id=run_id,
                        generation_lease_token=generation_lease_token,
                        existing_run=await self._db[_RUNS_COLLECTION].find_one(
                            {"run_id": run_id}
                        ),
                        submission_id=submission_id,
                        exam_id=exam_id,
                        questions=questions,
                        answer_pages=answer_pages,
                        student_assets=student_assets or [],
                        paper_bytes=paper_bytes,
                        solution_bytes=solution_bytes,
                        student_content=student_content,
                        document=document,
                        model_id=model_id,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        paper_file_hash=paper_file_hash,
                        solution_file_hash=solution_file_hash,
                    )
                    gate_response = None
                else:
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
                        json_schema=_evidence_ledger_schema(),
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
                        max_output_tokens=min(
                            30_000, max(8_000, 1_100 * len(questions))
                        ),
                        metadata={
                            "pcr_stage": "full_document_visual_grading",
                            "prompt_version": prompt_version,
                            "submission_id": submission_id,
                            "exam_id": exam_id,
                            "question_count": len(questions),
                            "page_count": len(answer_pages),
                            "run_id": run_id,
                        },
                    )
            except asyncio.CancelledError:
                await _fail_generation_run(
                    self._db,
                    run_id=run_id,
                    generation_lease_token=generation_lease_token,
                    error="Worker shutdown interrupted model generation",
                )
                raise
            except Exception as exc:
                await _fail_generation_run(
                    self._db,
                    run_id=run_id,
                    generation_lease_token=generation_lease_token,
                    error=exc,
                )
                raise FullDocumentGradingError(
                    f"Full-document model request failed: {str(exc)[:400]}"
                ) from exc

            if gate_response is not None:
                raw_llm = str(getattr(gate_response, "content", "") or "")
                raw_payload = _parse_json_object(raw_llm)
                if raw_payload is None:
                    await _fail_generation_run(
                        self._db,
                        run_id=run_id,
                        generation_lease_token=generation_lease_token,
                        error="Full-document model returned an invalid evidence ledger",
                    )
                    raise FullDocumentGradingError(
                        "Full-document model returned an invalid evidence ledger"
                    )
                usage = _usage_dict(gate_response, fallback_model=model_id)
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
                await _fail_generation_run(
                    self._db,
                    run_id=run_id,
                    generation_lease_token=generation_lease_token,
                    error=exc,
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
                        "processing_path": result.processing_path,
                        "materialization_id": result.materialization_id,
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

    async def _run_objective_answer_ledger(
        self,
        *,
        run_id: str,
        generation_lease_token: str,
        existing_run: Optional[Dict[str, Any]],
        submission_id: str,
        exam_id: str,
        questions: List[Dict[str, Any]],
        student_assets: List[_StudentPageAsset],
        model_id: str,
        temperature: float,
        reasoning_effort: str,
        paper_file_hash: str,
    ) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Read objective answers page-by-page, then merge and score server-side.

        Page requests are independent and run with bounded concurrency.  A
        one-page OMR therefore needs one model call, while a multi-page answer
        book needs one call per page rather than one call per question batch.
        The prompt contains permitted answer values but never the correct key.
        """

        if not student_assets:
            raise FullDocumentGradingError(
                "Objective answer-ledger grading requires canonical student pages"
            )
        current_run = dict(existing_run or {})
        saved_ledgers = dict(current_run.get("objective_page_ledgers") or {})
        saved_usages = dict(current_run.get("objective_page_ledger_usages") or {})
        catalog = objective_extraction_catalog(questions)
        cache_key = (
            "pcr-objective-ledger-"
            + hashlib.sha256(
                (
                    paper_file_hash
                    + "|"
                    + OBJECTIVE_PROMPT_VERSION
                    + "|"
                    + json.dumps(catalog, sort_keys=True, separators=(",", ":"))
                ).encode("utf-8")
            ).hexdigest()[:32]
        )
        resolved_model = str(current_run.get("model_used") or model_id)

        try:
            configured_concurrency = int(
                os.getenv("PCR_OBJECTIVE_PAGE_CONCURRENCY", "3") or 3
            )
        except (TypeError, ValueError):
            configured_concurrency = 3
        semaphore = asyncio.Semaphore(max(1, min(6, configured_concurrency)))

        async def _read_page(
            asset: _StudentPageAsset,
        ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
            async with semaphore:
                call_spec, request_bytes = build_objective_page_call_spec(
                    asset=asset,
                    catalog=catalog,
                    model_id=resolved_model,
                    prompt_cache_key=cache_key,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    submission_id=submission_id,
                    exam_id=exam_id,
                    run_id=run_id,
                    question_count=len(questions),
                )
                if request_bytes > _MAX_REQUEST_PAYLOAD_BYTES:
                    raise FullDocumentGradingError(
                        f"Objective page {asset.page_number} exceeds the visual "
                        "request size limit"
                    )
                request_manifest, image_assets, _ = build_llm_debug_request_manifest(
                    call_spec
                )
                trace_id = (
                    f"{run_id}:objective_answer_page_reading:"
                    f"{asset.page_number}"
                )
                requested_at = datetime.now(timezone.utc)
                await self._db[_LLM_DEBUG_TRACES_COLLECTION].update_one(
                    {"trace_id": trace_id},
                    {
                        "$setOnInsert": {
                            "trace_id": trace_id,
                            "run_id": run_id,
                            "submission_id": submission_id,
                            "exam_id": exam_id,
                            "stage": "objective_answer_page_reading",
                            "page_number": asset.page_number,
                            "created_at": requested_at,
                        },
                        "$set": {
                            "status": "requested",
                            "request": request_manifest,
                            "image_assets": image_assets,
                            "requested_at": requested_at,
                            "updated_at": requested_at,
                        },
                        "$unset": {
                            "raw_response": "",
                            "parsed_response": "",
                            "usage": "",
                            "response_error": "",
                            "completed_at": "",
                        },
                    },
                    upsert=True,
                )
                response_received = False
                trace_status = "failed"
                provider_status: Optional[str] = None
                incomplete_reason: Optional[str] = None
                try:
                    response = await self._gate.call(**call_spec)
                    response_received = True
                    raw = str(getattr(response, "content", "") or "")
                    usage = _usage_dict(response, fallback_model=resolved_model)
                    provider_status = (
                        str(getattr(response, "provider_status", "") or "").strip()
                        or None
                    )
                    incomplete_reason = (
                        str(getattr(response, "incomplete_reason", "") or "").strip()
                        or None
                    )
                    # Provider incomplete status always wins over partial JSON
                    # repair — a truncated stream is never a finished ledger.
                    if (
                        incomplete_reason == "max_output_tokens"
                        or provider_status == "incomplete"
                    ):
                        payload = None
                        trace_status = "incomplete"
                    else:
                        payload = _parse_json_object(raw)
                        trace_status = (
                            "completed" if payload is not None else "invalid_response"
                        )
                    completed_at = datetime.now(timezone.utc)
                    await self._db[_LLM_DEBUG_TRACES_COLLECTION].update_one(
                        {"trace_id": trace_id},
                        {
                            "$set": {
                                "status": trace_status,
                                "raw_response": raw,
                                "parsed_response": payload,
                                "usage": usage,
                                "provider_status": provider_status,
                                "incomplete_reason": incomplete_reason,
                                "completed_at": completed_at,
                                "updated_at": completed_at,
                            }
                        },
                    )
                    # A failed structured-output parse must still retain the
                    # provider-selected model on the immutable run audit.
                    if usage.get("model"):
                        await self._db[_RUNS_COLLECTION].update_one(
                            {
                                "run_id": run_id,
                                "generation_lease_token": generation_lease_token,
                            },
                            {
                                "$set": {
                                    "model_used": usage["model"],
                                    "updated_at": completed_at,
                                }
                            },
                        )
                    if payload is None:
                        if (
                            incomplete_reason == "max_output_tokens"
                            or provider_status == "incomplete"
                        ):
                            raise FullDocumentGradingError(
                                "Objective reader response was incomplete for page "
                                f"{asset.page_number}: provider reached "
                                f"{incomplete_reason or 'its output limit'}"
                            )
                        raise FullDocumentGradingError(
                            f"Objective reader returned invalid JSON for page "
                            f"{asset.page_number}"
                        )
                    return str(asset.page_number), payload, usage
                except Exception as exc:
                    failed_at = datetime.now(timezone.utc)
                    await self._db[_LLM_DEBUG_TRACES_COLLECTION].update_one(
                        {"trace_id": trace_id},
                        {
                            "$set": {
                                "status": (
                                    trace_status
                                    if response_received
                                    else "failed"
                                ),
                                "response_error": str(exc)[:1000],
                                "completed_at": failed_at,
                                "updated_at": failed_at,
                            }
                        },
                    )
                    raise

        missing_assets = [
            asset
            for asset in student_assets
            if str(asset.page_number) not in saved_ledgers
        ]
        if missing_assets:
            page_results = await asyncio.gather(
                *(_read_page(asset) for asset in missing_assets)
            )
            for page_key, page_payload, page_usage in page_results:
                saved_ledgers[page_key] = page_payload
                saved_usages[page_key] = page_usage
            resolved_model = str(
                next(
                    (
                        usage.get("model")
                        for usage in reversed(list(saved_usages.values()))
                        if isinstance(usage, dict) and usage.get("model")
                    ),
                    resolved_model,
                )
            )
            checkpoint = await self._db[_RUNS_COLLECTION].update_one(
                {
                    "run_id": run_id,
                    "generation_lease_token": generation_lease_token,
                },
                {
                    "$set": {
                        "objective_page_ledgers": saved_ledgers,
                        "objective_page_ledger_usages": saved_usages,
                        "model_used": resolved_model,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            if checkpoint.matched_count != 1:
                raise FullDocumentGradingError(
                    "Submission grading ownership expired while saving the "
                    "objective answer ledger"
                )

        ordered_page_payloads = [
            saved_ledgers[str(asset.page_number)]
            for asset in student_assets
            if str(asset.page_number) in saved_ledgers
        ]
        final_payload, _validation_errors = merge_objective_page_ledgers(
            ordered_page_payloads,
            questions=questions,
            page_count=len(student_assets),
        )
        usage = _aggregate_usages(
            saved_usages.values(),
            fallback_model=resolved_model,
        )
        raw_llm = json.dumps(
            {"page_ledgers": saved_ledgers},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return final_payload, raw_llm, usage

    async def _run_evidence_graph(
        self,
        *,
        run_id: str,
        generation_lease_token: str,
        existing_run: Optional[Dict[str, Any]],
        submission_id: str,
        exam_id: str,
        questions: List[Dict[str, Any]],
        answer_pages: List[Dict[str, Any]],
        student_assets: List[_StudentPageAsset],
        paper_bytes: bytes,
        solution_bytes: Optional[bytes],
        student_content: List[Dict[str, Any]],
        document: Dict[str, Any],
        model_id: str,
        temperature: float,
        reasoning_effort: str,
        paper_file_hash: str,
        solution_file_hash: Optional[str],
    ) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Build a global evidence graph, then grade bounded visual crop batches.

        The global request owns only question association and full-copy coverage.
        Question scoring receives high-resolution crops from the mapper's immutable
        regions. Intermediate results are checkpointed on the run so a provider
        retry does not repurchase completed work.
        """

        if not student_assets:
            raise FullDocumentGradingError(
                "Evidence-graph grading requires canonical student page assets"
            )
        current_run = dict(existing_run or {})
        static_content = _multistage_static_content(
            questions=questions,
            paper_bytes=paper_bytes,
            solution_bytes=solution_bytes,
            paper_filename=str(document.get("filename") or "question-paper.pdf"),
            solution_filename=str(
                document.get("answer_sheet_filename") or "teacher-solution.pdf"
            ),
        )
        cache_key = (
            "pcr-evidence-graph-"
            + _static_context_hash(
                {"paper_version_id": document.get("document_id")},
                paper_hash=paper_file_hash,
                solution_hash=solution_file_hash,
                prompt_version=_EVIDENCE_GRAPH_PROMPT_VERSION,
            )[:32]
        )

        mapping_payload = current_run.get("evidence_graph_mapping")
        mapping_usage = dict(current_run.get("evidence_graph_mapping_usage") or {})
        mapping_raw = str(current_run.get("evidence_graph_mapping_raw") or "")
        resolved_model = str(current_run.get("model_used") or model_id)
        if not isinstance(mapping_payload, dict):
            mapping_input = [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _multistage_system_instructions(),
                        }
                    ],
                },
                {"role": "user", "content": static_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "TASK: Build the complete student evidence graph. "
                                "Do not grade any question."
                            ),
                        },
                        *student_content,
                    ],
                },
            ]
            mapping_response = await self._gate.call(
                model_id=resolved_model,
                prompt="",
                caller_id=_CALLER_ID,
                responses_input=mapping_input,
                json_schema=evidence_mapping_schema(),
                prompt_cache_key=cache_key,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_output_tokens=min(
                    24_000,
                    max(6_000, 650 * len(questions)),
                ),
                metadata={
                    "pcr_stage": "full_document_evidence_mapping",
                    "prompt_version": _EVIDENCE_GRAPH_PROMPT_VERSION,
                    "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                    "submission_id": submission_id,
                    "exam_id": exam_id,
                    "question_count": len(questions),
                    "page_count": len(answer_pages),
                    "run_id": run_id,
                },
            )
            mapping_raw = str(getattr(mapping_response, "content", "") or "")
            mapping_payload = _parse_json_object(mapping_raw)
            if mapping_payload is None:
                raise FullDocumentGradingError(
                    "Evidence mapper returned an invalid evidence graph"
                )
            mapping_usage = _usage_dict(
                mapping_response,
                fallback_model=resolved_model,
            )
            resolved_model = str(mapping_usage.get("model") or resolved_model)
            checkpoint = await self._db[_RUNS_COLLECTION].update_one(
                {
                    "run_id": run_id,
                    "generation_lease_token": generation_lease_token,
                },
                {
                    "$set": {
                        "evidence_graph_mapping": mapping_payload,
                        "evidence_graph_mapping_raw": mapping_raw,
                        "evidence_graph_mapping_usage": mapping_usage,
                        "model_used": resolved_model,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            if checkpoint.matched_count != 1:
                raise FullDocumentGradingError(
                    "Submission grading ownership expired while saving evidence mapping"
                )

        mapping = validate_mapping_payload(
            mapping_payload,
            question_numbers=[
                int(question.get("question_number") or index)
                for index, question in enumerate(questions, start=1)
            ],
            page_count=len(answer_pages),
            absence_confidence=_ABSENCE_CONFIDENCE,
        )
        saved_grades = dict(
            current_run.get("evidence_graph_question_grades") or {}
        )
        saved_usages = dict(
            current_run.get("evidence_graph_question_grade_usages") or {}
        )
        attempted_numbers = [
            number
            for number, item in mapping.questions.items()
            if item.get("attempt_status") == "attempted"
        ]
        question_by_number = {
            int(question.get("question_number") or index): question
            for index, question in enumerate(questions, start=1)
        }
        missing_numbers = [
            number for number in attempted_numbers if str(number) not in saved_grades
        ]
        batch_index = 0
        for batch_numbers in _question_batches(missing_numbers):
            batch_index += 1
            resolved_model = await self._grade_evidence_batch_resilient(
                batch_numbers=batch_numbers,
                batch_index=batch_index,
                question_by_number=question_by_number,
                mapping_questions=mapping.questions,
                student_assets=student_assets,
                static_content=static_content,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                model_id=resolved_model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                cache_key=cache_key,
                submission_id=submission_id,
                exam_id=exam_id,
                run_id=run_id,
                generation_lease_token=generation_lease_token,
                saved_grades=saved_grades,
                saved_usages=saved_usages,
            )

        final_payload = merge_mapping_and_grading(
            mapping,
            [
                {
                    "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                    "questions": list(saved_grades.values()),
                }
            ],
        )
        if mapping.errors:
            final_payload["evidence_graph_validation_errors"] = mapping.errors
        usage = _aggregate_usages(
            [mapping_usage, *saved_usages.values()],
            fallback_model=resolved_model,
        )
        raw_llm = json.dumps(
            {
                "evidence_mapping": mapping_payload,
                "question_grades": saved_grades,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return final_payload, raw_llm, usage

    async def _grade_evidence_batch_resilient(
        self,
        *,
        batch_numbers: Sequence[int],
        batch_index: int,
        question_by_number: Mapping[int, Dict[str, Any]],
        mapping_questions: Mapping[int, Dict[str, Any]],
        student_assets: List[_StudentPageAsset],
        static_content: List[Dict[str, Any]],
        paper_bytes: bytes,
        solution_bytes: Optional[bytes],
        model_id: str,
        temperature: float,
        reasoning_effort: str,
        cache_key: str,
        submission_id: str,
        exam_id: str,
        run_id: str,
        generation_lease_token: str,
        saved_grades: Dict[str, Any],
        saved_usages: Dict[str, Any],
        depth: int = 0,
    ) -> str:
        """Grade one question batch with retry and split-on-failure recovery.

        Subjective visual grading frequently returns truncated or non-JSON
        payloads for large batches.  Retry once, then split the batch, and
        finally mark a single unrecoverable question unresolved so the rest of
        the student copy can still finish.
        """

        numbers = [int(number) for number in batch_numbers if int(number) in question_by_number]
        if not numbers:
            return model_id
        # Skip already-checkpointed questions (resume after partial failure).
        numbers = [number for number in numbers if str(number) not in saved_grades]
        if not numbers:
            return model_id

        last_error = "Question visual grader returned invalid structured output"
        resolved_model = model_id
        for attempt in range(1, _VISUAL_GRADE_BATCH_ATTEMPTS + 1):
            try:
                returned, batch_usage, resolved_model = await self._call_question_visual_grader(
                    batch_numbers=numbers,
                    batch_index=batch_index,
                    attempt=attempt,
                    question_by_number=question_by_number,
                    mapping_questions=mapping_questions,
                    student_assets=student_assets,
                    static_content=static_content,
                    paper_bytes=paper_bytes,
                    solution_bytes=solution_bytes,
                    model_id=resolved_model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    cache_key=cache_key,
                    submission_id=submission_id,
                    exam_id=exam_id,
                    run_id=run_id,
                )
                saved_grades.update(returned)
                usage_key = "questions-" + "-".join(
                    str(number) for number in sorted(numbers)
                )
                if attempt > 1:
                    usage_key = f"{usage_key}-retry{attempt}"
                saved_usages[usage_key] = batch_usage
                checkpoint = await self._db[_RUNS_COLLECTION].update_one(
                    {
                        "run_id": run_id,
                        "generation_lease_token": generation_lease_token,
                    },
                    {
                        "$set": {
                            "evidence_graph_question_grades": saved_grades,
                            "evidence_graph_question_grade_usages": saved_usages,
                            "model_used": resolved_model,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
                if checkpoint.matched_count != 1:
                    raise FullDocumentGradingError(
                        "Submission grading ownership expired while saving question grades"
                    )
                return resolved_model
            except FullDocumentGradingError as exc:
                last_error = str(exc)
                logger.warning(
                    "Subjective visual grade batch failed submission=%s batch=%s "
                    "attempt=%s questions=%s error=%s",
                    submission_id,
                    batch_index,
                    attempt,
                    numbers,
                    last_error[:240],
                )
                if attempt < _VISUAL_GRADE_BATCH_ATTEMPTS:
                    continue
                break

        # Persistent failure: split multi-question batches so one bad crop/schema
        # response cannot discard the whole student copy.
        if len(numbers) > 1:
            mid = max(1, len(numbers) // 2)
            left = numbers[:mid]
            right = numbers[mid:]
            logger.info(
                "Splitting subjective visual grade batch submission=%s %s -> %s | %s",
                submission_id,
                numbers,
                left,
                right,
            )
            resolved_model = await self._grade_evidence_batch_resilient(
                batch_numbers=left,
                batch_index=batch_index,
                question_by_number=question_by_number,
                mapping_questions=mapping_questions,
                student_assets=student_assets,
                static_content=static_content,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                model_id=resolved_model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                cache_key=cache_key,
                submission_id=submission_id,
                exam_id=exam_id,
                run_id=run_id,
                generation_lease_token=generation_lease_token,
                saved_grades=saved_grades,
                saved_usages=saved_usages,
                depth=depth + 1,
            )
            resolved_model = await self._grade_evidence_batch_resilient(
                batch_numbers=right,
                batch_index=batch_index,
                question_by_number=question_by_number,
                mapping_questions=mapping_questions,
                student_assets=student_assets,
                static_content=static_content,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                model_id=resolved_model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                cache_key=cache_key,
                submission_id=submission_id,
                exam_id=exam_id,
                run_id=run_id,
                generation_lease_token=generation_lease_token,
                saved_grades=saved_grades,
                saved_usages=saved_usages,
                depth=depth + 1,
            )
            return resolved_model

        # Single question still failing: record unresolved and continue.
        number = numbers[0]
        saved_grades[str(number)] = _unresolved_visual_grade_item(
            number,
            reason=(
                "Question visual grader failed after retries: "
                + last_error[:300]
            ),
        )
        usage_key = f"questions-{number}-failed"
        saved_usages[usage_key] = {
            "model": resolved_model,
            "caller": _CALLER_ID,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": last_error[:300],
        }
        checkpoint = await self._db[_RUNS_COLLECTION].update_one(
            {
                "run_id": run_id,
                "generation_lease_token": generation_lease_token,
            },
            {
                "$set": {
                    "evidence_graph_question_grades": saved_grades,
                    "evidence_graph_question_grade_usages": saved_usages,
                    "model_used": resolved_model,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        if checkpoint.matched_count != 1:
            raise FullDocumentGradingError(
                "Submission grading ownership expired while saving question grades"
            )
        logger.error(
            "Subjective visual grade left Q%s unresolved after retries submission=%s",
            number,
            submission_id,
        )
        return resolved_model

    async def _call_question_visual_grader(
        self,
        *,
        batch_numbers: Sequence[int],
        batch_index: int,
        attempt: int,
        question_by_number: Mapping[int, Dict[str, Any]],
        mapping_questions: Mapping[int, Dict[str, Any]],
        student_assets: List[_StudentPageAsset],
        static_content: List[Dict[str, Any]],
        paper_bytes: bytes,
        solution_bytes: Optional[bytes],
        model_id: str,
        temperature: float,
        reasoning_effort: str,
        cache_key: str,
        submission_id: str,
        exam_id: str,
        run_id: str,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], str]:
        batch_questions = [
            question_by_number[number]
            for number in batch_numbers
            if number in question_by_number
        ]
        if not batch_questions:
            return {}, {}, model_id
        grading_input, crop_bytes = _build_question_grading_input(
            static_content=static_content,
            questions=batch_questions,
            mappings=mapping_questions,
            student_assets=student_assets,
        )
        if len(paper_bytes) + len(solution_bytes or b"") + crop_bytes > _MAX_REQUEST_PAYLOAD_BYTES:
            raise FullDocumentGradingError(
                "Question-specific visual evidence exceeds the request size limit"
            )
        # Larger per-question budget on retries when the first response truncated.
        per_question = 1_600 if attempt > 1 else 1_400
        max_output_tokens = min(
            24_000,
            max(6_000, per_question * len(batch_questions)),
        )
        grading_response = await self._gate.call(
            model_id=model_id,
            prompt="",
            caller_id=_CALLER_ID,
            responses_input=grading_input,
            json_schema=question_grading_schema(),
            prompt_cache_key=cache_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            metadata={
                "pcr_stage": "question_visual_grading",
                "prompt_version": _EVIDENCE_GRAPH_PROMPT_VERSION,
                "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "question_numbers": list(batch_numbers),
                "batch_index": batch_index,
                "attempt": attempt,
                "run_id": run_id,
            },
        )
        grading_raw = str(getattr(grading_response, "content", "") or "")
        provider_status = str(
            getattr(grading_response, "provider_status", "") or ""
        ).strip()
        incomplete_reason = str(
            getattr(grading_response, "incomplete_reason", "") or ""
        ).strip()
        if incomplete_reason == "max_output_tokens" or provider_status == "incomplete":
            raise FullDocumentGradingError(
                "Question visual grader response was incomplete"
                + (f" ({incomplete_reason})" if incomplete_reason else "")
            )
        grading_payload = _parse_json_object(grading_raw)
        if grading_payload is None:
            raise FullDocumentGradingError(
                "Question visual grader returned invalid structured output"
            )
        allowed = {int(number) for number in batch_numbers}
        returned: Dict[str, Dict[str, Any]] = {}
        unexpected_numbers: set[int] = set()
        duplicate_numbers: set[int] = set()
        if grading_payload.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
            raise FullDocumentGradingError(
                "Question visual grader returned the wrong evidence contract"
            )
        for item in grading_payload.get("questions") or []:
            if not isinstance(item, dict):
                continue
            number = _positive_int(item.get("question_number"))
            if number not in allowed:
                if number is not None:
                    unexpected_numbers.add(number)
                continue
            if str(number) in returned:
                duplicate_numbers.add(number)
                continue
            returned[str(number)] = dict(item)
        if unexpected_numbers or duplicate_numbers:
            reasons: List[str] = []
            if unexpected_numbers:
                reasons.append(
                    "unexpected "
                    + ", ".join(f"Q{number}" for number in sorted(unexpected_numbers))
                )
            if duplicate_numbers:
                reasons.append(
                    "duplicate "
                    + ", ".join(f"Q{number}" for number in sorted(duplicate_numbers))
                )
            raise FullDocumentGradingError(
                "Question visual grader violated the requested batch: "
                + "; ".join(reasons)
            )
        if set(map(int, returned)) != allowed:
            missing = sorted(allowed - set(map(int, returned)))
            raise FullDocumentGradingError(
                "Question visual grader omitted requested question(s): "
                + ", ".join(f"Q{number}" for number in missing)
            )
        batch_usage = _usage_dict(grading_response, fallback_model=model_id)
        resolved_model = str(batch_usage.get("model") or model_id)
        return returned, batch_usage, resolved_model

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
        processing_path = (
            "objective_answer_ledger"
            if prompt_version == OBJECTIVE_PROMPT_VERSION
            else "full_document_visual"
        )

        raw_by_number = {
            int(item.get("question_number")): item
            for item in (raw_payload.get("questions") or [])
            if isinstance(item, dict) and _positive_int(item.get("question_number"))
        }
        for grade in grades:
            question_id = str(grade.question.get("question_id") or "")
            raw_question_result = raw_by_number.get(grade.question_number, {})
            visual_evidence = {
                "evidence_graph_version": raw_payload.get(
                    "evidence_graph_version"
                ),
                "source_page_numbers": list(
                    raw_question_result.get("source_page_numbers") or []
                ),
                "mapping_reason": raw_question_result.get("mapping_reason"),
                "interpretation_hypotheses": list(
                    raw_question_result.get("interpretation_hypotheses") or []
                ),
                "visual_semantics": dict(
                    raw_question_result.get("visual_semantics") or {}
                ),
            }
            semantic_evidence_signature = _semantic_evidence_signature(
                question_id=question_id,
                student_answer=grade.student_answer,
                source_pages=grade.source_pages,
                visual_evidence=visual_evidence,
                prompt_version=prompt_version,
            )
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
                        "objective_answer_ledger"
                        if prompt_version == OBJECTIVE_PROMPT_VERSION
                        else "full_document_visual"
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
                            "method": (
                                "objective_answer_ledger_coverage"
                                if prompt_version == OBJECTIVE_PROMPT_VERSION
                                else "full_document_visual_coverage"
                            ),
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
                "evidence_version": (
                    4
                    if prompt_version == OBJECTIVE_PROMPT_VERSION
                    else (
                        3
                        if prompt_version == _EVIDENCE_GRAPH_PROMPT_VERSION
                        else 2
                    )
                ),
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
                    "eval_path": (
                        f"{processing_path}_not_attempted"
                        if is_missing
                        else processing_path
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
                                "eval_path": processing_path,
                                "model_used": model_used,
                                "grading_run_id": run_id,
                                "manual_review_required": grade.manual_review_required,
                            },
                            "reason": (
                                "Objective answer-ledger evaluation against the "
                                "immutable answer key"
                                if processing_path == "objective_answer_ledger"
                                else (
                                    "Full-document visual evaluation against "
                                    "immutable paper and teacher solution"
                                )
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
            reason=f"{processing_path}_grading",
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
            if warnings
            else "ready"
        )
        await self._db["evalpen_submissions"].update_one(
            {"submission_id": submission_id},
            {
                "$set": {
                    "segmentation_status": "complete",
                    "processing_path": (
                        "objective_answer_ledger"
                        if prompt_version == OBJECTIVE_PROMPT_VERSION
                        else "full_document_visual"
                    ),
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
            # Technical processing completed successfully. Review and
            # publication eligibility are independent states.
            status="completed",
            page_count=page_count,
            response_count=len(response_docs),
            evaluated_count=evaluated,
            blocked_count=blocked,
            warning_count=warnings,
            processing_path=processing_path,
            run_id=run_id,
            materialization_id=materialization_id,
            errors=errors,
            document_review_required=document_review.required,
            review_state=review_state,
            review_reasons=list(dict.fromkeys(review_reasons)),
        )


async def _fail_generation_run(
    tenant_db: Any,
    *,
    run_id: str,
    generation_lease_token: str,
    error: Exception | str,
) -> bool:
    """Release a paid-call lease after failure or orderly worker shutdown."""

    result = await tenant_db[_RUNS_COLLECTION].update_one(
        {
            "run_id": run_id,
            "generation_lease_token": generation_lease_token,
        },
        {
            "$set": {
                "status": "failed",
                "generation_error": str(error)[:500],
                "updated_at": datetime.now(timezone.utc),
            },
            "$unset": {
                "generation_lease_token": "",
                "generation_lease_expires_at": "",
            },
        },
    )
    return result.matched_count == 1


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
    interpretation even when the immutable input bytes did not change.
    """

    now = datetime.now(timezone.utc)
    lease_token = uuid.uuid4().hex
    lease_expires_at = now + timedelta(minutes=15)
    collection = tenant_db[_RUNS_COLLECTION]
    existing = await collection.find_one({"run_id": run_id})
    try:
        existing_revision = int(
            (existing or {}).get("generation_revision")
            if (existing or {}).get("generation_revision") is not None
            else (existing or {}).get("grading_revision")
            or 0
        )
    except (TypeError, ValueError):
        existing_revision = -1
    existing_generation_fingerprint = str(
        (existing or {}).get("generation_fingerprint")
        or (existing or {}).get("input_fingerprint")
        or ""
    )
    existing_submission = str((existing or {}).get("submission_id") or "")
    if existing is not None and existing_submission and existing_submission != submission_id:
        raise FullDocumentGradingError(
            "Submission grading run ownership does not match the requested generation"
        )
    # Legacy rows used input_fingerprint as run_id and lacked generation_*.
    # When reprocess bumps generation_revision, the same run_id can collide with
    # a failed/expired row from an earlier generation. Reclaim that terminal row
    # for the new generation instead of looping on ownership errors forever.
    identity_mismatch = existing is not None and (
        existing_revision != generation_revision
        or (
            bool(existing_generation_fingerprint)
            and existing_generation_fingerprint != generation_fingerprint
        )
    )
    if identity_mismatch:
        status = str((existing or {}).get("status") or "")
        reclaimable = status in {"failed", "generating"}
        lease_expired = False
        expires_at = (existing or {}).get("generation_lease_expires_at")
        if expires_at is not None:
            try:
                if getattr(expires_at, "tzinfo", None) is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                lease_expired = expires_at <= now
            except Exception:
                lease_expired = True
        # Never silently overwrite a completed generation on a run_id collision.
        if reclaimable or (status == "generating" and lease_expired):
            reclaimed_legacy = await collection.update_one(
                {"run_id": run_id, "submission_id": submission_id},
                {
                    "$set": {
                        "status": "generating",
                        "student_id": student_id,
                        "exam_id": exam_id,
                        "grading_revision": generation_revision,
                        "generation_revision": generation_revision,
                        "generation_fingerprint": generation_fingerprint,
                        "prompt_version": prompt_version,
                        "requested_model_id": requested_model_id,
                        "input_fingerprint": input_fingerprint,
                        "page_count": page_count,
                        "generation_lease_token": lease_token,
                        "generation_lease_expires_at": lease_expires_at,
                        "generation_error": None,
                        "updated_at": now,
                    },
                    "$unset": {
                        "validated_payload": "",
                        "raw_llm_response": "",
                        "result": "",
                        "token_usage": "",
                        "completed_at": "",
                        "evidence_graph_mapping": "",
                        "evidence_graph_mapping_raw": "",
                        "evidence_graph_mapping_usage": "",
                        "evidence_graph_question_grades": "",
                        "evidence_graph_question_grade_usages": "",
                    },
                },
            )
            if reclaimed_legacy.matched_count == 1:
                logger.warning(
                    "Reclaimed legacy grading run_id=%s for submission=%s "
                    "generation_revision=%s (was revision=%s status=%s)",
                    run_id,
                    submission_id,
                    generation_revision,
                    existing_revision,
                    (existing or {}).get("status"),
                )
                return None, lease_token
        raise FullDocumentGradingError(
            "Submission grading run ownership does not match the requested generation"
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
                        # Keep grading_revision during the compatibility window
                        # so old reporting readers can still inspect new runs.
                        "grading_revision": generation_revision,
                        "generation_revision": generation_revision,
                        "generation_fingerprint": generation_fingerprint,
                        "prompt_version": prompt_version,
                        "requested_model_id": requested_model_id,
                        "input_fingerprint": input_fingerprint,
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
                "This submission generation is already being graded; retry after its "
                "current run finishes"
            )
        await asyncio.sleep(0.5)


async def _generation_revision(tenant_db: Any, submission_id: str) -> int:
    """Return the retry-stable model-generation revision for this submission.

    Technical retries keep the same generation. An explicit reprocess
    increments the generation and therefore creates both a fresh provider
    request and fresh immutable response/evaluation rows. ``reprocess_count``
    remains a backward-compatible fallback for jobs created before the
    generation contract was introduced.
    """

    jobs = await tenant_db[_PROCESSING_JOBS_COLLECTION].find(
        {"submission_id": submission_id}
    ).sort([("created_at", -1), ("updated_at", -1)]).to_list(length=1)
    if not jobs:
        return 0
    try:
        raw_revision = jobs[0].get("generation_revision")
        if raw_revision is None:
            raw_revision = jobs[0].get("reprocess_count")
        return max(0, int(raw_revision or 0))
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
            OBJECTIVE_PAPER_CONTEXT_VERSION,
        }
    )


def _paper_requires_evidence_graph(
    paper_version: Optional[Dict[str, Any]],
) -> bool:
    context = dict((paper_version or {}).get("paper_context") or {})
    return bool(
        context.get("ready")
        and str(context.get("version") or "")
        == "canonical-full-document-visual-v2"
    )


def _paper_requires_objective_ledger(
    paper_version: Optional[Dict[str, Any]],
) -> bool:
    context = dict((paper_version or {}).get("paper_context") or {})
    return bool(
        context.get("ready")
        and str(context.get("version") or "")
        == OBJECTIVE_PAPER_CONTEXT_VERSION
    )


def _is_openai_visual_model(model_id: str) -> bool:
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider and provider != "openai":
        return False
    normalized = model_id.strip().lower()
    return normalized.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))


def _model_supports_original_image_detail(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    return normalized.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.6"))


def _responses_temperature_is_effective(
    model_id: str,
    reasoning_effort: str,
) -> bool:
    normalized = model_id.strip().lower()
    if normalized.startswith("gpt-5"):
        return reasoning_effort.strip().lower() == "none"
    return True


async def _read_canonical_file(
    storage_path: str,
    *,
    expected_sha256: Any = None,
) -> Optional[bytes]:
    data = await read_canonical_asset(
        storage_path,
        max_bytes=_MAX_STATIC_PDF_BYTES,
    )
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
            raise CanonicalAssetUnavailableError(
                f"Canonical student page {page_number or '?'} has no image asset"
            )
        try:
            image_b64 = await _resolve_image_base64(
                raw_ref,
                expected_sha256=page.get("asset_sha256"),
            )
        except AssetIntegrityError:
            raise
        except Exception as exc:
            raise CanonicalAssetUnavailableError(
                f"Canonical student page {page_number} could not be loaded from storage"
            ) from exc
        if not image_b64:
            raise CanonicalAssetUnavailableError(
                f"Canonical student page {page_number} could not be loaded"
            )
        try:
            original = base64.b64decode(image_b64, validate=True)
        except Exception as exc:
            raise FullDocumentGradingError(
                f"Canonical student page {page_number} is not a valid image"
            ) from exc
        optimized, media_type = await asyncio.to_thread(_optimize_image, original)
        total_bytes += len(optimized)
        content.append(
            {
                "type": "input_text",
                "text": f"Student answer-copy page {page_number}:",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": (
                    f"data:{media_type};base64,"
                    + base64.b64encode(optimized).decode("ascii")
                ),
                "detail": "high",
            }
        )
    return content, total_bytes


async def _student_page_assets(
    answer_pages: List[Dict[str, Any]],
) -> tuple[List[_StudentPageAsset], int]:
    assets: List[_StudentPageAsset] = []
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
        optimized, media_type = await asyncio.to_thread(_optimize_image, original)
        total_bytes += len(optimized)
        assets.append(
            _StudentPageAsset(
                page_number=page_number,
                original_bytes=original,
                global_bytes=optimized,
                global_media_type=media_type,
                original_media_type=_detect_image_media_type(original),
            )
        )
    return assets, total_bytes


def _detect_image_media_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    raise FullDocumentGradingError(
        "Canonical student page uses an unsupported image format"
    )


def _student_content_from_assets(
    assets: Sequence[_StudentPageAsset],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "STUDENT ANSWER COPY. Inspect every page visually. Page labels below "
                "are authoritative source-page numbers, not question numbers."
            ),
        }
    ]
    for asset in assets:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": f"Student answer-copy page {asset.page_number}:",
                },
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{asset.global_media_type};base64,"
                        + base64.b64encode(asset.global_bytes).decode("ascii")
                    ),
                    "detail": "high",
                },
            ]
        )
    return content


def _build_objective_page_input(
    *,
    asset: _StudentPageAsset,
    catalog: Sequence[Mapping[str, Any]],
    model_id: str,
) -> tuple[List[Dict[str, Any]], int]:
    page_content, page_bytes = _objective_page_visual_content(
        asset,
        model_id=model_id,
    )
    return (
        [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": objective_reader_instructions(),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "IMMUTABLE EXTRACTION CATALOG. This catalog contains "
                            "only conducted question numbers, answer formats, and "
                            "permitted option labels. "
                            "It does not contain the correct-answer key:\n"
                            + json.dumps(
                                list(catalog),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"TASK: Inspect the complete submitted page "
                            f"{asset.page_number} once and return its compact answer "
                            f"ledger with page_number={asset.page_number}."
                        ),
                    },
                    *page_content,
                ],
            },
        ],
        page_bytes,
    )


def build_objective_page_call_spec(
    *,
    asset: _StudentPageAsset,
    catalog: Sequence[Mapping[str, Any]],
    model_id: str,
    prompt_cache_key: str,
    reasoning_effort: str,
    temperature: Optional[float],
    submission_id: str,
    exam_id: str,
    run_id: str,
    question_count: int,
) -> tuple[Dict[str, Any], int]:
    """Build the authoritative provider contract for one objective page.

    The grader and staff debugger both call this function.  This prevents the
    UI from displaying an approximation of what the provider saw.
    """

    request_input, request_bytes = _build_objective_page_input(
        asset=asset,
        catalog=catalog,
        model_id=model_id,
    )
    question_numbers = [
        int(item["question_number"])
        for item in catalog
        if _positive_int(item.get("question_number"))
    ]
    call_spec: Dict[str, Any] = {
        "model_id": model_id,
        "prompt": "",
        "caller_id": _CALLER_ID,
        "responses_input": request_input,
        "json_schema": objective_page_observation_schema(question_numbers),
        "prompt_cache_key": prompt_cache_key,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": _objective_page_output_token_budget(
            question_count=question_count,
            reasoning_effort=reasoning_effort,
        ),
        "metadata": {
            "pcr_stage": "objective_answer_page_reading",
            "prompt_version": OBJECTIVE_PROMPT_VERSION,
            "output_budget_policy": _OBJECTIVE_OUTPUT_BUDGET_POLICY,
            "submission_id": submission_id,
            "exam_id": exam_id,
            "page_number": asset.page_number,
            "question_count": question_count,
            "run_id": run_id,
        },
    }
    if _responses_temperature_is_effective(model_id, reasoning_effort):
        call_spec["temperature"] = temperature
    return call_spec, request_bytes


def _objective_page_output_token_budget(
    *,
    question_count: int,
    reasoning_effort: str,
) -> int:
    """Size an objective ledger response from its contracted row count.

    ``max_output_tokens`` covers both visible structured output and reasoning
    tokens for Responses reasoning models.  A fixed 60-token allowance per
    question left a 75-row OMR with almost no reasoning headroom and could cut
    valid JSON near the end of the ledger.  This policy reserves independent
    capacity for the JSON envelope, every row, and the configured reasoning
    effort.  It is a ceiling, not prepaid usage; a completed ledger stops
    naturally before consuming the allowance.
    """

    try:
        normalized_count = max(1, int(question_count))
    except (TypeError, ValueError):
        normalized_count = 1
    reasoning_reserve = _OBJECTIVE_REASONING_TOKEN_RESERVE.get(
        str(reasoning_effort or "").strip().lower(),
        _OBJECTIVE_REASONING_TOKEN_RESERVE["medium"],
    )
    requested = (
        _OBJECTIVE_OUTPUT_BASE_TOKENS
        + _OBJECTIVE_OUTPUT_TOKENS_PER_QUESTION * normalized_count
        + reasoning_reserve
    )
    return min(
        _OBJECTIVE_OUTPUT_TOKEN_CEILING,
        max(_OBJECTIVE_OUTPUT_TOKEN_FLOOR, requested),
    )


def build_llm_debug_request_manifest(
    call_spec: Mapping[str, Any],
) -> tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, tuple[bytes, str]],
]:
    """Replace inline image bodies with auditable asset descriptors.

    Request text, schema, model parameters and metadata remain exact.  Image
    bytes are not duplicated into MongoDB; their hashes are stored and the
    authorized preview endpoint regenerates the provider bytes from the
    immutable canonical page, then verifies the hash before returning them.
    """

    metadata = dict(call_spec.get("metadata") or {})
    page_number = int(metadata.get("page_number") or 0)
    image_assets: List[Dict[str, Any]] = []
    image_blobs: Dict[str, tuple[bytes, str]] = {}
    redacted_messages: List[Dict[str, Any]] = []
    image_index = 0

    for raw_message in call_spec.get("responses_input") or []:
        message = dict(raw_message) if isinstance(raw_message, Mapping) else {}
        redacted_content: List[Dict[str, Any]] = []
        preceding_text = ""
        for raw_item in message.get("content") or []:
            item = dict(raw_item) if isinstance(raw_item, Mapping) else {}
            if item.get("type") != "input_image":
                redacted_content.append(item)
                if item.get("type") == "input_text":
                    preceding_text = str(item.get("text") or "").strip()
                continue

            image_url = str(item.get("image_url") or "")
            if not image_url.startswith("data:") or ";base64," not in image_url:
                raise FullDocumentGradingError(
                    "LLM debug tracing received an unsupported image reference"
                )
            header, encoded = image_url.split(",", 1)
            media_type = header[5:].split(";", 1)[0].strip().lower()
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise FullDocumentGradingError(
                    "LLM debug tracing could not decode a provider image"
                ) from exc
            digest = hashlib.sha256(image_bytes).hexdigest()
            image_index += 1
            asset_id = (
                f"page-{page_number}-image-{image_index}-"
                f"{digest[:16]}"
            )
            asset = {
                "asset_id": asset_id,
                "page_number": page_number,
                "sequence": image_index,
                "label": preceding_text[:500],
                "media_type": media_type,
                "byte_count": len(image_bytes),
                "sha256": digest,
                "detail": str(item.get("detail") or "auto"),
            }
            image_assets.append(asset)
            image_blobs[asset_id] = (image_bytes, media_type)
            redacted_content.append(
                {
                    "type": "input_image",
                    "asset_id": asset_id,
                    "media_type": media_type,
                    "byte_count": len(image_bytes),
                    "sha256": digest,
                    "detail": str(item.get("detail") or "auto"),
                    "image_url": "[available through authorized debug asset endpoint]",
                }
            )
        message["content"] = redacted_content
        redacted_messages.append(message)

    manifest = {
        key: value
        for key, value in call_spec.items()
        if key != "responses_input"
    }
    manifest["responses_input"] = redacted_messages
    manifest["security"] = {
        "answer_key_included": False,
        "api_credentials_collected": False,
        "inline_image_bodies_persisted": False,
    }
    return manifest, image_assets, image_blobs


def _objective_page_visual_content(
    asset: _StudentPageAsset,
    *,
    model_id: str,
) -> tuple[List[Dict[str, Any]], int]:
    """Return exactly one canonical page image at the best supported detail."""

    detail = (
        "original"
        if _model_supports_original_image_detail(model_id)
        else "high"
    )
    return (
        [
            {
                "type": "input_text",
                "text": (
                    f"Complete original answer-copy page {asset.page_number}. "
                    "Inspect the whole sheet, including all OMR columns and any "
                    "handwritten numbered answers."
                ),
            },
            _input_image_content(
                asset.original_bytes,
                asset.original_media_type,
                detail=detail,
            ),
        ],
        len(asset.original_bytes),
    )


def _input_image_content(
    image_bytes: bytes,
    media_type: str,
    *,
    detail: str = "high",
) -> Dict[str, Any]:
    return {
        "type": "input_image",
        "image_url": (
            f"data:{media_type};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        ),
        "detail": detail,
    }


def _multistage_static_content(
    *,
    questions: List[Dict[str, Any]],
    paper_bytes: bytes,
    solution_bytes: Optional[bytes],
    paper_filename: str,
    solution_filename: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE MARKING CATALOG. Question IDs, ordering, maximum marks, "
                "criterion maximums, method policy, and acceptable evidence are "
                "authoritative. The PDFs remain visual evidence and may contain "
                "handwriting, formulae, diagrams, tables, or graphs.\n"
                + json.dumps(
                    [_catalog_question(question) for question in questions],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        },
        {"type": "input_text", "text": "ORIGINAL QUESTION PAPER PDF:"},
        {
            "type": "input_file",
            "filename": _safe_pdf_filename(paper_filename, "question-paper.pdf"),
            "file_data": (
                "data:application/pdf;base64,"
                + base64.b64encode(paper_bytes).decode("ascii")
            ),
        },
    ]
    if solution_bytes:
        content.extend(
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
                    "file_data": (
                        "data:application/pdf;base64,"
                        + base64.b64encode(solution_bytes).decode("ascii")
                    ),
                },
            ]
        )
    return content


def _multistage_system_instructions() -> str:
    return (
        mapping_system_instructions()
        + "\n\n"
        + grading_system_instructions()
        + "\n\nFollow only the TASK in the final user message for this request."
    )


def _question_batches(question_numbers: Sequence[int]) -> List[List[int]]:
    try:
        configured = int(
            os.getenv(
                "PCR_VISUAL_QUESTIONS_PER_BATCH",
                str(_DEFAULT_VISUAL_QUESTIONS_PER_BATCH),
            )
            or _DEFAULT_VISUAL_QUESTIONS_PER_BATCH
        )
    except (TypeError, ValueError):
        configured = _DEFAULT_VISUAL_QUESTIONS_PER_BATCH
    size = max(1, min(10, configured))
    numbers = [int(number) for number in question_numbers]
    return [numbers[index : index + size] for index in range(0, len(numbers), size)]


def _unresolved_visual_grade_item(question_number: int, *, reason: str) -> Dict[str, Any]:
    """Placeholder grade so a single bad model call cannot fail the whole copy."""

    return {
        "question_number": int(question_number),
        "confidence": 0.0,
        "student_answer": "",
        "interpretation_hypotheses": [],
        "visual_semantics": {
            "summary": "",
            "elements": [],
            "relationships": [],
            "confidence": 0,
        },
        "method_analysis": {
            "detected_method": "",
            "method_classification": "unresolved",
            "method_validity": "unresolved",
            "confidence": 0.0,
            "explanation": reason[:500],
            "error_carried_forward": "not_applicable",
            "error_carried_forward_reason": "",
        },
        "criterion_marks": [],
        "total_score": 0,
        "overall_feedback": "Automatic visual grading failed for this question.",
        "needs_review": True,
        "review_reason": reason[:500],
    }


def _build_question_grading_input(
    *,
    static_content: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    mappings: Mapping[int, Dict[str, Any]],
    student_assets: Sequence[_StudentPageAsset],
) -> tuple[List[Dict[str, Any]], int]:
    assets = {asset.page_number: asset for asset in student_assets}
    dynamic: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "TASK: Grade exactly the requested questions from the fixed evidence "
                "regions below. Region ownership is immutable for this stage. Use the "
                "original high-resolution crops, not the mapper's text description.\n"
                "REQUESTED QUESTION CATALOG:\n"
                + json.dumps(
                    [_catalog_question(question) for question in questions],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        }
    ]
    crop_bytes = 0
    for index, question in enumerate(questions, start=1):
        number = int(question.get("question_number") or index)
        mapped = mappings.get(number) or {}
        regions = list(mapped.get("evidence_regions") or [])
        dynamic.append(
            {
                "type": "input_text",
                "text": (
                    f"QUESTION {number} FIXED EVIDENCE MAP:\n"
                    + json.dumps(
                        {
                            "question_number": number,
                            "content_type": mapped.get("content_type"),
                            "mapping_reason": mapped.get("mapping_reason"),
                            "evidence_regions": regions,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
        )
        for region in regions:
            page_number = int(region.get("page_number") or 0)
            asset = assets.get(page_number)
            if asset is None:
                raise FullDocumentGradingError(
                    f"Mapped evidence for Q{number} refers to missing page {page_number}"
                )
            cropped, media_type = _crop_student_region(asset, region)
            crop_bytes += len(cropped)
            dynamic.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"Q{number} evidence region "
                            f"{region.get('region_id')} from page {page_number}:"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:{media_type};base64,"
                            + base64.b64encode(cropped).decode("ascii")
                        ),
                        "detail": "high",
                    },
                ]
            )
    return (
        [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": _multistage_system_instructions(),
                    }
                ],
            },
            {"role": "user", "content": static_content},
            {"role": "user", "content": dynamic},
        ],
        crop_bytes,
    )


def _crop_student_region(
    asset: _StudentPageAsset,
    region: Mapping[str, Any],
) -> tuple[bytes, str]:
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(asset.original_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            margin_x = max(8, int(width * 0.02))
            margin_y = max(8, int(height * 0.02))
            left = max(
                0,
                int(width * float(region.get("x_start") or 0) / 1000.0)
                - margin_x,
            )
            top = max(
                0,
                int(height * float(region.get("y_start") or 0) / 1000.0)
                - margin_y,
            )
            right = min(
                width,
                int(width * float(region.get("x_end") or 1000) / 1000.0)
                + margin_x,
            )
            bottom = min(
                height,
                int(height * float(region.get("y_end") or 1000) / 1000.0)
                + margin_y,
            )
            if right - left < 24 or bottom - top < 24:
                raise ValueError("evidence crop is too small")
            crop = image.crop((left, top, right, bottom))
            if crop.mode not in {"RGB", "L"}:
                background = Image.new("RGB", crop.size, "white")
                if "A" in crop.getbands():
                    background.paste(crop, mask=crop.getchannel("A"))
                else:
                    background.paste(crop.convert("RGB"))
                crop = background
            elif crop.mode == "L":
                crop = crop.convert("RGB")
            else:
                crop = crop.copy()
            crop.thumbnail((2600, 2600))
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=94, optimize=True)
            value = output.getvalue()
            if not value:
                raise ValueError("evidence crop is empty")
            return value, "image/jpeg"
    except Exception as exc:
        raise FullDocumentGradingError(
            "Could not create a high-resolution student evidence crop for "
            f"region {region.get('region_id') or '?'}"
        ) from exc


def _optimize_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Bound request size while retaining handwriting and diagram detail."""
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")
            else:
                image = image.copy()
            image.thumbnail((2400, 2400))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            optimized = output.getvalue()
            if optimized:
                return optimized, "image/jpeg"
    except Exception:
        logger.warning("Could not optimize a student page; using canonical bytes")
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return image_bytes, media_type


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
        "You are the primary visual examiner for a high-stakes handwritten exam. "
        "Read the original question paper, the teacher solution/marking scheme, "
        "and the student's complete answer copy directly. OCR text is not supplied "
        "and must not be treated as a gate. Use visual reasoning for handwriting, "
        "mathematics, arrows, tables, graphs, geometry, circuit diagrams, crossed-out "
        "work, and answers written out of order.\n\n"
        "Build a private evidence ledger across the entire student copy before grading. "
        "Match work to questions using visible question labels, given values, requested "
        "result, method, diagram semantics, and page continuity. Printed question or "
        "teacher-solution content is never student evidence. Do not copy the answer key "
        "into student_answer or evidence. A student may answer in any order and may put "
        "several questions on one page or one question across several pages.\n\n"
        "For catalog questions with grading_mode=objective, your role is extraction "
        "only: transcribe the student's selected option label into student_answer "
        "(for example A, B, C, or D). Do not decide whether it is correct and do not "
        "calculate marks. Return empty criterion_marks, total_score 0, and "
        "not_applicable method analysis; the server applies the immutable answer key "
        "and negative-marking rule. If more than one option is plausible, return "
        "unresolved instead of choosing one.\n\n"
        "Before awarding marks, reconstruct the student's own approach for each "
        "attempted question. Keep method identity separate from correctness: set "
        "method_classification to reference_method, alternative_method, "
        "specified_method, no_method_visible, not_applicable, or unresolved; then "
        "set method_validity independently to valid, partially_valid, invalid, "
        "not_applicable, or unresolved. alternative_method means only that the "
        "approach differs from the reference; it does not itself say that the "
        "approach is correct. The server derives whether the frozen method policy "
        "was satisfied, so do not return that policy decision. The "
        "teacher solution is an identity and correctness anchor, not a template that "
        "the student's working must resemble. Equivalent algebra, reordered steps, "
        "different valid formulae, concise mental steps, and valid diagram-based or "
        "verbal reasoning must receive the same criterion decision when they establish "
        "the same required fact. Never invent work that is not visible. Enforce a named "
        "method only when the catalog method_policy says specified_method_required. "
        "When error-carried-forward is enabled, isolate the earliest visible error and "
        "still award later method/reasoning criteria that are internally correct using "
        "the student's own value; do not repeatedly penalize one earlier error.\n\n"
        "For every catalog question return exactly one result. attempt_status=attempted "
        "only when student work is visibly present. Use not_attempted only after checking "
        "every student page and finding no work for that question. Use unresolved when "
        "ownership, handwriting, page coverage, or the correct award is genuinely "
        "uncertain. Never guess a zero. For attempted answers, apply only the locked "
        "criterion IDs and maximums from the catalog and return every locked criterion "
        "exactly once. For not_attempted return empty student_answer, evidence_regions, "
        "and criterion_marks with total_score 0. For unresolved return no award and empty "
        "criterion_marks with total_score 0. For each attempted answer compare visible "
        "student evidence to each criterion's acceptable_evidence independently; do not "
        "grade by overall impression. Use decision=met only with the full criterion mark, "
        "decision=not_met only with zero, decision=partially_met only when both achieved "
        "and missing parts are identified, and decision=unresolved when the criterion "
        "cannot be judged reliably. Any unresolved criterion makes the question review-only. "
        "Equivalent evidence must receive the same criterion decision regardless of student, "
        "handwriting style, answer order, or surrounding answers. Award step marks for "
        "correct visible work even when the final answer is wrong. Set each criterion's "
        "credit_basis to direct_evidence, error_carried_forward, no_credit, or unresolved "
        "so every awarded mark can be audited. Evaluate diagrams visually. "
        "Cite the student page and a short literal/visual description for every criterion, "
        "including why a zero was awarded. Never use the teacher solution as student evidence. "
        "Do not exceed any criterion or question maximum. Set needs_review for low-quality "
        "images, ambiguous ownership, contradictory work, unreadable evidence, or any "
        "uncertain award. Coordinates are approximate vertical bands from 0 at page top "
        "to 1000 at page bottom.\n\n"
        "For document_review, all_student_work_accounted means that every visible "
        "student mark on every submitted page has been assigned to a catalog question "
        "or the relevant questions were explicitly found not attempted. A routine note "
        "that some questions were not attempted is not uncertainty. The warnings array "
        "is explanatory only: if a warning describes cropped pages, unreadable work, "
        "unassigned writing, or any other real coverage uncertainty, also set "
        "all_student_work_accounted=false or lower confidence accordingly. Never put an "
        "ordinary not-attempted summary in warnings."
    )


def _evidence_ledger_schema() -> Dict[str, Any]:
    region = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_number": {"type": "integer", "minimum": 1},
            "y_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "evidence": {"type": "string"},
        },
        "required": ["page_number", "y_start", "y_end", "evidence"],
    }
    criterion = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": ["met", "partially_met", "not_met", "unresolved"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "marks_awarded": {"type": "number", "minimum": 0},
            "rationale": {"type": "string"},
            "evidence": {"type": "string"},
            "missing_evidence": {"type": "string"},
            "credit_basis": {
                "type": "string",
                "enum": [
                    "direct_evidence",
                    "error_carried_forward",
                    "no_credit",
                    "unresolved",
                ],
            },
        },
        "required": [
            "criterion_id",
            "decision",
            "confidence",
            "marks_awarded",
            "rationale",
            "evidence",
            "missing_evidence",
            "credit_basis",
        ],
    }
    method_analysis = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "detected_method": {"type": "string"},
            "method_classification": {
                "type": "string",
                "enum": [
                    "reference_method",
                    "alternative_method",
                    "specified_method",
                    "no_method_visible",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "method_validity": {
                "type": "string",
                "enum": [
                    "valid",
                    "partially_valid",
                    "invalid",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string"},
            "error_carried_forward": {
                "type": "string",
                "enum": [
                    "applied",
                    "not_applied",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "error_carried_forward_reason": {"type": "string"},
        },
        "required": [
            "detected_method",
            "method_classification",
            "method_validity",
            "confidence",
            "explanation",
            "error_carried_forward",
            "error_carried_forward_reason",
        ],
    }
    question = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "integer", "minimum": 1},
            "attempt_status": {
                "type": "string",
                "enum": ["attempted", "not_attempted", "unresolved"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "student_answer": {"type": "string"},
            "content_type": {
                "type": "string",
                "enum": ["TEXT_ONLY", "MIXED", "DIAGRAM_HEAVY", "TABLE_PRESENT"],
            },
            "evidence_regions": {"type": "array", "items": region},
            "method_analysis": method_analysis,
            "criterion_marks": {"type": "array", "items": criterion},
            "total_score": {"type": "number", "minimum": 0},
            "overall_feedback": {"type": "string"},
            "needs_review": {"type": "boolean"},
            "review_reason": {"type": "string"},
        },
        "required": [
            "question_number",
            "attempt_status",
            "confidence",
            "student_answer",
            "content_type",
            "evidence_regions",
            "method_analysis",
            "criterion_marks",
            "total_score",
            "overall_feedback",
            "needs_review",
            "review_reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_review": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "all_student_work_accounted": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["all_student_work_accounted", "confidence", "warnings"],
            },
            "questions": {"type": "array", "items": question},
        },
        "required": ["document_review", "questions"],
    }


def _validate_ledger(
    payload: Dict[str, Any],
    *,
    questions: List[Dict[str, Any]],
    page_count: int,
) -> tuple[List[_ValidatedGrade], List[str], _DocumentReview]:
    objective_ledger = (
        str(payload.get("evidence_graph_version") or "")
        == OBJECTIVE_LEDGER_VERSION
    )
    raw_document_review = payload.get("document_review")
    structural_errors = [
        str(error).strip()[:500]
        for error in (payload.get("evidence_graph_validation_errors") or [])
        if str(error).strip()
    ]
    document_warnings: List[str] = []
    coverage_complete = False
    coverage_confidence = 0.0
    if isinstance(raw_document_review, dict):
        coverage_complete = bool(
            raw_document_review.get("all_student_work_accounted")
        )
        coverage_confidence = _confidence(raw_document_review.get("confidence"))
        for warning in raw_document_review.get("warnings") or []:
            if str(warning).strip():
                document_warnings.append(str(warning).strip()[:300])
    else:
        document_warnings.append("Model omitted the full-copy coverage review")

    document_review = _DocumentReview(
        all_student_work_accounted=coverage_complete,
        confidence=coverage_confidence,
        warnings=document_warnings,
        required=(
            not coverage_complete
            or coverage_confidence < _AUTO_ACCEPT_CONFIDENCE
        ),
    )
    if structural_errors:
        document_review.required = True
        document_warnings.append(
            "The visual evidence graph has structural validation errors"
        )
    # Machine decisions use typed coverage fields, never the wording of a
    # free-text note. A note may explain that Q1/Q2 were absent without
    # contradicting high-confidence full-copy coverage. Genuine uncertainty
    # remains blocking through the typed boolean/confidence fields.
    absence_coverage_complete = coverage_complete

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
            coverage_complete=absence_coverage_complete,
            coverage_confidence=coverage_confidence,
            objective_ledger=objective_ledger,
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
    if not objective_ledger:
        _mark_overlapping_evidence_for_review(grades)
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


def _validate_method_analysis(
    raw: Any,
    *,
    method_policy: Dict[str, Any],
) -> tuple[Dict[str, Any], List[str], List[str]]:
    """Normalize method metadata without turning metadata defects into lost marks.

    Method identity and method validity are independent observations. Policy
    satisfaction is derived here from those observations, rather than trusted
    as another model-generated boolean that can contradict them.
    """

    if not isinstance(raw, dict):
        return (
            {
                **_not_applicable_method_analysis(),
                "method_classification": "unresolved",
                "method_validity": "unresolved",
                "method_requirement_satisfied": False,
                "confidence": 0.0,
                "explanation": "The model omitted method analysis.",
                "error_carried_forward": "unresolved",
            },
            [],
            ["The student's method could not be reconstructed reliably"],
        )

    classifications = {
        "reference_method",
        "alternative_method",
        # Accepted only to rematerialize ledgers produced by the previous
        # contract. It is normalized to alternative_method below.
        "valid_alternative",
        "specified_method",
        "no_method_visible",
        "not_applicable",
        "unresolved",
    }
    validities = {
        "valid",
        "partially_valid",
        "invalid",
        "not_applicable",
        "unresolved",
    }
    follow_through_states = {
        "applied",
        "not_applied",
        "not_applicable",
        "unresolved",
    }
    classification = str(raw.get("method_classification") or "").strip().lower()
    validity = str(raw.get("method_validity") or "").strip().lower()
    follow_through = str(raw.get("error_carried_forward") or "").strip().lower()
    explanation = str(raw.get("explanation") or "").strip()
    follow_through_reason = str(
        raw.get("error_carried_forward_reason") or ""
    ).strip()
    detected_method = str(raw.get("detected_method") or "").strip()
    confidence = _confidence(raw.get("confidence"))

    errors: List[str] = []
    review_reasons: List[str] = []
    if classification == "valid_alternative":
        classification = "alternative_method"
    if classification not in classifications:
        classification = "unresolved"
        review_reasons.append("Method analysis has an invalid classification")
    if validity not in validities:
        validity = "unresolved"
        review_reasons.append("Method analysis has an invalid validity decision")
    if follow_through not in follow_through_states:
        follow_through = "unresolved"
        review_reasons.append("Method analysis has an invalid follow-through decision")
    if not explanation:
        explanation = "The model did not explain its method decision."
        review_reasons.append("Method analysis has no explanation")
    if classification in {
        "reference_method",
        "alternative_method",
        "specified_method",
    } and not detected_method:
        review_reasons.append(
            "Method analysis named a method class without describing the method"
        )
    if follow_through == "applied":
        if not method_policy.get("allow_error_carried_forward", True):
            errors.append("Follow-through marks were applied although the frozen policy forbids them")
        if not follow_through_reason:
            errors.append("Applied follow-through credit has no explanation")

    mode = str(method_policy.get("mode") or ANY_VALID_METHOD)
    if mode == SPECIFIED_METHOD_REQUIRED:
        requirement_satisfied = (
            classification == "specified_method" and validity == "valid"
        )
        if not requirement_satisfied:
            review_reasons.append("The explicitly required method was not verified")
    elif mode == NO_METHOD_REQUIRED:
        requirement_satisfied = True
    else:
        requirement_satisfied = validity == "valid" and classification in {
            "reference_method",
            "alternative_method",
            "specified_method",
            "no_method_visible",
        }

    if classification == "unresolved" or validity == "unresolved":
        review_reasons.append("The student's method could not be reconstructed reliably")
    if confidence < _CRITERION_MIN_SCORE_CONFIDENCE:
        review_reasons.append("Method reconstruction confidence is below the review threshold")

    return (
        {
            "detected_method": detected_method[:2000],
            "method_classification": classification,
            "method_validity": validity,
            "method_requirement_satisfied": bool(requirement_satisfied),
            "confidence": confidence,
            "explanation": explanation[:3000],
            "error_carried_forward": follow_through,
            "error_carried_forward_reason": follow_through_reason[:3000],
        },
        errors,
        review_reasons,
    )


def _validate_question_grade(
    item: Dict[str, Any],
    *,
    question: Dict[str, Any],
    question_number: int,
    page_count: int,
    coverage_complete: bool,
    coverage_confidence: float,
    objective_ledger: bool = False,
) -> _ValidatedGrade:
    status = str(item.get("attempt_status") or "unresolved").strip().lower()
    if status not in {"attempted", "not_attempted", "unresolved"}:
        status = "unresolved"
    confidence = _confidence(item.get("confidence"))
    student_answer = str(item.get("student_answer") or "").strip()
    content_type = str(item.get("content_type") or ContentType.MIXED.value)
    if content_type not in {value.value for value in ContentType}:
        content_type = ContentType.MIXED.value
    source_pages, region_errors = _validate_regions(
        item.get("evidence_regions"),
        page_count=page_count,
    )
    validation_errors = list(region_errors)
    evidence_region_ids = {
        str(region.get("region_id") or "")
        for region in source_pages
        if str(region.get("region_id") or "")
    }
    evidence_graph_question = bool(evidence_region_ids)
    max_marks = _max_marks(question)
    criteria = _criteria(question)
    method_policy = _question_method_policy(question)
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
        absence_threshold = (
            0.55 if objective_ledger else _ABSENCE_CONFIDENCE
        )
        if not coverage_complete or coverage_confidence < absence_threshold:
            return _unresolved_grade(
                question,
                question_number,
                "The full-copy scan did not prove that this question was unattempted",
                confidence=min(confidence, coverage_confidence),
            )
        if confidence < absence_threshold:
            return _unresolved_grade(
                question,
                question_number,
                "The model was not confident enough to record a not-attempted zero",
                confidence=confidence,
            )
        criterion_marks = [
            {
                "criterion_id": criterion["criterion_id"],
                "description": criterion["description"],
                "marks_awarded": 0.0,
                "max_marks": criterion["max_marks"],
                "decision": "not_met",
                "confidence": min(confidence, coverage_confidence),
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

    if not student_answer:
        validation_errors.append("Attempted answer has no student transcription")
    if not source_pages and not objective_ledger:
        validation_errors.append("Attempted answer has no visual evidence region")
    if confidence < (0.55 if objective_ledger else 0.50):
        validation_errors.append("Question ownership confidence is below 0.50")
    if evidence_graph_question:
        hypotheses = item.get("interpretation_hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            validation_errors.append(
                "Attempted answer has no auditable visual interpretation hypothesis"
            )
        else:
            hypothesis_confidences: List[float] = []
            for hypothesis in hypotheses:
                if not isinstance(hypothesis, dict):
                    validation_errors.append(
                        "Visual interpretation hypothesis is not an object"
                    )
                    continue
                value = str(hypothesis.get("value") or "").strip()
                hypothesis_region_ids = {
                    str(region_id).strip()
                    for region_id in (hypothesis.get("evidence_region_ids") or [])
                    if str(region_id).strip()
                }
                hypothesis_confidence = _confidence(hypothesis.get("confidence"))
                if not value:
                    validation_errors.append(
                        "Visual interpretation hypothesis has no value"
                    )
                if not hypothesis_region_ids:
                    validation_errors.append(
                        "Visual interpretation hypothesis cites no evidence region"
                    )
                elif not hypothesis_region_ids.issubset(evidence_region_ids):
                    validation_errors.append(
                        "Visual interpretation hypothesis cites evidence outside "
                        "the fixed question map"
                    )
                hypothesis_confidences.append(hypothesis_confidence)
            ranked_hypotheses = sorted(hypothesis_confidences, reverse=True)
            if (
                ranked_hypotheses
                and ranked_hypotheses[0] < _CRITERION_MIN_SCORE_CONFIDENCE
            ):
                validation_errors.append(
                    "No visual interpretation is reliable enough to score"
                )
            elif (
                len(ranked_hypotheses) > 1
                and ranked_hypotheses[1] >= ranked_hypotheses[0] - 0.10
            ):
                # A close second reading (for example 2^3 versus 23) is a
                # genuine pixel-level ambiguity. Keep the calculated marks,
                # but prevent automatic publication instead of guessing.
                manual_review = True
                if not review_reason:
                    review_reason = (
                        "Two plausible visual readings are too close to resolve "
                        "automatically"
                    )
        visual_semantics = item.get("visual_semantics")
        if not objective_question and not isinstance(visual_semantics, dict):
            validation_errors.append(
                "Attempted answer has no structured visual semantics"
            )
        elif not objective_question:
            visual_confidence = _confidence(visual_semantics.get("confidence"))
            if visual_confidence < _CRITERION_MIN_SCORE_CONFIDENCE:
                validation_errors.append(
                    "Visual semantics could not be verified with sufficient confidence"
                )
            elif visual_confidence < _CRITERION_AUTO_ACCEPT_CONFIDENCE:
                manual_review = True
                if not review_reason:
                    review_reason = (
                        "Visual semantics confidence is below the automatic threshold"
                    )
            if content_type in {
                ContentType.DIAGRAM_HEAVY.value,
                ContentType.TABLE_PRESENT.value,
            } and not list(visual_semantics.get("elements") or []):
                validation_errors.append(
                    "Visual-heavy answer has no identified semantic elements"
                )
            visual_elements = visual_semantics.get("elements")
            visual_elements = (
                visual_elements if isinstance(visual_elements, list) else []
            )
            element_ids: set[str] = set()
            for element in visual_elements:
                if not isinstance(element, dict):
                    validation_errors.append(
                        "Visual semantic element is not an object"
                    )
                    continue
                element_id = str(element.get("element_id") or "").strip()
                region_id = str(element.get("region_id") or "").strip()
                if not element_id or element_id in element_ids:
                    validation_errors.append(
                        "Visual semantic elements have missing or duplicate IDs"
                    )
                else:
                    element_ids.add(element_id)
                if region_id not in evidence_region_ids:
                    validation_errors.append(
                        "Visual semantic element cites evidence outside the fixed "
                        "question map"
                    )
            visual_relationships = visual_semantics.get("relationships")
            visual_relationships = (
                visual_relationships
                if isinstance(visual_relationships, list)
                else []
            )
            for relationship in visual_relationships:
                if not isinstance(relationship, dict):
                    validation_errors.append(
                        "Visual semantic relationship is not an object"
                    )
                    continue
                if (
                    str(relationship.get("source_element_id") or "").strip()
                    not in element_ids
                    or str(relationship.get("target_element_id") or "").strip()
                    not in element_ids
                ):
                    validation_errors.append(
                        "Visual semantic relationship refers to an unknown element"
                    )

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
        if confidence < _AUTO_ACCEPT_CONFIDENCE and not objective_ledger:
            manual_review = True
            review_reason = review_reason or (
                "The selected option could not be read with automatic-publish confidence"
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

    method_analysis, method_errors, method_review_reasons = _validate_method_analysis(
        item.get("method_analysis"),
        method_policy=method_policy,
    )
    validation_errors.extend(method_errors)
    if method_review_reasons:
        manual_review = True
        if not review_reason:
            review_reason = "; ".join(dict.fromkeys(method_review_reasons))

    raw_marks = item.get("criterion_marks")
    raw_marks = raw_marks if isinstance(raw_marks, list) else []
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for raw in raw_marks:
        if isinstance(raw, dict):
            by_id.setdefault(str(raw.get("criterion_id") or "").strip(), []).append(raw)
    expected_ids = {criterion["criterion_id"] for criterion in criteria}
    if set(by_id) != expected_ids:
        validation_errors.append("Criterion IDs do not match the locked marking plan")
    criterion_review_reasons: List[str] = []
    criterion_unresolved_reasons: List[str] = []
    for criterion in criteria:
        rows = by_id.get(criterion["criterion_id"], [])
        if len(rows) != 1:
            validation_errors.append(
                f"Criterion {criterion['criterion_id']} is missing or duplicated"
            )
            continue
        raw = rows[0]
        criterion_id = criterion["criterion_id"]
        decision = str(raw.get("decision") or "").strip().lower()
        if decision not in {"met", "partially_met", "not_met", "unresolved"}:
            validation_errors.append(
                f"Criterion {criterion_id} has an invalid evidence decision"
            )
            continue
        criterion_confidence = _confidence(raw.get("confidence"))
        awarded = _finite_float(raw.get("marks_awarded"))
        if awarded is None or awarded < 0 or awarded > criterion["max_marks"]:
            validation_errors.append(
                f"Criterion {criterion_id} award is outside its locked range"
            )
            continue
        maximum = criterion["max_marks"]
        if decision == "met" and abs(awarded - maximum) > 0.01:
            validation_errors.append(
                f"Criterion {criterion_id} is met but was not awarded its full locked mark"
            )
        elif decision == "not_met" and abs(awarded) > 0.01:
            validation_errors.append(
                f"Criterion {criterion_id} is not met but was awarded marks"
            )
        elif decision == "partially_met" and not (
            awarded > 0.01 and awarded < maximum - 0.01
        ):
            validation_errors.append(
                f"Criterion {criterion_id} partial decision has no valid partial award"
            )
        elif decision == "unresolved" and abs(awarded) > 0.01:
            validation_errors.append(
                f"Criterion {criterion_id} is unresolved but was awarded marks"
            )
        rationale = str(raw.get("rationale") or "").strip()
        evidence = str(raw.get("evidence") or "").strip()
        cited_region_ids = [
            str(value).strip()
            for value in (raw.get("evidence_region_ids") or [])
            if str(value).strip()
        ]
        missing_evidence = str(raw.get("missing_evidence") or "").strip()
        credit_basis = str(raw.get("credit_basis") or "").strip().lower()
        if credit_basis not in {
            "direct_evidence",
            "error_carried_forward",
            "no_credit",
            "unresolved",
        }:
            validation_errors.append(
                f"Criterion {criterion_id} has an invalid credit basis"
            )
        elif decision in {"met", "partially_met"} and credit_basis not in {
            "direct_evidence",
            "error_carried_forward",
        }:
            validation_errors.append(
                f"Criterion {criterion_id} awarded marks without a positive credit basis"
            )
        elif decision == "not_met" and credit_basis != "no_credit":
            validation_errors.append(
                f"Criterion {criterion_id} gave no marks but has a positive credit basis"
            )
        elif decision == "unresolved" and credit_basis != "unresolved":
            validation_errors.append(
                f"Criterion {criterion_id} is unresolved but its credit basis is not"
            )
        if credit_basis == "error_carried_forward":
            if method_analysis.get("error_carried_forward") != "applied":
                validation_errors.append(
                    f"Criterion {criterion_id} claims follow-through credit without a question-level decision"
                )
            if not method_policy.get("allow_error_carried_forward", True):
                validation_errors.append(
                    f"Criterion {criterion_id} claims follow-through credit although the policy forbids it"
                )
        if not rationale:
            validation_errors.append(
                f"Criterion {criterion_id} has no decision rationale"
            )
        if decision != "unresolved" and not evidence:
            validation_errors.append(
                f"Criterion {criterion_id} has no cited student evidence"
            )
        if evidence_graph_question:
            if not cited_region_ids:
                validation_errors.append(
                    f"Criterion {criterion_id} has no cited visual evidence region"
                )
            elif not set(cited_region_ids).issubset(evidence_region_ids):
                validation_errors.append(
                    f"Criterion {criterion_id} cites evidence outside the fixed question map"
                )
        if decision == "partially_met" and not missing_evidence:
            validation_errors.append(
                f"Criterion {criterion_id} has no stated missing evidence for partial credit"
            )
        if decision == "unresolved" or criterion_confidence < _CRITERION_MIN_SCORE_CONFIDENCE:
            criterion_unresolved_reasons.append(
                f"Criterion {criterion_id} could not be judged with sufficient confidence"
            )
        elif criterion_confidence < _CRITERION_AUTO_ACCEPT_CONFIDENCE:
            criterion_review_reasons.append(
                f"Criterion {criterion_id} confidence is below the automatic threshold"
            )
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
        raw_total = _finite_float(item.get("total_score"))
        if raw_total is None or abs(raw_total - total_score) > 0.01:
            validation_errors.append("Criterion awards do not add up to total_score")
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

    if (
        total_score is not None
        and method_policy.get("mode") == SPECIFIED_METHOD_REQUIRED
        and not method_analysis.get("method_requirement_satisfied")
        and abs(total_score - max_marks) <= 0.01
    ):
        validation_errors.append(
            "Full marks cannot be awarded when the explicitly required method was not verified"
        )

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

    if criterion_unresolved_reasons:
        return _unresolved_grade(
            question,
            question_number,
            "; ".join(dict.fromkeys(criterion_unresolved_reasons)),
            confidence=min(
                [confidence]
                + [float(item.get("confidence") or 0.0) for item in criterion_marks]
            ),
            source_pages=source_pages,
            student_answer=student_answer,
            content_type=content_type,
        )

    if criterion_review_reasons:
        manual_review = True
        if not review_reason:
            review_reason = "; ".join(dict.fromkeys(criterion_review_reasons))

    if confidence < _AUTO_ACCEPT_CONFIDENCE:
        manual_review = True
        if not review_reason:
            review_reason = (
                "The question ownership or visual evidence is below the automatic "
                "acceptance threshold"
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


def _validate_regions(
    raw_regions: Any,
    *,
    page_count: int,
) -> tuple[List[Dict[str, Any]], List[str]]:
    regions: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not isinstance(raw_regions, list):
        return [], ["Evidence regions must be an array"]
    for item in raw_regions:
        if not isinstance(item, dict):
            errors.append("Evidence region is not an object")
            continue
        page_number = _positive_int(item.get("page_number"))
        x_start = _finite_float(item.get("x_start"))
        x_end = _finite_float(item.get("x_end"))
        start = _finite_float(item.get("y_start"))
        end = _finite_float(item.get("y_end"))
        if not page_number or page_number > page_count:
            errors.append("Evidence refers to a non-submitted page")
            continue
        if start is None or end is None or start < 0 or end > 1000 or end <= start:
            errors.append("Evidence has an invalid vertical page band")
            continue
        if x_start is None and x_end is None:
            x_start, x_end = 0.0, 1000.0
        if (
            x_start is None
            or x_end is None
            or x_start < 0
            or x_end > 1000
            or x_end <= x_start
        ):
            errors.append("Evidence has an invalid horizontal page band")
            continue
        region: Dict[str, Any] = {
            "page_number": page_number,
            "x_start": round((x_start / 1000.0) * _A4_WIDTH_MM, 3),
            "y_start": round((start / 1000.0) * _A4_HEIGHT_MM, 3),
            "x_end": round((x_end / 1000.0) * _A4_WIDTH_MM, 3),
            "y_end": round((end / 1000.0) * _A4_HEIGHT_MM, 3),
        }
        for key in (
            "region_id",
            "evidence_kind",
            "continuation_group",
            "evidence",
            "mapping_confidence",
        ):
            if item.get(key) is not None:
                region[key] = item.get(key)
        regions.append(region)
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


def _mark_overlapping_evidence_for_review(grades: List[_ValidatedGrade]) -> None:
    for left_index, left in enumerate(grades):
        if left.attempt_status != "attempted":
            continue
        for right in grades[left_index + 1 :]:
            if right.attempt_status != "attempted":
                continue
            if _regions_overlap(left.source_pages, right.source_pages):
                reason = (
                    f"Visual evidence overlaps Q{left.question_number} and "
                    f"Q{right.question_number}; teacher ownership review is required"
                )
                left.manual_review_required = True
                right.manual_review_required = True
                left.review_reason = left.review_reason or reason
                right.review_reason = right.review_reason or reason


def _regions_overlap(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
) -> bool:
    for a in left:
        for b in right:
            if a["page_number"] != b["page_number"]:
                continue
            overlap_y = min(a["y_end"], b["y_end"]) - max(
                a["y_start"], b["y_start"]
            )
            overlap_x = min(
                float(a.get("x_end", _A4_WIDTH_MM)),
                float(b.get("x_end", _A4_WIDTH_MM)),
            ) - max(float(a.get("x_start", 0.0)), float(b.get("x_start", 0.0)))
            if overlap_y <= 0 or overlap_x <= 0:
                continue
            area_overlap = overlap_x * overlap_y
            area_a = (
                float(a.get("x_end", _A4_WIDTH_MM))
                - float(a.get("x_start", 0.0))
            ) * (a["y_end"] - a["y_start"])
            area_b = (
                float(b.get("x_end", _A4_WIDTH_MM))
                - float(b.get("x_start", 0.0))
            ) * (b["y_end"] - b["y_start"])
            smaller = min(area_a, area_b)
            if smaller > 0 and area_overlap / smaller >= 0.50:
                return True
    return False


def _catalog_question(question: Dict[str, Any]) -> Dict[str, Any]:
    policy = _question_marking_policy(question)
    method_policy = _question_method_policy(question)
    objective = _is_objective_question(question)
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
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    candidates = [cleaned]
    # Extract the outermost JSON object when the model wraps prose around it.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            repaired = _repair_truncated_json_object(candidate)
            if repaired is None:
                continue
            try:
                parsed = json.loads(repaired)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _repair_truncated_json_object(raw: str) -> Optional[str]:
    """Best-effort close of truncated JSON objects from long visual grades.

    Only runs when the body already closed at least one nested structure, so a
    half-written top-level object like ``{"ledger_version":"x"`` is not
    falsely accepted as complete.
    """

    text = str(raw or "").strip()
    if not text.startswith("{"):
        return None
    # Only attempt when the payload looks truncated mid-object.
    if text.endswith("}"):
        return None
    # Require some already-closed structure so repair is not inventing a
    # finished document from a short prefix.
    if "}" not in text and "]" not in text:
        return None
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces <= 0 and open_brackets <= 0:
        return None
    # Drop a trailing incomplete key/value fragment after the last comma.
    last_comma = text.rfind(",")
    last_brace = max(text.rfind("}"), text.rfind("]"))
    if last_comma > last_brace:
        text = text[:last_comma]
    text = text.rstrip().rstrip(",")
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces < 0 or open_brackets < 0:
        return None
    return text + ("]" * open_brackets) + ("}" * open_braces)


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
        # immutable submission remains the ownership boundary. Reprocessing an
        # unchanged copy rematerializes this same ledger instead of purchasing
        # a new stochastic interpretation.
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


def _confidence(value: Any) -> float:
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
        processing_path=str(
            result.get("processing_path") or "full_document_visual"
        ),
        run_id=str(run.get("run_id") or "") or None,
        materialization_id=(
            str(result.get("materialization_id") or "") or None
        ),
        errors=[str(value) for value in (result.get("errors") or [])],
        document_review_required=bool(
            result.get("document_review_required")
        ),
        review_state=str(result.get("review_state") or "ready"),
        review_reasons=[
            str(value) for value in (result.get("review_reasons") or [])
        ],
    )
