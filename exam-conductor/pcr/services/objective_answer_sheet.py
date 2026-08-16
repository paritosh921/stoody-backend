"""Fast, deterministic PCR grading for objective answer sheets.

This module is deliberately separate from ``full_document_grading``.  It is
applicable only when every immutable catalog question is a multiple-choice
objective question.  The vision model transcribes answer states and evidence;
it never receives the answer key and never awards marks.  The server applies
the same deterministic scoring contract used by Online Test Series.

Subjective, integer-answer, and mixed papers are not handled here.  They fall
through to the existing full-document visual evidence-graph service unchanged.
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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from services.answer_mapping_contract import normalize_answer_label
from services.objective_scoring_service import (
    ObjectiveScoringContractError,
    is_integer_question,
    score_objective_response,
)

from ..storage.evaluation_repo import EvaluationRepository
from ..storage.response_repo import DetectedResponseRepository
from .ocr_service import AssetIntegrityError, _resolve_image_base64

logger = logging.getLogger(__name__)

OBJECTIVE_PROCESSING_PATH = "objective_answer_sheet"
OBJECTIVE_EXTRACTION_VERSION = "pcr-objective-answer-sheet-v1"
# The exam contract above remains immutable for finalized papers.  This
# implementation revision belongs to the extraction run fingerprint so a
# reprocess can safely replace an older whole-page vision read without
# rewriting the exam's scoring contract.
OBJECTIVE_EXTRACTOR_REVISION = "pcr-objective-hybrid-omr-v2"
OBJECTIVE_OMR_READER_VERSION = "deterministic-omr-grid-v1"

_RUNS_COLLECTION = "evalpen_objective_grading_runs"
_PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
_CALLER_ID = "pcr_objective_extraction"
_MAX_PAGE_COUNT = 40
_MAX_REQUEST_PAYLOAD_BYTES = 45 * 1024 * 1024
_SELECTED_MIN_CONFIDENCE = 0.70
_BLANK_MIN_CONFIDENCE = 0.90
_GENERATION_LEASE_MINUTES = 10


class ObjectiveAnswerSheetGateProtocol(Protocol):
    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        **kwargs: Any,
    ) -> Any: ...


class ObjectiveAnswerSheetError(RuntimeError):
    """Raised when an objective sheet cannot be processed safely."""


class ObjectiveRunIdentityError(ObjectiveAnswerSheetError):
    """Raised when a saved Objective run belongs to another generation."""

    retryable = False


@dataclass
class ObjectiveAnswerSheetResult:
    handled: bool
    submission_id: str
    processing_path: str = OBJECTIVE_PROCESSING_PATH
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


@dataclass(frozen=True)
class _PageAsset:
    page_number: int
    image_bytes: bytes
    media_type: str
    asset_hash: str


@dataclass
class _ValidatedAnswer:
    question: Dict[str, Any]
    question_number: int
    state: str
    selected_answer: str
    confidence: float
    source_pages: List[Dict[str, Any]]
    review_reason: str = ""


@dataclass(frozen=True)
class _OmrTrack:
    """One detected vertical option track in normalized image pixels."""

    x_reference: float
    slope: float
    support: int
    source_index: int


@dataclass(frozen=True)
class _OmrPoint:
    """The detected printed bubble nearest one option track."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class _OmrRow:
    """A complete, regularly spaced OMR row."""

    y: float
    points: tuple[_OmrPoint, ...]


@dataclass(frozen=True)
class _OmrGroup:
    """A left-to-right group of answer-option tracks and its rows."""

    tracks: tuple[_OmrTrack, ...]
    rows: tuple[_OmrRow, ...]
    score: float


class ObjectiveAnswerSheetGradingService:
    """Extract and deterministically score one pure-MCQ PCR answer copy."""

    def __init__(
        self,
        tenant_db: Any,
        gate: ObjectiveAnswerSheetGateProtocol,
        *,
        model_id: Optional[str] = None,
        response_repo: Optional[DetectedResponseRepository] = None,
        evaluation_repo: Optional[EvaluationRepository] = None,
    ) -> None:
        self._db = tenant_db
        self._gate = gate
        self._model_id = (
            model_id
            or os.getenv("PCR_OBJECTIVE_EXTRACTION_MODEL", "").strip()
            or os.getenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "gpt-5.1").strip()
        )
        self._responses = response_repo or DetectedResponseRepository(tenant_db)
        self._evaluations = evaluation_repo or EvaluationRepository(tenant_db)

    async def grade_submission(
        self,
        submission_id: str,
    ) -> ObjectiveAnswerSheetResult:
        submission = await self._db["evalpen_submissions"].find_one(
            {"submission_id": submission_id}
        )
        if submission is None:
            raise ObjectiveAnswerSheetError("Canonical submission was not found")

        source = str(submission.get("source") or "camera").strip().lower()
        if source not in {"camera", "pdf", "scan"}:
            return self._declined(
                submission_id,
                f"Submission source {source or 'unknown'} is not visual",
            )

        exam_id = str(submission.get("exam_id") or "")
        student_id = str(submission.get("student_id") or "")
        exam = await self._db["exampen_exams"].find_one({"exam_id": exam_id})
        if not exam or str(exam.get("exam_type") or "") != "pcr":
            return self._declined(
                submission_id,
                "Submission is not attached to a PCR exam",
            )

        questions = await self._db["evalpen_questions"].find(
            {"exam_id": exam_id}
        ).sort("question_number", 1).to_list(length=2000)
        questions = [question for question in questions if question.get("question_id")]
        if not questions:
            raise ObjectiveAnswerSheetError("Immutable PCR question catalog is empty")

        if any(not _is_objective_question(question) for question in questions):
            return self._declined(
                submission_id,
                "Paper is subjective or mixed and belongs to full-document grading",
            )
        if any(is_integer_question(question) for question in questions):
            return self._declined(
                submission_id,
                "Integer-answer objective papers remain on full-document grading",
            )

        catalog_errors = _validate_choice_catalog(questions)
        if catalog_errors:
            raise ObjectiveAnswerSheetError(
                "Immutable objective question catalog is invalid: "
                + "; ".join(catalog_errors[:10])
            )

        answer_pages = await self._db["evalpen_answer_pages"].find(
            {"submission_id": submission_id}
        ).sort("page_number", 1).to_list(length=_MAX_PAGE_COUNT + 1)
        if not answer_pages:
            raise ObjectiveAnswerSheetError("Canonical student answer pages are missing")
        if len(answer_pages) > _MAX_PAGE_COUNT:
            raise ObjectiveAnswerSheetError(
                f"Student copy has {len(answer_pages)} pages; maximum is "
                f"{_MAX_PAGE_COUNT}"
            )

        assets, payload_bytes = await _load_page_assets(answer_pages)
        if payload_bytes > _MAX_REQUEST_PAYLOAD_BYTES:
            raise ObjectiveAnswerSheetError(
                "Optimized student answer pages exceed the objective extraction limit"
            )

        contract = await self._objective_contract(exam)
        model_id = str(contract.get("model_id") or self._model_id).strip()
        reasoning_effort = str(
            contract.get("reasoning_effort") or "low"
        ).strip().lower()
        if reasoning_effort not in {"none", "minimal", "low"}:
            raise ObjectiveAnswerSheetError(
                "Objective extraction contract has an unsupported reasoning effort"
            )

        generation_revision = await _materialization_revision(
            self._db,
            submission_id,
        )
        input_fingerprint = _input_fingerprint(
            submission_id=submission_id,
            exam_id=exam_id,
            questions=questions,
            assets=assets,
            model_id=model_id,
        )
        generation_fingerprint = _generation_fingerprint(
            submission_id=submission_id,
            input_fingerprint=input_fingerprint,
            generation_revision=generation_revision,
        )
        run_id = f"OBJGR-{generation_fingerprint[:24]}"
        materialization_id = f"{run_id}:r{generation_revision}"

        await self._db[_RUNS_COLLECTION].create_index(
            "run_id",
            unique=True,
            name="uniq_objective_grading_run",
        )
        run = await self._db[_RUNS_COLLECTION].find_one({"run_id": run_id})
        if run:
            _assert_run_identity(
                run,
                submission_id=submission_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                generation_revision=generation_revision,
            )
        if run and run.get("status") == "completed":
            active_count = await self._db[
                "evalpen_detected_responses"
            ].count_documents(
                {
                    "submission_id": submission_id,
                    "mapping_version_id": materialization_id,
                    "superseded_at": {"$exists": False},
                }
            )
            if active_count == len(questions):
                return _result_from_run(run, submission_id)

        if not run or run.get("status") not in {
            "validated",
            "materializing",
            "completed",
        }:
            run, lease_token = await self._claim_or_wait_for_run(
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                submission_id=submission_id,
                student_id=student_id,
                exam_id=exam_id,
                generation_revision=generation_revision,
                model_id=model_id,
                page_count=len(assets),
            )
        else:
            lease_token = None

        if run and run.get("status") in {"validated", "materializing", "completed"}:
            payload = run.get("validated_payload")
            if not isinstance(payload, dict):
                raise ObjectiveAnswerSheetError(
                    "Saved objective extraction ledger is invalid"
                )
            usage = dict(run.get("token_usage") or {})
            raw_llm = str(run.get("raw_llm_response") or "")
        else:
            if not lease_token:
                raise ObjectiveAnswerSheetError(
                    "Objective extraction run has no generation ownership"
                )
            try:
                payload = await asyncio.to_thread(
                    _extract_omr_grid_payload,
                    questions,
                    assets,
                )
                if payload is not None:
                    raw_llm = json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    usage = {
                        "model": OBJECTIVE_OMR_READER_VERSION,
                        "caller": _CALLER_ID,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "extraction_method": "local_omr_grid",
                    }
                else:
                    request_input = _responses_input(
                        questions=questions,
                        assets=assets,
                    )
                    gate_response = await self._gate.call(
                        model_id=model_id,
                        prompt="",
                        caller_id=_CALLER_ID,
                        responses_input=request_input,
                        json_schema=_objective_extraction_schema(),
                        reasoning_effort=reasoning_effort,
                        temperature=0.0,
                        max_output_tokens=min(
                            18_000,
                            max(5_000, 180 * len(questions)),
                        ),
                        metadata={
                            "pcr_stage": "objective_answer_extraction",
                            "prompt_version": OBJECTIVE_EXTRACTION_VERSION,
                            "extractor_revision": OBJECTIVE_EXTRACTOR_REVISION,
                            "submission_id": submission_id,
                            "exam_id": exam_id,
                            "question_count": len(questions),
                            "page_count": len(assets),
                            "run_id": run_id,
                        },
                    )
                    raw_llm = str(getattr(gate_response, "content", "") or "")
                    payload = _parse_json_object(raw_llm)
                    if payload is None:
                        raise ObjectiveAnswerSheetError(
                            "Objective extractor returned invalid structured output"
                        )
                    usage = _usage_dict(gate_response, fallback_model=model_id)
                    usage["extraction_method"] = "vision_fallback"
            except Exception as exc:
                await self._db[_RUNS_COLLECTION].update_one(
                    {"run_id": run_id, "generation_lease_token": lease_token},
                    {
                        "$set": {
                            "status": "failed",
                            "generation_error": str(exc)[:500],
                            "updated_at": _now(),
                        },
                        "$unset": {
                            "generation_lease_token": "",
                            "generation_lease_expires_at": "",
                        },
                    },
                )
                if isinstance(exc, ObjectiveAnswerSheetError):
                    raise
                raise ObjectiveAnswerSheetError(
                    f"Objective answer extraction failed: {str(exc)[:400]}"
                ) from exc

            saved = await self._db[_RUNS_COLLECTION].update_one(
                {"run_id": run_id, "generation_lease_token": lease_token},
                {
                    "$set": {
                        "status": "validated",
                        "validated_payload": payload,
                        "raw_llm_response": raw_llm,
                        "token_usage": usage,
                        "model_used": usage.get("model") or model_id,
                        "extraction_method": usage.get("extraction_method"),
                        "updated_at": _now(),
                    },
                    "$unset": {
                        "generation_lease_token": "",
                        "generation_lease_expires_at": "",
                        "generation_error": "",
                    },
                },
            )
            if saved.matched_count != 1:
                raise ObjectiveAnswerSheetError(
                    "Objective extraction ownership expired before its ledger was saved"
                )

        answers, document_warnings = _validate_payload(
            payload,
            questions=questions,
            page_count=len(assets),
        )
        await self._db[_RUNS_COLLECTION].update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "materializing",
                    "validation_warnings": document_warnings,
                    "updated_at": _now(),
                }
            },
        )
        result = await self._materialize(
            run_id=run_id,
            materialization_id=materialization_id,
            submission=submission,
            answers=answers,
            payload=payload,
            usage=usage,
            input_fingerprint=input_fingerprint,
            page_count=len(assets),
            document_warnings=document_warnings,
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
                        "review_state": result.review_state,
                        "review_reasons": result.review_reasons,
                    },
                    "completed_at": _now(),
                    "updated_at": _now(),
                }
            },
        )
        return result

    async def _objective_contract(self, exam: Dict[str, Any]) -> Dict[str, Any]:
        exam_id = str(exam.get("exam_id") or "")
        existing = exam.get("pcr_objective_extraction_contract")
        if not isinstance(existing, dict):
            contract = {
                "version": OBJECTIVE_EXTRACTION_VERSION,
                "model_id": self._model_id,
                "reasoning_effort": "low",
                "temperature": 0.0,
                "scoring": "server_deterministic_v1",
                "created_at": _now(),
            }
            await self._db["exampen_exams"].update_one(
                {
                    "exam_id": exam_id,
                    "pcr_objective_extraction_contract": {"$exists": False},
                },
                {"$set": {"pcr_objective_extraction_contract": contract}},
            )
            refreshed = await self._db["exampen_exams"].find_one(
                {"exam_id": exam_id},
                {"pcr_objective_extraction_contract": 1},
            )
            existing = (refreshed or {}).get("pcr_objective_extraction_contract")
        contract = dict(existing or {})
        if contract.get("version") != OBJECTIVE_EXTRACTION_VERSION:
            raise ObjectiveAnswerSheetError(
                "Exam is locked to an unsupported Objective PCR extraction contract"
            )
        if contract.get("scoring") != "server_deterministic_v1":
            raise ObjectiveAnswerSheetError(
                "Exam Objective PCR scoring contract is invalid"
            )
        return contract

    async def _claim_or_wait_for_run(
        self,
        *,
        run_id: str,
        input_fingerprint: str,
        generation_fingerprint: str,
        submission_id: str,
        student_id: str,
        exam_id: str,
        generation_revision: int,
        model_id: str,
        page_count: int,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        collection = self._db[_RUNS_COLLECTION]
        lease_token = uuid.uuid4().hex
        now = _now()
        lease_expires_at = now + timedelta(minutes=_GENERATION_LEASE_MINUTES)
        base_doc = {
            "run_id": run_id,
            "input_fingerprint": input_fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "submission_id": submission_id,
            "student_id": student_id,
            "exam_id": exam_id,
            "grading_revision": generation_revision,
            "generation_revision": generation_revision,
            "prompt_version": OBJECTIVE_EXTRACTION_VERSION,
            "extractor_revision": OBJECTIVE_EXTRACTOR_REVISION,
            "requested_model_id": model_id,
            "page_count": page_count,
            "status": "generating",
            "generation_lease_token": lease_token,
            "generation_lease_expires_at": lease_expires_at,
            "created_at": now,
            "updated_at": now,
        }
        existing = await collection.find_one({"run_id": run_id})
        if existing is None:
            try:
                await collection.insert_one(base_doc)
                return base_doc, lease_token
            except DuplicateKeyError:
                existing = await collection.find_one({"run_id": run_id})

        if existing:
            _assert_run_identity(
                existing,
                submission_id=submission_id,
                input_fingerprint=input_fingerprint,
                generation_fingerprint=generation_fingerprint,
                generation_revision=generation_revision,
            )

        claimed = await collection.find_one_and_update(
            {
                "run_id": run_id,
                "$or": [
                    {"status": "failed"},
                    {"generation_lease_expires_at": {"$lte": now}},
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
                    "updated_at": now,
                },
                "$unset": {"generation_error": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
        if claimed is not None:
            return claimed, lease_token

        wait_seconds = _bounded_float(
            os.getenv("PCR_OBJECTIVE_SINGLEFLIGHT_WAIT_SECONDS", "120"),
            default=120.0,
            minimum=5.0,
            maximum=180.0,
        )
        deadline = asyncio.get_running_loop().time() + wait_seconds
        while asyncio.get_running_loop().time() < deadline:
            current = await collection.find_one({"run_id": run_id})
            if current:
                _assert_run_identity(
                    current,
                    submission_id=submission_id,
                    input_fingerprint=input_fingerprint,
                    generation_fingerprint=generation_fingerprint,
                    generation_revision=generation_revision,
                )
            if current and current.get("status") in {
                "validated",
                "materializing",
                "completed",
            }:
                return current, None
            if current and current.get("status") == "failed":
                return await self._claim_or_wait_for_run(
                    run_id=run_id,
                    input_fingerprint=input_fingerprint,
                    generation_fingerprint=generation_fingerprint,
                    submission_id=submission_id,
                    student_id=student_id,
                    exam_id=exam_id,
                    generation_revision=generation_revision,
                    model_id=model_id,
                    page_count=page_count,
                )
            await asyncio.sleep(0.5)
        raise ObjectiveAnswerSheetError(
            "This Objective PCR submission is already being extracted; retry after "
            "the current run finishes"
        )

    async def _materialize(
        self,
        *,
        run_id: str,
        materialization_id: str,
        submission: Dict[str, Any],
        answers: List[_ValidatedAnswer],
        payload: Dict[str, Any],
        usage: Dict[str, Any],
        input_fingerprint: str,
        page_count: int,
        document_warnings: List[str],
    ) -> ObjectiveAnswerSheetResult:
        submission_id = str(submission.get("submission_id") or "")
        exam_id = str(submission.get("exam_id") or "")
        student_id = str(submission.get("student_id") or "")
        model_used = str(usage.get("model") or self._model_id)
        raw_by_number = {
            _positive_int(item.get("question_number")): item
            for item in (payload.get("answers") or [])
            if isinstance(item, dict) and _positive_int(item.get("question_number"))
        }
        response_docs: List[Dict[str, Any]] = []
        evaluation_docs: List[Dict[str, Any]] = []
        review_reasons: List[str] = list(document_warnings)

        for answer in answers:
            question_id = str(answer.question.get("question_id") or "")
            response_id = _stable_id(
                "RESP-OBJ",
                submission_id,
                materialization_id,
                question_id,
            )
            unresolved = answer.state == "unresolved"
            not_attempted = answer.state == "blank"
            objective_result: Optional[Dict[str, Any]] = None
            if not unresolved:
                try:
                    objective_result = score_objective_response(
                        answer.question,
                        "" if not_attempted else answer.selected_answer,
                    )
                except ObjectiveScoringContractError as exc:
                    unresolved = True
                    answer.review_reason = str(exc)

            flags: List[Dict[str, Any]] = []
            if unresolved:
                reason = answer.review_reason or "Objective answer is ambiguous"
                flags.append(_blocking_flag(response_id, reason))
                review_reasons.append(f"Q{answer.question_number}: {reason}")

            response_doc = {
                "response_id": response_id,
                "submission_id": submission_id,
                "question_id": question_id,
                "question_number": answer.question_number,
                "sub_part": None,
                "question_assignment": {
                    "method": OBJECTIVE_PROCESSING_PATH,
                    "confidence": answer.confidence,
                    "prompt_version": OBJECTIVE_EXTRACTION_VERSION,
                    "model_used": model_used,
                    "grading_run_id": run_id,
                    "materialization_id": materialization_id,
                    "manual_review_required": unresolved,
                    "reason": answer.review_reason or None,
                    "absence_proof": (
                        {
                            "verified": True,
                            "method": "objective_full_sheet_coverage",
                            "confidence": answer.confidence,
                            "grading_run_id": run_id,
                        }
                        if not_attempted
                        else None
                    ),
                },
                "exam_id": exam_id,
                "student_id": student_id,
                "detected_text": (
                    "" if unresolved or not_attempted else answer.selected_answer
                ),
                "source_pages": answer.source_pages,
                "visual_evidence": {
                    "extraction_version": OBJECTIVE_EXTRACTION_VERSION,
                    "answer_state": answer.state,
                    "raw_answer": raw_by_number.get(answer.question_number, {}),
                },
                "evidence_version": 1,
                "evidence_atom_ids": [
                    _stable_id(
                        "region",
                        submission_id,
                        str(region.get("page_number") or ""),
                        str(region.get("x_start") or ""),
                        str(region.get("y_start") or ""),
                        str(region.get("x_end") or ""),
                        str(region.get("y_end") or ""),
                    )
                    for region in answer.source_pages
                ],
                "content_type": "TEXT_ONLY",
                "text_coverage_ratio": 1.0 if answer.selected_answer else 0.0,
                "segmentation_confidence": answer.confidence,
                "ocr_confidence": answer.confidence,
                "flags": flags,
                "word_count": 1 if answer.selected_answer else 0,
                "is_continuation": len(answer.source_pages) > 1,
                "is_missing_response": not_attempted,
                "absence_proven": not_attempted,
                "manual_review_required": unresolved,
                "manual_review_reason": answer.review_reason or None,
                "answer_state": (
                    "unresolved"
                    if unresolved
                    else "not_attempted"
                    if not_attempted
                    else "detected"
                ),
                "grading_mode": "objective",
                "objective_result": objective_result,
                "eval_status": "blocked" if unresolved else "pending",
                "mapping_version_id": materialization_id,
                "_immutable": True,
                "created_at": _now(),
            }
            response_docs.append(response_doc)
            if unresolved or objective_result is None:
                continue

            max_marks = float(objective_result["points"])
            total_score = float(objective_result["points_earned"])
            selected = str(objective_result.get("selected_answer") or "")
            correct = str(objective_result.get("correct_answer") or "")
            evaluation_docs.append(
                {
                    "evaluation_id": _stable_id(
                        "EVAL-OBJ",
                        submission_id,
                        materialization_id,
                        question_id,
                    ),
                    "evaluation_input_version": 2,
                    "mapping_version_id": materialization_id,
                    "response_id": response_id,
                    "question_id": question_id,
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "prompt_version": OBJECTIVE_EXTRACTION_VERSION,
                    "visual_evidence": response_doc["visual_evidence"],
                    "eval_path": (
                        "objective_answer_sheet_not_attempted"
                        if not_attempted
                        else "objective_answer_sheet"
                    ),
                    "model_used": "deterministic-objective-scorer-v1",
                    "total_score": total_score,
                    "max_score": max_marks,
                    "scoreable_max": max_marks,
                    "marking_policy": dict(
                        answer.question.get("marking_policy") or {}
                    ),
                    "method_policy": {},
                    "method_analysis": {},
                    "manual_review_required": False,
                    "step_marks": [],
                    "criterion_marks": [],
                    "overall_feedback": _objective_feedback(
                        attempted=bool(objective_result.get("attempted")),
                        correct=bool(objective_result.get("is_correct")),
                        selected=selected,
                        correct_answer=correct,
                        total_score=total_score,
                    ),
                    "grading_mode": "objective",
                    "objective_result": objective_result,
                    "reference_solution": correct,
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
                        raw_by_number.get(answer.question_number, {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "eval_flags": [],
                    "audit_trail": [
                        {
                            "actor_id": "system",
                            "timestamp": _now(),
                            "action": "evaluation_created",
                            "before": None,
                            "after": {
                                "total_score": total_score,
                                "max_score": max_marks,
                                "eval_path": OBJECTIVE_PROCESSING_PATH,
                                "selected_answer": selected,
                            },
                            "reason": (
                                "Option extracted visually and scored against the "
                                "immutable key by deterministic server code"
                            ),
                        }
                    ],
                    "created_at": _now(),
                }
            )

        await self._responses.insert_responses_bulk(response_docs)
        for evaluation_doc in evaluation_docs:
            await self._evaluations.insert_evaluation(evaluation_doc)
        for response_doc in response_docs:
            await self._responses.update_eval_status(
                response_doc["response_id"],
                "blocked"
                if response_doc["answer_state"] == "unresolved"
                else "evaluated",
            )
        await self._responses.supersede_responses_for_submission(
            submission_id,
            keep_response_ids=[doc["response_id"] for doc in response_docs],
            reason=OBJECTIVE_PROCESSING_PATH,
        )

        blocked_count = sum(
            1 for response in response_docs if response["answer_state"] == "unresolved"
        )
        review_state = "blocked" if blocked_count else "ready"
        await self._db["evalpen_submissions"].update_one(
            {"submission_id": submission_id},
            {
                "$set": {
                    "segmentation_status": "complete",
                    "processing_path": OBJECTIVE_PROCESSING_PATH,
                    "objective_grading_run_id": run_id,
                    "objective_grading_materialization_id": materialization_id,
                    "grading_input_hash": input_fingerprint,
                    "review_state": review_state,
                    "document_review": {
                        "status": "pending_review" if blocked_count else "verified",
                        "required": bool(blocked_count),
                        "all_student_work_accounted": not bool(document_warnings),
                        "confidence": _document_confidence(payload),
                        "warnings": document_warnings,
                        "grading_run_id": run_id,
                        "prompt_version": OBJECTIVE_EXTRACTION_VERSION,
                        "updated_at": _now(),
                    },
                    "updated_at": _now(),
                }
            },
        )
        return ObjectiveAnswerSheetResult(
            handled=True,
            submission_id=submission_id,
            status="completed",
            page_count=page_count,
            response_count=len(response_docs),
            evaluated_count=len(evaluation_docs),
            blocked_count=blocked_count,
            warning_count=len(document_warnings),
            run_id=run_id,
            errors=[],
            document_review_required=bool(blocked_count),
            review_state=review_state,
            review_reasons=list(dict.fromkeys(review_reasons)),
        )

    @staticmethod
    def _declined(
        submission_id: str,
        reason: str,
    ) -> ObjectiveAnswerSheetResult:
        return ObjectiveAnswerSheetResult(
            handled=False,
            submission_id=submission_id,
            skipped_reason=reason,
        )


def _is_objective_question(question: Dict[str, Any]) -> bool:
    return str(
        question.get("grading_mode")
        or question.get("question_type")
        or ""
    ).strip().lower() in {"objective", "mcq"}


def _option_labels(question: Dict[str, Any]) -> List[str]:
    options = question.get("options") or question.get("enhanced_options") or []
    if not isinstance(options, list):
        return []
    labels: List[str] = []
    for index, option in enumerate(options):
        if isinstance(option, dict):
            label = str(option.get("label") or chr(ord("A") + index))
        else:
            label = chr(ord("A") + index)
        normalized = normalize_answer_label(label)
        if normalized and normalized not in labels:
            labels.append(normalized)
    return labels


def _validate_choice_catalog(questions: Sequence[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen_numbers: set[int] = set()
    seen_ids: set[str] = set()
    for position, question in enumerate(questions, start=1):
        question_id = str(question.get("question_id") or "")
        number = _positive_int(question.get("question_number")) or position
        labels = _option_labels(question)
        correct = normalize_answer_label(
            str(
                question.get("correct_answer")
                or question.get("correctAnswer")
                or ""
            )
        )
        if not question_id:
            errors.append(f"Q{number} has no question_id")
        elif question_id in seen_ids:
            errors.append(f"duplicate question_id {question_id}")
        seen_ids.add(question_id)
        if number in seen_numbers:
            errors.append(f"duplicate question number Q{number}")
        seen_numbers.add(number)
        if len(labels) < 2:
            errors.append(f"Q{number} has fewer than two answer options")
        if not correct or correct not in labels:
            errors.append(f"Q{number} has an invalid correct-answer label")
        try:
            score_objective_response(question, "")
        except ObjectiveScoringContractError as exc:
            errors.append(f"Q{number}: {exc}")
    return errors


async def _load_page_assets(
    answer_pages: Sequence[Dict[str, Any]],
) -> tuple[List[_PageAsset], int]:
    assets: List[_PageAsset] = []
    total_bytes = 0
    for page in answer_pages:
        page_number = _positive_int(page.get("page_number"))
        raw_ref = page.get("raw_image_ref")
        if not page_number or not isinstance(raw_ref, str) or not raw_ref.strip():
            raise ObjectiveAnswerSheetError(
                f"Canonical student page {page_number or '?'} has no image asset"
            )
        try:
            image_b64 = await _resolve_image_base64(
                raw_ref,
                expected_sha256=page.get("asset_sha256"),
            )
        except AssetIntegrityError as exc:
            raise ObjectiveAnswerSheetError(str(exc)) from exc
        if not image_b64:
            raise ObjectiveAnswerSheetError(
                f"Canonical student page {page_number} could not be loaded"
            )
        try:
            original = base64.b64decode(image_b64, validate=True)
        except Exception as exc:
            raise ObjectiveAnswerSheetError(
                f"Canonical student page {page_number} is not a valid image"
            ) from exc
        optimized, media_type = await asyncio.to_thread(_optimize_image, original)
        total_bytes += len(optimized)
        assets.append(
            _PageAsset(
                page_number=page_number,
                image_bytes=optimized,
                media_type=media_type,
                asset_hash=hashlib.sha256(original).hexdigest(),
            )
        )
    return assets, total_bytes


def _optimize_image(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            else:
                image = image.copy()
            image.thumbnail((2600, 2600))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
            optimized = output.getvalue()
            if optimized:
                return optimized, "image/jpeg"
    except Exception:
        logger.warning("Could not optimize an objective answer page; using original")
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return image_bytes, media_type


def _extract_omr_grid_payload(
    questions: Sequence[Dict[str, Any]],
    assets: Sequence[_PageAsset],
) -> Optional[Dict[str, Any]]:
    """Read a regular OMR grid without consulting the answer key.

    This is deliberately a format detector, not a CIEL/template detector.  It
    discovers printed bubble tracks, row spacing, perspective, and fills from
    the submitted pixels.  If the page does not prove that structure, ``None``
    sends it to the existing vision fallback instead of guessing.
    """

    if not questions or not assets:
        return None

    label_sets = [_option_labels(question) for question in questions]
    first_labels = label_sets[0]
    if (
        len(first_labels) < 2
        or len(first_labels) > 6
        or any(labels != first_labels for labels in label_sets[1:])
    ):
        return None

    detected_pages: List[
        tuple[_PageAsset, Any, List[_OmrGroup], int, int]
    ] = []
    total_capacity = 0
    for asset in assets:
        detected = _detect_omr_page(
            asset.image_bytes,
            option_count=len(first_labels),
        )
        if detected is None:
            continue
        gray, groups = detected
        height, width = gray.shape[:2]
        detected_pages.append((asset, gray, groups, width, height))
        total_capacity += sum(len(group.rows) for group in groups)

    # A partial grid must not silently turn the remaining catalog into blanks.
    if not detected_pages or total_capacity < len(questions):
        return None

    cells: List[tuple[_PageAsset, Any, _OmrGroup, _OmrRow, int, int]] = []
    for asset, gray, groups, width, height in detected_pages:
        for group in groups:
            for row in group.rows:
                cells.append((asset, gray, group, row, width, height))
    if len(cells) < len(questions):
        return None

    answers: List[Dict[str, Any]] = []
    selected_confidences: List[float] = []
    for position, question in enumerate(questions):
        asset, gray, group, row, width, height = cells[position]
        state, option_indexes, confidence, reason = _read_omr_row(
            gray,
            row,
        )
        selected_options = [
            first_labels[index]
            for index in option_indexes
            if 0 <= index < len(first_labels)
        ]
        question_number = (
            _positive_int(question.get("question_number")) or position + 1
        )
        region = _omr_evidence_region(
            asset=asset,
            question_number=question_number,
            group=group,
            row=row,
            width=width,
            height=height,
            confidence=confidence,
            state=state,
            selected_options=selected_options,
        )
        answers.append(
            {
                "question_number": question_number,
                "state": state,
                "selected_options": selected_options,
                "confidence": confidence,
                "evidence_regions": [region],
                "reason": reason,
            }
        )
        if state in {"selected", "blank"}:
            selected_confidences.append(confidence)

    unresolved_count = sum(
        1
        for answer in answers
        if answer["state"] in {"multiple", "ambiguous"}
    )
    document_confidence = (
        min(selected_confidences)
        if selected_confidences
        else 0.0
    )
    if unresolved_count:
        document_confidence = min(document_confidence, 0.88)
    warnings = (
        [f"{unresolved_count} OMR rows need teacher review"]
        if unresolved_count
        else []
    )
    return {
        "version": OBJECTIVE_EXTRACTION_VERSION,
        "document": {
            "format": "omr_grid",
            "all_answer_areas_checked": True,
            "confidence": round(document_confidence, 4),
            "warnings": warnings,
        },
        "answers": answers,
    }


def _detect_omr_page(
    image_bytes: bytes,
    *,
    option_count: int,
) -> Optional[tuple[Any, List[_OmrGroup]]]:
    """Detect option-track groups and complete row lattices on one page."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning(
            "OpenCV is unavailable; Objective PCR will use the vision fallback"
        )
        return None

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest < 900 or longest > 1900:
        scale = 1600.0 / float(longest)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=(
                cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            ),
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circle_rows = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(5, int(width * 0.004)),
        param1=100,
        param2=10,
        minRadius=max(2, int(width * 0.0025)),
        maxRadius=max(7, int(width * 0.009)),
    )
    if circle_rows is None:
        return None
    circles = [
        _OmrPoint(float(x), float(y), float(radius))
        for x, y, radius in circle_rows[0]
        if float(y) >= height * 0.25
    ]
    if len(circles) < option_count * 8:
        return None

    edges = cv2.Canny(blurred, 50, 150)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(50, int(height * 0.05)),
        minLineLength=max(120, int(height * 0.12)),
        maxLineGap=max(20, int(height * 0.04)),
    )
    if raw_lines is None:
        return None

    y_reference = height * 0.70
    track_tolerance = max(6.0, width * 0.006)
    track_samples: List[tuple[float, float, int]] = []
    for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if (
            abs(dy) < max(40.0, height * 0.05)
            or abs(dx / dy) > 0.12
            or max(y1, y2) < height * 0.45
        ):
            continue
        slope = dx / dy
        x_reference = float(x1) + slope * (y_reference - float(y1))
        support = sum(
            1
            for point in circles
            if abs(
                point.x
                - (
                    x_reference
                    + slope * (point.y - y_reference)
                )
            )
            < track_tolerance
        )
        if support >= 8:
            track_samples.append((x_reference, slope, support))
    if len(track_samples) < option_count:
        return None

    track_samples.sort(key=lambda item: item[0])
    clusters: List[List[tuple[float, float, int]]] = []
    cluster_tolerance = max(8.0, width * 0.01)
    for sample in track_samples:
        if (
            clusters
            and abs(
                sample[0]
                - float(np.median([item[0] for item in clusters[-1]]))
            )
            < cluster_tolerance
        ):
            clusters[-1].append(sample)
        else:
            clusters.append([sample])

    tracks: List[_OmrTrack] = []
    for source_index, cluster in enumerate(clusters):
        best = max(cluster, key=lambda item: item[2])
        tracks.append(
            _OmrTrack(
                x_reference=float(
                    np.median([item[0] for item in cluster])
                ),
                slope=float(best[1]),
                support=int(best[2]),
                source_index=source_index,
            )
        )
    if len(tracks) < option_count:
        return None

    candidates: List[_OmrGroup] = []
    for start in range(0, len(tracks) - option_count + 1):
        group_tracks = tuple(tracks[start : start + option_count])
        gaps = np.diff([track.x_reference for track in group_tracks])
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 0:
            continue
        gap_cv = float(np.std(gaps) / mean_gap)
        if (
            mean_gap < width * 0.025
            or mean_gap > width * 0.09
            or gap_cv > 0.18
        ):
            continue
        rows = _detect_regular_omr_rows(
            circles,
            group_tracks,
            y_reference=y_reference,
            width=width,
            height=height,
        )
        if len(rows) < 8:
            continue
        supports = [track.support for track in group_tracks]
        score = (
            len(rows) * 2.0
            + float(np.mean(supports))
            + 0.5 * min(supports)
            - 100.0 * gap_cv
        )
        candidates.append(
            _OmrGroup(
                tracks=group_tracks,
                rows=tuple(rows),
                score=score,
            )
        )
    if not candidates:
        return None

    max_rows = max(len(candidate.rows) for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if len(candidate.rows) >= max(8, int(math.floor(max_rows * 0.70)))
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    selected: List[_OmrGroup] = []
    occupied_tracks: set[int] = set()
    for candidate in candidates:
        source_indexes = {
            track.source_index for track in candidate.tracks
        }
        if source_indexes & occupied_tracks:
            continue
        selected.append(candidate)
        occupied_tracks.update(source_indexes)
    if not selected:
        return None
    selected.sort(key=lambda group: group.tracks[0].x_reference)
    return gray, selected


def _detect_regular_omr_rows(
    circles: Sequence[_OmrPoint],
    tracks: Sequence[_OmrTrack],
    *,
    y_reference: float,
    width: int,
    height: int,
) -> List[_OmrRow]:
    """Find the longest complete, evenly spaced row run for a track group."""

    import numpy as np

    tolerance = max(6.0, width * 0.0067)
    hits: List[tuple[_OmrPoint, int, float]] = []
    for point in circles:
        distances = [
            abs(
                point.x
                - (
                    track.x_reference
                    + track.slope * (point.y - y_reference)
                )
            )
            for track in tracks
        ]
        option_index = int(np.argmin(distances))
        if distances[option_index] < tolerance:
            hits.append((point, option_index, distances[option_index]))
    if not hits:
        return []

    hits.sort(key=lambda item: item[0].y)
    row_clusters: List[List[tuple[_OmrPoint, int, float]]] = []
    row_tolerance = max(5.0, height * 0.0044)
    for hit in hits:
        if (
            row_clusters
            and abs(
                hit[0].y
                - float(
                    np.median(
                        [item[0].y for item in row_clusters[-1]]
                    )
                )
            )
            <= row_tolerance
        ):
            row_clusters[-1].append(hit)
        else:
            row_clusters.append([hit])

    complete_rows: List[_OmrRow] = []
    for cluster in row_clusters:
        by_option: Dict[int, tuple[_OmrPoint, float]] = {}
        for point, option_index, distance in cluster:
            current = by_option.get(option_index)
            if current is None or distance < current[1]:
                by_option[option_index] = (point, distance)
        # Header letters and random page graphics often imitate three circles.
        # Requiring the complete option row prevents them entering the lattice.
        if len(by_option) != len(tracks):
            continue
        points = tuple(
            by_option[index][0] for index in range(len(tracks))
        )
        complete_rows.append(
            _OmrRow(
                y=float(np.median([point.y for point in points])),
                points=points,
            )
        )
    if len(complete_rows) < 8:
        return []

    gaps = [
        complete_rows[index + 1].y - complete_rows[index].y
        for index in range(len(complete_rows) - 1)
    ]
    plausible_gaps = [
        gap
        for gap in gaps
        if height * 0.006 <= gap <= height * 0.055
    ]
    if not plausible_gaps:
        return []
    # Two-pixel buckets make the dominant printed row pitch resistant to
    # camera perspective and isolated header/footer circles.
    buckets = Counter(int(round(gap / 2.0) * 2) for gap in plausible_gaps)
    dominant_gap = float(buckets.most_common(1)[0][0])
    if dominant_gap <= 0:
        return []

    best: List[_OmrRow] = []
    for start in range(len(complete_rows)):
        sequence = [complete_rows[start]]
        for row in complete_rows[start + 1 :]:
            gap = row.y - sequence[-1].y
            if 0.72 * dominant_gap <= gap <= 1.28 * dominant_gap:
                sequence.append(row)
                continue
            if gap > 1.28 * dominant_gap:
                break
        if len(sequence) > len(best):
            best = sequence
    return best


def _read_omr_row(
    gray: Any,
    row: _OmrRow,
) -> tuple[str, List[int], float, str]:
    """Classify one row using fill darkness and correction strokes."""

    import numpy as np

    height, width = gray.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    measurements: List[tuple[float, float]] = []
    for point in row.points:
        distance = np.sqrt(
            (xx - point.x) ** 2 + (yy - point.y) ** 2
        )
        inner = gray[
            distance <= max(2.5, point.radius * 0.70)
        ]
        annulus = gray[
            (distance >= max(5.0, point.radius * 1.30))
            & (distance <= max(10.0, point.radius * 2.60))
        ]
        if not inner.size or not annulus.size:
            return (
                "ambiguous",
                [],
                0.0,
                "The OMR bubble crop could not be measured safely",
            )
        inner_darkness = float(np.mean(255.0 - inner))
        correction_fraction = float(np.mean(annulus < 120))
        measurements.append((inner_darkness, correction_fraction))

    darkness = [value[0] for value in measurements]
    baseline_count = max(1, len(darkness) // 2)
    baseline = float(np.median(sorted(darkness)[:baseline_count]))
    candidates = [
        index
        for index, (inner_darkness, _) in enumerate(measurements)
        if inner_darkness >= baseline + 45.0
    ]
    clean_candidates = [
        index
        for index in candidates
        if measurements[index][1] < 0.035
    ]
    ordered = sorted(darkness, reverse=True)
    top_margin = (
        ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    )

    if not candidates:
        if top_margin < 35.0:
            return "blank", [], 0.96, "No option is filled"
        return (
            "ambiguous",
            [],
            0.62,
            "A possible mark is too faint to read safely",
        )
    if len(candidates) == 1:
        chosen = candidates[0]
        if chosen not in clean_candidates:
            return (
                "ambiguous",
                [chosen],
                0.65,
                "The only marked option appears crossed out or erased",
            )
        confidence = min(0.99, 0.92 + max(0.0, top_margin - 35.0) / 500.0)
        return "selected", [chosen], round(confidence, 4), "One option is filled"
    if len(clean_candidates) == 1:
        # A dark candidate with a surrounding strike plus one clean candidate
        # is the standard visual form of a corrected OMR response.
        chosen = clean_candidates[0]
        return (
            "selected",
            [chosen],
            0.93,
            "One clean option remains after a visible correction",
        )
    if len(clean_candidates) > 1:
        return (
            "multiple",
            clean_candidates,
            0.90,
            "More than one option remains filled",
        )
    return (
        "ambiguous",
        candidates,
        0.68,
        "Several marked options appear crossed out or erased",
    )


def _omr_evidence_region(
    *,
    asset: _PageAsset,
    question_number: int,
    group: _OmrGroup,
    row: _OmrRow,
    width: int,
    height: int,
    confidence: float,
    state: str,
    selected_options: Sequence[str],
) -> Dict[str, Any]:
    left = min(point.x for point in row.points)
    right = max(point.x for point in row.points)
    radius = max(point.radius for point in row.points)
    x_start = max(0.0, left - max(28.0, (right - left) * 0.28))
    x_end = min(float(width), right + max(15.0, radius * 3.0))
    y_start = max(0.0, row.y - max(12.0, radius * 2.5))
    y_end = min(float(height), row.y + max(12.0, radius * 2.5))
    evidence = (
        f"OMR row: {','.join(selected_options)}"
        if selected_options
        else f"OMR row: {state}"
    )
    return {
        "region_id": f"objective-q{question_number}-page-{asset.page_number}",
        "page_number": asset.page_number,
        "coordinate_space": "normalized_1000",
        "x_start": round(x_start * 1000.0 / width, 3),
        "y_start": round(y_start * 1000.0 / height, 3),
        "x_end": round(x_end * 1000.0 / width, 3),
        "y_end": round(y_end * 1000.0 / height, 3),
        "evidence": evidence,
        "confidence": confidence,
    }


def _responses_input(
    *,
    questions: Sequence[Dict[str, Any]],
    assets: Sequence[_PageAsset],
) -> List[Dict[str, Any]]:
    catalog = [
        {
            "question_number": (
                _positive_int(question.get("question_number")) or position
            ),
            "allowed_options": _option_labels(question),
        }
        for position, question in enumerate(questions, start=1)
    ]
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": _objective_system_instructions(),
        },
        {
            "type": "input_text",
            "text": (
                "IMMUTABLE QUESTION NUMBER AND ALLOWED-LABEL CATALOG. It intentionally "
                "contains no answer key. Return exactly one answer-state record for "
                "every catalog question:\n"
                + json.dumps(catalog, separators=(",", ":"))
            ),
        },
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
                        f"data:{asset.media_type};base64,"
                        + base64.b64encode(asset.image_bytes).decode("ascii")
                    ),
                    "detail": "high",
                },
            ]
        )
    return [{"role": "user", "content": content}]


def _objective_system_instructions() -> str:
    return (
        "You are a strict answer-sheet transcription engine, not a grader. Inspect "
        "every supplied page and extract only the student's selected option for each "
        "question. Never decide correctness and never award marks. The answer key is "
        "not supplied. Support both formats: (1) OMR/bubble grids, including "
        "multi-column numbering, perspective, rotation, faint fills, erasures, "
        "crosses, and wrong-fill corrections; (2) handwritten numbered lists such as "
        "'1 B, 2 C'. A printed empty circle is not selected. A visibly darkened, "
        "ticked, crossed, or otherwise intentionally chosen option may be selected "
        "only when it is unambiguous. If two choices remain selected, use 'multiple'. "
        "If the mark cannot be read reliably, use 'ambiguous'. Use 'blank' only after "
        "checking the answer area for that exact question. Return every catalog "
        "question exactly once. Evidence boxes use normalized 0..1000 coordinates "
        "relative to the full source page, with the tightest box that shows the "
        "question number and selected mark. Set all_answer_areas_checked=true only if "
        "all pages and all expected answer areas were actually inspected. Do not infer "
        "missing answers from nearby patterns."
    )


def _objective_extraction_schema() -> Dict[str, Any]:
    region = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_number": {"type": "integer", "minimum": 1},
            "x_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "x_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "evidence": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "page_number",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
            "evidence",
            "confidence",
        ],
    }
    answer = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "integer", "minimum": 1},
            "state": {
                "type": "string",
                "enum": ["selected", "blank", "multiple", "ambiguous"],
            },
            "selected_options": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_regions": {"type": "array", "items": region},
            "reason": {"type": "string"},
        },
        "required": [
            "question_number",
            "state",
            "selected_options",
            "confidence",
            "evidence_regions",
            "reason",
        ],
    }
    # LLMGate owns the Responses API text.format envelope (name + strict).
    # Callers must pass only the raw JSON schema; wrapping it here would place
    # another schema object inside text.format.schema and OpenAI rejects that
    # payload because the resulting root has no `type`.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {
                "type": "string",
                "enum": [OBJECTIVE_EXTRACTION_VERSION],
            },
            "document": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": [
                            "omr_grid",
                            "numbered_option_list",
                            "mixed",
                            "unknown",
                        ],
                    },
                    "all_answer_areas_checked": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "format",
                    "all_answer_areas_checked",
                    "confidence",
                    "warnings",
                ],
            },
            "answers": {"type": "array", "items": answer},
        },
        "required": ["version", "document", "answers"],
    }


def _validate_payload(
    payload: Dict[str, Any],
    *,
    questions: Sequence[Dict[str, Any]],
    page_count: int,
) -> tuple[List[_ValidatedAnswer], List[str]]:
    if payload.get("version") != OBJECTIVE_EXTRACTION_VERSION:
        raise ObjectiveAnswerSheetError(
            "Objective extractor returned an unsupported contract version"
        )
    document = payload.get("document")
    if not isinstance(document, dict):
        raise ObjectiveAnswerSheetError(
            "Objective extractor omitted the document coverage record"
        )
    all_checked = bool(document.get("all_answer_areas_checked"))
    document_confidence = _finite_float(document.get("confidence"), default=0.0)
    document_warnings = [
        str(value).strip()
        for value in (document.get("warnings") or [])
        if str(value).strip()
    ]
    if not all_checked:
        document_warnings.append("Not all expected answer areas were verified")
    if document_confidence < _SELECTED_MIN_CONFIDENCE:
        document_warnings.append("Overall answer-sheet readability is low")

    raw_by_number: Dict[int, List[Dict[str, Any]]] = {}
    for item in payload.get("answers") or []:
        if not isinstance(item, dict):
            continue
        number = _positive_int(item.get("question_number"))
        if number:
            raw_by_number.setdefault(number, []).append(item)

    validated: List[_ValidatedAnswer] = []
    for position, question in enumerate(questions, start=1):
        number = _positive_int(question.get("question_number")) or position
        candidates = raw_by_number.get(number, [])
        if len(candidates) != 1:
            reason = (
                "No answer-state record was returned"
                if not candidates
                else "Duplicate answer-state records were returned"
            )
            validated.append(
                _ValidatedAnswer(
                    question=question,
                    question_number=number,
                    state="unresolved",
                    selected_answer="",
                    confidence=0.0,
                    source_pages=[],
                    review_reason=reason,
                )
            )
            continue

        raw = candidates[0]
        state = str(raw.get("state") or "ambiguous").strip().lower()
        confidence = _finite_float(raw.get("confidence"), default=0.0)
        selected_options = [
            normalize_answer_label(value)
            for value in (raw.get("selected_options") or [])
        ]
        selected_options = [
            value for value in selected_options if value is not None
        ]
        allowed_labels = set(_option_labels(question))
        source_pages = _valid_regions(
            raw.get("evidence_regions"),
            page_count=page_count,
            question_number=number,
        )
        reason = str(raw.get("reason") or "").strip()

        if state == "selected":
            if (
                len(selected_options) == 1
                and selected_options[0] in allowed_labels
                and confidence >= _SELECTED_MIN_CONFIDENCE
                and source_pages
            ):
                validated.append(
                    _ValidatedAnswer(
                        question=question,
                        question_number=number,
                        state="selected",
                        selected_answer=selected_options[0],
                        confidence=confidence,
                        source_pages=source_pages,
                    )
                )
                continue
            reason = reason or (
                "Selected option is missing, outside the allowed labels, or not "
                "visually reliable"
            )
        elif state == "blank":
            if (
                not selected_options
                and all_checked
                and document_confidence >= _BLANK_MIN_CONFIDENCE
                and confidence >= _BLANK_MIN_CONFIDENCE
            ):
                validated.append(
                    _ValidatedAnswer(
                        question=question,
                        question_number=number,
                        state="blank",
                        selected_answer="",
                        confidence=confidence,
                        source_pages=source_pages,
                    )
                )
                continue
            reason = reason or "Blank answer was not verified with full-sheet confidence"
        elif state == "multiple":
            reason = reason or "More than one option appears selected"
        else:
            reason = reason or "The selected option is visually ambiguous"

        validated.append(
            _ValidatedAnswer(
                question=question,
                question_number=number,
                state="unresolved",
                selected_answer="",
                confidence=confidence,
                source_pages=source_pages,
                review_reason=reason,
            )
        )
    return validated, list(dict.fromkeys(document_warnings))


def _valid_regions(
    value: Any,
    *,
    page_count: int,
    question_number: int,
) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return regions
    for item in value:
        if not isinstance(item, dict):
            continue
        page_number = _positive_int(item.get("page_number"))
        x_start = _bounded_coordinate(item.get("x_start"))
        y_start = _bounded_coordinate(item.get("y_start"))
        x_end = _bounded_coordinate(item.get("x_end"))
        y_end = _bounded_coordinate(item.get("y_end"))
        if (
            not page_number
            or page_number > page_count
            or x_start is None
            or y_start is None
            or x_end is None
            or y_end is None
            or x_end <= x_start
            or y_end <= y_start
        ):
            continue
        regions.append(
            {
                "region_id": f"objective-q{question_number}-region-{len(regions) + 1}",
                "page_number": page_number,
                "coordinate_space": "normalized_1000",
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "evidence": str(item.get("evidence") or "").strip()[:500],
                "confidence": _finite_float(
                    item.get("confidence"),
                    default=0.0,
                ),
            }
        )
    return regions


def _input_fingerprint(
    *,
    submission_id: str,
    exam_id: str,
    questions: Sequence[Dict[str, Any]],
    assets: Sequence[_PageAsset],
    model_id: str,
) -> str:
    payload = {
        "version": OBJECTIVE_EXTRACTION_VERSION,
        "extractor_revision": OBJECTIVE_EXTRACTOR_REVISION,
        "submission_id": submission_id,
        "exam_id": exam_id,
        "model_id": model_id,
        "questions": [
            {
                "question_id": question.get("question_id"),
                "question_number": question.get("question_number"),
                "allowed_options": _option_labels(question),
            }
            for question in questions
        ],
        "pages": [
            {"page_number": asset.page_number, "sha256": asset.asset_hash}
            for asset in assets
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
    payload = {
        "version": "pcr-objective-generation-v1",
        "submission_id": submission_id,
        "input_fingerprint": input_fingerprint,
        "generation_revision": max(0, int(generation_revision)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _assert_run_identity(
    run: Mapping[str, Any],
    *,
    submission_id: str,
    input_fingerprint: str,
    generation_fingerprint: str,
    generation_revision: int,
) -> None:
    try:
        saved_revision = int(
            run.get("generation_revision")
            if run.get("generation_revision") is not None
            else run.get("grading_revision")
        )
    except (TypeError, ValueError) as exc:
        raise ObjectiveRunIdentityError(
            "Saved Objective extraction run has an invalid generation revision"
        ) from exc
    if (
        str(run.get("submission_id") or "") != submission_id
        or str(run.get("input_fingerprint") or "") != input_fingerprint
        or str(run.get("generation_fingerprint") or "")
        != generation_fingerprint
        or saved_revision != generation_revision
    ):
        raise ObjectiveRunIdentityError(
            "Objective extraction run ownership does not match its requested generation"
        )


async def _materialization_revision(tenant_db: Any, submission_id: str) -> int:
    jobs = await tenant_db[_PROCESSING_JOBS_COLLECTION].find(
        {"submission_id": submission_id}
    ).sort([("created_at", -1), ("updated_at", -1)]).to_list(length=1)
    if not jobs:
        return 0
    try:
        return max(0, int(jobs[0].get("reprocess_count") or 0))
    except (TypeError, ValueError):
        return 0


def _result_from_run(
    run: Dict[str, Any],
    submission_id: str,
) -> ObjectiveAnswerSheetResult:
    result = dict(run.get("result") or {})
    return ObjectiveAnswerSheetResult(
        handled=True,
        submission_id=submission_id,
        status="completed",
        page_count=int(run.get("page_count") or 0),
        response_count=int(result.get("response_count") or 0),
        evaluated_count=int(result.get("evaluated_count") or 0),
        blocked_count=int(result.get("blocked_count") or 0),
        warning_count=int(result.get("warning_count") or 0),
        run_id=str(run.get("run_id") or "") or None,
        errors=[str(value) for value in result.get("errors") or []],
        document_review_required=bool(result.get("blocked_count")),
        review_state=str(result.get("review_state") or "ready"),
        review_reasons=[
            str(value) for value in result.get("review_reasons") or []
        ],
    )


def _blocking_flag(response_id: str, reason: str) -> Dict[str, Any]:
    return {
        "flag_id": _stable_id("FLG-OBJ", response_id, reason),
        "flag_type": "objective_answer_ambiguous",
        "severity": "blocking",
        "reason": reason,
        "source": OBJECTIVE_EXTRACTION_VERSION,
        "created_at": _now(),
    }


def _objective_feedback(
    *,
    attempted: bool,
    correct: bool,
    selected: str,
    correct_answer: str,
    total_score: float,
) -> str:
    if not attempted:
        return "Not attempted."
    if correct:
        return f"Selected {selected}. Correct."
    return (
        f"Selected {selected}. Correct answer: {correct_answer}. "
        f"Score: {total_score:g}."
    )


def _document_confidence(payload: Dict[str, Any]) -> float:
    document = payload.get("document")
    if not isinstance(document, dict):
        return 0.0
    return _finite_float(document.get("confidence"), default=0.0)


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


def _parse_json_object(value: str) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _finite_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bounded_coordinate(value: Any) -> Optional[float]:
    parsed = _finite_float(value, default=float("nan"))
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1000:
        return None
    return round(parsed, 3)


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _finite_float(value, default=default)
    return max(minimum, min(maximum, parsed))


def _now() -> datetime:
    return datetime.now(timezone.utc)
