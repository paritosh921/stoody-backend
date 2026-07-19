"""Submission-level visual grading for PCR answer copies.

This is the primary camera/PDF path for papers where handwriting, diagrams,
tables, and answer ownership cannot safely be reduced to OCR text first.  One
GPT-5 Responses request receives the immutable question paper, the teacher's
uploaded solution document (when present), and every canonical student page.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from ..domain.response_models import ContentType
from ..storage.evaluation_repo import EvaluationRepository
from ..storage.response_repo import DetectedResponseRepository
from .ocr_service import AssetIntegrityError, _resolve_image_base64

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "pcr-full-document-visual-v1"
_RUNS_COLLECTION = "evalpen_document_grading_runs"
_CALLER_ID = "pcr_eval_core"
_AUTO_ACCEPT_CONFIDENCE = 0.80
_ABSENCE_CONFIDENCE = 0.85
_MAX_PAGE_COUNT = 50
_MAX_STATIC_PDF_BYTES = 45 * 1024 * 1024
_MAX_REQUEST_PAYLOAD_BYTES = 45 * 1024 * 1024
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


@dataclass
class FullDocumentGradingResult:
    handled: bool
    submission_id: str
    status: str = "not_applicable"
    page_count: int = 0
    response_count: int = 0
    evaluated_count: int = 0
    blocked_count: int = 0
    warning_count: int = 0
    run_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class _ValidatedGrade:
    question: Dict[str, Any]
    question_number: int
    attempt_status: str
    confidence: float
    student_answer: str
    content_type: str
    source_pages: List[Dict[str, float]]
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
        if not _feature_enabled() or source not in {"camera", "pdf", "scan"}:
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
            )
        if not _is_openai_visual_model(self._model_id):
            logger.info(
                "Full-document grading skipped for non-OpenAI model %s",
                self._model_id,
            )
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
            )

        exam_id = str(submission.get("exam_id") or "")
        student_id = str(submission.get("student_id") or "")
        exam = await self._db["exampen_exams"].find_one({"exam_id": exam_id})
        if not exam or str(exam.get("exam_type") or "") != "pcr":
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
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

        answer_pages = await self._db["evalpen_answer_pages"].find(
            {"submission_id": submission_id}
        ).sort("page_number", 1).to_list(length=_MAX_PAGE_COUNT + 1)
        if not answer_pages:
            raise FullDocumentGradingError("Canonical student answer pages are missing")
        if len(answer_pages) > _MAX_PAGE_COUNT:
            raise FullDocumentGradingError(
                f"Student copy has {len(answer_pages)} pages; maximum is {_MAX_PAGE_COUNT}"
            )

        paper_version = await self._db["exampen_paper_versions"].find_one(
            {"paper_version_id": exam.get("paper_version_id")}
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
            # Legacy sessions without the original PDF remain on the existing
            # review-safe pipeline.  Do not accept a client-provided substitute.
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
            )

        paper_bytes = await _read_canonical_file(
            str(document.get("file_path") or ""),
            expected_sha256=document.get("sha256"),
        )
        if not paper_bytes:
            return FullDocumentGradingResult(
                handled=False,
                submission_id=submission_id,
            )
        solution_bytes = await _read_canonical_file(
            str(document.get("answer_sheet_path") or ""),
            expected_sha256=document.get("answer_sheet_sha256"),
        )
        if len(paper_bytes) + len(solution_bytes or b"") > _MAX_STATIC_PDF_BYTES:
            raise FullDocumentGradingError(
                "Question paper and teacher solution exceed the document-input size limit"
            )

        input_fingerprint = _input_fingerprint(
            submission=submission,
            exam=exam,
            document=document,
            answer_pages=answer_pages,
            model_id=self._model_id,
        )
        run_id = f"DOCGR-{input_fingerprint[:24]}"
        await self._db[_RUNS_COLLECTION].create_index(
            "run_id", unique=True, name="uniq_document_grading_run"
        )
        existing_run = await self._db[_RUNS_COLLECTION].find_one({"run_id": run_id})
        if existing_run and existing_run.get("status") == "completed":
            active_count = await self._db["evalpen_detected_responses"].count_documents(
                {
                    "submission_id": submission_id,
                    "mapping_version_id": run_id,
                    "superseded_at": {"$exists": False},
                }
            )
            if active_count == len(questions):
                return _result_from_run(existing_run, submission_id)

        if existing_run and existing_run.get("status") in {
            "validated",
            "materializing",
            "completed",
        }:
            raw_payload = existing_run.get("validated_payload")
            if not isinstance(raw_payload, dict):
                raise FullDocumentGradingError("Saved document grading ledger is invalid")
            usage = dict(existing_run.get("token_usage") or {})
            raw_llm = str(existing_run.get("raw_llm_response") or "")
        else:
            student_content, student_image_bytes = await _student_copy_content(answer_pages)
            if (
                len(paper_bytes)
                + len(solution_bytes or b"")
                + student_image_bytes
                > _MAX_REQUEST_PAYLOAD_BYTES
            ):
                raise FullDocumentGradingError(
                    "Paper, solution, and optimized student pages exceed the visual "
                    "request size limit"
                )
            request_input = _build_responses_input(
                questions=questions,
                paper_bytes=paper_bytes,
                solution_bytes=solution_bytes,
                student_content=student_content,
                paper_filename=str(document.get("filename") or "question-paper.pdf"),
                solution_filename=str(
                    document.get("answer_sheet_filename") or "teacher-solution.pdf"
                ),
            )
            try:
                gate_response = await self._gate.call(
                    model_id=self._model_id,
                    prompt="",
                    caller_id=_CALLER_ID,
                    responses_input=request_input,
                    json_schema=_evidence_ledger_schema(),
                    prompt_cache_key=f"pcr-paper-{_static_context_hash(exam, document)[:32]}",
                    reasoning_effort="medium",
                    max_output_tokens=min(30_000, max(8_000, 1_100 * len(questions))),
                    metadata={
                        "pcr_stage": "full_document_visual_grading",
                        "prompt_version": _PROMPT_VERSION,
                        "submission_id": submission_id,
                        "exam_id": exam_id,
                        "question_count": len(questions),
                        "page_count": len(answer_pages),
                        "run_id": run_id,
                    },
                )
            except Exception as exc:
                raise FullDocumentGradingError(
                    f"Full-document model request failed: {str(exc)[:400]}"
                ) from exc

            raw_llm = str(getattr(gate_response, "content", "") or "")
            raw_payload = _parse_json_object(raw_llm)
            if raw_payload is None:
                raise FullDocumentGradingError(
                    "Full-document model returned an invalid evidence ledger"
                )
            usage_obj = getattr(gate_response, "usage", None)
            usage = {
                "model": str(getattr(usage_obj, "model", self._model_id)),
                "caller": str(getattr(usage_obj, "caller", _CALLER_ID)),
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
            now = datetime.now(timezone.utc)
            await self._db[_RUNS_COLLECTION].update_one(
                {"run_id": run_id},
                {
                    "$setOnInsert": {
                        "run_id": run_id,
                        "submission_id": submission_id,
                        "exam_id": exam_id,
                        "student_id": student_id,
                        "input_fingerprint": input_fingerprint,
                        "page_count": len(answer_pages),
                        "created_at": now,
                    },
                    "$set": {
                        "status": "validated",
                        "prompt_version": _PROMPT_VERSION,
                        "model_used": usage.get("model") or self._model_id,
                        "validated_payload": raw_payload,
                        "raw_llm_response": raw_llm,
                        "token_usage": usage,
                        "updated_at": now,
                    },
                },
                upsert=True,
            )

        grades, document_errors = _validate_ledger(
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
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        result = await self._materialize(
            run_id=run_id,
            submission=submission,
            questions=questions,
            grades=grades,
            raw_payload=raw_payload,
            usage=usage,
            page_count=len(answer_pages),
            document_errors=document_errors,
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
        submission: Dict[str, Any],
        questions: List[Dict[str, Any]],
        grades: List[_ValidatedGrade],
        raw_payload: Dict[str, Any],
        usage: Dict[str, Any],
        page_count: int,
        document_errors: List[str],
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
            response_id = _stable_id("RESP-DOC", run_id, question_id)
            unresolved = grade.attempt_status == "unresolved"
            is_missing = grade.attempt_status == "not_attempted"
            flags: List[Dict[str, Any]] = []
            if unresolved:
                flags.append(
                    _review_flag(
                        response_id,
                        severity="blocking",
                        reason=grade.review_reason,
                    )
                )
            elif grade.manual_review_required:
                flags.append(
                    _review_flag(
                        response_id,
                        severity="warning",
                        reason=grade.review_reason,
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
                    "prompt_version": _PROMPT_VERSION,
                    "model_used": model_used,
                    "grading_run_id": run_id,
                    "manual_review_required": grade.manual_review_required or unresolved,
                    "reason": grade.review_reason or None,
                },
                "exam_id": exam_id,
                "student_id": student_id,
                "detected_text": grade.student_answer,
                "source_pages": grade.source_pages,
                "evidence_version": 2,
                "evidence_atom_ids": [
                    _stable_id(
                        "region",
                        submission_id,
                        str(item["page_number"]),
                        str(item["y_start"]),
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
                "eval_status": "blocked" if unresolved else "pending",
                "mapping_version_id": run_id,
                "_immutable": True,
                "created_at": datetime.now(timezone.utc),
            }
            response_docs.append(response_doc)

            if unresolved or grade.total_score is None:
                continue
            max_marks = _max_marks(grade.question)
            eval_id = _stable_id("EVAL-DOC", run_id, question_id)
            raw_question_result = raw_by_number.get(grade.question_number, {})
            evaluation_docs.append(
                {
                    "evaluation_id": eval_id,
                    "evaluation_input_version": 2,
                    "mapping_version_id": run_id,
                    "response_id": response_id,
                    "question_id": question_id,
                    "student_id": student_id,
                    "eval_path": (
                        "full_document_visual_not_attempted"
                        if is_missing
                        else "full_document_visual"
                    ),
                    "model_used": model_used,
                    "total_score": grade.total_score,
                    "max_score": max_marks,
                    "scoreable_max": max_marks,
                    "marking_policy": dict(grade.question.get("marking_policy") or {}),
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
                    "reference_solution": _reference_solution(grade.question),
                    "token_usage": {
                        "shared_document_call_id": run_id,
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
        await self._db["evalpen_submissions"].update_one(
            {"submission_id": submission_id},
            {
                "$set": {
                    "segmentation_status": "complete",
                    "processing_path": "full_document_visual",
                    "document_grading_run_id": run_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        blocked = sum(1 for grade in grades if grade.attempt_status == "unresolved")
        warnings = sum(
            1
            for grade in grades
            if grade.attempt_status != "unresolved" and grade.manual_review_required
        )
        evaluated = len(evaluation_docs)
        errors = list(document_errors)
        errors.extend(
            f"Q{grade.question_number}: {grade.review_reason}"
            for grade in grades
            if grade.attempt_status == "unresolved" and grade.review_reason
        )
        return FullDocumentGradingResult(
            handled=True,
            submission_id=submission_id,
            status="blocked_for_review" if blocked or warnings else "completed",
            page_count=page_count,
            response_count=len(response_docs),
            evaluated_count=evaluated,
            blocked_count=blocked,
            warning_count=warnings,
            run_id=run_id,
            errors=errors,
        )


def _feature_enabled() -> bool:
    return os.getenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
        "For every catalog question return exactly one result. attempt_status=attempted "
        "only when student work is visibly present. Use not_attempted only after checking "
        "every student page and finding no work for that question. Use unresolved when "
        "ownership, handwriting, page coverage, or the correct award is genuinely "
        "uncertain. Never guess a zero. For attempted answers, apply only the locked "
        "criterion IDs and maximums from the catalog and return every locked criterion "
        "exactly once. For not_attempted return empty student_answer, evidence_regions, "
        "and criterion_marks with total_score 0. For unresolved return no award and empty "
        "criterion_marks with total_score 0. Award step marks for correct visible "
        "work even when the final answer is wrong. Evaluate diagrams visually. Cite the "
        "student page and a short literal/visual description for every awarded mark. "
        "Do not exceed any criterion or question maximum. Set needs_review for low-quality "
        "images, ambiguous ownership, contradictory work, unreadable evidence, or any "
        "uncertain award. Coordinates are approximate vertical bands from 0 at page top "
        "to 1000 at page bottom."
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
            "marks_awarded": {"type": "number", "minimum": 0},
            "rationale": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["criterion_id", "marks_awarded", "rationale", "evidence"],
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
) -> tuple[List[_ValidatedGrade], List[str]]:
    document_review = payload.get("document_review")
    document_errors: List[str] = []
    coverage_complete = False
    coverage_confidence = 0.0
    if isinstance(document_review, dict):
        coverage_complete = bool(document_review.get("all_student_work_accounted"))
        coverage_confidence = _confidence(document_review.get("confidence"))
        for warning in document_review.get("warnings") or []:
            if str(warning).strip():
                document_errors.append(str(warning).strip()[:300])
    else:
        document_errors.append("Model omitted the full-copy coverage review")
    # A model-raised document warning means absence is not proven and every
    # otherwise valid attempted score must remain review-gated.
    coverage_complete = coverage_complete and not document_errors

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
            coverage_complete=coverage_complete,
            coverage_confidence=coverage_confidence,
        )
        grades.append(grade)

    expected_numbers = {
        _positive_int(question.get("question_number")) or position
        for position, question in enumerate(questions, start=1)
    }
    unexpected = sorted(set(candidates) - expected_numbers)
    if unexpected:
        document_errors.append(
            "Model returned non-catalog question numbers: "
            + ", ".join(str(value) for value in unexpected)
        )
    _mark_overlapping_evidence_for_review(grades)
    return grades, document_errors


def _validate_question_grade(
    item: Dict[str, Any],
    *,
    question: Dict[str, Any],
    question_number: int,
    page_count: int,
    coverage_complete: bool,
    coverage_confidence: float,
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
    max_marks = _max_marks(question)
    criteria = _criteria(question)
    criterion_marks: List[Dict[str, Any]] = []
    total_score: Optional[float] = None
    manual_review = bool(item.get("needs_review"))
    review_reason = str(item.get("review_reason") or "").strip()

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
        if not coverage_complete or coverage_confidence < _ABSENCE_CONFIDENCE:
            return _unresolved_grade(
                question,
                question_number,
                "The full-copy scan did not prove that this question was unattempted",
                confidence=min(confidence, coverage_confidence),
            )
        if confidence < _ABSENCE_CONFIDENCE:
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
                "rationale": "No student attempt was found after reviewing the full copy.",
                "evidence": "No student evidence located on any submitted page.",
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
    if not source_pages:
        validation_errors.append("Attempted answer has no visual evidence region")
    if confidence < 0.50:
        validation_errors.append("Question ownership confidence is below 0.50")

    raw_marks = item.get("criterion_marks")
    raw_marks = raw_marks if isinstance(raw_marks, list) else []
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for raw in raw_marks:
        if isinstance(raw, dict):
            by_id.setdefault(str(raw.get("criterion_id") or "").strip(), []).append(raw)
    expected_ids = {criterion["criterion_id"] for criterion in criteria}
    if set(by_id) != expected_ids:
        validation_errors.append("Criterion IDs do not match the locked marking plan")
    for criterion in criteria:
        rows = by_id.get(criterion["criterion_id"], [])
        if len(rows) != 1:
            validation_errors.append(
                f"Criterion {criterion['criterion_id']} is missing or duplicated"
            )
            continue
        raw = rows[0]
        awarded = _finite_float(raw.get("marks_awarded"))
        if awarded is None or awarded < 0 or awarded > criterion["max_marks"]:
            validation_errors.append(
                f"Criterion {criterion['criterion_id']} award is outside its locked range"
            )
            continue
        evidence = str(raw.get("evidence") or "").strip()
        if awarded > 0 and not evidence:
            validation_errors.append(
                f"Criterion {criterion['criterion_id']} awards marks without evidence"
            )
        criterion_marks.append(
            {
                "criterion_id": criterion["criterion_id"],
                "description": criterion["description"],
                "marks_awarded": round(awarded, 2),
                "max_marks": criterion["max_marks"],
                "rationale": str(raw.get("rationale") or "").strip(),
                "evidence": evidence,
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
                }
            ]

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

    if (
        confidence < _AUTO_ACCEPT_CONFIDENCE
        or not coverage_complete
        or coverage_confidence < _AUTO_ACCEPT_CONFIDENCE
    ):
        manual_review = True
        if not review_reason:
            review_reason = (
                "The visual evidence or whole-copy coverage is below the automatic "
                "publication threshold"
            )
    return _ValidatedGrade(
        question=question,
        question_number=question_number,
        attempt_status="attempted",
        confidence=confidence,
        student_answer=student_answer,
        content_type=content_type,
        source_pages=source_pages,
        criterion_marks=criterion_marks,
        total_score=total_score,
        overall_feedback=str(item.get("overall_feedback") or "").strip(),
        manual_review_required=manual_review,
        review_reason=review_reason,
    )


def _validate_regions(raw_regions: Any, *, page_count: int) -> tuple[List[Dict[str, float]], List[str]]:
    regions: List[Dict[str, float]] = []
    errors: List[str] = []
    if not isinstance(raw_regions, list):
        return [], ["Evidence regions must be an array"]
    for item in raw_regions:
        if not isinstance(item, dict):
            errors.append("Evidence region is not an object")
            continue
        page_number = _positive_int(item.get("page_number"))
        start = _finite_float(item.get("y_start"))
        end = _finite_float(item.get("y_end"))
        if not page_number or page_number > page_count:
            errors.append("Evidence refers to a non-submitted page")
            continue
        if start is None or end is None or start < 0 or end > 1000 or end <= start:
            errors.append("Evidence has an invalid vertical page band")
            continue
        regions.append(
            {
                "page_number": page_number,
                "y_start": round((start / 1000.0) * _A4_HEIGHT_MM, 3),
                "y_end": round((end / 1000.0) * _A4_HEIGHT_MM, 3),
            }
        )
    return regions, errors


def _unresolved_grade(
    question: Dict[str, Any],
    question_number: int,
    reason: str,
    *,
    confidence: float = 0.0,
    source_pages: Optional[List[Dict[str, float]]] = None,
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


def _regions_overlap(left: List[Dict[str, float]], right: List[Dict[str, float]]) -> bool:
    for a in left:
        for b in right:
            if a["page_number"] != b["page_number"]:
                continue
            overlap = min(a["y_end"], b["y_end"]) - max(a["y_start"], b["y_start"])
            if overlap <= 0:
                continue
            smaller = min(a["y_end"] - a["y_start"], b["y_end"] - b["y_start"])
            if smaller > 0 and overlap / smaller >= 0.50:
                return True
    return False


def _catalog_question(question: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_number": _positive_int(question.get("question_number")),
        "question_id": str(question.get("question_id") or ""),
        "question_text": str(question.get("question_text") or "")[:4000],
        "max_marks": _max_marks(question),
        "reference_solution": _reference_solution(question)[:5000],
        "marking_criteria": _criteria(question),
        "expects_diagram": bool(question.get("expects_diagram")),
    }


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
            criterion_total = round(sum(item["max_marks"] for item in criteria), 2)
            if abs(criterion_total - max_marks) > 0.01:
                errors.append(
                    f"Q{number} criterion maximums total {criterion_total:g}, "
                    f"question maximum is {max_marks:g}"
                )
    return errors


def _criteria(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = question.get("marking_criteria")
    if not isinstance(raw, list):
        return []
    criteria: List[Dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        criterion_id = str(item.get("criterion_id") or item.get("id") or f"c{index}").strip()
        max_marks = _finite_float(item.get("max_marks"))
        if not criterion_id or max_marks is None or max_marks < 0:
            continue
        criteria.append(
            {
                "criterion_id": criterion_id,
                "description": str(
                    item.get("description") or item.get("criterion") or ""
                ).strip(),
                "max_marks": round(max_marks, 2),
                "expected_evidence": str(
                    item.get("expected_evidence") or item.get("evidence") or ""
                ).strip(),
            }
        )
    return criteria


def _max_marks(question: Dict[str, Any]) -> float:
    value = _finite_float(question.get("max_marks"))
    return round(max(0.0, value or 0.0), 2)


def _reference_solution(question: Dict[str, Any]) -> str:
    return str(
        question.get("reference_solution")
        or question.get("teacher_reference_solution")
        or ""
    ).strip()


def _review_flag(response_id: str, *, severity: str, reason: str) -> Dict[str, Any]:
    return {
        "flag_id": _stable_id("FLG-DOC", response_id, reason),
        "response_id": response_id,
        "source": "full_document_visual",
        "flag_type": "llm_score_divergence",
        "severity": severity,
        "reason": reason,
        "suggested_action": "Review the cited pages against the original answer copy",
        "metadata": {"prompt_version": _PROMPT_VERSION},
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
    submission: Dict[str, Any],
    exam: Dict[str, Any],
    document: Dict[str, Any],
    answer_pages: List[Dict[str, Any]],
    model_id: str,
) -> str:
    payload = {
        "version": _PROMPT_VERSION,
        "model": model_id,
        "submission_id": submission.get("submission_id"),
        "submission_hash": submission.get("content_hash"),
        "paper_version_id": exam.get("paper_version_id"),
        "paper_hash": exam.get("paper_content_hash") or document.get("sha256"),
        "solution_hash": document.get("answer_sheet_sha256"),
        "pages": [
            [page.get("page_number"), page.get("asset_sha256") or page.get("content_hash")]
            for page in answer_pages
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _static_context_hash(exam: Dict[str, Any], document: Dict[str, Any]) -> str:
    value = "\x1f".join(
        [
            _PROMPT_VERSION,
            str(exam.get("paper_version_id") or ""),
            str(exam.get("paper_content_hash") or document.get("sha256") or ""),
            str(document.get("answer_sheet_sha256") or ""),
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
        status="blocked_for_review" if blocked or warnings else "completed",
        page_count=int(run.get("page_count") or 0),
        response_count=int(result.get("response_count") or 0),
        evaluated_count=int(result.get("evaluated_count") or 0),
        blocked_count=blocked,
        warning_count=warnings,
        run_id=str(run.get("run_id") or "") or None,
        errors=[str(value) for value in (result.get("errors") or [])],
    )
