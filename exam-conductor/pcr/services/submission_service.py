"""
PCR Submission Service — Orchestrate submission processing from
ingest artifacts through OCR to segmented detected responses.

This service coordinates the upstream portion of the PCR pipeline:

    canonical pages (from ingest) → OCR adapter → PageOCR → segmenter
    → detected responses → persisted to evalpen_detected_responses

It reads from the ingest substrate (server-side fetch, TAMPER_PROOF_SPEC
Layer 2) and writes detected responses through the storage layer.  It
does NOT perform evaluation — that is the responsibility of ``EvalCore``.

Spec authority:  new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md §3
Integrity:       new-docs/architecture/TAMPER_PROOF_SPEC.md (Layer 2)
Failure modes:   PCR-01 (boundary/marker detection failure → flags + review)
Test IDs:        I-PCR-01 (conducted artifact → PageOCR → detected responses),
                 I-PCR-02 (blocking flags prevent auto-eval),
                 I-TAMP-02 (conducted PCR eval fetches server-side artifact)
Hard constraints: C1 (MongoDB only), C3 (practice persistence untouched),
                  C5 (reads from ingest, writes to PCR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..domain.response_models import (
    FlagSeverity,
    SegmentationResult,
)
from ..domain.segmenter import segment_submission

from .ocr_service import OCRAdapter, OCRResult, VisionGateProtocol, create_ocr_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — decouple from concrete repository implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class IngestReader(Protocol):
    """Protocol for reading ingest artifacts (server-side fetch).

    Satisfied by ``ingest.service.IngestService``.
    """

    async def get_submission(
        self, submission_id: str
    ) -> Optional[Dict[str, Any]]:
        ...  # pragma: no cover

    async def get_answer_pages(
        self, submission_id: str
    ) -> List[Dict[str, Any]]:
        ...  # pragma: no cover

    async def update_segmentation_status(
        self, submission_id: str, status: Any
    ) -> bool:
        ...  # pragma: no cover


@runtime_checkable
class ResponseWriter(Protocol):
    """Protocol for persisting detected responses.

    Satisfied by ``pcr.storage.response_repo.DetectedResponseRepository``.
    """

    async def insert_response(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        ...  # pragma: no cover

    async def insert_responses_bulk(
        self, docs: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        ...  # pragma: no cover

    async def update_eval_status(
        self, response_id: str, eval_status: str
    ) -> bool:
        ...  # pragma: no cover

    async def supersede_responses_for_submission(
        self,
        submission_id: str,
        *,
        keep_response_ids: List[str],
        reason: str,
    ) -> int:
        ...  # pragma: no cover


@runtime_checkable
class QuestionReader(Protocol):
    """Protocol for reading question metadata.

    Satisfied by ``pcr.storage.question_repo.QuestionRepository``.
    """

    async def get_questions_by_exam(
        self, exam_id: str
    ) -> List[Dict[str, Any]]:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Processing result envelope
# ---------------------------------------------------------------------------


@dataclass
class SubmissionProcessingResult:
    """Result of processing a conducted-exam submission through the
    OCR → segmentation pipeline.

    Attributes
    ----------
    submission_id : str
        The submission that was processed.
    page_count : int
        Number of pages processed through OCR.
    response_count : int
        Number of detected responses produced.
    inserted_count : int
        Number of new responses persisted.
    duplicate_count : int
        Number of responses already in storage (idempotent).
    blocked_count : int
        Number of responses with blocking flags (I-PCR-02).
    warning_count : int
        Number of responses with warning flags.
    segmentation_result : SegmentationResult | None
        Full segmentation pipeline output (for downstream consumption).
    error : str | None
        Error message if processing failed.
    """

    submission_id: str
    page_count: int = 0
    response_count: int = 0
    inserted_count: int = 0
    duplicate_count: int = 0
    blocked_count: int = 0
    warning_count: int = 0
    segmentation_result: Optional[SegmentationResult] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Submission Service
# ---------------------------------------------------------------------------


class SubmissionService:
    """Orchestrate submission processing from ingest artifacts to
    detected responses.

    This service is responsible for the upstream pipeline:
    ingest artifacts → OCR → segmentation → detected response persistence.

    It enforces server-side fetch (TAMPER_PROOF_SPEC Layer 2) by reading
    canonical artifacts from the ingest substrate, never from client input.

    The OCR adapter is created internally via ``create_ocr_adapter()``
    using the provided LLM gate, selecting the correct adapter (camera
    vs pen) based on the submission's source field.

    Parameters
    ----------
    ingest : IngestReader
        For fetching canonical submission and page artifacts.
    response_repo : ResponseWriter
        For persisting detected responses.
    question_repo : QuestionReader
        For fetching question metadata (used by segmenter for manifest
        question numbers and expected word counts).
    gate : VisionGateProtocol
        An initialised LLM gate instance for OCR vision calls (C4).
    ocr_adapter : OCRAdapter, optional
        Explicit OCR adapter override.  When provided, ``gate`` is not
        used and this adapter is used for all submissions regardless of
        source.  Primarily useful for testing.

    Usage
    -----
    ::

        gate = LLMGate(tenant_db)
        await gate.initialize()
        service = SubmissionService(
            ingest=ingest_service,
            response_repo=detected_response_repo,
            question_repo=question_repo,
            gate=gate,
        )
        result = await service.process_submission("submission-id-123")
    """

    def __init__(
        self,
        ingest: IngestReader,
        response_repo: ResponseWriter,
        question_repo: QuestionReader,
        gate: VisionGateProtocol,
        ocr_adapter: Optional[OCRAdapter] = None,
    ) -> None:
        self._ingest = ingest
        self._response_repo = response_repo
        self._question_repo = question_repo
        self._gate = gate
        self._ocr_override = ocr_adapter

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process_submission(
        self,
        submission_id: str,
    ) -> SubmissionProcessingResult:
        """Process a conducted-exam submission through OCR and segmentation.

        Steps:
        1. Fetch canonical submission from ingest (server-side fetch, I-TAMP-02)
        2. Fetch canonical answer pages from ingest
        3. Run OCR/HWR on answer pages → PageOCR
        4. Fetch question metadata for the exam
        5. Run segmentation pipeline → detected responses + flags
        6. Persist detected responses to ``evalpen_detected_responses``
        7. Set ``eval_status`` on each response based on flag severity
        8. Update submission ``segmentation_status``

        Parameters
        ----------
        submission_id : str
            The canonical submission ID from the ingest substrate.

        Returns
        -------
        SubmissionProcessingResult
            Summary of the processing outcome.
        """
        # Step 1: Server-side fetch of canonical submission (I-TAMP-02)
        submission = await self._ingest.get_submission(submission_id)
        if submission is None:
            logger.error(
                "Submission %s not found in ingest substrate", submission_id
            )
            return SubmissionProcessingResult(
                submission_id=submission_id,
                error=f"Submission {submission_id} not found",
            )

        exam_id = submission.get("exam_id", "")
        student_id = submission.get("student_id", "")
        source = submission.get("source", "camera")

        # Step 2: Fetch canonical answer pages
        answer_pages = await self._ingest.get_answer_pages(submission_id)
        if not answer_pages:
            logger.warning(
                "No answer pages found for submission %s", submission_id
            )
            await self._ingest.update_segmentation_status(
                submission_id, "failed"
            )
            return SubmissionProcessingResult(
                submission_id=submission_id,
                error="No answer pages found for submission",
            )

        # Step 3: Run OCR/HWR → PageOCR
        # Select the appropriate adapter based on submission source.
        normalized_source = _normalize_source(source)
        ocr_adapter = (
            self._ocr_override
            if self._ocr_override is not None
            else create_ocr_adapter(normalized_source, gate=self._gate)
        )
        try:
            ocr_result: OCRResult = await ocr_adapter.recognize_pages(
                answer_pages,
                source=normalized_source,
            )
        except Exception:
            logger.exception(
                "OCR failed for submission %s", submission_id
            )
            await self._ingest.update_segmentation_status(
                submission_id, "failed"
            )
            return SubmissionProcessingResult(
                submission_id=submission_id,
                page_count=len(answer_pages),
                error="OCR recognition failed",
            )

        pages = ocr_result.pages
        if not pages:
            logger.warning(
                "OCR produced no pages for submission %s", submission_id
            )
            await self._ingest.update_segmentation_status(
                submission_id, "failed"
            )
            return SubmissionProcessingResult(
                submission_id=submission_id,
                page_count=len(answer_pages),
                error="OCR produced no recognizable pages",
            )
        if all(not page.text_blocks for page in pages):
            logger.warning(
                "OCR produced no text blocks for submission %s", submission_id
            )
            await self._ingest.update_segmentation_status(
                submission_id, "failed"
            )
            return SubmissionProcessingResult(
                submission_id=submission_id,
                page_count=len(pages),
                error="OCR produced no text blocks",
            )

        # Step 4: Fetch question metadata for manifest awareness
        questions = await self._question_repo.get_questions_by_exam(exam_id)
        manifest_question_numbers = {
            q.get("question_number", i + 1) for i, q in enumerate(questions)
        }
        expected_max_words = {
            q.get("question_number", i + 1): (
                q.get("expected_word_range", {}).get("max")
                if isinstance(q.get("expected_word_range"), dict)
                else None
            )
            for i, q in enumerate(questions)
        }

        # Step 5: Run segmentation pipeline
        seg_result = segment_submission(
            pages=pages,
            stroke_lines=None,  # TODO: extract from pen pages when HWR is live
            hough_lines=None,   # TODO: extract from camera pages when OCR is live
            manifest_question_numbers=(
                manifest_question_numbers if manifest_question_numbers else None
            ),
            expected_max_words_by_question=(
                expected_max_words if any(expected_max_words.values()) else None
            ),
        )

        # Step 6: Persist detected responses
        response_docs = _build_response_docs(
            seg_result, submission_id, exam_id, student_id
        )

        inserted_count = 0
        duplicate_count = 0

        if response_docs:
            inserted_count, duplicate_count = (
                await self._response_repo.insert_responses_bulk(response_docs)
            )
            logger.info(
                "Persisted %d detected responses for submission %s "
                "(%d duplicates)",
                inserted_count,
                submission_id,
                duplicate_count,
            )

        # Step 7: Set eval_status based on flags (I-PCR-02)
        blocked_count = 0
        warning_count = 0

        for response in seg_result.responses:
            has_blocking = any(
                f.severity == FlagSeverity.BLOCKING for f in response.flags
            )
            has_warning = any(
                f.severity == FlagSeverity.WARNING for f in response.flags
            )

            if has_blocking:
                eval_status = "blocked"
                blocked_count += 1
            elif has_warning:
                eval_status = "ready_with_warnings"
                warning_count += 1
            else:
                eval_status = "ready"

            await self._response_repo.update_eval_status(
                response.response_id, eval_status
            )

        if response_docs:
            await self._response_repo.supersede_responses_for_submission(
                submission_id,
                keep_response_ids=[
                    doc["response_id"] for doc in response_docs
                ],
                reason="submission_reprocessed",
            )

        # Step 8: Update submission segmentation status
        seg_status = "complete"
        await self._ingest.update_segmentation_status(
            submission_id, seg_status
        )

        logger.info(
            "Submission %s processed: %d pages, %d responses "
            "(%d blocked, %d warnings)",
            submission_id,
            len(pages),
            len(seg_result.responses),
            blocked_count,
            warning_count,
        )

        return SubmissionProcessingResult(
            submission_id=submission_id,
            page_count=len(pages),
            response_count=len(seg_result.responses),
            inserted_count=inserted_count,
            duplicate_count=duplicate_count,
            blocked_count=blocked_count,
            warning_count=warning_count,
            segmentation_result=seg_result,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def process_pending_submissions(
        self,
        submission_ids: List[str],
    ) -> List[SubmissionProcessingResult]:
        """Process a batch of pending submissions sequentially.

        Parameters
        ----------
        submission_ids : list[str]
            Submission IDs to process.

        Returns
        -------
        list[SubmissionProcessingResult]
            One result per submission.
        """
        results: List[SubmissionProcessingResult] = []

        for sid in submission_ids:
            try:
                result = await self.process_submission(sid)
                results.append(result)
            except Exception:
                logger.exception(
                    "Failed to process submission %s", sid
                )
                results.append(
                    SubmissionProcessingResult(
                        submission_id=sid,
                        error="Unhandled exception during processing",
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_source(source: str) -> str:
    """Normalize source string to ``pen`` or ``camera``."""
    if source in ("pen", "ble_pen"):
        return "pen"
    return "camera"


def _build_response_docs(
    seg_result: SegmentationResult,
    submission_id: str,
    exam_id: str,
    student_id: str,
) -> List[Dict[str, Any]]:
    """Convert segmentation responses into MongoDB documents for
    persistence.

    Each document matches the ``evalpen_detected_responses`` schema
    from PCR_EVAL_ENGINE_SPEC §7.2.
    """
    docs: List[Dict[str, Any]] = []

    for response in seg_result.responses:
        # Build question_id from exam_id + question_number
        question_id: Optional[str] = None
        if response.question_number is not None:
            sub_suffix = f".{response.sub_part}" if response.sub_part else ""
            question_id = f"{exam_id}_Q{response.question_number}{sub_suffix}"

        # Serialize flags
        flags_serialized = [
            {
                "flag_id": f.flag_id,
                "response_id": f.response_id,
                "source": f.source,
                "flag_type": f.flag_type.value if hasattr(f.flag_type, "value") else str(f.flag_type),
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                "reason": f.reason,
                "suggested_action": f.suggested_action,
                "metadata": f.metadata,
            }
            for f in response.flags
        ]

        # Serialize source pages
        source_pages_serialized = [
            {
                "page_number": sp.page_number,
                "y_start": sp.y_start,
                "y_end": sp.y_end,
            }
            for sp in response.source_pages
        ]

        doc: Dict[str, Any] = {
            "response_id": response.response_id,
            "submission_id": submission_id,
            "question_id": question_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "detected_text": response.detected_text,
            "source_pages": source_pages_serialized,
            "content_type": response.content_type.value,
            "text_coverage_ratio": response.text_coverage_ratio,
            "segmentation_confidence": response.segmentation_confidence,
            "ocr_confidence": response.ocr_confidence,
            "flags": flags_serialized,
            "word_count": response.word_count,
            "is_continuation": response.is_continuation,
            "eval_status": "pending",
            "_immutable": True,
            "created_at": datetime.now(timezone.utc),
        }
        docs.append(doc)

    return docs
