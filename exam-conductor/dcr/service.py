"""
DCR Service — orchestration layer.

Orchestrates the full DCR evaluation pipeline:
    fetch artifacts → render strokes → overlay on template → LLM Vision OCR
    → match → score → store results

Architecture: DUAL_MODE_ARCHITECTURE.md §4
Ownership: STATE_OWNERSHIP_MAP.md — DCR engine writes to ``exampen_dcr_results``
Test IDs:
  - U-DCR-03 — default path does not require gate
  - I-DCR-01 — canonical artifact -> DCR result commit
  - I-DCR-02 — fallback path logs gate usage when enabled
  - E2E-DCR-01 — conducted DCR exam -> canonical artifact -> DCR score
Failure modes:
  - DCR-01 — low confidence → gate fallback
  - DCR-02 — numeric tolerance (delegated to matcher)
  - DCR-03 — scope guard (no deep PCR semantics)

Hard constraints:
  - C1: MongoDB only
  - C2: archiveDCR is backup only
  - C3: No practice persistence
  - C4: All LLM calls go through the gate (recognizer + fallback)
  - C5: Reads canonical artifacts; does not own artifact persistence
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol

from .matcher import TemplateMatcher
from .models import (
    AnswerKey,
    AuditAction,
    DCRAuditEntry,
    DCREvaluateRequest,
    DCREvaluateResponse,
    DCRQuestionResult,
    DCRResult,
    DCRSubmission,
    MatchOutput,
    MatchType,
    RecognitionOutput,
)
from .recognizer import HWRRecognizer
from .repository import DCRRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate interface dependency (C4 — interface only, not implemented here)
# ---------------------------------------------------------------------------
# The actual gate implementation lives in ``exam_conductor.llm_gate.LLMGate``.
# DCR defines a Protocol so it can depend on the gate without importing the
# concrete class at module level (avoids circular imports and keeps DCR
# independently testable).
#
# The real ``LLMGate.call()`` returns a ``GateResponse`` (pydantic model
# with ``.content: str`` and ``.usage: TokenUsage``).  The protocol below
# matches that signature.
# ---------------------------------------------------------------------------

class GateInterface(Protocol):
    """
    Protocol for the shared LLM gate (LLM_GATE_SPEC.md §4).

    DCR uses ``caller_id="dcr_ai"`` for both primary Vision OCR recognition
    and low-confidence fallback (§5).  The concrete implementation is
    ``exam_conductor.llm_gate.LLMGate`` — only the interface dependency is
    declared here (C4).

    The return type is ``GateResponse`` which has:
      - ``content: str``   — the LLM text output
      - ``usage: TokenUsage`` — full token accounting
    """

    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Invoke the LLM gate.

        Parameters
        ----------
        messages
            Optional pre-built messages array for multimodal / vision calls.
            When provided, the array is forwarded to the provider as-is and
            *prompt* is used only for logging context.

        Returns a gate response object with ``.content`` (str) and
        ``.usage`` (with ``.total_tokens``, ``.model``, etc.).
        """
        ...


# Type alias for the answer-key loader callback
AnswerKeyLoader = Callable[
    [str, Optional[List[str]]],
    Coroutine[Any, Any, List[AnswerKey]],
]

# Type alias for the question-region loader callback
QuestionRegionLoader = Callable[
    [str],
    Coroutine[Any, Any, Dict[int, List[Dict[str, Any]]]],
]

# Type alias for the template-image loader callback
# Takes exam_id and returns PNG bytes or None.
TemplateImageLoader = Callable[
    [str],
    Coroutine[Any, Any, Optional[bytes]],
]


# ---------------------------------------------------------------------------
# DCR Service
# ---------------------------------------------------------------------------

class DCRService:
    """
    Orchestrates DCR evaluation for conducted exams.

    Primary path: template overlay + LLM Vision OCR via the gate.
    Fallback path (I-DCR-02): gate-mediated text refinement when primary
    recognition confidence is below threshold.

    Usage::

        from exam_conductor.llm_gate import LLMGate

        repo = DCRRepository(tenant_db)
        gate = LLMGate(tenant_db)
        await gate.initialize()
        recognizer = HWRRecognizer(gate)
        matcher = TemplateMatcher()
        service = DCRService(repo, recognizer, matcher, gate=gate)
        response = await service.evaluate(
            request, answer_key_loader, region_loader, template_loader
        )
    """

    def __init__(
        self,
        repository: DCRRepository,
        recognizer: HWRRecognizer,
        matcher: TemplateMatcher,
        *,
        gate: Optional[GateInterface] = None,
        gate_model_id: str = "gpt-4o",
        enable_gate_fallback: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        repository
            DCR MongoDB repository.
        recognizer
            Vision-based HWR recognizer (uses the gate internally).
        matcher
            Deterministic template matcher.
        gate
            Optional LLM gate interface for the low-confidence text-based
            fallback (C4: interface-only dep).  Note: the primary Vision
            OCR uses the gate directly via the recognizer.
        gate_model_id
            Model ID to pass to the gate for fallback calls.
        enable_gate_fallback
            Whether to invoke the text-based gate fallback when Vision OCR
            confidence is below threshold.  Even when ``True``, the gate
            must also be provided.
        """
        self._repo = repository
        self._recognizer = recognizer
        self._matcher = matcher
        self._gate = gate
        self._gate_model_id = gate_model_id
        self._enable_gate_fallback = enable_gate_fallback

    # ------------------------------------------------------------------
    # Primary evaluation entry point
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        request: DCREvaluateRequest,
        answer_key_loader: AnswerKeyLoader,
        region_loader: QuestionRegionLoader,
        template_loader: Optional[TemplateImageLoader] = None,
    ) -> DCREvaluateResponse:
        """
        Run the full DCR evaluation pipeline for a conducted exam.

        Pipeline:
            1. Fetch canonical submission from repository (C5 -- read-only).
            2. Load answer keys from exam metadata.
            3. Load question spatial regions for the submission pages.
            3b. Optionally load template image for the exam.
            4. Render strokes + overlay on template + LLM Vision OCR.
            5. Optionally route low-confidence outputs through text gate (DCR-01).
            6. Match recognized text against answer keys (DCR-02 tolerance).
            7. Store results atomically (I-DCR-01).
            8. Return evaluation response.

        Parameters
        ----------
        request
            Evaluation request with submission/exam/student identifiers.
        answer_key_loader
            Async callback ``(exam_id, question_ids?) -> list[AnswerKey]``.
            Loads answer keys from exam metadata (not owned by DCR -- C5).
        region_loader
            Async callback ``(exam_id) -> {page_number: [region_dicts]}``.
            Loads question bounding-box regions for the exam pages.
        template_loader
            Optional async callback ``(exam_id) -> bytes | None``.
            Loads the answer-sheet template PNG for the exam.  When ``None``
            or when the callback returns ``None``, recognition proceeds
            without a template overlay (strokes-only mode).

        Returns
        -------
        DCREvaluateResponse

        Raises
        ------
        ValueError
            If the submission is not found.
        """
        now = datetime.now(timezone.utc)

        # ── 1. Fetch canonical submission ────────────────────────────
        submission = await self._repo.get_submission(request.submission_id)
        if submission is None:
            raise ValueError(
                f"DCR submission not found: {request.submission_id}"
            )

        # Verify identity consistency
        if submission.exam_id != request.exam_id:
            raise ValueError(
                f"Exam ID mismatch: submission has {submission.exam_id}, "
                f"request has {request.exam_id}"
            )
        if submission.student_id != request.student_id:
            raise ValueError(
                f"Student ID mismatch: submission has {submission.student_id}, "
                f"request has {request.student_id}"
            )

        # ── 2. Load answer keys ─────────────────────────────────────
        answer_keys = await answer_key_loader(
            request.exam_id, request.question_ids
        )
        if not answer_keys:
            logger.warning(
                "No answer keys returned for exam %s; evaluation will produce "
                "no results.",
                request.exam_id,
            )
            return DCREvaluateResponse(
                submission_id=request.submission_id,
                exam_id=request.exam_id,
                student_id=request.student_id,
                evaluated_at=now,
            )

        # ── 3. Load question regions ────────────────────────────────
        question_regions_by_page = await region_loader(request.exam_id)

        # ── 3b. Fetch answer pages from evalpen_answer_pages ───────
        pages = await self._repo.get_submission_pages(request.submission_id)

        # ── 3c. Load template image (if loader provided) ──────────
        template_image: Optional[bytes] = None
        if template_loader is not None:
            try:
                template_image = await template_loader(request.exam_id)
                if template_image:
                    logger.info(
                        "Template image loaded for exam %s (%d bytes).",
                        request.exam_id,
                        len(template_image),
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load template image for exam %s: %s. "
                    "Proceeding without template overlay.",
                    request.exam_id,
                    exc,
                )

        # ── 4. Recognize (template overlay + LLM Vision OCR) ──────
        recognitions = await self._recognizer.recognize_submission(
            pages, question_regions_by_page, template_image=template_image
        )

        # ── 5. Gate fallback for low-confidence (DCR-01) ────────────
        recognitions = await self._apply_gate_fallback(
            recognitions, answer_keys, submission
        )

        # ── 6. Match ────────────────────────────────────────────────
        match_outputs = self._matcher.match_batch(recognitions, answer_keys)

        # ── 7. Store results (I-DCR-01) ─────────────────────────────
        results: List[DCRResult] = []
        question_results: List[DCRQuestionResult] = []
        errors: List[Dict[str, Any]] = []

        for mo in match_outputs:
            # Find if gate was used for this question
            gate_used = self._was_gate_used(mo.question_id, recognitions)

            audit_entry = DCRAuditEntry(
                action=(
                    AuditAction.GATE_FALLBACK
                    if gate_used
                    else AuditAction.ENGINE_SCORED
                ),
                actor="engine",
                new_score=mo.score,
                new_match_type=mo.match_type,
                gate_call_ref=gate_used,
                occurred_at=now,
            )

            result = DCRResult(
                exam_id=request.exam_id,
                student_id=request.student_id,
                question_id=mo.question_id,
                recognized_text=mo.recognized_text,
                confidence=mo.confidence,
                match_type=mo.match_type,
                score=mo.score,
                max_score=mo.max_score,
                audit_trail=[audit_entry],
                created_at=now,
                updated_at=now,
            )
            results.append(result)

            question_results.append(
                DCRQuestionResult(
                    question_id=mo.question_id,
                    recognized_text=mo.recognized_text,
                    confidence=mo.confidence,
                    match_type=mo.match_type,
                    score=mo.score,
                    max_score=mo.max_score,
                    used_gate_fallback=gate_used is not None,
                )
            )

        # Persist batch
        try:
            persisted = await self._repo.upsert_results_batch(results)
            logger.info(
                "DCR results persisted: %d/%d for exam=%s student=%s",
                persisted,
                len(results),
                request.exam_id,
                request.student_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to persist DCR results for exam=%s student=%s: %s",
                request.exam_id,
                request.student_id,
                exc,
            )
            errors.append({
                "stage": "persist",
                "error": str(exc),
            })

        # ── 8. Build response ───────────────────────────────────────
        total_score = sum(qr.score for qr in question_results)
        total_max = sum(qr.max_score for qr in question_results)

        return DCREvaluateResponse(
            submission_id=request.submission_id,
            exam_id=request.exam_id,
            student_id=request.student_id,
            results=question_results,
            total_score=total_score,
            total_max_score=total_max,
            evaluated_at=now,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Read-only accessors (delegate to repository)
    # ------------------------------------------------------------------

    async def get_results(
        self,
        exam_id: str,
        student_id: str,
    ) -> List[DCRResult]:
        """Fetch all stored DCR results for an exam + student pair."""
        return await self._repo.get_results_for_exam_student(exam_id, student_id)

    async def get_total_score(
        self,
        exam_id: str,
        student_id: str,
    ) -> Dict[str, float]:
        """Aggregate total score for an exam + student."""
        return await self._repo.get_total_score(exam_id, student_id)

    # ------------------------------------------------------------------
    # Gate fallback (DCR-01 / I-DCR-02)
    # ------------------------------------------------------------------

    async def _apply_gate_fallback(
        self,
        recognitions: List[RecognitionOutput],
        answer_keys: List[AnswerKey],
        submission: DCRSubmission,
    ) -> List[RecognitionOutput]:
        """
        For low-confidence recognitions, optionally invoke the LLM gate.

        This is the *only* place in DCR where an LLM call can occur (C4).
        The default path skips this entirely (U-DCR-03).

        When enabled and gate is available:
          - Build a prompt containing the expected answer and context
          - Call the gate with caller_id = "dcr_ai" (LLM_GATE_SPEC.md §5)
          - Replace recognized_text and upgrade confidence if the gate
            returns a viable result

        All gate calls are logged as audit trail entries (I-DCR-02).
        """
        if not self._enable_gate_fallback or self._gate is None:
            return recognitions

        key_map = {ak.question_id: ak for ak in answer_keys}
        updated: List[RecognitionOutput] = []
        # Track gate refs for audit trail assembly later
        self._gate_call_refs: Dict[str, str] = {}

        for rec in recognitions:
            if not self._recognizer.is_low_confidence(rec):
                updated.append(rec)
                continue

            ak = key_map.get(rec.question_id)
            if ak is None:
                updated.append(rec)
                continue

            # Build gate prompt
            prompt = self._build_gate_prompt(rec, ak)

            try:
                gate_response = await self._gate.call(
                    model_id=self._gate_model_id,
                    prompt=prompt,
                    caller_id="dcr_ai",
                    max_output_tokens=256,
                    metadata={
                        "exam_id": submission.exam_id,
                        "student_id": submission.student_id,
                        "question_id": rec.question_id,
                    },
                )

                # gate_response is a GateResponse with .content and .usage
                gate_text = gate_response.content.strip()
                if gate_text:
                    original_conf = rec.confidence
                    # Build a gate call reference from the usage metadata
                    # for audit trail traceability (I-DCR-02).
                    gate_ref = (
                        f"dcr_ai:{rec.question_id}:"
                        f"{gate_response.usage.timestamp.isoformat()}"
                        if hasattr(gate_response, "usage")
                        and hasattr(gate_response.usage, "timestamp")
                        else f"dcr_ai:{rec.question_id}"
                    )
                    rec = RecognitionOutput(
                        question_id=rec.question_id,
                        recognized_text=gate_text,
                        confidence=0.80,  # Gate-assisted confidence cap
                        page_number=rec.page_number,
                        raw_logits=None,
                    )
                    self._gate_call_refs[rec.question_id] = gate_ref
                    logger.info(
                        "DCR gate fallback used for question %s "
                        "(original confidence %.3f → gate-assisted).",
                        rec.question_id,
                        original_conf,
                    )
            except Exception as exc:
                logger.warning(
                    "DCR gate fallback failed for question %s: %s. "
                    "Keeping original low-confidence recognition.",
                    rec.question_id,
                    exc,
                )

            updated.append(rec)

        return updated

    def _was_gate_used(
        self,
        question_id: str,
        recognitions: List[RecognitionOutput],
    ) -> Optional[str]:
        """
        Return the gate call reference if the gate was used for this question,
        otherwise ``None``.
        """
        refs = getattr(self, "_gate_call_refs", {})
        return refs.get(question_id)

    @staticmethod
    def _build_gate_prompt(
        recognition: RecognitionOutput,
        answer_key: AnswerKey,
    ) -> str:
        """
        Build a prompt for the LLM gate fallback.

        The prompt provides the low-confidence HWR output and asks the LLM
        to determine the most likely intended answer.  This is used only
        when the deterministic path confidence is below threshold (DCR-01).
        """
        return (
            "A handwriting recognition system produced the following output "
            "with low confidence.\n\n"
            f"Recognized text: \"{recognition.recognized_text}\"\n"
            f"Confidence: {recognition.confidence:.3f}\n"
            f"Expected answer format: similar to \"{answer_key.expected_text}\"\n\n"
            "Based on the recognized text and the expected answer format, "
            "what is the most likely intended answer? "
            "Respond with ONLY the corrected answer text, nothing else."
        )

    # ------------------------------------------------------------------
    # Index management delegation
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Ensure DCR collection indexes exist."""
        await self._repo.ensure_indexes()
