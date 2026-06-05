"""
PCR Eval Core — Core evaluation orchestration from detected responses
to scored evaluations.

Implements the evaluation core pipeline from PCR_EVAL_ENGINE_SPEC §5:

    1. Fetch detected responses from storage (server-side, I-TAMP-02)
    2. For each response: check blocking flags → if yes, skip auto-eval,
       mark for teacher review (I-PCR-02)
    3. Look up solution cache → hit: use cached reference solution;
       miss: generate via gate (pcr_cache_warmup)
    4. Route by complexity: L1 (Haiku), L2 (Sonnet), L3 (Opus) —
       from question metadata (PCR_EVAL_ENGINE_SPEC §5.2)
    5. Build eval prompt using subject template family (§5.3)
    6. Call LLM gate with pcr_eval_core caller_id (C4)
    7. Parse response → extract step_marks[], feedback, total_score
    8. Store evaluation result with token usage and audit trail

All LLM calls go through the gate with registered caller IDs (C4):
    - ``pcr_eval_core``    — evaluation calls
    - ``pcr_cache_warmup`` — solution generation (via SolutionCache)

Spec authority:   new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md §5
Gate authority:   new-docs/architecture/LLM_GATE_SPEC.md §5
Integrity:        new-docs/architecture/TAMPER_PROOF_SPEC.md (Layer 2, 3)
Failure modes:    PCR-01 (flags + review), PCR-02 (clubbed),
                  PCR-03 (diagram), PCR-04 (practice boundary),
                  GATE-01 (budget exhaustion)
Test IDs:         U-EVAL-01 (eval result parsing and scoring envelope),
                  I-PCR-01 (artifact → PageOCR → responses),
                  I-PCR-02 (blocking flags prevent auto-eval),
                  I-PCR-03 (practice stateless, no new persistence)
Hard constraints: C1 (MongoDB only), C3 (practice untouched),
                  C4 (gate), C5 (ownership boundaries)
"""

from __future__ import annotations

import json
import logging
import importlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..domain.content_classifier import compute_scoreable_marks
from ..domain.response_models import ContentType, FlagSeverity, FlagType

from .solution_cache import CacheLookupResult, SolutionCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — decouple from concrete implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class ResponseReader(Protocol):
    """Protocol for reading detected responses.

    Satisfied by ``pcr.storage.response_repo.DetectedResponseRepository``.
    """

    async def get_response(
        self, response_id: str
    ) -> Optional[Dict[str, Any]]:
        ...  # pragma: no cover

    async def get_responses_by_submission(
        self, submission_id: str
    ) -> List[Dict[str, Any]]:
        ...  # pragma: no cover

    async def update_eval_status(
        self, response_id: str, eval_status: str
    ) -> bool:
        ...  # pragma: no cover


@runtime_checkable
class EvaluationWriter(Protocol):
    """Protocol for persisting evaluation results.

    Satisfied by ``pcr.storage.evaluation_repo.EvaluationRepository``.
    """

    async def insert_evaluation(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        ...  # pragma: no cover


@runtime_checkable
class QuestionReader(Protocol):
    """Protocol for reading question metadata.

    Satisfied by ``pcr.storage.question_repo.QuestionRepository``.
    """

    async def get_question(
        self, question_id: str
    ) -> Optional[Dict[str, Any]]:
        ...  # pragma: no cover


@runtime_checkable
class GateProtocol(Protocol):
    """Protocol for the LLM gate.

    Satisfied by ``llm_gate.gate.LLMGate``.
    """

    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Complexity router — model selection (PCR_EVAL_ENGINE_SPEC §5.2)
# ---------------------------------------------------------------------------

# Model mapping for evaluation calls.
# cache-hit always uses Haiku-class for compare-only.
# L1/L2/L3 tiers are used on cache miss.
EVAL_MODEL_MAP: Dict[str, str] = {
    "cache_hit": "claude-haiku-4-20250514",   # Compare-only (any cache hit)
    "L1":        "claude-haiku-4-20250514",   # Factual / single-step
    "L2":        "claude-sonnet-4-20250514",  # Multi-step short answer
    "L3":        "claude-sonnet-4-20250514",  # Essay / open-ended
}

DEFAULT_EVAL_MODEL: str = "claude-sonnet-4-20250514"


def _get_gate_provider_default_model() -> str:
    """Resolve the active gate provider's default model."""
    import_errors: list[str] = []
    for module_name in (
        "exam-conductor.llm_gate.provider",
        "llm_gate.provider",
    ):
        try:
            provider = importlib.import_module(module_name)
            return provider.get_default_model()
        except (ImportError, AttributeError) as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
    raise RuntimeError(
        "Gate provider default model resolver unavailable: "
        + "; ".join(import_errors)
    )


def _select_eval_model(
    complexity: str,
    cache_hit: bool,
) -> str:
    """Select the LLM model based on complexity tier and cache status.

    From PCR_EVAL_ENGINE_SPEC §5.2:
        cache-hit  → Haiku-class compare-only
        L1 miss    → Haiku-class
        L2 miss    → Sonnet-class
        L3 miss    → Sonnet/Opus-class
    """
    active_provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if active_provider != "anthropic":
        return _get_gate_provider_default_model()
    if cache_hit:
        return EVAL_MODEL_MAP.get("cache_hit", DEFAULT_EVAL_MODEL)
    return EVAL_MODEL_MAP.get(complexity, DEFAULT_EVAL_MODEL)


# ---------------------------------------------------------------------------
# Template families (PCR_EVAL_ENGINE_SPEC §5.3)
# ---------------------------------------------------------------------------

TEMPLATE_FAMILIES: Dict[str, str] = {
    "stepwise_numerical": """You are an expert exam evaluator for {subject}.

You are evaluating a student's response to a numerical/calculation question.

Question (worth {max_marks} marks):
{question_text}

Reference Solution:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors — be tolerant of l/1, O/0, rn/m, missing superscripts):
{student_response}

Evaluate the student's response step by step. For each step, assign partial marks.

IMPORTANT: Be tolerant of common OCR/HWR errors such as:
- 'l' mistaken for '1' or vice versa
- 'O' mistaken for '0'
- 'rn' mistaken for 'm'
- Missing or malformed superscripts/subscripts
- Devanagari matra errors

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "description of step", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",

    "essay_rubric": """You are an expert exam evaluator for {subject}.

You are evaluating a student's essay/long-form answer.

Question (worth {max_marks} marks):
{question_text}

Reference Solution / Key Points:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors):
{student_response}

Evaluate against the rubric criteria. Award marks for content, organization, expression, and completeness.

IMPORTANT: Be tolerant of common OCR/HWR errors.

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "Content & Key Points", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}},
    {{"step": "Organization & Structure", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}},
    {{"step": "Expression & Language", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}},
    {{"step": "Completeness", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",

    "factual_recall": """You are an expert exam evaluator for {subject}.

You are evaluating a short factual answer.

Question (worth {max_marks} marks):
{question_text}

Reference Solution:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors):
{student_response}

Evaluate whether the student's response contains the key factual points.

IMPORTANT: Be tolerant of common OCR/HWR errors.

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "key point description", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",

    "keyword_coverage": """You are an expert exam evaluator for {subject}.

You are evaluating a definition or process-based answer.

Question (worth {max_marks} marks):
{question_text}

Reference Solution / Key Terms:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors):
{student_response}

Evaluate based on coverage of key terms and concepts. Award marks for each key term/concept present and correctly explained.

IMPORTANT: Be tolerant of common OCR/HWR errors.

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "keyword/concept", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",

    "ledger_tabular": """You are an expert exam evaluator for {subject} (Accountancy).

You are evaluating a ledger, journal entry, or tabular answer.

Question (worth {max_marks} marks):
{question_text}

Reference Solution:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors — tables may have alignment issues):
{student_response}

Evaluate the entries, totals, and format. Award partial marks for correct entries even if formatting is imperfect.

IMPORTANT: Be tolerant of OCR errors, especially in tabular alignment and number recognition.

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "entry/section", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",

    "proof_derivation": """You are an expert exam evaluator for {subject}.

You are evaluating a mathematical proof or derivation.

Question (worth {max_marks} marks):
{question_text}

Reference Solution:
{reference_solution}

Student Response (OCR-extracted, may contain OCR errors — mathematical notation may be garbled):
{student_response}

Evaluate the logical progression, correct use of theorems/formulas, and final result. Award marks for each logical step that is correct, even if notation is imperfect due to OCR.

IMPORTANT: Be especially tolerant of OCR errors in mathematical notation (superscripts, subscripts, Greek letters, operators).

Respond in the following JSON format ONLY (no other text):
{{
  "step_marks": [
    {{"step": "logical step description", "marks_awarded": 0.0, "max_marks": 0.0, "rationale": "why"}}
  ],
  "total_score": 0.0,
  "max_score": {max_marks},
  "overall_feedback": "brief feedback for the student"
}}""",
}

# Default template when eval_template is not specified or unrecognized
DEFAULT_TEMPLATE = "factual_recall"


# ---------------------------------------------------------------------------
# Eval result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StepMark:
    """A single step in the step-wise marking breakdown."""

    step: str
    marks_awarded: float
    max_marks: float
    rationale: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single detected response.

    Attributes
    ----------
    evaluation_id : str
        Unique evaluation identifier.
    response_id : str
        The evaluated response.
    question_id : str | None
        The question this response is associated with.
    student_id : str
        Student who wrote the response.
    eval_path : str
        Evaluation path taken (``cache_hit``, ``L1``, ``L2``, ``L3``).
    model_used : str
        LLM model used for evaluation.
    total_score : float
        Total marks awarded.
    max_score : float
        Maximum possible marks for the question.
    scoreable_max : float
        Prorated scoreable marks (reduced if diagram excluded).
    step_marks : list[StepMark]
        Step-by-step marking breakdown.
    overall_feedback : str
        Feedback text for the student.
    reference_solution : str | None
        The reference solution used for comparison.
    token_usage : dict
        Token accounting from the gate call.
    raw_llm_response : str
        Raw LLM output for audit purposes.
    skipped : bool
        True if evaluation was skipped (blocking flags).
    skip_reason : str | None
        Reason for skipping (if applicable).
    error : str | None
        Error message if evaluation failed.
    """

    evaluation_id: str
    response_id: str
    question_id: Optional[str] = None
    student_id: str = ""
    eval_path: str = ""
    model_used: str = ""
    total_score: float = 0.0
    max_score: float = 0.0
    scoreable_max: float = 0.0
    step_marks: List[StepMark] = field(default_factory=list)
    overall_feedback: str = ""
    reference_solution: Optional[str] = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    raw_llm_response: str = ""
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Batch result envelope
# ---------------------------------------------------------------------------


@dataclass
class BatchEvalResult:
    """Result of evaluating all responses for a submission.

    Attributes
    ----------
    submission_id : str
        The submission that was evaluated.
    total_responses : int
        Total responses in the submission.
    evaluated_count : int
        Number of responses successfully evaluated.
    blocked_count : int
        Number of responses blocked by flags.
    error_count : int
        Number of responses that failed evaluation.
    results : list[EvalResult]
        Individual evaluation results.
    """

    submission_id: str = ""
    total_responses: int = 0
    evaluated_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    results: List[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------


def _parse_eval_response(raw: str, max_score: float) -> Dict[str, Any]:
    """Parse the LLM evaluation response JSON.

    Tolerates markdown code fences and leading/trailing whitespace.

    Returns a dict with keys: step_marks, total_score, max_score,
    overall_feedback.  On parse failure, returns a degraded result.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (possibly with language tag)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed content
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM eval response as JSON")
                return {
                    "step_marks": [],
                    "total_score": 0.0,
                    "max_score": max_score,
                    "overall_feedback": "Evaluation response could not be parsed",
                    "parse_error": True,
                }
        else:
            logger.warning("No JSON found in LLM eval response")
            return {
                "step_marks": [],
                "total_score": 0.0,
                "max_score": max_score,
                "overall_feedback": "Evaluation response could not be parsed",
                "parse_error": True,
            }

    # Validate and clamp total_score
    total_score = float(parsed.get("total_score", 0.0))
    total_score = max(0.0, min(total_score, max_score))

    # Parse step marks
    step_marks_raw = parsed.get("step_marks", [])
    step_marks = []
    for sm in step_marks_raw:
        if isinstance(sm, dict):
            step_marks.append({
                "step": sm.get("step", ""),
                "marks_awarded": float(sm.get("marks_awarded", 0.0)),
                "max_marks": float(sm.get("max_marks", 0.0)),
                "rationale": sm.get("rationale", ""),
            })

    return {
        "step_marks": step_marks,
        "total_score": total_score,
        "max_score": float(parsed.get("max_score", max_score)),
        "overall_feedback": parsed.get("overall_feedback", ""),
        "parse_error": False,
    }


# ---------------------------------------------------------------------------
# Eval Core
# ---------------------------------------------------------------------------


class EvalCore:
    """Core evaluation orchestrator for PCR.

    Coordinates the flow from detected responses to scored evaluations:
    server-side response fetch → blocking flag check → solution cache
    lookup → complexity routing → LLM gate call → parse → persist.

    All LLM calls go through the gate with ``pcr_eval_core`` caller_id.
    Solution cache warmup uses ``pcr_cache_warmup`` via ``SolutionCache``.

    Parameters
    ----------
    response_repo : ResponseReader
        For reading detected responses (server-side fetch).
    eval_repo : EvaluationWriter
        For persisting evaluation results.
    question_repo : QuestionReader
        For reading question metadata (complexity, template, marks).
    solution_cache : SolutionCache
        For solution cache lookup and warmup.
    gate : GateProtocol
        LLM gate for evaluation calls.
    eval_models : dict[str, str] | None
        Override complexity-to-model mapping for evaluation.

    Usage
    -----
    ::

        core = EvalCore(
            response_repo=response_repo,
            eval_repo=eval_repo,
            question_repo=question_repo,
            solution_cache=solution_cache,
            gate=gate,
        )
        result = await core.evaluate_response("RESP-abc123")
        batch = await core.evaluate_submission("submission-id-456")
    """

    CALLER_ID = "pcr_eval_core"
    """Registered gate caller identity for evaluation (LLM_GATE_SPEC §5)."""

    def __init__(
        self,
        response_repo: ResponseReader,
        eval_repo: EvaluationWriter,
        question_repo: QuestionReader,
        solution_cache: SolutionCache,
        gate: GateProtocol,
        *,
        eval_models: Optional[Dict[str, str]] = None,
    ) -> None:
        self._responses = response_repo
        self._evals = eval_repo
        self._questions = question_repo
        self._cache = solution_cache
        self._gate = gate
        self._eval_models = eval_models or EVAL_MODEL_MAP

    # ------------------------------------------------------------------
    # Single response evaluation
    # ------------------------------------------------------------------

    async def evaluate_response(
        self,
        response_id: str,
        *,
        question_id: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> EvalResult:
        """Evaluate a single detected response.

        Steps (PCR_EVAL_ENGINE_SPEC §5):
        1. Fetch response from storage (server-side, I-TAMP-02)
        2. Validate question_id matches stored response (if provided)
        3. Check blocking flags → skip if present (I-PCR-02)
        4. Fetch question metadata
        5. Solution cache lookup
        6. Complexity routing → model selection
        7. Build eval prompt from template family
        8. Call gate with pcr_eval_core
        9. Parse response → step_marks, feedback, total_score
        10. Store evaluation result

        Parameters
        ----------
        response_id : str
            The detected response to evaluate.
        question_id : str, optional
            Expected question_id from the wire request. If provided and
            it does not match the stored response's question_id, the
            request is rejected (tamper-proof validation).
        student_id : str, optional
            Override student_id (if not in response doc).

        Returns
        -------
        EvalResult
            Complete evaluation result.
        """
        eval_id = f"EVAL-{uuid.uuid4().hex[:12]}"

        # Step 1: Server-side fetch (TAMPER_PROOF_SPEC Layer 2)
        response_doc = await self._responses.get_response(response_id)
        if response_doc is None:
            logger.error("Response %s not found in storage", response_id)
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                error=f"Response {response_id} not found",
            )
        if response_doc.get("eval_status") == "superseded":
            logger.info("Response %s is superseded; skipping evaluation", response_id)
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                error=f"Response {response_id} has been superseded",
            )

        resolved_student_id = student_id or response_doc.get("student_id", "")
        stored_question_id = response_doc.get("question_id")

        # Validate question_id if provided by the caller (tamper-proof cross-check)
        if question_id and stored_question_id and question_id != stored_question_id:
            logger.warning(
                "question_id mismatch: wire=%s stored=%s for response=%s",
                question_id, stored_question_id, response_id,
            )
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                question_id=stored_question_id,
                error=(
                    f"question_id mismatch: request specified {question_id} "
                    f"but response {response_id} belongs to {stored_question_id}"
                ),
            )

        question_id = stored_question_id
        detected_text = response_doc.get("detected_text", "")
        content_type = response_doc.get("content_type", "TEXT_ONLY")
        flags = response_doc.get("flags", [])

        # Step 2: Check blocking flags (I-PCR-02, PCR_EVAL_ENGINE_SPEC §6.3)
        has_blocking = any(
            (f.get("severity") == "blocking" or
             f.get("severity") == FlagSeverity.BLOCKING.value)
            for f in flags
        )

        if has_blocking:
            logger.info(
                "Response %s has blocking flags — skipping auto-eval "
                "(I-PCR-02)",
                response_id,
            )
            await self._responses.update_eval_status(
                response_id, "blocked"
            )
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                question_id=question_id,
                student_id=resolved_student_id,
                skipped=True,
                skip_reason="Blocking flags present — routed to teacher review",
            )

        # Step 3: Fetch question metadata
        question_doc: Dict[str, Any] = {}
        if question_id:
            question_doc = (
                await self._questions.get_question(question_id) or {}
            )

        max_marks = float(question_doc.get("max_marks", 10.0))
        complexity = question_doc.get("complexity", "L2")
        eval_template = question_doc.get("eval_template", DEFAULT_TEMPLATE)
        subject = question_doc.get("subject", "General")
        question_text = question_doc.get(
            "question_text", "(question text not available)"
        )
        diagram_weight = float(question_doc.get("diagram_weight", 0.0))

        # Compute scoreable marks (prorating for diagram exclusion)
        scoreable_max = max_marks
        if content_type == ContentType.MIXED.value and diagram_weight > 0:
            scoreable_max = compute_scoreable_marks(max_marks, diagram_weight)

        # Step 4: Solution cache lookup
        cache_result: CacheLookupResult = await self._cache.lookup(
            question_id=question_id or response_id,
            question_metadata=question_doc if question_doc else {
                "subject": subject,
                "question_type": "subjective",
                "max_marks": max_marks,
                "question_text": question_text,
                "complexity": complexity,
            },
        )
        reference_solution = cache_result.reference_solution or ""

        # Step 5: Complexity routing (PCR_EVAL_ENGINE_SPEC §5.2)
        eval_path = "cache_hit" if cache_result.hit else complexity
        model_id = _select_eval_model(complexity, cache_result.hit)

        # Step 6: Build eval prompt from template family (§5.3)
        template = TEMPLATE_FAMILIES.get(eval_template)
        if template is None:
            logger.warning(
                "Unknown eval_template %r for question %s — "
                "using %s",
                eval_template,
                question_id,
                DEFAULT_TEMPLATE,
            )
            template = TEMPLATE_FAMILIES[DEFAULT_TEMPLATE]

        prompt = template.format(
            subject=subject,
            max_marks=scoreable_max,
            question_text=question_text,
            reference_solution=reference_solution,
            student_response=detected_text,
        )

        # Step 7: Call gate with pcr_eval_core (C4)
        try:
            gate_response = await self._gate.call(
                model_id=model_id,
                prompt=prompt,
                caller_id=self.CALLER_ID,
                max_output_tokens=2048,
                temperature=0.1,
                metadata={
                    "response_id": response_id,
                    "question_id": question_id,
                    "eval_path": eval_path,
                    "eval_template": eval_template,
                },
            )
        except Exception as exc:
            logger.exception(
                "Gate call failed for response %s", response_id
            )
            await self._responses.update_eval_status(
                response_id, "manual_review"
            )
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                question_id=question_id,
                student_id=resolved_student_id,
                eval_path=eval_path,
                model_used=model_id,
                max_score=max_marks,
                scoreable_max=scoreable_max,
                error=f"Gate call failed: {exc}",
            )

        raw_llm = gate_response.content
        token_usage = {
            "model": gate_response.usage.model,
            "caller": gate_response.usage.caller,
            "input_tokens": gate_response.usage.input_tokens,
            "output_tokens": gate_response.usage.output_tokens,
            "total_tokens": gate_response.usage.total_tokens,
            "estimated_cost_usd": gate_response.usage.estimated_cost_usd,
        }

        # Step 8: Parse response
        parsed = _parse_eval_response(raw_llm, scoreable_max)

        step_marks = [
            StepMark(
                step=sm.get("step", ""),
                marks_awarded=sm.get("marks_awarded", 0.0),
                max_marks=sm.get("max_marks", 0.0),
                rationale=sm.get("rationale", ""),
            )
            for sm in parsed.get("step_marks", [])
        ]

        total_score = parsed.get("total_score", 0.0)
        overall_feedback = parsed.get("overall_feedback", "")

        # Check for score divergence (flag if significantly off)
        eval_flags: List[Dict[str, Any]] = []
        if parsed.get("parse_error"):
            eval_flags.append({
                "flag_type": FlagType.LLM_SCORE_DIVERGENCE.value,
                "severity": FlagSeverity.WARNING.value,
                "reason": "LLM response could not be fully parsed",
            })

        # Partial eval diagram exclusion flag
        if content_type == ContentType.MIXED.value and diagram_weight > 0:
            eval_flags.append({
                "flag_type": FlagType.PARTIAL_EVAL_DIAGRAM_EXCLUDED.value,
                "severity": FlagSeverity.INFO.value,
                "reason": (
                    f"Diagram portion excluded; "
                    f"scoreable_max prorated from {max_marks} to {scoreable_max}"
                ),
            })

        # Step 9: Store evaluation result
        eval_doc: Dict[str, Any] = {
            "evaluation_id": eval_id,
            "response_id": response_id,
            "question_id": question_id,
            "student_id": resolved_student_id,
            "eval_path": eval_path,
            "model_used": gate_response.usage.model,
            "total_score": total_score,
            "max_score": max_marks,
            "scoreable_max": scoreable_max,
            "step_marks": [
                {
                    "step": sm.step,
                    "marks_awarded": sm.marks_awarded,
                    "max_marks": sm.max_marks,
                    "rationale": sm.rationale,
                }
                for sm in step_marks
            ],
            "overall_feedback": overall_feedback,
            "reference_solution": reference_solution,
            "token_usage": token_usage,
            "raw_llm_response": raw_llm,
            "eval_flags": eval_flags,
            "audit_trail": [
                {
                    "actor_id": "system",
                    "timestamp": datetime.now(timezone.utc),
                    "action": "evaluation_created",
                    "before": None,
                    "after": {
                        "total_score": total_score,
                        "max_score": max_marks,
                        "scoreable_max": scoreable_max,
                        "eval_path": eval_path,
                        "model_used": gate_response.usage.model,
                    },
                    "reason": "Automated PCR evaluation",
                },
            ],
            "created_at": datetime.now(timezone.utc),
        }

        try:
            await self._evals.insert_evaluation(eval_doc)
        except Exception:
            logger.exception(
                "Failed to persist evaluation %s for response %s",
                eval_id,
                response_id,
            )
            # Do NOT lose the eval result — return it even if persistence fails
            return EvalResult(
                evaluation_id=eval_id,
                response_id=response_id,
                question_id=question_id,
                student_id=resolved_student_id,
                eval_path=eval_path,
                model_used=gate_response.usage.model,
                total_score=total_score,
                max_score=max_marks,
                scoreable_max=scoreable_max,
                step_marks=step_marks,
                overall_feedback=overall_feedback,
                reference_solution=reference_solution,
                token_usage=token_usage,
                raw_llm_response=raw_llm,
                error="Evaluation completed but persistence failed",
            )

        # Update response eval_status
        has_warnings = any(
            f.get("severity") == FlagSeverity.WARNING.value
            for f in flags
        ) or eval_flags
        eval_status = (
            "evaluated_with_warnings" if has_warnings else "evaluated"
        )
        await self._responses.update_eval_status(response_id, eval_status)

        logger.info(
            "Evaluated response %s: %.1f/%.1f (path=%s, model=%s, "
            "tokens=%d)",
            response_id,
            total_score,
            max_marks,
            eval_path,
            gate_response.usage.model,
            gate_response.usage.total_tokens,
        )

        return EvalResult(
            evaluation_id=eval_id,
            response_id=response_id,
            question_id=question_id,
            student_id=resolved_student_id,
            eval_path=eval_path,
            model_used=gate_response.usage.model,
            total_score=total_score,
            max_score=max_marks,
            scoreable_max=scoreable_max,
            step_marks=step_marks,
            overall_feedback=overall_feedback,
            reference_solution=reference_solution,
            token_usage=token_usage,
            raw_llm_response=raw_llm,
        )

    # ------------------------------------------------------------------
    # Submission-level batch evaluation
    # ------------------------------------------------------------------

    async def evaluate_submission(
        self,
        submission_id: str,
        *,
        student_id: Optional[str] = None,
    ) -> BatchEvalResult:
        """Evaluate all detected responses for a submission.

        Fetches all detected responses from storage (server-side fetch),
        evaluates each one, and returns a batch result.  Responses with
        blocking flags are skipped and counted separately.

        Budget exhaustion mid-batch is handled gracefully — already-
        evaluated responses are preserved and the batch returns with
        partial results and an error count.

        Parameters
        ----------
        submission_id : str
            The submission whose responses to evaluate.
        student_id : str, optional
            Override student_id for all responses.

        Returns
        -------
        BatchEvalResult
            Aggregate result with individual evaluation details.
        """
        responses = await self._responses.get_responses_by_submission(
            submission_id
        )

        if not responses:
            logger.warning(
                "No detected responses found for submission %s",
                submission_id,
            )
            return BatchEvalResult(
                submission_id=submission_id,
                total_responses=0,
            )

        batch = BatchEvalResult(
            submission_id=submission_id,
            total_responses=len(responses),
        )

        for resp_doc in responses:
            response_id = resp_doc.get("response_id", "")
            resolved_student_id = (
                student_id or resp_doc.get("student_id", "")
            )

            try:
                result = await self.evaluate_response(
                    response_id,
                    student_id=resolved_student_id,
                )

                if result.skipped:
                    batch.blocked_count += 1
                elif result.error:
                    batch.error_count += 1
                else:
                    batch.evaluated_count += 1

                batch.results.append(result)

            except Exception as exc:
                logger.exception(
                    "Unhandled error evaluating response %s in "
                    "submission %s",
                    response_id,
                    submission_id,
                )
                batch.error_count += 1
                batch.results.append(
                    EvalResult(
                        evaluation_id=f"EVAL-{uuid.uuid4().hex[:12]}",
                        response_id=response_id,
                        student_id=resolved_student_id,
                        error=f"Unhandled error: {exc}",
                    )
                )

        logger.info(
            "Batch eval for submission %s: %d/%d evaluated, "
            "%d blocked, %d errors",
            submission_id,
            batch.evaluated_count,
            batch.total_responses,
            batch.blocked_count,
            batch.error_count,
        )

        return batch

    # ------------------------------------------------------------------
    # Practice evaluation (stateless — C3, PCR-04)
    # ------------------------------------------------------------------

    async def evaluate_practice(
        self,
        *,
        student_response: str,
        question_id: str,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate a practice response statelessly.

        Practice mode (PCR_EVAL_ENGINE_SPEC §2.2, §8.2):
        - Request → evaluate → return
        - No new ``evalpen_submissions``
        - No immutable practice artifact store
        - Token logging may still occur through the gate
        - NO persistence of evaluation results (C3, PCR-04)

        The student_response is accepted directly (not from server-side
        storage) because practice mode is stateless and not subject to
        the conducted-exam tamper-proofing requirements.

        Parameters
        ----------
        student_response : str
            The student's answer text (client-supplied is OK for practice).
        question_id : str
            The question being practiced.
        question_metadata : dict, optional
            Question metadata override.  If not provided, fetched from
            ``evalpen_questions``.

        Returns
        -------
        EvalResult
            Evaluation result (NOT persisted — C3, PCR-04).
        """
        eval_id = f"PRACTICE-{uuid.uuid4().hex[:12]}"

        # Fetch question metadata if not provided
        q_meta = question_metadata
        if q_meta is None and question_id:
            q_meta = await self._questions.get_question(question_id) or {}
        q_meta = q_meta or {}

        max_marks = float(q_meta.get("max_marks", 10.0))
        complexity = q_meta.get("complexity", "L2")
        eval_template = q_meta.get("eval_template", DEFAULT_TEMPLATE)
        subject = q_meta.get("subject", "General")
        question_text = q_meta.get(
            "question_text", "(question text not available)"
        )

        # Solution cache lookup (solutions may be shared — they are NOT
        # new practice persistence, just read access)
        cache_result = await self._cache.lookup(
            question_id=question_id,
            question_metadata=q_meta,
        )
        reference_solution = cache_result.reference_solution or ""

        # Model selection
        eval_path = "cache_hit" if cache_result.hit else complexity
        model_id = _select_eval_model(complexity, cache_result.hit)

        # Build prompt
        template = TEMPLATE_FAMILIES.get(
            eval_template, TEMPLATE_FAMILIES[DEFAULT_TEMPLATE]
        )
        prompt = template.format(
            subject=subject,
            max_marks=max_marks,
            question_text=question_text,
            reference_solution=reference_solution,
            student_response=student_response,
        )

        # Gate call with pcr_practice — the spec (LLM_GATE_SPEC §5) reserves
        # pcr_practice for the stateless practice path, separate from
        # pcr_eval_core used for conducted-exam evaluation. This ensures
        # correct token attribution and distinguishable usage tracking.
        gate_response = await self._gate.call(
            model_id=model_id,
            prompt=prompt,
            caller_id="pcr_practice",
            max_output_tokens=2048,
            temperature=0.1,
            metadata={
                "question_id": question_id,
                "eval_path": eval_path,
                "mode": "practice",
            },
        )

        raw_llm = gate_response.content
        parsed = _parse_eval_response(raw_llm, max_marks)

        step_marks = [
            StepMark(
                step=sm.get("step", ""),
                marks_awarded=sm.get("marks_awarded", 0.0),
                max_marks=sm.get("max_marks", 0.0),
                rationale=sm.get("rationale", ""),
            )
            for sm in parsed.get("step_marks", [])
        ]

        # Practice results are NOT persisted (C3, PCR-04)
        return EvalResult(
            evaluation_id=eval_id,
            response_id="practice",
            question_id=question_id,
            eval_path=eval_path,
            model_used=gate_response.usage.model,
            total_score=parsed.get("total_score", 0.0),
            max_score=max_marks,
            scoreable_max=max_marks,
            step_marks=step_marks,
            overall_feedback=parsed.get("overall_feedback", ""),
            reference_solution=reference_solution,
            token_usage={
                "model": gate_response.usage.model,
                "input_tokens": gate_response.usage.input_tokens,
                "output_tokens": gate_response.usage.output_tokens,
                "total_tokens": gate_response.usage.total_tokens,
                "estimated_cost_usd": gate_response.usage.estimated_cost_usd,
            },
            raw_llm_response=raw_llm,
        )
