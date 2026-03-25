"""
PCR Solution Cache — Cache hit/miss logic and warmup via LLM gate.

Implements the solution cache strategy from PCR_EVAL_ENGINE_SPEC §5.1:

    Cache key is question-centric.  On cache hit, PCR runs compare-only
    evaluation.  On cache miss, PCR routes to a model tier, generates or
    refreshes a reference solution, stores it, then evaluates.

All LLM calls for solution generation go through the gate with
``caller_id = pcr_cache_warmup`` (C4, LLM_GATE_SPEC §5).

Spec authority:  new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md §5.1
Gate authority:  new-docs/architecture/LLM_GATE_SPEC.md §5
Failure modes:   GATE-01 (budget exhaustion during warmup)
Test IDs:        U-EVAL-01 (eval result parsing and scoring envelope)
Hard constraints: C1 (MongoDB only), C4 (gate), C5 (ownership boundaries)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — decouple from concrete repository and gate implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class SolutionStore(Protocol):
    """Protocol for the solution repository interface.

    Satisfied by ``pcr.storage.solution_repo.SolutionRepository``.
    """

    async def get_latest_solution(
        self, question_id: str
    ) -> Optional[Dict[str, Any]]:
        ...  # pragma: no cover

    async def has_solution(self, question_id: str) -> bool:
        ...  # pragma: no cover

    async def upsert_solution(
        self, doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...  # pragma: no cover


@runtime_checkable
class GateProtocol(Protocol):
    """Protocol for the LLM gate call interface.

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
# Cache result envelope
# ---------------------------------------------------------------------------


@dataclass
class CacheLookupResult:
    """Result of a solution cache lookup.

    Attributes
    ----------
    hit : bool
        True if a cached reference solution was found.
    reference_solution : str | None
        The reference solution text (from cache or freshly generated).
    version : int
        Solution version number.
    solution_source : str
        ``"teacher"`` or ``"llm"``.
    model_used : str | None
        Model that generated the solution (for LLM-sourced solutions).
    was_generated : bool
        True if the solution was generated on this call (cache miss).
    """

    hit: bool
    reference_solution: Optional[str] = None
    version: int = 0
    solution_source: str = "llm"
    model_used: Optional[str] = None
    was_generated: bool = False


# ---------------------------------------------------------------------------
# Complexity-to-model mapping (PCR_EVAL_ENGINE_SPEC §5.2)
# ---------------------------------------------------------------------------

# Default model tiers for solution generation (cache warmup).
# These can be overridden via configuration.
DEFAULT_WARMUP_MODELS: Dict[str, str] = {
    "L1": "claude-haiku-4-20250514",     # Haiku-class for factual/single-step
    "L2": "claude-sonnet-4-20250514",    # Sonnet-class for multi-step short answer
    "L3": "claude-sonnet-4-20250514",    # Sonnet/Opus-class for essay/open-ended
}

# Fallback model when complexity is unknown
DEFAULT_WARMUP_MODEL: str = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Prompt templates for solution generation
# ---------------------------------------------------------------------------

SOLUTION_GENERATION_PROMPT = """You are an expert teacher generating a reference solution for an exam question.

Subject: {subject}
Question Type: {question_type}
Maximum Marks: {max_marks}

Question:
{question_text}

{rubric_section}

Generate a complete, well-structured reference solution that:
1. Covers all key points worth marks
2. Shows clear step-by-step working where applicable
3. Is appropriate for the marks allocation ({max_marks} marks)
4. Can be used as a benchmark to evaluate student answers

Provide the reference solution only, no additional commentary."""


# ---------------------------------------------------------------------------
# Solution Cache
# ---------------------------------------------------------------------------


class SolutionCache:
    """Question-centric solution cache with LLM gate-backed warmup.

    The cache uses ``evalpen_solutions`` (MongoDB) as its backing store.
    On cache miss, it generates a reference solution via the LLM gate
    with ``caller_id = pcr_cache_warmup`` and persists it for future hits.

    Parameters
    ----------
    solution_store : SolutionStore
        Async repository for ``evalpen_solutions``.
    gate : GateProtocol
        LLM gate instance for solution generation on cache miss.
    warmup_models : dict[str, str] | None
        Override complexity-to-model mapping.  Keys are ``L1``, ``L2``, ``L3``.

    Usage
    -----
    ::

        cache = SolutionCache(solution_repo, gate)
        result = await cache.lookup(
            question_id="Q-001",
            question_metadata=question_doc,
        )
        if result.hit:
            # compare-only evaluation with result.reference_solution
            ...
        else:
            # result.reference_solution is freshly generated
            ...
    """

    CALLER_ID = "pcr_cache_warmup"
    """Registered gate caller identity for cache warmup (LLM_GATE_SPEC §5)."""

    def __init__(
        self,
        solution_store: SolutionStore,
        gate: GateProtocol,
        *,
        warmup_models: Optional[Dict[str, str]] = None,
    ) -> None:
        self._store = solution_store
        self._gate = gate
        self._warmup_models = warmup_models or DEFAULT_WARMUP_MODELS

    # ------------------------------------------------------------------
    # Cache lookup
    # ------------------------------------------------------------------

    async def lookup(
        self,
        question_id: str,
        question_metadata: Dict[str, Any],
    ) -> CacheLookupResult:
        """Look up or generate a reference solution for a question.

        Cache hit path:
            1. Find latest solution in ``evalpen_solutions``
            2. Return it for compare-only evaluation

        Cache miss path:
            1. No solution found
            2. Generate via gate with ``pcr_cache_warmup``
            3. Store generated solution with version increment
            4. Return for evaluation

        Parameters
        ----------
        question_id : str
            The question to look up.
        question_metadata : dict
            Question document from ``evalpen_questions``.  Must include
            at minimum: ``subject``, ``question_type``, ``max_marks``.
            May include: ``complexity``, ``rubric``, ``question_text``.

        Returns
        -------
        CacheLookupResult
            Contains the reference solution (cached or generated) and
            metadata about the cache operation.
        """
        # -- Cache hit check --
        existing = await self._store.get_latest_solution(question_id)

        if existing is not None:
            logger.info(
                "Solution cache HIT for question %s (v%d, source=%s)",
                question_id,
                existing.get("version", 0),
                existing.get("solution_source", "unknown"),
            )
            return CacheLookupResult(
                hit=True,
                reference_solution=existing.get("reference_solution"),
                version=existing.get("version", 0),
                solution_source=existing.get("solution_source", "unknown"),
                model_used=existing.get("model_used"),
                was_generated=False,
            )

        # -- Cache miss: generate via gate --
        logger.info(
            "Solution cache MISS for question %s — generating via gate",
            question_id,
        )
        return await self._generate_and_store(question_id, question_metadata)

    # ------------------------------------------------------------------
    # Warmup (explicit pre-generation for a batch of questions)
    # ------------------------------------------------------------------

    async def warmup(
        self,
        questions: list[Dict[str, Any]],
    ) -> Dict[str, CacheLookupResult]:
        """Pre-warm the solution cache for a batch of questions.

        Skips questions that already have a cached solution.

        Parameters
        ----------
        questions : list[dict]
            Question metadata documents.  Each must have ``question_id``.

        Returns
        -------
        dict[str, CacheLookupResult]
            Mapping from question_id to cache result.
        """
        results: Dict[str, CacheLookupResult] = {}

        for q in questions:
            question_id = q["question_id"]

            if await self._store.has_solution(question_id):
                logger.debug(
                    "Warmup: skipping question %s (already cached)",
                    question_id,
                )
                existing = await self._store.get_latest_solution(question_id)
                if existing:
                    results[question_id] = CacheLookupResult(
                        hit=True,
                        reference_solution=existing.get("reference_solution"),
                        version=existing.get("version", 0),
                        solution_source=existing.get("solution_source", "unknown"),
                        model_used=existing.get("model_used"),
                        was_generated=False,
                    )
                continue

            try:
                result = await self._generate_and_store(question_id, q)
                results[question_id] = result
            except Exception:
                logger.exception(
                    "Warmup: failed to generate solution for question %s",
                    question_id,
                )
                results[question_id] = CacheLookupResult(
                    hit=False,
                    was_generated=False,
                )

        return results

    # ------------------------------------------------------------------
    # Internal — generate and store
    # ------------------------------------------------------------------

    async def _generate_and_store(
        self,
        question_id: str,
        question_metadata: Dict[str, Any],
    ) -> CacheLookupResult:
        """Generate a reference solution via the LLM gate and store it.

        Uses ``caller_id = pcr_cache_warmup`` for all gate calls (C4).

        Raises
        ------
        BudgetExhaustedError
            If the gate budget is exhausted (GATE-01).
        """
        # Determine model from complexity tier
        complexity = question_metadata.get("complexity", "L2")
        model_id = self._warmup_models.get(complexity, DEFAULT_WARMUP_MODEL)

        # Build generation prompt
        prompt = self._build_generation_prompt(question_metadata)

        # Call gate with pcr_cache_warmup caller_id
        gate_response = await self._gate.call(
            model_id=model_id,
            prompt=prompt,
            caller_id=self.CALLER_ID,
            max_output_tokens=2048,
            temperature=0.3,
            metadata={
                "question_id": question_id,
                "complexity": complexity,
                "operation": "solution_generation",
            },
        )

        reference_solution = gate_response.content.strip()

        # Determine next version
        existing = await self._store.get_latest_solution(question_id)
        next_version = (existing.get("version", 0) + 1) if existing else 1

        # Persist to evalpen_solutions
        solution_doc = {
            "question_id": question_id,
            "version": next_version,
            "reference_solution": reference_solution,
            "solution_source": "llm",
            "model_used": gate_response.usage.model,
            "created_at": datetime.now(timezone.utc),
        }
        await self._store.upsert_solution(solution_doc)

        logger.info(
            "Generated and stored solution v%d for question %s "
            "(model=%s, tokens=%d)",
            next_version,
            question_id,
            model_id,
            gate_response.usage.total_tokens,
        )

        return CacheLookupResult(
            hit=False,
            reference_solution=reference_solution,
            version=next_version,
            solution_source="llm",
            model_used=gate_response.usage.model,
            was_generated=True,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generation_prompt(question_metadata: Dict[str, Any]) -> str:
        """Build the solution generation prompt from question metadata."""
        subject = question_metadata.get("subject", "General")
        question_type = question_metadata.get("question_type", "subjective")
        max_marks = question_metadata.get("max_marks", 10)
        question_text = question_metadata.get(
            "question_text", "(question text not available)"
        )

        rubric = question_metadata.get("rubric")
        rubric_section = ""
        if rubric:
            if isinstance(rubric, dict):
                # Structured rubric — format as key-value pairs
                rubric_lines = [f"- {k}: {v}" for k, v in rubric.items()]
                rubric_section = "Rubric:\n" + "\n".join(rubric_lines)
            elif isinstance(rubric, str):
                rubric_section = f"Rubric:\n{rubric}"

        return SOLUTION_GENERATION_PROMPT.format(
            subject=subject,
            question_type=question_type,
            max_marks=max_marks,
            question_text=question_text,
            rubric_section=rubric_section,
        )
