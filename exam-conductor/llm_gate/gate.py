"""
LLM Gate — Core orchestrator.

Implements the full gate contract:

    call(model_id, prompt, caller_id, **kwargs) -> GateResponse

Steps (LLM_GATE_SPEC.md §3):
  1. Validate caller_id
  2. Estimate prompt size
  3. Enforce per-call limits
  4. Enforce daily / weekly / monthly budgets
  5. Call provider
  6. Append token log
  7. Return content + usage metadata

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §3-§4
Ownership       : LLM gate (STATE_OWNERSHIP_MAP.md)
Test IDs        : U-GATE-01 (caller validation), U-GATE-02 (per-call limits),
                  U-GATE-03 (budget checks), U-GATE-04 (log shape),
                  I-GATE-01 (DCR through gate), I-GATE-02 (PCR through gate)
Failure modes   : GATE-01 (budget exhaustion), GATE-02 (caller bypass),
                  GATE-03 (log consistency)
Hard constraints: C1 (MongoDB only), C4 (all LLM calls through gate),
                  C5 (gate owns token budget state only)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from .budget import BudgetChecker
from .models import (
    ALLOWED_CALLER_IDS,
    GateConfig,
    GateResponse,
    TokenUsage,
    TokenUsageLogEntry,
    UnregisteredCallerError,
)
from .provider import (
    ProviderResponse,
    build_openai_responses_payload,
    call_provider,
    estimate_cost,
    estimate_tokens,
    estimate_tokens_for_messages,
)
from .repository import GateRepository

logger = logging.getLogger(__name__)


class LLMGate:
    """
    Shared LLM gate for ExamPen.

    Every LLM call in the system — DCR fallback, PCR evaluation, practice,
    cache warmup — MUST go through ``gate.call()``.  No module may import
    or invoke an LLM provider directly (C4).

    Typical usage::

        db = await db_manager.get_tenant_db(db_name)
        gate = LLMGate(db)
        await gate.initialize()
        response = await gate.call(
            model_id="gpt-4o",
            prompt="Evaluate this student response...",
            caller_id="pcr_eval_core",
        )
        print(response.content, response.usage.total_tokens)

    Ownership declaration (STATE_OWNERSHIP_MAP.md §5):
        Writes : llm_gate_config, llm_token_usage_log, llm_token_usage_rollup
        Reads  : llm_gate_config, llm_token_usage_log
        Never writes to : conducted-exam artifacts, DCR/PCR results, practice persistence
        Transactional boundaries : provider response + append-only usage log
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._repo = GateRepository(db)
        self._budget = BudgetChecker(self._repo)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Bootstrap indexes.  Safe to call multiple times (idempotent)."""
        if not self._initialized:
            await self._repo.ensure_indexes()
            self._initialized = True

    # ------------------------------------------------------------------
    # Config access (delegated to repository)
    # ------------------------------------------------------------------

    async def get_config(self) -> GateConfig:
        return await self._repo.get_config()

    async def update_config(self, config: GateConfig) -> None:
        await self._repo.upsert_config(config)

    # ------------------------------------------------------------------
    # Usage query (used by SWM-007 route layer)
    # ------------------------------------------------------------------

    async def current_usage(self) -> dict:
        """Return ``CurrentUsage`` dict for the tenant."""
        config = await self._repo.get_config()
        return await self._budget.current_usage(config)

    # ------------------------------------------------------------------
    # Core gate contract (§4.1)
    # ------------------------------------------------------------------

    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        responses_input: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        prompt_cache_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GateResponse:
        """
        Execute an LLM call through the gate.

        Parameters
        ----------
        model_id : str
            Provider model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
        prompt : str
            The full prompt text.  For pure vision/multimodal calls where
            all content is in *messages*, pass an empty string.
        caller_id : str
            One of the registered caller identities.
        messages : list of dict, optional
            Pre-built messages array for multimodal / vision calls.  When
            provided the array is forwarded to the provider as-is and
            *prompt* is used only for logging context (not sent to the LLM).
            Token estimation accounts for text parts plus a conservative
            per-image heuristic.
        responses_input : list of dict, optional
            Native OpenAI Responses input for PDF/image document calls. This
            is mutually exclusive with ``messages`` and remains subject to
            the same caller, budget, and usage-log controls.
        json_schema : dict, optional
            Strict structured-output schema for a Responses API call.
        prompt_cache_key : str, optional
            Stable non-PII routing key for repeated static prompt prefixes.
        reasoning_effort : str, optional
            Provider reasoning effort for supported Responses models.
        max_output_tokens : int, optional
            Override for the max output token count.
        temperature : float, optional
            Temperature override (only if caller-specific override is allowed).
        metadata : dict, optional
            Opaque metadata forwarded to the token log.

        Returns
        -------
        GateResponse
            Contains ``content`` (the LLM text) and ``usage`` (full accounting).

        Raises
        ------
        UnregisteredCallerError
            If ``caller_id`` is not in the allow-list (GATE-02).
        TokenLimitExceededError
            If per-call input or output limits are breached (§9.2).
        BudgetExhaustedError
            If any budget period is exhausted (GATE-01, §9.1).
        RuntimeError
            If the provider API key is missing.
        ProviderHTTPError
            On provider-level HTTP failures. The error is sanitised and
            carries the provider retry contract.
        """

        # ── Step 1: Validate caller_id (GATE-02 mitigation) ────────────
        if caller_id not in ALLOWED_CALLER_IDS:
            raise UnregisteredCallerError(caller_id)

        # ── Step 2: Estimate prompt size ────────────────────────────────
        # For multimodal calls, estimate text tokens + conservative per-image
        # heuristic.  For text-only calls, use the simple heuristic.
        if messages is not None and responses_input is not None:
            raise ValueError("messages and responses_input are mutually exclusive")
        estimated_input = estimate_tokens_for_messages(
            prompt,
            messages,
            responses_input,
        )

        # ── Load config once ────────────────────────────────────────────
        config = await self._repo.get_config()

        # ── Step 3: Enforce per-call limits ─────────────────────────────
        BudgetChecker.check_per_call_input(config, estimated_input)
        BudgetChecker.check_per_call_output(config, max_output_tokens)

        # Clamp max_output_tokens to config ceiling if caller didn't
        # specify one but config has a limit.
        effective_max_output = max_output_tokens
        if effective_max_output is None and config.max_output_tokens is not None:
            effective_max_output = config.max_output_tokens

        # ── Step 4: Enforce period budgets ──────────────────────────────
        await self._budget.check_budgets(config)

        # ── Step 5: Call provider ───────────────────────────────────────
        provider_resp: ProviderResponse = await call_provider(
            model_id,
            prompt,
            messages=messages,
            responses_input=responses_input,
            json_schema=json_schema,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            max_output_tokens=effective_max_output,
            temperature=temperature,
        )

        # ── Compute usage metadata ──────────────────────────────────────
        total_tokens = provider_resp.input_tokens + provider_resp.output_tokens
        cost = estimate_cost(
            model_id,
            provider_resp.input_tokens,
            provider_resp.output_tokens,
            cache_read_tokens=provider_resp.cache_read_tokens,
        )
        now = datetime.utcnow()

        usage = TokenUsage(
            model=provider_resp.model or model_id,
            caller=caller_id,
            input_tokens=provider_resp.input_tokens,
            output_tokens=provider_resp.output_tokens,
            cache_read_tokens=provider_resp.cache_read_tokens,
            cache_creation_tokens=provider_resp.cache_creation_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            timestamp=now,
        )

        # ── Step 6: Append token log (GATE-03 mitigation) ──────────────
        log_entry = TokenUsageLogEntry(
            model=usage.model,
            caller=usage.caller,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            called_at=now,
            metadata=metadata,
        )
        try:
            await self._repo.append_log(log_entry)
        except Exception:
            # Log persistence failure must not lose the LLM response that
            # was already generated — callers still get the content.
            # This aligns with GATE-03 mitigation: we log the failure so
            # operators can investigate, but do not discard the response.
            logger.exception(
                "GATE-03: Failed to persist token usage log for caller=%s model=%s",
                caller_id,
                model_id,
            )

        # ── Step 7: Return content + usage metadata ─────────────────────
        return GateResponse(
            content=provider_resp.content,
            usage=usage,
            completion_status=provider_resp.completion_status,
            incomplete_reason=provider_resp.incomplete_reason,
        )

    async def prepare_batch_responses_call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        responses_input: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
        prompt_cache_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Authorize and build one delayed `/v1/responses` Batch request.

        Batch work still passes the same caller allow-list, per-call limits and
        tenant budgets as an immediate call. Usage is appended later, when the
        provider result is imported, because only then are actual tokens known.
        """

        if caller_id not in ALLOWED_CALLER_IDS:
            raise UnregisteredCallerError(caller_id)
        estimated_input = estimate_tokens_for_messages(
            prompt,
            None,
            responses_input,
        )
        config = await self._repo.get_config()
        BudgetChecker.check_per_call_input(config, estimated_input)
        BudgetChecker.check_per_call_output(config, max_output_tokens)
        effective_max_output = max_output_tokens
        if effective_max_output is None and config.max_output_tokens is not None:
            effective_max_output = config.max_output_tokens
        await self._budget.check_budgets(config)
        return build_openai_responses_payload(
            model_id,
            responses_input=responses_input,
            json_schema=json_schema,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            max_output_tokens=effective_max_output,
            temperature=temperature,
        )

    async def record_batch_response(
        self,
        *,
        requested_model_id: str,
        caller_id: str,
        provider_response: ProviderResponse,
        metadata: Optional[Dict[str, Any]] = None,
        persist_log: bool = True,
    ) -> GateResponse:
        """Record actual usage for one successfully imported Batch item."""

        if caller_id not in ALLOWED_CALLER_IDS:
            raise UnregisteredCallerError(caller_id)
        total_tokens = provider_response.input_tokens + provider_response.output_tokens
        cost = estimate_cost(
            requested_model_id,
            provider_response.input_tokens,
            provider_response.output_tokens,
            cache_read_tokens=provider_response.cache_read_tokens,
            batch=True,
        )
        now = datetime.utcnow()
        usage = TokenUsage(
            model=provider_response.model or requested_model_id,
            caller=caller_id,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
            cache_read_tokens=provider_response.cache_read_tokens,
            cache_creation_tokens=provider_response.cache_creation_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            timestamp=now,
        )
        log_metadata = dict(metadata or {})
        log_metadata["billing_mode"] = "openai_batch"
        try:
            if persist_log:
                await self._repo.append_log(
                    TokenUsageLogEntry(
                        model=usage.model,
                        caller=usage.caller,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_read_tokens=usage.cache_read_tokens,
                        cache_creation_tokens=usage.cache_creation_tokens,
                        total_tokens=usage.total_tokens,
                        estimated_cost_usd=usage.estimated_cost_usd,
                        called_at=now,
                        metadata=log_metadata,
                    )
                )
        except Exception:
            logger.exception(
                "GATE-03: Failed to persist Batch token usage for caller=%s model=%s",
                caller_id,
                requested_model_id,
            )
        return GateResponse(
            content=provider_response.content,
            usage=usage,
            completion_status=provider_response.completion_status,
            incomplete_reason=provider_response.incomplete_reason,
        )

    async def check_batch_reservation(self, reserved_tokens: int) -> None:
        """Check a delayed group's upper-bound tokens against tenant budgets."""

        config = await self._repo.get_config()
        await self._budget.check_reservation(config, reserved_tokens)

    # ------------------------------------------------------------------
    # Repository access (for rollup jobs and usage APIs)
    # ------------------------------------------------------------------

    @property
    def repo(self) -> GateRepository:
        """Expose repository for rollup jobs and usage API queries."""
        return self._repo

    @property
    def budget(self) -> BudgetChecker:
        """Expose budget checker for usage API current-usage queries."""
        return self._budget
