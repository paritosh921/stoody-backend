"""Deferred OpenAI Batch transport for calls that already use :class:`LLMGate`.

The wrapper never sends a provider request itself. It either replays a saved
Batch response through the gate's usage ledger, or raises ``DeferredBatchCall``
with the exact `/v1/responses` body that may be written to Batch JSONL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .provider import ProviderResponse, provider_response_from_responses_body


class DeferredBatchCall(RuntimeError):
    """Carries one authorized provider request to the durable batch coordinator."""

    deferred_batch_call = True

    def __init__(
        self,
        *,
        call_index: int,
        request_body: Dict[str, Any],
        caller_id: str,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"Provider call {call_index + 1} is waiting for OpenAI Batch")
        self.call_index = call_index
        self.request_body = request_body
        self.caller_id = caller_id
        self.model_id = model_id
        self.metadata = dict(metadata or {})


class BatchReplayGate:
    """Replay imported responses, then defer the next missing provider call."""

    resume_deferred_run = True

    def __init__(
        self,
        gate: Any,
        *,
        response_bodies: Optional[Sequence[Mapping[str, Any]]] = None,
        recorded_call_indexes: Optional[Sequence[int]] = None,
    ) -> None:
        self._gate = gate
        self._response_bodies = [dict(item) for item in (response_bodies or [])]
        self._recorded_call_indexes: Set[int] = {
            max(0, int(value)) for value in (recorded_call_indexes or [])
        }
        self._call_index = 0

    async def initialize(self) -> None:
        if hasattr(self._gate, "initialize"):
            await self._gate.initialize()

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
    ) -> Any:
        if messages is not None or responses_input is None:
            raise ValueError("Economy checking supports native OpenAI Responses calls only")

        request_body = await self._gate.prepare_batch_responses_call(
            model_id,
            prompt,
            caller_id,
            responses_input=responses_input,
            json_schema=json_schema,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        current_index = self._call_index
        self._call_index += 1
        if current_index >= len(self._response_bodies):
            raise DeferredBatchCall(
                call_index=current_index,
                request_body=request_body,
                caller_id=caller_id,
                model_id=model_id,
                metadata=metadata,
            )

        provider_response: ProviderResponse = provider_response_from_responses_body(
            self._response_bodies[current_index],
            fallback_model=model_id,
        )
        replay_metadata = dict(metadata or {})
        replay_metadata["provider_call_index"] = current_index
        return await self._gate.record_batch_response(
            requested_model_id=model_id,
            caller_id=caller_id,
            provider_response=provider_response,
            metadata=replay_metadata,
            persist_log=current_index not in self._recorded_call_indexes,
        )


__all__ = ["BatchReplayGate", "DeferredBatchCall"]
