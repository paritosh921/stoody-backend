"""
LLM Gate — Provider adapter layer.

Wraps OpenAI, Anthropic, Google Gemini, and Mistral behind a uniform interface
so that the gate never leaks provider-specific types to callers.

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §3 (no direct provider calls)
Failure modes   : GATE-02 — if a caller invokes a provider directly, the gate cannot
                  log usage.  This adapter is the only sanctioned path.
Hard constraint : C4 — All LLM calls through the gate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified provider response
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    """Normalised output from any LLM provider."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cost estimation table (USD per 1 M tokens — approximate, updated as needed)
# ---------------------------------------------------------------------------

# Format: model_prefix -> (input_cost_per_M, output_cost_per_M)
_COST_TABLE: Dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    # Gemini
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # Mistral
    "mistral-ocr-latest": (1.00, 1.00),
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    "mistral-medium": (2.70, 8.10),
    "pixtral": (0.15, 0.60),
    "open-mistral-nemo": (0.15, 0.15),
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost for a call.  Matches the longest prefix in the cost
    table so that e.g. ``gpt-4o-2024-08-06`` maps to ``gpt-4o``.
    """
    best_match = ""
    for prefix in _COST_TABLE:
        if model_id.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
    if not best_match:
        return 0.0
    inp_rate, out_rate = _COST_TABLE[best_match]
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Token estimation heuristic
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Quick heuristic: ~4 characters per token for English, ~3 for mixed/code.
    Uses 3.5 as a middle ground.  Intentionally does not import tiktoken to
    avoid a hard dependency; callers wanting exact counts can use tiktoken
    externally.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


# Conservative per-image token estimate for multimodal calls.
# Vision models typically use 765-1105 tokens per image depending on detail;
# 1000 is a safe middle-ground heuristic.
_TOKENS_PER_IMAGE = 1000


def estimate_tokens_for_messages(
    prompt: str,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Estimate input tokens for a gate call.

    When *messages* is ``None`` (text-only call), falls back to the simple
    ``estimate_tokens(prompt)`` heuristic.

    For multimodal *messages*, the estimate sums:
    - text tokens (from all ``text`` content parts across all messages)
    - a conservative fixed count per image (``_TOKENS_PER_IMAGE``)
    """
    if messages is None:
        return estimate_tokens(prompt)

    text_chars = 0
    image_count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type", "")
                if ptype == "text":
                    text_chars += len(part.get("text", ""))
                elif ptype in ("image_url", "image"):
                    image_count += 1
    text_tokens = max(1, int(text_chars / 3.5)) if text_chars else 0
    return text_tokens + image_count * _TOKENS_PER_IMAGE


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def _detect_provider(model_id: str) -> str:
    """Return ``'openai'``, ``'anthropic'``, ``'gemini'``, or ``'mistral'``.

    If the ``AI_PROVIDER`` env var is set to one of the recognised provider
    names it takes precedence over model-name heuristics, making the gate
    fully model-agnostic.
    """
    override = os.getenv("AI_PROVIDER", "").strip().lower()
    if override in ("openai", "anthropic", "gemini", "mistral"):
        return override

    lower = model_id.lower()
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("gemini"):
        return "gemini"
    if lower.startswith(("mistral", "pixtral", "open-mistral")):
        return "mistral"
    # Default to OpenAI for gpt-*, o1-*, o3-*, etc.
    return "openai"


# ---------------------------------------------------------------------------
# Individual provider implementations (all use httpx for uniformity)
# ---------------------------------------------------------------------------

async def _call_openai(
    model_id: str,
    prompt: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> ProviderResponse:
    """Call OpenAI Chat Completions API via httpx (no SDK leak).

    When *messages* is provided it is used as-is for the ``messages`` field
    in the API payload (supports vision / multimodal content).  Otherwise
    *prompt* is wrapped in a single user message.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code >= 400:
            error_body = resp.text
            logger.error(
                f"OpenAI API error {resp.status_code} for model={model_id}: {error_body[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"] if data.get("choices") else ""
    usage = data.get("usage", {})
    return ProviderResponse(
        content=choice,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        cache_read_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        cache_creation_tokens=0,
        model=data.get("model", model_id),
        raw=data,
    )


async def _call_mistral(
    model_id: str,
    prompt: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> ProviderResponse:
    """Call Mistral Chat Completions API via httpx.

    Mistral exposes an OpenAI-compatible chat completions endpoint at
    ``https://api.mistral.ai/v1/chat/completions``.  The request/response
    format mirrors OpenAI, so multimodal messages are forwarded as-is.
    """
    key = api_key or os.getenv("MISTRAL_API_KEY", "")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"] if data.get("choices") else ""
    usage = data.get("usage", {})
    return ProviderResponse(
        content=choice,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        cache_read_tokens=0,
        cache_creation_tokens=0,
        model=data.get("model", model_id),
        raw=data,
    )


async def _call_anthropic(
    model_id: str,
    prompt: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> ProviderResponse:
    """Call Anthropic Messages API via httpx.

    When *messages* is provided it is used as-is for the ``messages`` field
    in the API payload (supports vision / multimodal content).  Otherwise
    *prompt* is wrapped in a single user message.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens or 4096,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract text from content blocks
    content_blocks = data.get("content", [])
    content = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    usage = data.get("usage", {})
    return ProviderResponse(
        content=content,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        model=data.get("model", model_id),
        raw=data,
    )


async def _call_gemini(
    model_id: str,
    prompt: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> ProviderResponse:
    """Call Google Gemini generateContent API via httpx.

    When *messages* is provided the Gemini ``contents`` array is built from
    them.  Text parts map to ``{"text": ...}`` and image parts (OpenAI-style
    ``image_url`` with a ``data:`` URI) map to ``{"inline_data": ...}``.
    This lets the gate forward multimodal payloads originally built for
    OpenAI's schema to Gemini as well.

    When *messages* is ``None``, *prompt* is wrapped as a single text part.
    """
    key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY / GEMINI_API_KEY is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={key}"

    generation_config: Dict[str, Any] = {}
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if temperature is not None:
        generation_config["temperature"] = temperature

    # Build contents array
    if messages is not None:
        contents = _openai_messages_to_gemini_contents(messages)
    else:
        contents = [{"parts": [{"text": prompt}]}]

    payload: Dict[str, Any] = {
        "contents": contents,
    }
    if generation_config:
        payload["generationConfig"] = generation_config

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract text from candidates
    candidates = data.get("candidates", [])
    content = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)

    usage_metadata = data.get("usageMetadata", {})
    return ProviderResponse(
        content=content,
        input_tokens=usage_metadata.get("promptTokenCount", 0),
        output_tokens=usage_metadata.get("candidatesTokenCount", 0),
        cache_read_tokens=usage_metadata.get("cachedContentTokenCount", 0),
        cache_creation_tokens=0,
        model=model_id,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Gemini message format conversion
# ---------------------------------------------------------------------------

def _openai_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert an OpenAI-style messages array to Gemini ``contents`` format.

    Handles:
    - Plain string ``content`` (text-only messages)
    - List ``content`` with ``text`` and ``image_url`` parts (multimodal)

    System messages are prepended as text parts in the first user turn since
    Gemini does not have a separate system role in the contents array.
    """
    import re as _re

    contents: List[Dict[str, Any]] = []
    pending_system_text: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        # Accumulate system messages — they will be prepended to the next user turn
        if role == "system":
            if isinstance(raw_content, str):
                pending_system_text.append(raw_content)
            elif isinstance(raw_content, list):
                for part in raw_content:
                    if part.get("type") == "text":
                        pending_system_text.append(part.get("text", ""))
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: List[Dict[str, Any]] = []

        # Flush any accumulated system text into this turn
        if pending_system_text:
            parts.append({"text": "\n\n".join(pending_system_text)})
            pending_system_text = []

        if isinstance(raw_content, str):
            parts.append({"text": raw_content})
        elif isinstance(raw_content, list):
            for part in raw_content:
                ptype = part.get("type", "")
                if ptype == "text":
                    parts.append({"text": part.get("text", "")})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    # Parse data URI: data:<media_type>;base64,<data>
                    m = _re.match(r"data:([^;]+);base64,(.+)", url, _re.DOTALL)
                    if m:
                        parts.append({
                            "inline_data": {
                                "mime_type": m.group(1),
                                "data": m.group(2),
                            }
                        })
                    else:
                        # Non-data URL — pass as text reference (Gemini doesn't
                        # natively fetch URLs in inline_data)
                        parts.append({"text": f"[Image URL: {url}]"})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    # If there's leftover system text with no subsequent user message, add it
    if pending_system_text:
        contents.append({"role": "user", "parts": [{"text": "\n\n".join(pending_system_text)}]})

    return contents


# ---------------------------------------------------------------------------
# Default model resolution
# ---------------------------------------------------------------------------

# Maps provider name -> (env var, fallback default)
_DEFAULT_MODEL_MAP: Dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_MODEL", "gpt-4o"),
    "mistral": ("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "gemini": ("GEMINI_MODEL", "gemini-2.5-flash"),
}


def get_default_model() -> str:
    """Return the default model identifier for the active provider.

    The active provider is determined by ``AI_PROVIDER`` env var (falling
    back to ``"openai"`` when unset).  Each provider reads its own env var
    for the model name, with a sensible fallback:

    - ``openai``    -> ``OPENAI_MODEL``       (default ``gpt-4o``)
    - ``mistral``   -> ``MISTRAL_OCR_MODEL``  (default ``mistral-ocr-latest``)
    - ``anthropic`` -> ``ANTHROPIC_MODEL``     (default ``claude-sonnet-4-20250514``)
    - ``gemini``    -> ``GEMINI_MODEL``        (default ``gemini-2.5-flash``)
    """
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    env_var, fallback = _DEFAULT_MODEL_MAP.get(provider, ("OPENAI_MODEL", "gpt-4o"))
    return os.getenv(env_var, fallback)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

async def call_provider(
    model_id: str,
    prompt: str,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> ProviderResponse:
    """
    Route an LLM call to the correct provider and return a normalised
    ``ProviderResponse``.

    Parameters
    ----------
    model_id : str
        Provider model identifier.
    prompt : str
        Text prompt (used when *messages* is ``None``).
    messages : list, optional
        Pre-built messages array for multimodal / vision calls.  When
        provided the array is forwarded to the provider as-is (OpenAI /
        Anthropic) or translated (Gemini).  *prompt* is ignored in this case.

    This is the ONLY function external code should use.  It is called
    exclusively by ``gate.py`` — no other module may import or call
    the internal ``_call_*`` functions.
    """
    provider = _detect_provider(model_id)
    logger.debug("LLM gate dispatching to %s for model %s (multimodal=%s)", provider, model_id, messages is not None)

    if provider == "openai":
        return await _call_openai(
            model_id, prompt,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    elif provider == "mistral":
        return await _call_mistral(
            model_id, prompt,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    elif provider == "anthropic":
        return await _call_anthropic(
            model_id, prompt,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    elif provider == "gemini":
        return await _call_gemini(
            model_id, prompt,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider for model: {model_id}")
