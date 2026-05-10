"""
OCR Service

Provides OCR (Optical Character Recognition) capabilities for analyzing
question images. Uses OpenAI GPT vision as the provider.

SWM-011: All LLM calls — text and vision — are routed through the shared
gate (caller_id ``dcr_ai``) per C4 policy.  Vision calls use
``gate.call(messages=...)`` which forwards the multimodal messages array
to the provider.  If the gate module is not deployed, the direct OpenAI
provider is used as a fallback with a CRITICAL log so the C4 violation
is visible in monitoring.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Provider selection
DEFAULT_PROVIDER = os.getenv("OCR_PROVIDER", "openai")  # prioritize openai due to stability


# ---------------------------------------------------------------------------
# LLM Gate bridge (SWM-011)
#
# All OCR LLM calls (text and vision) are routed through the shared gate
# with caller_id ``dcr_ai``.  Vision calls use ``gate.call(messages=...)``
# which forwards the multimodal messages array to the provider.
#
# If the gate module is not deployed, the direct OpenAI provider is used
# as a fallback with a CRITICAL log so the C4 violation is visible in
# monitoring.
# ---------------------------------------------------------------------------

_gate_module = None
_gate_import_attempted = False  # True once we have tried (success or failure)
_gate_unavailable = False       # True when import was attempted and failed


def _try_load_gate_module():
    """Attempt to import the LLM gate module once.  Returns the module or None."""
    global _gate_module, _gate_import_attempted, _gate_unavailable
    if _gate_module is not None:
        return _gate_module
    if _gate_import_attempted:
        return None
    _gate_import_attempted = True
    try:
        from api.v1._exampen_imports import load_exampen
        _gate_module = load_exampen("llm_gate")
        logger.info("LLM gate module loaded for OCR bridge (SWM-011)")
        return _gate_module
    except ImportError:
        _gate_unavailable = True
        # CRITICAL: C4 requires all LLM calls through the gate.  If the gate
        # is not importable the deployment is non-compliant.
        logger.critical(
            "SWM-011 C4 VIOLATION: exam-conductor.llm_gate not importable — "
            "OCR LLM calls (text + vision) will bypass the gate.  "
            "Deploy exam-conductor to restore compliance."
        )
        return None


async def _gate_text_call(
    tenant_db: Any,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Route a text-only LLM call through the shared gate with caller ``dcr_ai``.

    Returns the response content string, or ``None`` **only** when the gate
    module was never importable (deployment issue).

    If the gate *is* loaded but the call fails at runtime, the exception
    propagates — there is no silent fallback to a direct provider.
    """
    gate_mod = _try_load_gate_module()
    if gate_mod is None:
        raise RuntimeError(
            "SWM-011 C4: LLM gate not available — exam-conductor not deployed. "
            "All LLM calls must go through the shared gate (C4)."
        )

    gate = gate_mod.LLMGate(tenant_db)
    await gate.initialize()

    model_id = OPENAI_MODEL

    gate_resp = await gate.call(
        model_id=model_id,
        prompt=prompt,
        caller_id="dcr_ai",
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return gate_resp.content


async def _svg_to_png_b64(svg_b64: str) -> str:
    """Convert SVG base64 (raw, no data URI prefix) to PNG base64 using cairosvg."""
    import base64 as _b64
    import cairosvg
    svg_bytes = _b64.b64decode(svg_b64)
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=1024)
    return _b64.b64encode(png_bytes).decode("ascii")


async def _normalize_image_b64(image_b64: str) -> tuple:
    """
    Strip data URI prefix, detect MIME, and convert SVG to PNG if needed.
    Returns (png_b64_raw, mime_type).
    """
    raw = image_b64
    mime = "image/png"
    if "," in raw:
        prefix, raw = raw.split(",", 1)
        if "svg+xml" in prefix:
            mime = "image/svg+xml"
        elif "jpeg" in prefix or "jpg" in prefix:
            mime = "image/jpeg"
        elif "webp" in prefix:
            mime = "image/webp"
        elif "gif" in prefix:
            mime = "image/gif"

    if mime == "image/svg+xml":
        raw = await _svg_to_png_b64(raw)
        mime = "image/png"

    return raw, mime


async def _gate_vision_call(
    tenant_db: Any,
    image_b64: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Route a vision/multimodal LLM call through the shared gate with caller ``dcr_ai``.

    Builds an OpenAI-style messages array with text + image_url parts and
    forwards it via ``gate.call(messages=...)``.

    Handles SVG→PNG conversion when the input is SVG-base64.

    Returns the response content string, or ``None`` **only** when the gate
    module was never importable (deployment issue).

    If the gate *is* loaded but the call fails at runtime, the exception
    propagates — there is no silent fallback to a direct provider.
    """
    gate_mod = _try_load_gate_module()
    if gate_mod is None:
        raise RuntimeError(
            "SWM-011 C4: LLM gate not available — exam-conductor not deployed. "
            "All LLM calls (including vision) must go through the shared gate (C4)."
        )

    gate = gate_mod.LLMGate(tenant_db)
    await gate.initialize()

    model_id = OPENAI_MODEL

    img_raw, mime = await _normalize_image_b64(image_b64)

    # Build multimodal messages array (OpenAI format)
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img_raw}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    gate_resp = await gate.call(
        model_id=model_id,
        prompt=prompt,
        caller_id="dcr_ai",
        messages=messages,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return gate_resp.content


class OCRService:
    """OCR service using OpenAI GPT vision.

    SWM-011 ownership declaration:
        Writes : nothing (OCR service is stateless)
        Reads  : question images (via caller-supplied base64)
        Never writes to : practice_attempts, conducted-exam artifacts
        Transactional boundaries : none — pure inference
    """

    def __init__(self):
        self.openai_available = bool(OPENAI_API_KEY)
        self.http_client = httpx.AsyncClient(timeout=60.0)

        if self.openai_available:
            logger.info("OCRService initialized with OpenAI")
        else:
            logger.warning("OCRService: No OpenAI API key configured, OCR will be unavailable")

    async def analyze_image(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        *,
        tenant_db: Any = None,
    ) -> dict:
        """
        Analyze an image and extract text/mathematical content.

        Args:
            image_b64: Base64-encoded image data (with or without data URI prefix)
            prompt: Optional custom prompt for the analysis
            tenant_db: Optional Motor database for LLM gate routing (SWM-011)

        Returns:
            dict with 'text' (extracted text) and 'success' (bool)
        """
        default_prompt = """Extract all text and mathematical equations from this image exactly as written.
Return ONLY the extracted text, nothing else. No explanations, no comments, no formatting notes."""

        analysis_prompt = prompt or default_prompt

        # SWM-011: All LLM calls MUST go through the shared gate (C4).
        # No direct provider fallback.
        if not image_b64:
            # Text-only path (rare — OCR usually has an image)
            gate_content = await _gate_text_call(
                tenant_db, analysis_prompt, max_tokens=1024
            )
            return {"success": True, "text": gate_content, "provider": "gate:dcr_ai"}

        # Vision path — the common case for OCR
        gate_content = await _gate_vision_call(
            tenant_db, image_b64, analysis_prompt, max_tokens=1024
        )
        return {"success": True, "text": gate_content, "provider": "gate:dcr_ai"}




    async def _analyze_with_openai(self, image_b64: str, prompt: str) -> dict:
        """Analyze image using OpenAI GPT vision API."""
        url = f"{OPENAI_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        img_raw, mime = await _normalize_image_b64(image_b64)

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_raw}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }

        # Use max_completion_tokens for newer models, max_tokens for older ones
        # GPT-5.x, GPT-4o, GPT-4-turbo, o1-preview, o1-mini use the new parameter
        model_lower = OPENAI_MODEL.lower()
        if any(m in model_lower for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
            payload['max_completion_tokens'] = 1024
        else:
            payload['max_tokens'] = 1024

        response = await self.http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            error_text = response.text
            logger.error(f"OpenAI vision error: {response.status_code} - {error_text}")
            return {"success": False, "text": "", "error": error_text}

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        return {"success": True, "text": text, "provider": "openai"}

    async def evaluate_answer(
        self,
        question_text: str,
        answer_image_b64: str,
        *,
        tenant_db: Any = None,
        correct_answer: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a student's handwritten answer.

        Args:
            question_text: The question that was asked
            answer_image_b64: Base64-encoded image of student's answer
            tenant_db: Optional Motor database for LLM gate routing (SWM-011)

        Returns:
            dict with 'score' (correct/incorrect/partial), 'feedback', 'success'
        """
        eval_prompt = f"""You are a teacher evaluating a student's handwritten answer.

QUESTION: {question_text}
"""
        if correct_answer:
            eval_prompt += f"\nCORRECT ANSWER: {correct_answer}\n"

        eval_prompt += """
The attached image shows the student's handwritten response to this question.

Please:
1. Read and interpret the student's handwritten answer
2. Evaluate if the answer is correct, incorrect, partially correct, or inconclusive (unreadable)
3. Provide brief, helpful feedback

Respond in this exact JSON format:
{
  "score": "correct" or "incorrect" or "partial" or "inconclusive",
  "extracted_answer": "what you read from the handwriting",
  "correct_answer": "the correct answer if known",
  "feedback": "brief feedback for the student (1-2 sentences)"
}

Only respond with the JSON, nothing else."""

        # SWM-011: All LLM calls MUST go through the shared gate (C4).
        # No direct provider fallback.
        gate_content = await _gate_vision_call(
            tenant_db, answer_image_b64, eval_prompt, max_tokens=512
        )
        # Parse the gate response into the expected dict shape
        import json as json_module
        try:
            clean_text = gate_content.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            clean_text = clean_text.strip()
            result = json_module.loads(clean_text)
            score = result.get("score", "inconclusive")
            if score not in ("correct", "incorrect", "partial", "inconclusive"):
                score = "inconclusive"
            return {
                "success": True,
                "score": score,
                "extracted_answer": result.get("extracted_answer", ""),
                "correct_answer": result.get("correct_answer", correct_answer or ""),
                "feedback": result.get("feedback", ""),
                "provider": "gate:dcr_ai",
            }
        except json_module.JSONDecodeError:
            lower_text = gate_content.lower()
            if "correct" in lower_text and "incorrect" not in lower_text:
                score = "correct"
            elif "incorrect" in lower_text:
                score = "incorrect"
            elif "inconclusive" in lower_text or "unreadable" in lower_text:
                score = "inconclusive"
            else:
                score = "partial"
            return {
                "success": True,
                "score": score,
                "extracted_answer": "",
                "correct_answer": correct_answer or "",
                "feedback": gate_content[:200],
                "provider": "gate:dcr_ai",
            }

    async def _evaluate_with_openai(self, image_b64: str, prompt: str) -> dict:
        """Evaluate using OpenAI."""
        import json as json_module

        url = f"{OPENAI_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        img_raw, mime = await _normalize_image_b64(image_b64)

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_raw}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }

        # Use max_completion_tokens for newer models
        model_lower = OPENAI_MODEL.lower()
        if any(m in model_lower for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
            payload['max_completion_tokens'] = 512
        else:
            payload['max_tokens'] = 512

        response = await self.http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            return {"success": False, "error": response.text}

        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

        # Parse the JSON response
        try:
            # Clean up response - sometimes models add markdown code blocks
            clean_text = response_text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            clean_text = clean_text.strip()

            result = json_module.loads(clean_text)
            return {
                "success": True,
                "score": result.get("score", "partial"),
                "extracted_answer": result.get("extracted_answer", ""),
                "feedback": result.get("feedback", ""),
                "provider": "openai",
            }
        except json_module.JSONDecodeError:
            # Fallback: try to extract score from text
            lower_text = response_text.lower()
            if "correct" in lower_text and "incorrect" not in lower_text:
                score = "correct"
            elif "incorrect" in lower_text:
                score = "incorrect"
            else:
                score = "partial"

            return {
                "success": True,
                "score": score,
                "extracted_answer": "",
                "feedback": response_text[:200],
                "provider": "openai",
            }

    async def generate_solution(
        self,
        question_text: str,
        question_image_b64: Optional[str] = None,
        *,
        tenant_db: Any = None,
    ) -> dict:
        solution_prompt = f"""You are an expert teacher. Generate a step-by-step solution for this question.

QUESTION: {question_text}

Provide:
1. A clear step-by-step solution
2. The final answer clearly stated

Respond in this exact JSON format:
{{
  "solution": "step-by-step solution explanation",
  "final_answer": "the final answer"
}}

Only respond with the JSON, nothing else."""

        if question_image_b64:
            gate_content = await _gate_vision_call(
                tenant_db, question_image_b64, solution_prompt, max_tokens=2048
            )
        else:
            gate_content = await _gate_text_call(
                tenant_db, solution_prompt, max_tokens=2048
            )

        import json as json_module
        try:
            clean_text = gate_content.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            clean_text = clean_text.strip()
            result = json_module.loads(clean_text)
            return {
                "success": True,
                "solution": result.get("solution", ""),
                "final_answer": result.get("final_answer", ""),
                "provider": "gate:dcr_ai",
            }
        except json_module.JSONDecodeError:
            return {
                "success": True,
                "solution": gate_content,
                "final_answer": "",
                "provider": "gate:dcr_ai",
            }

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Singleton instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create the OCR service singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
