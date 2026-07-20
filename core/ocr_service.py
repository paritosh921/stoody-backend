"""
OCR Service

Provides OCR (Optical Character Recognition) capabilities for analyzing
question images. Uses OpenAI GPT vision as the provider.

SWM-011: All LLM calls — text and vision — are routed through the shared
gate (caller_id ``dcr_ai``) per C4 policy.  Vision calls use
``gate.call(messages=...)`` which forwards the multimodal messages array
to the provider. If the gate module is not deployed, inference fails closed
so no provider call bypasses shared accounting and budget enforcement.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# LLM Gate bridge (SWM-011)
#
# All OCR LLM calls (text and vision) are routed through the shared gate
# with caller_id ``dcr_ai``.  Vision calls use ``gate.call(messages=...)``
# which forwards the multimodal messages array to the provider.
#
# If the gate module is not deployed, inference fails closed.
# ---------------------------------------------------------------------------

_gate_module = None
_gate_import_attempted = False  # True once we have tried (success or failure)


def _try_load_gate_module():
    """Attempt to import the LLM gate module once.  Returns the module or None."""
    global _gate_module, _gate_import_attempted
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
        logger.critical(
            "SWM-011 C4: exam-conductor.llm_gate not importable; "
            "OCR inference is unavailable until exam-conductor is deployed."
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


async def _gate_vision_images_call(
    tenant_db: Any,
    images: List[Dict[str, Any]],
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Optional[str]:
    """Route a multimodal LLM call with multiple labeled images through the shared gate."""
    gate_mod = _try_load_gate_module()
    if gate_mod is None:
        raise RuntimeError(
            "SWM-011 C4: LLM gate not available — exam-conductor not deployed. "
            "All LLM calls (including vision) must go through the shared gate (C4)."
        )

    gate = gate_mod.LLMGate(tenant_db)
    await gate.initialize()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, image in enumerate(images, start=1):
        image_b64 = str(image.get("image_b64") or "")
        if not image_b64:
            continue
        label = str(image.get("label") or f"image_{index}")
        description = str(image.get("description") or "")
        img_raw, mime = await _normalize_image_b64(image_b64)
        label_text = f"Image {index}: {label}"
        if description:
            label_text = f"{label_text} — {description}"
        content.extend(
            [
                {"type": "text", "text": label_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img_raw}",
                        "detail": "high",
                    },
                },
            ]
        )

    if len(content) == 1:
        return await _gate_text_call(
            tenant_db,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": content,
        }
    ]

    gate_resp = await gate.call(
        model_id=OPENAI_MODEL,
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
        logger.info("OCRService initialized with shared LLM gate routing")

    @property
    def gate_available(self) -> bool:
        """Return whether the required exam-conductor LLM gate is loadable."""
        return _try_load_gate_module() is not None

    async def analyze_image(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        *,
        tenant_db: Any = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """
        Analyze an image and extract text/mathematical content.

        Args:
            image_b64: Base64-encoded image data (with or without data URI prefix)
            prompt: Optional custom prompt for the analysis
            tenant_db: Optional Motor database for LLM gate routing (SWM-011)
            temperature: Sampling temperature forwarded to the vision model

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
                tenant_db,
                analysis_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"success": True, "text": gate_content, "provider": "gate:dcr_ai"}

        # Vision path — the common case for OCR
        gate_content = await _gate_vision_call(
            tenant_db,
            image_b64,
            analysis_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"success": True, "text": gate_content, "provider": "gate:dcr_ai"}

    async def analyze_images(
        self,
        images: List[Dict[str, Any]],
        prompt: Optional[str] = None,
        *,
        tenant_db: Any = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Analyze multiple labeled images in one multimodal OCR request."""
        default_prompt = """Extract all text and mathematical equations from these images exactly as written.
Return ONLY the extracted text, nothing else. No explanations, no comments, no formatting notes."""
        analysis_prompt = prompt or default_prompt

        gate_content = await _gate_vision_images_call(
            tenant_db,
            images,
            analysis_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"success": True, "text": gate_content, "provider": "gate:dcr_ai"}

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

    async def evaluate_answers(
        self,
        question_text: str,
        answer_images: List[Dict[str, str]],
        *,
        tenant_db: Any = None,
        correct_answer: Optional[str] = None,
    ) -> Dict[str, dict]:
        """Evaluate multiple labeled student images in one gate call."""
        import json as json_module

        if not answer_images:
            return {}

        labels = [str(image.get("label") or "") for image in answer_images]
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Grouped evaluation requires unique non-empty labels")

        prompt = f"""You are a teacher evaluating multiple handwritten responses to one question.

QUESTION: {question_text}
"""
        if correct_answer:
            prompt += f"\nCORRECT ANSWER: {correct_answer}\n"
        prompt += f"""
Each following image is labeled with its pen ID. Evaluate every image independently.
Return one result for each of these labels: {json_module.dumps(labels)}.

Respond with JSON only in this exact shape:
{{
  "results": [
    {{
      "pen_id": "the exact image label",
      "score": "correct or incorrect or partial or inconclusive",
      "extracted_answer": "what was read",
      "correct_answer": "the correct answer if known",
      "feedback": "brief feedback"
    }}
  ]
}}
"""

        gate_content = await _gate_vision_images_call(
            tenant_db,
            answer_images,
            prompt,
            max_tokens=min(4096, max(768, len(answer_images) * 512)),
            temperature=0.2,
        )
        clean_text = (gate_content or "").strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```", 2)[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
            clean_text = clean_text.strip()

        try:
            decoded = json_module.loads(clean_text)
        except json_module.JSONDecodeError as exc:
            raise ValueError("Grouped evaluation returned invalid JSON") from exc

        rows = decoded.get("results") if isinstance(decoded, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Grouped evaluation response has no results list")

        expected = set(labels)
        results: Dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            pen_id = str(row.get("pen_id") or "")
            if pen_id not in expected or pen_id in results:
                continue
            score = str(row.get("score") or "inconclusive").lower()
            if score not in ("correct", "incorrect", "partial", "inconclusive"):
                score = "inconclusive"
            results[pen_id] = {
                "success": True,
                "score": score,
                "extracted_answer": row.get("extracted_answer", ""),
                "correct_answer": row.get("correct_answer", correct_answer or ""),
                "feedback": row.get("feedback", ""),
                "provider": "gate:dcr_ai",
            }
        return results

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
        """Keep the service lifecycle API; the shared gate owns its clients."""
        return None


# Singleton instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create the OCR service singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
