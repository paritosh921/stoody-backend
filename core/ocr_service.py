"""
OCR Service

Provides OCR (Optical Character Recognition) capabilities for analyzing
question images. Uses Mistral OCR as primary, OpenAI GPT vision as fallback.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional
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


class OCRService:
    """OCR service using OpenAI GPT vision."""

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
    ) -> dict:
        """
        Analyze an image and extract text/mathematical content.
        
        Args:
            image_b64: Base64-encoded image data (with or without data URI prefix)
            prompt: Optional custom prompt for the analysis
            
        Returns:
            dict with 'text' (extracted text) and 'success' (bool)
        """
        # Clean up base64 data - remove data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        default_prompt = """Extract all text and mathematical equations from this image exactly as written.
Return ONLY the extracted text, nothing else. No explanations, no comments, no formatting notes."""

        analysis_prompt = prompt or default_prompt

        # Use OpenAI
        if self.openai_available:
            try:
                result = await self._analyze_with_openai(image_b64, analysis_prompt)
                return result
            except Exception as e:
                logger.error(f"OpenAI vision error: {e}")
                return {"success": False, "text": "", "error": str(e)}

        return {
            "success": False,
            "text": "",
            "error": "No OCR providers configured",
        }




    async def _analyze_with_openai(self, image_b64: str, prompt: str) -> dict:
        """Analyze image using OpenAI GPT vision API."""
        url = f"{OPENAI_BASE_URL}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
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
                                "url": f"data:image/png;base64,{image_b64}",
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
    ) -> dict:
        """
        Evaluate a student's handwritten answer.
        
        Args:
            question_text: The question that was asked
            answer_image_b64: Base64-encoded image of student's answer
            
        Returns:
            dict with 'score' (correct/incorrect/partial), 'feedback', 'success'
        """
        # Clean up base64 data
        if "," in answer_image_b64:
            answer_image_b64 = answer_image_b64.split(",", 1)[1]

        eval_prompt = f"""You are a teacher evaluating a student's handwritten answer.

QUESTION: {question_text}

The attached image shows the student's handwritten response to this question.

Please:
1. Read and interpret the student's handwritten answer
2. Evaluate if the answer is correct, incorrect, or partially correct
3. Provide brief, helpful feedback

Respond in this exact JSON format:
{{
  "score": "correct" or "incorrect" or "partial",
  "extracted_answer": "what you read from the handwriting",
  "feedback": "brief feedback for the student (1-2 sentences)"
}}

Only respond with the JSON, nothing else."""



        # Use OpenAI for evaluation
        if self.openai_available:
            try:
                result = await self._evaluate_with_openai(answer_image_b64, eval_prompt)
                return result
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "No AI providers configured"}

    async def _evaluate_with_openai(self, image_b64: str, prompt: str) -> dict:
        """Evaluate using OpenAI."""
        import json as json_module
        
        url = f"{OPENAI_BASE_URL}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
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
                                "url": f"data:image/png;base64,{image_b64}",
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
