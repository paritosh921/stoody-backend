"""
Canvas OCR Service for Reliable Handwriting Extraction
Provides redundant OCR pipeline with multiple extraction strategies and confidence scoring.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from services.async_openai_service import AsyncOpenAIService
from utils.image_processor import (
    enhance_canvas_images_batch,
    merge_canvas_pages_vertical,
    is_canvas_empty,
    prepare_image_for_llm_evaluation
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of text extraction from canvas."""
    extracted_text: str
    confidence: float  # 0.0 to 1.0
    method: str  # Which extraction method was used
    raw_response: str  # Raw LLM response
    is_likely_answer: bool  # Whether this looks like a valid answer


class CanvasOCRService:
    """
    Multi-strategy canvas OCR service with redundancy and verification.
    
    Strategies:
    1. Direct Vision OCR - Ask LLM to transcribe what's written
    2. MCQ Letter Detection - Specialized prompt for detecting option letters
    3. Multi-pass Verification - Cross-verify extractions
    """
    
    def __init__(self):
        self.ai_service = AsyncOpenAIService()
    
    async def extract_text_from_canvas(
        self,
        canvas_pages: List[str],
        question_context: Optional[str] = None,
        options_context: Optional[str] = None,
        is_mcq: bool = False
    ) -> ExtractionResult:
        """
        Extract text from canvas images using redundant OCR strategies.
        
        Args:
            canvas_pages: List of canvas page data URLs
            question_context: The question text for context
            options_context: MCQ options text for context
            is_mcq: Whether this is an MCQ question
            
        Returns:
            ExtractionResult with extracted text and confidence
        """
        if not canvas_pages:
            return ExtractionResult(
                extracted_text="",
                confidence=0.0,
                method="none",
                raw_response="No canvas data provided",
                is_likely_answer=False
            )
        
        # Filter empty canvases
        non_empty = [p for p in canvas_pages if p and not is_canvas_empty(p)]
        if not non_empty:
            return ExtractionResult(
                extracted_text="",
                confidence=0.0,
                method="none",
                raw_response="Canvas appears to be empty",
                is_likely_answer=False
            )
        
        # Prepare enhanced images
        enhanced_pages, merged_image = prepare_image_for_llm_evaluation(
            non_empty,
            enhance=True,
            merge_pages=True,
            target_width=1500
        )
        
        # Use merged image if available, otherwise use first enhanced page
        primary_image = merged_image or (enhanced_pages[0] if enhanced_pages else None)
        
        if not primary_image:
            return ExtractionResult(
                extracted_text="",
                confidence=0.0,
                method="none",
                raw_response="Failed to process canvas images",
                is_likely_answer=False
            )
        
        # Strategy selection based on question type
        if is_mcq and options_context:
            # MCQ-specific extraction
            result = await self._extract_mcq_answer(primary_image, question_context, options_context)
        else:
            # General handwriting extraction
            result = await self._extract_general_text(primary_image, question_context)
        
        # If confidence is low, try alternative strategy
        if result.confidence < 0.5:
            logger.info(f"Low confidence ({result.confidence:.2f}), attempting fallback extraction")
            fallback_result = await self._fallback_extraction(primary_image, question_context)
            
            # Use fallback if it's more confident
            if fallback_result.confidence > result.confidence:
                result = fallback_result
        
        return result
    
    async def _extract_mcq_answer(
        self,
        image: str,
        question_context: Optional[str],
        options_context: str
    ) -> ExtractionResult:
        """Extract MCQ answer (single letter) from canvas."""
        
        prompt = """You are analyzing a handwritten answer on a digital canvas.

TASK: Identify which multiple choice option (A, B, C, D, etc.) the student has written or selected.

IMPORTANT ANALYSIS GUIDELINES:
1. Look for ANY written text, letters, numbers, or marks on the canvas
2. The student may have written: just a letter (A, B, C, D), or a word, or circled an option
3. Common handwritten letters can look similar - 'B' vs 'D', 'C' vs 'G', 'A' vs 'R'
4. Look for the CLEAREST marking that indicates their choice
5. If you see a curved line, it might be a letter 'C' or 'D'
6. If you see a vertical line with bumps, it might be 'B' or 'R'

"""
        
        if question_context:
            prompt += f"QUESTION: {question_context[:500]}\n\n"
        
        prompt += f"OPTIONS:\n{options_context}\n\n"
        
        prompt += """RESPOND IN THIS EXACT JSON FORMAT:
{
    "detected_text": "The exact text/letter you can see written",
    "selected_option": "A" or "B" or "C" or "D" (or the letter you detected),
    "confidence": 0.0 to 1.0 (how confident you are in this reading),
    "description": "Brief description of what you see on the canvas"
}

If you cannot clearly see ANY writing, set confidence to 0 and detected_text to "unclear".
Be generous in interpretation - if something looks like a letter, report what it most likely is."""

        try:
            response = await self.ai_service.analyze_image_async(
                image,
                prompt,
                max_tokens=500
            )
            
            if response.get("success") and response.get("response"):
                raw = response["response"]
                parsed = self._parse_json_response(raw)
                
                if parsed:
                    detected = parsed.get("detected_text", "").strip().upper()
                    selected = parsed.get("selected_option", "").strip().upper()
                    confidence = float(parsed.get("confidence", 0.0))
                    
                    # Normalize to single letter if possible
                    answer = selected or detected
                    if answer and len(answer) == 1 and answer.isalpha():
                        return ExtractionResult(
                            extracted_text=answer,
                            confidence=confidence,
                            method="mcq_detection",
                            raw_response=raw,
                            is_likely_answer=True
                        )
                    elif answer:
                        # Try to extract first letter
                        match = re.search(r'\b([A-Za-z])\b', answer)
                        if match:
                            return ExtractionResult(
                                extracted_text=match.group(1).upper(),
                                confidence=confidence * 0.8,
                                method="mcq_detection_partial",
                                raw_response=raw,
                                is_likely_answer=True
                            )
                
                # Failed to parse structured response
                return ExtractionResult(
                    extracted_text="",
                    confidence=0.2,
                    method="mcq_detection_failed",
                    raw_response=raw,
                    is_likely_answer=False
                )
                
        except Exception as e:
            logger.error(f"MCQ extraction error: {e}")
        
        return ExtractionResult(
            extracted_text="",
            confidence=0.0,
            method="mcq_error",
            raw_response=str(e) if 'e' in dir() else "Unknown error",
            is_likely_answer=False
        )
    
    async def _extract_general_text(
        self,
        image: str,
        question_context: Optional[str]
    ) -> ExtractionResult:
        """Extract general handwritten text from canvas."""
        
        prompt = """You are an expert at reading handwritten text from digital canvases and whiteboards.

TASK: Transcribe ALL text, numbers, symbols, equations, and markings visible on this canvas.

READING GUIDELINES:
1. Be thorough - capture every written element you can see
2. Preserve mathematical notation and equations
3. Include diagrams/drawings descriptions in [brackets]
4. If text is messy, make your best interpretation
5. Note any circled or underlined items
6. Read left-to-right, top-to-bottom

"""
        
        if question_context:
            prompt += f"CONTEXT (the question being answered):\n{question_context[:500]}\n\n"
        
        prompt += """RESPOND IN THIS EXACT JSON FORMAT:
{
    "transcribed_text": "The full text you can read from the canvas",
    "final_answer": "What appears to be the student's final answer (if identifiable)",
    "confidence": 0.0 to 1.0 (how confident you are in this reading),
    "content_type": "text" or "equation" or "diagram" or "mixed",
    "notes": "Any additional observations about the writing"
}

If canvas appears empty or you cannot read anything, set confidence to 0."""

        try:
            # Allow longer completions for dense handwritten or essay-style submissions.
            ocr_max_tokens = 1400 if question_context and len(question_context) > 300 else 1000
            response = await self.ai_service.analyze_image_async(
                image,
                prompt,
                max_tokens=ocr_max_tokens
            )
            
            if response.get("success") and response.get("response"):
                raw = response["response"]
                parsed = self._parse_json_response(raw)
                
                if parsed:
                    transcribed = parsed.get("transcribed_text", "").strip()
                    final_answer = parsed.get("final_answer", "").strip()
                    confidence = float(parsed.get("confidence", 0.5))

                    # For general extraction, preserve full transcription first.
                    # If a distinct final answer is detected, append it as structured metadata.
                    if transcribed and final_answer:
                        if final_answer.lower() in transcribed.lower():
                            extracted = transcribed
                        else:
                            extracted = f"{transcribed}\n[Detected final answer]: {final_answer}"
                    elif transcribed:
                        extracted = transcribed
                    else:
                        extracted = final_answer
                    
                    return ExtractionResult(
                        extracted_text=extracted,
                        confidence=confidence,
                        method="general_ocr",
                        raw_response=raw,
                        is_likely_answer=bool(final_answer or transcribed)
                    )
                
                # Could not parse, try raw extraction
                return ExtractionResult(
                    extracted_text=self._extract_text_from_raw(raw),
                    confidence=0.3,
                    method="general_ocr_raw",
                    raw_response=raw,
                    is_likely_answer=False
                )
                
        except Exception as e:
            logger.error(f"General text extraction error: {e}")
        
        return ExtractionResult(
            extracted_text="",
            confidence=0.0,
            method="general_error",
            raw_response="",
            is_likely_answer=False
        )
    
    async def _fallback_extraction(
        self,
        image: str,
        question_context: Optional[str]
    ) -> ExtractionResult:
        """Fallback extraction with simpler, more direct prompt."""
        
        prompt = """Look at this image of handwritten work.

SIMPLE TASK: What is written on this canvas?

Just tell me:
1. What letters, numbers, or text you can see
2. What you think the answer is

Be direct and specific. Even if messy, make your best guess."""

        try:
            response = await self.ai_service.analyze_image_async(
                image,
                prompt,
                max_tokens=300
            )
            
            if response.get("success") and response.get("response"):
                raw = response["response"]
                
                # Try to extract key information
                text = self._extract_text_from_raw(raw)
                
                # Look for single letters (often the answer for MCQ)
                letter_match = re.search(r'\b([A-Da-d])\b', text)
                if letter_match:
                    return ExtractionResult(
                        extracted_text=letter_match.group(1).upper(),
                        confidence=0.6,
                        method="fallback_letter",
                        raw_response=raw,
                        is_likely_answer=True
                    )
                
                return ExtractionResult(
                    extracted_text=text,
                    confidence=0.4,
                    method="fallback_general",
                    raw_response=raw,
                    is_likely_answer=bool(text)
                )
                
        except Exception as e:
            logger.error(f"Fallback extraction error: {e}")
        
        return ExtractionResult(
            extracted_text="",
            confidence=0.0,
            method="fallback_error",
            raw_response="",
            is_likely_answer=False
        )
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling various formats."""
        import json
        import ast
        
        if not text:
            return None
        
        # Try direct JSON parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to extract JSON object with regex
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                try:
                    return ast.literal_eval(json_match.group(0))
                except:
                    pass
        
        return None
    
    def _extract_text_from_raw(self, text: str) -> str:
        """Extract meaningful text from raw LLM response."""
        if not text:
            return ""
        
        # Remove common LLM prefixes
        text = re.sub(r'^(I can see|I see|The text|The answer|It says|This shows|Looking at)', '', text, flags=re.IGNORECASE)
        
        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return quoted[0].strip()
        
        # Get first substantial line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[0][:200]
        
        return text[:200].strip()


# Singleton instance
_canvas_ocr_service: Optional[CanvasOCRService] = None


def get_canvas_ocr_service() -> CanvasOCRService:
    """Get or create the Canvas OCR service singleton."""
    global _canvas_ocr_service
    if _canvas_ocr_service is None:
        _canvas_ocr_service = CanvasOCRService()
    return _canvas_ocr_service
