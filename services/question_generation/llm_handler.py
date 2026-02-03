"""
LLM Handler Service

Handles LLM interactions and response parsing for question generation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .models.config import QuestionType
from .models.question import GeneratedQuestion, QuestionOption, MarkingStep, DiagramSpec, GenerationSource

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM interactions and response parsing."""
    
    def __init__(self, openai_service=None, model: str = "gpt-5-mini", temperature: float = 1.0):
        self._openai_service = openai_service
        self._kimi_service = None
        self._use_kimi = False
        self._model = model
        self._temperature = temperature
    
    def set_service(self, openai_service):
        """Set the OpenAI service after initialization."""
        self._openai_service = openai_service

    def set_kimi_service(self, kimi_service, use_kimi: bool = True):
        """Set the Kimi service for question generation.
        
        Args:
            kimi_service: The Kimi service instance
            use_kimi: Whether to use Kimi for LLM calls (default True)
        """
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi
        if use_kimi:
            logger.info("LLMHandler configured to use Kimi service")
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Temperature for generation (uses default if not specified)
            max_tokens: Maximum tokens in response
            
        Returns:
            The response text or None if failed
        """
        # Use Kimi if configured
        if self._use_kimi and self._kimi_service:
            try:
                logger.info("Using Kimi service for LLM completion")
                response = await self._kimi_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature or 0.6,  # Recommended for kimi-k2.5 instant mode
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                
                if response.get("success"):
                    return response["response"]
                else:
                    logger.error(f"Kimi LLM call failed: {response.get('error')}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error in Kimi LLM completion: {e}")
                return None
        
        # Fall back to OpenAI
        if not self._openai_service:
            logger.error("No LLM service initialized (neither Kimi nor OpenAI)")
            return None
        
        try:
            logger.info("Using OpenAI service for LLM completion")
            response = await self._openai_service.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self._temperature,
                max_tokens=max_tokens,
                model=self._model,
                system_prompt=system_prompt,
            )
            
            if response.get("success"):
                return response["response"]
            else:
                logger.error(f"OpenAI LLM call failed: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in OpenAI LLM completion: {e}")
    
    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 1000,
        temperature: float = 1.0,  # gpt-5-nano only supports temperature=1
    ) -> Optional[str]:
        """
        Generate a completion with an image input.
        
        Args:
            prompt: The prompt text
            image_base64: Base64 encoded image
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Returns:
            The response text or None if failed
        """
        if not self._openai_service:
            logger.error("OpenAI service not initialized")
            return None
        
        try:
            response = await self._openai_service.chat_completion_with_image_async(
                prompt=prompt,
                image_base64=image_base64,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            if response.get("success"):
                return response["response"]
            else:
                logger.error(f"LLM image call failed: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM image completion: {e}")
            return None
    
    def parse_single_question_response(
        self,
        response_text: str,
        question_type: QuestionType,
        subject: str,
    ) -> Optional[GeneratedQuestion]:
        """
        Parse LLM response for a single question.
        
        Args:
            response_text: The raw LLM response
            question_type: Expected question type
            subject: Subject for context
            
        Returns:
            GeneratedQuestion or None if parsing failed
        """
        try:
            json_str = self.extract_json(response_text)
            data = json.loads(json_str)
            
            # The response might be the question directly or wrapped
            if "questions" in data:
                q_data = data["questions"][0] if data["questions"] else None
            elif "question" in data:
                q_data = data["question"]
            else:
                q_data = data
            
            if not q_data:
                return None
            
            question = GeneratedQuestion.from_dict(q_data)
            question.question_type = question_type.value
            question.subject = subject
            question.source_type = GenerationSource.NOTES
            
            if question.diagram_spec:
                question.diagram_spec.subject = subject.lower()
            
            return question
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse single question response: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return None
    
    def parse_questions_response(
        self,
        response_text: str,
        question_type: QuestionType,
        subject: str,
        expected_count: int = 0,
    ) -> List[GeneratedQuestion]:
        """
        Parse LLM response for multiple questions.
        
        Args:
            response_text: The raw LLM response
            question_type: Expected question type
            subject: Subject for context
            expected_count: Expected number of questions (for validation)
            
        Returns:
            List of GeneratedQuestion objects
        """
        questions = []
        
        try:
            json_str = self.extract_json(response_text)
            data = json.loads(json_str)
            
            questions_data = data.get("questions", [])
            
            for q_data in questions_data:
                try:
                    question = self._parse_question_data(q_data, question_type, subject)
                    if question:
                        questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to parse question: {e}")
                    continue
            
            if expected_count > 0 and len(questions) < expected_count:
                logger.warning(
                    f"Expected {expected_count} questions but got {len(questions)} "
                    f"for type {question_type.value}"
                )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            logger.debug(f"Response: {response_text[:500]}...")
        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
        
        return questions
    
    def _parse_question_data(
        self,
        q_data: Dict[str, Any],
        question_type: QuestionType,
        subject: str,
    ) -> Optional[GeneratedQuestion]:
        """Parse individual question data dict into GeneratedQuestion."""
        try:
            # Parse options for MCQ
            options = None
            if question_type == QuestionType.MCQ and "options" in q_data:
                options = [
                    QuestionOption(
                        label=opt.get("label", ""),
                        content=opt.get("content", ""),
                        is_correct=opt.get("is_correct", False),
                    )
                    for opt in q_data.get("options", [])
                ]
            
            # Parse marking scheme
            marking_steps = None
            if "marking_scheme" in q_data:
                marking_steps = [
                    MarkingStep(
                        step=ms.get("step", ""),
                        marks=float(ms.get("marks", 0)),
                        criteria=ms.get("criteria", ""),
                    )
                    for ms in q_data.get("marking_scheme", [])
                ]
            
            # Parse diagram spec
            diagram_spec = None
            if q_data.get("has_diagram") and q_data.get("diagram_spec"):
                ds = q_data["diagram_spec"]
                diagram_spec = DiagramSpec(
                    subject=ds.get("subject", subject.lower()),
                    diagram_type=ds.get("diagram_type", ""),
                    title=ds.get("title"),
                    description=ds.get("description"),
                    parameters={k: v for k, v in ds.items() 
                               if k not in ["subject", "diagram_type", "title", "description"]}
                )
            
            question = GeneratedQuestion(
                question_text=q_data.get("question_text", ""),
                question_type=question_type.value,
                subject=subject,
                marks=int(q_data.get("marks", 1)),
                difficulty=q_data.get("difficulty", "medium"),
                bloom_level=q_data.get("bloom_level", "understand"),
                topic=q_data.get("topic", ""),
                options=options,
                correct_answer=q_data.get("correct_answer"),
                solution=q_data.get("solution"),
                solution_steps=q_data.get("solution_steps", []),
                marking_steps=marking_steps,
                has_diagram=q_data.get("has_diagram", False),
                diagram_spec=diagram_spec,
                source_type=GenerationSource.NOTES,
            )
            
            return question
            
        except Exception as e:
            logger.error(f"Error parsing question data: {e}")
            return None
    
    def extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM response that may contain markdown or other text.
        
        Args:
            text: Raw response text
            
        Returns:
            Extracted JSON string
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object or array
        # First try to find { ... } pattern
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            return brace_match.group()
        
        # Try to find [ ... ] pattern
        bracket_match = re.search(r'\[[\s\S]*\]', text)
        if bracket_match:
            return bracket_match.group()
        
        # Return original text if no JSON found
        return text.strip()
    
    def validate_json_response(self, response_text: str) -> bool:
        """
        Validate that the response contains valid JSON.
        
        Args:
            response_text: The response to validate
            
        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json_str = self.extract_json(response_text)
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, Exception):
            return False


# Global instance
_llm_handler: Optional[LLMHandler] = None


def get_llm_handler() -> LLMHandler:
    """Get or create the LLM handler singleton."""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler()
    return _llm_handler
