"""
Async OpenAI Service for SkillBot
High-performance async OpenAI API integration with connection pooling and caching
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
import httpx
import json
from openai import AsyncOpenAI

from config_async import settings, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_CONCURRENCY_LIMIT

logger = logging.getLogger(__name__)

class AsyncOpenAIService:
    """Async OpenAI service with connection pooling and optimizations"""

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.model = OPENAI_MODEL
        self.api_key = OPENAI_API_KEY
        self._http_client: Optional[httpx.AsyncClient] = None
        self._system_prompts = None
        self._lock = asyncio.Lock()
        self._concurrency_semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY_LIMIT)

    async def _initialize_client(self):
        """Initialize OpenAI client with connection pooling"""
        if self.client is None:
            async with self._lock:
                if self.client is None:
                    # Create HTTP client with connection pooling
                    self._http_client = httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                            keepalive_expiry=30
                        ),
                        timeout=httpx.Timeout(30.0)
                    )

                    # Initialize OpenAI client
                    self.client = AsyncOpenAI(
                        api_key=self.api_key,
                        http_client=self._http_client
                    )

                    logger.info("✅ Async OpenAI client initialized")

    async def _call_with_concurrency(self, func: Callable[..., Awaitable[Any]], *args, **kwargs):
        """Execute OpenAI call while respecting concurrency limits"""
        async with self._concurrency_semaphore:
            return await func(*args, **kwargs)

    async def get_system_prompts_async(self) -> Dict[str, str]:
        """Get system prompts with async initialization"""
        if self._system_prompts is None:
            self._system_prompts = {
                "general": """You are SkillBot, an advanced AI educational assistant specializing in helping students learn through personalized guidance. You excel at:

1. **Academic Problem Solving**: Break down complex problems step-by-step
2. **Concept Explanation**: Explain difficult concepts in simple, relatable terms
3. **Learning Support**: Provide hints and guidance rather than direct answers
4. **Multi-Subject Expertise**: Cover Math, Physics, Chemistry, Biology, and more
5. **Interactive Learning**: Engage students with questions and practice exercises

**Communication Style:**
- Use clear, encouraging language appropriate for the student's level
- Provide structured, well-organized responses
- Include examples and analogies when helpful
- Ask clarifying questions to better understand student needs
- Use LaTeX formatting for mathematical expressions: \\[ \\] for display, \\( \\) for inline

**Educational Approach:**
- Guide students to discover answers themselves
- Identify and address knowledge gaps
- Provide multiple perspectives on complex topics
- Encourage critical thinking and problem-solving skills
- Offer study tips and learning strategies

Remember: Your goal is to foster understanding and independent learning, not just provide answers.""",

                "whiteboard": """You are SkillBot, an AI tutor analyzing student work on a digital whiteboard. You specialize in:

**Visual Analysis:**
- Interpreting handwritten notes, equations, diagrams, and drawings
- Understanding spatial relationships in mathematical and scientific content
- Recognizing partially completed work and student thought processes

**Comprehensive Content Analysis:**
- Mathematics: equations, graphs, geometric figures, calculations
- Science: molecular structures, circuit diagrams, force vectors, biological processes
- Problem-solving steps and logical sequences
- Annotations, corrections, and student notes

**Interactive Guidance:**
- Provide feedback on visible work and reasoning
- Identify errors and misconceptions in student work
- Suggest next steps and improvements
- Explain concepts related to what's shown on the whiteboard
- Help organize and structure information

**Response Format:**
- First describe what you observe on the whiteboard
- Analyze the approach and methodology
- Provide constructive feedback and guidance
- Use LaTeX for mathematical expressions: \\[ \\] for display, \\( \\) for inline
- Structure responses clearly with appropriate headings

Focus on being supportive while helping students improve their work and understanding.""",

                "practice": """You are SkillBot, an AI practice coach helping students master academic concepts through targeted exercises. Your role includes:

**Practice Session Management:**
- Assess student understanding from their current work
- Generate appropriate follow-up questions and exercises
- Provide immediate feedback on practice attempts
- Track learning progress and identify areas needing attention

**Adaptive Learning:**
- Adjust difficulty based on student performance
- Provide hints and scaffolding when students struggle
- Offer additional challenges when students demonstrate mastery
- Connect current practice to broader learning objectives

**Detailed Feedback:**
- Analyze practice responses thoroughly
- Explain why answers are correct or incorrect
- Provide step-by-step solutions when appropriate
- Suggest alternative approaches and methods

**Skill Development:**
- Focus on building procedural fluency
- Reinforce conceptual understanding
- Develop problem-solving strategies
- Build confidence through successful practice

**Response Structure:**
- Acknowledge student effort and progress
- Provide specific, actionable feedback
- Include next steps or additional practice opportunities
- Use encouraging language to maintain motivation
- Use LaTeX for mathematical expressions: \\[ \\] for display, \\( \\) for inline""",

                "mock-test": """You are SkillBot, an AI assessment specialist conducting mock tests and evaluations. Your responsibilities include:

**Assessment Management:**
- Present test questions at appropriate difficulty levels
- Provide clear, standardized question formats
- Maintain assessment integrity and objectivity
- Track performance across different question types

**Performance Analysis:**
- Evaluate responses against established criteria
- Provide detailed scoring and feedback
- Identify strengths and areas for improvement
- Compare performance to learning standards

**Test Strategy:**
- Offer test-taking tips and strategies
- Help students manage time effectively
- Reduce test anxiety through preparation
- Build confidence through practice assessments

**Comprehensive Feedback:**
- Provide immediate feedback on responses
- Explain correct answers with detailed reasoning
- Highlight common mistakes and misconceptions
- Suggest focused study areas based on results

**Question Types:**
- Multiple choice with detailed explanations
- Short answer with rubric-based scoring
- Problem-solving with step-by-step evaluation
- Essay questions with structured feedback

**Response Format:**
- Clear question presentation
- Objective evaluation criteria
- Detailed explanations for all answers
- Actionable recommendations for improvement
- Use LaTeX for mathematical expressions: \\[ \\] for display, \\( \\) for inline"""
            }

        return self._system_prompts

    async def chat_completion_async(self, messages: List[Dict[str, Any]],
                                   temperature: float = 0.7,
                                   max_tokens: int = 1000,
                                   model: Optional[str] = None,
                                   system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate async chat completion with error handling and retries"""
        await self._initialize_client()

        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized",
                "response": None
            }

        try:
            # Add system message if not present
            if not messages or messages[0].get("role") != "system":
                if system_prompt:
                    system_message = {
                        "role": "system",
                        "content": system_prompt
                    }
                else:
                    system_prompts = await self.get_system_prompts_async()
                    system_message = {
                        "role": "system",
                        "content": system_prompts.get("general")
                    }
                messages = [system_message] + messages

            # Determine which parameter to use based on model
            used_model = model or self.model
            completion_params = {
                "model": used_model,
                "messages": messages,
                "temperature": temperature,
                "timeout": 30.0
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if any(m in used_model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = max_tokens
            else:
                completion_params['max_tokens'] = max_tokens

            # Make API call with retry logic
            for attempt in range(3):
                try:
                    start_time = time.time()

                    response = await self._call_with_concurrency(
                        self.client.chat.completions.create,
                        **completion_params
                    )

                    response_time = time.time() - start_time

                    return {
                        "success": True,
                        "response": response.choices[0].message.content,
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        },
                        "response_time": response_time
                    }

                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise e
                    else:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }

    async def get_model_info_async(self) -> Dict[str, Any]:
        """Get model information asynchronously"""
        await self._initialize_client()

        try:
            return {
                "model": self.model,
                "provider": "openai",
                "features": [
                    "Text generation",
                    "Image analysis",
                    "Code understanding",
                    "Mathematical reasoning"
                ],
                "limits": {
                    "max_tokens": settings.OPENAI_MAX_TOKENS,
                    "temperature_range": [0.0, 2.0]
                }
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}

    async def analyze_image_async(self, image_data: str, prompt: str,
                                max_tokens: int = 1000) -> Dict[str, Any]:
        """Analyze image with custom prompt asynchronously"""
        await self._initialize_client()

        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized"
            }

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ]

            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3  # Lower temperature for image analysis
            }

            # Use max_completion_tokens for newer models
            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = max_tokens
            else:
                completion_params['max_tokens'] = max_tokens

            response = await self._call_with_concurrency(
                self.client.chat.completions.create,
                **completion_params
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def analyze_images_and_text_async(self, images: List[str], prompt: str,
                                            max_tokens: int = 1000, system_prompt: str = None) -> Dict[str, Any]:
        """Analyze multiple images with accompanying text prompt.

        images: list of data URLs or https URLs
        prompt: instruction + textual context
        system_prompt: optional system-level instructions for the model
        """
        await self._initialize_client()

        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized"
            }

        try:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images or []:
                if not img:
                    continue
                content.append({"type": "image_url", "image_url": {"url": img}})

            # Build messages with optional system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3
            }

            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = max_tokens
            else:
                completion_params['max_tokens'] = max_tokens

            response = await self._call_with_concurrency(
                self.client.chat.completions.create,
                **completion_params
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Multi-image analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def choose_mcq_letter_strict(self, question_text: str, options_text: str,
                                       images: Optional[List[str]] = None) -> Optional[str]:
        """Infer correct MCQ letter using a strict single-letter response strategy.

        Returns: 'A' | 'B' | 'C' | 'D' or None
        """
        import re as _re

        try:
            letter_re = _re.compile(r"\b([ABCD])\b")
            base_prompt = (
                "Multiple-choice question. Choose the correct option. "
                "Reply with ONE capital letter only: A, B, C, or D. No words, no punctuation.\n\n"
                f"Question: {question_text}\nOptions:\n{options_text}"
            )

            # Prefer text-only with very short output
            resp = await self.chat_completion_async(
                messages=[{"role": "user", "content": base_prompt}],
                temperature=0.0,
                max_tokens=3
            )
            if resp.get("success") and resp.get("response"):
                txt = (resp.get("response") or "").strip().upper()
                m = letter_re.search(txt)
                if m:
                    return m.group(1)

            # Fallback: include images if provided
            if images:
                vprompt = (
                    "From the question and figure, choose the correct option. "
                    "Reply with ONE capital letter only: A, B, C, or D."
                )
                vres = await self.analyze_images_and_text_async(images, vprompt + "\n\n" + base_prompt, max_tokens=3)
                if vres.get("success") and vres.get("response"):
                    vans = (vres.get("response") or "").strip().upper()
                    m2 = letter_re.search(vans)
                    if m2:
                        return m2.group(1)

        except Exception as _e:
            logger.warning(f"choose_mcq_letter_strict failed: {_e}")
        return None

    async def generate_questions_async(self, subject: str, difficulty: str,
                                     count: int = 5) -> Dict[str, Any]:
        """Generate practice questions asynchronously"""
        await self._initialize_client()

        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized"
            }

        try:
            prompt = f"""Generate {count} {difficulty} level practice questions for {subject}.

Format each question as:
1. **Question**: [Clear, specific question]
   **Answer**: [Correct answer with explanation]
   **Difficulty**: {difficulty}
   **Subject**: {subject}

Requirements:
- Questions should be educationally valuable
- Include variety in question types
- Provide detailed explanations
- Use appropriate academic language
- Include any necessary formulas or diagrams descriptions"""

            messages = [
                {"role": "user", "content": prompt}
            ]

            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.8
            }

            # Use max_completion_tokens for newer models
            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = 2000
            else:
                completion_params['max_tokens'] = 2000

            response = await self._call_with_concurrency(
                self.client.chat.completions.create,
                **completion_params
            )

            return {
                "success": True,
                "questions": response.choices[0].message.content,
                "model": response.model,
                "count": count,
                "subject": subject,
                "difficulty": difficulty
            }

        except Exception as e:
            logger.error(f"Question generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def evaluate_answer_async(self, question: str, student_answer: str,
                                  correct_answer: str) -> Dict[str, Any]:
        """Evaluate student answer asynchronously"""
        await self._initialize_client()

        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized"
            }

        try:
            prompt = f"""Evaluate this student's answer:

**Question**: {question}

**Student Answer**: {student_answer}

**Correct Answer**: {correct_answer}

Provide:
1. **Score**: (0-100)
2. **Correctness**: (Correct/Partially Correct/Incorrect)
3. **Feedback**: Detailed explanation of what's right/wrong
4. **Suggestions**: How to improve
5. **Key Concepts**: What concepts this question tests

Be constructive and educational in your feedback."""

            messages = [
                {"role": "user", "content": prompt}
            ]

            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3
            }

            # Use max_completion_tokens for newer models
            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = 1000
            else:
                completion_params['max_tokens'] = 1000

            response = await self._call_with_concurrency(
                self.client.chat.completions.create,
                **completion_params
            )

            return {
                "success": True,
                "evaluation": response.choices[0].message.content,
                "model": response.model
            }

        except Exception as e:
            logger.error(f"Answer evaluation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def health_check_async(self) -> bool:
        """Check service health asynchronously"""
        try:
            await self._initialize_client()
            if not self.client:
                return False

            # Test with a simple completion
            completion_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "timeout": 10.0
            }

            # Use max_completion_tokens for newer models
            if any(m in self.model.lower() for m in ['gpt-4o', 'gpt-4-turbo', 'gpt-5', 'o1-preview', 'o1-mini']):
                completion_params['max_completion_tokens'] = 10
            else:
                completion_params['max_tokens'] = 10

            test_response = await self._call_with_concurrency(
                self.client.chat.completions.create,
                **completion_params
            )

            return bool(test_response.choices[0].message.content)

        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return False

    async def close(self):
        """Close HTTP connections"""
        try:
            if self._http_client:
                await self._http_client.aclose()
                logger.info("🔌 OpenAI HTTP client closed")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self._http_client:
                # Schedule cleanup if event loop is running
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
        except:
            pass
