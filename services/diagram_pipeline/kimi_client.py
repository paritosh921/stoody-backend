"""
Kimi 2.5 LLM Client for Diagram Pipeline

Handles both:
- LLM1: Generating diagram instructions from question text
- LLM2: Reviewing generated diagrams and providing feedback

Constraints:
- Uses ONLY Kimi 2.5 model
- Temperature: 0.6 (fixed)
"""

import json
import logging
import httpx
import os
from typing import Optional, Dict, Any
from datetime import datetime

from .models import (
    DiagramInstructions,
    DiagramReview,
    ReviewIssue,
    IssueCode,
    SubjectType,
)
from config_async import KIMI_API_URL, KIMI_API_KEY

logger = logging.getLogger(__name__)

# Kimi K2.5 API Configuration
KIMI_MODEL = "kimi-k2.5"  # Kimi K2.5
KIMI_TEMPERATURE = 0.6  # Instant mode (thinking disabled) uses 0.6
KIMI_BASE_URL = (KIMI_API_URL or "https://api.moonshot.ai/v1").rstrip("/")
KIMI_FALLBACK_BASE_URLS = [
    "https://api.moonshot.ai/v1",
    "https://api.moonshot.cn/v1",
]


class KimiClient:
    """
    Client for Kimi 2.5 API.
    
    Used for both instruction generation (LLM1) and review (LLM2).
    """
    
    def __init__(self):
        # Endpoint candidates:
        # 1) configured URL, 2) common Moonshot Open Platform URLs.
        endpoint_candidates = []
        configured_url = (KIMI_BASE_URL or "").strip().rstrip("/")
        if configured_url:
            endpoint_candidates.append(configured_url)
        for fallback_url in KIMI_FALLBACK_BASE_URLS:
            normalized = fallback_url.strip().rstrip("/")
            if normalized and normalized not in endpoint_candidates:
                endpoint_candidates.append(normalized)
        self.api_urls = endpoint_candidates
        self.api_url = self.api_urls[0] if self.api_urls else "https://api.moonshot.ai/v1"

        # Accept both env var names and sanitize common copy/paste issues.
        raw_key = KIMI_API_KEY or os.getenv("MOONSHOT_API_KEY", "")
        raw_key = (raw_key or "").strip().strip('"').strip("'")
        if raw_key.lower().startswith("bearer "):
            raw_key = raw_key.split(" ", 1)[1].strip()
        self.api_key = raw_key
        self.model = KIMI_MODEL
        self.temperature = KIMI_TEMPERATURE
        
        if not self.api_key:
            logger.warning("KIMI_API_KEY not set - diagram pipeline will not work")
        elif not self.api_key.startswith("sk-"):
            logger.warning(
                "Kimi key format looks unusual (expected prefix 'sk-'). "
                "Verify KIMI_API_KEY or MOONSHOT_API_KEY."
            )
    
    async def _call_api(
        self,
        messages: list,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Make a call to the Kimi K2.5 API with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            The assistant's response text
        """
        import asyncio
        
        if not self.api_key:
            raise ValueError("KIMI_API_KEY not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Instant mode (thinking disabled): temperature=0.6, top_p=0.95
        # Must explicitly pass thinking: {"type": "disabled"} for instant mode
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "thinking": {"type": "disabled"},  # Required for instant mode with temp=0.6
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        api_endpoints = [f"{base}/chat/completions" for base in self.api_urls] or [
            "https://api.moonshot.ai/v1/chat/completions"
        ]

        logger.info(
            f"Calling Kimi K2.5 API with {len(api_endpoints)} endpoint candidate(s). "
            f"Primary endpoint: {api_endpoints[0]}"
        )
        logger.debug(f"Payload: model={self.model}, temp={self.temperature}, max_tokens={max_tokens}")

        # Retry logic for timeouts
        max_retries = 3
        base_timeout = 90.0  # 90 seconds per attempt
        auth_errors = []

        for endpoint_idx, api_endpoint in enumerate(api_endpoints):
            for attempt in range(max_retries):
                try:
                    timeout = base_timeout * (attempt + 1)  # Increase timeout with each retry
                    logger.info(
                        f"Kimi API endpoint {endpoint_idx + 1}/{len(api_endpoints)}, "
                        f"attempt {attempt + 1}/{max_retries}, timeout={timeout}s"
                    )

                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(
                            api_endpoint,
                            headers=headers,
                            json=payload,
                        )

                        if response.status_code == 401:
                            # Usually not retryable on same endpoint; try the next endpoint.
                            error_text = response.text
                            logger.error(f"Kimi API auth error at {api_endpoint}: {error_text}")
                            auth_errors.append(f"{api_endpoint} -> {error_text}")
                            break

                        if response.status_code != 200:
                            error_text = response.text
                            logger.error(f"Kimi API error: {response.status_code} - {error_text}")
                            raise Exception(f"Kimi API error {response.status_code}: {error_text}")

                        data = response.json()
                        return data["choices"][0]["message"]["content"]

                except httpx.TimeoutException as e:
                    logger.warning(f"Kimi API timeout on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error(f"Kimi API timeout after {max_retries} attempts for endpoint {api_endpoint}")
                    # Try next endpoint before failing.
                    break
                except httpx.RequestError as e:
                    logger.warning(f"Kimi API request error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)
                        continue
                    logger.error(f"Kimi API request error at endpoint {api_endpoint}: {e}")
                    # Try next endpoint before failing.
                    break
                except KeyError as e:
                    logger.error(f"Kimi API response parsing error: {e}")
                    raise Exception(f"Kimi API response missing expected field: {str(e)}")
                except Exception as e:
                    logger.error(f"Kimi API unexpected error: {type(e).__name__}: {e}")
                    raise

        # Exhausted all endpoints.
        if auth_errors:
            raise Exception(
                "Kimi API authentication failed on all endpoints. "
                "Verify KIMI_API_KEY/MOONSHOT_API_KEY belongs to Moonshot Open Platform, "
                "is active, and has no extra spaces/quotes. "
                f"Details: {' | '.join(auth_errors)}"
            )
        raise Exception("Kimi API request failed on all configured endpoints")


class LLM1InstructionGenerator(KimiClient):
    """
    LLM1: Generates diagram instructions from question text.
    
    Creates clean, precise, exam-style instructions for Nano Banan Pro.
    """
    
    SYSTEM_PROMPT = """You are a Technical Illustrator for high-stakes exams (JEE/NEET).
Your task is to describe the VISUAL SCENE to a blind artist who will draw it.
DO NOT solve the problem. DO NOT explain the concept. DESCRIBE THE DRAWING.

STRICT VISUAL RULES:
1. IMAGERY ONLY: Describe shapes, lines, arrows, and spatial layout.
2. LABELS: Specify exactly what text to place and where (e.g., "label 'A' at top vertex").
3. NO CLUTTER: Do not include decorative elements.
4. EXAM STYLE: Black lines on white background. High contrast.
5. FIDELITY: Do NOT add elements, labels, or values not stated or directly implied by the question.
6. COVERAGE: Include ALL named points/labels and ALL numeric values (angles, lengths, coordinates, forces, resistances, etc.) from the question.
7. CONFLICTS: If hints conflict with the question, ignore the hints and follow the question.
8. LABEL SAFETY: Do NOT treat grammatical articles (e.g., leading "A", "An", "The") as diagram labels.
9. ARROWS: Add arrows only when direction is explicitly relevant (forces, velocity, acceleration, current, ray direction, etc.).

SUBJECT SPECIFIC GUIDES:
- GEOMETRY: Describe the exact shape (e.g. "a triangle where top angle is obtuse"), label vertices, mark known angles/lengths visually.
- PHYSICS: Describe the setup (e.g. "block on inclined plane"), force vectors (arrows), and circuit components clearly.
- CHEMISTRY: Describe the molecular structure clearly (bonds, atoms).

OUTPUT FORMAT:
Return a single paragraph describing the visual elements to be drawn.
Example: "Draw a right-angled triangle ABC with the right angle at B. AB is vertical, BC is horizontal. Label A at top, B at bottom-left, C at bottom-right. Draw a circle inscribed within the triangle."
"""

    async def generate_instructions(
        self,
        question_text: str,
        subject: Optional[SubjectType] = None,
        diagram_hints: Optional[str] = None,
        previous_instructions: Optional[str] = None,
        review_feedback: Optional[str] = None,
        required_labels: Optional[list] = None,
        required_numbers: Optional[list] = None,
    ) -> DiagramInstructions:
        """
        Generate diagram instructions for a question.
        
        Args:
            question_text: The question requiring a diagram
            subject: The subject (physics/chemistry/maths/biology)
            diagram_hints: Optional hints about what diagram to create
            previous_instructions: Previous instructions if refining
            review_feedback: Feedback from LLM2 review if refining
            required_labels: Explicit labels that must appear (optional)
            required_numbers: Explicit numeric values to include (optional)
        
        Returns:
            DiagramInstructions object
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Build the user message
        user_content = f"Question: {question_text}\n"
        
        if subject:
            user_content += f"Subject: {subject.value}\n"
        
        if diagram_hints:
            user_content += f"Hints (use only if consistent with the question): {diagram_hints}\n"

        if required_labels:
            user_content += f"Required labels (must appear exactly): {', '.join(required_labels)}\n"
        if required_numbers:
            user_content += f"Required numeric values (include where diagram-relevant): {', '.join(required_numbers)}\n"
        
        if previous_instructions and review_feedback:
            user_content += f"\n--- REFINEMENT REQUEST ---\n"
            user_content += f"Previous instructions that need improvement:\n{previous_instructions}\n\n"
            user_content += f"Review feedback:\n{review_feedback}\n\n"
            user_content += "Please write improved instructions that address the feedback."
        else:
            user_content += "\nWrite the diagram instructions for this question."
        
        messages.append({"role": "user", "content": user_content})
        
        # Call Kimi 2.5
        instruction_text = await self._call_api(messages, max_tokens=1000)
        
        # Clean up the response
        instruction_text = instruction_text.strip()
        if instruction_text.startswith('"') and instruction_text.endswith('"'):
            instruction_text = instruction_text[1:-1]
        
        version = 1
        refined_from = None
        
        if previous_instructions:
            # This is a refinement, increment version
            version = 2  # Will be updated by caller if needed
            refined_from = 1
        
        return DiagramInstructions(
            instructions=instruction_text,
            version=version,
            subject=subject,
            refined_from_version=refined_from,
            refinement_reason=review_feedback[:200] if review_feedback else None,
        )


class LLM2DiagramReviewer(KimiClient):
    """
    LLM2: Reviews generated diagrams and provides feedback.
    
    Determines if the diagram is acceptable for exam use and suggests improvements.
    """
    
    SYSTEM_PROMPT = """You are an expert exam diagram reviewer for JEE/NEET questions.

Your job is to review diagram instructions AND the rendered diagram image (if provided) to determine if the diagram is acceptable for exam use.

REVIEW CRITERIA:
1. CLARITY: Is the diagram clear and legible when printed? Check for overlapping text, thin lines, or cluttered regions.
2. LABELS: Are all necessary labels present, correctly placed, and readable? No overlaps?
3. ACCURACY: Does the rendered diagram correctly represent what the question describes?
4. SIMPLICITY: Is it simple enough for an exam? No unnecessary decoration or clutter?
5. COMPLETENESS: Are all required elements (shapes, labels, arrows, values) present?
6. COVERAGE: Only flag MISSING_LABEL/INCOMPLETE_DIAGRAM if a label/value appears in the question BUT is absent from the instructions or the rendered image.
7. VISUAL QUALITY: If a rendered image is provided, verify that shapes are correctly drawn, labels are legible, lines are clean, and the spatial layout matches the instructions.
8. CONSISTENCY: If both instructions and a rendered image are available, flag any discrepancy where the image does not match the instructions.
9. EVIDENCE: If you flag an issue, cite the exact missing/incorrect element with specifics (e.g., "label 'B' is overlapping with the angle mark at vertex B").
10. LABEL SAFETY: Do NOT treat grammatical articles (e.g., sentence-start "A") as labels.
11. ARROWS: Flag MISSING_ARROW only when the question explicitly requires directional arrows (force/velocity/acceleration/current/rays, etc.).

ISSUE CODES (use exactly these):
- LABEL_OVERLAP: Labels overlap each other or other elements
- MISSING_LABEL: A required label is missing from the diagram
- WRONG_ANGLE: Angles or orientations are visually incorrect
- LOW_CLARITY: The diagram is hard to read (thin lines, small text, poor contrast)
- WRONG_CONFIGURATION: Elements are incorrectly arranged or positioned
- TOO_CLUTTERED: Too many elements or unnecessary decoration
- MISSING_ARROW: Required direction indicators are missing
- WRONG_PROPORTION: Size ratios or relative dimensions are wrong
- INCOMPLETE_DIAGRAM: Missing required shapes, components, or structural elements
- STYLE_ISSUE: Not suitable for exam printing (color, background, decorative elements)
- OTHER: Other issues not covered above

OUTPUT FORMAT (JSON only):
{
  "is_acceptable": true or false,
  "issues": [
    {
      "code": "ISSUE_CODE",
      "details": "Brief explanation citing the specific element"
    }
  ],
  "suggested_instruction_update": "Revised instruction text that fixes the issues, or null if acceptable"
}

Be STRICT but FAIR. Only mark as unacceptable if there are real problems that would affect exam usability.
Output ONLY the JSON, no other text."""

    async def review_diagram(
        self,
        question_text: str,
        current_instructions: str,
        image_description: Optional[str] = None,
        subject: Optional[SubjectType] = None,
        image_base64: Optional[str] = None,
    ) -> DiagramReview:
        """
        Review a diagram's instructions and (optionally) the rendered image.

        When *image_base64* is provided the rendered PNG is sent directly to
        KIMI K2.5 so the reviewer can visually inspect shapes, labels, and
        layout in addition to reasoning over the textual instructions.

        Args:
            question_text: The original question
            current_instructions: The instructions used to generate the diagram
            image_description: Optional text description of the generated image
            subject: The subject for context
            image_base64: Optional base64-encoded PNG of the rendered diagram
        
        Returns:
            DiagramReview object
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Build the textual part of the user message
        user_text = f"Question: {question_text}\n\n"
        user_text += f"Diagram Instructions:\n{current_instructions}\n"
        
        if subject:
            user_text += f"\nSubject: {subject.value}"
        
        if image_description:
            user_text += f"\n\nGenerated Image Description:\n{image_description}"

        if image_base64:
            user_text += (
                "\n\nThe rendered diagram image is attached below. "
                "Visually inspect it against the question and instructions, "
                "then provide your assessment as JSON."
            )
        else:
            user_text += "\n\nReview these instructions and provide your assessment as JSON."

        # Build the user message — multimodal when an image is available
        if image_base64:
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                },
            ]
        else:
            user_content = user_text

        messages.append({"role": "user", "content": user_content})
        
        # Call Kimi 2.5 with JSON response format
        response_text = await self._call_api(
            messages, 
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        
        # Parse the JSON response
        try:
            review_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                # Fallback to accepting if we can't parse
                logger.warning(f"Could not parse review response: {response_text}")
                return DiagramReview(
                    is_acceptable=True,
                    issues=[],
                    suggested_instruction_update=None,
                )
        
        # Build the review object
        issues = []
        for issue_data in review_data.get("issues", []):
            code_str = issue_data.get("code", "OTHER")
            try:
                code = IssueCode(code_str)
            except ValueError:
                code = IssueCode.OTHER
            
            issues.append(ReviewIssue(
                code=code,
                details=issue_data.get("details", ""),
            ))
        
        return DiagramReview(
            is_acceptable=review_data.get("is_acceptable", True),
            issues=issues,
            suggested_instruction_update=review_data.get("suggested_instruction_update"),
        )
