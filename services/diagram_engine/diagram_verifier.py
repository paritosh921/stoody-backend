"""
Diagram Verifier - LLM-based verification of generated diagrams.

This module provides verification of diagrams against their corresponding questions
using vision capabilities. If a diagram doesn't correctly represent the question,
it provides feedback for regeneration.

Enhanced with:
- Structured feedback (what/why/fix)
- Category-based scoring (conceptual, labeling, visual, alignment, academic)
- JEE/NEET academic standards checking
- Support for DiagramPlan verification
"""

import base64
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .diagram_planner import DiagramPlan

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of diagram verification."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    ERROR = "error"


class VerificationCategory(str, Enum):
    """Categories for verification issues"""
    CONCEPTUAL = "conceptual"      # Physics/math/chem principles
    LABELING = "labeling"          # Text, units, symbols
    VISUAL = "visual"              # Clarity, clutter, proportions
    ALIGNMENT = "alignment"        # Match with question text
    ACADEMIC = "academic"          # JEE/NEET standards


class DetailedIssue(BaseModel):
    """Structured issue with what/why/fix"""
    category: VerificationCategory
    severity: str = Field(..., description="critical, major, or minor")
    what: str = Field(..., description="What is wrong")
    why: str = Field(..., description="Why it matters for students")
    fix: str = Field(..., description="Specific correction needed")


class EnhancedVerificationResult(BaseModel):
    """Enhanced verification result with structured feedback"""
    status: VerificationStatus
    is_acceptable: bool
    confidence: float = 0.5
    
    # Category scores (0-1)
    conceptual_score: float = 0.5
    labeling_score: float = 0.5
    visual_score: float = 0.5
    alignment_score: float = 0.5
    academic_score: float = 0.5
    
    # Structured issues
    issues: List[DetailedIssue] = Field(default_factory=list)
    
    # Prioritized actions
    critical_fixes: List[str] = Field(default_factory=list)
    suggested_improvements: List[str] = Field(default_factory=list)
    
    # For regeneration
    regeneration_guidance: str = ""
    feedback: str = ""  # Legacy field for compatibility
    
    class Config:
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "is_acceptable": self.is_acceptable,
            "confidence": self.confidence,
            "scores": {
                "conceptual": self.conceptual_score,
                "labeling": self.labeling_score,
                "visual": self.visual_score,
                "alignment": self.alignment_score,
                "academic": self.academic_score,
            },
            "issues": [i.model_dump() for i in self.issues],
            "critical_fixes": self.critical_fixes,
            "suggested_improvements": self.suggested_improvements,
            "regeneration_guidance": self.regeneration_guidance,
            "feedback": self.feedback,
        }
    
    def get_composite_score(self) -> float:
        """Calculate weighted composite score"""
        weights = {
            "conceptual": 0.35,
            "alignment": 0.25,
            "labeling": 0.20,
            "visual": 0.10,
            "academic": 0.10
        }
        
        score = (
            self.conceptual_score * weights["conceptual"] +
            self.alignment_score * weights["alignment"] +
            self.labeling_score * weights["labeling"] +
            self.visual_score * weights["visual"] +
            self.academic_score * weights["academic"]
        )
        
        # Penalty for critical issues
        critical_count = len([i for i in self.issues if i.severity == "critical"])
        score -= critical_count * 0.15
        
        return max(0.0, min(1.0, score))


@dataclass
class VerificationResult:
    """Legacy verification result for compatibility."""
    status: VerificationStatus
    is_acceptable: bool
    feedback: Optional[str] = None
    issues: Optional[list] = None
    suggestions: Optional[list] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "is_acceptable": self.is_acceptable,
            "feedback": self.feedback,
            "issues": self.issues or [],
            "suggestions": self.suggestions or [],
            "confidence": self.confidence
        }


VERIFICATION_PROMPT = """You are a STRICT educational diagram reviewer for exam papers. Your job is to ensure diagrams are CLEAR and will NOT CONFUSE students.

QUESTION:
{question_text}

SUBJECT: {subject}

{diagram_description}

=== VERIFICATION CHECKLIST ===

1. CORRECTNESS
   - Does the diagram match the question EXACTLY?
   - Are all numerical values correct?
   - Are proportions reasonably accurate?
   - Is the concept being shown correct?

2. CLARITY (This is CRITICAL!)
   - Can a student understand the diagram at a glance?
   - Are labels large enough to read?
   - Are labels positioned clearly (not overlapping)?
   - Is there adequate spacing between elements?

3. COMPLETENESS
   - Are all important elements from the question shown?
   - Are all elements properly labeled?
   - Are units included where needed?

4. NO CONFUSION
   - Could ANY part of this diagram mislead a student?
   - Are there any ambiguous representations?
   - Is anything drawn in a misleading way?

=== REJECTION CRITERIA (REJECT if ANY of these apply) ===

MUST REJECT:
- Wrong numerical values (would cause wrong calculations)
- Missing critical elements (forces not shown, components missing)
- Misleading proportions (something twice as big drawn same size)
- Unlabeled important elements
- Cluttered/messy layout that's hard to read
- Labels too small or overlapping
- Confusing representation that could mislead students
- Incorrect diagram type for the question

ACCEPT:
- Clean, clear diagrams with all required elements
- Correctly labeled with readable text
- Accurate representation of the concept
- Would genuinely help students understand

=== RESPONSE FORMAT ===
Respond in JSON format ONLY (no markdown):
{{
    "status": "correct" | "incorrect" | "partially_correct",
    "is_acceptable": true | false,
    "confidence": 0.0-1.0,
    "issues": ["list every problem you find - be specific"],
    "suggestions": ["concrete fixes: what EXACTLY should change"],
    "feedback": "Main problem in one sentence"
}}

IMPORTANT: Be STRICT. A diagram that confuses students is worse than no diagram.
If there are clarity issues that could confuse students, mark is_acceptable=false.
"""


# Enhanced verification prompt with structured feedback
ENHANCED_VERIFICATION_PROMPT = """You are an expert JEE/NEET diagram reviewer. Evaluate this diagram with STRICT academic standards.

QUESTION:
{question_text}

SUBJECT: {subject}
{diagram_description}
{plan_section}

=== EVALUATION CRITERIA (JEE/NEET STANDARD) ===

1. CONCEPTUAL CORRECTNESS (Most Critical - 35% weight)
   - Are physics/math/chemistry principles correctly represented?
   - Are numerical values from the question shown accurately?
   - Are proportions and relationships correct?
   - Would this diagram HELP or HURT a student's understanding?
   - For circuits: Are all components connected correctly with proper symbols?
   - For chemistry: Are molecular structures/bonds accurate?
   - For physics: Are force directions, angles, and vectors correct?

2. QUESTION ALIGNMENT (25% weight)
   - Does the diagram match ALL details in the question?
   - Are ALL specified values shown (masses, angles, distances, etc.)?
   - Is the scenario correctly interpreted?
   - Are there any omissions from the question?
   - Are the GIVEN values shown (not calculated results)?

3. LABELING STANDARDS (20% weight)
   - Are all important elements labeled?
   - Are labels readable (fontsize >= 14)?
   - Do labels include units where needed (N, kg, m, etc.)?
   - Are standard symbols used (F for force, θ for angle, m for mass)?
   - Are labels positioned clearly (not overlapping shapes or each other)?

4. VISUAL CLARITY (10% weight)
   - Is the diagram clean and uncluttered?
   - Is there adequate spacing between elements?
   - Do labels overlap with shapes or lines?
   - Are proportions visually reasonable?
   - Are colors used appropriately (not too many, high contrast)?
   - Is the diagram resolution and quality good?
   - Are lines crisp and shapes well-defined?

5. ACADEMIC STANDARDS - JEE/NEET (10% weight)
   - Does this follow standard textbook conventions?
   - Is this how the concept is typically shown in exams?
   - Is complexity appropriate (not over-simplified, not cluttered)?
   - Are standard diagram conventions followed (e.g., circuit symbols)?

=== FOR EACH ISSUE FOUND ===
Provide structured feedback:
- WHAT: Exactly what is wrong (be specific)
- WHY: Why this matters for student understanding
- FIX: Specific instruction to correct it

=== RESPONSE FORMAT (JSON) ===
{{
    "status": "correct" | "incorrect" | "partially_correct",
    "is_acceptable": true | false,
    "confidence": 0.0-1.0,
    
    "scores": {{
        "conceptual": 0.0-1.0,
        "labeling": 0.0-1.0,
        "visual": 0.0-1.0,
        "alignment": 0.0-1.0,
        "academic": 0.0-1.0
    }},
    
    "issues": [
        {{
            "category": "conceptual|labeling|visual|alignment|academic",
            "severity": "critical|major|minor",
            "what": "description of the problem",
            "why": "why this matters for students",
            "fix": "specific correction needed"
        }}
    ],
    
    "critical_fixes": ["must do 1", "must do 2"],
    "suggested_improvements": ["nice to have 1"],
    
    "regeneration_guidance": "Overall guidance for creating a better diagram",
    "feedback": "Main problem in one sentence"
}}

=== DECISION RULES ===
- If ANY conceptual issue exists → is_acceptable = false
- If conceptual_score < 0.7 → is_acceptable = false
- If alignment_score < 0.6 → is_acceptable = false
- If 2+ critical issues → is_acceptable = false
- Otherwise, accept if composite score >= 0.7

IMPORTANT:
- Be STRICT on conceptual correctness - a wrong diagram is worse than no diagram
- Prioritize: Correctness > Clarity > Beauty
- Students' understanding is the ultimate goal
"""


class DiagramVerifier:
    """
    Verifies generated diagrams against their questions using LLM vision.
    
    This class provides:
    - Image analysis to check if diagram matches question
    - Detailed feedback for regeneration (what/why/fix structure)
    - Category-based scoring (conceptual, labeling, visual, alignment, academic)
    - JEE/NEET academic standards verification
    - Confidence scoring
    """
    
    def __init__(self, openai_service=None, kimi_service=None, use_kimi: bool = True):
        """
        Initialize the verifier.

        Args:
            openai_service: AsyncOpenAIService instance with vision capability (fallback)
            kimi_service: KimiService instance for vision (preferred)
            use_kimi: Whether to use Kimi for verification (default: True)
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None
        self._max_retries = 2  # Max parsing retries for JSON response

        if self._use_kimi:
            logger.info("DiagramVerifier initialized with Kimi K2.5 for vision")
        else:
            logger.info("DiagramVerifier initialized with OpenAI for vision")
    
    async def verify_diagram(
        self,
        image_bytes: bytes,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify if a diagram correctly represents the question.
        
        Args:
            image_bytes: PNG image data of the diagram
            question_text: The question text the diagram should represent
            subject: Subject area (physics, maths, etc.)
            diagram_description: Optional description of what the diagram should show
            
        Returns:
            VerificationResult with status, feedback, and suggestions
        """
        try:
            # Convert image to base64 data URL
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_base64}"
            
            # Build the verification prompt
            desc_section = ""
            if diagram_description:
                desc_section = f"\nDIAGRAM SHOULD SHOW:\n{diagram_description}"
            
            prompt = VERIFICATION_PROMPT.format(
                question_text=question_text,
                subject=subject.upper(),
                diagram_description=desc_section
            )
            
            # Call vision API (Kimi or OpenAI)
            if self._use_kimi and self._kimi_service:
                response = await self._kimi_service.analyze_image_async(
                    image_data=image_data_url,
                    prompt=prompt,
                    max_tokens=1000
                )
            else:
                response = await self._openai_service.analyze_image_async(
                    image_data=image_data_url,
                    prompt=prompt,
                    max_tokens=1000
                )
            
            if not response.get("success"):
                logger.error(f"Vision API failed: {response.get('error')}")
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    is_acceptable=False,
                    feedback=f"Verification failed: {response.get('error')}"
                )
            
            # Parse the JSON response
            return self._parse_verification_response(response.get("response", ""))
            
        except Exception as e:
            logger.error(f"Diagram verification error: {e}", exc_info=True)
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_acceptable=False,
                feedback=f"Verification error: {str(e)}"
            )
    
    def _parse_verification_response(self, response_text: str) -> VerificationResult:
        """Parse the LLM's JSON response into a VerificationResult."""
        try:
            # Try to extract JSON from the response
            json_str = response_text.strip()
            
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Map status string to enum
            status_map = {
                "correct": VerificationStatus.CORRECT,
                "incorrect": VerificationStatus.INCORRECT,
                "partially_correct": VerificationStatus.PARTIALLY_CORRECT
            }
            status = status_map.get(data.get("status", "").lower(), VerificationStatus.INCORRECT)
            
            return VerificationResult(
                status=status,
                is_acceptable=data.get("is_acceptable", False),
                feedback=data.get("feedback"),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                confidence=float(data.get("confidence", 0.5))
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification JSON: {e}")
            # Try to infer from text response
            response_lower = response_text.lower()
            if "correct" in response_lower and "incorrect" not in response_lower:
                return VerificationResult(
                    status=VerificationStatus.CORRECT,
                    is_acceptable=True,
                    feedback="Diagram appears correct",
                    confidence=0.6
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.INCORRECT,
                    is_acceptable=False,
                    feedback=response_text[:500],  # Use raw response as feedback
                    confidence=0.4
                )
    
    async def verify_diagram_enhanced(
        self,
        image_bytes: bytes,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None,
        plan: Optional["DiagramPlan"] = None
    ) -> EnhancedVerificationResult:
        """
        Enhanced verification with structured feedback and category scoring.
        
        Args:
            image_bytes: PNG image data of the diagram
            question_text: The question text the diagram should represent
            subject: Subject area (physics, maths, etc.)
            diagram_description: Optional description of what the diagram should show
            plan: Optional DiagramPlan to verify against
            
        Returns:
            EnhancedVerificationResult with detailed feedback and scores
        """
        try:
            # Convert image to base64 data URL
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_base64}"
            
            # Build the verification prompt
            desc_section = ""
            if diagram_description:
                desc_section = f"\nDIAGRAM SHOULD SHOW:\n{diagram_description}"
            
            plan_section = ""
            if plan:
                plan_section = f"\n\n=== PLANNED ELEMENTS (should all be present) ===\n"
                plan_section += f"Extracted Values: {plan.extracted_values}\n"
                plan_section += f"Planned Objects: {len(plan.objects)}\n"
                plan_section += f"Planned Labels: {len(plan.labels)}\n"
                if plan.academic_notes:
                    plan_section += f"Academic Notes: {plan.academic_notes}\n"
            
            prompt = ENHANCED_VERIFICATION_PROMPT.format(
                question_text=question_text,
                subject=subject.upper(),
                diagram_description=desc_section,
                plan_section=plan_section
            )
            
            # Call vision API (Kimi or OpenAI)
            if self._use_kimi and self._kimi_service:
                response = await self._kimi_service.analyze_image_async(
                    image_data=image_data_url,
                    prompt=prompt,
                    max_tokens=1500  # More tokens for detailed response
                )
            else:
                response = await self._openai_service.analyze_image_async(
                    image_data=image_data_url,
                    prompt=prompt,
                    max_tokens=1500  # More tokens for detailed response
                )
            
            if not response.get("success"):
                logger.error(f"Vision API failed: {response.get('error')}")
                return EnhancedVerificationResult(
                    status=VerificationStatus.ERROR,
                    is_acceptable=False,
                    feedback=f"Verification failed: {response.get('error')}"
                )
            
            # Parse the enhanced JSON response
            return self._parse_enhanced_verification_response(response.get("response", ""))
            
        except Exception as e:
            logger.error(f"Enhanced diagram verification error: {e}", exc_info=True)
            return EnhancedVerificationResult(
                status=VerificationStatus.ERROR,
                is_acceptable=False,
                feedback=f"Verification error: {str(e)}"
            )
    
    def _parse_enhanced_verification_response(self, response_text: str) -> EnhancedVerificationResult:
        """Parse enhanced verification response into EnhancedVerificationResult."""
        import re
        
        try:
            # Extract JSON from response
            json_str = response_text.strip()
            
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                parts = json_str.split("```")
                if len(parts) >= 2:
                    json_str = parts[1].strip()
            
            # Try to find JSON object using regex if direct parsing fails
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response")
            
            # Map status string to enum
            status_map = {
                "correct": VerificationStatus.CORRECT,
                "incorrect": VerificationStatus.INCORRECT,
                "partially_correct": VerificationStatus.PARTIALLY_CORRECT
            }
            status = status_map.get(data.get("status", "").lower(), VerificationStatus.INCORRECT)
            
            # Parse scores
            scores = data.get("scores", {})
            
            # Parse detailed issues
            issues = []
            for issue_data in data.get("issues", []):
                try:
                    category = VerificationCategory(issue_data.get("category", "conceptual"))
                    issues.append(DetailedIssue(
                        category=category,
                        severity=issue_data.get("severity", "major"),
                        what=issue_data.get("what", "Unknown issue"),
                        why=issue_data.get("why", "Could affect student understanding"),
                        fix=issue_data.get("fix", "Review and correct")
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse issue: {e}")
            
            # Determine if acceptable based on scores if not explicitly provided
            is_acceptable = data.get("is_acceptable", None)
            if is_acceptable is None:
                # Calculate based on scores - acceptable if conceptual >= 0.7 and no critical issues
                conceptual = float(scores.get("conceptual", 0.5))
                has_critical = any(i.severity == "critical" for i in issues)
                is_acceptable = conceptual >= 0.7 and not has_critical
            
            return EnhancedVerificationResult(
                status=status,
                is_acceptable=is_acceptable,
                confidence=float(data.get("confidence", 0.5)),
                conceptual_score=float(scores.get("conceptual", 0.5)),
                labeling_score=float(scores.get("labeling", 0.5)),
                visual_score=float(scores.get("visual", 0.5)),
                alignment_score=float(scores.get("alignment", 0.5)),
                academic_score=float(scores.get("academic", 0.5)),
                issues=issues,
                critical_fixes=data.get("critical_fixes", []),
                suggested_improvements=data.get("suggested_improvements", []),
                regeneration_guidance=data.get("regeneration_guidance", ""),
                feedback=data.get("feedback", "")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse enhanced verification JSON: {e}")
            
            # Try to determine acceptability from text content
            text_lower = response_text.lower()
            is_acceptable = "acceptable" in text_lower and "not acceptable" not in text_lower
            
            # Return a default result with some useful feedback
            return EnhancedVerificationResult(
                status=VerificationStatus.PARTIALLY_CORRECT if is_acceptable else VerificationStatus.INCORRECT,
                is_acceptable=is_acceptable,
                feedback=response_text[:500],
                confidence=0.4,
                regeneration_guidance="Could not parse verification response. Review diagram manually."
            )
    
    def build_targeted_feedback(
        self,
        verification: EnhancedVerificationResult,
        plan: Optional["DiagramPlan"] = None
    ) -> str:
        """
        Build targeted feedback for regeneration from enhanced verification.
        
        ENHANCED: More specific and actionable feedback for better diagram quality.
        
        Args:
            verification: EnhancedVerificationResult with detailed issues
            plan: Optional DiagramPlan to reference
            
        Returns:
            Structured feedback string for regeneration prompt
        """
        parts = []
        
        # Overall score context
        composite = verification.get_composite_score()
        parts.append(f"=== DIAGRAM QUALITY SCORE: {composite:.0%} ===")
        parts.append("")
        
        # Critical fixes first (most important)
        if verification.critical_fixes:
            parts.append("⛔ CRITICAL ISSUES (MUST FIX IMMEDIATELY) ⛔")
            for i, fix in enumerate(verification.critical_fixes, 1):
                parts.append(f"  {i}. {fix}")
            parts.append("")
        
        # Detailed issues by category (prioritized)
        priority_categories = [
            VerificationCategory.CONCEPTUAL,
            VerificationCategory.ALIGNMENT,
            VerificationCategory.LABELING,
            VerificationCategory.VISUAL,
            VerificationCategory.ACADEMIC
        ]
        
        for category in priority_categories:
            cat_issues = [i for i in verification.issues if i.category == category]
            if cat_issues:
                parts.append(f"=== {category.value.upper()} ISSUES ===")
                for issue in cat_issues:
                    severity_marker = "🔴" if issue.severity == "critical" else "🟡" if issue.severity == "major" else "🟢"
                    parts.append(f"{severity_marker} WHAT: {issue.what}")
                    parts.append(f"   WHY: {issue.why}")
                    parts.append(f"   FIX: {issue.fix}")
                    parts.append("")
        
        # Score breakdown for context
        parts.append("=== SCORE BREAKDOWN ===")
        parts.append(f"  Conceptual: {verification.conceptual_score:.0%}")
        parts.append(f"  Alignment:  {verification.alignment_score:.0%}")
        parts.append(f"  Labeling:   {verification.labeling_score:.0%}")
        parts.append(f"  Visual:     {verification.visual_score:.0%}")
        parts.append(f"  Academic:   {verification.academic_score:.0%}")
        parts.append("")
        
        # Regeneration guidance
        if verification.regeneration_guidance:
            parts.append(f"OVERALL GUIDANCE: {verification.regeneration_guidance}")
            parts.append("")
        
        # Reference the plan if available
        if plan and plan.extracted_values:
            parts.append("=== VALUES FROM QUESTION (MUST INCLUDE) ===")
            for key, value in plan.extracted_values.items():
                parts.append(f"  ✓ {key}: {value}")
            parts.append("")
        
        # Add specific rendering suggestions based on low scores
        rendering_suggestions = []
        if verification.visual_score < 0.6:
            rendering_suggestions.append("- Increase spacing between elements")
            rendering_suggestions.append("- Use larger font sizes (minimum 14pt)")
            rendering_suggestions.append("- Ensure labels don't overlap shapes")
            rendering_suggestions.append("- Use cleaner, simpler shapes")
        if verification.labeling_score < 0.6:
            rendering_suggestions.append("- Add units to all numerical values")
            rendering_suggestions.append("- Use standard symbols (F, θ, m, etc.)")
            rendering_suggestions.append("- Position labels clearly outside shapes")
        if verification.conceptual_score < 0.7:
            rendering_suggestions.append("- Verify all force/vector directions")
            rendering_suggestions.append("- Check numerical values match the question")
            rendering_suggestions.append("- Ensure correct physical/chemical representation")
        
        if rendering_suggestions:
            parts.append("=== RENDERING IMPROVEMENTS ===")
            parts.extend(rendering_suggestions)
            parts.append("")
        
        # Final instruction
        parts.append("=== ACTION REQUIRED ===")
        parts.append("Redraw the diagram addressing ALL issues above.")
        parts.append("Focus on CORRECTNESS first, then CLARITY.")
        
        return "\n".join(parts)
    
    def build_regeneration_prompt(
        self,
        verification_result: VerificationResult,
        original_question: str,
        subject: str
    ) -> str:
        """
        Build a prompt for regenerating an incorrect diagram.
        
        Args:
            verification_result: The verification result with feedback
            original_question: The original question text
            subject: Subject area
            
        Returns:
            A prompt that includes the feedback for regeneration
        """
        issues_text = ""
        if verification_result.issues:
            issues_text = "ISSUES FOUND:\n" + "\n".join(f"- {issue}" for issue in verification_result.issues)
        
        suggestions_text = ""
        if verification_result.suggestions:
            suggestions_text = "SUGGESTIONS:\n" + "\n".join(f"- {s}" for s in verification_result.suggestions)
        
        return f"""The previous diagram was incorrect. Please regenerate it with the following corrections.

ORIGINAL QUESTION:
{original_question}

SUBJECT: {subject.upper()}

FEEDBACK FROM VERIFICATION:
{verification_result.feedback or "The diagram did not accurately represent the question."}

{issues_text}

{suggestions_text}

INSTRUCTIONS:
1. Carefully read the feedback and understand what was wrong
2. Create a new diagram that addresses ALL the issues mentioned
3. Ensure all numerical values from the question are accurately shown
4. Label all important elements clearly
5. Make the diagram educational and clear

Start by calling create_canvas, then draw the corrected diagram step by step.
"""


def get_diagram_verifier(openai_service) -> DiagramVerifier:
    """Factory function to create a DiagramVerifier instance."""
    return DiagramVerifier(openai_service)
