"""
Spec-Based Diagram Service

This module provides a refactored diagram generation service that uses
the new structured DiagramSpec system.

Key improvements over the original verified_diagram_service:
1. LLMs fill JSON specs, NOT write drawing code
2. Strict routing ensures correct library selection
3. Feedback updates the spec, not regenerates prompts
4. Persisted state across generation attempts

This can be used as a drop-in replacement or alongside the existing service.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .specs.diagram_spec import (
    DiagramSpec,
    DiagramSubject,
    normalize_diagram_type,
    get_valid_types_for_subject,
    create_spec_from_plan,
    PARAMETER_MODELS,
    get_smiles_for_molecule,
)
from .spec_router import get_spec_router, RenderingDecision, RendererType
from .feedback_handler import get_feedback_handler, FeedbackState

# Import existing components
from .diagram_planner import DiagramPlanner, DiagramPlan
from .diagram_verifier import DiagramVerifier, EnhancedVerificationResult, VerificationStatus
from .tool_based_generator import ToolBasedDiagramGenerator

# Import renderers
try:
    from .engine import get_diagram_engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

logger = logging.getLogger(__name__)


@dataclass
class SpecBasedResult:
    """Result from spec-based diagram generation."""
    success: bool
    image_bytes: Optional[bytes] = None
    spec: Optional[DiagramSpec] = None
    verification: Optional[EnhancedVerificationResult] = None
    attempts: int = 0
    renderer_used: str = ""
    corrections_applied: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "has_image": self.image_bytes is not None,
            "attempts": self.attempts,
            "renderer_used": self.renderer_used,
            "corrections_applied": self.corrections_applied,
            "error": self.error,
        }

        if self.spec:
            result["spec"] = {
                "subject": self.spec.subject.value,
                "diagram_type": self.spec.diagram_type,
                "parameters": self.spec.parameters,
                "iteration": self.spec.iteration,
            }

        if self.verification:
            result["verification"] = {
                "status": self.verification.status.value if hasattr(self.verification.status, 'value') else str(self.verification.status),
                "is_acceptable": self.verification.is_acceptable,
                "score": self.verification.get_composite_score(),
                "scores": {
                    "conceptual": self.verification.conceptual_score,
                    "labeling": self.verification.labeling_score,
                    "visual": self.verification.visual_score,
                    "alignment": self.verification.alignment_score,
                    "academic": self.verification.academic_score,
                },
            }

        return result


# LLM prompt for spec generation (NOT drawing code)
SPEC_GENERATION_PROMPT = """You are creating a DIAGRAM SPECIFICATION (not drawing code) for a JEE/NEET exam diagram.

QUESTION:
{question_text}

SUBJECT: {subject}
{diagram_description_section}

Your task is to create a JSON specification that describes WHAT to draw, not HOW to draw it.

=== RULES ===
1. DO NOT write any drawing code
2. DO NOT solve the question - only extract GIVEN values
3. Keep it MINIMAL - like real exam diagrams
4. Extract all numerical values with units

=== AVAILABLE DIAGRAM TYPES FOR {subject} ===
{available_types}

=== PARAMETER REQUIREMENTS ===
{parameter_hints}

=== OUTPUT FORMAT (JSON ONLY) ===
{{
    "subject": "{subject}",
    "diagram_type": "select_from_available_types",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "labels": [
        {{"text": "Label text", "position": "above"}}
    ],
    "title": "Diagram Title"
}}

{specialized_hints}

RESPOND WITH ONLY VALID JSON - no markdown, no explanation.
"""


class SpecBasedDiagramService:
    """
    Diagram service using structured specifications.

    This service:
    1. Generates specs from questions using LLM
    2. Routes specs to correct renderers
    3. Applies feedback as spec corrections
    4. Maintains state across iterations
    """

    # Quality thresholds
    MIN_ATTEMPTS = 3
    MAX_ATTEMPTS = 7
    EARLY_ACCEPT_THRESHOLD = 0.75
    MINIMUM_QUALITY = 0.5

    def __init__(
        self,
        openai_service=None,
        kimi_service=None,
        use_kimi: bool = True,
        max_attempts: int = 5,
    ):
        """
        Initialize the service.

        Args:
            openai_service: OpenAI service for LLM calls
            kimi_service: Kimi service for LLM calls (preferred)
            use_kimi: Whether to use Kimi (default: True)
            max_attempts: Default max attempts per diagram
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None

        # Initialize components
        self._router = get_spec_router()
        self._feedback_handler = get_feedback_handler()
        self._verifier = DiagramVerifier(
            openai_service=openai_service,
            kimi_service=kimi_service,
            use_kimi=self._use_kimi
        )

        # For fallback to tool-based generation
        self._tool_generator = ToolBasedDiagramGenerator(
            openai_service=openai_service,
            kimi_service=kimi_service,
            use_kimi=self._use_kimi
        )

        self._max_attempts = max_attempts

        # Get diagram engine for specialized rendering
        self._engine = None
        if HAS_ENGINE:
            try:
                self._engine = get_diagram_engine()
                logger.info("DiagramEngine initialized for specialized rendering")
            except Exception as e:
                logger.warning(f"Failed to init DiagramEngine: {e}")

        if self._use_kimi:
            logger.info("SpecBasedDiagramService using Kimi K2.5")
        else:
            logger.info("SpecBasedDiagramService using OpenAI")

    async def generate_diagram(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None,
        provided_spec: Optional[Dict[str, Any]] = None,
    ) -> SpecBasedResult:
        """
        Generate a diagram using the spec-based approach.

        Args:
            question_text: The question requiring a diagram
            subject: Subject area (physics, chemistry, maths, biology)
            diagram_description: Optional description of what to show
            provided_spec: Optional pre-built spec dict

        Returns:
            SpecBasedResult with image, spec, and verification details
        """
        # Step 1: Create or validate the spec
        if provided_spec:
            spec = DiagramSpec(
                subject=DiagramSubject(subject.lower()),
                question_text=question_text,
                **provided_spec
            )
            logger.info(f"Using provided spec: {spec.diagram_type}")
        else:
            spec = await self._generate_spec_from_question(
                question_text, subject, diagram_description
            )
            if spec is None:
                return SpecBasedResult(
                    success=False,
                    error="Failed to generate diagram specification"
                )
            logger.info(f"Generated spec: {spec.diagram_type}")

        # Step 2: Create feedback state for tracking
        feedback_state = self._feedback_handler.create_state(spec)

        # Step 3: Route to renderer and generate
        best_result = None
        best_score = 0.0
        attempt = 0

        while attempt < self._max_attempts:
            attempt += 1
            current_spec = feedback_state.get_updated_spec()

            logger.info(f"Attempt {attempt}/{self._max_attempts}: {current_spec.diagram_type}")

            # Route the spec
            decision = self._router.route(current_spec)

            # Generate image using appropriate renderer
            image_bytes, metadata = await self._render_from_decision(
                decision, question_text, diagram_description
            )

            if image_bytes is None:
                logger.warning(f"Attempt {attempt} failed to render: {metadata.get('error')}")
                continue

            logger.info(f"Rendered with {decision.renderer_type.value}")

            # Verify the diagram
            verification = await self._verifier.verify_diagram_enhanced(
                image_bytes=image_bytes,
                question_text=question_text,
                subject=subject,
                diagram_description=diagram_description,
            )

            score = verification.get_composite_score()
            logger.info(f"Verification score: {score:.2f}, acceptable: {verification.is_acceptable}")

            # Track best result
            if score > best_score:
                best_score = score
                best_result = {
                    "image_bytes": image_bytes,
                    "spec": current_spec,
                    "verification": verification,
                    "renderer": decision.renderer_type.value,
                }

            # Accept if good enough
            if verification.is_acceptable:
                return SpecBasedResult(
                    success=True,
                    image_bytes=image_bytes,
                    spec=current_spec,
                    verification=verification,
                    attempts=attempt,
                    renderer_used=decision.renderer_type.value,
                    corrections_applied=len(feedback_state.corrections_applied),
                )

            # Early accept if we've done enough attempts and quality is good
            if attempt >= self.MIN_ATTEMPTS and best_score >= self.EARLY_ACCEPT_THRESHOLD:
                logger.info(f"Early accept with score {best_score:.2f}")
                return SpecBasedResult(
                    success=True,
                    image_bytes=best_result["image_bytes"],
                    spec=best_result["spec"],
                    verification=best_result["verification"],
                    attempts=attempt,
                    renderer_used=best_result["renderer"],
                    corrections_applied=len(feedback_state.corrections_applied),
                )

            # Apply feedback for next iteration
            if attempt < self._max_attempts:
                feedback_text = self._verifier.build_targeted_feedback(verification)
                updated_spec = self._feedback_handler.process_verification_feedback(
                    feedback_state,
                    feedback_text,
                    score,
                    verification
                )
                logger.info(f"Applied {len(feedback_state.corrections_applied)} total corrections")

        # Max attempts reached - return best effort
        if best_result and best_score >= self.MINIMUM_QUALITY:
            return SpecBasedResult(
                success=False,  # Not fully verified
                image_bytes=best_result["image_bytes"],
                spec=best_result["spec"],
                verification=best_result["verification"],
                attempts=attempt,
                renderer_used=best_result["renderer"],
                corrections_applied=len(feedback_state.corrections_applied),
                error=f"Best effort after {attempt} attempts (score: {best_score:.2f})"
            )

        return SpecBasedResult(
            success=False,
            error=f"Failed to generate acceptable diagram after {attempt} attempts"
        )

    async def _generate_spec_from_question(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str]
    ) -> Optional[DiagramSpec]:
        """
        Use LLM to generate a DiagramSpec from a question.

        The LLM fills in the spec structure, NOT drawing code.
        """
        try:
            subject_enum = DiagramSubject(subject.lower())
        except ValueError:
            logger.warning(f"Invalid subject '{subject}', defaulting to physics")
            subject_enum = DiagramSubject.PHYSICS

        # Get available types for this subject
        available_types = get_valid_types_for_subject(subject_enum)

        # Get parameter hints for common types
        parameter_hints = self._get_parameter_hints(subject_enum)

        # Get specialized hints for libraries
        specialized_hints = self._get_specialized_hints(subject_enum)

        # Build prompt
        desc_section = ""
        if diagram_description:
            desc_section = f"\nDIAGRAM REQUIREMENTS: {diagram_description}"

        prompt = SPEC_GENERATION_PROMPT.format(
            question_text=question_text,
            subject=subject,
            diagram_description_section=desc_section,
            available_types="\n".join(f"- {t}" for t in available_types),
            parameter_hints=parameter_hints,
            specialized_hints=specialized_hints,
        )

        # Call LLM
        try:
            if self._use_kimi and self._kimi_service:
                response = await self._kimi_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=1500,
                )
            else:
                response = await self._openai_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                    model="gpt-4o",
                )

            if not response.get("success"):
                logger.error(f"LLM spec generation failed: {response.get('error')}")
                return None

            # Parse response
            return self._parse_spec_response(response.get("response", ""), subject_enum, question_text)

        except Exception as e:
            logger.error(f"Spec generation error: {e}")
            return None

    def _parse_spec_response(
        self,
        response_text: str,
        subject: DiagramSubject,
        question_text: str
    ) -> Optional[DiagramSpec]:
        """Parse LLM response into DiagramSpec."""
        import json
        import re

        try:
            # Clean response
            text = response_text.strip()

            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)

            # Find JSON object
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                logger.error("No JSON found in response")
                return None

            data = json.loads(match.group())

            # Build spec
            spec = DiagramSpec(
                subject=subject,
                diagram_type=normalize_diagram_type(data.get("diagram_type", "diagram")),
                question_text=question_text,
                parameters=data.get("parameters", {}),
                labels=data.get("labels", []),
                title=data.get("title"),
                show_labels=data.get("show_labels", True),
            )

            logger.info(f"Parsed spec: type={spec.diagram_type}, params={list(spec.parameters.keys())}")
            return spec

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"Spec parse error: {e}")
            return None

    async def _render_from_decision(
        self,
        decision: RenderingDecision,
        question_text: str,
        diagram_description: Optional[str]
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Render a diagram based on the routing decision.

        Returns:
            Tuple of (image_bytes, metadata) or (None, error_dict)
        """
        render_params = decision.render_params

        # Try specialized renderer first
        if decision.renderer_type in [RendererType.SCHEMDRAW, RendererType.RDKIT]:
            if self._engine:
                try:
                    result = await self._render_with_engine(decision)
                    if result[0] is not None:
                        return result
                except Exception as e:
                    logger.warning(f"Specialized render failed: {e}")

        # Fall back to tool-based generator
        logger.info("Using tool-based generator as fallback")
        return await self._tool_generator.generate_diagram(
            question_text=question_text,
            subject=decision.subject.value,
            diagram_description=diagram_description,
        )

    async def _render_with_engine(
        self,
        decision: RenderingDecision
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Render using the DiagramEngine with specialized renderers."""
        from .specs.base_spec import DiagramSubject as BaseSubject

        try:
            # Get renderer for subject
            subject_enum = BaseSubject(decision.subject.value)
            renderer = self._engine.get_renderer(subject_enum)

            if renderer is None:
                return None, {"error": f"No renderer for {decision.subject.value}"}

            if not renderer.supports_type(decision.diagram_type):
                return None, {"error": f"Type {decision.diagram_type} not supported"}

            # Render
            result = await renderer.render(decision.render_params)

            if result and result.image_data:
                return result.image_data, {
                    "renderer": decision.renderer_type.value,
                    "width": result.width,
                    "height": result.height,
                }

            return None, {"error": "Renderer returned no image"}

        except Exception as e:
            logger.error(f"Engine render error: {e}")
            return None, {"error": str(e)}

    def _get_parameter_hints(self, subject: DiagramSubject) -> str:
        """Get parameter hints for a subject."""
        hints = {
            DiagramSubject.PHYSICS: """
For CIRCUITS: Provide "components" list with type, name, value. Provide "voltage".
For INCLINED_PLANE: Provide "angle", "mass", "object_type" (box/sphere/cylinder).
For OPTICS: Provide "focal_length", "object_distance", "lens_type" or "mirror_type".
For PROJECTILE: Provide "initial_velocity", "angle".
""",
            DiagramSubject.CHEMISTRY: """
For MOLECULES: Provide "smiles" string (REQUIRED). Example: "CCO" for ethanol, "ClC(Cl)(Cl)Cl" for CCl4.
For REACTIONS: Provide "reactants" and "products" lists.
For LAB SETUPS: Provide equipment labels and connection info.
""",
            DiagramSubject.MATHS: """
For GRAPHS: Provide "functions" list with expressions.
For GEOMETRY: Provide "shapes" list with type, position, dimensions.
For NUMBER_LINE: Provide "min_value", "max_value", "points" to mark.
""",
            DiagramSubject.BIOLOGY: """
For CELLS: Just specify diagram_type, renderer has templates.
For ORGANS: Specify any labels or highlights needed.
""",
        }
        return hints.get(subject, "")

    def _get_specialized_hints(self, subject: DiagramSubject) -> str:
        """Get hints about specialized libraries."""
        hints = {
            DiagramSubject.PHYSICS: """
IMPORTANT: Circuit diagrams use SchemDraw for professional quality.
Ensure "components" is a list of {"type": "resistor/capacitor/inductor", "name": "R1", "value": "10 Ω"}
""",
            DiagramSubject.CHEMISTRY: """
IMPORTANT: Molecules use RDKit for accurate structures. The "smiles" field is REQUIRED.
Common SMILES:
- Methane: C
- Ethanol: CCO
- Carbon tetrachloride (CCl4): ClC(Cl)(Cl)Cl
- Benzene: c1ccccc1
- Water: O
""",
        }
        return hints.get(subject, "")


# Factory function
def get_spec_based_service(
    openai_service=None,
    kimi_service=None,
    use_kimi: bool = True,
    max_attempts: int = 5,
) -> SpecBasedDiagramService:
    """Create a SpecBasedDiagramService instance."""
    return SpecBasedDiagramService(
        openai_service=openai_service,
        kimi_service=kimi_service,
        use_kimi=use_kimi,
        max_attempts=max_attempts,
    )
