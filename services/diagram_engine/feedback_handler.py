"""
Diagram Feedback Handler

Handles the feedback loop between diagram verification and regeneration.

Key responsibilities:
1. Parse verification feedback into actionable corrections
2. Update diagram specs with corrections (NOT create new drawing code)
3. Persist feedback state across regeneration attempts
4. Extract specific parameter fixes from LLM feedback

This ensures LLM2's critique UPDATES the diagram_spec and triggers regeneration
with corrected parameters, rather than being ignored.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .specs.diagram_spec import (
    DiagramSpec,
    DiagramSubject,
    normalize_diagram_type,
    PARAMETER_MODELS,
)

logger = logging.getLogger(__name__)


class CorrectionCategory(str, Enum):
    """Categories of corrections that can be applied."""
    VALUE = "value"           # Numerical value corrections (angle, mass, etc.)
    COMPONENT = "component"   # Component additions/removals (circuit parts, etc.)
    LABEL = "label"           # Label text/position corrections
    SMILES = "smiles"         # Molecule SMILES corrections
    LAYOUT = "layout"         # Layout/spacing corrections
    CONCEPTUAL = "conceptual" # Conceptual fixes (wrong diagram type, etc.)


@dataclass
class Correction:
    """A single correction to apply to a spec."""
    category: CorrectionCategory
    parameter: str
    old_value: Any
    new_value: Any
    source: str  # Which part of feedback triggered this
    confidence: float = 0.8  # How confident we are in this correction


@dataclass
class FeedbackState:
    """
    Persisted state for feedback across regeneration attempts.

    This ensures feedback history is maintained and corrections compound.
    """
    spec: DiagramSpec
    corrections_applied: List[Correction] = field(default_factory=list)
    feedback_texts: List[str] = field(default_factory=list)
    iteration: int = 0
    scores: List[float] = field(default_factory=list)

    def add_feedback(self, feedback: str, score: float = 0.0):
        """Add new feedback to history."""
        self.feedback_texts.append(feedback)
        self.scores.append(score)
        self.iteration += 1

    def apply_correction(self, correction: Correction) -> bool:
        """
        Apply a correction to the spec.

        Returns True if correction was applied successfully.
        """
        try:
            # Update the spec parameters
            if correction.parameter in self.spec.parameters:
                old = self.spec.parameters[correction.parameter]
                if old != correction.new_value:
                    self.spec.parameters[correction.parameter] = correction.new_value
                    correction.old_value = old
                    self.corrections_applied.append(correction)
                    logger.info(
                        f"Applied correction: {correction.parameter} = "
                        f"{correction.new_value} (was {old})"
                    )
                    return True
            else:
                # Add new parameter
                self.spec.parameters[correction.parameter] = correction.new_value
                self.corrections_applied.append(correction)
                logger.info(f"Added parameter: {correction.parameter} = {correction.new_value}")
                return True

        except Exception as e:
            logger.error(f"Failed to apply correction: {e}")
            return False

    def get_updated_spec(self) -> DiagramSpec:
        """Get the spec with all corrections applied."""
        # Update iteration count
        self.spec.iteration = self.iteration
        self.spec.feedback_history = self.feedback_texts

        return self.spec


class FeedbackHandler:
    """
    Handles parsing and applying feedback to diagram specs.

    The handler:
    1. Extracts corrections from verification feedback
    2. Maintains state across regeneration attempts
    3. Ensures feedback is never dropped
    4. Prioritizes critical corrections
    """

    # Patterns for extracting values from feedback
    VALUE_PATTERNS = {
        "angle": [
            r'angle\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)\s*°?',
            r'(\d+(?:\.\d+)?)\s*°?\s*(?:degrees?|angle)',
            r'θ\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "mass": [
            r'mass\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)\s*kg',
            r'(\d+(?:\.\d+)?)\s*kg',
            r'm\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "voltage": [
            r'voltage\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)\s*V',
            r'(\d+(?:\.\d+)?)\s*V(?:olts?)?',
            r'emf\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "resistance": [
            r'resistance\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)\s*Ω',
            r'(\d+(?:\.\d+)?)\s*(?:Ω|ohms?)',
            r'R\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "focal_length": [
            r'focal\s*length\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'f\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "initial_velocity": [
            r'initial\s*velocity\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'v[0₀]\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*m/s',
        ],
        "amplitude": [
            r'amplitude\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'A\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "wavelength": [
            r'wavelength\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'λ\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "height": [
            r'height\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'h\s*=\s*(\d+(?:\.\d+)?)',
        ],
        "h_diff": [
            r'height\s*difference\s*(?:should be|is|=)\s*(\d+(?:\.\d+)?)',
            r'Δh\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:mm|cm)\s*(?:height|difference)',
        ],
    }

    # Patterns for SMILES corrections
    SMILES_PATTERNS = [
        r'SMILES\s*(?:should be|is|:)\s*([A-Za-z0-9\(\)\[\]=@#+\\/-]+)',
        r'molecule\s+(?:as\s+)?([A-Z][A-Za-z0-9\(\)\[\]=@#+\\/-]+)',
        r'structure\s+(?:is\s+)?([A-Z][A-Za-z0-9\(\)\[\]=@#+\\/-]+)',
    ]

    # Common molecule name to SMILES
    MOLECULE_CORRECTIONS = {
        "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
        "ccl4": "ClC(Cl)(Cl)Cl",
        "methane": "C",
        "ethane": "CC",
        "ethanol": "CCO",
        "water": "O",
        "ammonia": "N",
        "benzene": "c1ccccc1",
        "chloroform": "ClC(Cl)Cl",
    }

    def __init__(self):
        """Initialize feedback handler."""
        self._active_states: Dict[str, FeedbackState] = {}

    def create_state(self, spec: DiagramSpec) -> FeedbackState:
        """
        Create a new feedback state for a spec.

        Call this when starting a new generation attempt.
        """
        state = FeedbackState(spec=spec)

        # Store by a unique key (could use question hash)
        state_key = f"{spec.subject.value}_{spec.diagram_type}_{id(spec)}"
        self._active_states[state_key] = state

        return state

    def parse_feedback(
        self,
        feedback: str,
        state: FeedbackState,
        enhanced_result: Optional[Any] = None
    ) -> List[Correction]:
        """
        Parse feedback text into actionable corrections.

        Args:
            feedback: Text feedback from verification
            state: Current feedback state
            enhanced_result: Optional EnhancedVerificationResult with structured issues

        Returns:
            List of Correction objects to apply
        """
        corrections = []
        feedback_lower = feedback.lower()

        # 1. Extract value corrections from patterns
        for param, patterns in self.VALUE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, feedback, re.IGNORECASE)
                if match:
                    try:
                        new_value = float(match.group(1))
                        old_value = state.spec.parameters.get(param)

                        if old_value != new_value:
                            corrections.append(Correction(
                                category=CorrectionCategory.VALUE,
                                parameter=param,
                                old_value=old_value,
                                new_value=new_value,
                                source=f"pattern: {pattern}",
                                confidence=0.9
                            ))
                            break  # Found a match for this param
                    except (ValueError, IndexError):
                        pass

        # 2. Extract SMILES corrections for molecules
        if state.spec.diagram_type in ["molecule_2d", "molecule_3d"]:
            corrections.extend(self._extract_smiles_corrections(feedback, state))

        # 3. Parse structured issues if available
        if enhanced_result and hasattr(enhanced_result, 'issues'):
            corrections.extend(self._parse_structured_issues(enhanced_result.issues, state))

        # 4. Label corrections
        if "label" in feedback_lower or "text" in feedback_lower:
            corrections.extend(self._extract_label_corrections(feedback, state))

        # 5. Layout corrections
        if any(word in feedback_lower for word in ["overlap", "spacing", "crowded", "small"]):
            corrections.extend(self._extract_layout_corrections(feedback, state))

        # Log all extracted corrections
        if corrections:
            logger.info(f"Extracted {len(corrections)} corrections from feedback")
            for c in corrections:
                logger.debug(f"  - {c.parameter}: {c.old_value} -> {c.new_value}")

        return corrections

    def _extract_smiles_corrections(
        self,
        feedback: str,
        state: FeedbackState
    ) -> List[Correction]:
        """Extract SMILES corrections for molecule diagrams."""
        corrections = []
        feedback_lower = feedback.lower()

        # Check for molecule name mentions
        for name, smiles in self.MOLECULE_CORRECTIONS.items():
            if name in feedback_lower:
                current_smiles = state.spec.parameters.get("smiles", "")
                if current_smiles != smiles:
                    corrections.append(Correction(
                        category=CorrectionCategory.SMILES,
                        parameter="smiles",
                        old_value=current_smiles,
                        new_value=smiles,
                        source=f"molecule name: {name}",
                        confidence=0.95
                    ))
                    break

        # Check for explicit SMILES in feedback
        for pattern in self.SMILES_PATTERNS:
            match = re.search(pattern, feedback, re.IGNORECASE)
            if match:
                new_smiles = match.group(1)
                current = state.spec.parameters.get("smiles", "")
                if current != new_smiles:
                    corrections.append(Correction(
                        category=CorrectionCategory.SMILES,
                        parameter="smiles",
                        old_value=current,
                        new_value=new_smiles,
                        source=f"SMILES pattern",
                        confidence=0.85
                    ))
                break

        return corrections

    def _parse_structured_issues(
        self,
        issues: List[Any],
        state: FeedbackState
    ) -> List[Correction]:
        """Parse structured issues from EnhancedVerificationResult."""
        corrections = []

        for issue in issues:
            if not hasattr(issue, 'fix') or not issue.fix:
                continue

            fix = issue.fix
            category = getattr(issue, 'category', None)

            # Extract corrections from fix text
            if category and hasattr(category, 'value'):
                cat_val = category.value

                if cat_val == "conceptual":
                    # Look for value corrections in fix
                    for param, patterns in self.VALUE_PATTERNS.items():
                        for pattern in patterns:
                            match = re.search(pattern, fix, re.IGNORECASE)
                            if match:
                                try:
                                    new_value = float(match.group(1))
                                    old_value = state.spec.parameters.get(param)
                                    if old_value != new_value:
                                        corrections.append(Correction(
                                            category=CorrectionCategory.VALUE,
                                            parameter=param,
                                            old_value=old_value,
                                            new_value=new_value,
                                            source=f"issue fix: {fix[:50]}",
                                            confidence=0.9
                                        ))
                                except (ValueError, IndexError):
                                    pass

                elif cat_val == "labeling":
                    # Label-related corrections
                    if "unit" in fix.lower():
                        corrections.append(Correction(
                            category=CorrectionCategory.LABEL,
                            parameter="include_units",
                            old_value=False,
                            new_value=True,
                            source="labeling fix",
                            confidence=0.8
                        ))

                    if "font" in fix.lower() or "size" in fix.lower():
                        match = re.search(r'(\d+)\s*(?:pt|point|px)?', fix)
                        if match:
                            corrections.append(Correction(
                                category=CorrectionCategory.LABEL,
                                parameter="font_size",
                                old_value=state.spec.parameters.get("font_size", 12),
                                new_value=max(int(match.group(1)), 14),
                                source="font size fix",
                                confidence=0.85
                            ))

        return corrections

    def _extract_label_corrections(
        self,
        feedback: str,
        state: FeedbackState
    ) -> List[Correction]:
        """Extract label-related corrections."""
        corrections = []

        if "font" in feedback.lower() and "small" in feedback.lower():
            corrections.append(Correction(
                category=CorrectionCategory.LABEL,
                parameter="font_size",
                old_value=state.spec.parameters.get("font_size", 12),
                new_value=16,
                source="font size feedback",
                confidence=0.7
            ))

        return corrections

    def _extract_layout_corrections(
        self,
        feedback: str,
        state: FeedbackState
    ) -> List[Correction]:
        """Extract layout-related corrections."""
        corrections = []
        feedback_lower = feedback.lower()

        if "overlap" in feedback_lower or "crowded" in feedback_lower:
            # Increase spacing
            current_spacing = state.spec.parameters.get("element_spacing", 1.0)
            corrections.append(Correction(
                category=CorrectionCategory.LAYOUT,
                parameter="element_spacing",
                old_value=current_spacing,
                new_value=current_spacing * 1.5,
                source="spacing feedback",
                confidence=0.7
            ))

        return corrections

    def apply_corrections(
        self,
        state: FeedbackState,
        corrections: List[Correction]
    ) -> FeedbackState:
        """
        Apply a list of corrections to the state.

        Corrections are applied in priority order:
        1. Conceptual (most important)
        2. Value
        3. SMILES
        4. Label
        5. Layout
        
        Note: This increments the iteration counter when corrections are applied.
        """
        # Sort by priority
        priority_order = [
            CorrectionCategory.CONCEPTUAL,
            CorrectionCategory.VALUE,
            CorrectionCategory.SMILES,
            CorrectionCategory.LABEL,
            CorrectionCategory.LAYOUT,
        ]

        sorted_corrections = sorted(
            corrections,
            key=lambda c: (
                priority_order.index(c.category) if c.category in priority_order else 99,
                -c.confidence
            )
        )

        # Apply each correction
        applied_count = 0
        for correction in sorted_corrections:
            if state.apply_correction(correction):
                applied_count += 1

        # Increment iteration if any corrections were applied
        if applied_count > 0:
            state.iteration += 1
            logger.info(f"Applied {applied_count} corrections, now at iteration {state.iteration}")

        return state

    def process_verification_feedback(
        self,
        state: FeedbackState,
        feedback: str,
        score: float,
        enhanced_result: Optional[Any] = None
    ) -> DiagramSpec:
        """
        Process verification feedback and return updated spec.

        This is the main entry point for the feedback loop.

        Args:
            state: Current feedback state
            feedback: Text feedback from verification
            score: Verification score (0-1)
            enhanced_result: Optional structured verification result

        Returns:
            Updated DiagramSpec with corrections applied
        """
        # Add feedback to history
        state.add_feedback(feedback, score)

        # Parse feedback into corrections
        corrections = self.parse_feedback(feedback, state, enhanced_result)

        # Apply corrections
        if corrections:
            state = self.apply_corrections(state, corrections)
            logger.info(f"Applied {len(corrections)} corrections to spec")
        else:
            logger.warning("No corrections extracted from feedback - consider manual review")

        return state.get_updated_spec()

    def get_correction_summary(self, state: FeedbackState) -> str:
        """Get a summary of all corrections applied."""
        if not state.corrections_applied:
            return "No corrections applied."

        lines = [f"Corrections applied (iteration {state.iteration}):"]
        for c in state.corrections_applied:
            lines.append(f"  - {c.parameter}: {c.old_value} -> {c.new_value}")

        return "\n".join(lines)


# Singleton instance
_handler_instance: Optional[FeedbackHandler] = None


def get_feedback_handler() -> FeedbackHandler:
    """Get or create the feedback handler singleton."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = FeedbackHandler()
    return _handler_instance
