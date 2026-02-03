"""
Diagram Planner - Creates and validates structured diagram plans before generation.

This module implements the planning phase of the enhanced diagram generation system.
A plan is created before any drawing to ensure:
1. All values from the question are extracted
2. Objects, labels, and relationships are defined
3. Academic standards are considered
4. The plan is validated before generation begins
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .validation_rules import (
    get_validation_rules,
    ValidationResult,
    ValidationIssue,
    ValidationCategory,
    IssueSeverity
)
from .academic_standards import AcademicStandards

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR DIAGRAM PLAN
# ============================================================================

class DiagramObject(BaseModel):
    """An object to be drawn in the diagram"""
    name: str = Field(..., description="Unique name for this object")
    type: str = Field(..., description="Shape type: rectangle, circle, arrow, line, polygon, arc")
    position: List[float] = Field(default=[0, 0], description="[x, y] position")
    dimensions: Dict[str, float] = Field(default_factory=dict, description="Size params: width, height, radius, etc.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Style: color, line_width, fill, etc.")
    
    class Config:
        extra = "allow"


class DiagramLabel(BaseModel):
    """A text label in the diagram"""
    text: str = Field(..., description="Label text content")
    target_object: Optional[str] = Field(None, description="Object this label refers to")
    position: str = Field("above", description="Position relative to target: above, below, left, right, inside")
    absolute_position: Optional[List[float]] = Field(None, description="[x, y] if not relative to object")
    font_size: int = Field(14, description="Font size (minimum 12 for readability)")
    includes_units: bool = Field(False, description="Whether label includes units")
    
    class Config:
        extra = "allow"


class DiagramAnnotation(BaseModel):
    """An annotation like angle marker, dimension line, etc."""
    type: str = Field(..., description="Annotation type: angle, dimension, marker, right_angle")
    value: Optional[str] = Field(None, description="Value to display: '30°', '5 cm', etc.")
    position: Dict[str, float] = Field(default_factory=dict, description="Position parameters")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    
    class Config:
        extra = "allow"


class DiagramRelationship(BaseModel):
    """Relationship between objects"""
    from_object: str = Field(..., description="Source object name")
    to_object: str = Field(..., description="Target object name")
    relationship_type: str = Field(..., description="Type: connected, parallel, perpendicular, equal")
    
    class Config:
        extra = "allow"


class CoordinateRange(BaseModel):
    """Coordinate system range for the diagram"""
    x_min: float = Field(0, description="Minimum x value")
    x_max: float = Field(10, description="Maximum x value")
    y_min: float = Field(0, description="Minimum y value")
    y_max: float = Field(8, description="Maximum y value")


class DiagramPlan(BaseModel):
    """
    Complete structured plan for a diagram.
    
    This plan is created before generation and validated to ensure
    all necessary information is captured correctly.
    """
    # Core identification
    diagram_type: str = Field(..., description="Type of diagram: force_diagram, number_line, circuit, etc.")
    subject: str = Field(..., description="Subject: physics, maths, chemistry, biology")
    
    # Extracted information from question
    extracted_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Numerical values extracted from question: {'force': '50 N', 'angle': '30°'}"
    )
    
    # Diagram elements
    objects: List[DiagramObject] = Field(default_factory=list, description="Objects to draw")
    labels: List[DiagramLabel] = Field(default_factory=list, description="Text labels")
    annotations: List[DiagramAnnotation] = Field(default_factory=list, description="Annotations (angles, dimensions)")
    relationships: List[DiagramRelationship] = Field(default_factory=list, description="Object relationships")
    
    # Layout
    coordinate_range: CoordinateRange = Field(default_factory=CoordinateRange, description="Coordinate system")
    layout_strategy: str = Field("centered", description="Layout: centered, left_to_right, radial")
    
    # Academic compliance
    physics_principles: List[str] = Field(default_factory=list, description="Physics principles involved")
    math_concepts: List[str] = Field(default_factory=list, description="Math concepts involved")
    academic_notes: str = Field("", description="Notes on JEE/NEET compliance")
    
    # Metadata
    complexity_level: str = Field("moderate", description="Complexity: simple, moderate, complex")
    layout_notes: str = Field("", description="Explanation of layout choices")
    
    class Config:
        extra = "allow"
    
    def get_object_count(self) -> int:
        return len(self.objects)
    
    def get_label_count(self) -> int:
        return len(self.labels)
    
    def is_simple(self) -> bool:
        return self.get_object_count() <= 3 and self.get_label_count() <= 5
    
    def is_complex(self) -> bool:
        return self.get_object_count() > 6 or self.get_label_count() > 10


# ============================================================================
# PLANNING PROMPTS
# ============================================================================

PLANNING_PROMPT = """You are creating a PLAN for a JEE/NEET/CBSE exam-style diagram. Do NOT draw - just plan.

⚠️ CRITICAL: This diagram is for a QUESTION PAPER. It must be MINIMAL like real exam diagrams.
⚠️ DO NOT SOLVE THE QUESTION - only show what is GIVEN.

QUESTION:
{question_text}

SUBJECT: {subject}
{diagram_description_section}

=== PLANNING REQUIREMENTS ===

1. EXTRACT ONLY GIVEN VALUES (DO NOT CALCULATE):
   ✓ Values explicitly stated: "mass = 5 kg", "force = 50 N", "angle = 30°"
   ✓ Given relationships: "A is twice B"
   ✓ For MOLECULES: Provide "smiles" string if possible (e.g. "C(Cl)(Cl)(Cl)Cl" for CCl4)
   ❌ DO NOT calculate derived values (like F=ma results)
   ❌ DO NOT solve for unknowns
   ❌ DO NOT show answers

2. IDENTIFY DIAGRAM ELEMENTS (MINIMAL):
   - Only ESSENTIAL objects to show the setup
   - Only labels for GIVEN values
   - Minimal annotations (only if explicitly needed)
   - Think: "What would a JEE paper show?"

3. KEEP IT SIMPLE:
   - Exam diagrams are SIMPLE and CLEAN
   - NO extra decorations or details
   - NO formulas or calculations shown
   - White background, black lines, minimal colors

4. ACADEMIC STANDARDS (JEE/NEET/CBSE):
   - Follow standard conventions (force arrows from center, etc.)
   - Match what students see in actual exam papers
   - Complexity should be MINIMAL - this is to SHOW the setup, not SOLVE it

5. DETERMINE COMPLEXITY:
   - simple: 1-3 objects, 1-5 labels (PREFERRED for most diagrams)
   - moderate: 4-6 objects, 6-10 labels (only if question demands it)
   - complex: 7+ objects (RARE - only for detailed anatomy etc.)

6. SELECT CORRECT DIAGRAM TYPE:
   Select from these supported types based on subject (or use generic 'diagram' if none fit):
   - PHYSICS: series_circuit, parallel_circuit, mixed_circuit, ray_diagram_lens, ray_diagram_mirror, free_body_diagram, projectile_motion, wave_diagram, inclined_plane, electric_field, magnetic_field
   - CHEMISTRY: molecule_2d, reaction_scheme, orbital_diagram, periodic_table_section, lab_setup_titration, lab_setup_distillation
   - MATHS: coordinate_graph, geometry_construction, number_line, venn_diagram, bar_chart, pie_chart
   - BIOLOGY: plant_cell, animal_cell, human_heart, human_brain, dna_replication, mitosis_stages, digestive_system, respiratory_system

=== RESPONSE FORMAT (JSON) ===
{{
    "diagram_type": "type_name",
    "subject": "{subject}",
    
    "extracted_values": {{
        "value_name": "value with unit",
        "example": "50 N"
    }},
    
    "objects": [
        {{
            "name": "unique_object_name",
            "type": "rectangle|circle|arrow|line|polygon|arc",
            "position": [x, y],
            "dimensions": {{"width": 2, "height": 1}},
            "properties": {{"color": "#FF0000", "fill": true}}
        }}
    ],
    
    "labels": [
        {{
            "text": "F = 50 N",
            "target_object": "force_arrow",
            "position": "above",
            "font_size": 14,
            "includes_units": true
        }}
    ],
    
    "annotations": [
        {{
            "type": "angle",
            "value": "30°",
            "position": {{"x": 5, "y": 3, "radius": 1}}
        }}
    ],
    
    "relationships": [
        {{
            "from_object": "obj1",
            "to_object": "obj2",
            "relationship_type": "perpendicular"
        }}
    ],
    
    "coordinate_range": {{
        "x_min": 0, "x_max": 10,
        "y_min": 0, "y_max": 8
    }},
    
    "layout_strategy": "centered",
    "complexity_level": "simple|moderate|complex",
    
    "physics_principles": ["Newton's second law", "..."],
    "math_concepts": ["Inequalities", "..."],
    
    "layout_notes": "Explanation of why you chose this layout",
    "academic_notes": "Notes on JEE/NEET compliance"
}}

⚠️ CRITICAL RULES:
1. Extract ONLY explicitly given values - DO NOT calculate anything
2. DO NOT solve the question or show answers
3. Keep diagram MINIMAL - like a real JEE/NEET exam paper
4. Labels should be fontsize >= 14 and clearly placed
5. SIMPLE is better - fewer elements, cleaner diagram
6. Think: "Would this appear in an actual exam paper?" If not, remove it.
"""

PLANNING_VALIDATION_PROMPT = """Review this diagram plan for correctness and completeness.

ORIGINAL QUESTION:
{question_text}

SUBJECT: {subject}

PROPOSED PLAN:
{plan_json}

Check for:
1. Are ALL values from the question captured in extracted_values?
2. Are all necessary objects planned?
3. Are labels clear and include units where needed?
4. Does the layout avoid overlapping elements?
5. Does it follow {subject} conventions?

Respond with:
{{
    "is_valid": true|false,
    "missing_values": ["list of values from question not in plan"],
    "missing_elements": ["list of required elements not planned"],
    "issues": ["list of specific issues"],
    "suggestions": ["list of improvements"]
}}
"""


PLANNING_UPDATE_PROMPT = """You are refining a diagram plan based on feedback.

ORIGINAL QUESTION:
{question_text}

CURRENT PLAN (JSON):
{plan_json}

FEEDBACK / ISSUES TO FIX:
{feedback}

TASK:
Update the JSON plan to address the feedback.
- Keep the structure valid
- Only change what needs to be fixed
- RETAIN all existing objects, labels, and values unless explicitly asked to remove them
- Ensure all extraction rules still apply (no calculations, only given values)
- For specialized diagrams (molecules, circuits), UPDATE the 'extracted_values' (e.g. 'smiles', 'formula') to fix the content
- Ensure the complexity remains appropriate

RESPONSE FORMAT:
Return ONLY the full valid JSON of the updated plan.
"""


# ============================================================================
# DIAGRAM PLANNER CLASS
# ============================================================================

class DiagramPlanner:
    """
    Creates and validates structured diagram plans before generation.
    
    The planner:
    1. Analyzes the question text
    2. Creates a structured plan (DiagramPlan)
    3. Validates the plan against subject rules
    4. Returns the validated plan for generation
    """
    
    def __init__(self, openai_service=None, kimi_service=None, use_kimi: bool = False):
        """
        Initialize the diagram planner.
        
        Args:
            openai_service: OpenAI service for LLM calls
            kimi_service: Kimi service for LLM calls (alternative)
            use_kimi: Whether to use Kimi instead of OpenAI
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None
        self._model = "gpt-4o" if not self._use_kimi else None
    
    async def create_plan(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None
    ) -> Tuple[Optional[DiagramPlan], List[str]]:
        """
        Create a structured diagram plan from question text.
        
        Args:
            question_text: The question requiring a diagram
            subject: Subject (physics, maths, chemistry, biology)
            diagram_description: Optional specific diagram requirements
            
        Returns:
            Tuple of (DiagramPlan or None, list of warnings)
        """
        logger.info(f"Creating diagram plan for {subject}")
        
        # Build the planning prompt
        description_section = ""
        if diagram_description:
            description_section = f"\nDIAGRAM REQUIREMENTS: {diagram_description}"
        
        prompt = PLANNING_PROMPT.format(
            question_text=question_text,
            subject=subject,
            diagram_description_section=description_section
        )
        
        # Add subject-specific style instructions
        style_instructions = AcademicStandards.get_style_instructions(subject, "general")
        prompt += f"\n\n{style_instructions}"
        
        try:
            # Call LLM to create plan
            if self._use_kimi and self._kimi_service:
                response = await self._kimi_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,  # Kimi K2.5 only supports temperature=0.6
                    max_tokens=2000,
                )
            else:
                response = await self._openai_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Low temperature for structured output
                    max_tokens=2000,
                    model=self._model,
                )
            
            if not response.get("success"):
                logger.error(f"Failed to create plan: {response.get('error')}")
                return None, [f"LLM error: {response.get('error')}"]
            
            # Parse the response
            plan_text = response.get("response", "")
            plan, parse_warnings = self._parse_plan_response(plan_text, subject)
            
            if plan is None:
                return None, parse_warnings
            
            # Validate the plan
            validation_result = self.validate_plan(plan)
            
            all_warnings = parse_warnings + validation_result.warnings
            
            if not validation_result.is_valid:
                logger.warning(f"Plan validation failed: {[str(i) for i in validation_result.issues]}")
                # Still return the plan but with warnings
                for issue in validation_result.issues:
                    all_warnings.append(f"{issue.category.value}: {issue.what}")
            
            logger.info(f"Plan created: {plan.diagram_type}, {len(plan.objects)} objects, {len(plan.labels)} labels")
            return plan, all_warnings
            
            return plan, all_warnings
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}", exc_info=True)
            return None, [f"Exception: {str(e)}"]

    async def update_plan(
        self,
        current_plan: DiagramPlan,
        feedback: str,
        question_text: str,
    ) -> Tuple[Optional[DiagramPlan], List[str]]:
        """
        Update a diagram plan based on feedback using LLM.
        
        Args:
            current_plan: The current plan to update
            feedback: Feedback describing text/issues
            question_text: Original question text for context
            
        Returns:
            Tuple of (Updated DiagramPlan or None, list of warnings)
        """
        logger.info("Updating diagram plan based on feedback")
        
        # Convert current plan to JSON for prompt
        plan_json = current_plan.model_dump_json(indent=2)
        
        prompt = PLANNING_UPDATE_PROMPT.format(
            question_text=question_text,
            plan_json=plan_json,
            feedback=feedback
        )
        
        try:
            # Call LLM to update plan
            if self._use_kimi and self._kimi_service:
                response = await self._kimi_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,  # Kimi K2.5 only supports temperature=0.6
                    max_tokens=2000,
                )
            else:
                response = await self._openai_service.chat_completion_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # Low temperature for structured output
                    max_tokens=2000,
                    model=self._model,
                )
            
            if not response.get("success"):
                logger.error(f"Failed to update plan: {response.get('error')}")
                return None, [f"LLM error: {response.get('error')}"]
            
            # Parse the response
            plan_text = response.get("response", "")
            plan, parse_warnings = self._parse_plan_response(plan_text, current_plan.subject)
            
            if plan is None:
                return None, parse_warnings
            
            # Validate the updated plan
            validation_result = self.validate_plan(plan)
            all_warnings = parse_warnings + validation_result.warnings
            
            if not validation_result.is_valid:
                logger.warning(f"Updated plan validation failed: {[str(i) for i in validation_result.issues]}")
                for issue in validation_result.issues:
                    all_warnings.append(f"{issue.category.value}: {issue.what}")
                
            logger.info("Plan updated successfully")
            return plan, all_warnings
            
        except Exception as e:
            logger.error(f"Error updating plan: {e}", exc_info=True)
            return None, [f"Exception: {str(e)}"]
    
    def _parse_plan_response(self, response_text: str, subject: str) -> Tuple[Optional[DiagramPlan], List[str]]:
        """Parse LLM response into DiagramPlan"""
        warnings = []
        
        try:
            # Extract JSON from response
            json_str = self._extract_json(response_text)
            
            if not json_str:
                warnings.append("Could not extract JSON from planning response")
                return None, warnings
            
            data = json.loads(json_str)
            
            # Parse objects
            objects = []
            for obj_data in data.get("objects", []):
                try:
                    objects.append(DiagramObject(**obj_data))
                except Exception as e:
                    warnings.append(f"Failed to parse object: {e}")
            
            # Parse labels
            labels = []
            for label_data in data.get("labels", []):
                try:
                    labels.append(DiagramLabel(**label_data))
                except Exception as e:
                    warnings.append(f"Failed to parse label: {e}")
            
            # Parse annotations
            annotations = []
            for ann_data in data.get("annotations", []):
                try:
                    annotations.append(DiagramAnnotation(**ann_data))
                except Exception as e:
                    warnings.append(f"Failed to parse annotation: {e}")
            
            # Parse relationships
            relationships = []
            for rel_data in data.get("relationships", []):
                try:
                    relationships.append(DiagramRelationship(**rel_data))
                except Exception as e:
                    warnings.append(f"Failed to parse relationship: {e}")
            
            # Parse coordinate range
            coord_data = data.get("coordinate_range", {})
            coordinate_range = CoordinateRange(
                x_min=coord_data.get("x_min", 0),
                x_max=coord_data.get("x_max", 10),
                y_min=coord_data.get("y_min", 0),
                y_max=coord_data.get("y_max", 8),
            )
            
            # Create the plan
            plan = DiagramPlan(
                diagram_type=data.get("diagram_type", "unknown"),
                subject=data.get("subject", subject),
                extracted_values=data.get("extracted_values", {}),
                objects=objects,
                labels=labels,
                annotations=annotations,
                relationships=relationships,
                coordinate_range=coordinate_range,
                layout_strategy=data.get("layout_strategy", "centered"),
                physics_principles=data.get("physics_principles", []),
                math_concepts=data.get("math_concepts", []),
                academic_notes=data.get("academic_notes", ""),
                complexity_level=data.get("complexity_level", "moderate"),
                layout_notes=data.get("layout_notes", ""),
            )
            
            return plan, warnings
            
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error: {e}")
            return None, warnings
        except Exception as e:
            warnings.append(f"Parse error: {e}")
            return None, warnings
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            return brace_match.group()
        
        return None
    
    def validate_plan(self, plan: DiagramPlan) -> ValidationResult:
        """
        Validate a diagram plan against subject-specific rules.
        
        Args:
            plan: The DiagramPlan to validate
            
        Returns:
            ValidationResult with issues and warnings
        """
        issues = []
        warnings = []
        
        # Get subject-specific validator
        validator = get_validation_rules(plan.subject)
        
        # Convert plan to dict for validation
        plan_dict = plan.model_dump()
        
        # Run subject-specific validation
        subject_result = validator.validate_plan(plan_dict)
        issues.extend(subject_result.issues)
        warnings.extend(subject_result.warnings)
        
        # General validation checks
        general_issues = self._validate_general_requirements(plan)
        issues.extend(general_issues)
        
        # Check academic standards compliance
        compliance_warnings = AcademicStandards.validate_compliance(
            plan.diagram_type, plan.subject, plan_dict
        )
        warnings.extend(compliance_warnings)
        
        # Calculate overall validity
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        score = max(0, 1.0 - len(issues) * 0.1 - len([i for i in issues if i.severity == IssueSeverity.CRITICAL]) * 0.3)
        
        return ValidationResult(
            is_valid=not has_critical,
            issues=issues,
            warnings=warnings,
            score=score
        )
    
    def _validate_general_requirements(self, plan: DiagramPlan) -> List[ValidationIssue]:
        """Validate general diagram requirements"""
        issues = []
        
        # Check for extracted values
        if not plan.extracted_values:
            issues.append(ValidationIssue(
                category=ValidationCategory.ALIGNMENT,
                severity=IssueSeverity.MAJOR,
                what="No numerical values extracted from question",
                why="Diagram may not match specific values in the question",
                fix="Extract all numbers, measurements, and values from the question",
                rule_id="extracted_values_required"
            ))
        
        # Check for objects
        # Specialized diagrams (molecules, reactions, charts) may not need explicit objects
        # if they rely on extracted_values (e.g. SMILES, reaction parameters)
        DATA_DRIVEN_TYPES = {
            "molecule_2d", "molecule_3d", "reaction_scheme", 
            "lab_setup_titration", "lab_setup_distillation", "lab_setup_electrolysis", "lab_setup_manometer",
            "periodic_table_section", "orbital_diagram", "crystal_structure",
            "venn_diagram", "number_line", "bar_chart", "pie_chart",
            "coordinate_graph"
        }
        
        # Check if it's a data-driven type
        is_data_driven = plan.diagram_type in DATA_DRIVEN_TYPES
        
        if not plan.objects and not is_data_driven:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONCEPTUAL,
                severity=IssueSeverity.CRITICAL,
                what="No objects planned for diagram",
                why="Cannot create a diagram without any objects to draw",
                fix="Add objects (shapes, arrows, lines) to the plan",
                rule_id="objects_required"
            ))
        
        # Check label font sizes
        for label in plan.labels:
            if label.font_size < 12:
                issues.append(ValidationIssue(
                    category=ValidationCategory.VISUAL,
                    severity=IssueSeverity.MAJOR,
                    what=f"Label '{label.text}' has small font size ({label.font_size})",
                    why="Small text is hard to read in exam diagrams",
                    fix="Use font size >= 14 for all labels",
                    rule_id="minimum_font_size"
                ))
        
        # Check for labels on all objects that should have them
        object_names = {obj.name for obj in plan.objects}
        labeled_objects = {label.target_object for label in plan.labels if label.target_object}
        
        # Most objects should have labels
        unlabeled = object_names - labeled_objects
        if len(unlabeled) > len(object_names) / 2:
            issues.append(ValidationIssue(
                category=ValidationCategory.LABELING,
                severity=IssueSeverity.MAJOR,
                what=f"Many objects lack labels: {unlabeled}",
                why="Unlabeled elements confuse students",
                fix="Add labels to important objects",
                rule_id="label_coverage"
            ))
        
        return issues
    
    def get_plan_summary(self, plan: DiagramPlan) -> str:
        """Get a human-readable summary of the plan for prompts"""
        lines = [
            f"=== DIAGRAM PLAN: {plan.diagram_type.upper()} ===",
            "",
            "EXTRACTED VALUES:",
        ]
        
        for key, value in plan.extracted_values.items():
            lines.append(f"  - {key}: {value}")
        
        lines.extend(["", "OBJECTS TO DRAW:"])
        for obj in plan.objects:
            lines.append(f"  - {obj.name} ({obj.type}) at {obj.position}")
        
        lines.extend(["", "LABELS TO ADD:"])
        for label in plan.labels:
            pos = f"on {label.target_object}" if label.target_object else f"at {label.absolute_position}"
            lines.append(f"  - '{label.text}' {label.position} {pos}")
        
        if plan.annotations:
            lines.extend(["", "ANNOTATIONS:"])
            for ann in plan.annotations:
                lines.append(f"  - {ann.type}: {ann.value}")
        
        lines.extend([
            "",
            f"COORDINATE RANGE: x=[{plan.coordinate_range.x_min}, {plan.coordinate_range.x_max}], "
            f"y=[{plan.coordinate_range.y_min}, {plan.coordinate_range.y_max}]",
            f"COMPLEXITY: {plan.complexity_level}",
        ])
        
        if plan.academic_notes:
            lines.extend(["", f"ACADEMIC NOTES: {plan.academic_notes}"])
        
        return "\n".join(lines)


# Singleton instance
_planner_instance: Optional[DiagramPlanner] = None


def get_diagram_planner() -> DiagramPlanner:
    """Get or create the diagram planner singleton"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = DiagramPlanner()
    return _planner_instance
