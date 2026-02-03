"""
Subject-Specific Validation Rules for Diagram Generation

Contains JEE/NEET/CBSE academic standards for physics, maths, chemistry, and biology diagrams.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationCategory(str, Enum):
    """Categories of validation issues"""
    CONCEPTUAL = "conceptual"      # Physics/math/chem principles
    LABELING = "labeling"          # Text, units, symbols
    VISUAL = "visual"              # Clarity, clutter, proportions
    ALIGNMENT = "alignment"        # Match with question text
    ACADEMIC = "academic"          # JEE/NEET standards


class IssueSeverity(str, Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"   # Must fix - diagram is wrong/misleading
    MAJOR = "major"         # Should fix - impacts understanding
    MINOR = "minor"         # Nice to fix - improves quality


@dataclass
class ValidationIssue:
    """Structured validation issue with remediation"""
    category: ValidationCategory
    severity: IssueSeverity
    what: str       # What is wrong
    why: str        # Why it matters for students
    fix: str        # How to fix it
    rule_id: str    # Reference to validation rule


@dataclass
class ValidationResult:
    """Result of plan/diagram validation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0-1, overall validation score
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
    
    @property
    def has_critical_issues(self) -> bool:
        return len(self.critical_issues) > 0


class SubjectValidationRules(ABC):
    """Base class for subject-specific validation rules"""
    
    @abstractmethod
    def get_rules(self) -> Dict[str, Dict[str, str]]:
        """Get all rules for this subject"""
        pass
    
    @abstractmethod
    def validate_plan(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate a diagram plan against subject rules"""
        pass
    
    def _create_issue(
        self,
        category: ValidationCategory,
        severity: IssueSeverity,
        what: str,
        why: str,
        fix: str,
        rule_id: str
    ) -> ValidationIssue:
        return ValidationIssue(
            category=category,
            severity=severity,
            what=what,
            why=why,
            fix=fix,
            rule_id=rule_id
        )


class PhysicsValidationRules(SubjectValidationRules):
    """Physics-specific validation for JEE/NEET standards"""
    
    # Force diagram rules
    FORCE_DIAGRAM_RULES = {
        "arrows_from_center": {
            "description": "Force arrows must originate from object center or point of application",
            "why": "Shows where force acts on the body - fundamental to free body diagrams",
            "fix": "Redraw arrows starting from center of the object"
        },
        "proportional_lengths": {
            "description": "Arrow lengths must be proportional to force magnitude",
            "why": "Students need to visually compare force magnitudes",
            "fix": "Scale arrow lengths: larger force = longer arrow"
        },
        "weight_downward": {
            "description": "Weight/gravity (mg) must point straight downward",
            "why": "Gravity always acts toward Earth's center",
            "fix": "Draw weight arrow pointing directly down from center"
        },
        "normal_perpendicular": {
            "description": "Normal force must be perpendicular to contact surface",
            "why": "Normal force is always perpendicular by definition",
            "fix": "Draw normal at 90 degrees to the surface"
        },
        "friction_opposes_motion": {
            "description": "Friction must oppose direction of motion/impending motion",
            "why": "Friction always resists relative motion between surfaces",
            "fix": "Draw friction arrow opposite to velocity/applied force direction"
        },
        "action_reaction_separate": {
            "description": "Action-reaction pairs must be on different bodies",
            "why": "Newton's 3rd law - pairs act on different objects",
            "fix": "Show action on one body, reaction on the other body"
        },
        "force_colors": {
            "description": "Use standard colors: weight=red, normal=blue, friction=orange, applied=green",
            "why": "Consistent color coding helps identify force types quickly",
            "fix": "Apply standard color scheme for force arrows"
        }
    }
    
    # Circuit diagram rules
    CIRCUIT_RULES = {
        "proper_symbols": {
            "description": "Use standard IEC circuit symbols (zigzag for resistor, etc.)",
            "why": "Standard symbols are universally recognized in physics",
            "fix": "Replace rectangles with proper circuit symbols"
        },
        "complete_loop": {
            "description": "Circuit must form complete closed loop(s)",
            "why": "Current can only flow in complete circuits",
            "fix": "Ensure all components connect to form closed path"
        },
        "labeled_values": {
            "description": "All component values must be labeled (R=10Ω, V=12V)",
            "why": "Values are essential for circuit analysis problems",
            "fix": "Add value labels near each component"
        },
        "current_direction": {
            "description": "Show conventional current direction (+ to -) if relevant",
            "why": "Current direction affects calculations and understanding",
            "fix": "Add arrow showing current flow direction"
        },
        "series_single_path": {
            "description": "Series circuits: single continuous path through all components",
            "why": "Defines series configuration - same current through all",
            "fix": "Arrange components in single loop"
        },
        "parallel_branches": {
            "description": "Parallel circuits: clear branch points with multiple paths",
            "why": "Parallel means same voltage, split current",
            "fix": "Show clear junction points where current divides"
        }
    }
    
    # Optics/ray diagram rules
    OPTICS_RULES = {
        "ray_direction": {
            "description": "Light rays travel left to right (standard convention)",
            "why": "Universal convention in optics diagrams",
            "fix": "Place object on left, image forms on right side"
        },
        "object_arrow": {
            "description": "Object shown as upward arrow on principal axis",
            "why": "Standard representation to show object orientation",
            "fix": "Draw object as vertical arrow pointing up"
        },
        "focal_points_marked": {
            "description": "Focal points clearly marked with 'F' or 'F1', 'F2'",
            "why": "Focal points are essential for ray construction",
            "fix": "Mark focal points on both sides of lens/mirror"
        },
        "principal_axis": {
            "description": "Principal axis shown as horizontal line through center",
            "why": "Reference line for all ray constructions",
            "fix": "Draw dashed horizontal line through optical center"
        },
        "three_standard_rays": {
            "description": "Use standard rays: parallel, through center, through focus",
            "why": "These rays have predictable paths for image location",
            "fix": "Draw at least 2 of the 3 standard rays"
        },
        "real_solid_virtual_dashed": {
            "description": "Real rays/images: solid lines. Virtual: dashed lines",
            "why": "Distinguishes between actual and apparent light paths",
            "fix": "Use solid for real, dashed for virtual rays/images"
        }
    }
    
    # Inclined plane rules
    INCLINED_PLANE_RULES = {
        "angle_at_base": {
            "description": "Inclination angle shown at base of plane",
            "why": "Standard convention for inclined plane problems",
            "fix": "Mark angle between plane and horizontal at bottom"
        },
        "components_parallel_perpendicular": {
            "description": "Weight components parallel and perpendicular to surface",
            "why": "Essential decomposition for inclined plane analysis",
            "fix": "Show mg·sinθ along plane, mg·cosθ perpendicular"
        },
        "coordinate_axes": {
            "description": "Show coordinate axes aligned with plane if used",
            "why": "Clarifies sign conventions for calculations",
            "fix": "Add x-axis along plane, y-axis perpendicular"
        }
    }
    
    # Wave diagram rules
    WAVE_RULES = {
        "wavelength_marked": {
            "description": "Wavelength (λ) clearly marked between two crests/troughs",
            "why": "Wavelength is fundamental wave property",
            "fix": "Add double-headed arrow showing one complete wavelength"
        },
        "amplitude_marked": {
            "description": "Amplitude (A) shown from equilibrium to crest/trough",
            "why": "Amplitude represents maximum displacement",
            "fix": "Mark vertical distance from center line to peak"
        },
        "equilibrium_line": {
            "description": "Show horizontal equilibrium/mean position line",
            "why": "Reference for measuring displacement",
            "fix": "Draw dashed horizontal line at center of wave"
        }
    }
    
    def get_rules(self) -> Dict[str, Dict[str, str]]:
        return {
            "force_diagram": self.FORCE_DIAGRAM_RULES,
            "free_body_diagram": self.FORCE_DIAGRAM_RULES,
            "circuit": self.CIRCUIT_RULES,
            "series_circuit": self.CIRCUIT_RULES,
            "parallel_circuit": self.CIRCUIT_RULES,
            "mixed_circuit": self.CIRCUIT_RULES,
            "ray_diagram": self.OPTICS_RULES,
            "ray_diagram_lens": self.OPTICS_RULES,
            "ray_diagram_mirror": self.OPTICS_RULES,
            "inclined_plane": self.INCLINED_PLANE_RULES,
            "wave_diagram": self.WAVE_RULES,
        }
    
    def validate_plan(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate physics diagram plan"""
        issues = []
        warnings = []
        
        diagram_type = plan.get("diagram_type", "").lower()
        objects = plan.get("objects", [])
        labels = plan.get("labels", [])
        extracted_values = plan.get("extracted_values", {})
        
        # Check for force diagrams
        if "force" in diagram_type or "free_body" in diagram_type:
            issues.extend(self._validate_force_diagram(plan))
        
        # Check for circuits
        if "circuit" in diagram_type:
            issues.extend(self._validate_circuit(plan))
        
        # Check for ray diagrams
        if "ray" in diagram_type or "lens" in diagram_type or "mirror" in diagram_type:
            issues.extend(self._validate_ray_diagram(plan))
        
        # General physics checks
        if not extracted_values:
            warnings.append("No numerical values extracted from question")
        
        # Check labels have units where needed
        for label in labels:
            text = label.get("text", "") if isinstance(label, dict) else str(label)
            if any(char.isdigit() for char in text):
                if not any(unit in text.lower() for unit in ["n", "m", "kg", "s", "v", "a", "ω", "°"]):
                    warnings.append(f"Label '{text}' may need units")
        
        is_valid = not any(i.severity == IssueSeverity.CRITICAL for i in issues)
        score = max(0, 1.0 - len(issues) * 0.1 - len([i for i in issues if i.severity == IssueSeverity.CRITICAL]) * 0.3)
        
        return ValidationResult(is_valid=is_valid, issues=issues, warnings=warnings, score=score)
    
    def _validate_force_diagram(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate force diagram specific rules"""
        issues = []
        objects = plan.get("objects", [])
        
        # Check for weight force pointing down
        has_weight = False
        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get("name", "").lower()
                obj_type = obj.get("type", "").lower()
                if "weight" in name or "gravity" in name or "mg" in name:
                    has_weight = True
                    # Check direction
                    props = obj.get("properties", {})
                    direction = props.get("direction", "")
                    if direction and "down" not in direction.lower():
                        issues.append(self._create_issue(
                            ValidationCategory.CONCEPTUAL,
                            IssueSeverity.CRITICAL,
                            "Weight force not pointing downward",
                            "Gravity always acts toward Earth's center (downward)",
                            "Change weight arrow direction to point straight down",
                            "weight_downward"
                        ))
        
        return issues
    
    def _validate_circuit(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate circuit diagram specific rules"""
        issues = []
        labels = plan.get("labels", [])
        
        # Check for component value labels
        has_value_labels = False
        for label in labels:
            text = label.get("text", "") if isinstance(label, dict) else str(label)
            if any(unit in text for unit in ["Ω", "V", "A", "F", "H", "ohm"]):
                has_value_labels = True
                break
        
        if not has_value_labels:
            issues.append(self._create_issue(
                ValidationCategory.LABELING,
                IssueSeverity.MAJOR,
                "Circuit components missing value labels",
                "Students need component values for calculations",
                "Add labels like 'R = 10Ω', 'V = 12V' near components",
                "labeled_values"
            ))
        
        return issues
    
    def _validate_ray_diagram(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate ray diagram specific rules"""
        issues = []
        labels = plan.get("labels", [])
        
        # Check for focal point labels
        has_focal_label = False
        for label in labels:
            text = label.get("text", "") if isinstance(label, dict) else str(label)
            if text.upper() in ["F", "F1", "F2", "FOCUS"]:
                has_focal_label = True
                break
        
        if not has_focal_label:
            issues.append(self._create_issue(
                ValidationCategory.LABELING,
                IssueSeverity.MAJOR,
                "Focal point(s) not labeled",
                "Focal points are essential reference for ray construction",
                "Add 'F' label at focal point position(s)",
                "focal_points_marked"
            ))
        
        return issues


class MathsValidationRules(SubjectValidationRules):
    """Maths-specific validation rules for JEE/NEET standards"""
    
    # Geometry rules
    GEOMETRY_RULES = {
        "vertex_labels": {
            "description": "Vertices labeled with capital letters (A, B, C)",
            "why": "Standard mathematical notation for referencing points",
            "fix": "Use uppercase letters A, B, C... for vertices"
        },
        "side_labels": {
            "description": "Sides labeled with lowercase letters or measurements",
            "why": "Distinguishes sides from vertices in problem solving",
            "fix": "Use lowercase a, b, c or actual measurements like '5 cm'"
        },
        "right_angle_square": {
            "description": "Right angles marked with small square symbol",
            "why": "Universal convention for indicating 90-degree angles",
            "fix": "Add small square at right angle corners"
        },
        "equal_sides_ticks": {
            "description": "Equal sides marked with matching tick marks",
            "why": "Shows congruent sides without cluttering with numbers",
            "fix": "Add single/double/triple ticks on equal sides"
        },
        "angle_arcs": {
            "description": "Angles shown with arc and degree or variable label",
            "why": "Clearly indicates which angle is being referenced",
            "fix": "Draw arc at angle and label with degrees or θ, α, etc."
        },
        "perpendicular_symbol": {
            "description": "Perpendicular lines marked with ⊥ symbol or small square",
            "why": "Standard notation for perpendicularity",
            "fix": "Add ⊥ symbol or small square at intersection"
        },
        "parallel_arrows": {
            "description": "Parallel lines marked with matching arrows (→→)",
            "why": "Standard notation for parallel lines",
            "fix": "Add matching arrows on parallel line segments"
        }
    }
    
    # Number line rules
    NUMBER_LINE_RULES = {
        "arrow_direction": {
            "description": "Arrow at positive (right) end of number line",
            "why": "Shows number line extends to infinity",
            "fix": "Add arrow pointing right at end of line"
        },
        "even_spacing": {
            "description": "Scale points must be evenly spaced",
            "why": "Consistent scale ensures accurate representation",
            "fix": "Space all tick marks at equal intervals"
        },
        "endpoint_markers": {
            "description": "Filled circle ● for included (≤, ≥), empty ○ for excluded (<, >)",
            "why": "Critical distinction for inequality solutions",
            "fix": "Use ● for closed/included, ○ for open/excluded endpoints"
        },
        "solution_highlight": {
            "description": "Solution region clearly highlighted or bolded",
            "why": "Students need to quickly identify the answer region",
            "fix": "Use thick line or shading for solution set"
        },
        "critical_point_labeled": {
            "description": "Critical/boundary point value labeled",
            "why": "Shows exact value where condition changes",
            "fix": "Add number label below the critical point"
        }
    }
    
    # Graph/coordinate rules
    GRAPH_RULES = {
        "axes_labeled": {
            "description": "Both axes labeled with variable names (x, y) or quantities",
            "why": "Identifies what each axis represents",
            "fix": "Add labels like 'x' and 'y' or 'time (s)' at axis ends"
        },
        "axes_arrows": {
            "description": "Arrows at positive ends of both axes",
            "why": "Shows axes extend to infinity",
            "fix": "Add arrows pointing right (x) and up (y)"
        },
        "scale_marked": {
            "description": "Scale clearly marked on axes with numbers",
            "why": "Allows reading values from graph",
            "fix": "Add tick marks with numbers at regular intervals"
        },
        "origin_labeled": {
            "description": "Origin marked as O or (0,0)",
            "why": "Reference point for coordinate system",
            "fix": "Label intersection point as 'O' or '0'"
        },
        "key_points_labeled": {
            "description": "Key points (intercepts, vertices, intersections) labeled",
            "why": "These points are often the answer or key to solution",
            "fix": "Add coordinate labels like (2, 0) at important points"
        },
        "smooth_curves": {
            "description": "Functions drawn as smooth curves, not jagged/angular",
            "why": "Mathematical functions are continuous",
            "fix": "Use smooth curve for function graphs"
        }
    }
    
    # Venn diagram rules
    VENN_RULES = {
        "sets_labeled": {
            "description": "Each set clearly labeled (A, B, C or descriptive names)",
            "why": "Identifies what each circle represents",
            "fix": "Add set labels inside or near each circle"
        },
        "regions_identifiable": {
            "description": "All regions (intersections, complements) visually distinct",
            "why": "Each region represents a different set operation",
            "fix": "Use shading or colors to distinguish regions"
        },
        "universal_set_shown": {
            "description": "Rectangle showing universal set if needed",
            "why": "Context for complement operations",
            "fix": "Draw rectangle around all circles for universal set"
        }
    }
    
    def get_rules(self) -> Dict[str, Dict[str, str]]:
        return {
            "geometry": self.GEOMETRY_RULES,
            "geometry_construction": self.GEOMETRY_RULES,
            "triangle": self.GEOMETRY_RULES,
            "number_line": self.NUMBER_LINE_RULES,
            "coordinate_graph": self.GRAPH_RULES,
            "graph": self.GRAPH_RULES,
            "venn_diagram": self.VENN_RULES,
        }
    
    def validate_plan(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate maths diagram plan"""
        issues = []
        warnings = []
        
        diagram_type = plan.get("diagram_type", "").lower()
        labels = plan.get("labels", [])
        
        # Check for number lines
        if "number" in diagram_type and "line" in diagram_type:
            issues.extend(self._validate_number_line(plan))
        
        # Check for geometry
        if any(g in diagram_type for g in ["geometry", "triangle", "polygon", "circle"]):
            issues.extend(self._validate_geometry(plan))
        
        # Check for graphs
        if "graph" in diagram_type or "coordinate" in diagram_type:
            issues.extend(self._validate_graph(plan))
        
        is_valid = not any(i.severity == IssueSeverity.CRITICAL for i in issues)
        score = max(0, 1.0 - len(issues) * 0.1)
        
        return ValidationResult(is_valid=is_valid, issues=issues, warnings=warnings, score=score)
    
    def _validate_number_line(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate number line specific rules"""
        issues = []
        extracted = plan.get("extracted_values", {})
        
        # Check for critical point in extracted values
        if not extracted:
            issues.append(self._create_issue(
                ValidationCategory.ALIGNMENT,
                IssueSeverity.MAJOR,
                "No critical point value extracted from question",
                "Number line must show the boundary value from inequality",
                "Extract the numerical value (e.g., 3 from 'x > 3')",
                "critical_point_labeled"
            ))
        
        return issues
    
    def _validate_geometry(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate geometry diagram specific rules"""
        issues = []
        labels = plan.get("labels", [])
        
        # Check for vertex labels (uppercase)
        has_vertex_labels = False
        for label in labels:
            text = label.get("text", "") if isinstance(label, dict) else str(label)
            if len(text) == 1 and text.isupper():
                has_vertex_labels = True
                break
        
        if not has_vertex_labels:
            issues.append(self._create_issue(
                ValidationCategory.LABELING,
                IssueSeverity.MAJOR,
                "Vertices not labeled with capital letters",
                "Standard math notation uses A, B, C for vertices",
                "Add uppercase letter labels (A, B, C) at each vertex",
                "vertex_labels"
            ))
        
        return issues
    
    def _validate_graph(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate graph/coordinate diagram"""
        issues = []
        labels = plan.get("labels", [])
        
        # Check for axis labels
        has_x_label = False
        has_y_label = False
        for label in labels:
            text = label.get("text", "").lower() if isinstance(label, dict) else str(label).lower()
            if text in ["x", "x-axis"]:
                has_x_label = True
            if text in ["y", "y-axis"]:
                has_y_label = True
        
        if not has_x_label or not has_y_label:
            issues.append(self._create_issue(
                ValidationCategory.LABELING,
                IssueSeverity.MAJOR,
                "Axes not properly labeled",
                "Graph axes must show what variables they represent",
                "Add 'x' label at right end, 'y' label at top of axes",
                "axes_labeled"
            ))
        
        return issues


class ChemistryValidationRules(SubjectValidationRules):
    """Chemistry-specific validation rules"""
    
    MOLECULE_RULES = {
        "bond_lines": {
            "description": "Bonds shown as lines (single=-, double==, triple≡)",
            "why": "Standard chemical bond notation",
            "fix": "Use appropriate line notation for bond order"
        },
        "atom_symbols": {
            "description": "Use standard element symbols (C, H, O, not words)",
            "why": "Universal chemical notation",
            "fix": "Use periodic table symbols"
        },
        "lone_pairs": {
            "description": "Show lone pairs as dots when relevant",
            "why": "Important for understanding reactivity and structure",
            "fix": "Add dot pairs for non-bonding electrons"
        }
    }
    
    REACTION_RULES = {
        "arrow_direction": {
            "description": "Arrow points from reactants to products (→)",
            "why": "Shows direction of chemical change",
            "fix": "Arrow should point left to right"
        },
        "equilibrium_arrows": {
            "description": "Reversible reactions use double arrows (⇌)",
            "why": "Indicates equilibrium, not complete reaction",
            "fix": "Use ⇌ for equilibrium reactions"
        },
        "state_symbols": {
            "description": "State symbols (s), (l), (g), (aq) shown",
            "why": "Physical states are part of complete equation",
            "fix": "Add state symbols after each species"
        },
        "balanced": {
            "description": "Equation must be balanced",
            "why": "Conservation of mass is fundamental",
            "fix": "Verify atom counts equal on both sides"
        }
    }
    
    LAB_SETUP_RULES = {
        "apparatus_labeled": {
            "description": "All apparatus clearly labeled",
            "why": "Students need to identify equipment",
            "fix": "Add labels to beakers, flasks, burettes, etc."
        },
        "proper_symbols": {
            "description": "Use standard lab apparatus symbols",
            "why": "Consistent representation across diagrams",
            "fix": "Use standard scientific apparatus symbols"
        }
    }
    
    def get_rules(self) -> Dict[str, Dict[str, str]]:
        return {
            "molecule": self.MOLECULE_RULES,
            "molecule_2d": self.MOLECULE_RULES,
            "molecule_3d": self.MOLECULE_RULES,
            "reaction": self.REACTION_RULES,
            "reaction_scheme": self.REACTION_RULES,
            "lab_setup": self.LAB_SETUP_RULES,
        }
    
    def validate_plan(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate chemistry diagram plan"""
        issues = []
        warnings = []
        
        diagram_type = plan.get("diagram_type", "").lower()
        
        # Basic validation for chemistry
        if "molecule" in diagram_type:
            # Check for atom symbols
            pass
        
        if "reaction" in diagram_type:
            # Check for proper arrow usage
            pass
        
        is_valid = not any(i.severity == IssueSeverity.CRITICAL for i in issues)
        score = max(0, 1.0 - len(issues) * 0.1)
        
        return ValidationResult(is_valid=is_valid, issues=issues, warnings=warnings, score=score)


class BiologyValidationRules(SubjectValidationRules):
    """Biology-specific validation rules"""
    
    CELL_RULES = {
        "organelles_labeled": {
            "description": "All visible organelles must be labeled",
            "why": "Students need to identify cell structures",
            "fix": "Add labels with leader lines to each organelle"
        },
        "membrane_shown": {
            "description": "Cell membrane clearly visible",
            "why": "Boundary of cell is fundamental",
            "fix": "Draw clear outer membrane"
        },
        "nucleus_distinct": {
            "description": "Nucleus clearly distinguishable",
            "why": "Nucleus is key organelle in eukaryotes",
            "fix": "Show nucleus with distinct boundary"
        }
    }
    
    ANATOMY_RULES = {
        "parts_labeled": {
            "description": "All relevant parts labeled with leader lines",
            "why": "Anatomical diagrams require complete labeling",
            "fix": "Add labels to all significant structures"
        },
        "proportions": {
            "description": "Relative sizes approximately correct",
            "why": "Misleading proportions confuse understanding",
            "fix": "Adjust sizes to match actual proportions"
        },
        "orientation": {
            "description": "Standard anatomical orientation",
            "why": "Consistent orientation aids learning",
            "fix": "Use standard anatomical position"
        }
    }
    
    def get_rules(self) -> Dict[str, Dict[str, str]]:
        return {
            "cell": self.CELL_RULES,
            "plant_cell": self.CELL_RULES,
            "animal_cell": self.CELL_RULES,
            "anatomy": self.ANATOMY_RULES,
            "human_heart": self.ANATOMY_RULES,
            "human_brain": self.ANATOMY_RULES,
        }
    
    def validate_plan(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate biology diagram plan"""
        issues = []
        warnings = []
        
        diagram_type = plan.get("diagram_type", "").lower()
        labels = plan.get("labels", [])
        
        # Check for sufficient labels
        if len(labels) < 3:
            warnings.append("Biology diagrams typically need multiple labels")
        
        is_valid = not any(i.severity == IssueSeverity.CRITICAL for i in issues)
        score = max(0, 1.0 - len(issues) * 0.1)
        
        return ValidationResult(is_valid=is_valid, issues=issues, warnings=warnings, score=score)


def get_validation_rules(subject: str) -> SubjectValidationRules:
    """Factory function to get subject-specific validation rules"""
    subject_lower = subject.lower()
    
    if subject_lower in ["physics", "phy"]:
        return PhysicsValidationRules()
    elif subject_lower in ["maths", "math", "mathematics"]:
        return MathsValidationRules()
    elif subject_lower in ["chemistry", "chem"]:
        return ChemistryValidationRules()
    elif subject_lower in ["biology", "bio"]:
        return BiologyValidationRules()
    else:
        # Default to physics rules as fallback
        return PhysicsValidationRules()
