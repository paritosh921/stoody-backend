"""
Diagram Specification Helpers for JEE/NEET-style Question Generation.

Provides:
1. Diagram type to tool mapping (matplotlib, schemdraw, RDKit)
2. Subject-specific diagram spec templates
3. Validation functions for diagram specs
4. Example specs for LLM prompts
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Mapping: Diagram Type → Rendering Tool
# =============================================================================

DIAGRAM_TOOL_MAPPING = {
    # Physics - Circuits (schemdraw)
    "series_circuit": "schemdraw",
    "parallel_circuit": "schemdraw",
    "mixed_circuit": "schemdraw",
    "rc_circuit": "schemdraw",
    "rl_circuit": "schemdraw",
    "rlc_circuit": "schemdraw",
    "wheatstone_bridge": "schemdraw",
    "potentiometer": "schemdraw",
    "meter_bridge": "schemdraw",
    
    # Physics - Optics (matplotlib)
    "ray_diagram_lens": "matplotlib",
    "ray_diagram_mirror": "matplotlib",
    "lens_maker": "matplotlib",
    "prism_refraction": "matplotlib",
    "total_internal_reflection": "matplotlib",
    
    # Physics - Mechanics (matplotlib)
    "free_body_diagram": "matplotlib",
    "inclined_plane": "matplotlib",
    "projectile_motion": "matplotlib",
    "circular_motion": "matplotlib",
    "shm_diagram": "matplotlib",
    "pulley_system": "matplotlib",
    
    # Physics - Waves (matplotlib)
    "wave_diagram": "matplotlib",
    "standing_wave": "matplotlib",
    "interference_pattern": "matplotlib",
    "diffraction_pattern": "matplotlib",
    
    # Physics - Fields (matplotlib)
    "electric_field": "matplotlib",
    "magnetic_field": "matplotlib",
    "equipotential_lines": "matplotlib",
    
    # Chemistry - Molecules (RDKit)
    "molecule_2d": "rdkit",
    "molecule_3d": "rdkit",
    "reaction_mechanism": "rdkit",
    "resonance_structure": "rdkit",
    "isomer": "rdkit",
    "stereoisomer": "rdkit",
    
    # Chemistry - Other (matplotlib)
    "orbital_diagram": "matplotlib",
    "energy_level": "matplotlib",
    "titration_curve": "matplotlib",
    "phase_diagram": "matplotlib",
    "electrochemical_cell": "matplotlib",
    "lab_setup_distillation": "matplotlib",
    "lab_setup_titration": "matplotlib",
    "periodic_table_highlight": "matplotlib",
    
    # Math - Graphs (matplotlib)
    "coordinate_graph": "matplotlib",
    "function_graph": "matplotlib",
    "parametric_curve": "matplotlib",
    "polar_graph": "matplotlib",
    "3d_surface": "matplotlib",
    
    # Math - Geometry (matplotlib)
    "geometry_construction": "matplotlib",
    "triangle": "matplotlib",
    "circle_geometry": "matplotlib",
    "conic_section": "matplotlib",
    "solid_geometry": "matplotlib",
    
    # Math - Other (matplotlib)
    "number_line": "matplotlib",
    "venn_diagram": "matplotlib",
    "bar_chart": "matplotlib",
    "pie_chart": "matplotlib",
    "histogram": "matplotlib",
    "trig_circle": "matplotlib",
    
    # Biology (matplotlib)
    "cell_diagram": "matplotlib",
    "organ_system": "matplotlib",
    "genetics_cross": "matplotlib",
    "phylogenetic_tree": "matplotlib",
    "ecosystem_food_web": "matplotlib",
}


def get_tool_for_diagram_type(diagram_type: str) -> str:
    """
    Get the appropriate rendering tool for a diagram type.
    
    Args:
        diagram_type: Type of diagram
        
    Returns:
        Tool name: 'matplotlib', 'schemdraw', 'rdkit', or 'none'
    """
    # Normalize the diagram type
    normalized = diagram_type.lower().replace("-", "_").replace(" ", "_")
    
    # Check direct mapping
    if normalized in DIAGRAM_TOOL_MAPPING:
        return DIAGRAM_TOOL_MAPPING[normalized]
    
    # Check partial matches
    for key, tool in DIAGRAM_TOOL_MAPPING.items():
        if key in normalized or normalized in key:
            return tool
    
    # Default to matplotlib for unknown types
    logger.warning(f"Unknown diagram type '{diagram_type}', defaulting to matplotlib")
    return "matplotlib"


def get_subject_diagram_types(subject: str) -> List[str]:
    """
    Get supported diagram types for a subject.
    
    Args:
        subject: Subject name (physics, chemistry, math, biology)
        
    Returns:
        List of supported diagram types
    """
    subject_lower = subject.lower()
    
    if "phys" in subject_lower:
        return [
            "series_circuit", "parallel_circuit", "mixed_circuit",
            "ray_diagram_lens", "ray_diagram_mirror",
            "free_body_diagram", "inclined_plane", "projectile_motion",
            "wave_diagram", "electric_field", "magnetic_field",
        ]
    elif "chem" in subject_lower:
        return [
            "molecule_2d", "molecule_3d", "reaction_mechanism",
            "orbital_diagram", "energy_level", "titration_curve",
            "electrochemical_cell", "lab_setup_distillation",
        ]
    elif "math" in subject_lower:
        return [
            "coordinate_graph", "function_graph", "geometry_construction",
            "triangle", "circle_geometry", "conic_section",
            "number_line", "venn_diagram", "trig_circle",
        ]
    elif "bio" in subject_lower:
        return [
            "cell_diagram", "organ_system", "genetics_cross",
            "phylogenetic_tree", "ecosystem_food_web",
        ]
    else:
        return list(DIAGRAM_TOOL_MAPPING.keys())


# =============================================================================
# Diagram Spec Templates (For LLM Prompts)
# =============================================================================

DIAGRAM_SPEC_TEMPLATES = {
    # -------------------------------------------------------------------------
    # PHYSICS - Circuits (schemdraw)
    # -------------------------------------------------------------------------
    "series_circuit": {
        "tool": "schemdraw",
        "template": {
            "subject": "physics",
            "diagram_type": "series_circuit",
            "title": "Series Circuit",
            "params": {
                "circuit_type": "series",
                "components": [
                    {"type": "battery", "voltage": "12V", "label": "V"},
                    {"type": "resistor", "value": "100Ω", "label": "R1"},
                    {"type": "resistor", "value": "200Ω", "label": "R2"},
                ],
                "show_current_arrows": True,
                "show_labels": True,
            },
            "output": {"format": "svg", "width": 600, "height": 400},
            "rendering_notes": "Clear labels, no clutter, exam-ready",
        },
        "description": "Series circuit with battery and resistors in series",
    },
    
    "parallel_circuit": {
        "tool": "schemdraw",
        "template": {
            "subject": "physics",
            "diagram_type": "parallel_circuit",
            "title": "Parallel Circuit",
            "params": {
                "circuit_type": "parallel",
                "components": [
                    {"type": "battery", "voltage": "6V", "label": "V"},
                    {"branch_1": [{"type": "resistor", "value": "10Ω", "label": "R1"}]},
                    {"branch_2": [{"type": "resistor", "value": "20Ω", "label": "R2"}]},
                ],
                "show_current_arrows": True,
            },
            "output": {"format": "svg", "width": 600, "height": 400},
        },
        "description": "Parallel circuit with branching resistors",
    },
    
    # -------------------------------------------------------------------------
    # PHYSICS - Optics (matplotlib)
    # -------------------------------------------------------------------------
    "ray_diagram_lens": {
        "tool": "matplotlib",
        "template": {
            "subject": "physics",
            "diagram_type": "ray_diagram_lens",
            "title": "Convex Lens Ray Diagram",
            "params": {
                "lens_type": "convex",  # or "concave"
                "focal_length": 15,  # cm
                "object_distance": 30,  # cm
                "object_height": 5,  # cm
                "show_principal_axis": True,
                "show_focal_points": True,
                "show_optical_center": True,
                "ray_types": ["parallel", "center", "focal"],
                "labels": {
                    "object": "O",
                    "image": "I",
                    "focal_points": ["F", "F'"],
                },
            },
            "output": {"format": "svg", "width": 800, "height": 400},
            "rendering_notes": "Show all three principal rays clearly",
        },
        "description": "Ray diagram for lens showing image formation",
    },
    
    # -------------------------------------------------------------------------
    # PHYSICS - Mechanics (matplotlib)
    # -------------------------------------------------------------------------
    "inclined_plane": {
        "tool": "matplotlib",
        "template": {
            "subject": "physics",
            "diagram_type": "inclined_plane",
            "title": "Block on Inclined Plane",
            "params": {
                "angle": 30,  # degrees
                "mass": 5,  # kg
                "object_shape": "block",  # or "ball", "cylinder"
                "show_forces": True,
                "show_components": True,
                "friction": True,
                "friction_coefficient": 0.3,
                "labels": {
                    "angle": "θ = 30°",
                    "mass": "m = 5 kg",
                    "forces": ["mg", "N", "f", "mg sin θ", "mg cos θ"],
                },
            },
            "output": {"format": "svg", "width": 600, "height": 500},
            "rendering_notes": "Force vectors clearly labeled with components",
        },
        "description": "Inclined plane with force diagram",
    },
    
    "projectile_motion": {
        "tool": "matplotlib",
        "template": {
            "subject": "physics",
            "diagram_type": "projectile_motion",
            "title": "Projectile Motion",
            "params": {
                "initial_velocity": 20,  # m/s
                "launch_angle": 45,  # degrees
                "gravity": 10,  # m/s²
                "show_trajectory": True,
                "show_velocity_components": True,
                "show_max_height": True,
                "show_range": True,
                "time_markers": [0, 1, 2, 3, 4],  # seconds
            },
            "output": {"format": "svg", "width": 800, "height": 500},
        },
        "description": "Projectile motion trajectory with velocity components",
    },
    
    # -------------------------------------------------------------------------
    # CHEMISTRY - Molecules (RDKit)
    # -------------------------------------------------------------------------
    "molecule_2d": {
        "tool": "rdkit",
        "template": {
            "subject": "chemistry",
            "diagram_type": "molecule_2d",
            "title": "Molecular Structure",
            "params": {
                "smiles": "CCO",  # Ethanol
                "molecule_name": "Ethanol",
                "show_hydrogens": True,
                "show_atom_indices": False,
                "highlight_atoms": [],  # indices to highlight
                "highlight_color": "#FF0000",
                "bond_color": "#000000",
            },
            "output": {"format": "svg", "width": 400, "height": 300},
            "rendering_notes": "Clear bond lines, readable atom labels",
        },
        "description": "2D molecular structure using SMILES notation",
    },
    
    # Common SMILES for JEE/NEET
    "common_smiles": {
        "ethanol": "CCO",
        "methanol": "CO",
        "acetic_acid": "CC(=O)O",
        "benzene": "c1ccccc1",
        "phenol": "Oc1ccccc1",
        "aniline": "Nc1ccccc1",
        "acetone": "CC(=O)C",
        "formaldehyde": "C=O",
        "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "ethene": "C=C",
        "ethyne": "C#C",
        "butane": "CCCC",
        "isobutane": "CC(C)C",
        "cyclohexane": "C1CCCCC1",
        "toluene": "Cc1ccccc1",
    },
    
    # -------------------------------------------------------------------------
    # CHEMISTRY - Lab/Other (matplotlib)
    # -------------------------------------------------------------------------
    "orbital_diagram": {
        "tool": "matplotlib",
        "template": {
            "subject": "chemistry",
            "diagram_type": "orbital_diagram",
            "title": "Electron Configuration",
            "params": {
                "element": "Fe",
                "atomic_number": 26,
                "show_orbitals": ["1s", "2s", "2p", "3s", "3p", "4s", "3d"],
                "show_arrows": True,
                "highlight_unpaired": True,
            },
            "output": {"format": "svg", "width": 600, "height": 400},
        },
        "description": "Orbital diagram showing electron configuration",
    },
    
    # -------------------------------------------------------------------------
    # MATH - Graphs (matplotlib)
    # -------------------------------------------------------------------------
    "coordinate_graph": {
        "tool": "matplotlib",
        "template": {
            "subject": "math",
            "diagram_type": "coordinate_graph",
            "title": "Function Graph",
            "params": {
                "functions": [
                    {"expression": "x**2", "label": "y = x²", "color": "blue"},
                    {"expression": "2*x + 1", "label": "y = 2x + 1", "color": "red"},
                ],
                "x_range": [-5, 5],
                "y_range": [-2, 10],
                "show_grid": True,
                "show_axes": True,
                "axis_labels": {"x": "x", "y": "y"},
                "points": [
                    {"x": 2, "y": 4, "label": "(2, 4)"},
                ],
                "show_legend": True,
            },
            "output": {"format": "svg", "width": 600, "height": 600},
            "rendering_notes": "Clear grid lines, distinct curve colors",
        },
        "description": "Coordinate graph with functions and points",
    },
    
    # -------------------------------------------------------------------------
    # MATH - Geometry (matplotlib)
    # -------------------------------------------------------------------------
    "triangle": {
        "tool": "matplotlib",
        "template": {
            "subject": "math",
            "diagram_type": "triangle",
            "title": "Triangle ABC",
            "params": {
                "vertices": {
                    "A": [0, 0],
                    "B": [4, 0],
                    "C": [2, 3],
                },
                "show_labels": True,
                "show_sides": True,
                "show_angles": True,
                "angle_labels": ["∠A", "∠B", "∠C"],
                "side_lengths": {"AB": 4, "BC": 3.6, "CA": 3.6},
                "construction_lines": [],  # e.g., ["altitude_from_C", "median_from_A"]
            },
            "output": {"format": "svg", "width": 500, "height": 400},
        },
        "description": "Triangle with labeled vertices, sides, and angles",
    },
    
    "circle_geometry": {
        "tool": "matplotlib",
        "template": {
            "subject": "math",
            "diagram_type": "circle_geometry",
            "title": "Circle with Inscribed Angle",
            "params": {
                "center": [0, 0],
                "radius": 3,
                "points_on_circle": [
                    {"name": "A", "angle_degrees": 0},
                    {"name": "B", "angle_degrees": 90},
                    {"name": "P", "angle_degrees": 180},
                ],
                "chords": [["A", "B"]],
                "tangents": [],
                "show_center": True,
                "show_radii": ["A"],
                "inscribed_angles": [{"vertex": "P", "arc": ["A", "B"]}],
            },
            "output": {"format": "svg", "width": 500, "height": 500},
        },
        "description": "Circle with geometric constructions",
    },
}


def get_diagram_spec_template(diagram_type: str) -> Optional[Dict[str, Any]]:
    """
    Get the template for a diagram type.
    
    Args:
        diagram_type: Type of diagram
        
    Returns:
        Template dict or None
    """
    normalized = diagram_type.lower().replace("-", "_").replace(" ", "_")
    
    if normalized in DIAGRAM_SPEC_TEMPLATES:
        return DIAGRAM_SPEC_TEMPLATES[normalized]
    
    return None


def get_diagram_examples_for_subject(subject: str) -> str:
    """
    Get example diagram specs for a subject (for LLM prompts).
    
    Args:
        subject: Subject name
        
    Returns:
        Formatted examples string
    """
    import json
    
    subject_lower = subject.lower()
    examples = []
    
    if "phys" in subject_lower:
        types = ["series_circuit", "ray_diagram_lens", "inclined_plane"]
    elif "chem" in subject_lower:
        types = ["molecule_2d", "orbital_diagram"]
    elif "math" in subject_lower:
        types = ["coordinate_graph", "triangle"]
    else:
        types = ["coordinate_graph"]
    
    for dtype in types:
        template = DIAGRAM_SPEC_TEMPLATES.get(dtype)
        if template:
            examples.append(f"--- {dtype} ({template['tool']}) ---")
            examples.append(json.dumps(template["template"], indent=2))
            examples.append("")
    
    return "\n".join(examples)


# =============================================================================
# Diagram Spec Validation
# =============================================================================

def validate_diagram_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a diagram specification.
    
    Args:
        spec: Diagram specification to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["subject", "diagram_type"]
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Check tool validity
    diagram_type = spec.get("diagram_type", "")
    expected_tool = get_tool_for_diagram_type(diagram_type)
    
    # Check params based on diagram type
    params = spec.get("params", {})
    
    # RDKit diagrams need valid SMILES
    if expected_tool == "rdkit":
        if "smiles" not in params:
            errors.append("RDKit diagrams require 'smiles' in params")
        else:
            smiles = params["smiles"]
            # Basic SMILES validation (actual validation done by RDKit)
            if not smiles or not isinstance(smiles, str):
                errors.append("Invalid SMILES string")
    
    # Schemdraw circuits need components
    if expected_tool == "schemdraw":
        if "components" not in params:
            errors.append("Circuit diagrams require 'components' in params")
    
    # Graph diagrams need functions or data
    if "graph" in diagram_type.lower() or "plot" in diagram_type.lower():
        if "functions" not in params and "data" not in params and "points" not in params:
            errors.append("Graph diagrams require 'functions', 'data', or 'points' in params")
    
    # Geometry diagrams need vertices or points
    if "geometry" in diagram_type.lower() or "triangle" in diagram_type.lower():
        if "vertices" not in params and "points" not in params:
            errors.append("Geometry diagrams require 'vertices' or 'points' in params")
    
    # Check output format
    output = spec.get("output", {})
    if output:
        format = output.get("format", "svg")
        if format not in ["svg", "png", "pdf"]:
            errors.append(f"Invalid output format: {format}")
    
    return len(errors) == 0, errors


def validate_diagram_for_question(
    diagram_spec: Dict[str, Any],
    question_text: str,
    subject: str,
) -> Tuple[bool, List[str]]:
    """
    Validate that a diagram spec is appropriate for a question.
    
    Checks:
    - Diagram type matches subject
    - Key values from question are in diagram params
    - Diagram is not generic/placeholder
    
    Args:
        diagram_spec: Diagram specification
        question_text: Question text
        subject: Subject name
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    if not diagram_spec:
        return True, []  # No diagram is valid
    
    # Check subject match
    spec_subject = diagram_spec.get("subject", "").lower()
    if spec_subject and subject.lower() not in spec_subject and spec_subject not in subject.lower():
        warnings.append(f"Diagram subject '{spec_subject}' may not match question subject '{subject}'")
    
    # Check for placeholder/generic values
    params = diagram_spec.get("params", {})
    title = diagram_spec.get("title", "")
    
    generic_titles = ["diagram", "figure", "image", "graph", "chart"]
    if title.lower() in generic_titles:
        warnings.append("Diagram title is too generic - should be specific to the question")
    
    # Check for default/example values that should be customized
    if params.get("angle") == 30 and "45" in question_text:
        warnings.append("Diagram angle (30°) does not match question (mentions 45°)")
    
    if params.get("mass") == 5 and "10 kg" in question_text:
        warnings.append("Diagram mass does not match question value")
    
    # For molecules, check SMILES is provided
    diagram_type = diagram_spec.get("diagram_type", "")
    tool = get_tool_for_diagram_type(diagram_type)
    if tool == "rdkit":
        smiles = params.get("smiles", "")
        if not smiles:
            warnings.append("Molecule diagram missing SMILES string")
        elif smiles == "CCO":  # Default example
            # Check if question mentions ethanol
            if "ethanol" not in question_text.lower():
                warnings.append("Diagram uses ethanol SMILES but question may be about a different molecule")
    
    return len(warnings) == 0, warnings


# =============================================================================
# Prompt Enhancement: Diagram Spec Instructions
# =============================================================================

def get_enhanced_diagram_prompt(subject: str, diagram_required: bool) -> str:
    """
    Get enhanced diagram generation instructions for LLM prompts.
    
    Args:
        subject: Subject name
        diagram_required: Whether diagram is needed
        
    Returns:
        Formatted instruction string
    """
    if not diagram_required:
        return """
DIAGRAM: Not required for this question.
Set diagram_required=false and diagram_spec=null.
"""
    
    tool_hint = ""
    if "phys" in subject.lower():
        tool_hint = """
PHYSICS DIAGRAM TOOLS:
- Circuits (series, parallel, RC, RL): Use schemdraw
- Optics (ray diagrams, lenses, mirrors): Use matplotlib
- Mechanics (FBD, inclined plane, projectile): Use matplotlib
- Waves and Fields: Use matplotlib
"""
    elif "chem" in subject.lower():
        tool_hint = """
CHEMISTRY DIAGRAM TOOLS:
- Molecular structures: Use RDKit with valid SMILES string
  Common SMILES: ethanol=CCO, benzene=c1ccccc1, phenol=Oc1ccccc1
- Lab setups, orbital diagrams, titration curves: Use matplotlib
"""
    elif "math" in subject.lower():
        tool_hint = """
MATH DIAGRAM TOOLS:
- Function graphs, coordinate plots: Use matplotlib
- Geometry (triangles, circles, constructions): Use matplotlib
- Number lines, Venn diagrams: Use matplotlib
"""
    
    examples = get_diagram_examples_for_subject(subject)
    
    return f"""
DIAGRAM SPECIFICATION REQUIRED

{tool_hint}

CRITICAL DIAGRAM RULES:
1. Use the CORRECT tool for the diagram type
2. Include EXACT values from the question (NOT defaults)
3. All labels must be crisp and readable
4. Avoid clutter - exam-style simplicity
5. Use valid SMILES for RDKit molecules (do NOT hallucinate)

OUTPUT FORMAT:
"diagram_spec": {{
    "subject": "{subject.lower()}",
    "diagram_type": "specific_type_from_list",
    "title": "Descriptive title matching the question",
    "params": {{
        // ALL required parameters with values from the question
    }},
    "output": {{"format": "svg", "width": 600, "height": 400}},
    "rendering_notes": "print-friendly, clear labels"
}}

EXAMPLE SPECS:
{examples}
"""
