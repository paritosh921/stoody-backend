"""
Diagram Specification Validation and Feedback System.

This module provides:
1. Schema validation for diagram specs before rendering
2. Parameter extraction and correction suggestions
3. Structured feedback for LLM revision prompts
4. Integration with the LLM1-LLM2 feedback loop

Key principles:
- Validate EARLY (before rendering, not after)
- Provide SPECIFIC corrections (not generic "fix diagram")
- Track state across iterations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# SUPPORTED DIAGRAM TYPES BY SUBJECT
# ============================================================================

SUPPORTED_DIAGRAM_TYPES = {
    "physics": {
        "circuits": ["series_circuit", "parallel_circuit", "mixed_circuit"],
        "optics": ["ray_diagram_lens", "ray_diagram_mirror"],
        "wave_optics": ["interference_pattern", "double_slit_experiment", "diffraction_pattern", "single_slit_diffraction"],
        "mechanics": ["free_body_diagram", "inclined_plane", "projectile_motion"],
        "waves": ["wave_diagram", "standing_wave", "wave_superposition"],
        "fields": ["electric_field", "magnetic_field"],
        "thermodynamics": ["pv_diagram", "carnot_cycle"],
        "modern_physics": ["photoelectric_effect", "atomic_spectrum"],
    },
    "chemistry": {
        "molecules": ["molecule_2d", "molecule_3d"],
        "reactions": ["reaction_scheme"],
        "lab": ["lab_setup_titration", "lab_setup_distillation", "lab_setup_electrolysis"],
        "other": ["orbital_diagram", "crystal_structure", "periodic_table_section"],
    },
    "math": {
        "graphs": ["coordinate_graph", "number_line", "bar_chart", "pie_chart", "3d_plot"],
        "geometry": ["geometry_construction", "trigonometric_circle", "venn_diagram"],
    },
    "maths": {  # Alias
        "graphs": ["coordinate_graph", "number_line", "bar_chart", "pie_chart", "3d_plot"],
        "geometry": ["geometry_construction", "trigonometric_circle", "venn_diagram"],
    },
    "biology": {
        "cells": ["plant_cell", "animal_cell"],
        "anatomy": ["human_heart", "human_brain", "nephron", "neuron", "eye_structure", "ear_structure"],
        "systems": ["digestive_system", "respiratory_system"],
        "processes": ["dna_replication", "mitosis_stages", "meiosis_stages"],
        "botany": ["flower_structure"],
    },
}

# Flatten for quick lookup
ALL_SUPPORTED_TYPES = {}
for subject, categories in SUPPORTED_DIAGRAM_TYPES.items():
    if subject not in ALL_SUPPORTED_TYPES:
        ALL_SUPPORTED_TYPES[subject] = set()
    for types in categories.values():
        ALL_SUPPORTED_TYPES[subject].update(types)


# ============================================================================
# REQUIRED PARAMETERS BY DIAGRAM TYPE
# ============================================================================

REQUIRED_PARAMETERS = {
    # Circuits
    "series_circuit": ["components", "voltage"],
    "parallel_circuit": ["components", "voltage"],
    "mixed_circuit": ["components", "voltage"],
    
    # Optics
    "ray_diagram_lens": ["lens_type", "focal_length", "object_distance"],
    "ray_diagram_mirror": ["mirror_type", "focal_length", "object_distance"],
    
    # Mechanics
    "free_body_diagram": ["forces"],
    "inclined_plane": ["angle", "mass"],
    "projectile_motion": ["initial_velocity", "angle"],
    
    # Waves
    "wave_diagram": ["amplitude", "wavelength"],
    "standing_wave": ["amplitude", "wavelength", "nodes"],
    "wave_superposition": ["wave1_amplitude", "wave2_amplitude"],
    
    # Wave Optics
    "interference_pattern": ["slit_separation", "wavelength", "screen_distance"],
    "double_slit_experiment": ["slit_separation", "wavelength", "screen_distance"],
    "diffraction_pattern": ["slit_width", "wavelength"],
    "single_slit_diffraction": ["slit_width", "wavelength"],
    
    # Thermodynamics
    "pv_diagram": ["process_type"],
    "carnot_cycle": ["t_hot", "t_cold"],
    
    # Modern Physics
    "photoelectric_effect": ["metal", "wavelength"],
    "atomic_spectrum": ["element"],
    
    # Fields
    "electric_field": ["charges"],
    "magnetic_field": ["field_type"],
    
    # Chemistry - Molecules (CRITICAL: must have SMILES)
    "molecule_2d": ["smiles"],
    "molecule_3d": ["smiles"],
    
    # Chemistry - Other
    "reaction_scheme": ["reactants", "products"],
    "orbital_diagram": ["element", "orbitals"],
    
    # Math - Graphs
    "coordinate_graph": ["functions"],
    "number_line": ["range", "points"],
    "bar_chart": ["data", "labels"],
    "pie_chart": ["data", "labels"],
    
    # Math - Geometry
    "geometry_construction": ["shapes"],
    "trigonometric_circle": ["angle"],
    "venn_diagram": ["sets"],
}

# Default values for optional parameters
DEFAULT_PARAMETERS = {
    "voltage": 12.0,
    "focal_length": 10.0,
    "object_distance": 15.0,
    "angle": 30.0,
    "mass": 1.0,
    "initial_velocity": 10.0,
    "amplitude": 1.0,
    "wavelength": 2.0,
    "gravity": 9.8,
    # Wave optics defaults
    "slit_separation": 0.5,  # mm
    "screen_distance": 1.0,  # m
    "slit_width": 0.1,  # mm
    # Thermodynamics defaults
    "t_hot": 500,  # K
    "t_cold": 300,  # K
}


# ============================================================================
# COMMON SMILES DICTIONARY FOR CHEMISTRY
# ============================================================================

COMMON_SMILES = {
    # Simple molecules
    "water": "O",
    "h2o": "O",
    "methane": "C",
    "ch4": "C",
    "ammonia": "N",
    "nh3": "N",
    "carbon dioxide": "O=C=O",
    "co2": "O=C=O",
    
    # Alcohols
    "methanol": "CO",
    "ethanol": "CCO",
    "propanol": "CCCO",
    "isopropanol": "CC(C)O",
    
    # Organic basics
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    
    # Carboxylic acids
    "acetic acid": "CC(=O)O",
    "formic acid": "C(=O)O",
    
    # Ketones/Aldehydes
    "acetone": "CC(=O)C",
    "formaldehyde": "C=O",
    "acetaldehyde": "CC=O",
    
    # Amines
    "methylamine": "CN",
    "ethylamine": "CCN",
    
    # Esters
    "ethyl acetate": "CCOC(=O)C",
    "methyl acetate": "COC(=O)C",
    
    # Common exam molecules
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    
    # Hydrocarbons
    "ethene": "C=C",
    "ethyne": "C#C",
    "propene": "CC=C",
    "butane": "CCCC",
    "pentane": "CCCCC",
    "hexane": "CCCCCC",
    "cyclohexane": "C1CCCCC1",
}


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class DiagramValidationResult:
    """Result of diagram spec validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)
    suggested_type: Optional[str] = None
    
    def to_feedback(self) -> str:
        """Convert to human-readable feedback for LLM revision."""
        lines = []
        
        if self.errors:
            lines.append("DIAGRAM ERRORS (must fix):")
            for error in self.errors:
                lines.append(f"  - {error}")
        
        if self.warnings:
            lines.append("DIAGRAM WARNINGS (should fix):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        if self.corrections:
            lines.append("SUGGESTED PARAMETER CORRECTIONS:")
            for param, value in self.corrections.items():
                lines.append(f"  - Set {param} to: {value}")
        
        if self.suggested_type:
            lines.append(f"SUGGESTED DIAGRAM TYPE: {self.suggested_type}")
        
        return "\n".join(lines) if lines else "No diagram issues found."


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_diagram_spec(
    diagram_spec: Optional[Dict[str, Any]],
    subject: str,
    question_text: str = "",
) -> DiagramValidationResult:
    """
    Validate a diagram specification before rendering.
    
    Args:
        diagram_spec: The diagram spec from LLM
        subject: Subject (physics, chemistry, math, biology)
        question_text: Question text for context extraction
        
    Returns:
        DiagramValidationResult with validity, errors, and corrections
    """
    result = DiagramValidationResult(is_valid=True)
    
    if not diagram_spec:
        result.is_valid = False
        result.errors.append("No diagram_spec provided")
        return result
    
    # Normalize subject - detect base subject from topic names
    # Handles: "Physics", "Wave Optics", "Physics Wave Optics", "Organic Chemistry", etc.
    subject_lower = subject.lower().strip()
    
    # Direct subject matches
    if subject_lower in ["physics", "chemistry", "biology", "math", "maths", "mathematics"]:
        if subject_lower in ["maths", "mathematics"]:
            subject_lower = "math"
    else:
        # Topic-based detection - map common topics to their base subjects
        PHYSICS_TOPICS = {
            "optics", "wave", "mechanics", "thermodynamics", "electromagnetism", 
            "magnetism", "electricity", "circuit", "kinematics", "dynamics",
            "gravitation", "oscillation", "sound", "light", "heat", "fluid",
            "quantum", "nuclear", "atomic", "relativity", "interference",
            "diffraction", "polarization", "ray", "lens", "mirror", "prism",
            "capacitor", "resistor", "inductor", "semiconductor", "photoelectric",
            "motion", "force", "energy", "momentum", "torque", "rotation",
        }
        CHEMISTRY_TOPICS = {
            "organic", "inorganic", "physical chemistry", "electrochemistry",
            "thermochemistry", "stoichiometry", "equilibrium", "kinetics",
            "acid", "base", "salt", "redox", "oxidation", "reduction",
            "bond", "molecule", "atom", "element", "compound", "reaction",
            "periodic", "metal", "nonmetal", "polymer", "hydrocarbon",
            "alcohol", "aldehyde", "ketone", "carboxylic", "ester", "amine",
        }
        BIOLOGY_TOPICS = {
            "cell", "genetics", "evolution", "ecology", "anatomy", "physiology",
            "botany", "zoology", "microbiology", "biochemistry", "molecular",
            "dna", "rna", "protein", "enzyme", "hormone", "neuron", "tissue",
            "organ", "system", "photosynthesis", "respiration", "digestion",
            "circulation", "excretion", "reproduction", "heredity", "mutation",
            "plant", "animal", "bacteria", "virus", "fungi", "ecosystem",
        }
        MATH_TOPICS = {
            "algebra", "geometry", "trigonometry", "calculus", "statistics",
            "probability", "arithmetic", "number", "equation", "function",
            "graph", "coordinate", "vector", "matrix", "polynomial", "logarithm",
            "exponential", "differentiation", "integration", "limit", "series",
            "sequence", "set", "logic", "permutation", "combination",
        }
        
        # Check which subject the topic belongs to
        detected_subject = None
        for topic in PHYSICS_TOPICS:
            if topic in subject_lower:
                detected_subject = "physics"
                break
        if not detected_subject:
            for topic in CHEMISTRY_TOPICS:
                if topic in subject_lower:
                    detected_subject = "chemistry"
                    break
        if not detected_subject:
            for topic in BIOLOGY_TOPICS:
                if topic in subject_lower:
                    detected_subject = "biology"
                    break
        if not detected_subject:
            for topic in MATH_TOPICS:
                if topic in subject_lower:
                    detected_subject = "math"
                    break
        
        if detected_subject:
            subject_lower = detected_subject
        # else: keep original subject_lower and let validation fail with helpful message
    
    # 1. Check diagram_type exists
    diagram_type = diagram_spec.get("diagram_type", "")
    if not diagram_type:
        result.is_valid = False
        result.errors.append("Missing diagram_type field")
        # Try to suggest based on subject
        result.suggested_type = _suggest_diagram_type(subject_lower, question_text)
        return result
    
    diagram_type_lower = diagram_type.lower().strip()
    
    # 2. Check diagram_type is supported for subject
    supported = ALL_SUPPORTED_TYPES.get(subject_lower, set())
    if diagram_type_lower not in supported:
        result.is_valid = False
        result.errors.append(
            f"Unsupported diagram_type '{diagram_type}' for subject '{subject}'. "
            f"Supported types: {', '.join(sorted(supported))}"
        )
        result.suggested_type = _suggest_diagram_type(subject_lower, question_text)
        return result
    
    # 3. Check required parameters (with variant name support)
    params = diagram_spec.get("params", diagram_spec.get("parameters", {}))
    if not isinstance(params, dict):
        params = {}
    
    required = REQUIRED_PARAMETERS.get(diagram_type_lower, [])
    missing_params = []
    
    # Parameter variant mappings - check base name and common variants
    PARAM_VARIANTS = {
        "wavelength": ["wavelength", "wavelength_nm", "wavelength_m", "lambda", "λ"],
        "slit_separation": ["slit_separation", "slit_separation_mm", "slit_separation_m", "d", "slit_distance"],
        "screen_distance": ["screen_distance", "screen_distance_m", "screen_distance_cm", "D", "distance_to_screen"],
        "slit_width": ["slit_width", "slit_width_mm", "slit_width_m", "a", "aperture_width"],
        "focal_length": ["focal_length", "focal_length_cm", "focal_length_m", "f"],
        "object_distance": ["object_distance", "object_distance_cm", "u"],
        "voltage": ["voltage", "V", "emf"],
        "angle": ["angle", "angle_deg", "angle_degrees", "theta", "θ"],
    }
    
    def has_param_variant(param_name: str, params_dict: dict, spec_dict: dict) -> bool:
        """Check if any variant of the parameter exists."""
        variants = PARAM_VARIANTS.get(param_name, [param_name])
        for variant in variants:
            if variant in params_dict or variant in spec_dict:
                return True
        return False
    
    for param in required:
        if not has_param_variant(param, params, diagram_spec):
            missing_params.append(param)
    
    # Only add defaults for TRULY missing parameters (no variants found)
    if missing_params:
        result.warnings.append(
            f"Missing recommended parameters for {diagram_type}: {', '.join(missing_params)}"
        )
        # Add default corrections ONLY if no variant exists
        for param in missing_params:
            if param in DEFAULT_PARAMETERS:
                result.corrections[param] = DEFAULT_PARAMETERS[param]
    
    # 4. Special validation for chemistry molecules
    if diagram_type_lower in ["molecule_2d", "molecule_3d"]:
        smiles = params.get("smiles") or diagram_spec.get("smiles")
        formula = params.get("formula") or diagram_spec.get("formula")
        molecule_name = params.get("molecule") or params.get("name") or diagram_spec.get("molecule")
        
        if not smiles:
            if molecule_name:
                # Try to convert molecule name to SMILES
                smiles = _get_smiles_for_molecule(molecule_name)
                if smiles:
                    result.corrections["smiles"] = smiles
                    result.warnings.append(f"Converted molecule name '{molecule_name}' to SMILES: {smiles}")
                else:
                    result.is_valid = False
                    result.errors.append(
                        f"Unknown molecule '{molecule_name}'. Please provide a valid SMILES string. "
                        f"Common molecules: {', '.join(list(COMMON_SMILES.keys())[:10])}"
                    )
            elif formula:
                # Try to get SMILES from formula
                smiles = _get_smiles_from_formula(formula)
                if smiles:
                    result.corrections["smiles"] = smiles
                else:
                    result.warnings.append(
                        f"Could not convert formula '{formula}' to SMILES. "
                        "Please provide a valid SMILES string."
                    )
            else:
                result.is_valid = False
                result.errors.append(
                    "Molecule diagrams require a 'smiles' parameter. "
                    "Example: 'smiles': 'CCO' for ethanol"
                )
    
    # 5. Special validation for circuits
    if diagram_type_lower in ["series_circuit", "parallel_circuit", "mixed_circuit"]:
        components = params.get("components") or diagram_spec.get("components")
        voltage = params.get("voltage") or params.get("V") or diagram_spec.get("voltage")
        
        if not components:
            result.warnings.append(
                "Circuit diagram has no components specified. "
                "Add components like: [{'type': 'resistor', 'name': 'R1', 'value': '10 Ω'}]"
            )
        
        if voltage is None:
            result.corrections["voltage"] = 12.0
            result.warnings.append("No voltage specified, defaulting to 12V")
    
    # 6. Special validation for optics
    if diagram_type_lower in ["ray_diagram_lens", "ray_diagram_mirror"]:
        focal_length = params.get("focal_length") or params.get("f")
        object_distance = params.get("object_distance") or params.get("u")
        
        if focal_length is None:
            result.corrections["focal_length"] = 10.0
            result.warnings.append("No focal_length specified, defaulting to 10 units")
        
        if object_distance is None:
            result.corrections["object_distance"] = 15.0
            result.warnings.append("No object_distance specified, defaulting to 15 units")
    
    # 7. Special validation for graphs
    if diagram_type_lower == "coordinate_graph":
        functions = params.get("functions") or diagram_spec.get("functions")
        if not functions:
            result.warnings.append(
                "Coordinate graph has no functions specified. "
                "Add functions like: [{'expression': 'x**2', 'label': 'f(x) = x²'}]"
            )
    
    return result


def _suggest_diagram_type(subject: str, question_text: str) -> Optional[str]:
    """Suggest a diagram type based on subject and question context."""
    question_lower = question_text.lower()
    
    if subject == "physics":
        if any(w in question_lower for w in ["circuit", "resistor", "capacitor", "voltage", "current"]):
            return "series_circuit"
        if any(w in question_lower for w in ["lens", "mirror", "focal", "image", "object"]):
            return "ray_diagram_lens"
        if any(w in question_lower for w in ["projectile", "trajectory", "throw", "launch"]):
            return "projectile_motion"
        if any(w in question_lower for w in ["incline", "slope", "friction", "block"]):
            return "inclined_plane"
        if any(w in question_lower for w in ["force", "tension", "weight", "normal"]):
            return "free_body_diagram"
        if any(w in question_lower for w in ["wave", "frequency", "amplitude"]):
            return "wave_diagram"
        return "free_body_diagram"  # Default for physics
    
    elif subject in ["chemistry", "chem"]:
        if any(w in question_lower for w in ["molecule", "structure", "compound", "organic", "formula"]):
            return "molecule_2d"
        if any(w in question_lower for w in ["reaction", "equation", "product"]):
            return "reaction_scheme"
        if any(w in question_lower for w in ["titration", "acid", "base"]):
            return "lab_setup_titration"
        if any(w in question_lower for w in ["orbital", "electron", "configuration"]):
            return "orbital_diagram"
        return "molecule_2d"  # Default for chemistry
    
    elif subject in ["math", "maths", "mathematics"]:
        if any(w in question_lower for w in ["graph", "function", "plot", "curve"]):
            return "coordinate_graph"
        if any(w in question_lower for w in ["triangle", "circle", "angle", "geometry"]):
            return "geometry_construction"
        if any(w in question_lower for w in ["number line", "integer"]):
            return "number_line"
        if any(w in question_lower for w in ["venn", "set", "union", "intersection"]):
            return "venn_diagram"
        if any(w in question_lower for w in ["bar", "histogram"]):
            return "bar_chart"
        if any(w in question_lower for w in ["pie", "percentage", "sector"]):
            return "pie_chart"
        return "coordinate_graph"  # Default for math
    
    elif subject in ["biology", "bio"]:
        if any(w in question_lower for w in ["cell", "nucleus", "mitochondria"]):
            if "plant" in question_lower:
                return "plant_cell"
            return "animal_cell"
        if any(w in question_lower for w in ["heart", "blood", "cardiac"]):
            return "human_heart"
        if any(w in question_lower for w in ["brain", "neuron", "nerve"]):
            return "human_brain"
        if any(w in question_lower for w in ["kidney", "nephron", "urine"]):
            return "nephron"
        if any(w in question_lower for w in ["dna", "replication", "genetic"]):
            return "dna_replication"
        if any(w in question_lower for w in ["mitosis", "cell division"]):
            return "mitosis_stages"
        if any(w in question_lower for w in ["eye", "vision", "retina"]):
            return "eye_structure"
        return "animal_cell"  # Default for biology
    
    return None


def _get_smiles_for_molecule(molecule_name: str) -> Optional[str]:
    """Convert molecule name to SMILES string."""
    name_lower = molecule_name.lower().strip()
    return COMMON_SMILES.get(name_lower)


def _get_smiles_from_formula(formula: str) -> Optional[str]:
    """Try to get SMILES from chemical formula."""
    formula_lower = formula.lower().strip()
    
    # Simple mapping for common formulas
    formula_to_smiles = {
        "h2o": "O",
        "ch4": "C",
        "nh3": "N",
        "co2": "O=C=O",
        "c2h5oh": "CCO",
        "ch3oh": "CO",
        "c6h6": "c1ccccc1",
        "c2h4": "C=C",
        "c2h2": "C#C",
    }
    
    return formula_to_smiles.get(formula_lower)


# ============================================================================
# FEEDBACK EXTRACTION FROM RENDERING ERRORS
# ============================================================================

def extract_diagram_feedback_from_error(error_message: str) -> Dict[str, Any]:
    """
    Extract structured feedback from rendering error messages.
    
    Args:
        error_message: Error message from diagram renderer
        
    Returns:
        Dict with corrections and suggestions
    """
    feedback = {
        "corrections": {},
        "suggestions": [],
    }
    
    error_lower = error_message.lower()
    
    # Check for unsupported type
    if "unsupported" in error_lower and "type" in error_lower:
        # Extract the type mentioned
        import re
        match = re.search(r"type['\"]?\s*[:=]\s*['\"]?(\w+)", error_lower)
        if match:
            feedback["suggestions"].append(f"Change diagram_type from '{match.group(1)}' to a supported type")
    
    # Check for missing SMILES
    if "smiles" in error_lower:
        feedback["suggestions"].append("Add a valid SMILES string for the molecule")
        feedback["corrections"]["smiles"] = "CCO"  # Default to ethanol as example
    
    # Check for invalid parameters
    if "invalid" in error_lower and "param" in error_lower:
        feedback["suggestions"].append("Check parameter values are numeric and within reasonable ranges")
    
    # Check for missing components
    if "component" in error_lower:
        feedback["suggestions"].append("Add circuit components with type, name, and value")
    
    return feedback


# ============================================================================
# INTEGRATION WITH REVISION PROMPT
# ============================================================================

def build_diagram_revision_instructions(
    validation_result: DiagramValidationResult,
    subject: str,
    current_diagram_type: Optional[str] = None,
) -> str:
    """
    Build specific diagram revision instructions for the LLM.
    
    Args:
        validation_result: Result from validate_diagram_spec
        subject: Subject for supported types info
        current_diagram_type: Current diagram type if any
        
    Returns:
        Formatted instructions string for revision prompt
    """
    lines = [
        "",
        "=== DIAGRAM REVISION INSTRUCTIONS ===",
        "The diagram specification needs corrections. Please update diagram_spec:",
        "",
    ]
    
    # Add validation feedback
    lines.append(validation_result.to_feedback())
    lines.append("")
    
    # Add supported types for reference
    supported = ALL_SUPPORTED_TYPES.get(subject.lower(), set())
    if supported:
        lines.append(f"SUPPORTED DIAGRAM TYPES for {subject}:")
        for dtype in sorted(supported):
            lines.append(f"  - {dtype}")
        lines.append("")
    
    # Add parameter requirements for current type
    if current_diagram_type and current_diagram_type in REQUIRED_PARAMETERS:
        required = REQUIRED_PARAMETERS[current_diagram_type]
        lines.append(f"REQUIRED PARAMETERS for {current_diagram_type}:")
        for param in required:
            default = DEFAULT_PARAMETERS.get(param, "no default")
            lines.append(f"  - {param} (default: {default})")
        lines.append("")
    
    # Add special instructions for chemistry
    if subject.lower() in ["chemistry", "chem"]:
        lines.append("CHEMISTRY DIAGRAM NOTES:")
        lines.append("  - Molecule diagrams REQUIRE a valid SMILES string")
        lines.append("  - Common SMILES: ethanol='CCO', benzene='c1ccccc1', water='O'")
        lines.append("  - If unsure of SMILES, use a simple molecule")
        lines.append("")
    
    # Add special instructions for circuits
    if subject.lower() == "physics" and current_diagram_type in ["series_circuit", "parallel_circuit", "mixed_circuit"]:
        lines.append("CIRCUIT DIAGRAM NOTES:")
        lines.append("  - Components should be: [{type, name, value}]")
        lines.append("  - Types: resistor, capacitor, inductor, lamp, switch, ammeter, voltmeter")
        lines.append("  - Include voltage source value")
        lines.append("")
    
    return "\n".join(lines)
