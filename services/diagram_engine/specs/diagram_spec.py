"""
Structured Diagram Specification System

This module implements the core diagram_spec architecture described in diagram_refactor_spec.md.

Key principles:
1. LLMs fill JSON specs, NOT write raw drawing code
2. Strict mapping from diagram_spec to correct rendering function
3. Each subject has specialized parameter schemas
4. Feedback updates the spec, triggering re-render with corrected parameters

Usage:
    spec = DiagramSpec.from_question(question_text, subject)
    renderer = RENDERER_MAPPING[spec.diagram_type]
    result = renderer.render(spec.to_render_params())
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DIAGRAM TYPE ENUMS - Strict types for each subject
# ============================================================================

class PhysicsDiagramType(str, Enum):
    """Supported physics diagram types - each maps to a specific renderer."""
    # Circuits - uses SchemDraw
    SERIES_CIRCUIT = "series_circuit"
    PARALLEL_CIRCUIT = "parallel_circuit"
    MIXED_CIRCUIT = "mixed_circuit"

    # Optics - uses matplotlib
    RAY_DIAGRAM_LENS = "ray_diagram_lens"
    RAY_DIAGRAM_MIRROR = "ray_diagram_mirror"

    # Mechanics - uses matplotlib
    FREE_BODY_DIAGRAM = "free_body_diagram"
    INCLINED_PLANE = "inclined_plane"
    PROJECTILE_MOTION = "projectile_motion"

    # Waves & Fields - uses matplotlib
    WAVE_DIAGRAM = "wave_diagram"
    ELECTRIC_FIELD = "electric_field"
    MAGNETIC_FIELD = "magnetic_field"


class ChemistryDiagramType(str, Enum):
    """Supported chemistry diagram types - each maps to a specific renderer."""
    # Molecules - uses RDKit
    MOLECULE_2D = "molecule_2d"
    MOLECULE_3D = "molecule_3d"

    # Reactions - uses matplotlib
    REACTION_SCHEME = "reaction_scheme"

    # Lab setups - uses matplotlib
    LAB_SETUP_TITRATION = "lab_setup_titration"
    LAB_SETUP_DISTILLATION = "lab_setup_distillation"
    LAB_SETUP_ELECTROLYSIS = "lab_setup_electrolysis"
    LAB_SETUP_MANOMETER = "lab_setup_manometer"

    # Other - uses matplotlib
    PERIODIC_TABLE_SECTION = "periodic_table_section"
    ORBITAL_DIAGRAM = "orbital_diagram"
    CRYSTAL_STRUCTURE = "crystal_structure"


class MathsDiagramType(str, Enum):
    """Supported maths diagram types - all use matplotlib."""
    COORDINATE_GRAPH = "coordinate_graph"
    GEOMETRY_CONSTRUCTION = "geometry_construction"
    NUMBER_LINE = "number_line"
    VENN_DIAGRAM = "venn_diagram"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    TRIGONOMETRIC_CIRCLE = "trigonometric_circle"


class BiologyDiagramType(str, Enum):
    """Supported biology diagram types - all use matplotlib."""
    PLANT_CELL = "plant_cell"
    ANIMAL_CELL = "animal_cell"
    HUMAN_HEART = "human_heart"
    HUMAN_BRAIN = "human_brain"
    NEPHRON = "nephron"
    NEURON = "neuron"
    DNA_REPLICATION = "dna_replication"
    MITOSIS_STAGES = "mitosis_stages"
    MEIOSIS_STAGES = "meiosis_stages"
    DIGESTIVE_SYSTEM = "digestive_system"
    RESPIRATORY_SYSTEM = "respiratory_system"
    EYE_STRUCTURE = "eye_structure"
    EAR_STRUCTURE = "ear_structure"
    FLOWER_STRUCTURE = "flower_structure"


class DiagramSubject(str, Enum):
    """Subject categories for diagrams."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATHS = "maths"
    BIOLOGY = "biology"


# ============================================================================
# PARAMETER MODELS - Specialized parameters for each diagram type
# ============================================================================

class CircuitComponent(BaseModel):
    """Component in a circuit diagram."""
    type: Literal["resistor", "capacitor", "inductor", "lamp", "switch",
                  "ammeter", "voltmeter", "led", "diode", "fuse"] = Field(
        default="resistor",
        description="Type of circuit component"
    )
    name: str = Field(default="", description="Component name/label (e.g., R1, C1)")
    value: Optional[str] = Field(default=None, description="Component value with unit (e.g., '10 Ω', '5 µF')")

    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        if v is not None and isinstance(v, str):
            # Ensure value has proper format
            return v.strip()
        return v


class CircuitParameters(BaseModel):
    """Parameters for circuit diagrams (series, parallel, mixed)."""
    components: List[CircuitComponent] = Field(
        default_factory=list,
        description="List of circuit components"
    )
    voltage: float = Field(default=12.0, description="Source voltage in volts")
    show_values: bool = Field(default=True, description="Show component values on diagram")
    circuit_topology: Optional[str] = Field(
        default=None,
        description="Optional topology description (e.g., 'R1 series with parallel(R2, R3)')"
    )


class OpticsParameters(BaseModel):
    """Parameters for ray diagrams (lens and mirror)."""
    lens_type: Optional[Literal["convex", "concave"]] = None
    mirror_type: Optional[Literal["concave", "convex"]] = None
    focal_length: float = Field(default=3.0, description="Focal length in cm")
    object_distance: float = Field(default=6.0, description="Object distance from lens/mirror in cm")
    object_height: float = Field(default=2.0, description="Object height in cm")
    show_labels: bool = Field(default=True, description="Show F, 2F, etc. labels")

    @model_validator(mode='after')
    def validate_optics_type(self):
        if self.lens_type is None and self.mirror_type is None:
            self.lens_type = "convex"  # Default to convex lens
        return self


class InclinedPlaneParameters(BaseModel):
    """Parameters for inclined plane diagrams."""
    angle: float = Field(default=30.0, ge=0, le=90, description="Angle of incline in degrees")
    mass: float = Field(default=10.0, ge=0, description="Mass of object in kg")
    object_type: Literal["box", "cylinder", "sphere", "ball", "disc"] = Field(
        default="box",
        description="Type of object on the plane"
    )
    radius: Optional[float] = Field(default=None, description="Radius for cylinder/sphere in m")
    plane_length: Optional[float] = Field(default=None, description="Length of plane in m")
    height: Optional[float] = Field(default=None, description="Height of plane in m")
    show_forces: bool = Field(default=True, description="Show force vectors")
    friction: bool = Field(default=True, description="Show friction force")
    friction_coefficient: Optional[float] = Field(default=None, description="Coefficient of friction")


class ProjectileParameters(BaseModel):
    """Parameters for projectile motion diagrams."""
    initial_velocity: float = Field(default=20.0, ge=0, description="Initial velocity in m/s")
    angle: float = Field(default=45.0, ge=0, le=90, description="Launch angle in degrees")
    initial_height: float = Field(default=0.0, ge=0, description="Initial height in m")
    gravity: float = Field(default=9.8, description="Gravitational acceleration in m/s²")
    show_trajectory: bool = Field(default=True, description="Show parabolic path")
    show_components: bool = Field(default=True, description="Show velocity components")


class WaveParameters(BaseModel):
    """Parameters for wave diagrams."""
    wave_type: Literal["transverse", "longitudinal", "standing"] = Field(
        default="transverse",
        description="Type of wave"
    )
    amplitude: float = Field(default=1.0, ge=0, description="Wave amplitude")
    wavelength: float = Field(default=2.0, ge=0, description="Wavelength")
    frequency: Optional[float] = Field(default=None, ge=0, description="Frequency in Hz")
    num_waves: int = Field(default=3, ge=1, le=10, description="Number of wavelengths to show")
    show_labels: bool = Field(default=True, description="Show wavelength, amplitude labels")


class FreeBodyParameters(BaseModel):
    """Parameters for free body diagrams."""
    forces: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of forces with direction and magnitude"
    )
    body_shape: Literal["box", "circle", "point"] = Field(
        default="box",
        description="Shape of the body"
    )
    body_label: str = Field(default="", description="Label for the body (e.g., 'm')")


class ElectricFieldParameters(BaseModel):
    """Parameters for electric field diagrams."""
    charges: List[Dict[str, float]] = Field(
        default_factory=lambda: [{"x": 0, "y": 0, "q": 1}],
        description="List of charges with x, y positions and charge value q"
    )
    show_field_lines: bool = Field(default=True)
    num_lines: int = Field(default=8, ge=4, le=16)


class MagneticFieldParameters(BaseModel):
    """Parameters for magnetic field diagrams."""
    field_type: Literal["bar_magnet", "wire", "solenoid"] = Field(default="bar_magnet")


# Chemistry Parameters

class MoleculeParameters(BaseModel):
    """Parameters for molecule diagrams - uses RDKit."""
    smiles: str = Field(
        ...,
        description="SMILES string for the molecule (REQUIRED for RDKit rendering)"
    )
    molecule_name: Optional[str] = Field(default=None, description="Common name of molecule")
    show_hydrogens: bool = Field(default=False, description="Show hydrogen atoms")
    highlight_atoms: List[int] = Field(default_factory=list, description="Atom indices to highlight")

    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v):
        """Validate SMILES is not empty."""
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        return v.strip()


class ReactionParameters(BaseModel):
    """Parameters for reaction scheme diagrams."""
    reactants: List[str] = Field(default_factory=list, description="List of reactant formulas")
    products: List[str] = Field(default_factory=list, description="List of product formulas")
    conditions: List[str] = Field(default_factory=list, description="Reaction conditions")
    arrow_type: Literal["single", "double", "equilibrium"] = Field(default="single")


class LabSetupParameters(BaseModel):
    """Parameters for lab setup diagrams."""
    # Titration
    burette_label: Optional[str] = None
    flask_label: Optional[str] = None
    indicator: Optional[str] = None

    # Distillation
    mixture_label: Optional[str] = None
    distillate_label: Optional[str] = None

    # Electrolysis
    electrolyte: Optional[str] = None
    anode_material: Optional[str] = None
    cathode_material: Optional[str] = None
    show_bubbles: bool = True

    # Manometer
    fluid_label: Optional[str] = None
    h_diff: Optional[float] = None
    h_diff_label: Optional[str] = None
    left_arm_connection: Optional[str] = None
    right_arm_connection: Optional[str] = None


class OrbitalDiagramParameters(BaseModel):
    """Parameters for orbital diagrams."""
    element: Union[str, int] = Field(
        default="C",
        description="Element symbol or atomic number"
    )
    show_boxes: bool = Field(default=True)


class PeriodicTableParameters(BaseModel):
    """Parameters for periodic table sections."""
    elements: List[int] = Field(
        default_factory=list,
        description="Atomic numbers to highlight"
    )
    highlight_groups: List[int] = Field(default_factory=list)
    highlight_periods: List[int] = Field(default_factory=list)


class CrystalParameters(BaseModel):
    """Parameters for crystal structure diagrams."""
    structure_type: Literal["cubic", "bcc", "fcc", "hcp"] = Field(default="cubic")
    element: str = Field(default="Na")
    show_unit_cell: bool = Field(default=True)


# Maths Parameters

class CoordinateGraphParameters(BaseModel):
    """Parameters for coordinate graphs."""
    functions: List[str] = Field(
        default_factory=list,
        description="List of function expressions to plot (e.g., 'x**2', 'sin(x)')"
    )
    x_range: tuple = Field(default=(-10, 10), description="X-axis range")
    y_range: tuple = Field(default=(-10, 10), description="Y-axis range")
    points: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Points to plot as [{'x': 1, 'y': 2, 'label': 'A'}]"
    )
    show_grid: bool = Field(default=True)


class GeometryParameters(BaseModel):
    """Parameters for geometry construction diagrams."""
    shapes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of shapes to draw"
    )
    angles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Angles to mark"
    )
    labels: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Labels to add"
    )


class NumberLineParameters(BaseModel):
    """Parameters for number line diagrams."""
    min_value: float = Field(default=-5)
    max_value: float = Field(default=5)
    points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Points to mark [{'value': 2, 'label': 'A'}]"
    )
    intervals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Intervals to highlight"
    )


class VennDiagramParameters(BaseModel):
    """Parameters for Venn diagrams."""
    sets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sets with labels and elements"
    )
    num_sets: int = Field(default=2, ge=2, le=3)


class ChartParameters(BaseModel):
    """Parameters for bar/pie charts."""
    data: Dict[str, float] = Field(
        default_factory=dict,
        description="Data as {label: value}"
    )
    title: Optional[str] = None
    colors: Optional[List[str]] = None


# ============================================================================
# MAIN DIAGRAM SPEC - The core structured specification
# ============================================================================

class DiagramSpec(BaseModel):
    """
    Structured diagram specification.

    This is the SINGLE SOURCE OF TRUTH for diagram generation.
    LLMs should fill this spec, NOT write drawing code.

    The spec maps directly to rendering functions:
    - diagram_type determines which renderer to use
    - parameters contain all values needed for rendering
    - Feedback updates the parameters, not the rendering logic
    """

    # Core identification
    subject: DiagramSubject = Field(
        ...,
        description="Subject area: physics, chemistry, maths, biology"
    )
    diagram_type: str = Field(
        ...,
        description="Specific diagram type from the subject's enum"
    )

    # Question context (for reference, not directly used in rendering)
    question_text: str = Field(
        default="",
        description="Original question text for context"
    )

    # Extracted parameters - the KEY data for rendering
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="All parameters needed for rendering (angle, mass, SMILES, etc.)"
    )

    # Labels and annotations
    labels: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Labels to add to the diagram"
    )

    # Visual settings
    title: Optional[str] = Field(default=None, description="Diagram title")
    show_labels: bool = Field(default=True, description="Whether to show labels")

    # Feedback tracking
    feedback_history: List[str] = Field(
        default_factory=list,
        description="History of feedback applied to this spec"
    )
    iteration: int = Field(default=0, description="Current iteration number")

    # Validation status
    is_validated: bool = Field(default=False, description="Whether spec has been validated")
    validation_errors: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"  # Allow extra fields for flexibility

    @model_validator(mode='after')
    def validate_diagram_type(self):
        """Ensure diagram_type is valid for the subject."""
        valid_types = get_valid_types_for_subject(self.subject)

        # Normalize the diagram type
        normalized_type = normalize_diagram_type(self.diagram_type)

        if normalized_type not in valid_types:
            # Log warning but don't fail - might be a new/custom type
            logger.warning(
                f"Diagram type '{self.diagram_type}' (normalized: {normalized_type}) "
                f"not in standard types for {self.subject.value}. "
                f"Valid types: {valid_types}"
            )
            self.validation_errors.append(
                f"Non-standard diagram type: {self.diagram_type}"
            )
        else:
            self.diagram_type = normalized_type
            self.is_validated = True

        return self

    def get_typed_parameters(self) -> BaseModel:
        """
        Get parameters as the correct typed model for this diagram type.
        This ensures all required fields are present with proper types.
        """
        param_class = PARAMETER_MODELS.get(self.diagram_type)
        if param_class:
            try:
                return param_class(**self.parameters)
            except Exception as e:
                logger.warning(f"Failed to parse parameters as {param_class.__name__}: {e}")
                return None
        return None

    def to_render_params(self) -> Dict[str, Any]:
        """
        Convert spec to parameters for rendering.
        This is what gets passed to the renderer.
        """
        render_params = {
            "subject": self.subject.value,
            "diagram_type": self.diagram_type,
            "title": self.title or "",
            "show_labels": self.show_labels,
            **self.parameters
        }

        # Add labels if present
        if self.labels:
            render_params["labels"] = self.labels

        return render_params

    def apply_feedback(self, feedback: str, corrections: Dict[str, Any]) -> "DiagramSpec":
        """
        Apply feedback to create an updated spec.

        Args:
            feedback: Textual feedback from verification
            corrections: Specific parameter corrections to apply

        Returns:
            New DiagramSpec with corrections applied
        """
        # Create a copy of current parameters
        updated_params = self.parameters.copy()

        # Apply corrections
        for key, value in corrections.items():
            if value is not None:
                updated_params[key] = value
                logger.info(f"Applied correction: {key} = {value}")

        # Create new spec with updated parameters
        return DiagramSpec(
            subject=self.subject,
            diagram_type=self.diagram_type,
            question_text=self.question_text,
            parameters=updated_params,
            labels=self.labels,
            title=self.title,
            show_labels=self.show_labels,
            feedback_history=self.feedback_history + [feedback],
            iteration=self.iteration + 1,
            is_validated=False  # Needs re-validation after corrections
        )

    def get_renderer_key(self) -> str:
        """
        Get the key used to look up the correct renderer.
        This is the primary routing mechanism.
        """
        return self.diagram_type


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_valid_types_for_subject(subject: DiagramSubject) -> List[str]:
    """Get list of valid diagram types for a subject."""
    type_enums = {
        DiagramSubject.PHYSICS: PhysicsDiagramType,
        DiagramSubject.CHEMISTRY: ChemistryDiagramType,
        DiagramSubject.MATHS: MathsDiagramType,
        DiagramSubject.BIOLOGY: BiologyDiagramType,
    }

    enum_class = type_enums.get(subject)
    if enum_class:
        return [e.value for e in enum_class]
    return []


def normalize_diagram_type(raw_type: str) -> str:
    """
    Normalize diagram type string to standard format.

    Handles common variations:
    - "series circuit" -> "series_circuit"
    - "Series_Circuit" -> "series_circuit"
    - "series-circuit" -> "series_circuit"
    """
    if not raw_type:
        return ""

    # Lowercase, replace spaces/hyphens with underscores
    normalized = raw_type.lower().replace(" ", "_").replace("-", "_")

    # Remove duplicate underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


# Mapping from diagram types to their parameter models
PARAMETER_MODELS: Dict[str, type] = {
    # Physics
    "series_circuit": CircuitParameters,
    "parallel_circuit": CircuitParameters,
    "mixed_circuit": CircuitParameters,
    "ray_diagram_lens": OpticsParameters,
    "ray_diagram_mirror": OpticsParameters,
    "free_body_diagram": FreeBodyParameters,
    "inclined_plane": InclinedPlaneParameters,
    "projectile_motion": ProjectileParameters,
    "wave_diagram": WaveParameters,
    "electric_field": ElectricFieldParameters,
    "magnetic_field": MagneticFieldParameters,

    # Chemistry
    "molecule_2d": MoleculeParameters,
    "molecule_3d": MoleculeParameters,
    "reaction_scheme": ReactionParameters,
    "lab_setup_titration": LabSetupParameters,
    "lab_setup_distillation": LabSetupParameters,
    "lab_setup_electrolysis": LabSetupParameters,
    "lab_setup_manometer": LabSetupParameters,
    "periodic_table_section": PeriodicTableParameters,
    "orbital_diagram": OrbitalDiagramParameters,
    "crystal_structure": CrystalParameters,

    # Maths
    "coordinate_graph": CoordinateGraphParameters,
    "geometry_construction": GeometryParameters,
    "number_line": NumberLineParameters,
    "venn_diagram": VennDiagramParameters,
    "bar_chart": ChartParameters,
    "pie_chart": ChartParameters,

    # Biology - uses generic parameters with shapes
    # (most biology diagrams are template-based)
}


# Diagram types that MUST use specialized libraries
SPECIALIZED_LIBRARY_TYPES: Dict[str, str] = {
    # SchemDraw for circuits
    "series_circuit": "schemdraw",
    "parallel_circuit": "schemdraw",
    "mixed_circuit": "schemdraw",

    # RDKit for molecules
    "molecule_2d": "rdkit",
    "molecule_3d": "rdkit",
}


def get_required_library(diagram_type: str) -> Optional[str]:
    """
    Get the required library for a diagram type.
    Returns None if matplotlib fallback is acceptable.
    """
    return SPECIALIZED_LIBRARY_TYPES.get(normalize_diagram_type(diagram_type))


def create_spec_from_plan(plan: Any) -> DiagramSpec:
    """
    Create a DiagramSpec from a DiagramPlan.

    This bridges the existing planning system with the new spec system.
    """
    # Extract parameters from plan's extracted_values and objects
    parameters = plan.extracted_values.copy() if plan.extracted_values else {}

    # Add object information if relevant
    if plan.objects:
        # For circuit diagrams, convert objects to components
        if "circuit" in plan.diagram_type.lower():
            components = []
            for obj in plan.objects:
                if obj.type.lower() not in ["battery", "cell", "source"]:
                    component = {
                        "type": _map_object_to_component_type(obj.type),
                        "name": obj.name,
                    }
                    if obj.properties and "value" in obj.properties:
                        component["value"] = str(obj.properties["value"])
                    components.append(component)
            parameters["components"] = components

    # Create labels list
    labels = []
    if plan.labels:
        for label in plan.labels:
            labels.append({
                "text": label.text,
                "position": label.position,
                "target_object": label.target_object,
                "font_size": label.font_size,
            })

    return DiagramSpec(
        subject=DiagramSubject(plan.subject.lower()),
        diagram_type=normalize_diagram_type(plan.diagram_type),
        question_text="",  # Will be filled by caller
        parameters=parameters,
        labels=labels,
        title=plan.diagram_type.replace("_", " ").title(),
    )


def _map_object_to_component_type(obj_type: str) -> str:
    """Map plan object types to circuit component types."""
    obj_type_lower = obj_type.lower()

    mappings = {
        "resistor": "resistor",
        "r": "resistor",
        "capacitor": "capacitor",
        "c": "capacitor",
        "inductor": "inductor",
        "l": "inductor",
        "lamp": "lamp",
        "bulb": "lamp",
        "light": "lamp",
        "switch": "switch",
        "ammeter": "ammeter",
        "voltmeter": "voltmeter",
        "led": "led",
        "diode": "diode",
        "fuse": "fuse",
    }

    for key, value in mappings.items():
        if key in obj_type_lower:
            return value

    return "resistor"  # Default


# Export common names matching chemistry conventions
COMMON_MOLECULE_SMILES: Dict[str, str] = {
    # Common organic molecules
    "methane": "C",
    "ethane": "CC",
    "propane": "CCC",
    "butane": "CCCC",
    "ethanol": "CCO",
    "methanol": "CO",
    "ethylene": "C=C",
    "acetylene": "C#C",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "c1ccc(O)cc1",
    "acetic acid": "CC(=O)O",
    "formic acid": "C(=O)O",

    # Common inorganic
    "water": "O",
    "ammonia": "N",
    "carbon dioxide": "O=C=O",
    "hydrogen sulfide": "S",
    "hydrogen peroxide": "OO",

    # Halogens
    "chloromethane": "CCl",
    "dichloromethane": "ClCCl",
    "chloroform": "ClC(Cl)Cl",
    "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
    "ccl4": "ClC(Cl)(Cl)Cl",

    # Alcohols
    "propanol": "CCCO",
    "isopropanol": "CC(C)O",
    "butanol": "CCCCO",

    # Acids
    "hydrochloric acid": "Cl",
    "sulfuric acid": "OS(=O)(=O)O",
    "nitric acid": "O=[N+]([O-])O",
}


def get_smiles_for_molecule(name: str) -> Optional[str]:
    """Get SMILES string for a common molecule name."""
    return COMMON_MOLECULE_SMILES.get(name.lower().strip())
