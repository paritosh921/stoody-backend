"""
Verified Diagram Service - Generates diagrams with verification loop.

Enhanced with Plan → Generate → Verify → Correct workflow:
1. Plan: Creates a structured diagram plan before generation
2. Generate: Uses validated plan for accurate generation
   - Routes to specialized renderers (SchemDraw for circuits, RDKit for molecules)
   - Falls back to matplotlib-based generator for custom diagrams
3. Verify: Checks with structured feedback (what/why/fix)
4. Correct: Regenerates with targeted feedback if needed

Supports adaptive iterations (3-7 based on complexity) and tracks best result.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from .tool_based_generator import ToolBasedDiagramGenerator
from .diagram_verifier import (
    DiagramVerifier, 
    VerificationResult, 
    VerificationStatus,
    EnhancedVerificationResult,
)
from .diagram_planner import DiagramPlanner, DiagramPlan

# Import for specialized rendering via DiagramEngine
try:
    from .engine import get_diagram_engine, DiagramEngine
    from .specs.base_spec import DiagramSubject, SUPPORTED_DIAGRAM_TYPES
    HAS_DIAGRAM_ENGINE = True
except ImportError:
    HAS_DIAGRAM_ENGINE = False
    DiagramEngine = None

logger = logging.getLogger(__name__)


@dataclass
class DiagramGenerationResult:
    """Result of verified diagram generation."""
    success: bool
    image_bytes: Optional[bytes] = None
    verification_result: Optional[VerificationResult] = None
    enhanced_verification: Optional[EnhancedVerificationResult] = None
    plan: Optional[DiagramPlan] = None
    attempts: int = 0
    final_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    is_best_effort: bool = False  # True if returned best result despite not being fully verified
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "has_image": self.image_bytes is not None,
            "verification": self.verification_result.to_dict() if self.verification_result else None,
            "attempts": self.attempts,
            "metadata": self.final_metadata,
            "error": self.error,
            "is_best_effort": self.is_best_effort,
        }
        
        if self.enhanced_verification:
            result["enhanced_verification"] = {
                "status": self.enhanced_verification.status.value if hasattr(self.enhanced_verification.status, 'value') else self.enhanced_verification.status,
                "is_acceptable": self.enhanced_verification.is_acceptable,
                "overall_score": self.enhanced_verification.get_composite_score(),
                "scores": {
                    "conceptual": self.enhanced_verification.conceptual_score,
                    "labeling": self.enhanced_verification.labeling_score,
                    "visual": self.enhanced_verification.visual_score,
                    "alignment": self.enhanced_verification.alignment_score,
                    "academic": self.enhanced_verification.academic_score,
                },
                "issues_count": len(self.enhanced_verification.issues),
                "critical_issues": [i.what for i in self.enhanced_verification.issues if i.severity == "critical"]
            }
        
        if self.plan:
            result["plan"] = {
                "diagram_type": self.plan.diagram_type,
                "complexity": self.plan.complexity_level,
                "objects_count": self.plan.get_object_count(),
                "labels_count": self.plan.get_label_count(),
            }
        
        return result


class VerifiedDiagramService:
    """
    Service that generates diagrams with verification and feedback loop.
    
    Enhanced with Plan → Generate → Verify → Correct workflow:
    - Creates a structured plan before generation
    - Routes to specialized renderers (SchemDraw, RDKit) when appropriate
    - Uses adaptive iterations (3-7 based on complexity)
    - Tracks best result across attempts
    - Uses enhanced verification with category scores
    
    This ensures diagrams are accurate, properly labeled, and follow
    JEE/NEET/CBSE academic standards.
    """
    
    # Adaptive iteration settings
    MIN_ATTEMPTS = 3
    MAX_ATTEMPTS = 7
    EARLY_ACCEPT_THRESHOLD = 0.75  # Accept best effort if score >= this (raised for quality)
    
    # Mapping from plan diagram types to engine-supported types
    # This allows the planner to use descriptive names that map to renderer types
    # ENHANCED: Comprehensive mapping with fuzzy matching support for common variations
    DIAGRAM_TYPE_MAPPING = {
        # Physics - Circuits (use SchemDraw - HIGH PRIORITY for quality)
        "circuit": "series_circuit",
        "series_circuit": "series_circuit",
        "series circuit": "series_circuit",
        "series-circuit": "series_circuit",
        "parallel_circuit": "parallel_circuit",
        "parallel circuit": "parallel_circuit",
        "parallel-circuit": "parallel_circuit",
        "mixed_circuit": "mixed_circuit",
        "mixed circuit": "mixed_circuit",
        "electrical_circuit": "series_circuit",
        "electrical circuit": "series_circuit",
        "circuit_diagram": "series_circuit",
        "circuit diagram": "series_circuit",
        "resistor_circuit": "series_circuit",
        "resistor circuit": "series_circuit",
        "rc_circuit": "series_circuit",
        "rl_circuit": "series_circuit",
        "rlc_circuit": "mixed_circuit",
        "wheatstone_bridge": "mixed_circuit",
        "wheatstone bridge": "mixed_circuit",
        "potentiometer": "series_circuit",
        "meter_bridge": "mixed_circuit",
        "meter bridge": "mixed_circuit",

        # Physics - Optics
        "ray_diagram": "ray_diagram_lens",
        "ray diagram": "ray_diagram_lens",
        "lens_diagram": "ray_diagram_lens",
        "lens diagram": "ray_diagram_lens",
        "convex_lens": "ray_diagram_lens",
        "convex lens": "ray_diagram_lens",
        "concave_lens": "ray_diagram_lens",
        "concave lens": "ray_diagram_lens",
        "mirror_diagram": "ray_diagram_mirror",
        "mirror diagram": "ray_diagram_mirror",
        "concave_mirror": "ray_diagram_mirror",
        "concave mirror": "ray_diagram_mirror",
        "convex_mirror": "ray_diagram_mirror",
        "convex mirror": "ray_diagram_mirror",
        "ray_diagram_lens": "ray_diagram_lens",
        "ray_diagram_mirror": "ray_diagram_mirror",
        "optics": "ray_diagram_lens",
        "optics_diagram": "ray_diagram_lens",
        "refraction": "ray_diagram_lens",
        "reflection": "ray_diagram_mirror",

        # Physics - Mechanics
        "force_diagram": "free_body_diagram",
        "force diagram": "free_body_diagram",
        "free_body_diagram": "free_body_diagram",
        "free body diagram": "free_body_diagram",
        "fbd": "free_body_diagram",
        "forces": "free_body_diagram",
        "force_analysis": "free_body_diagram",
        "inclined_plane": "inclined_plane",
        "inclined plane": "inclined_plane",
        "inclined_planes": "inclined_plane",
        "inclined_plane_experiment": "inclined_plane",
        "inclined_planes_experiment": "inclined_plane",
        "double_inclined_plane": "inclined_plane",
        "ramp": "inclined_plane",
        "ramp_diagram": "inclined_plane",
        "slope": "inclined_plane",
        "sliding_block": "inclined_plane",
        "rolling_cylinder": "inclined_plane",
        "rolling_sphere": "inclined_plane",
        "projectile_motion": "projectile_motion",
        "projectile motion": "projectile_motion",
        "projectile": "projectile_motion",
        "projectile_trajectory": "projectile_motion",
        "trajectory": "projectile_motion",
        "parabolic_motion": "projectile_motion",

        # Physics - Waves & Fields
        "wave_diagram": "wave_diagram",
        "wave diagram": "wave_diagram",
        "wave": "wave_diagram",
        "waves": "wave_diagram",
        "transverse_wave": "wave_diagram",
        "transverse wave": "wave_diagram",
        "longitudinal_wave": "wave_diagram",
        "longitudinal wave": "wave_diagram",
        "standing_wave": "wave_diagram",
        "electric_field": "electric_field",
        "electric field": "electric_field",
        "magnetic_field": "magnetic_field",
        "magnetic field": "magnetic_field",
        "field_lines": "electric_field",
        "field lines": "electric_field",
        "electromagnetic": "electric_field",

        # Chemistry - Molecules (use RDKit - HIGH PRIORITY for quality)
        "molecule": "molecule_2d",
        "molecule_2d": "molecule_2d",
        "molecule 2d": "molecule_2d",
        "molecule_3d": "molecule_3d",
        "molecule 3d": "molecule_3d",
        "molecular_structure": "molecule_2d",
        "molecular structure": "molecule_2d",
        "lewis_structure": "molecule_2d",
        "lewis structure": "molecule_2d",
        "structural_formula": "molecule_2d",
        "structural formula": "molecule_2d",
        "organic_compound": "molecule_2d",
        "organic compound": "molecule_2d",
        "chemical_structure": "molecule_2d",
        "chemical structure": "molecule_2d",
        "bond_structure": "molecule_2d",
        "covalent_structure": "molecule_2d",
        "benzene": "molecule_2d",
        "alkane": "molecule_2d",
        "alkene": "molecule_2d",
        "alkyne": "molecule_2d",
        "alcohol": "molecule_2d",
        "carboxylic_acid": "molecule_2d",
        "amine": "molecule_2d",
        "ester": "molecule_2d",

        # Chemistry - Reactions
        "reaction": "reaction_scheme",
        "reaction_scheme": "reaction_scheme",
        "reaction scheme": "reaction_scheme",
        "chemical_equation": "reaction_scheme",
        "chemical equation": "reaction_scheme",
        "chemical_reaction": "reaction_scheme",
        "chemical reaction": "reaction_scheme",
        "mechanism": "reaction_scheme",
        "reaction_mechanism": "reaction_scheme",

        # Chemistry - Lab
        "titration": "lab_setup_titration",
        "lab_setup_titration": "lab_setup_titration",
        "titration_setup": "lab_setup_titration",
        "acid_base_titration": "lab_setup_titration",
        "distillation": "lab_setup_distillation",
        "lab_setup_distillation": "lab_setup_distillation",
        "distillation_setup": "lab_setup_distillation",
        "fractional_distillation": "lab_setup_distillation",
        "electrolysis": "lab_setup_electrolysis",
        "lab_setup_electrolysis": "lab_setup_electrolysis",
        "electrolysis_setup": "lab_setup_electrolysis",
        "electrolytic_cell": "lab_setup_electrolysis",
        "manometer": "lab_setup_manometer",
        "lab_setup_manometer": "lab_setup_manometer",
        "u_tube_manometer": "lab_setup_manometer",
        "u-tube_manometer": "lab_setup_manometer",
        "pressure_measurement": "lab_setup_manometer",
        "gas_pressure": "lab_setup_manometer",

        # Chemistry - Other
        "periodic_table": "periodic_table_section",
        "periodic_table_section": "periodic_table_section",
        "periodic table": "periodic_table_section",
        "elements": "periodic_table_section",
        "orbital_diagram": "orbital_diagram",
        "orbital diagram": "orbital_diagram",
        "electron_configuration": "orbital_diagram",
        "electron configuration": "orbital_diagram",
        "aufbau": "orbital_diagram",
        "electron_orbital": "orbital_diagram",
        "crystal_structure": "crystal_structure",
        "crystal structure": "crystal_structure",
        "lattice": "crystal_structure",
        "unit_cell": "crystal_structure",
        "bcc": "crystal_structure",
        "fcc": "crystal_structure",
        "hcp": "crystal_structure",

        # Maths - Graphs and Coordinate Geometry
        "coordinate_graph": "coordinate_graph",
        "coordinate graph": "coordinate_graph",
        "graph": "coordinate_graph",
        "function_graph": "coordinate_graph",
        "function graph": "coordinate_graph",
        "plot": "coordinate_graph",
        "xy_plane": "coordinate_graph",
        "cartesian_plane": "coordinate_graph",
        "cartesian plane": "coordinate_graph",
        "coordinate_plane": "coordinate_graph",
        "coordinate plane": "coordinate_graph",
        "linear_graph": "coordinate_graph",
        "quadratic_graph": "coordinate_graph",
        "parabola": "coordinate_graph",
        "line_graph": "coordinate_graph",
        "vector_diagram": "coordinate_graph",
        "vector diagram": "coordinate_graph",

        # Maths - Geometry
        "geometry": "geometry_construction",
        "geometry_construction": "geometry_construction",
        "geometry construction": "geometry_construction",
        "triangle": "geometry_construction",
        "triangles": "geometry_construction",
        "quadrilateral": "geometry_construction",
        "polygon": "geometry_construction",
        "circle_geometry": "geometry_construction",
        "angle": "geometry_construction",
        "angles": "geometry_construction",
        "construction": "geometry_construction",
        "geometric_figure": "geometry_construction",

        # Maths - Number Line
        "number_line": "number_line",
        "number line": "number_line",
        "numberline": "number_line",
        "inequality": "number_line",
        "inequalities": "number_line",
        "interval": "number_line",
        "real_line": "number_line",
        "real line": "number_line",

        # Maths - Sets and Charts
        "venn_diagram": "venn_diagram",
        "venn diagram": "venn_diagram",
        "venn": "venn_diagram",
        "sets": "venn_diagram",
        "set_diagram": "venn_diagram",
        "bar_chart": "bar_chart",
        "bar chart": "bar_chart",
        "histogram": "bar_chart",
        "bar_graph": "bar_chart",
        "pie_chart": "pie_chart",
        "pie chart": "pie_chart",
        "pie_graph": "pie_chart",
        "circle_chart": "pie_chart",

        # Maths - Trigonometry
        "trigonometric_circle": "trigonometric_circle",
        "trig_circle": "trigonometric_circle",
        "unit_circle": "trigonometric_circle",
        "unit circle": "trigonometric_circle",

        # Biology - Cells
        "cell": "plant_cell",
        "plant_cell": "plant_cell",
        "plant cell": "plant_cell",
        "animal_cell": "animal_cell",
        "animal cell": "animal_cell",
        "eukaryotic_cell": "animal_cell",
        "prokaryotic_cell": "animal_cell",

        # Biology - Organs
        "heart": "human_heart",
        "human_heart": "human_heart",
        "human heart": "human_heart",
        "cardiac": "human_heart",
        "brain": "human_brain",
        "human_brain": "human_brain",
        "human brain": "human_brain",
        "cerebrum": "human_brain",
        "nephron": "nephron",
        "kidney_nephron": "nephron",
        "kidney": "nephron",
        "neuron": "neuron",
        "nerve_cell": "neuron",
        "nerve cell": "neuron",

        # Biology - Molecular
        "dna": "dna_replication",
        "dna_replication": "dna_replication",
        "dna replication": "dna_replication",
        "dna_structure": "dna_replication",
        "double_helix": "dna_replication",

        # Biology - Cell Division
        "mitosis": "mitosis_stages",
        "mitosis_stages": "mitosis_stages",
        "cell_division": "mitosis_stages",
        "meiosis": "meiosis_stages",
        "meiosis_stages": "meiosis_stages",

        # Biology - Systems
        "digestive_system": "digestive_system",
        "digestive system": "digestive_system",
        "alimentary_canal": "digestive_system",
        "respiratory_system": "respiratory_system",
        "respiratory system": "respiratory_system",
        "lungs": "respiratory_system",
        "eye": "eye_structure",
        "eye_structure": "eye_structure",
        "human_eye": "eye_structure",
        "ear": "ear_structure",
        "ear_structure": "ear_structure",
        "human_ear": "ear_structure",
        "flower": "flower_structure",
        "flower_structure": "flower_structure",
        "plant_flower": "flower_structure",
    }
    
    # Diagram types that should use specialized renderers (not matplotlib)
    # EXPANDED: Include ALL types that have dedicated renderer implementations
    SPECIALIZED_TYPES: Set[str] = {
        # Physics - SchemDraw for circuits
        "series_circuit", "parallel_circuit", "mixed_circuit",
        # Physics - Custom renderers (better quality than matplotlib tools)
        "ray_diagram_lens", "ray_diagram_mirror",
        "free_body_diagram",
        "inclined_plane",
        "projectile_motion",
        "wave_diagram",
        "electric_field", "magnetic_field",
        # Chemistry - RDKit for molecules  
        "molecule_2d", "molecule_3d",
        # Chemistry - Custom renderers
        "reaction_scheme",
        "lab_setup_titration", "lab_setup_distillation", "lab_setup_electrolysis",
        "lab_setup_manometer",
        "periodic_table_section",
        "orbital_diagram",
        "crystal_structure",
        # Maths - Custom renderers
        "coordinate_graph",
        "geometry_construction",
        "number_line",
        "venn_diagram",
        "bar_chart", "pie_chart",
        # Biology - Custom renderers
        "plant_cell", "animal_cell",
        "human_heart", "human_brain",
        "nephron", "neuron",
        "dna_replication",
        "mitosis_stages", "meiosis_stages",
        "digestive_system", "respiratory_system",
        "eye_structure", "ear_structure",
        "flower_structure",
    }
    
    def __init__(
        self,
        openai_service=None,
        kimi_service=None,
        use_kimi: bool = True,
        max_attempts: int = 5,
        verify_all: bool = True,
        use_planning: bool = True,
        use_enhanced_verification: bool = True,
        use_specialized_renderers: bool = True
    ):
        """
        Initialize the service.

        Args:
            openai_service: AsyncOpenAIService instance (fallback, used for embeddings only)
            kimi_service: KimiService instance (preferred for all LLM calls)
            use_kimi: Whether to use Kimi for diagram operations (default: True)
            max_attempts: Default maximum generation attempts (can be overridden by adaptive logic)
            verify_all: If True, verify all diagrams. If False, skip verification.
            use_planning: If True, create a plan before generation (recommended).
            use_enhanced_verification: If True, use enhanced verification with category scores.
            use_specialized_renderers: If True, use SchemDraw/RDKit for circuits/molecules.
        """
        self._openai_service = openai_service
        self._kimi_service = kimi_service
        self._use_kimi = use_kimi and kimi_service is not None

        # Initialize components with Kimi support
        self._generator = ToolBasedDiagramGenerator(
            openai_service=openai_service,
            kimi_service=kimi_service,
            use_kimi=self._use_kimi
        )
        self._verifier = DiagramVerifier(
            openai_service=openai_service,
            kimi_service=kimi_service,
            use_kimi=self._use_kimi
        )
        self._planner = DiagramPlanner(
            openai_service=openai_service,
            kimi_service=kimi_service,
            use_kimi=self._use_kimi
        )

        if self._use_kimi:
            logger.info("VerifiedDiagramService initialized with Kimi K2.5 for all LLM calls")
        else:
            logger.info("VerifiedDiagramService initialized with OpenAI for LLM calls")
        self._default_max_attempts = max_attempts
        self._verify_all = verify_all
        self._use_planning = use_planning
        self._use_enhanced_verification = use_enhanced_verification
        self._use_specialized_renderers = use_specialized_renderers and HAS_DIAGRAM_ENGINE
        
        # Initialize diagram engine for specialized renderers
        self._diagram_engine: Optional[DiagramEngine] = None
        if self._use_specialized_renderers:
            try:
                self._diagram_engine = get_diagram_engine()
                logger.info("DiagramEngine initialized for specialized rendering (SchemDraw, RDKit)")
            except Exception as e:
                logger.warning(f"Failed to initialize DiagramEngine: {e}. Using matplotlib fallback.")
                self._use_specialized_renderers = False
    
    def _get_engine_diagram_type(self, plan_type: str) -> Optional[str]:
        """Map plan diagram type to engine-supported type."""
        plan_type_lower = plan_type.lower().replace(" ", "_").replace("-", "_")
        return self.DIAGRAM_TYPE_MAPPING.get(plan_type_lower)
    
    def _should_use_specialized_renderer(self, diagram_type: str, subject: str) -> bool:
        """Check if specialized renderer (SchemDraw/RDKit) should be used."""
        if not self._use_specialized_renderers or not self._diagram_engine:
            return False
        
        engine_type = self._get_engine_diagram_type(diagram_type)
        if not engine_type:
            return False
        
        # Check if this is a specialized type (circuits, molecules)
        if engine_type in self.SPECIALIZED_TYPES:
            return True
        
        # Also use engine for other supported types as they may have better rendering
        try:
            subject_enum = DiagramSubject(subject.lower())
            supported = SUPPORTED_DIAGRAM_TYPES.get(subject_enum, [])
            return engine_type in supported
        except (ValueError, KeyError):
            return False
    
    async def _generate_with_engine(
        self,
        plan: DiagramPlan,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate diagram using specialized renderers (SchemDraw, RDKit).
        
        Args:
            plan: The diagram plan
            question_text: The question text
            subject: Subject area
            diagram_description: Optional description
            
        Returns:
            Tuple of (image_bytes, metadata) or (None, error_dict)
        """
        if not self._diagram_engine:
            return None, {"error": "DiagramEngine not available"}
        
        engine_type = self._get_engine_diagram_type(plan.diagram_type)
        if not engine_type:
            return None, {"error": f"No engine mapping for type: {plan.diagram_type}"}
        
        try:
            # Build spec from plan
            spec = self._build_engine_spec(plan, engine_type, subject, question_text, diagram_description)
            
            logger.info(f"Generating with specialized renderer: type={engine_type}, subject={subject}")
            
            # Get the appropriate renderer from the engine
            try:
                subject_enum = DiagramSubject(subject.lower())
            except ValueError:
                return None, {"error": f"Invalid subject: {subject}"}
            
            renderer = self._diagram_engine.get_renderer(subject_enum)
            if renderer is None:
                return None, {"error": f"No renderer for subject: {subject}"}
            
            # Check if diagram type is supported
            if not renderer.supports_type(engine_type):
                return None, {"error": f"Type {engine_type} not supported by {renderer.__class__.__name__}"}
            
            # Render directly to get image bytes
            render_result = await renderer.render(spec)
            
            if render_result and render_result.image_data:
                metadata = {
                    "tool_calls": 0,
                    "elements_drawn": len(plan.objects),
                    "title": plan.diagram_type.replace("_", " ").title(),
                    "renderer": "specialized",
                    "renderer_class": renderer.__class__.__name__,
                    "engine_type": engine_type,
                    "used_plan": True,
                    "width": render_result.width,
                    "height": render_result.height,
                }
                logger.info(f"Specialized renderer produced {len(render_result.image_data)} bytes")
                return render_result.image_data, metadata
            else:
                return None, {"error": "Renderer returned no image data"}
                
        except Exception as e:
            logger.error(f"Specialized renderer failed: {e}", exc_info=True)
            return None, {"error": str(e)}
    
    def _build_engine_spec(
        self,
        plan: DiagramPlan,
        engine_type: str,
        subject: str,
        question_text: str,
        diagram_description: Optional[str]
    ) -> Dict[str, Any]:
        """Build a spec dict for DiagramEngine from a plan.
        
        ENHANCED: Better integration between plan and specialized renderers.
        Ensures all extracted values, labels, and objects are properly passed.
        """
        spec = {
            "subject": subject.lower(),
            "diagram_type": engine_type,
            "title": plan.diagram_type.replace("_", " ").title(),
            "question_text": question_text,  # Include for context
        }
        
        if diagram_description:
            spec["description"] = diagram_description
        
        # Flatten extracted values directly into spec (renderers expect top-level keys)
        # ENHANCED: More thorough value extraction
        if plan.extracted_values:
            spec.update(plan.extracted_values)
            # Also keep a copy under 'parameters' for compatibility
            spec["parameters"] = plan.extracted_values
        
        # ENHANCED: Include labels from plan for renderers to use
        if plan.labels:
            spec["labels"] = [
                {
                    "text": label.text,
                    "position": label.position,
                    "target_object": label.target_object,
                    "font_size": label.font_size,
                    "includes_units": label.includes_units,
                }
                for label in plan.labels
            ]
        
        # ENHANCED: Include objects from plan
        if plan.objects:
            spec["objects"] = [
                {
                    "name": obj.name,
                    "type": obj.type,
                    "position": obj.position,
                    "dimensions": obj.dimensions,
                    "properties": obj.properties,
                }
                for obj in plan.objects
            ]
        
        # ENHANCED: Include annotations from plan
        if plan.annotations:
            spec["annotations"] = [
                {
                    "type": ann.type,
                    "value": ann.value,
                    "position": ann.position,
                    "properties": ann.properties,
                }
                for ann in plan.annotations
            ]
        
        # Add type-specific configuration
        if "circuit" in engine_type:
            spec["components"] = self._build_circuit_components(plan)
            # ENHANCED: Extract voltage from various possible sources
            self._extract_circuit_voltage(spec, plan)
        elif "molecule" in engine_type:
            spec["molecule"] = self._build_molecule_spec(plan, question_text)
        elif engine_type == "reaction_scheme":
            # ENHANCED: Extract reactants and products for chemical reactions
            self._extract_reaction_components(spec, plan, question_text)
        elif "ray_diagram" in engine_type:
            spec["optics"] = self._build_optics_spec(plan)
            # ENHANCED: Pass focal length, object distance etc from extracted values
            self._extract_optics_values(spec, plan)
        elif "free_body" in engine_type or "force" in plan.diagram_type.lower():
            spec["forces"] = self._build_forces_spec(plan)
        elif engine_type == "number_line":
            spec["number_line"] = self._build_number_line_spec(plan)
        elif "geometry" in engine_type:
            spec["geometry"] = self._build_geometry_spec(plan)
        elif engine_type == "inclined_plane":
            spec = self._build_inclined_plane_spec(spec, plan)
        elif engine_type == "projectile_motion":
            spec = self._build_projectile_spec(spec, plan)
        elif engine_type == "wave_diagram":
            spec = self._build_wave_spec(spec, plan)
        elif "manometer" in engine_type:
            spec = self._build_manometer_spec(spec, plan)
        
        return spec
    
    def _build_circuit_components(self, plan: DiagramPlan) -> list:
        """Build circuit components from plan objects.
        
        ENHANCED: Better extraction of component values and filtering.
        """
        import re
        
        components = []
        for obj in plan.objects:
            obj_type = obj.type.lower()
            
            # Skip battery - it's added separately by the renderer
            if 'battery' in obj_type or 'cell' in obj_type or 'source' in obj_type:
                continue
            
            # Map object type to component type
            component_type = 'resistor'  # default
            if 'resistor' in obj_type or obj_type == 'r':
                component_type = 'resistor'
            elif 'capacitor' in obj_type or obj_type == 'c':
                component_type = 'capacitor'
            elif 'inductor' in obj_type or obj_type == 'l':
                component_type = 'inductor'
            elif 'lamp' in obj_type or 'bulb' in obj_type or 'light' in obj_type:
                component_type = 'lamp'
            elif 'switch' in obj_type:
                component_type = 'switch'
            elif 'ammeter' in obj_type:
                component_type = 'ammeter'
            elif 'voltmeter' in obj_type:
                component_type = 'voltmeter'
            elif 'diode' in obj_type or 'led' in obj_type:
                component_type = 'led' if 'led' in obj_type else 'diode'
            elif 'fuse' in obj_type:
                component_type = 'fuse'
            
            component = {
                "type": component_type,
                "name": obj.name,
            }
            
            # Extract value from properties
            if obj.properties:
                value = obj.properties.get('value') or obj.properties.get('resistance') or obj.properties.get('capacitance')
                if value:
                    component['value'] = str(value)
                label = obj.properties.get('label')
                if label:
                    component['label'] = label
            
            # Extract value from dimensions (sometimes stored here)
            if obj.dimensions:
                for key in ['value', 'resistance', 'capacitance', 'inductance']:
                    if key in obj.dimensions:
                        component['value'] = str(obj.dimensions[key])
                        break
            
            components.append(component)
        
        return components
    
    def _extract_circuit_voltage(self, spec: Dict[str, Any], plan: DiagramPlan) -> None:
        """Extract circuit voltage from plan extracted values.
        
        Looks for voltage in various formats: 'voltage', 'V', 'EMF', 'battery_voltage', etc.
        """
        import re
        
        extracted = plan.extracted_values or {}
        
        # Look for voltage under various keys
        voltage_keys = ['voltage', 'V', 'v', 'emf', 'EMF', 'battery_voltage', 'battery', 'source_voltage', 'supply']
        
        for key in voltage_keys:
            if key in extracted:
                value = extracted[key]
                if isinstance(value, (int, float)):
                    spec['voltage'] = float(value)
                    return
                elif isinstance(value, str):
                    # Extract numeric value from string like "9V" or "9 volts"
                    match = re.search(r'(-?\d+(?:\.\d+)?)', value)
                    if match:
                        spec['voltage'] = float(match.group(1))
                        return
        
        # Also check labels for voltage
        for label in plan.labels:
            text = label.text.lower()
            if 'v' in text or 'volt' in text:
                match = re.search(r'(\d+(?:\.\d+)?)\s*v', text, re.IGNORECASE)
                if match:
                    spec['voltage'] = float(match.group(1))
                    return
    
    def _build_molecule_spec(self, plan: DiagramPlan, question_text: str) -> Dict[str, Any]:
        """Build molecule spec, trying to extract SMILES or formula."""
        import re

        mol_spec = {}

        # Try to find SMILES in extracted values
        if plan.extracted_values:
            for key, val in plan.extracted_values.items():
                if "smiles" in key.lower():
                    mol_spec["smiles"] = val
                elif "formula" in key.lower():
                    mol_spec["formula"] = val
                elif "name" in key.lower() and not mol_spec.get("name"):
                    mol_spec["name"] = val

        # Try to extract common molecule names or formulas from question
        if not mol_spec.get("smiles") and not mol_spec.get("formula"):
            # Common molecule patterns
            formula_pattern = r'\b([A-Z][a-z]?\d*)+\b'
            matches = re.findall(formula_pattern, question_text)
            for match in matches:
                if len(match) > 1 and any(c.isdigit() for c in match):
                    mol_spec["formula"] = match
                    break

        return mol_spec

    def _extract_reaction_components(
        self,
        spec: Dict[str, Any],
        plan: DiagramPlan,
        question_text: str
    ) -> None:
        """
        Extract reactants and products for chemical reaction diagrams.

        Looks for reaction components in:
        - plan.extracted_values under various keys
        - Direct 'equation' or 'reaction' strings that can be parsed
        - Question text for chemical formulas

        Args:
            spec: The spec dict to update (modified in place)
            plan: The diagram plan
            question_text: The original question text
        """
        import re

        extracted = plan.extracted_values or {}
        reactants = []
        products = []
        conditions = []

        # 1. Check for direct 'reactants'/'products' lists
        if extracted.get('reactants'):
            val = extracted['reactants']
            if isinstance(val, list):
                reactants = [r for r in val if r and r not in ['A', 'B', '?']]
            elif isinstance(val, str):
                reactants = [r.strip() for r in re.split(r'[,+]', val) if r.strip() and r.strip() not in ['A', 'B', '?']]

        if extracted.get('products'):
            val = extracted['products']
            if isinstance(val, list):
                products = [p for p in val if p and p not in ['C', '?']]
            elif isinstance(val, str):
                products = [p.strip() for p in re.split(r'[,+]', val) if p.strip() and p.strip() not in ['C', '?']]

        # 2. Check for 'equation' or 'reaction' string to parse
        equation = extracted.get('equation') or extracted.get('reaction') or extracted.get('chemical_equation')
        if equation and isinstance(equation, str):
            # Parse equation like "H2 + O2 -> H2O" or "2Na + Cl2 → 2NaCl"
            arrow_pattern = r'(?:->|→|⟶|=|⇌|-->)'
            parts = re.split(arrow_pattern, equation)
            if len(parts) >= 2:
                left_side = parts[0].strip()
                right_side = parts[-1].strip()

                # Parse left side (reactants)
                if not reactants:
                    left_comps = re.split(r'\s*\+\s*', left_side)
                    for comp in left_comps:
                        # Remove leading coefficient
                        clean = re.sub(r'^\d+\s*', '', comp.strip())
                        if clean and clean not in ['A', 'B', '?', '']:
                            reactants.append(clean)

                # Parse right side (products)
                if not products:
                    right_comps = re.split(r'\s*\+\s*', right_side)
                    for comp in right_comps:
                        # Remove leading coefficient
                        clean = re.sub(r'^\d+\s*', '', comp.strip())
                        if clean and clean not in ['C', '?', '']:
                            products.append(clean)

        # 3. Check for numbered keys like 'reactant_1', 'reactant_2', etc.
        for key, val in extracted.items():
            key_lower = key.lower()
            if 'reactant' in key_lower and val:
                if isinstance(val, str) and val not in ['A', 'B', '?', '']:
                    if val not in reactants:
                        reactants.append(val)
            elif 'product' in key_lower and val:
                if isinstance(val, str) and val not in ['C', '?', '']:
                    if val not in products:
                        products.append(val)
            elif 'condition' in key_lower or 'catalyst' in key_lower or 'temp' in key_lower:
                if val:
                    conditions.append(str(val))

        # 4. If still empty, try to extract from question text
        if not reactants or not products:
            # Look for chemical formulas in question
            formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b'
            all_formulas = re.findall(formula_pattern, question_text)
            # Deduplicate while preserving order
            seen = set()
            unique_formulas = []
            for f in all_formulas:
                if f not in seen:
                    seen.add(f)
                    unique_formulas.append(f)

            # Look for patterns like "reaction of X with Y to form Z"
            reaction_pattern = r'reaction\s+(?:of\s+)?(.+?)\s+(?:with|and)\s+(.+?)\s+(?:to\s+(?:form|produce|give)\s+)?(.+?)(?:\.|$)'
            match = re.search(reaction_pattern, question_text, re.IGNORECASE)
            if match:
                if not reactants:
                    reactants = [match.group(1).strip(), match.group(2).strip()]
                if not products and match.group(3):
                    products = [match.group(3).strip()]
            elif unique_formulas and not reactants:
                # Assume first half are reactants, second half are products
                mid = max(1, len(unique_formulas) // 2)
                reactants = unique_formulas[:mid]
                if not products:
                    products = unique_formulas[mid:]

        # 5. Check for arrow type based on keywords
        arrow_type = 'single'
        if extracted.get('arrow_type'):
            arrow_type = extracted['arrow_type']
        elif 'equilibrium' in question_text.lower() or 'reversible' in question_text.lower():
            arrow_type = 'equilibrium'

        # Update spec with extracted values
        if reactants:
            spec['reactants'] = reactants
            logger.info(f"Extracted reactants: {reactants}")
        if products:
            spec['products'] = products
            logger.info(f"Extracted products: {products}")
        if conditions:
            spec['conditions'] = conditions
        spec['arrow_type'] = arrow_type
    
    def _build_optics_spec(self, plan: DiagramPlan) -> Dict[str, Any]:
        """Build optics spec from plan."""
        optics = {}
        if plan.extracted_values:
            optics.update(plan.extracted_values)
        return optics
    
    def _build_forces_spec(self, plan: DiagramPlan) -> list:
        """Build forces list from plan objects."""
        forces = []
        for obj in plan.objects:
            if "force" in obj.type.lower() or "arrow" in obj.type.lower():
                force = {
                    "name": obj.name,
                    "direction": obj.properties.get("direction", "up"),
                }
                if obj.dimensions.get("magnitude"):
                    force["magnitude"] = obj.dimensions["magnitude"]
                forces.append(force)
        return forces

    def _build_inclined_plane_spec(self, spec: Dict[str, Any], plan: DiagramPlan) -> Dict[str, Any]:
        """Build inclined plane spec from plan.
        
        The PhysicsRenderer expects these top-level keys:
        - angle: Angle of incline in degrees
        - mass: Mass of object (kg)
        - object_type: 'box', 'cylinder', 'sphere', 'ball', 'disc'
        - radius: Radius of cylinder/sphere
        - plane_length: Length of the inclined plane
        - show_forces: Whether to show force vectors
        - friction: Whether to show friction force
        - labels: List of custom labels
        """
        # Extract angle from various possible keys in extracted_values
        extracted = plan.extracted_values or {}
        
        # Look for angle in different formats
        angle = extracted.get('angle') or extracted.get('theta') or extracted.get('θ')
        if angle:
            # Handle string values like "30°" or "30 degrees"
            if isinstance(angle, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', angle)
                if match:
                    spec['angle'] = float(match.group(1))
            else:
                spec['angle'] = float(angle)
        
        # Look for mass
        mass = extracted.get('mass') or extracted.get('m')
        if mass:
            if isinstance(mass, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', mass)
                if match:
                    spec['mass'] = float(match.group(1))
            else:
                spec['mass'] = float(mass)
        
        # Look for height
        height = extracted.get('height') or extracted.get('h')
        if height:
            if isinstance(height, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', height)
                if match:
                    spec['height'] = float(match.group(1))
            else:
                spec['height'] = float(height)
        
        # Check for object type from plan objects
        for obj in plan.objects:
            obj_type = obj.type.lower()
            if 'ball' in obj_type or 'sphere' in obj_type:
                spec['object_type'] = 'sphere'
            elif 'cylinder' in obj_type or 'disc' in obj_type:
                spec['object_type'] = 'cylinder'
            elif 'box' in obj_type or 'block' in obj_type:
                spec['object_type'] = 'box'
        
        # Add labels from plan
        if plan.labels:
            spec['labels'] = [
                {'text': label.text, 'position': label.position or 'bottom'}
                for label in plan.labels
            ]
        
        # Default to showing forces for educational diagrams
        spec.setdefault('show_forces', True)
        spec.setdefault('friction', True)
        
        return spec
    
    def _build_number_line_spec(self, plan: DiagramPlan) -> Dict[str, Any]:
        """Build number line spec from plan."""
        nl_spec = {
            "min": plan.coordinate_range.x_min if plan.coordinate_range else -5,
            "max": plan.coordinate_range.x_max if plan.coordinate_range else 5,
        }
        if plan.extracted_values:
            nl_spec.update(plan.extracted_values)
        return nl_spec
    
    def _build_geometry_spec(self, plan: DiagramPlan) -> Dict[str, Any]:
        """Build geometry spec from plan."""
        geo_spec = {
            "shapes": [],
        }
        for obj in plan.objects:
            shape = {
                "type": obj.type,
                "name": obj.name,
            }
            if obj.position:
                shape["position"] = obj.position
            if obj.dimensions:
                shape.update(obj.dimensions)
            geo_spec["shapes"].append(shape)
        return geo_spec

    def _extract_optics_values(self, spec: Dict[str, Any], plan: DiagramPlan) -> None:
        """Extract optics-specific values from plan into spec.
        
        Looks for focal length, object distance, image distance, magnification, etc.
        """
        import re
        
        extracted = plan.extracted_values or {}
        
        # Common optics parameters to extract
        optics_params = {
            'focal_length': ['focal_length', 'f', 'focal'],
            'object_distance': ['object_distance', 'u', 'do', 'object_dist'],
            'image_distance': ['image_distance', 'v', 'di', 'image_dist'],
            'magnification': ['magnification', 'm', 'mag'],
            'radius_of_curvature': ['radius_of_curvature', 'R', 'radius', 'curvature'],
            'lens_type': ['lens_type', 'type', 'lens'],
            'mirror_type': ['mirror_type', 'type', 'mirror'],
        }
        
        for target_key, source_keys in optics_params.items():
            for src_key in source_keys:
                if src_key in extracted:
                    value = extracted[src_key]
                    # Try to extract numeric value
                    if isinstance(value, str):
                        match = re.search(r'(-?\d+(?:\.\d+)?)', value)
                        if match:
                            spec[target_key] = float(match.group(1))
                    elif isinstance(value, (int, float)):
                        spec[target_key] = float(value)
                    break
    
    def _build_projectile_spec(self, spec: Dict[str, Any], plan: DiagramPlan) -> Dict[str, Any]:
        """Build projectile motion spec from plan.
        
        Extracts initial velocity, angle, height, range etc.
        """
        import re
        
        extracted = plan.extracted_values or {}
        
        # Extract projectile parameters
        param_mappings = {
            'initial_velocity': ['initial_velocity', 'velocity', 'v0', 'u', 'v'],
            'angle': ['angle', 'theta', 'θ', 'launch_angle'],
            'initial_height': ['initial_height', 'height', 'h', 'h0'],
            'range': ['range', 'R', 'horizontal_range'],
            'max_height': ['max_height', 'H', 'peak_height'],
            'time_of_flight': ['time_of_flight', 't', 'T', 'time'],
        }
        
        for target_key, source_keys in param_mappings.items():
            for src_key in source_keys:
                if src_key in extracted:
                    value = extracted[src_key]
                    if isinstance(value, str):
                        match = re.search(r'(-?\d+(?:\.\d+)?)', value)
                        if match:
                            spec[target_key] = float(match.group(1))
                    elif isinstance(value, (int, float)):
                        spec[target_key] = float(value)
                    break
        
        # Default show trajectory
        spec.setdefault('show_trajectory', True)
        spec.setdefault('show_components', True)
        
        return spec
    
    
    def _build_wave_spec(self, spec: Dict[str, Any], plan: DiagramPlan) -> Dict[str, Any]:
        """Build wave diagram spec from plan.
        
        Extracts wavelength, amplitude, frequency, etc.
        """
        import re
        
        extracted = plan.extracted_values or {}
        
        # Extract wave parameters
        param_mappings = {
            'wavelength': ['wavelength', 'lambda', 'λ'],
            'amplitude': ['amplitude', 'A'],
            'frequency': ['frequency', 'f', 'nu', 'ν'],
            'period': ['period', 'T'],
            'speed': ['speed', 'velocity', 'v', 'c'],
        }
        
        for target_key, source_keys in param_mappings.items():
            for src_key in source_keys:
                if src_key in extracted:
                    value = extracted[src_key]
                    if isinstance(value, str):
                        match = re.search(r'(-?\d+(?:\.\d+)?)', value)
                        if match:
                            spec[target_key] = float(match.group(1))
                    elif isinstance(value, (int, float)):
                        spec[target_key] = float(value)
                    break
        
        # Default wave type
        spec.setdefault('wave_type', 'transverse')
        
        return spec


    def _build_manometer_spec(self, spec: Dict[str, Any], plan: DiagramPlan) -> Dict[str, Any]:
        """Build manometer spec from plan."""
        extracted = plan.extracted_values or {}
        
        # Extract h_diff
        h_diff = extracted.get('height_difference') or extracted.get('h') or extracted.get('diff') or extracted.get('delta_h')
        if h_diff:
            spec['h_diff'] = h_diff
            
        # Extract fluid name
        fluid = extracted.get('fluid') or extracted.get('liquid')
        if fluid:
            spec['fluid_label'] = str(fluid).title()
            
        # Extract extracted values to connections if relevant
        for key, val in extracted.items():
            key_lower = key.lower()
            if 'gas' in key_lower and 'pressure' not in key_lower:
                spec['left_arm_connection'] = str(val)
            elif 'atm' in key_lower:
                spec['right_arm_connection'] = 'Atmosphere'
                
        # Manually check labels for h_diff label
        for label in plan.labels:
            if 'h' in label.text.lower() or 'mm' in label.text.lower() or 'cm' in label.text.lower():
                spec['h_diff_label'] = label.text
                
        return spec
        

        

    
    async def generate_verified_diagram(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None,
        diagram_spec: Optional[Dict[str, Any]] = None,
        skip_verification: bool = False,
        skip_planning: bool = False
    ) -> DiagramGenerationResult:
        """
        Generate a diagram with Plan → Generate → Verify → Correct workflow.
        
        The enhanced process:
        1. PLAN: Create a structured diagram plan (extracts values, objects, labels)
        2. GENERATE: Use validated plan for accurate generation
        3. VERIFY: Check with structured feedback (what/why/fix)
        4. CORRECT: Regenerate with targeted feedback if needed
        
        Uses adaptive iterations (3-7 based on complexity) and tracks best result.
        
        Args:
            question_text: The question that needs a diagram
            subject: Subject area (physics, maths, chemistry, biology)
            diagram_description: Optional description of what the diagram should show
            skip_verification: If True, skip verification (useful for simple diagrams)
            skip_planning: If True, skip the planning phase
            
        Returns:
            DiagramGenerationResult with image, verification, and plan details
        """
        should_verify = self._verify_all and not skip_verification
        should_plan = self._use_planning and not skip_planning
        
        plan: Optional[DiagramPlan] = None
        max_attempts = self._default_max_attempts
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: PLAN - Create and validate diagram plan
        # ═══════════════════════════════════════════════════════════════
        if diagram_spec and not plan:
            try:
                # Convert provided spec to plan
                plan = DiagramPlan(**diagram_spec)
                logger.info(f"Using provided diagram spec as plan: {plan.diagram_type}")
                # Re-validate to ensure it matches current rules
                if should_plan:
                    validation = self._planner.validate_plan(plan)
                    if not validation.is_valid:
                        logger.warning(f"Provided spec has validation issues: {[i.what for i in validation.issues]}")
            except Exception as e:
                logger.warning(f"Failed to convert diagram_spec to plan: {e}")

        if should_plan and not plan:
            try:
                logger.info("Phase 1: Creating diagram plan...")
                plan, plan_warnings = await self._planner.create_plan(question_text, subject, diagram_description)
                
                if plan is None:
                    logger.warning(f"Plan creation returned None: {plan_warnings}")
                else:
                    logger.info(
                        f"Plan created: type={plan.diagram_type}, "
                        f"objects={plan.get_object_count()}, labels={plan.get_label_count()}, "
                        f"complexity={plan.complexity_level}"
                    )
                    
                    if plan_warnings:
                        logger.info(f"Plan warnings: {plan_warnings}")
                    
                    # Adaptive attempts based on complexity
                    max_attempts = self._get_adaptive_attempts(plan)
                    logger.info(f"Adaptive attempts set to {max_attempts} based on complexity")
                    
            except Exception as e:
                logger.error(f"Planning failed: {e}. Continuing without plan.")
                plan = None
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2-4: GENERATE → VERIFY → CORRECT loop
        # ═══════════════════════════════════════════════════════════════
        attempt = 0
        feedback = None
        last_verification: Optional[VerificationResult] = None
        last_enhanced: Optional[EnhancedVerificationResult] = None
        
        # Track best result across attempts
        best_result: Optional[Dict[str, Any]] = None
        best_score: float = 0.0
        
        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Phase 2: Generation attempt {attempt}/{max_attempts}")
            
            # ═══════════════════════════════════════════════════════════
            # RENDERING STRATEGY (FIXED):
            # - Use specialized renderers (SchemDraw/RDKit) ONLY on first attempt
            # - For retries with feedback, use tool-based generator which can
            #   incorporate feedback into the generation process
            # - Specialized renderers are deterministic and cannot learn from feedback
            # ═══════════════════════════════════════════════════════════
            use_specialized = False

            # CRITICAL FIX: Only use specialized renderer on FIRST attempt (no feedback)
            # Specialized renderers are deterministic - they produce the same output
            # for the same input spec. They cannot incorporate verification feedback.
            # Tool-based generator can use feedback to adjust its approach.
            if plan and feedback is None:
                use_specialized = self._should_use_specialized_renderer(plan.diagram_type, subject)

            # Apply feedback to improve the plan (if any)
            if feedback and plan:
                logger.info(f"Updating plan based on feedback (attempt {attempt})...")
                try:
                    # Try LLM-based update first
                    new_plan, warnings = await self._planner.update_plan(plan, feedback, question_text)
                    if new_plan:
                        plan = new_plan
                        logger.info("Plan updated successfully via LLM")
                    else:
                        raise Exception("LLM returned None for plan update")
                except Exception as e:
                    logger.warning(f"LLM plan update failed: {e}. Falling back to heuristic update.")
                    plan = self._apply_heuristic_feedback(plan, feedback, last_enhanced)
                    logger.info("Applied heuristic feedback to plan")
            
            if use_specialized and plan:
                logger.info(f"Using specialized renderer for {plan.diagram_type} (attempt {attempt})")
                image_bytes, metadata = await self._generate_with_engine(
                    plan=plan,
                    question_text=question_text,
                    subject=subject,
                    diagram_description=diagram_description
                )
                
                if image_bytes is None:
                    logger.warning(f"Specialized renderer failed: {metadata.get('error')}. Falling back to tool-based generator.")
                    use_specialized = False
            
            # Use tool-based generator when:
            # 1. Specialized renderer not applicable for this diagram type
            # 2. Specialized renderer failed
            # 3. No plan available
            # 4. CRITICAL: On retry attempts WITH FEEDBACK (specialized can't learn from feedback)
            if not use_specialized or image_bytes is None:
                if feedback:
                    logger.info(f"Using tool-based generator with feedback (attempt {attempt}) - can incorporate verification feedback")
                else:
                    logger.info(f"Using tool-based generator (attempt {attempt})")
                    
                image_bytes, metadata = await self._generator.generate_diagram(
                    question_text=question_text,
                    subject=subject,
                    diagram_description=diagram_description,
                    feedback=feedback,
                    plan=plan
                )
            
            if image_bytes is None:
                error_msg = metadata.get("error", "Unknown generation error")
                logger.error(f"Attempt {attempt} failed to generate: {error_msg}")
                
                # If generation failed, don't retry - it's likely a system issue
                return DiagramGenerationResult(
                    success=False,
                    plan=plan,
                    attempts=attempt,
                    error=error_msg
                )
            
            renderer_info = metadata.get("renderer", "matplotlib")
            logger.info(f"Generated diagram with {metadata.get('tool_calls', 0)} tool calls (renderer: {renderer_info})")
            
            # Skip verification if requested
            if not should_verify:
                logger.info("Skipping verification (skip_verification=True)")
                return DiagramGenerationResult(
                    success=True,
                    image_bytes=image_bytes,
                    plan=plan,
                    attempts=attempt,
                    final_metadata=metadata
                )
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 3: VERIFY - Check with structured feedback
            # ═══════════════════════════════════════════════════════════
            logger.info("Phase 3: Verifying diagram...")
            
            if self._use_enhanced_verification:
                enhanced_verification = await self._verifier.verify_diagram_enhanced(
                    image_bytes=image_bytes,
                    question_text=question_text,
                    subject=subject,
                    diagram_description=diagram_description,
                    plan=plan
                )
                
                last_enhanced = enhanced_verification
                current_score = enhanced_verification.get_composite_score()
                
                status_str = enhanced_verification.status.value if hasattr(enhanced_verification.status, 'value') else enhanced_verification.status
                logger.info(
                    f"Attempt {attempt} verification: "
                    f"status={status_str}, "
                    f"acceptable={enhanced_verification.is_acceptable}, "
                    f"score={current_score:.2f} "
                    f"(concept={enhanced_verification.conceptual_score:.2f}, "
                    f"label={enhanced_verification.labeling_score:.2f}, "
                    f"visual={enhanced_verification.visual_score:.2f})"
                )
                
                # Track best result
                if current_score > best_score:
                    best_score = current_score
                    best_result = {
                        "image_bytes": image_bytes,
                        "metadata": metadata,
                        "enhanced_verification": enhanced_verification,
                        "attempt": attempt
                    }
                
                # Check if diagram is acceptable
                if enhanced_verification.is_acceptable:
                    logger.info(f"Diagram verified successfully after {attempt} attempt(s)")
                    return DiagramGenerationResult(
                        success=True,
                        image_bytes=image_bytes,
                        enhanced_verification=enhanced_verification,
                        plan=plan,
                        attempts=attempt,
                        final_metadata=metadata
                    )
                
                # ═══════════════════════════════════════════════════════
                # PHASE 4: CORRECT - Prepare targeted feedback
                # ═══════════════════════════════════════════════════════
                if attempt < max_attempts:
                    # Check for early acceptance based on best score
                    min_attempts_done = attempt >= self.MIN_ATTEMPTS
                    if min_attempts_done and best_score >= self.EARLY_ACCEPT_THRESHOLD:
                        logger.info(
                            f"Early accept: best_score={best_score:.2f} >= threshold={self.EARLY_ACCEPT_THRESHOLD} "
                            f"after {attempt} attempts"
                        )
                        return DiagramGenerationResult(
                            success=True,
                            image_bytes=best_result["image_bytes"],
                            enhanced_verification=best_result["enhanced_verification"],
                            plan=plan,
                            attempts=attempt,
                            final_metadata=best_result["metadata"],
                            is_best_effort=True
                        )
                    
                    # Build targeted feedback from enhanced verification
                    feedback = self._verifier.build_targeted_feedback(enhanced_verification)
                    logger.info(f"Diagram rejected. Feedback: {feedback[:200]}...")
                else:
                    logger.warning(f"Max attempts ({max_attempts}) reached")
                    
            else:
                # Standard verification (backward compatibility)
                verification = await self._verifier.verify_diagram(
                    image_bytes=image_bytes,
                    question_text=question_text,
                    subject=subject,
                    diagram_description=diagram_description
                )
                
                last_verification = verification
                
                logger.info(
                    f"Attempt {attempt} verification: status={verification.status.value if hasattr(verification.status, 'value') else verification.status}, "
                    f"acceptable={verification.is_acceptable}, confidence={verification.confidence:.2f}"
                )
                
                if verification.is_acceptable:
                    logger.info(f"Diagram verified successfully after {attempt} attempt(s)")
                    return DiagramGenerationResult(
                        success=True,
                        image_bytes=image_bytes,
                        verification_result=verification,
                        plan=plan,
                        attempts=attempt,
                        final_metadata=metadata
                    )
                
                if attempt < max_attempts:
                    feedback = self._build_feedback_prompt(verification)
                    logger.info(f"Diagram rejected. Feedback: {feedback[:200]}...")
        
        # ═══════════════════════════════════════════════════════════════
        # Max attempts reached - return best result if quality is acceptable
        # ═══════════════════════════════════════════════════════════════
        
        # Minimum quality threshold - below this, don't return any image
        # A diagram that confuses students is worse than no diagram
        MINIMUM_QUALITY_THRESHOLD = 0.5  # Raised from 0.3 - quality is critical for education
        
        if best_result and best_score >= 0.6:
            logger.info(f"Returning best effort result with score {best_score:.2f}")
            return DiagramGenerationResult(
                success=False,  # Not fully verified
                image_bytes=best_result["image_bytes"],
                enhanced_verification=best_result.get("enhanced_verification"),
                verification_result=last_verification,
                plan=plan,
                attempts=attempt,
                final_metadata=best_result["metadata"],
                error=f"Best effort after {attempt} attempts (score: {best_score:.2f})",
                is_best_effort=True
            )
        
        # If best score is below minimum threshold, don't return any image
        # A wrong diagram is worse than no diagram for students
        if best_score < MINIMUM_QUALITY_THRESHOLD:
            logger.error(
                f"Diagram generation failed completely - best score {best_score:.2f} "
                f"is below minimum threshold {MINIMUM_QUALITY_THRESHOLD}. "
                f"NOT returning diagram to avoid confusing students."
            )
            return DiagramGenerationResult(
                success=False,
                image_bytes=None,  # Don't return bad diagrams!
                enhanced_verification=last_enhanced,
                verification_result=last_verification,
                plan=plan,
                attempts=attempt,
                final_metadata=metadata,
                error=f"Diagram quality too low ({best_score:.0%}) - below minimum threshold ({MINIMUM_QUALITY_THRESHOLD:.0%})"
            )
        
        # Score is between MINIMUM_QUALITY_THRESHOLD and 0.5 - return with warning
        logger.warning(f"Returning low-quality diagram with score {best_score:.2f}")
        return DiagramGenerationResult(
            success=False,
            image_bytes=image_bytes,
            enhanced_verification=last_enhanced,
            verification_result=last_verification,
            plan=plan,
            attempts=attempt,
            final_metadata=metadata,
            error=f"Diagram verification failed after {attempt} attempts (best score: {best_score:.2f})",
            is_best_effort=True
        )
    

    def _get_adaptive_attempts(self, plan: DiagramPlan) -> int:
        """
        Calculate the number of attempts based on diagram complexity.
        
        Simple diagrams (≤3 objects, ≤5 labels): 3 attempts
        Moderate diagrams: 5 attempts
        Complex diagrams (>6 objects or >10 labels): 7 attempts
        
        Args:
            plan: The validated diagram plan
            
        Returns:
            Number of attempts (3-7)
        """
        if plan.is_simple():
            return self.MIN_ATTEMPTS  # 3
        elif plan.is_complex():
            return self.MAX_ATTEMPTS  # 7
        else:
            return 5  # Moderate
    
    def _apply_heuristic_feedback(
        self,
        plan: DiagramPlan,
        feedback: str,
        enhanced_verification: Optional[EnhancedVerificationResult] = None
    ) -> DiagramPlan:
        """
        Apply verification feedback to improve the diagram plan.
        
        This allows specialized renderers to be used on subsequent attempts
        by modifying the plan based on what the verifier identified as issues.
        
        Args:
            plan: The current diagram plan
            feedback: Text feedback from verification
            enhanced_verification: Structured verification result with scores and issues
            
        Returns:
            Modified DiagramPlan with improvements based on feedback
        """
        import copy
        import re
        
        # Create a deep copy to avoid modifying the original
        modified_plan = copy.deepcopy(plan)
        
        if not enhanced_verification:
            return modified_plan
        
        # Apply fixes based on specific issue categories
        for issue in enhanced_verification.issues:
            if issue.severity != "critical":
                continue
                
            fix_lower = issue.fix.lower()
            what_lower = issue.what.lower()
            
            # LABELING ISSUES - increase font sizes, improve positions
            if issue.category.value == "labeling":
                # Increase font sizes for all labels
                for label in modified_plan.labels:
                    if label.font_size < 16:
                        label.font_size = 16
                
                # Check for specific label issues
                if "overlap" in what_lower or "overlap" in fix_lower:
                    # Improve label positions
                    for label in modified_plan.labels:
                        if label.position == "inside":
                            label.position = "above"
                
                if "unit" in what_lower or "unit" in fix_lower:
                    # Mark labels as needing units
                    for label in modified_plan.labels:
                        if not label.includes_units:
                            label.includes_units = True
            
            # VISUAL ISSUES - improve layout and spacing
            elif issue.category.value == "visual":
                if "spacing" in what_lower or "crowded" in fix_lower:
                    # Expand coordinate range for more spacing
                    modified_plan.coordinate_range.x_min -= 1
                    modified_plan.coordinate_range.x_max += 1
                    modified_plan.coordinate_range.y_min -= 1
                    modified_plan.coordinate_range.y_max += 1
                
                if "small" in what_lower or "size" in fix_lower:
                    # Increase object dimensions
                    for obj in modified_plan.objects:
                        if "width" in obj.dimensions:
                            obj.dimensions["width"] = obj.dimensions["width"] * 1.2
                        if "height" in obj.dimensions:
                            obj.dimensions["height"] = obj.dimensions["height"] * 1.2
                        if "radius" in obj.dimensions:
                            obj.dimensions["radius"] = obj.dimensions["radius"] * 1.2
            
            # CONCEPTUAL ISSUES - these are hardest to fix automatically
            elif issue.category.value == "conceptual":
                # Look for specific value corrections in the fix
                value_patterns = [
                    (r'(\d+(?:\.\d+)?)\s*(?:N|newton)', 'force'),
                    (r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram)', 'mass'),
                    (r'(\d+(?:\.\d+)?)\s*(?:°|degree)', 'angle'),
                    (r'(\d+(?:\.\d+)?)\s*(?:m|meter|cm|centimeter)', 'length'),
                    (r'(\d+(?:\.\d+)?)\s*(?:V|volt)', 'voltage'),
                    (r'(\d+(?:\.\d+)?)\s*(?:Ω|ohm)', 'resistance'),
                    # Match "molecule as CCl4" or "formula CCl4"
                    (r'(?:molecule|formula)\s+(?:as\s+)?([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)', 'formula'),
                ]

                for pattern, key in value_patterns:
                    match = re.search(pattern, issue.fix, re.IGNORECASE)
                    if match:
                        modified_plan.extracted_values[key] = match.group(0)

                # ENHANCED: Handle chemistry reaction feedback
                # Look for reactants/products mentioned in feedback
                if 'reactant' in what_lower or 'product' in what_lower or 'reaction' in what_lower:
                    self._apply_chemistry_feedback(modified_plan, issue.fix, what_lower)
            
            # ALIGNMENT ISSUES - add missing elements
            elif issue.category.value == "alignment":
                if "missing" in what_lower:
                    # Add note about missing element
                    new_note = f"MUST ADD: {issue.what}"
                    if modified_plan.academic_notes:
                        modified_plan.academic_notes += f"; {new_note}"
                    else:
                        modified_plan.academic_notes = new_note
        
        # Apply improvements based on low category scores
        if enhanced_verification.labeling_score < 0.6:
            # Ensure all labels have good font size
            for label in modified_plan.labels:
                label.font_size = max(label.font_size, 16)
        
        if enhanced_verification.visual_score < 0.5:
            # Increase spacing
            modified_plan.coordinate_range.x_max += 2
            modified_plan.coordinate_range.y_max += 2
        
        logger.debug(f"Plan modified based on feedback: {len(enhanced_verification.issues)} issues processed")
        return modified_plan

    def _apply_chemistry_feedback(
        self,
        plan: DiagramPlan,
        fix_text: str,
        issue_text: str
    ) -> None:
        """
        Apply chemistry-specific feedback to update reaction components.

        This handles cases where the verifier identifies wrong reactants/products
        and provides corrections like:
        - "Show actual reactants H2 and O2"
        - "Products should be H2O, not just placeholder"
        - "Replace A with the actual formula Na"

        Args:
            plan: The plan to modify (modified in place)
            fix_text: The fix suggestion from verification
            issue_text: The issue description
        """
        import re

        # Pattern to find chemical formulas in feedback
        formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'

        # Look for reactant corrections
        reactant_patterns = [
            r'reactants?\s+(?:should\s+be\s+|are\s+|:)?\s*([A-Z][\w\d\s,+]+)',
            r'show\s+(?:actual\s+)?reactants?\s+([A-Z][\w\d\s,+]+)',
            r'replace\s+[AB]\s+with\s+([A-Z][\w\d]+)',
        ]

        for pattern in reactant_patterns:
            match = re.search(pattern, fix_text, re.IGNORECASE)
            if match:
                # Extract formulas from the matched text
                formulas = re.findall(formula_pattern, match.group(1))
                valid_formulas = [f for f in formulas if len(f) > 1 and f not in ['The', 'And', 'For', 'With']]
                if valid_formulas:
                    plan.extracted_values['reactants'] = valid_formulas
                    logger.info(f"Updated reactants from feedback: {valid_formulas}")
                    break

        # Look for product corrections
        product_patterns = [
            r'products?\s+(?:should\s+be\s+|are\s+|:)?\s*([A-Z][\w\d\s,+]+)',
            r'show\s+(?:actual\s+)?products?\s+([A-Z][\w\d\s,+]+)',
            r'forms?\s+([A-Z][\w\d]+)',
            r'produces?\s+([A-Z][\w\d]+)',
        ]

        for pattern in product_patterns:
            match = re.search(pattern, fix_text, re.IGNORECASE)
            if match:
                formulas = re.findall(formula_pattern, match.group(1))
                valid_formulas = [f for f in formulas if len(f) > 1 and f not in ['The', 'And', 'For', 'With']]
                if valid_formulas:
                    plan.extracted_values['products'] = valid_formulas
                    logger.info(f"Updated products from feedback: {valid_formulas}")
                    break

        # Also extract any standalone formulas mentioned
        if 'placeholder' in issue_text or 'generic' in issue_text or "'A'" in issue_text or "'B'" in issue_text:
            # The feedback is saying we're using placeholders - find real formulas
            all_formulas = re.findall(formula_pattern, fix_text)
            valid_formulas = [f for f in all_formulas if len(f) > 1 and f not in ['The', 'And', 'For', 'With', 'Show', 'Use']]

            if valid_formulas and 'reactants' not in plan.extracted_values:
                # Assume first formula is reactant
                plan.extracted_values['reactants'] = valid_formulas[:max(1, len(valid_formulas)//2)]
            if valid_formulas and 'products' not in plan.extracted_values:
                # Assume last formula is product
                if len(valid_formulas) > 1:
                    plan.extracted_values['products'] = valid_formulas[len(valid_formulas)//2:]

    def _build_feedback_prompt(self, verification: VerificationResult) -> str:
        """Build a detailed feedback prompt from verification result."""
        parts = []
        
        # Add main issue prominently
        if verification.feedback:
            parts.append(f"MAIN PROBLEM: {verification.feedback}")
        
        # List all specific issues with numbers for clarity
        if verification.issues:
            parts.append("\nSPECIFIC ISSUES TO FIX:")
            for i, issue in enumerate(verification.issues, 1):
                parts.append(f"  {i}. {issue}")
        
        # Add concrete suggestions - these are the most important
        if verification.suggestions:
            parts.append("\nEXACT CHANGES NEEDED:")
            for i, suggestion in enumerate(verification.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        
        # Add clear instructions
        parts.append("\nWHEN REDRAWING:")
        parts.append("- Focus on the MAIN PROBLEM first")
        parts.append("- Make sure all numerical values match the question")
        parts.append("- Keep the diagram simple and clear")
        parts.append("- Label important points/values")
        
        if not verification.issues and not verification.feedback:
            parts = [
                "The previous diagram had issues.",
                "Please redraw with more attention to:",
                "- Correct numerical values from the question",
                "- Clear labels on important elements",
                "- Proper representation of the concept"
            ]
        
        return "\n".join(parts)
    
    async def generate_simple_diagram(
        self,
        question_text: str,
        subject: str,
        diagram_description: Optional[str] = None
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate a diagram without verification loop.
        
        Use this for simple diagrams where verification overhead isn't needed.
        
        Args:
            question_text: The question that needs a diagram
            subject: Subject area
            diagram_description: Optional description
            
        Returns:
            Tuple of (image_bytes, metadata)
        """
        return await self._generator.generate_diagram(
            question_text=question_text,
            subject=subject,
            diagram_description=diagram_description
        )


def get_verified_diagram_service(
    openai_service=None,
    kimi_service=None,
    use_kimi: bool = True,
    max_attempts: int = 5,
    use_planning: bool = True,
    use_enhanced_verification: bool = True,
    use_specialized_renderers: bool = True
) -> VerifiedDiagramService:
    """
    Factory function to create a VerifiedDiagramService instance.

    Args:
        openai_service: AsyncOpenAIService instance (fallback, used for embeddings only)
        kimi_service: KimiService instance (preferred for all LLM calls)
        use_kimi: Whether to use Kimi for diagram operations (default: True)
        max_attempts: Default max attempts (can be adjusted by adaptive logic)
        use_planning: Enable Plan → Generate → Verify → Correct workflow
        use_enhanced_verification: Enable enhanced verification with category scores
        use_specialized_renderers: Enable SchemDraw (circuits) and RDKit (molecules)

    Returns:
        Configured VerifiedDiagramService
    """
    return VerifiedDiagramService(
        openai_service=openai_service,
        kimi_service=kimi_service,
        use_kimi=use_kimi,
        max_attempts=max_attempts,
        use_planning=use_planning,
        use_enhanced_verification=use_enhanced_verification,
        use_specialized_renderers=use_specialized_renderers
    )
