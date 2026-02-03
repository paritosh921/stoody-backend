"""
Diagram Spec Router

Provides strict mapping from diagram specifications to rendering functions.
This ensures the correct library is ALWAYS used:
- SchemDraw for circuits
- RDKit for molecules
- Matplotlib for everything else

The router:
1. Validates the spec before routing
2. Enforces library requirements
3. Falls back gracefully when specialized libraries unavailable
4. Logs all routing decisions for debugging
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable, Awaitable
from enum import Enum

from .specs.diagram_spec import (
    DiagramSpec,
    DiagramSubject,
    PhysicsDiagramType,
    ChemistryDiagramType,
    MathsDiagramType,
    BiologyDiagramType,
    SPECIALIZED_LIBRARY_TYPES,
    normalize_diagram_type,
)
from .specs.base_spec import OutputFormat

logger = logging.getLogger(__name__)


class RendererType(str, Enum):
    """Types of renderers available."""
    SCHEMDRAW = "schemdraw"
    RDKIT = "rdkit"
    MATPLOTLIB = "matplotlib"
    FALLBACK = "fallback"


class RenderingDecision:
    """
    Encapsulates the routing decision for a diagram spec.

    Contains:
    - Which renderer to use
    - Whether specialized library is required
    - Fallback options if specialized unavailable
    - Pre-processed parameters for the renderer
    """

    def __init__(
        self,
        spec: DiagramSpec,
        renderer_type: RendererType,
        subject: DiagramSubject,
        diagram_type: str,
        render_params: Dict[str, Any],
        requires_specialized: bool = False,
        fallback_available: bool = True,
        warnings: list = None
    ):
        self.spec = spec
        self.renderer_type = renderer_type
        self.subject = subject
        self.diagram_type = diagram_type
        self.render_params = render_params
        self.requires_specialized = requires_specialized
        self.fallback_available = fallback_available
        self.warnings = warnings or []

    def __str__(self):
        return (
            f"RenderingDecision(type={self.diagram_type}, "
            f"renderer={self.renderer_type.value}, "
            f"specialized={self.requires_specialized})"
        )


class SpecRouter:
    """
    Routes diagram specs to appropriate renderers.

    The router ensures:
    1. Circuits ALWAYS go to SchemDraw (when available)
    2. Molecules ALWAYS go to RDKit (when available)
    3. Other diagrams use matplotlib-based renderers
    4. Library availability is checked before routing
    """

    def __init__(self):
        """Initialize router with library availability checks."""
        self._check_libraries()

        # Build routing tables
        self._build_routing_tables()

    def _check_libraries(self):
        """Check which specialized libraries are available."""
        # Check SchemDraw
        try:
            import schemdraw
            import schemdraw.elements as elm
            self._has_schemdraw = True
            logger.info("SchemDraw available for circuit diagrams")
        except ImportError:
            self._has_schemdraw = False
            logger.warning("SchemDraw not available - circuits will use matplotlib fallback")

        # Check RDKit
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            self._has_rdkit = True
            logger.info("RDKit available for molecular structures")
        except ImportError:
            self._has_rdkit = False
            logger.warning("RDKit not available - molecules will use matplotlib fallback")

    def _build_routing_tables(self):
        """Build routing tables for each subject."""

        # Types that REQUIRE specialized libraries (no fallback acceptable)
        self._critical_specialized = {
            "molecule_2d": RendererType.RDKIT,
            "molecule_3d": RendererType.RDKIT,
        }

        # Types that PREFER specialized libraries but have fallback
        self._preferred_specialized = {
            "series_circuit": RendererType.SCHEMDRAW,
            "parallel_circuit": RendererType.SCHEMDRAW,
            "mixed_circuit": RendererType.SCHEMDRAW,
        }

        # Renderer for each diagram type
        self._type_to_renderer: Dict[str, RendererType] = {
            # Physics
            "series_circuit": RendererType.SCHEMDRAW,
            "parallel_circuit": RendererType.SCHEMDRAW,
            "mixed_circuit": RendererType.SCHEMDRAW,
            "ray_diagram_lens": RendererType.MATPLOTLIB,
            "ray_diagram_mirror": RendererType.MATPLOTLIB,
            "free_body_diagram": RendererType.MATPLOTLIB,
            "inclined_plane": RendererType.MATPLOTLIB,
            "projectile_motion": RendererType.MATPLOTLIB,
            "wave_diagram": RendererType.MATPLOTLIB,
            "electric_field": RendererType.MATPLOTLIB,
            "magnetic_field": RendererType.MATPLOTLIB,

            # Chemistry
            "molecule_2d": RendererType.RDKIT,
            "molecule_3d": RendererType.RDKIT,
            "reaction_scheme": RendererType.MATPLOTLIB,
            "lab_setup_titration": RendererType.MATPLOTLIB,
            "lab_setup_distillation": RendererType.MATPLOTLIB,
            "lab_setup_electrolysis": RendererType.MATPLOTLIB,
            "lab_setup_manometer": RendererType.MATPLOTLIB,
            "periodic_table_section": RendererType.MATPLOTLIB,
            "orbital_diagram": RendererType.MATPLOTLIB,
            "crystal_structure": RendererType.MATPLOTLIB,

            # Maths - all matplotlib
            "coordinate_graph": RendererType.MATPLOTLIB,
            "geometry_construction": RendererType.MATPLOTLIB,
            "number_line": RendererType.MATPLOTLIB,
            "venn_diagram": RendererType.MATPLOTLIB,
            "bar_chart": RendererType.MATPLOTLIB,
            "pie_chart": RendererType.MATPLOTLIB,
            "trigonometric_circle": RendererType.MATPLOTLIB,

            # Biology - all matplotlib
            "plant_cell": RendererType.MATPLOTLIB,
            "animal_cell": RendererType.MATPLOTLIB,
            "human_heart": RendererType.MATPLOTLIB,
            "human_brain": RendererType.MATPLOTLIB,
            "nephron": RendererType.MATPLOTLIB,
            "neuron": RendererType.MATPLOTLIB,
            "dna_replication": RendererType.MATPLOTLIB,
            "mitosis_stages": RendererType.MATPLOTLIB,
            "meiosis_stages": RendererType.MATPLOTLIB,
            "digestive_system": RendererType.MATPLOTLIB,
            "respiratory_system": RendererType.MATPLOTLIB,
            "eye_structure": RendererType.MATPLOTLIB,
            "ear_structure": RendererType.MATPLOTLIB,
            "flower_structure": RendererType.MATPLOTLIB,
        }

    def route(self, spec: DiagramSpec) -> RenderingDecision:
        """
        Route a diagram spec to the appropriate renderer.

        Args:
            spec: The DiagramSpec to route

        Returns:
            RenderingDecision with renderer type and parameters
        """
        diagram_type = normalize_diagram_type(spec.diagram_type)
        warnings = []

        # Get the preferred renderer for this type
        preferred_renderer = self._type_to_renderer.get(
            diagram_type,
            RendererType.MATPLOTLIB
        )

        # Check if specialized library is available
        actual_renderer = preferred_renderer
        requires_specialized = False
        fallback_available = True

        if preferred_renderer == RendererType.SCHEMDRAW:
            requires_specialized = True
            if not self._has_schemdraw:
                actual_renderer = RendererType.MATPLOTLIB
                warnings.append(
                    f"SchemDraw not available for {diagram_type}, using matplotlib fallback"
                )
                logger.warning(warnings[-1])

        elif preferred_renderer == RendererType.RDKIT:
            requires_specialized = True
            if not self._has_rdkit:
                # Molecules with matplotlib fallback are very limited
                actual_renderer = RendererType.MATPLOTLIB
                fallback_available = True  # We have a fallback, but it's not great
                warnings.append(
                    f"RDKit not available for {diagram_type}, using matplotlib fallback "
                    "(quality will be reduced)"
                )
                logger.warning(warnings[-1])

        # Build render parameters from spec
        render_params = self._build_render_params(spec, diagram_type, actual_renderer)

        decision = RenderingDecision(
            spec=spec,
            renderer_type=actual_renderer,
            subject=spec.subject,
            diagram_type=diagram_type,
            render_params=render_params,
            requires_specialized=requires_specialized,
            fallback_available=fallback_available,
            warnings=warnings
        )

        logger.info(f"Routing decision: {decision}")
        return decision

    def _build_render_params(
        self,
        spec: DiagramSpec,
        diagram_type: str,
        renderer_type: RendererType
    ) -> Dict[str, Any]:
        """
        Build render parameters from spec.

        This ensures all required parameters are present and properly formatted.
        """
        # Start with base parameters
        params = {
            "subject": spec.subject.value,
            "diagram_type": diagram_type,
            "title": spec.title or diagram_type.replace("_", " ").title(),
            "show_labels": spec.show_labels,
            "dimensions": {"width": 800, "height": 600},
            "output_format": "png",
        }

        # Merge spec parameters
        if spec.parameters:
            params.update(spec.parameters)

        # Add labels
        if spec.labels:
            params["labels"] = spec.labels

        # Type-specific parameter processing
        if diagram_type in ["series_circuit", "parallel_circuit", "mixed_circuit"]:
            params = self._process_circuit_params(params, spec)
        elif diagram_type in ["molecule_2d", "molecule_3d"]:
            params = self._process_molecule_params(params, spec)
        elif diagram_type == "inclined_plane":
            params = self._process_inclined_plane_params(params, spec)
        elif diagram_type in ["ray_diagram_lens", "ray_diagram_mirror"]:
            params = self._process_optics_params(params, spec)

        return params

    def _process_circuit_params(
        self,
        params: Dict[str, Any],
        spec: DiagramSpec
    ) -> Dict[str, Any]:
        """Process circuit-specific parameters."""
        # Ensure components list exists
        if "components" not in params:
            params["components"] = []

        # Extract voltage from various possible keys
        if "voltage" not in params:
            for key in ["voltage", "V", "v", "emf", "EMF", "battery_voltage"]:
                if key in params:
                    try:
                        params["voltage"] = float(str(params[key]).replace("V", "").strip())
                        break
                    except (ValueError, TypeError):
                        pass
            else:
                params["voltage"] = 12.0  # Default

        # Ensure show_values is set
        params.setdefault("show_values", True)

        return params

    def _process_molecule_params(
        self,
        params: Dict[str, Any],
        spec: DiagramSpec
    ) -> Dict[str, Any]:
        """Process molecule-specific parameters."""
        # SMILES is required - try to extract from various sources
        if "smiles" not in params:
            # Check for formula and convert
            formula = params.get("formula") or params.get("molecule") or params.get("compound")
            if formula:
                from .specs.diagram_spec import get_smiles_for_molecule
                smiles = get_smiles_for_molecule(str(formula))
                if smiles:
                    params["smiles"] = smiles
                    logger.info(f"Converted molecule name '{formula}' to SMILES: {smiles}")

        # Validate SMILES if present
        if "smiles" in params and self._has_rdkit:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(params["smiles"])
            if mol is None:
                logger.warning(f"Invalid SMILES: {params['smiles']}")
                # Try to find common name variant
                from .specs.diagram_spec import COMMON_MOLECULE_SMILES
                for name, smiles in COMMON_MOLECULE_SMILES.items():
                    if params["smiles"].lower() in name.lower():
                        params["smiles"] = smiles
                        logger.info(f"Corrected SMILES to: {smiles}")
                        break

        return params

    def _process_inclined_plane_params(
        self,
        params: Dict[str, Any],
        spec: DiagramSpec
    ) -> Dict[str, Any]:
        """Process inclined plane parameters."""
        import re

        # Extract angle from various possible formats
        if "angle" not in params or params["angle"] is None:
            for key in ["angle", "theta", "θ", "inclination"]:
                if key in params:
                    val = params[key]
                    if isinstance(val, (int, float)):
                        params["angle"] = float(val)
                        break
                    elif isinstance(val, str):
                        match = re.search(r'(\d+(?:\.\d+)?)', val)
                        if match:
                            params["angle"] = float(match.group(1))
                            break

        params.setdefault("angle", 30.0)
        params.setdefault("show_forces", True)
        params.setdefault("friction", True)

        return params

    def _process_optics_params(
        self,
        params: Dict[str, Any],
        spec: DiagramSpec
    ) -> Dict[str, Any]:
        """Process optics (ray diagram) parameters."""
        import re

        # Extract focal length
        for key in ["focal_length", "f", "F", "focal"]:
            if key in params:
                val = params[key]
                if isinstance(val, (int, float)):
                    params["focal_length"] = float(val)
                    break
                elif isinstance(val, str):
                    match = re.search(r'(\d+(?:\.\d+)?)', val)
                    if match:
                        params["focal_length"] = float(match.group(1))
                        break

        params.setdefault("focal_length", 3.0)
        params.setdefault("object_distance", 6.0)
        params.setdefault("object_height", 2.0)
        params.setdefault("show_labels", True)

        return params

    def is_specialized_available(self, diagram_type: str) -> bool:
        """Check if specialized library is available for a diagram type."""
        diagram_type = normalize_diagram_type(diagram_type)

        if diagram_type in ["series_circuit", "parallel_circuit", "mixed_circuit"]:
            return self._has_schemdraw
        elif diagram_type in ["molecule_2d", "molecule_3d"]:
            return self._has_rdkit

        return True  # matplotlib is always available

    def get_renderer_info(self, diagram_type: str) -> Dict[str, Any]:
        """Get information about the renderer for a diagram type."""
        diagram_type = normalize_diagram_type(diagram_type)

        renderer = self._type_to_renderer.get(diagram_type, RendererType.MATPLOTLIB)

        return {
            "diagram_type": diagram_type,
            "preferred_renderer": renderer.value,
            "specialized_available": self.is_specialized_available(diagram_type),
            "requires_specialized": diagram_type in self._critical_specialized,
        }


# Singleton instance
_router_instance: Optional[SpecRouter] = None


def get_spec_router() -> SpecRouter:
    """Get or create the spec router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = SpecRouter()
    return _router_instance
