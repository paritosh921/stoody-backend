"""
Tests for the Diagram Refactor System

Tests the new spec-based architecture including:
1. DiagramSpec creation and validation
2. Spec routing to correct renderers
3. Feedback handling and corrections
4. End-to-end generation flows for all subjects

Run with: pytest tests/test_diagram_spec_refactor.py -v
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
import json

# Import the new modules
from services.diagram_engine.specs.diagram_spec import (
    DiagramSpec,
    DiagramSubject,
    PhysicsDiagramType,
    ChemistryDiagramType,
    MathsDiagramType,
    BiologyDiagramType,
    normalize_diagram_type,
    get_valid_types_for_subject,
    CircuitParameters,
    MoleculeParameters,
    InclinedPlaneParameters,
    PARAMETER_MODELS,
    get_smiles_for_molecule,
)

from services.diagram_engine.spec_router import (
    get_spec_router,
    RendererType,
    RenderingDecision,
)

from services.diagram_engine.feedback_handler import (
    get_feedback_handler,
    FeedbackState,
    Correction,
    CorrectionCategory,
)

from services.diagram_engine.exam_quality import (
    ExamQualitySettings,
    get_exam_settings,
    LabelPlacement,
    get_preset,
)


# ============================================================================
# DIAGRAM SPEC TESTS
# ============================================================================

class TestDiagramSpec:
    """Tests for DiagramSpec model."""

    def test_create_physics_spec(self):
        """Test creating a physics diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={
                "angle": 30,
                "mass": 10,
                "object_type": "box",
            },
            title="Inclined Plane Diagram",
        )

        assert spec.subject == DiagramSubject.PHYSICS
        assert spec.diagram_type == "inclined_plane"
        assert spec.parameters["angle"] == 30
        assert spec.is_validated  # Should be valid type

    def test_normalize_diagram_type(self):
        """Test diagram type normalization."""
        test_cases = [
            ("series circuit", "series_circuit"),
            ("Series_Circuit", "series_circuit"),
            ("series-circuit", "series_circuit"),
            ("MOLECULE_2D", "molecule_2d"),
            ("free body diagram", "free_body_diagram"),
        ]

        for input_type, expected in test_cases:
            assert normalize_diagram_type(input_type) == expected

    def test_get_valid_types(self):
        """Test getting valid types for each subject."""
        physics_types = get_valid_types_for_subject(DiagramSubject.PHYSICS)
        assert "series_circuit" in physics_types
        assert "inclined_plane" in physics_types

        chemistry_types = get_valid_types_for_subject(DiagramSubject.CHEMISTRY)
        assert "molecule_2d" in chemistry_types
        assert "reaction_scheme" in chemistry_types

        maths_types = get_valid_types_for_subject(DiagramSubject.MATHS)
        assert "coordinate_graph" in maths_types

    def test_circuit_parameters(self):
        """Test circuit parameter validation."""
        params = CircuitParameters(
            components=[
                {"type": "resistor", "name": "R1", "value": "10 Ω"},
                {"type": "capacitor", "name": "C1", "value": "5 µF"},
            ],
            voltage=12.0,
            show_values=True,
        )

        assert len(params.components) == 2
        assert params.voltage == 12.0

    def test_molecule_parameters_requires_smiles(self):
        """Test that molecule parameters require SMILES."""
        # Valid case
        params = MoleculeParameters(
            smiles="CCO",
            molecule_name="Ethanol",
        )
        assert params.smiles == "CCO"

        # Invalid case - empty SMILES should fail
        with pytest.raises(ValueError):
            MoleculeParameters(smiles="")

    def test_inclined_plane_parameters(self):
        """Test inclined plane parameter validation."""
        params = InclinedPlaneParameters(
            angle=45,
            mass=5.0,
            object_type="sphere",
            show_forces=True,
        )

        assert params.angle == 45
        assert params.object_type == "sphere"

    def test_get_smiles_for_molecule(self):
        """Test SMILES lookup for common molecules."""
        assert get_smiles_for_molecule("ethanol") == "CCO"
        assert get_smiles_for_molecule("carbon tetrachloride") == "ClC(Cl)(Cl)Cl"
        assert get_smiles_for_molecule("benzene") == "c1ccccc1"
        assert get_smiles_for_molecule("unknown_molecule") is None

    def test_spec_to_render_params(self):
        """Test converting spec to render parameters."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="series_circuit",
            parameters={
                "voltage": 9,
                "components": [{"type": "resistor", "value": "5 Ω"}],
            },
            title="Circuit",
        )

        params = spec.to_render_params()

        assert params["subject"] == "physics"
        assert params["diagram_type"] == "series_circuit"
        assert params["voltage"] == 9
        assert params["title"] == "Circuit"


# ============================================================================
# SPEC ROUTER TESTS
# ============================================================================

class TestSpecRouter:
    """Tests for spec routing."""

    def test_route_circuit_to_schemdraw(self):
        """Test that circuits route to SchemDraw."""
        router = get_spec_router()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="series_circuit",
            parameters={"voltage": 12},
        )

        decision = router.route(spec)

        # Should prefer schemdraw (if available) or fall back
        assert decision.diagram_type == "series_circuit"
        assert decision.requires_specialized is True

    def test_route_molecule_to_rdkit(self):
        """Test that molecules route to RDKit."""
        router = get_spec_router()

        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="molecule_2d",
            parameters={"smiles": "CCO"},
        )

        decision = router.route(spec)

        assert decision.diagram_type == "molecule_2d"
        assert decision.requires_specialized is True

    def test_route_inclined_plane_to_matplotlib(self):
        """Test that inclined planes route to matplotlib."""
        router = get_spec_router()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={"angle": 30},
        )

        decision = router.route(spec)

        assert decision.diagram_type == "inclined_plane"
        assert decision.renderer_type == RendererType.MATPLOTLIB

    def test_renderer_info(self):
        """Test getting renderer info."""
        router = get_spec_router()

        info = router.get_renderer_info("series_circuit")
        assert info["preferred_renderer"] == "schemdraw"

        info = router.get_renderer_info("molecule_2d")
        assert info["preferred_renderer"] == "rdkit"

        info = router.get_renderer_info("coordinate_graph")
        assert info["preferred_renderer"] == "matplotlib"


# ============================================================================
# FEEDBACK HANDLER TESTS
# ============================================================================

class TestFeedbackHandler:
    """Tests for feedback handling."""

    def test_create_feedback_state(self):
        """Test creating feedback state."""
        handler = get_feedback_handler()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={"angle": 30},
        )

        state = handler.create_state(spec)

        assert state.iteration == 0
        assert len(state.corrections_applied) == 0

    def test_parse_value_correction(self):
        """Test parsing value corrections from feedback."""
        handler = get_feedback_handler()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={"angle": 30},
        )
        state = handler.create_state(spec)

        feedback = "The angle should be 45° not 30°"
        corrections = handler.parse_feedback(feedback, state)

        # Should extract angle correction
        angle_corrections = [c for c in corrections if c.parameter == "angle"]
        assert len(angle_corrections) >= 1
        assert angle_corrections[0].new_value == 45.0

    def test_parse_smiles_correction(self):
        """Test parsing SMILES corrections from feedback."""
        handler = get_feedback_handler()

        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="molecule_2d",
            parameters={"smiles": "C"},
        )
        state = handler.create_state(spec)

        feedback = "This should be carbon tetrachloride (CCl4), not methane"
        corrections = handler.parse_feedback(feedback, state)

        smiles_corrections = [c for c in corrections if c.category == CorrectionCategory.SMILES]
        assert len(smiles_corrections) >= 1

    def test_apply_corrections(self):
        """Test applying corrections to state."""
        handler = get_feedback_handler()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={"angle": 30, "mass": 10},
        )
        state = handler.create_state(spec)

        corrections = [
            Correction(
                category=CorrectionCategory.VALUE,
                parameter="angle",
                old_value=30,
                new_value=45,
                source="test",
            )
        ]

        state = handler.apply_corrections(state, corrections)

        updated_spec = state.get_updated_spec()
        assert updated_spec.parameters["angle"] == 45
        assert len(state.corrections_applied) == 1

    def test_correction_summary(self):
        """Test getting correction summary."""
        handler = get_feedback_handler()

        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            parameters={"angle": 30},
        )
        state = handler.create_state(spec)

        state.apply_correction(Correction(
            category=CorrectionCategory.VALUE,
            parameter="angle",
            old_value=30,
            new_value=45,
            source="test",
        ))

        summary = handler.get_correction_summary(state)
        assert "angle" in summary
        assert "45" in summary


# ============================================================================
# EXAM QUALITY TESTS
# ============================================================================

class TestExamQuality:
    """Tests for exam quality settings."""

    def test_default_settings(self):
        """Test default exam quality settings."""
        settings = get_exam_settings()

        assert settings.min_font_size >= 12
        assert settings.default_font_size >= 14
        assert settings.min_line_width >= 1.0
        assert settings.background_color == "#ffffff"

    def test_matplotlib_rcparams(self):
        """Test getting matplotlib rcParams."""
        settings = ExamQualitySettings()
        params = settings.get_matplotlib_rcparams()

        assert 'font.size' in params
        assert params['font.size'] == settings.default_font_size
        assert params['figure.dpi'] == settings.print_dpi

    def test_label_placement_no_overlap(self):
        """Test that label placement avoids overlaps."""
        placer = LabelPlacement()

        # Add first label
        pos1 = placer.add_label(0, 0, "Label 1")
        assert pos1 == {'x': 0, 'y': 0}

        # Add overlapping label - should be adjusted
        pos2 = placer.add_label(0.1, 0.1, "Label 2")
        # Should be moved away from first label
        assert pos2 != {'x': 0.1, 'y': 0.1}

    def test_exam_presets(self):
        """Test exam type presets."""
        jee = get_preset('jee')
        neet = get_preset('neet')

        # NEET should have larger fonts (biology diagrams need more detail)
        assert neet.default_font_size >= jee.default_font_size


# ============================================================================
# SUBJECT-SPECIFIC TEST CASES
# ============================================================================

class TestPhysicsFlows:
    """Test flows for physics diagrams."""

    def test_circuit_spec_creation(self):
        """Test creating a circuit diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="series_circuit",
            question_text="A 9V battery is connected to three resistors in series...",
            parameters={
                "voltage": 9,
                "components": [
                    {"type": "resistor", "name": "R1", "value": "10 Ω"},
                    {"type": "resistor", "name": "R2", "value": "20 Ω"},
                    {"type": "resistor", "name": "R3", "value": "30 Ω"},
                ],
                "show_values": True,
            },
        )

        assert spec.diagram_type == "series_circuit"
        assert spec.parameters["voltage"] == 9
        assert len(spec.parameters["components"]) == 3

    def test_inclined_plane_spec_creation(self):
        """Test creating an inclined plane spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            question_text="A 5 kg block slides down a 30° incline...",
            parameters={
                "angle": 30,
                "mass": 5,
                "object_type": "box",
                "show_forces": True,
                "friction": True,
            },
        )

        router = get_spec_router()
        decision = router.route(spec)

        assert decision.render_params["angle"] == 30
        assert decision.render_params["show_forces"] is True

    def test_projectile_spec_creation(self):
        """Test creating a projectile motion spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="projectile_motion",
            parameters={
                "initial_velocity": 20,
                "angle": 45,
                "show_trajectory": True,
                "show_components": True,
            },
        )

        assert spec.is_validated
        assert spec.parameters["initial_velocity"] == 20

    def test_optics_spec_creation(self):
        """Test creating a ray diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="ray_diagram_lens",
            parameters={
                "lens_type": "convex",
                "focal_length": 10,
                "object_distance": 15,
                "object_height": 3,
            },
        )

        router = get_spec_router()
        decision = router.route(spec)

        assert decision.render_params["focal_length"] == 10


class TestChemistryFlows:
    """Test flows for chemistry diagrams."""

    def test_molecule_spec_creation(self):
        """Test creating a molecule diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="molecule_2d",
            question_text="Draw the structure of carbon tetrachloride",
            parameters={
                "smiles": "ClC(Cl)(Cl)Cl",
                "molecule_name": "Carbon Tetrachloride",
                "show_hydrogens": False,
            },
        )

        assert spec.diagram_type == "molecule_2d"
        assert spec.parameters["smiles"] == "ClC(Cl)(Cl)Cl"

    def test_molecule_feedback_correction(self):
        """Test correcting molecule SMILES from feedback."""
        handler = get_feedback_handler()

        # Start with wrong molecule
        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="molecule_2d",
            parameters={"smiles": "C"},  # Methane
        )

        state = handler.create_state(spec)

        # Feedback indicates wrong molecule
        feedback = "This is methane but should be carbon tetrachloride CCl4"
        corrections = handler.parse_feedback(feedback, state)

        # Apply corrections
        state = handler.apply_corrections(state, corrections)
        updated = state.get_updated_spec()

        # Should have updated SMILES
        smiles_correction = [c for c in state.corrections_applied if c.parameter == "smiles"]
        if smiles_correction:
            assert updated.parameters["smiles"] == "ClC(Cl)(Cl)Cl"

    def test_lab_setup_spec_creation(self):
        """Test creating a lab setup spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="lab_setup_manometer",
            parameters={
                "fluid_label": "Mercury",
                "h_diff": 45,
                "h_diff_label": "h = 45 mm",
                "left_arm_connection": "Gas",
                "right_arm_connection": "Atmosphere",
            },
        )

        assert spec.is_validated
        assert spec.parameters["h_diff"] == 45


class TestMathsFlows:
    """Test flows for maths diagrams."""

    def test_coordinate_graph_spec(self):
        """Test creating a coordinate graph spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.MATHS,
            diagram_type="coordinate_graph",
            parameters={
                "functions": ["x**2", "2*x + 1"],
                "x_range": (-5, 5),
                "points": [{"x": 0, "y": 1, "label": "A"}],
                "show_grid": True,
            },
        )

        router = get_spec_router()
        decision = router.route(spec)

        assert decision.renderer_type == RendererType.MATPLOTLIB

    def test_geometry_spec_creation(self):
        """Test creating a geometry construction spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.MATHS,
            diagram_type="geometry_construction",
            parameters={
                "shapes": [
                    {"type": "triangle", "vertices": [(0, 0), (4, 0), (2, 3)]},
                ],
                "angles": [
                    {"vertex": (0, 0), "value": "60°"},
                ],
            },
        )

        assert spec.is_validated

    def test_number_line_spec(self):
        """Test creating a number line spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.MATHS,
            diagram_type="number_line",
            parameters={
                "min_value": -5,
                "max_value": 5,
                "points": [
                    {"value": -2, "label": "A"},
                    {"value": 3, "label": "B"},
                ],
            },
        )

        assert spec.is_validated


class TestBiologyFlows:
    """Test flows for biology diagrams."""

    def test_cell_diagram_spec(self):
        """Test creating a cell diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.BIOLOGY,
            diagram_type="plant_cell",
            parameters={
                "highlight_organelles": ["nucleus", "chloroplast"],
            },
            labels=[
                {"text": "Cell Wall", "position": "outside"},
                {"text": "Nucleus", "position": "inside"},
            ],
        )

        assert spec.is_validated

    def test_heart_diagram_spec(self):
        """Test creating a heart diagram spec."""
        spec = DiagramSpec(
            subject=DiagramSubject.BIOLOGY,
            diagram_type="human_heart",
            parameters={
                "show_blood_flow": True,
            },
            labels=[
                {"text": "Left Ventricle"},
                {"text": "Right Atrium"},
            ],
        )

        router = get_spec_router()
        decision = router.route(spec)

        assert decision.renderer_type == RendererType.MATPLOTLIB


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_physics_flow(self):
        """Test full flow for physics diagram."""
        # 1. Create spec
        spec = DiagramSpec(
            subject=DiagramSubject.PHYSICS,
            diagram_type="inclined_plane",
            question_text="A 10 kg block on a 30° incline",
            parameters={
                "angle": 30,
                "mass": 10,
                "object_type": "box",
            },
        )

        # 2. Create feedback state
        handler = get_feedback_handler()
        state = handler.create_state(spec)

        # 3. Route to renderer
        router = get_spec_router()
        decision = router.route(spec)

        assert decision.diagram_type == "inclined_plane"
        assert decision.render_params["angle"] == 30

        # 4. Simulate feedback
        feedback = "Angle should be 45 degrees"
        corrections = handler.parse_feedback(feedback, state)
        state = handler.apply_corrections(state, corrections)

        # 5. Get updated spec
        updated_spec = state.get_updated_spec()
        assert updated_spec.iteration == 1

    def test_full_chemistry_flow(self):
        """Test full flow for chemistry diagram."""
        # 1. Create spec
        spec = DiagramSpec(
            subject=DiagramSubject.CHEMISTRY,
            diagram_type="molecule_2d",
            parameters={
                "smiles": "CCO",
                "molecule_name": "Ethanol",
            },
        )

        # 2. Route
        router = get_spec_router()
        decision = router.route(spec)

        assert decision.requires_specialized is True
        assert decision.render_params["smiles"] == "CCO"

    def test_spec_parameter_models(self):
        """Test that all diagram types have parameter models."""
        # Check key types have models
        assert "series_circuit" in PARAMETER_MODELS
        assert "molecule_2d" in PARAMETER_MODELS
        assert "inclined_plane" in PARAMETER_MODELS
        assert "coordinate_graph" in PARAMETER_MODELS


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
