"""
Tests for Diagram Renderers

Tests for MathRenderer, PhysicsRenderer, ChemistryRenderer, and BiologyRenderer.
"""

import pytest
import asyncio
from typing import Dict, Any

# Import renderers
from services.diagram_engine.renderers import (
    MathRenderer,
    PhysicsRenderer,
    ChemistryRenderer,
    BiologyRenderer,
)
from services.diagram_engine.base_renderer import RenderResult, RenderError
from services.diagram_engine.specs.base_spec import DiagramSubject, OutputFormat


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def math_renderer():
    """Create MathRenderer instance."""
    return MathRenderer()


@pytest.fixture
def physics_renderer():
    """Create PhysicsRenderer instance."""
    return PhysicsRenderer()


@pytest.fixture
def chemistry_renderer():
    """Create ChemistryRenderer instance."""
    return ChemistryRenderer()


@pytest.fixture
def biology_renderer():
    """Create BiologyRenderer instance."""
    return BiologyRenderer()


def base_spec(diagram_type: str, subject: str = "maths", **kwargs) -> Dict[str, Any]:
    """Create a base diagram specification."""
    spec = {
        "subject": subject,
        "diagram_type": diagram_type,
        "output_format": "png",
        "quality": "high",
        "dimensions": {"width": 800, "height": 600},
        "style": {
            "background_color": "#ffffff",
            "line_color": "#000000",
        },
    }
    spec.update(kwargs)
    return spec


# ============================================================================
# MathRenderer Tests
# ============================================================================

class TestMathRenderer:
    """Tests for MathRenderer."""
    
    def test_subject_property(self, math_renderer):
        """Test renderer subject is correct."""
        assert math_renderer.subject == DiagramSubject.MATHS
    
    def test_supported_types(self, math_renderer):
        """Test supported diagram types."""
        types = math_renderer.get_supported_types()
        expected = [
            "coordinate_graph", "geometry_construction", "number_line",
            "venn_diagram", "bar_chart", "pie_chart", "trigonometric_circle", "3d_plot"
        ]
        for t in expected:
            assert t in types
    
    @pytest.mark.asyncio
    async def test_render_coordinate_graph(self, math_renderer):
        """Test rendering a coordinate graph."""
        spec = base_spec(
            "coordinate_graph",
            functions=["x**2", "sin(x)"],
            x_range=[-5, 5],
            y_range=[-10, 10],
            show_grid=True,
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert result.format == OutputFormat.PNG
        assert len(result.image_data) > 0
        assert result.width == 800
        assert result.height == 600
    
    @pytest.mark.asyncio
    async def test_render_geometry_construction(self, math_renderer):
        """Test rendering geometry shapes."""
        spec = base_spec(
            "geometry_construction",
            shapes=[
                {"type": "circle", "center": [0, 0], "radius": 3, "color": "#FF0000"},
                {"type": "triangle", "vertices": [[0, 0], [4, 0], [2, 3]], "color": "#0000FF"},
            ],
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_number_line(self, math_renderer):
        """Test rendering a number line."""
        spec = base_spec(
            "number_line",
            range=[-10, 10],
            points=[{"value": 3, "label": "A"}, {"value": -5, "label": "B"}],
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_bar_chart(self, math_renderer):
        """Test rendering a bar chart."""
        spec = base_spec(
            "bar_chart",
            data=[10, 25, 15, 30, 20],
            labels=["A", "B", "C", "D", "E"],
            title="Sample Bar Chart",
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_pie_chart(self, math_renderer):
        """Test rendering a pie chart."""
        spec = base_spec(
            "pie_chart",
            data=[30, 25, 20, 15, 10],
            labels=["A", "B", "C", "D", "E"],
            title="Sample Pie Chart",
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_3d_plot(self, math_renderer):
        """Test rendering a 3D plot."""
        spec = base_spec(
            "3d_plot",
            expression="sin(sqrt(x**2 + y**2))",
            x_range=[-5, 5],
            y_range=[-5, 5],
        )
        
        result = await math_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_invalid_diagram_type(self, math_renderer):
        """Test error handling for invalid diagram type."""
        spec = base_spec("invalid_type")
        
        with pytest.raises(RenderError) as excinfo:
            await math_renderer.render(spec)
        
        assert "invalid_type" in str(excinfo.value.message).lower() or "unsupported" in str(excinfo.value.message).lower()


# ============================================================================
# PhysicsRenderer Tests
# ============================================================================

class TestPhysicsRenderer:
    """Tests for PhysicsRenderer."""
    
    def test_subject_property(self, physics_renderer):
        """Test renderer subject is correct."""
        assert physics_renderer.subject == DiagramSubject.PHYSICS
    
    def test_supported_types(self, physics_renderer):
        """Test supported diagram types."""
        types = physics_renderer.get_supported_types()
        expected = [
            "series_circuit", "parallel_circuit", "mixed_circuit",
            "ray_diagram_lens", "ray_diagram_mirror", "free_body_diagram",
            "inclined_plane", "projectile_motion", "wave_diagram",
            "electric_field", "magnetic_field"
        ]
        for t in expected:
            assert t in types
    
    @pytest.mark.asyncio
    async def test_render_series_circuit(self, physics_renderer):
        """Test rendering a series circuit."""
        spec = base_spec(
            "series_circuit",
            subject="physics",
            components=[
                {"type": "battery", "voltage": 12},
                {"type": "resistor", "resistance": 100},
                {"type": "resistor", "resistance": 200},
            ],
        )
        
        result = await physics_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_ray_diagram_lens(self, physics_renderer):
        """Test rendering a lens ray diagram."""
        spec = base_spec(
            "ray_diagram_lens",
            subject="physics",
            lens_type="convex",
            focal_length=5,
            object_distance=10,
            object_height=2,
        )
        
        result = await physics_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_free_body_diagram(self, physics_renderer):
        """Test rendering a free body diagram."""
        spec = base_spec(
            "free_body_diagram",
            subject="physics",
            forces=[
                {"name": "Weight", "magnitude": 100, "angle": -90, "color": "#FF0000"},
                {"name": "Normal", "magnitude": 100, "angle": 90, "color": "#0000FF"},
                {"name": "Friction", "magnitude": 30, "angle": 180, "color": "#00FF00"},
            ],
        )
        
        result = await physics_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_wave_diagram(self, physics_renderer):
        """Test rendering a wave diagram."""
        spec = base_spec(
            "wave_diagram",
            subject="physics",
            wavelength=2,
            amplitude=1,
            num_cycles=3,
        )
        
        result = await physics_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_projectile_motion(self, physics_renderer):
        """Test rendering projectile motion."""
        spec = base_spec(
            "projectile_motion",
            subject="physics",
            initial_velocity=20,
            angle=45,
        )
        
        result = await physics_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0


# ============================================================================
# ChemistryRenderer Tests
# ============================================================================

class TestChemistryRenderer:
    """Tests for ChemistryRenderer."""
    
    def test_subject_property(self, chemistry_renderer):
        """Test renderer subject is correct."""
        assert chemistry_renderer.subject == DiagramSubject.CHEMISTRY
    
    def test_supported_types(self, chemistry_renderer):
        """Test supported diagram types."""
        types = chemistry_renderer.get_supported_types()
        expected = [
            "molecule_2d", "molecule_3d", "reaction_scheme",
            "lab_setup_titration", "lab_setup_distillation", "lab_setup_electrolysis",
            "periodic_table_section", "orbital_diagram", "crystal_structure"
        ]
        for t in expected:
            assert t in types
    
    @pytest.mark.asyncio
    async def test_render_molecule_2d(self, chemistry_renderer):
        """Test rendering a 2D molecule."""
        spec = base_spec(
            "molecule_2d",
            subject="chemistry",
            smiles="CCO",  # Ethanol
            molecule_name="Ethanol",
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_reaction_scheme(self, chemistry_renderer):
        """Test rendering a reaction scheme."""
        spec = base_spec(
            "reaction_scheme",
            subject="chemistry",
            reactants=["2H₂", "O₂"],
            products=["2H₂O"],
            conditions=["Heat", "Catalyst"],
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_lab_titration(self, chemistry_renderer):
        """Test rendering titration apparatus."""
        spec = base_spec(
            "lab_setup_titration",
            subject="chemistry",
            burette_label="NaOH (aq)",
            flask_label="HCl (aq)",
            indicator="Phenolphthalein",
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_periodic_table(self, chemistry_renderer):
        """Test rendering periodic table section."""
        spec = base_spec(
            "periodic_table_section",
            subject="chemistry",
            elements=[1, 6, 7, 8],  # H, C, N, O
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_orbital_diagram(self, chemistry_renderer):
        """Test rendering orbital diagram."""
        spec = base_spec(
            "orbital_diagram",
            subject="chemistry",
            element="C",  # Carbon
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_crystal_structure(self, chemistry_renderer):
        """Test rendering crystal structure."""
        spec = base_spec(
            "crystal_structure",
            subject="chemistry",
            structure_type="fcc",
            element="Cu",
        )
        
        result = await chemistry_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0


# ============================================================================
# BiologyRenderer Tests
# ============================================================================

class TestBiologyRenderer:
    """Tests for BiologyRenderer."""
    
    def test_subject_property(self, biology_renderer):
        """Test renderer subject is correct."""
        assert biology_renderer.subject == DiagramSubject.BIOLOGY
    
    def test_supported_types(self, biology_renderer):
        """Test supported diagram types."""
        types = biology_renderer.get_supported_types()
        expected = [
            "human_heart", "human_brain", "nephron", "neuron",
            "plant_cell", "animal_cell", "dna_replication",
            "mitosis_stages", "meiosis_stages", "digestive_system",
            "respiratory_system", "eye_structure", "ear_structure", "flower_structure"
        ]
        for t in expected:
            assert t in types
    
    @pytest.mark.asyncio
    async def test_render_plant_cell(self, biology_renderer):
        """Test rendering a plant cell."""
        spec = base_spec(
            "plant_cell",
            subject="biology",
            show_labels=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_animal_cell(self, biology_renderer):
        """Test rendering an animal cell."""
        spec = base_spec(
            "animal_cell",
            subject="biology",
            show_labels=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_neuron(self, biology_renderer):
        """Test rendering a neuron."""
        spec = base_spec(
            "neuron",
            subject="biology",
            show_labels=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_heart(self, biology_renderer):
        """Test rendering human heart."""
        spec = base_spec(
            "human_heart",
            subject="biology",
            show_labels=True,
            show_blood_flow=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_brain(self, biology_renderer):
        """Test rendering human brain."""
        spec = base_spec(
            "human_brain",
            subject="biology",
            show_labels=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_dna_replication(self, biology_renderer):
        """Test rendering DNA replication."""
        spec = base_spec(
            "dna_replication",
            subject="biology",
            show_labels=True,
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_mitosis(self, biology_renderer):
        """Test rendering mitosis stages."""
        spec = base_spec(
            "mitosis_stages",
            subject="biology",
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_eye(self, biology_renderer):
        """Test rendering eye structure."""
        spec = base_spec(
            "eye_structure",
            subject="biology",
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0
    
    @pytest.mark.asyncio
    async def test_render_flower(self, biology_renderer):
        """Test rendering flower structure."""
        spec = base_spec(
            "flower_structure",
            subject="biology",
        )
        
        result = await biology_renderer.render(spec)
        
        assert isinstance(result, RenderResult)
        assert len(result.image_data) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestRendererIntegration:
    """Integration tests for all renderers."""
    
    @pytest.mark.asyncio
    async def test_all_renderers_svg_output(
        self,
        math_renderer,
        physics_renderer,
        chemistry_renderer,
        biology_renderer
    ):
        """Test all renderers can output SVG format."""
        renderers_and_specs = [
            (math_renderer, base_spec("bar_chart", data=[1, 2, 3], labels=["A", "B", "C"], output_format="svg")),
            (physics_renderer, base_spec("wave_diagram", subject="physics", output_format="svg")),
            (chemistry_renderer, base_spec("reaction_scheme", subject="chemistry", reactants=["A"], products=["B"], output_format="svg")),
            (biology_renderer, base_spec("plant_cell", subject="biology", output_format="svg")),
        ]
        
        for renderer, spec in renderers_and_specs:
            result = await renderer.render(spec)
            assert result.format == OutputFormat.SVG
            assert len(result.image_data) > 0
            # SVG should start with XML declaration or svg tag
            content = result.image_data.decode('utf-8', errors='ignore')
            assert '<?xml' in content or '<svg' in content
    
    @pytest.mark.asyncio
    async def test_all_renderers_custom_dimensions(
        self,
        math_renderer,
        physics_renderer,
        chemistry_renderer,
        biology_renderer
    ):
        """Test all renderers respect custom dimensions."""
        custom_width = 1200
        custom_height = 800
        
        renderers_and_specs = [
            (math_renderer, base_spec("bar_chart", data=[1, 2, 3], labels=["A", "B", "C"])),
            (physics_renderer, base_spec("wave_diagram", subject="physics")),
            (chemistry_renderer, base_spec("reaction_scheme", subject="chemistry", reactants=["A"], products=["B"])),
            (biology_renderer, base_spec("plant_cell", subject="biology")),
        ]
        
        for renderer, spec in renderers_and_specs:
            spec["dimensions"] = {"width": custom_width, "height": custom_height}
            result = await renderer.render(spec)
            assert result.width == custom_width
            assert result.height == custom_height


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
