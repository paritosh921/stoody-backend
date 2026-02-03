"""
Physics Diagram Renderer

Renders physics diagrams using schemdraw (circuits) and matplotlib:
- Circuit diagrams (series, parallel, mixed)
- Ray diagrams (lens, mirror)
- Free body diagrams
- Inclined plane
- Projectile motion
- Wave diagrams
- Electric/Magnetic fields
"""

import logging
import io
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Arc, Wedge, Ellipse
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import schemdraw
    import schemdraw.elements as elm
    HAS_SCHEMDRAW = True
except ImportError:
    HAS_SCHEMDRAW = False

from ..base_renderer import BaseRenderer, RenderResult, RenderError
from ..specs.base_spec import DiagramSubject, OutputFormat

logger = logging.getLogger(__name__)


class PhysicsRenderer(BaseRenderer):
    """
    Renderer for physics diagrams.
    
    Uses schemdraw for circuit diagrams and matplotlib for other physics diagrams.
    
    Supports:
    - series_circuit, parallel_circuit, mixed_circuit
    - ray_diagram_lens, ray_diagram_mirror
    - free_body_diagram
    - inclined_plane
    - projectile_motion
    - wave_diagram
    - electric_field, magnetic_field
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for PhysicsRenderer")
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.linewidth': 1.5,
        })
    
    @property
    def subject(self) -> DiagramSubject:
        return DiagramSubject.PHYSICS
    
    def get_supported_types(self) -> List[str]:
        return [
            "series_circuit",
            "parallel_circuit",
            "mixed_circuit",
            "ray_diagram_lens",
            "ray_diagram_mirror",
            "free_body_diagram",
            "inclined_plane",
            "projectile_motion",
            "wave_diagram",
            "electric_field",
            "magnetic_field",
        ]
    
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """Render a physics diagram based on spec."""
        is_valid, error = self.validate_spec(spec)
        if not is_valid:
            raise RenderError(error, spec.get('diagram_type', 'unknown'))
        
        spec = self._apply_style(spec)
        diagram_type = spec.get('diagram_type')
        
        render_methods = {
            'series_circuit': self._render_circuit,
            'parallel_circuit': self._render_circuit,
            'mixed_circuit': self._render_circuit,
            'ray_diagram_lens': self._render_ray_diagram_lens,
            'ray_diagram_mirror': self._render_ray_diagram_mirror,
            'free_body_diagram': self._render_free_body_diagram,
            'inclined_plane': self._render_inclined_plane,
            'projectile_motion': self._render_projectile_motion,
            'wave_diagram': self._render_wave_diagram,
            'electric_field': self._render_electric_field,
            'magnetic_field': self._render_magnetic_field,
        }
        
        render_method = render_methods.get(diagram_type)
        if not render_method:
            raise RenderError(f"Unsupported diagram type: {diagram_type}", diagram_type)
        
        try:
            return await render_method(spec)
        except Exception as e:
            logger.error(f"Render error for {diagram_type}: {e}", exc_info=True)
            raise RenderError(str(e), diagram_type, {'exception': type(e).__name__})
    
    def _create_figure(self, spec: Dict[str, Any]) -> Tuple[plt.Figure, Any]:
        """Create a matplotlib figure."""
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800) / 100
        height = dimensions.get('height', 600) / 100
        dpi = self._get_dpi(spec)
        
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        
        fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        
        return fig, ax
    
    def _save_figure(self, fig: plt.Figure, spec: Dict[str, Any]) -> RenderResult:
        """Save figure to bytes."""
        output_format = OutputFormat(spec.get('output_format', 'png'))
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800)
        height = dimensions.get('height', 600)
        
        buffer = io.BytesIO()
        
        if output_format == OutputFormat.SVG:
            fig.savefig(buffer, format='svg', bbox_inches='tight')
        elif output_format == OutputFormat.PDF:
            fig.savefig(buffer, format='pdf', bbox_inches='tight')
        else:
            fig.savefig(buffer, format='png', bbox_inches='tight')
        
        plt.close(fig)
        buffer.seek(0)
        image_data = buffer.read()
        
        return RenderResult(
            image_data=image_data,
            format=output_format,
            width=width,
            height=height,
            metadata={'diagram_type': spec.get('diagram_type')}
        )
    
    async def _render_circuit(self, spec: Dict[str, Any]) -> RenderResult:
        """Render circuit diagram using schemdraw or matplotlib fallback."""
        if HAS_SCHEMDRAW:
            return await self._render_circuit_schemdraw(spec)
        else:
            return await self._render_circuit_matplotlib(spec)
    
    async def _render_circuit_schemdraw(self, spec: Dict[str, Any]) -> RenderResult:
        """Render circuit using schemdraw with proper circuit topology."""
        diagram_type = spec.get('diagram_type')
        components = spec.get('components', [])
        voltage = spec.get('voltage', 12)
        show_values = spec.get('show_values', True)
        title = spec.get('title', '')
        
        # Filter out battery from components (we add it separately)
        circuit_components = [c for c in components if c.get('type') != 'battery']
        
        with schemdraw.Drawing() as d:
            d.config(unit=3, fontsize=11)
            
            if diagram_type == 'series_circuit':
                # SERIES CIRCUIT: Battery and all components in a single loop
                #
                #    +--[R1]--[R2]--[R3]--+
                #    |                    |
                #   [V]                   |
                #    |                    |
                #    +--------------------+
                
                # Start at origin, go right with battery
                d += elm.Ground().at((0, 0))
                d += elm.Battery().up().label(f'{voltage}V' if show_values else 'V').reverse()
                
                # Top wire going right
                d += elm.Line().right().length(d.unit * 0.5)
                
                # Add components going right
                for i, comp in enumerate(circuit_components):
                    comp_type = comp.get('type', 'resistor')
                    value = comp.get('value', '')
                    label = comp.get('label', '')
                    
                    # Build label
                    if show_values and value and label:
                        lbl = f'{label}={value}'
                    elif value:
                        lbl = str(value)
                    elif label:
                        lbl = label
                    else:
                        lbl = ''
                    
                    if comp_type == 'resistor':
                        d += elm.Resistor().right().label(lbl)
                    elif comp_type == 'capacitor':
                        d += elm.Capacitor().right().label(lbl)
                    elif comp_type == 'inductor':
                        d += elm.Inductor().right().label(lbl)
                    elif comp_type == 'lamp' or comp_type == 'bulb':
                        d += elm.Lamp().right().label(lbl)
                    elif comp_type == 'switch':
                        d += elm.Switch().right().label(lbl)
                    elif comp_type == 'ammeter':
                        d += elm.MeterA().right().label(lbl or 'A')
                    elif comp_type == 'voltmeter':
                        d += elm.MeterV().right().label(lbl or 'V')
                    elif comp_type == 'led':
                        d += elm.LED().right().label(lbl)
                    elif comp_type == 'diode':
                        d += elm.Diode().right().label(lbl)
                    elif comp_type == 'fuse':
                        d += elm.Fuse().right().label(lbl)
                    else:
                        # Default to resistor
                        d += elm.Resistor().right().label(lbl)
                
                # Complete the loop: down, left, back to start
                d += elm.Line().right().length(d.unit * 0.5)
                d += elm.Line().down().toy(0)
                d += elm.Line().left().tox(0)
            
            elif diagram_type == 'parallel_circuit':
                # PARALLEL CIRCUIT: Components in parallel branches
                #
                #         +---[R1]---+
                #         |         |
                #    +----+---[R2]---+----+
                #    |    |         |    |
                #   [V]   +---[R3]---+    |
                #    |                   |
                #    +-------------------+
                
                n = len(circuit_components) if circuit_components else 1
                total_width = d.unit * (n + 1)
                
                # Ground and battery on left
                d += elm.Ground().at((0, 0))
                d += elm.Battery().up().label(f'{voltage}V' if show_values else 'V').reverse()
                battery_top = d.here
                
                # Top rail
                d += elm.Line().right().length(total_width)
                top_right = d.here
                
                # Draw each parallel branch
                for i, comp in enumerate(circuit_components):
                    comp_type = comp.get('type', 'resistor')
                    value = comp.get('value', '')
                    label = comp.get('label', '')
                    
                    # Build label
                    if show_values and value and label:
                        lbl = f'{label}={value}'
                    elif value:
                        lbl = str(value)
                    elif label:
                        lbl = label
                    else:
                        lbl = ''
                    
                    # Position for this branch
                    branch_x = battery_top[0] + (i + 1) * d.unit
                    
                    # Draw from top rail down through component to bottom rail
                    d.push()
                    d += elm.Line().at((branch_x, battery_top[1])).down().length(d.unit * 0.5)
                    
                    if comp_type == 'resistor':
                        d += elm.Resistor().down().label(lbl)
                    elif comp_type == 'capacitor':
                        d += elm.Capacitor().down().label(lbl)
                    elif comp_type == 'lamp' or comp_type == 'bulb':
                        d += elm.Lamp().down().label(lbl)
                    elif comp_type == 'inductor':
                        d += elm.Inductor().down().label(lbl)
                    elif comp_type == 'led':
                        d += elm.LED().down().label(lbl)
                    else:
                        d += elm.Resistor().down().label(lbl)
                    
                    d += elm.Line().down().toy(0)
                    d.pop()
                
                # Bottom rail
                d += elm.Line().at((0, 0)).right().length(total_width)
                
                # Close circuit on right side
                d += elm.Line().at(top_right).down().toy(0)
            
            # Save to buffer
            buffer = io.BytesIO()
            d.save(buffer, fmt='png', dpi=150)
            buffer.seek(0)
            image_data = buffer.read()
        
        dimensions = spec.get('dimensions', {})
        return RenderResult(
            image_data=image_data,
            format=OutputFormat.PNG,
            width=dimensions.get('width', 800),
            height=dimensions.get('height', 600),
            metadata={'diagram_type': diagram_type, 'renderer': 'schemdraw'}
        )
    
    async def _render_circuit_matplotlib(self, spec: Dict[str, Any]) -> RenderResult:
        """Fallback circuit rendering using matplotlib."""
        fig, ax = self._create_figure(spec)
        
        diagram_type = spec.get('diagram_type')
        components = spec.get('components', [])
        voltage = spec.get('voltage', 12)
        title = spec.get('title', diagram_type.replace('_', ' ').title())
        
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if diagram_type == 'series_circuit':
            # Draw battery
            ax.plot([0, 0], [0, 2], 'k-', linewidth=2)
            ax.plot([-0.2, 0.2], [1.8, 1.8], 'k-', linewidth=4)
            ax.plot([-0.4, 0.4], [2.2, 2.2], 'k-', linewidth=2)
            ax.text(0.5, 2, f'{voltage}V', fontsize=10)
            
            # Draw wire to components
            ax.plot([0, 0], [2.2, 4], 'k-', linewidth=2)
            ax.plot([0, 2], [4, 4], 'k-', linewidth=2)
            
            # Draw resistors in series
            x_pos = 2
            for i, comp in enumerate(components):
                # Zigzag for resistor
                x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
                y = np.array([0, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, 0]) + 4
                ax.plot(x + x_pos, y, 'k-', linewidth=2)
                
                value = comp.get('value', '')
                if value:
                    ax.text(x_pos + 0.8, 4.5, f'{value}Ω', fontsize=9, ha='center')
                
                x_pos += 2
            
            # Complete circuit
            ax.plot([x_pos, x_pos], [4, 0], 'k-', linewidth=2)
            ax.plot([0, x_pos], [0, 0], 'k-', linewidth=2)
        
        elif diagram_type == 'parallel_circuit':
            # Simplified parallel circuit
            ax.plot([0, 0], [0, 5], 'k-', linewidth=2)
            ax.plot([0, 8], [5, 5], 'k-', linewidth=2)
            ax.plot([8, 8], [5, 0], 'k-', linewidth=2)
            ax.plot([0, 8], [0, 0], 'k-', linewidth=2)
            
            # Battery
            ax.plot([-0.2, 0.2], [2.3, 2.3], 'k-', linewidth=4)
            ax.plot([-0.4, 0.4], [2.7, 2.7], 'k-', linewidth=2)
            ax.text(-0.8, 2.5, f'{voltage}V', fontsize=10)
            
            # Parallel branches
            n_branches = min(len(components), 3)
            branch_spacing = 6 / (n_branches + 1)
            
            for i, comp in enumerate(components[:3]):
                x_branch = 1 + (i + 1) * branch_spacing
                ax.plot([x_branch, x_branch], [5, 3.5], 'k-', linewidth=2)
                ax.plot([x_branch, x_branch], [1.5, 0], 'k-', linewidth=2)
                
                # Resistor symbol
                y = np.array([0, 0.2, -0.2, 0.2, -0.2, 0.2, 0]) + 2.5
                x = np.linspace(x_branch - 0.3, x_branch + 0.3, 7)
                ax.plot([x_branch] * len(y), y, 'k-', linewidth=2)
                
                # Draw as box for simplicity
                ax.add_patch(plt.Rectangle((x_branch - 0.3, 1.5), 0.6, 2, 
                            fill=False, edgecolor='black', linewidth=2))
                
                value = comp.get('value', '')
                if value:
                    ax.text(x_branch, 0.8, f'{value}Ω', fontsize=9, ha='center')
        
        ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_ray_diagram_lens(self, spec: Dict[str, Any]) -> RenderResult:
        """Render ray diagram for a lens."""
        fig, ax = self._create_figure(spec)
        
        lens_type = spec.get('lens_type', 'convex')
        focal_length = spec.get('focal_length', 3)
        object_distance = spec.get('object_distance', 6)
        object_height = spec.get('object_height', 2)
        show_labels = spec.get('show_labels', True)
        title = spec.get('title', f'{lens_type.title()} Lens Ray Diagram')
        
        # DEFENSIVE: Ensure numeric values are actually numbers
        try:
            focal_length = float(focal_length) if focal_length is not None else 3.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid focal_length '{focal_length}', defaulting to 3")
            focal_length = 3.0
        
        try:
            object_distance = float(object_distance) if object_distance is not None else 6.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid object_distance '{object_distance}', defaulting to 6")
            object_distance = 6.0
        
        try:
            object_height = float(object_height) if object_height is not None else 2.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid object_height '{object_height}', defaulting to 2")
            object_height = 2.0
        
        # Calculate image position using lens formula
        if lens_type == 'convex':
            f = focal_length
        else:
            f = -focal_length
        
        u = -object_distance
        v = 1 / (1/f - 1/u) if (1/f - 1/u) != 0 else float('inf')
        magnification = -v / u if u != 0 else 0
        image_height = object_height * magnification
        
        ax.set_xlim(-object_distance - 2, max(abs(v) + 2, object_distance + 2))
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        
        # Draw principal axis
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Draw lens
        if lens_type == 'convex':
            # Convex lens shape
            lens_height = 3
            ax.plot([0, 0], [-lens_height, lens_height], 'b-', linewidth=2)
            ax.annotate('', xy=(0.3, lens_height), xytext=(0, lens_height),
                       arrowprops=dict(arrowstyle='->', color='blue'))
            ax.annotate('', xy=(0.3, -lens_height), xytext=(0, -lens_height),
                       arrowprops=dict(arrowstyle='->', color='blue'))
        else:
            # Concave lens shape
            lens_height = 3
            ax.plot([0, 0], [-lens_height, lens_height], 'b-', linewidth=2)
            ax.annotate('', xy=(-0.3, lens_height), xytext=(0, lens_height),
                       arrowprops=dict(arrowstyle='->', color='blue'))
            ax.annotate('', xy=(-0.3, -lens_height), xytext=(0, -lens_height),
                       arrowprops=dict(arrowstyle='->', color='blue'))
        
        # Mark focal points
        ax.plot([f, f], [-0.2, 0.2], 'g-', linewidth=2)
        ax.plot([-f, -f], [-0.2, 0.2], 'g-', linewidth=2)
        if show_labels:
            ax.text(f, -0.5, 'F', fontsize=10, ha='center')
            ax.text(-f, -0.5, "F'", fontsize=10, ha='center')
        
        # Draw object (arrow)
        ax.annotate('', xy=(-object_distance, object_height), xytext=(-object_distance, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        if show_labels:
            ax.text(-object_distance, object_height + 0.3, 'Object', fontsize=9, ha='center')
        
        # Draw rays
        # Ray 1: Parallel to principal axis, through focal point
        ax.plot([-object_distance, 0], [object_height, object_height], 'r--', linewidth=1)
        if lens_type == 'convex' and v > 0:
            ax.plot([0, v], [object_height, 0], 'r--', linewidth=1)
        
        # Ray 2: Through center of lens
        ax.plot([-object_distance, v], [object_height, image_height], 'g--', linewidth=1)
        
        # Draw image
        if abs(v) < 20 and abs(image_height) < 10:
            ax.annotate('', xy=(v, image_height), xytext=(v, 0),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=2))
            if show_labels:
                ax.text(v, image_height + np.sign(image_height) * 0.3, 'Image', fontsize=9, ha='center')
        
        ax.set_title(title)
        ax.set_xlabel('Distance (cm)')
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, spec)
    
    async def _render_ray_diagram_mirror(self, spec: Dict[str, Any]) -> RenderResult:
        """Render ray diagram for a mirror."""
        fig, ax = self._create_figure(spec)
        
        mirror_type = spec.get('mirror_type', 'concave')
        radius = spec.get('radius', 6)
        object_distance = spec.get('object_distance', 4)
        object_height = spec.get('object_height', 1.5)
        show_labels = spec.get('show_labels', True)
        title = spec.get('title', f'{mirror_type.title()} Mirror Ray Diagram')
        
        # DEFENSIVE: Ensure numeric values are actually numbers
        try:
            radius = float(radius) if radius is not None else 6.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid radius '{radius}', defaulting to 6")
            radius = 6.0
        
        try:
            object_distance = float(object_distance) if object_distance is not None else 4.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid object_distance '{object_distance}', defaulting to 4")
            object_distance = 4.0
        
        try:
            object_height = float(object_height) if object_height is not None else 1.5
        except (TypeError, ValueError):
            logger.warning(f"Invalid object_height '{object_height}', defaulting to 1.5")
            object_height = 1.5
        
        f = radius / 2
        
        ax.set_xlim(-object_distance - 2, 4)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        
        # Draw principal axis
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Draw mirror (arc)
        if mirror_type == 'concave':
            theta = np.linspace(-np.pi/4, np.pi/4, 50)
            x_mirror = radius - radius * np.cos(theta)
            y_mirror = radius * np.sin(theta)
            ax.plot(x_mirror, y_mirror, 'b-', linewidth=2)
        else:
            theta = np.linspace(3*np.pi/4, 5*np.pi/4, 50)
            x_mirror = -radius - radius * np.cos(theta)
            y_mirror = radius * np.sin(theta)
            ax.plot(x_mirror, y_mirror, 'b-', linewidth=2)
        
        # Mark focal point and center
        ax.plot([-f], [0], 'go', markersize=6)
        ax.plot([-radius], [0], 'bo', markersize=6)
        if show_labels:
            ax.text(-f, -0.4, 'F', fontsize=10, ha='center')
            ax.text(-radius, -0.4, 'C', fontsize=10, ha='center')
        
        # Draw object
        ax.annotate('', xy=(-object_distance, object_height), xytext=(-object_distance, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        if show_labels:
            ax.text(-object_distance, object_height + 0.3, 'O', fontsize=10, ha='center')
        
        # Calculate image using mirror formula
        u = -object_distance
        v = 1 / (1/f + 1/u) if (1/f + 1/u) != 0 else float('inf')
        m = -v / u
        image_height = object_height * m
        
        # Draw rays
        ax.plot([-object_distance, 0], [object_height, object_height], 'r--', linewidth=1)
        ax.plot([0, v], [object_height, 0], 'r--', linewidth=1)
        
        # Draw image
        if abs(v) < 15 and abs(image_height) < 5:
            ax.annotate('', xy=(v, image_height), xytext=(v, 0),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=2))
            if show_labels:
                ax.text(v, image_height + np.sign(image_height) * 0.3, 'I', fontsize=10, ha='center')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, spec)
    
    async def _render_free_body_diagram(self, spec: Dict[str, Any]) -> RenderResult:
        """Render free body diagram."""
        fig, ax = self._create_figure(spec)
        
        forces = spec.get('forces', [])
        body_shape = spec.get('body_shape', 'box')
        body_label = spec.get('body_label', '')
        title = spec.get('title', 'Free Body Diagram')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw body
        if body_shape == 'box':
            ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=True, 
                        facecolor='lightgray', edgecolor='black', linewidth=2))
        elif body_shape == 'circle':
            ax.add_patch(Circle((0, 0), 0.5, fill=True,
                        facecolor='lightgray', edgecolor='black', linewidth=2))
        
        if body_label:
            ax.text(0, 0, body_label, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw forces
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, force in enumerate(forces):
            direction = force.get('direction', 'up')
            magnitude = force.get('magnitude', 1)
            label = force.get('label', f'F{i+1}')
            color = force.get('color', colors[i % len(colors)])
            
            # DEFENSIVE: Ensure magnitude is a number (LLM sometimes returns strings)
            try:
                magnitude = float(magnitude)
            except (TypeError, ValueError):
                logger.warning(f"Invalid magnitude '{magnitude}' for force {label}, defaulting to 1")
                magnitude = 1.0
            
            # Calculate arrow direction
            scale = min(magnitude / 10, 3)  # Scale arrow length
            
            directions = {
                'up': (0, scale),
                'down': (0, -scale),
                'left': (-scale, 0),
                'right': (scale, 0),
                'up_left': (-scale * 0.7, scale * 0.7),
                'up_right': (scale * 0.7, scale * 0.7),
                'down_left': (-scale * 0.7, -scale * 0.7),
                'down_right': (scale * 0.7, -scale * 0.7),
            }
            
            dx, dy = directions.get(direction, (0, scale))
            
            # Start point on body surface
            start_offsets = {
                'up': (0, 0.5),
                'down': (0, -0.5),
                'left': (-0.5, 0),
                'right': (0.5, 0),
                'up_left': (-0.35, 0.35),
                'up_right': (0.35, 0.35),
                'down_left': (-0.35, -0.35),
                'down_right': (0.35, -0.35),
            }
            sx, sy = start_offsets.get(direction, (0, 0.5))
            
            # Draw arrow
            ax.annotate('', xy=(sx + dx, sy + dy), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # Label
            ax.text(sx + dx * 1.2, sy + dy * 1.2, f'{label}\n({magnitude}N)',
                   ha='center', va='center', fontsize=9, color=color)
        
        ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_inclined_plane(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render inclined plane diagram with DYNAMIC object shapes and labels.
        
        Supported spec parameters:
        - angle: Angle of incline in degrees
        - mass: Mass of object (kg)
        - object_type: 'box', 'cylinder', 'sphere', 'ball', 'disc' (default: 'box')
        - radius: Radius of cylinder/sphere (if applicable)
        - plane_length: Length of the inclined plane
        - show_forces: Whether to show force vectors
        - friction: Whether to show friction force
        - labels: List of custom labels to display
        - title: Diagram title
        """
        logger.info(f"[INCLINED_PLANE] Received spec keys: {list(spec.keys())}")
        logger.info(f"[INCLINED_PLANE] Full spec: {spec}")
        
        fig, ax = self._create_figure(spec)
        
        # Extract parameters with defaults
        angle = spec.get('angle', 30)
        mass = spec.get('mass', 10)
        object_type = spec.get('object_type', spec.get('object_shape', 'box')).lower()
        radius = spec.get('radius')
        plane_length_value = spec.get('plane_length', spec.get('length'))
        show_forces = spec.get('show_forces', True)
        friction = spec.get('friction', True)
        custom_labels = spec.get('labels', [])
        title = spec.get('title', '')
        
        # DEFENSIVE: Ensure numeric values are actually numbers
        try:
            angle = float(angle) if angle is not None else 30.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid angle '{angle}', defaulting to 30")
            angle = 30.0
        
        try:
            mass = float(mass) if mass is not None else 10.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid mass '{mass}', defaulting to 10")
            mass = 10.0
        
        if radius is not None:
            try:
                radius = float(radius)
            except (TypeError, ValueError):
                radius = None
        
        if plane_length_value is not None:
            try:
                plane_length_value = float(plane_length_value)
            except (TypeError, ValueError):
                plane_length_value = None
        
        logger.info(f"[INCLINED_PLANE] Using: angle={angle}, mass={mass}, object_type={object_type}, radius={radius}, plane_length={plane_length_value}")
        
        # Setup figure
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        angle_rad = math.radians(angle)
        display_plane_length = 7  # Visual length for drawing
        
        # Draw inclined plane (triangle)
        plane_height = display_plane_length * math.sin(angle_rad)
        plane_base = display_plane_length * math.cos(angle_rad)
        
        plane_vertices = [
            [0, 0],
            [plane_base, 0],
            [plane_base, plane_height]
        ]
        ax.add_patch(Polygon(plane_vertices, fill=True, facecolor='#c9a78c', 
                            edgecolor='black', linewidth=2, alpha=0.7))
        
        # Draw angle arc
        arc = Arc((0, 0), 1.5, 1.5, angle=0, theta1=0, theta2=angle, color='black', linewidth=1.5)
        ax.add_patch(arc)
        ax.text(1.2, 0.25, f'{angle}°', fontsize=11, fontweight='bold')
        
        # Calculate object position on plane
        obj_dist_along_plane = display_plane_length * 0.45
        obj_x = obj_dist_along_plane * math.cos(angle_rad)
        obj_y = obj_dist_along_plane * math.sin(angle_rad)
        
        # Draw object based on type
        object_size = 0.5  # Base size for objects
        
        if object_type in ['cylinder', 'disc', 'solid cylinder', 'solid disc']:
            # Draw cylinder/disc as a circle (side view shows as ellipse/circle)
            obj_y += object_size + 0.1  # Lift to sit on plane
            circle = Circle((obj_x, obj_y), object_size, fill=True, 
                           facecolor='#87CEEB', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            # Add center point and radius line to show it's a cylinder
            ax.plot(obj_x, obj_y, 'ko', markersize=3)
            # Draw radius line
            ax.plot([obj_x, obj_x + object_size * 0.8], [obj_y, obj_y], 'k-', linewidth=1.5)
            label_text = f'{mass} kg'
            if radius:
                label_text += f'\nr = {radius} m'
            ax.text(obj_x, obj_y - 0.15, label_text, ha='center', va='top', fontsize=9, fontweight='bold')
            
        elif object_type in ['sphere', 'ball', 'solid sphere']:
            # Draw sphere as a circle
            obj_y += object_size + 0.1
            circle = Circle((obj_x, obj_y), object_size, fill=True,
                           facecolor='#FFB6C1', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.plot(obj_x, obj_y, 'ko', markersize=3)  # Center point
            label_text = f'{mass} kg'
            if radius:
                label_text += f'\nr = {radius} m'
            ax.text(obj_x, obj_y - 0.15, label_text, ha='center', va='top', fontsize=9, fontweight='bold')
            
        else:
            # Default: Draw box/block (rotated rectangle)
            obj_y += object_size/2 + 0.15
            box_size = object_size * 1.2
            corners = np.array([
                [-box_size/2, -box_size/2],
                [box_size/2, -box_size/2],
                [box_size/2, box_size/2],
                [-box_size/2, box_size/2]
            ])
            
            # Rotate corners to align with plane
            rotation = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad)],
                [math.sin(angle_rad), math.cos(angle_rad)]
            ])
            rotated = corners @ rotation.T
            rotated[:, 0] += obj_x
            rotated[:, 1] += obj_y
            
            ax.add_patch(Polygon(rotated, fill=True, facecolor='#87CEEB',
                                edgecolor='black', linewidth=2))
            ax.text(obj_x, obj_y, f'{mass} kg', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw forces if requested
        if show_forces:
            force_scale = 1.2
            
            # Weight (mg) - always downward
            ax.annotate('', xy=(obj_x, obj_y - force_scale * 1.3), xytext=(obj_x, obj_y),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
            ax.text(obj_x + 0.25, obj_y - force_scale * 0.8, 'mg', fontsize=11, color='red', fontweight='bold')
            
            # Normal force (perpendicular to plane)
            normal_dx = -math.sin(angle_rad) * force_scale
            normal_dy = math.cos(angle_rad) * force_scale
            ax.annotate('', xy=(obj_x + normal_dx, obj_y + normal_dy), xytext=(obj_x, obj_y),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
            ax.text(obj_x + normal_dx - 0.2, obj_y + normal_dy + 0.15, 'N', fontsize=11, color='blue', fontweight='bold')
            
            # Component along plane (mg sin θ)
            comp_scale = force_scale * 0.7
            comp_dx = -math.cos(angle_rad) * comp_scale
            comp_dy = -math.sin(angle_rad) * comp_scale
            ax.annotate('', xy=(obj_x + comp_dx, obj_y + comp_dy), xytext=(obj_x, obj_y),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(obj_x + comp_dx - 0.4, obj_y + comp_dy - 0.15, 'mg sinθ', fontsize=9, color='green', fontweight='bold')
            
            if friction:
                # Friction force (up the plane, opposing motion)
                fric_scale = force_scale * 0.5
                fric_dx = math.cos(angle_rad) * fric_scale
                fric_dy = math.sin(angle_rad) * fric_scale
                ax.annotate('', xy=(obj_x + fric_dx, obj_y + fric_dy), xytext=(obj_x, obj_y),
                           arrowprops=dict(arrowstyle='->', color='orange', lw=2))
                ax.text(obj_x + fric_dx + 0.15, obj_y + fric_dy + 0.1, 'f', fontsize=11, color='orange', fontweight='bold')
        
        # Add plane length label if provided
        if plane_length_value:
            # Draw length along the incline
            mid_x = plane_base * 0.5 + 0.3
            mid_y = plane_height * 0.5 + 0.5
            ax.text(mid_x, mid_y, f'L = {plane_length_value} m', fontsize=10, fontweight='bold',
                   rotation=angle, ha='center', va='bottom', color='#333333')
        
        # Add any custom labels from spec
        for label_info in custom_labels:
            if isinstance(label_info, dict):
                text = label_info.get('text', '')
                pos = label_info.get('position', 'bottom')
                if text:
                    if pos == 'bottom':
                        ax.text(plane_base/2, -0.5, text, ha='center', fontsize=10)
                    elif pos == 'top':
                        ax.text(plane_base/2, plane_height + 0.5, text, ha='center', fontsize=10)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        return self._save_figure(fig, spec)
    
    async def _render_projectile_motion(self, spec: Dict[str, Any]) -> RenderResult:
        """Render projectile motion diagram."""
        fig, ax = self._create_figure(spec)
        
        initial_velocity = spec.get('initial_velocity', 20)
        angle = spec.get('angle', 45)
        g = spec.get('gravity', 9.8)
        show_components = spec.get('show_components', True)
        title = spec.get('title', 'Projectile Motion')
        
        # DEFENSIVE: Ensure numeric values are actually numbers
        try:
            initial_velocity = float(initial_velocity) if initial_velocity is not None else 20.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid initial_velocity '{initial_velocity}', defaulting to 20")
            initial_velocity = 20.0
        
        try:
            angle = float(angle) if angle is not None else 45.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid angle '{angle}', defaulting to 45")
            angle = 45.0
        
        try:
            g = float(g) if g is not None else 9.8
        except (TypeError, ValueError):
            logger.warning(f"Invalid gravity '{g}', defaulting to 9.8")
            g = 9.8
        
        angle_rad = math.radians(angle)
        v0x = initial_velocity * math.cos(angle_rad)
        v0y = initial_velocity * math.sin(angle_rad)
        
        # Time of flight
        t_flight = 2 * v0y / g
        
        # Trajectory
        t = np.linspace(0, t_flight, 100)
        x = v0x * t
        y = v0y * t - 0.5 * g * t**2
        
        max_x = v0x * t_flight
        max_y = v0y**2 / (2 * g)
        
        ax.set_xlim(-1, max_x + 2)
        ax.set_ylim(-1, max_y + 2)
        
        # Draw trajectory
        ax.plot(x, y, 'b-', linewidth=2, label='Trajectory')
        
        # Draw ground
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], [-1, -1], [0, 0], 
                       color='brown', alpha=0.3)
        
        # Initial velocity vector
        scale = max_x / (3 * initial_velocity)
        ax.annotate('', xy=(v0x * scale, v0y * scale), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(v0x * scale * 0.5, v0y * scale * 0.5 + 0.5, f'v₀ = {initial_velocity} m/s', 
               fontsize=10, color='red')
        
        if show_components:
            # v0x component
            ax.annotate('', xy=(v0x * scale, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            ax.text(v0x * scale * 0.5, -0.5, f'v₀ₓ', fontsize=9, color='green')
            
            # v0y component
            ax.annotate('', xy=(0, v0y * scale), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
            ax.text(-0.5, v0y * scale * 0.5, f'v₀ᵧ', fontsize=9, color='orange')
        
        # Mark maximum height
        ax.plot([max_x/2], [max_y], 'ro', markersize=6)
        ax.annotate(f'H_max = {max_y:.1f}m', xy=(max_x/2, max_y), 
                   xytext=(max_x/2 + 1, max_y + 0.5), fontsize=9)
        
        # Mark range
        ax.annotate(f'R = {max_x:.1f}m', xy=(max_x, 0), 
                   xytext=(max_x - 2, -0.8), fontsize=9)
        
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Vertical Distance (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, spec)
    
    async def _render_wave_diagram(self, spec: Dict[str, Any]) -> RenderResult:
        """Render wave diagram."""
        fig, ax = self._create_figure(spec)
        
        wave_type = spec.get('wave_type', 'transverse')
        amplitude = spec.get('amplitude', 1)
        wavelength = spec.get('wavelength', 2)
        num_waves = spec.get('num_waves', 3)
        show_labels = spec.get('show_labels', True)
        title = spec.get('title', f'{wave_type.title()} Wave')
        
        # DEFENSIVE: Ensure numeric values are actually numbers
        try:
            amplitude = float(amplitude) if amplitude is not None else 1.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid amplitude '{amplitude}', defaulting to 1")
            amplitude = 1.0
        
        try:
            wavelength = float(wavelength) if wavelength is not None else 2.0
        except (TypeError, ValueError):
            logger.warning(f"Invalid wavelength '{wavelength}', defaulting to 2")
            wavelength = 2.0
        
        try:
            num_waves = int(num_waves) if num_waves is not None else 3
        except (TypeError, ValueError):
            logger.warning(f"Invalid num_waves '{num_waves}', defaulting to 3")
            num_waves = 3
        
        x = np.linspace(0, num_waves * wavelength, 500)
        y = amplitude * np.sin(2 * np.pi * x / wavelength)
        
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=1)
        
        if show_labels:
            # Mark wavelength
            ax.annotate('', xy=(wavelength, amplitude + 0.3), xytext=(0, amplitude + 0.3),
                       arrowprops=dict(arrowstyle='<->', color='red'))
            ax.text(wavelength/2, amplitude + 0.5, f'λ = {wavelength}', fontsize=10, ha='center', color='red')
            
            # Mark amplitude
            ax.annotate('', xy=(-0.2, amplitude), xytext=(-0.2, 0),
                       arrowprops=dict(arrowstyle='<->', color='green'))
            ax.text(-0.5, amplitude/2, f'A = {amplitude}', fontsize=10, ha='right', color='green', rotation=90, va='center')
            
            # Mark crest and trough
            ax.plot(wavelength/4, amplitude, 'ro', markersize=6)
            ax.text(wavelength/4, amplitude + 0.2, 'Crest', fontsize=9, ha='center')
            
            ax.plot(3*wavelength/4, -amplitude, 'ro', markersize=6)
            ax.text(3*wavelength/4, -amplitude - 0.2, 'Trough', fontsize=9, ha='center', va='top')
        
        ax.set_xlim(-0.5, num_waves * wavelength + 0.5)
        ax.set_ylim(-amplitude * 1.5, amplitude * 1.5)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Displacement')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, spec)
    
    async def _render_electric_field(self, spec: Dict[str, Any]) -> RenderResult:
        """Render electric field diagram."""
        fig, ax = self._create_figure(spec)
        
        charges = spec.get('charges', [{'x': 0, 'y': 0, 'q': 1}])
        show_field_lines = spec.get('show_field_lines', True)
        num_lines = spec.get('num_lines', 8)
        title = spec.get('title', 'Electric Field')
        
        # DEFENSIVE: Ensure num_lines is an integer
        try:
            num_lines = int(num_lines) if num_lines is not None else 8
        except (TypeError, ValueError):
            logger.warning(f"Invalid num_lines '{num_lines}', defaulting to 8")
            num_lines = 8
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        
        for charge in charges:
            x, y, q = charge.get('x', 0), charge.get('y', 0), charge.get('q', 1)
            
            # DEFENSIVE: Ensure charge position and magnitude are numbers
            try:
                x = float(x) if x is not None else 0.0
            except (TypeError, ValueError):
                x = 0.0
            try:
                y = float(y) if y is not None else 0.0
            except (TypeError, ValueError):
                y = 0.0
            try:
                q = float(q) if q is not None else 1.0
            except (TypeError, ValueError):
                q = 1.0
            color = 'red' if q > 0 else 'blue'
            symbol = '+' if q > 0 else '-'
            
            ax.plot(x, y, 'o', color=color, markersize=20)
            ax.text(x, y, symbol, fontsize=14, ha='center', va='center', color='white', fontweight='bold')
            
            if show_field_lines:
                for i in range(num_lines):
                    angle = 2 * np.pi * i / num_lines
                    
                    if q > 0:
                        # Field lines go outward
                        dx = 2 * np.cos(angle)
                        dy = 2 * np.sin(angle)
                        ax.annotate('', xy=(x + dx, y + dy), xytext=(x + 0.3 * np.cos(angle), y + 0.3 * np.sin(angle)),
                                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
                    else:
                        # Field lines go inward
                        dx = 2 * np.cos(angle)
                        dy = 2 * np.sin(angle)
                        ax.annotate('', xy=(x + 0.3 * np.cos(angle), y + 0.3 * np.sin(angle)), xytext=(x + dx, y + dy),
                                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, spec)
    
    async def _render_magnetic_field(self, spec: Dict[str, Any]) -> RenderResult:
        """Render magnetic field diagram."""
        fig, ax = self._create_figure(spec)
        
        field_type = spec.get('field_type', 'bar_magnet')
        title = spec.get('title', 'Magnetic Field')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        
        if field_type == 'bar_magnet':
            # Draw bar magnet
            ax.add_patch(plt.Rectangle((-2, -0.5), 2, 1, fill=True, facecolor='red', edgecolor='black'))
            ax.add_patch(plt.Rectangle((0, -0.5), 2, 1, fill=True, facecolor='blue', edgecolor='black'))
            ax.text(-1, 0, 'N', fontsize=14, ha='center', va='center', color='white', fontweight='bold')
            ax.text(1, 0, 'S', fontsize=14, ha='center', va='center', color='white', fontweight='bold')
            
            # Draw field lines
            for y_start in [-0.3, 0, 0.3]:
                # Lines from N pole
                t = np.linspace(0, np.pi, 50)
                for scale in [1, 1.5, 2]:
                    x_line = -2 - scale * np.cos(t)
                    y_line = y_start + scale * np.sin(t)
                    ax.plot(x_line, y_line, 'gray', linewidth=1)
                
                # Lines from S pole
                for scale in [1, 1.5, 2]:
                    x_line = 2 + scale * np.cos(t)
                    y_line = y_start + scale * np.sin(t)
                    ax.plot(x_line, y_line, 'gray', linewidth=1)
        
        elif field_type == 'wire':
            # Current-carrying wire (cross-section)
            ax.add_patch(Circle((0, 0), 0.3, fill=True, facecolor='orange', edgecolor='black', linewidth=2))
            ax.text(0, 0, '⊙', fontsize=20, ha='center', va='center')
            
            # Concentric field lines
            for r in [1, 1.5, 2, 2.5, 3]:
                circle = Circle((0, 0), r, fill=False, edgecolor='gray', linestyle='--', linewidth=1)
                ax.add_patch(circle)
                # Arrow to show direction
                ax.annotate('', xy=(0, r + 0.1), xytext=(0.1, r),
                           arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.set_title(title)
        ax.axis('off')
        
        return self._save_figure(fig, spec)
