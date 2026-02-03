"""
Math Diagram Renderer

Renders mathematical diagrams using matplotlib:
- Coordinate graphs (functions, plots)
- Geometry constructions
- Number lines
- Venn diagrams
- Charts (bar, pie)
- Trigonometric circles
- 3D plots
"""

import logging
import io
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Arc, FancyArrowPatch, Wedge
    from matplotlib.lines import Line2D
    from matplotlib.collections import PatchCollection
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib_venn import venn2, venn3
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..base_renderer import BaseRenderer, RenderResult, RenderError
from ..specs.base_spec import DiagramSubject, OutputFormat

logger = logging.getLogger(__name__)


class MathRenderer(BaseRenderer):
    """
    Renderer for mathematical diagrams using matplotlib.
    
    Supports:
    - coordinate_graph: Function plots, parametric curves
    - geometry_construction: Shapes, angles, constructions
    - number_line: Number line with points and intervals
    - venn_diagram: 2-set and 3-set Venn diagrams
    - bar_chart: Vertical/horizontal bar charts
    - pie_chart: Pie charts with labels
    - trigonometric_circle: Unit circle with angles
    - 3d_plot: 3D surface and parametric plots
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for MathRenderer. "
                "Install with: pip install matplotlib matplotlib-venn"
            )
        
        # Configure matplotlib defaults
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
        })
    
    @property
    def subject(self) -> DiagramSubject:
        return DiagramSubject.MATHS
    
    def get_supported_types(self) -> List[str]:
        return [
            "coordinate_graph",
            "geometry_construction",
            "number_line",
            "venn_diagram",
            "bar_chart",
            "pie_chart",
            "trigonometric_circle",
            "3d_plot",
        ]
    
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """Render a math diagram based on spec."""
        # Validate spec
        is_valid, error = self.validate_spec(spec)
        if not is_valid:
            raise RenderError(error, spec.get('diagram_type', 'unknown'))
        
        # Apply style defaults
        spec = self._apply_style(spec)
        
        diagram_type = spec.get('diagram_type')
        
        # Route to specific renderer
        render_methods = {
            'coordinate_graph': self._render_coordinate_graph,
            'geometry_construction': self._render_geometry,
            'number_line': self._render_number_line,
            'venn_diagram': self._render_venn_diagram,
            'bar_chart': self._render_bar_chart,
            'pie_chart': self._render_pie_chart,
            'trigonometric_circle': self._render_trig_circle,
            '3d_plot': self._render_3d_plot,
        }
        
        render_method = render_methods.get(diagram_type)
        if not render_method:
            raise RenderError(
                f"Unsupported diagram type: {diagram_type}",
                diagram_type
            )
        
        try:
            return await render_method(spec)
        except Exception as e:
            logger.error(f"Render error for {diagram_type}: {e}", exc_info=True)
            raise RenderError(str(e), diagram_type, {'exception': type(e).__name__})
    
    def _create_figure(self, spec: Dict[str, Any], is_3d: bool = False) -> Tuple[plt.Figure, Any]:
        """Create a matplotlib figure with proper dimensions."""
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800) / 100  # Convert to inches
        height = dimensions.get('height', 600) / 100
        dpi = self._get_dpi(spec)
        
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        
        fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=bg_color)
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        
        ax.set_facecolor(bg_color)
        
        return fig, ax
    
    def _save_figure(self, fig: plt.Figure, spec: Dict[str, Any]) -> RenderResult:
        """Save figure to bytes and return RenderResult."""
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
    
    async def _render_coordinate_graph(self, spec: Dict[str, Any]) -> RenderResult:
        """Render coordinate graph with functions, points, and vectors."""
        # DEBUG: Log what spec we receive
        logger.info(f"[COORDINATE_GRAPH] Received spec keys: {list(spec.keys())}")
        logger.info(f"[COORDINATE_GRAPH] points: {spec.get('points', [])}")
        logger.info(f"[COORDINATE_GRAPH] lines: {spec.get('lines', [])}")
        
        fig, ax = self._create_figure(spec)
        
        # Get graph parameters
        x_range = spec.get('x_range', [-10, 10])
        y_range = spec.get('y_range', [-10, 10])
        functions = spec.get('functions', [])
        points = spec.get('points', [])
        lines = spec.get('lines', [])  # For vectors and line segments
        vectors = spec.get('vectors', [])  # Alternative way to specify vectors
        show_grid = spec.get('show_grid', True)
        show_axes = spec.get('show_axes', True)
        title = spec.get('title', '')
        x_label = spec.get('x_label', 'x')
        y_label = spec.get('y_label', 'y')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Set up the coordinate system
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        if show_axes:
            ax.axhline(y=0, color=line_color, linewidth=1.0)
            ax.axvline(x=0, color=line_color, linewidth=1.0)
            # Add axis arrows
            ax.annotate('', xy=(x_range[1], 0), xytext=(x_range[1]-0.5, 0),
                       arrowprops=dict(arrowstyle='->', color=line_color, lw=1))
            ax.annotate('', xy=(0, y_range[1]), xytext=(0, y_range[1]-0.5),
                       arrowprops=dict(arrowstyle='->', color=line_color, lw=1))
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot functions
        x = np.linspace(x_range[0], x_range[1], 1000)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, func_spec in enumerate(functions):
            expr = func_spec.get('expression', '')
            label = func_spec.get('label', f'f{i+1}(x)')
            color = func_spec.get('color', colors[i % len(colors)])
            line_style = func_spec.get('line_style', '-')
            
            try:
                y = self._safe_eval_expression(expr, x)
                ax.plot(x, y, color=color, label=label, linestyle=line_style, linewidth=2)
            except Exception as e:
                logger.warning(f"Failed to plot function '{expr}': {e}")
        
        # Draw vectors/arrows (lines with arrowheads)
        vector_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        # Handle 'lines' parameter (vectors as line segments)
        for i, line_spec in enumerate(lines):
            from_point = line_spec.get('from', [0, 0])
            to_point = line_spec.get('to', [1, 1])
            label = line_spec.get('label', '')
            color = line_spec.get('color', vector_colors[i % len(vector_colors)])
            line_style = line_spec.get('style', 'solid')
            show_arrow = line_spec.get('arrow', True)  # Default: show arrow
            
            # Handle both list and dict formats for points
            if isinstance(from_point, dict):
                x1, y1 = from_point.get('x', 0), from_point.get('y', 0)
            else:
                x1, y1 = from_point[0], from_point[1]
            
            if isinstance(to_point, dict):
                x2, y2 = to_point.get('x', 1), to_point.get('y', 1)
            else:
                x2, y2 = to_point[0], to_point[1]
            
            # Draw the vector with arrow
            if show_arrow:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                          connectionstyle='arc3,rad=0'))
            else:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2,
                       linestyle='-' if line_style == 'solid' else '--')
            
            # Add label at midpoint
            if label:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(label, (mid_x, mid_y), textcoords="offset points",
                           xytext=(5, 5), fontsize=11, fontweight='bold', color=color)
        
        # Handle 'vectors' parameter (explicit vector specification)
        for i, vec_spec in enumerate(vectors):
            origin = vec_spec.get('origin', [0, 0])
            components = vec_spec.get('components', [1, 1])
            label = vec_spec.get('label', '')
            color = vec_spec.get('color', vector_colors[i % len(vector_colors)])
            
            x1, y1 = origin[0], origin[1]
            dx, dy = components[0], components[1]
            
            ax.arrow(x1, y1, dx, dy, head_width=0.3, head_length=0.2,
                    fc=color, ec=color, linewidth=2)
            
            if label:
                ax.annotate(label, (x1 + dx/2, y1 + dy/2), textcoords="offset points",
                           xytext=(5, 5), fontsize=11, fontweight='bold', color=color)
        
        # Plot points
        for point in points:
            px, py = point.get('x', 0), point.get('y', 0)
            label = point.get('label', '')
            color = point.get('color', '#e74c3c')
            marker = point.get('marker', 'o')
            size = point.get('size', 10)
            
            ax.plot(px, py, marker=marker, color=color, markersize=size,
                   markeredgecolor='black', markeredgewidth=1)
            if label:
                ax.annotate(label, (px, py), textcoords="offset points",
                           xytext=(8, 8), fontsize=11, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if functions:
            ax.legend(loc='best')
        
        # Make the plot look cleaner
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        return self._save_figure(fig, spec)
    
    def _safe_eval_expression(self, expr: str, x: np.ndarray) -> np.ndarray:
        """Safely evaluate a mathematical expression."""
        # Define allowed functions
        safe_dict = {
            'x': x,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e,
            'pow': np.power,
        }
        
        # Replace common math notation
        expr = expr.replace('^', '**')
        expr = expr.replace('ln', 'log')
        
        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return np.where(np.isfinite(result), result, np.nan)
        except Exception as e:
            raise ValueError(f"Invalid expression '{expr}': {e}")
    
    async def _render_geometry(self, spec: Dict[str, Any]) -> RenderResult:
        """Render geometry constructions."""
        fig, ax = self._create_figure(spec)
        
        shapes = spec.get('shapes', [])
        labels = spec.get('labels', [])
        show_grid = spec.get('show_grid', False)
        title = spec.get('title', '')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        line_width = style.get('line_width', 1.5)
        
        ax.set_aspect('equal')
        ax.grid(show_grid, alpha=0.3)
        
        for shape in shapes:
            shape_type = shape.get('type', '')
            color = shape.get('color', line_color)
            fill = shape.get('fill', False)
            fill_color = shape.get('fill_color', color)
            fill_alpha = shape.get('fill_alpha', 0.3)
            
            if shape_type == 'point':
                x, y = shape.get('x', 0), shape.get('y', 0)
                ax.plot(x, y, 'o', color=color, markersize=6)
            
            elif shape_type == 'line':
                points = shape.get('points', [[0, 0], [1, 1]])
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                ax.plot(x_coords, y_coords, color=color, linewidth=line_width)
            
            elif shape_type == 'circle':
                center = shape.get('center', [0, 0])
                radius = shape.get('radius', 1)
                circle = Circle(center, radius, fill=fill, 
                               facecolor=fill_color if fill else 'none',
                               edgecolor=color, linewidth=line_width,
                               alpha=fill_alpha if fill else 1)
                ax.add_patch(circle)
            
            elif shape_type == 'triangle':
                vertices = shape.get('vertices', [[0, 0], [1, 0], [0.5, 1]])
                polygon = Polygon(vertices, fill=fill,
                                 facecolor=fill_color if fill else 'none',
                                 edgecolor=color, linewidth=line_width,
                                 alpha=fill_alpha if fill else 1)
                ax.add_patch(polygon)
            
            elif shape_type == 'rectangle':
                vertices = shape.get('vertices', [[0, 0], [2, 0], [2, 1], [0, 1]])
                polygon = Polygon(vertices, fill=fill,
                                 facecolor=fill_color if fill else 'none',
                                 edgecolor=color, linewidth=line_width,
                                 alpha=fill_alpha if fill else 1)
                ax.add_patch(polygon)
            
            elif shape_type == 'arc':
                center = shape.get('center', [0, 0])
                radius = shape.get('radius', 1)
                theta1 = shape.get('theta1', 0)
                theta2 = shape.get('theta2', 90)
                arc = Arc(center, 2*radius, 2*radius, angle=0,
                         theta1=theta1, theta2=theta2,
                         color=color, linewidth=line_width)
                ax.add_patch(arc)
            
            elif shape_type == 'angle':
                vertex = shape.get('vertex', [0, 0])
                radius = shape.get('radius', 0.5)
                angle1 = shape.get('angle1', 0)
                angle2 = shape.get('angle2', 90)
                wedge = Wedge(vertex, radius, angle1, angle2,
                             facecolor=fill_color, edgecolor=color,
                             alpha=0.3, linewidth=line_width)
                ax.add_patch(wedge)
        
        # Add labels
        for label in labels:
            x, y = label.get('x', 0), label.get('y', 0)
            text = label.get('text', '')
            fontsize = label.get('fontsize', 12)
            ax.annotate(text, (x, y), fontsize=fontsize, ha='center', va='center')
        
        if title:
            ax.set_title(title)
        
        ax.autoscale_view()
        ax.set_axis_off()
        
        return self._save_figure(fig, spec)
    
    async def _render_number_line(self, spec: Dict[str, Any]) -> RenderResult:
        """Render number line with points and intervals."""
        fig, ax = self._create_figure(spec)
        
        # Robust handling of x_range - can be list, dict, or other formats
        x_range_raw = spec.get('range', [-10, 10])
        if isinstance(x_range_raw, list) and len(x_range_raw) >= 2:
            x_range = [float(x_range_raw[0]), float(x_range_raw[1])]
        elif isinstance(x_range_raw, dict):
            # Handle dict formats like {0: -10, 1: 10} or {"start": -10, "end": 10}
            if 0 in x_range_raw and 1 in x_range_raw:
                x_range = [float(x_range_raw[0]), float(x_range_raw[1])]
            elif '0' in x_range_raw and '1' in x_range_raw:
                x_range = [float(x_range_raw['0']), float(x_range_raw['1'])]
            elif 'start' in x_range_raw and 'end' in x_range_raw:
                x_range = [float(x_range_raw['start']), float(x_range_raw['end'])]
            elif 'min' in x_range_raw and 'max' in x_range_raw:
                x_range = [float(x_range_raw['min']), float(x_range_raw['max'])]
            else:
                logger.warning(f"Invalid x_range dict format: {x_range_raw}, using default")
                x_range = [-10, 10]
        elif isinstance(x_range_raw, (int, float)):
            # Single value - create symmetric range
            x_range = [-abs(float(x_range_raw)), abs(float(x_range_raw))]
        else:
            logger.warning(f"Invalid x_range format: {type(x_range_raw)}, using default")
            x_range = [-10, 10]
        
        # Ensure x_range[0] < x_range[1]
        if x_range[0] >= x_range[1]:
            x_range = [x_range[1], x_range[0]] if x_range[0] > x_range[1] else [-10, 10]
        
        points = spec.get('points', [])
        intervals = spec.get('intervals', [])
        tick_interval = spec.get('tick_interval', 1)
        title = spec.get('title', '')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Draw main line with arrows on both ends
        ax.axhline(y=0, color=line_color, linewidth=2)
        # Arrow at right end
        ax.annotate('', xy=(x_range[1] + 0.3, 0), xytext=(x_range[1], 0),
                   arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
        # Arrow at left end (for infinity)
        ax.annotate('', xy=(x_range[0] - 0.3, 0), xytext=(x_range[0], 0),
                   arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
        
        # Draw ticks
        ticks = np.arange(x_range[0], x_range[1] + tick_interval, tick_interval)
        for tick in ticks:
            ax.plot([tick, tick], [-0.1, 0.1], color=line_color, linewidth=1)
            ax.annotate(str(int(tick)), (tick, -0.3), ha='center', va='top', fontsize=10)
        
        # Draw intervals (FIXED: proper open/closed circles)
        for interval in intervals:
            start = interval.get('start', x_range[0])
            end = interval.get('end', x_range[1])
            color = interval.get('color', '#1f77b4')
            include_start = interval.get('include_start', True)
            include_end = interval.get('include_end', True)
            
            # Clamp to visible range but allow extending to edges
            draw_start = max(start, x_range[0] - 0.2)
            draw_end = min(end, x_range[1] + 0.2)
            
            # Draw the interval line (thick, above the number line)
            ax.plot([draw_start, draw_end], [0.15, 0.15], color=color, linewidth=4, solid_capstyle='butt')
            
            # Draw arrow if interval extends to infinity
            if start <= x_range[0]:
                # Extends to -infinity, draw arrow pointing left
                ax.annotate('', xy=(x_range[0] - 0.3, 0.15), xytext=(x_range[0], 0.15),
                           arrowprops=dict(arrowstyle='->', color=color, lw=3))
            else:
                # Finite start - draw open or closed circle
                # FIXED: fillstyle determines if circle is filled or hollow
                if include_start:
                    ax.plot(start, 0.15, 'o', color=color, markersize=10, markeredgewidth=2)
                else:
                    ax.plot(start, 0.15, 'o', markerfacecolor='white', markeredgecolor=color, 
                           markersize=10, markeredgewidth=2)
            
            if end >= x_range[1]:
                # Extends to +infinity, draw arrow pointing right
                ax.annotate('', xy=(x_range[1] + 0.3, 0.15), xytext=(x_range[1], 0.15),
                           arrowprops=dict(arrowstyle='->', color=color, lw=3))
            else:
                # Finite end - draw open or closed circle
                # FIXED: fillstyle determines if circle is filled or hollow
                if include_end:
                    ax.plot(end, 0.15, 'o', color=color, markersize=10, markeredgewidth=2)
                else:
                    ax.plot(end, 0.15, 'o', markerfacecolor='white', markeredgecolor=color,
                           markersize=10, markeredgewidth=2)
        
        # Draw highlight points (on the number line itself)
        for point in points:
            x = point.get('x', 0)
            label = point.get('label', '')
            color = point.get('color', '#d62728')
            # Check if this is an open or closed point
            is_open = point.get('open', False)
            
            if is_open:
                ax.plot(x, 0, 'o', markerfacecolor='white', markeredgecolor=color,
                       markersize=10, markeredgewidth=2)
            else:
                ax.plot(x, 0, 'o', color=color, markersize=10)
            
            if label:
                ax.annotate(label, (x, -0.4), ha='center', va='top', fontsize=10, fontweight='bold')
        
        ax.set_xlim(x_range[0] - 0.8, x_range[1] + 0.8)
        ax.set_ylim(-0.8, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        return self._save_figure(fig, spec)
    
    async def _render_venn_diagram(self, spec: Dict[str, Any]) -> RenderResult:
        """Render Venn diagram."""
        fig, ax = self._create_figure(spec)
        
        sets = spec.get('sets', [])
        title = spec.get('title', '')
        
        if len(sets) == 2:
            set_labels = (sets[0].get('label', 'A'), sets[1].get('label', 'B'))
            subsets = (
                sets[0].get('only', 10),
                sets[1].get('only', 10),
                spec.get('intersection', 5)
            )
            venn2(subsets=subsets, set_labels=set_labels, ax=ax)
        
        elif len(sets) == 3:
            set_labels = (
                sets[0].get('label', 'A'),
                sets[1].get('label', 'B'),
                sets[2].get('label', 'C')
            )
            subsets = (
                sets[0].get('only', 10),
                sets[1].get('only', 10),
                spec.get('ab_only', 3),
                sets[2].get('only', 10),
                spec.get('ac_only', 2),
                spec.get('bc_only', 2),
                spec.get('abc', 1)
            )
            venn3(subsets=subsets, set_labels=set_labels, ax=ax)
        else:
            raise RenderError("Venn diagram supports 2 or 3 sets only", "venn_diagram")
        
        if title:
            ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_bar_chart(self, spec: Dict[str, Any]) -> RenderResult:
        """Render bar chart."""
        fig, ax = self._create_figure(spec)
        
        categories = spec.get('categories', ['A', 'B', 'C'])
        values = spec.get('values', [10, 20, 15])
        colors = spec.get('colors', None)
        orientation = spec.get('orientation', 'vertical')
        title = spec.get('title', '')
        x_label = spec.get('x_label', '')
        y_label = spec.get('y_label', '')
        
        x = np.arange(len(categories))
        
        if orientation == 'horizontal':
            ax.barh(x, values, color=colors)
            ax.set_yticks(x)
            ax.set_yticklabels(categories)
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        else:
            ax.bar(x, values, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        
        if title:
            ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_pie_chart(self, spec: Dict[str, Any]) -> RenderResult:
        """Render pie chart."""
        fig, ax = self._create_figure(spec)
        
        labels = spec.get('labels', ['A', 'B', 'C'])
        values = spec.get('values', [30, 40, 30])
        colors = spec.get('colors', None)
        explode = spec.get('explode', None)
        title = spec.get('title', '')
        show_percentages = spec.get('show_percentages', True)
        
        autopct = '%1.1f%%' if show_percentages else None
        
        ax.pie(values, labels=labels, colors=colors, explode=explode,
               autopct=autopct, startangle=90)
        ax.axis('equal')
        
        if title:
            ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_trig_circle(self, spec: Dict[str, Any]) -> RenderResult:
        """Render trigonometric (unit) circle."""
        fig, ax = self._create_figure(spec)
        
        angle_degrees = spec.get('angle', 45)
        show_sin = spec.get('show_sin', True)
        show_cos = spec.get('show_cos', True)
        show_tan = spec.get('show_tan', False)
        title = spec.get('title', 'Unit Circle')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color=line_color, linewidth=1.5)
        
        # Draw axes
        ax.axhline(y=0, color=line_color, linewidth=0.5)
        ax.axvline(x=0, color=line_color, linewidth=0.5)
        
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)
        
        # Draw radius line
        ax.plot([0, cos_val], [0, sin_val], color='#1f77b4', linewidth=2, label='r = 1')
        ax.plot(cos_val, sin_val, 'o', color='#1f77b4', markersize=8)
        
        # Draw angle arc
        arc_theta = np.linspace(0, angle_rad, 30)
        arc_r = 0.2
        ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), 
               color='#2ca02c', linewidth=1.5)
        ax.annotate(f'{angle_degrees}°', (arc_r * 1.5, arc_r * 0.5), fontsize=10)
        
        # Draw sin and cos
        if show_cos:
            ax.plot([0, cos_val], [0, 0], color='#ff7f0e', linewidth=2, 
                   label=f'cos({angle_degrees}°) = {cos_val:.3f}')
            ax.plot([cos_val, cos_val], [0, sin_val], '--', color='#ff7f0e', linewidth=1)
        
        if show_sin:
            ax.plot([0, 0], [0, sin_val], color='#d62728', linewidth=2,
                   label=f'sin({angle_degrees}°) = {sin_val:.3f}')
            ax.plot([0, cos_val], [sin_val, sin_val], '--', color='#d62728', linewidth=1)
        
        if show_tan and abs(cos_val) > 0.01:
            tan_val = sin_val / cos_val
            if abs(tan_val) <= 3:  # Only show if reasonable
                ax.plot([1, 1], [0, tan_val], color='#9467bd', linewidth=2,
                       label=f'tan({angle_degrees}°) = {tan_val:.3f}')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    async def _render_3d_plot(self, spec: Dict[str, Any]) -> RenderResult:
        """Render 3D surface or parametric plot."""
        fig, ax = self._create_figure(spec, is_3d=True)
        
        plot_type = spec.get('plot_type', 'surface')
        expression = spec.get('expression', 'sin(sqrt(x**2 + y**2))')
        x_range = spec.get('x_range', [-5, 5])
        y_range = spec.get('y_range', [-5, 5])
        title = spec.get('title', '')
        colormap = spec.get('colormap', 'viridis')
        
        x = np.linspace(x_range[0], x_range[1], 50)
        y = np.linspace(y_range[0], y_range[1], 50)
        X, Y = np.meshgrid(x, y)
        
        try:
            Z = self._safe_eval_3d(expression, X, Y)
        except Exception as e:
            raise RenderError(f"Invalid 3D expression: {e}", "3d_plot")
        
        if plot_type == 'surface':
            ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.8, edgecolor='none')
        elif plot_type == 'wireframe':
            ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.7)
        elif plot_type == 'contour':
            ax.contour3D(X, Y, Z, 50, cmap=colormap)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if title:
            ax.set_title(title)
        
        return self._save_figure(fig, spec)
    
    def _safe_eval_3d(self, expr: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Safely evaluate a 3D mathematical expression."""
        safe_dict = {
            'x': X, 'y': Y,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'abs': np.abs, 'pi': np.pi, 'e': np.e,
        }
        
        expr = expr.replace('^', '**')
        
        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return np.where(np.isfinite(result), result, np.nan)
        except Exception as e:
            raise ValueError(f"Invalid expression '{expr}': {e}")
