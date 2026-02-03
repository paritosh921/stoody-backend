"""
Biology Diagram Renderer

Renders biology diagrams using matplotlib with SVG output capability
for anatomical and cellular diagrams.

Supported diagram types:
- human_heart: Heart structure with labeled chambers
- human_brain: Brain regions and structures
- nephron: Kidney nephron structure
- neuron: Nerve cell structure
- plant_cell: Plant cell with organelles
- animal_cell: Animal cell with organelles
- dna_replication: DNA replication process
- mitosis_stages: Stages of mitosis
- meiosis_stages: Stages of meiosis
- digestive_system: Human digestive tract
- respiratory_system: Human respiratory system
- eye_structure: Human eye anatomy
- ear_structure: Human ear anatomy
- flower_structure: Flower parts diagram
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import io
import math
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import (
        Circle, Ellipse, Rectangle, Polygon, FancyArrowPatch, 
        Arc, Wedge, PathPatch, FancyBboxPatch
    )
    from matplotlib.path import Path
    import matplotlib.colors as mcolors
    from matplotlib.collections import PatchCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..base_renderer import BaseRenderer, RenderResult, RenderError
from ..specs.base_spec import (
    DiagramSubject,
    OutputFormat,
    SUPPORTED_DIAGRAM_TYPES,
)

logger = logging.getLogger(__name__)


# Color palettes for biological diagrams
BIO_COLORS = {
    # Cell colors
    'cell_membrane': '#F4A460',
    'cytoplasm': '#FFFACD',
    'nucleus': '#9370DB',
    'nucleolus': '#663399',
    'mitochondria': '#FF6347',
    'chloroplast': '#228B22',
    'vacuole': '#87CEEB',
    'cell_wall': '#8B4513',
    'ribosome': '#FF69B4',
    'endoplasmic_reticulum': '#DDA0DD',
    'golgi': '#FFD700',
    'lysosome': '#FF4500',
    'centriole': '#4169E1',
    
    # Organ colors
    'blood': '#DC143C',
    'oxygenated_blood': '#FF0000',
    'deoxygenated_blood': '#8B0000',
    'muscle': '#CD5C5C',
    'tissue': '#FFB6C1',
    'bone': '#F5F5DC',
    'nerve': '#FFFF00',
    
    # DNA colors
    'adenine': '#FF6B6B',
    'thymine': '#4ECDC4',
    'guanine': '#45B7D1',
    'cytosine': '#96CEB4',
    'backbone': '#6C5B7B',
}


class BiologyRenderer(BaseRenderer):
    """
    Renderer for biology diagrams.
    
    Uses matplotlib to create detailed biological diagrams
    with SVG output support for high-quality anatomical illustrations.
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for BiologyRenderer. "
                "Install with: pip install matplotlib"
            )
    
    @property
    def subject(self) -> DiagramSubject:
        return DiagramSubject.BIOLOGY
    
    def get_supported_types(self) -> List[str]:
        return SUPPORTED_DIAGRAM_TYPES.get(DiagramSubject.BIOLOGY, [])
    
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render a biology diagram based on specification.
        """
        is_valid, error = self.validate_spec(spec)
        if not is_valid:
            raise RenderError(error, spec.get('diagram_type', 'unknown'))
        
        spec = self._apply_style(spec)
        diagram_type = spec['diagram_type']
        
        render_methods = {
            'human_heart': self._render_heart,
            'human_brain': self._render_brain,
            'nephron': self._render_nephron,
            'neuron': self._render_neuron,
            'plant_cell': self._render_plant_cell,
            'animal_cell': self._render_animal_cell,
            'dna_replication': self._render_dna_replication,
            'mitosis_stages': self._render_mitosis,
            'meiosis_stages': self._render_meiosis,
            'digestive_system': self._render_digestive_system,
            'respiratory_system': self._render_respiratory_system,
            'eye_structure': self._render_eye,
            'ear_structure': self._render_ear,
            'flower_structure': self._render_flower,
        }
        
        render_method = render_methods.get(diagram_type)
        if not render_method:
            raise RenderError(
                f"No renderer for diagram type: {diagram_type}",
                diagram_type
            )
        
        try:
            return await render_method(spec)
        except Exception as e:
            logger.error(f"Error rendering {diagram_type}: {e}")
            raise RenderError(str(e), diagram_type, {'exception': type(e).__name__})
    
    def _create_figure(
        self,
        spec: Dict[str, Any],
        figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create matplotlib figure with standard settings."""
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800)
        height = dimensions.get('height', 600)
        dpi = self._get_dpi(spec)
        
        if figsize is None:
            figsize = (width / dpi, height / dpi)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        return fig, ax
    
    def _save_figure(
        self,
        fig: plt.Figure,
        spec: Dict[str, Any]
    ) -> RenderResult:
        """Save figure and return RenderResult."""
        output_format = OutputFormat(spec.get('output_format', 'png'))
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800)
        height = dimensions.get('height', 600)
        
        buffer = io.BytesIO()
        
        if output_format == OutputFormat.SVG:
            fig.savefig(buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
        elif output_format == OutputFormat.PDF:
            fig.savefig(buffer, format='pdf', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
            output_format = OutputFormat.PNG
        
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
    
    def _add_label(
        self,
        ax: plt.Axes,
        text: str,
        position: Tuple[float, float],
        target: Tuple[float, float],
        fontsize: int = 9,
        color: str = 'black'
    ):
        """Add a label with a leader line to a target point."""
        ax.annotate(
            text,
            xy=target,
            xytext=position,
            fontsize=fontsize,
            ha='center',
            va='center',
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                lw=0.5
            )
        )
    
    # =========================================================================
    # Cell Diagrams
    # =========================================================================
    
    async def _render_plant_cell(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render plant cell structure.
        
        Spec parameters:
            - show_labels: Whether to show organelle labels
            - highlight_organelles: List of organelles to highlight
        """
        show_labels = spec.get('show_labels', True)
        highlight = spec.get('highlight_organelles', [])
        
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Cell wall (rectangular)
        cell_wall = FancyBboxPatch(
            (1, 1), 10, 8,
            boxstyle="round,pad=0.1",
            facecolor=BIO_COLORS['cell_wall'],
            edgecolor='#5D4037',
            linewidth=3
        )
        ax.add_patch(cell_wall)
        
        # Cell membrane
        membrane = FancyBboxPatch(
            (1.3, 1.3), 9.4, 7.4,
            boxstyle="round,pad=0.1",
            facecolor=BIO_COLORS['cytoplasm'],
            edgecolor=BIO_COLORS['cell_membrane'],
            linewidth=2
        )
        ax.add_patch(membrane)
        
        # Large central vacuole
        vacuole = Ellipse(
            (6, 5), 5, 4,
            facecolor=BIO_COLORS['vacuole'],
            edgecolor='#4682B4',
            linewidth=1.5,
            alpha=0.6
        )
        ax.add_patch(vacuole)
        
        # Nucleus
        nucleus = Circle(
            (3, 6.5), 1,
            facecolor=BIO_COLORS['nucleus'],
            edgecolor='#4B0082',
            linewidth=1.5
        )
        ax.add_patch(nucleus)
        
        # Nucleolus
        nucleolus = Circle(
            (3, 6.5), 0.3,
            facecolor=BIO_COLORS['nucleolus'],
            edgecolor='#2E0854'
        )
        ax.add_patch(nucleolus)
        
        # Chloroplasts
        chloroplast_positions = [(2.5, 3), (9, 7), (9.5, 3.5), (2, 8)]
        for pos in chloroplast_positions:
            chloroplast = Ellipse(
                pos, 0.8, 0.4,
                facecolor=BIO_COLORS['chloroplast'],
                edgecolor='#006400',
                linewidth=1
            )
            ax.add_patch(chloroplast)
            # Granum lines
            for i in range(3):
                offset = -0.15 + i * 0.15
                ax.plot(
                    [pos[0] - 0.25, pos[0] + 0.25],
                    [pos[1] + offset, pos[1] + offset],
                    color='#004d00', linewidth=1
                )
        
        # Mitochondria
        mito_positions = [(9, 5), (2.5, 4.5)]
        for pos in mito_positions:
            mito = Ellipse(
                pos, 0.6, 0.35,
                facecolor=BIO_COLORS['mitochondria'],
                edgecolor='#8B0000',
                linewidth=1
            )
            ax.add_patch(mito)
            # Cristae
            for i in range(2):
                ax.plot(
                    [pos[0] - 0.15 + i * 0.2, pos[0] - 0.15 + i * 0.2],
                    [pos[1] - 0.12, pos[1] + 0.12],
                    color='#8B0000', linewidth=0.8
                )
        
        # Endoplasmic reticulum
        er_x = np.linspace(4, 5, 20)
        er_y = 7.5 + 0.3 * np.sin(er_x * 8)
        ax.plot(er_x, er_y, color=BIO_COLORS['endoplasmic_reticulum'], linewidth=2)
        ax.plot(er_x, er_y - 0.2, color=BIO_COLORS['endoplasmic_reticulum'], linewidth=2)
        
        # Ribosomes (small dots)
        for _ in range(20):
            rx = np.random.uniform(1.8, 10)
            ry = np.random.uniform(1.8, 8.5)
            # Skip if inside vacuole
            if ((rx - 6)**2 / 6.25 + (ry - 5)**2 / 4) < 0.8:
                continue
            ax.add_patch(Circle((rx, ry), 0.05, facecolor=BIO_COLORS['ribosome']))
        
        # Golgi apparatus
        for i in range(4):
            golgi = Arc(
                (8, 2.5), 1.5, 0.4 + i * 0.15,
                angle=0, theta1=30, theta2=150,
                color=BIO_COLORS['golgi'], linewidth=2
            )
            ax.add_patch(golgi)
        
        # Labels
        if show_labels:
            labels = [
                ('Cell Wall', (0.3, 5), (1, 5)),
                ('Cell Membrane', (0.3, 4), (1.4, 4)),
                ('Vacuole', (6, 5), (6, 5)),
                ('Nucleus', (3, 8.5), (3, 7.5)),
                ('Chloroplast', (11, 7), (9, 7)),
                ('Mitochondria', (10.5, 5), (9.4, 5)),
                ('Golgi Apparatus', (10, 2.5), (8.5, 2.5)),
                ('ER', (5.5, 8), (4.5, 7.5)),
            ]
            
            for label, text_pos, target in labels:
                self._add_label(ax, label, text_pos, target)
        
        ax.set_title('Plant Cell Structure', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    async def _render_animal_cell(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render animal cell structure.
        """
        show_labels = spec.get('show_labels', True)
        
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Cell membrane (irregular shape using polygon)
        theta = np.linspace(0, 2 * np.pi, 50)
        r = 4 + 0.3 * np.sin(5 * theta) + 0.2 * np.cos(3 * theta)
        cx, cy = 6, 5
        membrane_x = cx + r * np.cos(theta)
        membrane_y = cy + r * np.sin(theta)
        
        ax.fill(membrane_x, membrane_y, 
               facecolor=BIO_COLORS['cytoplasm'],
               edgecolor=BIO_COLORS['cell_membrane'],
               linewidth=2)
        
        # Nucleus
        nucleus = Ellipse(
            (5, 5.5), 2.5, 2,
            facecolor=BIO_COLORS['nucleus'],
            edgecolor='#4B0082',
            linewidth=2
        )
        ax.add_patch(nucleus)
        
        # Nuclear membrane pores
        for angle in [30, 90, 150, 210, 270, 330]:
            px = 5 + 1.25 * np.cos(np.radians(angle))
            py = 5.5 + 1 * np.sin(np.radians(angle))
            ax.add_patch(Circle((px, py), 0.08, facecolor='white'))
        
        # Nucleolus
        ax.add_patch(Circle((5, 5.5), 0.5, facecolor=BIO_COLORS['nucleolus']))
        
        # Mitochondria
        mito_positions = [(8, 6.5), (3, 3.5), (8.5, 3.5), (2.5, 6)]
        for pos in mito_positions:
            angle = np.random.uniform(0, 360)
            mito = Ellipse(
                pos, 0.8, 0.4,
                angle=angle,
                facecolor=BIO_COLORS['mitochondria'],
                edgecolor='#8B0000',
                linewidth=1
            )
            ax.add_patch(mito)
        
        # Centrioles
        for i, pos in enumerate([(7.5, 5), (7.8, 5.3)]):
            angle = 45 + i * 90
            centriole = Rectangle(
                (pos[0] - 0.15, pos[1] - 0.3), 0.3, 0.6,
                angle=angle,
                facecolor=BIO_COLORS['centriole'],
                edgecolor='#000080'
            )
            ax.add_patch(centriole)
        
        # Endoplasmic reticulum (rough ER near nucleus)
        er_x = np.linspace(6.5, 9, 30)
        for offset in [0, 0.3, 0.6]:
            er_y = 6.5 + offset + 0.15 * np.sin(er_x * 6)
            ax.plot(er_x, er_y, color=BIO_COLORS['endoplasmic_reticulum'], 
                   linewidth=2, alpha=0.8)
        
        # Ribosomes on rough ER
        for _ in range(15):
            rx = np.random.uniform(6.5, 9)
            ry = np.random.uniform(6.3, 7.3)
            ax.add_patch(Circle((rx, ry), 0.06, facecolor=BIO_COLORS['ribosome']))
        
        # Golgi apparatus
        for i in range(5):
            golgi = Arc(
                (3.5, 6.5), 1.2, 0.3 + i * 0.12,
                angle=30, theta1=30, theta2=150,
                color=BIO_COLORS['golgi'], linewidth=2
            )
            ax.add_patch(golgi)
        
        # Lysosomes
        lyso_positions = [(4, 3), (8, 4.5), (3.5, 7.5)]
        for pos in lyso_positions:
            ax.add_patch(Circle(pos, 0.25, 
                               facecolor=BIO_COLORS['lysosome'],
                               edgecolor='#CC3300'))
        
        # Small vacuoles
        for _ in range(3):
            vx = np.random.uniform(3, 9)
            vy = np.random.uniform(2.5, 7)
            if ((vx - 5)**2 / 1.5 + (vy - 5.5)**2) > 1.5:  # Outside nucleus
                ax.add_patch(Circle((vx, vy), 0.2,
                                   facecolor=BIO_COLORS['vacuole'],
                                   edgecolor='#4682B4', alpha=0.5))
        
        # Labels
        if show_labels:
            labels = [
                ('Cell Membrane', (11, 5), (9.8, 5)),
                ('Nucleus', (5, 8), (5, 6.5)),
                ('Nucleolus', (6.5, 5.5), (5.5, 5.5)),
                ('Mitochondria', (10, 6.5), (8.5, 6.5)),
                ('Centrioles', (9, 5.2), (8, 5.2)),
                ('Rough ER', (10, 7.5), (8, 7)),
                ('Golgi Apparatus', (1.5, 6.5), (3, 6.5)),
                ('Lysosome', (1.5, 3), (3.5, 3)),
            ]
            
            for label, text_pos, target in labels:
                self._add_label(ax, label, text_pos, target)
        
        ax.set_title('Animal Cell Structure', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Neuron Diagram
    # =========================================================================
    
    async def _render_neuron(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render nerve cell (neuron) structure.
        """
        show_labels = spec.get('show_labels', True)
        
        fig, ax = self._create_figure(spec, figsize=(14, 8))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Cell body (soma)
        soma = Circle((3, 4), 1.2, facecolor='#FFE4B5', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(soma)
        
        # Nucleus
        ax.add_patch(Circle((3, 4), 0.5, facecolor=BIO_COLORS['nucleus'], edgecolor='#4B0082'))
        
        # Dendrites
        dendrite_starts = [(2, 4.8), (1.8, 4.2), (2, 3.2), (2.2, 2.5), (2.8, 2)]
        for start in dendrite_starts:
            # Main branch
            end_x = start[0] - 0.8
            end_y = start[1] + np.random.uniform(-0.3, 0.3)
            ax.plot([start[0], end_x], [start[1], end_y], 
                   color='#8B4513', linewidth=2)
            # Sub-branches
            for _ in range(2):
                branch_end_x = end_x - 0.4
                branch_end_y = end_y + np.random.uniform(-0.4, 0.4)
                ax.plot([end_x, branch_end_x], [end_y, branch_end_y],
                       color='#8B4513', linewidth=1)
        
        # Axon hillock
        ax.add_patch(Polygon(
            [(4.2, 4.3), (4.2, 3.7), (5, 4)],
            closed=True, facecolor='#FFE4B5', edgecolor='#8B4513', linewidth=2
        ))
        
        # Axon
        axon_x = np.linspace(5, 12, 100)
        axon_y = 4 + 0.1 * np.sin(axon_x * 2)
        ax.plot(axon_x, axon_y, color='#8B4513', linewidth=2)
        
        # Myelin sheath segments
        myelin_starts = [5.5, 6.8, 8.1, 9.4, 10.7]
        for start in myelin_starts:
            myelin = FancyBboxPatch(
                (start, 3.7), 0.8, 0.6,
                boxstyle="round,pad=0.02",
                facecolor='#E6E6FA',
                edgecolor='#9370DB',
                linewidth=1
            )
            ax.add_patch(myelin)
        
        # Nodes of Ranvier (gaps)
        for i, start in enumerate(myelin_starts[:-1]):
            ax.plot([start + 0.8, start + 1.3], [4, 4], 
                   color='#8B4513', linewidth=2)
        
        # Schwann cell nuclei
        for start in myelin_starts:
            ax.add_patch(Ellipse((start + 0.4, 3.5), 0.2, 0.1,
                                facecolor=BIO_COLORS['nucleus']))
        
        # Axon terminal (synaptic end bulbs)
        terminal_positions = [(12.3, 4.3), (12.3, 4), (12.3, 3.7), (12.6, 4.15), (12.6, 3.85)]
        for pos in terminal_positions:
            ax.add_patch(Circle(pos, 0.15, facecolor='#FFD700', edgecolor='#B8860B'))
        
        # Synaptic vesicles
        for pos in terminal_positions[:3]:
            for _ in range(3):
                vx = pos[0] + np.random.uniform(-0.08, 0.08)
                vy = pos[1] + np.random.uniform(-0.08, 0.08)
                ax.add_patch(Circle((vx, vy), 0.03, facecolor='#FF6347'))
        
        # Labels
        if show_labels:
            labels = [
                ('Dendrites', (0.5, 4.5), (1.2, 4.5)),
                ('Cell Body\n(Soma)', (3, 2), (3, 2.8)),
                ('Nucleus', (3, 5.5), (3, 4.5)),
                ('Axon Hillock', (4.6, 5), (4.6, 4.3)),
                ('Axon', (7.5, 5), (7.5, 4.1)),
                ('Myelin Sheath', (6.8, 3), (6.8, 3.7)),
                ('Node of Ranvier', (6.3, 4.8), (6.3, 4.1)),
                ('Axon Terminals', (13, 4), (12.5, 4)),
            ]
            
            for label, text_pos, target in labels:
                self._add_label(ax, label, text_pos, target, fontsize=8)
        
        ax.set_title('Neuron (Nerve Cell) Structure', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Heart Diagram
    # =========================================================================
    
    async def _render_heart(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render human heart structure.
        """
        show_labels = spec.get('show_labels', True)
        show_blood_flow = spec.get('show_blood_flow', True)
        
        fig, ax = self._create_figure(spec, figsize=(12, 12))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Right atrium
        ra = FancyBboxPatch(
            (2, 6), 2.5, 2.5,
            boxstyle="round,pad=0.2",
            facecolor=BIO_COLORS['deoxygenated_blood'],
            edgecolor='#4B0082',
            linewidth=2, alpha=0.8
        )
        ax.add_patch(ra)
        
        # Left atrium
        la = FancyBboxPatch(
            (7.5, 6), 2.5, 2.5,
            boxstyle="round,pad=0.2",
            facecolor=BIO_COLORS['oxygenated_blood'],
            edgecolor='#4B0082',
            linewidth=2, alpha=0.8
        )
        ax.add_patch(la)
        
        # Right ventricle
        rv_points = [(2, 6), (5, 6), (5, 3), (3.5, 1.5), (2, 3)]
        rv = Polygon(rv_points, closed=True,
                    facecolor=BIO_COLORS['deoxygenated_blood'],
                    edgecolor='#4B0082', linewidth=2, alpha=0.8)
        ax.add_patch(rv)
        
        # Left ventricle
        lv_points = [(7, 6), (10, 6), (10, 3), (8.5, 1.5), (7, 3)]
        lv = Polygon(lv_points, closed=True,
                    facecolor=BIO_COLORS['oxygenated_blood'],
                    edgecolor='#4B0082', linewidth=2, alpha=0.8)
        ax.add_patch(lv)
        
        # Septum
        ax.add_patch(Rectangle((5, 1.5), 2, 7, facecolor=BIO_COLORS['muscle'],
                               edgecolor='#8B4513', linewidth=2))
        
        # Superior vena cava
        ax.add_patch(Rectangle((2.5, 8.5), 0.8, 2, 
                               facecolor=BIO_COLORS['deoxygenated_blood'],
                               edgecolor='#4B0082', linewidth=1.5))
        ax.text(2.9, 11, 'Superior\nVena Cava', ha='center', fontsize=8)
        
        # Inferior vena cava
        ax.add_patch(Rectangle((3.2, 0), 0.8, 1.5,
                               facecolor=BIO_COLORS['deoxygenated_blood'],
                               edgecolor='#4B0082', linewidth=1.5))
        ax.text(3.6, -0.5, 'Inferior\nVena Cava', ha='center', fontsize=8)
        
        # Pulmonary artery
        pa_start = (4.5, 8.5)
        ax.add_patch(FancyBboxPatch(
            (4, 8.5), 1, 2,
            boxstyle="round,pad=0.1",
            facecolor=BIO_COLORS['deoxygenated_blood'],
            edgecolor='#4B0082', linewidth=1.5
        ))
        ax.text(4.5, 11, 'Pulmonary\nArtery', ha='center', fontsize=8)
        
        # Pulmonary veins
        ax.add_patch(Rectangle((8.2, 8.5), 0.6, 2,
                               facecolor=BIO_COLORS['oxygenated_blood'],
                               edgecolor='#8B0000', linewidth=1.5))
        ax.add_patch(Rectangle((9, 8.5), 0.6, 2,
                               facecolor=BIO_COLORS['oxygenated_blood'],
                               edgecolor='#8B0000', linewidth=1.5))
        ax.text(8.9, 11, 'Pulmonary\nVeins', ha='center', fontsize=8)
        
        # Aorta
        aorta_points = [(6.5, 8.5), (6.5, 10), (8, 10.5), (9.5, 10), (9.5, 8.5)]
        ax.plot(*zip(*aorta_points), color=BIO_COLORS['oxygenated_blood'], 
               linewidth=8, solid_capstyle='round')
        ax.text(8, 11.2, 'Aorta', ha='center', fontsize=9, fontweight='bold')
        
        # Valves
        # Tricuspid valve
        ax.plot([2.5, 4.5], [6, 6], color='#FFD700', linewidth=3)
        # Mitral valve  
        ax.plot([7.5, 9.5], [6, 6], color='#FFD700', linewidth=3)
        
        # Labels for chambers
        if show_labels:
            ax.text(3.2, 7.2, 'Right\nAtrium', ha='center', fontsize=10, fontweight='bold')
            ax.text(8.7, 7.2, 'Left\nAtrium', ha='center', fontsize=10, fontweight='bold')
            ax.text(3.5, 4, 'Right\nVentricle', ha='center', fontsize=10, fontweight='bold')
            ax.text(8.5, 4, 'Left\nVentricle', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, 5, 'Septum', ha='center', fontsize=9, rotation=90)
        
        # Blood flow arrows
        if show_blood_flow:
            # Into right atrium
            ax.annotate('', xy=(3, 8), xytext=(3, 9.5),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            # RA to RV
            ax.annotate('', xy=(3.5, 5), xytext=(3.5, 6.5),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            # RV to pulmonary
            ax.annotate('', xy=(4.5, 9), xytext=(4.5, 7),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            # Pulmonary veins to LA
            ax.annotate('', xy=(8.7, 8), xytext=(8.7, 9.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            # LA to LV
            ax.annotate('', xy=(8.5, 5), xytext=(8.5, 6.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            # LV to aorta
            ax.annotate('', xy=(7, 9), xytext=(7, 7),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_title('Human Heart Structure', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Brain Diagram
    # =========================================================================
    
    async def _render_brain(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render human brain regions.
        """
        show_labels = spec.get('show_labels', True)
        highlight_region = spec.get('highlight_region', None)
        
        fig, ax = self._create_figure(spec, figsize=(14, 10))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Brain outline (side view)
        brain_outline = [
            (2, 4), (3, 6), (4, 7.5), (6, 8.5), (9, 8.5), (11, 7.5),
            (12, 6), (12.5, 4.5), (12, 3), (10, 2), (7, 1.5), (4, 2), (2, 3)
        ]
        brain = Polygon(brain_outline, closed=True,
                       facecolor='#FFC0CB', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(brain)
        
        # Cerebrum folds (gyri)
        fold_lines = [
            [(3, 5), (4, 6.5), (6, 7)],
            [(4, 4), (5.5, 5.5), (7, 6)],
            [(6, 4.5), (8, 5.5), (10, 5.5)],
            [(7, 7), (9, 7.5), (11, 7)],
            [(8, 3.5), (9.5, 4.5), (11, 5)],
        ]
        for fold in fold_lines:
            xs, ys = zip(*fold)
            ax.plot(xs, ys, color='#CD5C5C', linewidth=1.5)
        
        # Frontal lobe
        ax.add_patch(Ellipse((4, 5.5), 3, 4, angle=15,
                            facecolor='#FFB6C1' if highlight_region != 'frontal' else '#FF69B4',
                            edgecolor='#8B4513', linewidth=1, alpha=0.3))
        
        # Parietal lobe
        ax.add_patch(Ellipse((7.5, 6.5), 3.5, 3, angle=0,
                            facecolor='#98FB98' if highlight_region != 'parietal' else '#32CD32',
                            edgecolor='#8B4513', linewidth=1, alpha=0.3))
        
        # Occipital lobe
        ax.add_patch(Ellipse((11, 5.5), 2, 3,
                            facecolor='#87CEEB' if highlight_region != 'occipital' else '#00BFFF',
                            edgecolor='#8B4513', linewidth=1, alpha=0.3))
        
        # Temporal lobe
        ax.add_patch(Ellipse((6, 3), 4, 2, angle=-10,
                            facecolor='#DDA0DD' if highlight_region != 'temporal' else '#DA70D6',
                            edgecolor='#8B4513', linewidth=1, alpha=0.3))
        
        # Cerebellum
        cerebellum = Ellipse((10.5, 2.5), 3, 2, angle=-20,
                            facecolor='#F0E68C', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(cerebellum)
        # Cerebellum folds
        for i in range(5):
            arc = Arc((10.5, 2.5), 2 - i * 0.3, 1.5 - i * 0.2,
                     angle=-20, theta1=-60, theta2=60,
                     color='#DAA520', linewidth=1)
            ax.add_patch(arc)
        
        # Brain stem
        brainstem = Polygon(
            [(8, 2), (8.5, 1), (9.5, 1), (10, 2)],
            closed=True, facecolor='#DEB887', edgecolor='#8B4513', linewidth=2
        )
        ax.add_patch(brainstem)
        
        # Labels
        if show_labels:
            ax.text(3.5, 6, 'Frontal\nLobe', ha='center', fontsize=10, fontweight='bold')
            ax.text(7.5, 7, 'Parietal\nLobe', ha='center', fontsize=10, fontweight='bold')
            ax.text(11, 6, 'Occipital\nLobe', ha='center', fontsize=10, fontweight='bold')
            ax.text(5.5, 2.5, 'Temporal\nLobe', ha='center', fontsize=10, fontweight='bold')
            ax.text(10.5, 2.5, 'Cerebellum', ha='center', fontsize=9)
            ax.text(9, 0.5, 'Brain\nStem', ha='center', fontsize=9)
        
        ax.set_title('Human Brain Structure (Lateral View)', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # DNA Replication Diagram
    # =========================================================================
    
    async def _render_dna_replication(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render DNA replication process.
        """
        show_labels = spec.get('show_labels', True)
        
        fig, ax = self._create_figure(spec, figsize=(14, 10))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Original DNA strand (double helix simplified)
        y_center = 5
        
        # Draw the replication fork
        # Parent strand left side
        x_left = np.linspace(0, 4, 50)
        strand1_y = y_center + 0.3 * np.sin(x_left * 3)
        strand2_y = y_center - 0.3 * np.sin(x_left * 3)
        
        ax.plot(x_left, strand1_y, color=BIO_COLORS['backbone'], linewidth=3)
        ax.plot(x_left, strand2_y, color=BIO_COLORS['backbone'], linewidth=3)
        
        # Base pairs on parent strand
        for i in range(0, len(x_left), 5):
            x = x_left[i]
            y1 = strand1_y[i]
            y2 = strand2_y[i]
            ax.plot([x, x], [y1, y2], color='gray', linewidth=1)
            
            # Alternating base colors
            bases = [(BIO_COLORS['adenine'], BIO_COLORS['thymine']),
                    (BIO_COLORS['guanine'], BIO_COLORS['cytosine'])]
            b1, b2 = bases[i % 2]
            ax.add_patch(Circle((x, (y1 + y_center) / 2), 0.15, facecolor=b1))
            ax.add_patch(Circle((x, (y2 + y_center) / 2), 0.15, facecolor=b2))
        
        # Replication fork opening
        # Top strand unwinding
        x_fork = np.linspace(4, 7, 30)
        top_y = y_center + 0.3 + (x_fork - 4) * 0.5
        bottom_y = y_center - 0.3 - (x_fork - 4) * 0.5
        
        ax.plot(x_fork, top_y, color=BIO_COLORS['backbone'], linewidth=3)
        ax.plot(x_fork, bottom_y, color=BIO_COLORS['backbone'], linewidth=3)
        
        # New complementary strands being synthesized
        # Leading strand (continuous)
        x_new = np.linspace(5, 7, 20)
        new_top_y = top_y[-20:] - 0.4
        ax.plot(x_new, new_top_y, color='#32CD32', linewidth=3, linestyle='--')
        
        # Lagging strand (Okazaki fragments)
        for start in [5, 5.8, 6.6]:
            x_frag = np.linspace(start, start + 0.6, 10)
            frag_y = y_center - 0.3 - (x_frag - 4) * 0.5 + 0.4
            ax.plot(x_frag, frag_y, color='#FF6347', linewidth=3)
        
        # Helicase
        ax.add_patch(Ellipse((4.2, y_center), 0.6, 1.2, angle=0,
                            facecolor='#FFD700', edgecolor='#B8860B', linewidth=2))
        ax.text(4.2, y_center, 'H', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # DNA Polymerase
        ax.add_patch(Circle((6.5, top_y[-10] - 0.3), 0.3,
                           facecolor='#9370DB', edgecolor='#4B0082'))
        ax.add_patch(Circle((6, bottom_y[-20] + 0.5), 0.3,
                           facecolor='#9370DB', edgecolor='#4B0082'))
        
        # Right side (replicated DNA)
        x_right = np.linspace(7, 14, 70)
        
        # Top replicated strand
        r_top1_y = 7 + 0.3 * np.sin(x_right * 3)
        r_top2_y = 7 - 0.3 * np.sin(x_right * 3)
        ax.plot(x_right, r_top1_y, color=BIO_COLORS['backbone'], linewidth=3)
        ax.plot(x_right, r_top2_y, color='#32CD32', linewidth=3)
        
        # Bottom replicated strand
        r_bot1_y = 3 + 0.3 * np.sin(x_right * 3)
        r_bot2_y = 3 - 0.3 * np.sin(x_right * 3)
        ax.plot(x_right, r_bot1_y, color='#FF6347', linewidth=3)
        ax.plot(x_right, r_bot2_y, color=BIO_COLORS['backbone'], linewidth=3)
        
        # Labels
        if show_labels:
            ax.text(2, 6.5, 'Parent DNA\n(Template)', ha='center', fontsize=10)
            ax.text(4.2, 6.5, 'Helicase', ha='center', fontsize=9)
            ax.text(7, 5, 'Replication\nFork', ha='center', fontsize=10)
            ax.text(10, 8.5, 'Leading Strand\n(Continuous)', ha='center', fontsize=9)
            ax.text(10, 1.5, 'Lagging Strand\n(Okazaki Fragments)', ha='center', fontsize=9)
            ax.text(6.5, 8, 'DNA Pol III', ha='center', fontsize=8)
            
            # Legend
            ax.add_patch(Rectangle((0.5, 0.5), 0.3, 0.3, facecolor=BIO_COLORS['backbone']))
            ax.text(1, 0.65, 'Original strand', fontsize=8)
            ax.add_patch(Rectangle((3.5, 0.5), 0.3, 0.3, facecolor='#32CD32'))
            ax.text(4, 0.65, 'New strand', fontsize=8)
        
        ax.set_title('DNA Replication', fontsize=14, fontweight='bold', pad=15)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Mitosis/Meiosis Stages
    # =========================================================================
    
    async def _render_mitosis(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render stages of mitosis.
        """
        stages = ['Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase', 'Cytokinesis']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self._get_dpi(spec))
        axes = axes.flatten()
        
        for i, (ax, stage) in enumerate(zip(axes, stages)):
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(stage, fontsize=12, fontweight='bold')
            
            # Cell outline
            if stage == 'Cytokinesis':
                # Two cells forming
                ax.add_patch(Ellipse((1.3, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
                ax.add_patch(Ellipse((2.7, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
            else:
                ax.add_patch(Ellipse((2, 2), 3, 3.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
            
            # Stage-specific content
            if stage == 'Interphase':
                # Intact nucleus with chromatin
                ax.add_patch(Circle((2, 2), 0.8, facecolor='#E6E6FA', edgecolor='#4B0082', linewidth=1.5))
                # Chromatin threads
                for _ in range(5):
                    x = 2 + np.random.uniform(-0.5, 0.5)
                    y = 2 + np.random.uniform(-0.5, 0.5)
                    ax.plot([x, x + 0.2], [y, y + 0.1], color='#800080', linewidth=2)
                
            elif stage == 'Prophase':
                # Condensing chromosomes, nuclear envelope breaking down
                ax.add_patch(Circle((2, 2), 0.8, facecolor='none', edgecolor='#4B0082', 
                                   linewidth=1.5, linestyle='--'))
                # X-shaped chromosomes
                positions = [(1.7, 2.2), (2.3, 1.8), (2, 2.3), (2.1, 1.7)]
                for pos in positions:
                    ax.plot([pos[0]-0.15, pos[0]+0.15], [pos[1]-0.15, pos[1]+0.15], 
                           color='#800080', linewidth=3)
                    ax.plot([pos[0]-0.15, pos[0]+0.15], [pos[1]+0.15, pos[1]-0.15], 
                           color='#800080', linewidth=3)
                
            elif stage == 'Metaphase':
                # Chromosomes aligned at metaphase plate
                ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
                ax.text(0.3, 2, 'Metaphase\nplate', fontsize=7, va='center')
                for x in [1.3, 1.7, 2.1, 2.5, 2.9]:
                    ax.plot([x-0.1, x+0.1], [1.85, 2.15], color='#800080', linewidth=3)
                    ax.plot([x-0.1, x+0.1], [2.15, 1.85], color='#800080', linewidth=3)
                # Spindle fibers
                for x in [1.3, 1.7, 2.1, 2.5, 2.9]:
                    ax.plot([x, 0.5], [2, 3], color='#90EE90', linewidth=0.5, alpha=0.5)
                    ax.plot([x, 3.5], [2, 3], color='#90EE90', linewidth=0.5, alpha=0.5)
                    ax.plot([x, 0.5], [2, 1], color='#90EE90', linewidth=0.5, alpha=0.5)
                    ax.plot([x, 3.5], [2, 1], color='#90EE90', linewidth=0.5, alpha=0.5)
                
            elif stage == 'Anaphase':
                # Sister chromatids separating
                for x in [1.4, 1.8, 2.2, 2.6]:
                    # Moving up
                    ax.plot([x, x+0.1], [2.8, 2.6], color='#800080', linewidth=2)
                    # Moving down
                    ax.plot([x, x+0.1], [1.2, 1.4], color='#800080', linewidth=2)
                # Spindle fibers
                ax.plot([2, 2], [0.8, 3.2], color='#90EE90', linewidth=0.5, linestyle='--')
                
            elif stage == 'Telophase':
                # Two nuclei forming
                ax.add_patch(Circle((2, 2.8), 0.5, facecolor='#E6E6FA', edgecolor='#4B0082', linewidth=1))
                ax.add_patch(Circle((2, 1.2), 0.5, facecolor='#E6E6FA', edgecolor='#4B0082', linewidth=1))
                # Chromatin decondensing
                for center_y in [2.8, 1.2]:
                    for _ in range(3):
                        x = 2 + np.random.uniform(-0.3, 0.3)
                        y = center_y + np.random.uniform(-0.3, 0.3)
                        ax.plot([x, x + 0.1], [y, y + 0.05], color='#800080', linewidth=1.5)
                # Cleavage furrow forming
                ax.plot([0.8, 3.2], [2, 2], color='#8B4513', linewidth=2, linestyle='--')
                
            elif stage == 'Cytokinesis':
                # Two daughter cells with nuclei
                ax.add_patch(Circle((1.3, 2), 0.4, facecolor='#E6E6FA', edgecolor='#4B0082', linewidth=1))
                ax.add_patch(Circle((2.7, 2), 0.4, facecolor='#E6E6FA', edgecolor='#4B0082', linewidth=1))
        
        fig.suptitle('Stages of Mitosis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return self._save_figure(fig, spec)
    
    async def _render_meiosis(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render stages of meiosis (simplified).
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 10), dpi=self._get_dpi(spec))
        
        stages = [
            ('Prophase I', 'Crossing over'),
            ('Metaphase I', 'Homologous pairs align'),
            ('Anaphase I', 'Homologs separate'),
            ('Telophase I', 'Two cells form'),
            ('Prophase II', 'Chromosomes condense'),
            ('Metaphase II', 'Chromosomes align'),
            ('Anaphase II', 'Sisters separate'),
            ('Telophase II', 'Four haploid cells')
        ]
        
        for ax, (stage, desc) in zip(axes.flatten(), stages):
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{stage}\n{desc}', fontsize=9, fontweight='bold')
            
            # Draw cells and chromosomes based on stage
            if 'I' in stage and 'II' not in stage:
                # Meiosis I - draw single cell
                ax.add_patch(Ellipse((2, 2), 3, 3.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
                
                if stage == 'Prophase I':
                    # Homologous pairs with crossing over
                    ax.plot([1.5, 2.5], [2.5, 1.5], color='#FF0000', linewidth=3)
                    ax.plot([1.5, 2.5], [1.5, 2.5], color='#0000FF', linewidth=3)
                    ax.add_patch(Circle((2, 2), 0.1, facecolor='#FFD700'))  # Chiasma
                    
                elif stage == 'Metaphase I':
                    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
                    # Bivalents at plate
                    for x in [1.5, 2.5]:
                        ax.plot([x-0.1, x+0.1], [2.1, 1.9], color='#FF0000', linewidth=3)
                        ax.plot([x-0.15, x+0.05], [2.15, 1.85], color='#0000FF', linewidth=2)
                        
                elif stage == 'Anaphase I':
                    # Homologs separating
                    ax.plot([1.5, 1.7], [2.8, 2.6], color='#FF0000', linewidth=3)
                    ax.plot([2.3, 2.5], [2.8, 2.6], color='#FF0000', linewidth=3)
                    ax.plot([1.5, 1.7], [1.2, 1.4], color='#0000FF', linewidth=3)
                    ax.plot([2.3, 2.5], [1.2, 1.4], color='#0000FF', linewidth=3)
                    
                elif stage == 'Telophase I':
                    # Two cells
                    ax.add_patch(Ellipse((1.3, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
                    ax.add_patch(Ellipse((2.7, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
            else:
                # Meiosis II - draw two/four cells
                if stage == 'Telophase II':
                    # Four cells
                    for pos in [(1, 2.8), (3, 2.8), (1, 1.2), (3, 1.2)]:
                        ax.add_patch(Ellipse(pos, 1.4, 1.4, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=1.5))
                        ax.add_patch(Circle(pos, 0.3, facecolor='#E6E6FA', edgecolor='#4B0082'))
                else:
                    # Two cells
                    ax.add_patch(Ellipse((1.3, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
                    ax.add_patch(Ellipse((2.7, 2), 1.8, 2.5, facecolor='#FFFACD', edgecolor='#8B4513', linewidth=2))
        
        fig.suptitle('Stages of Meiosis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Additional Diagrams (Simplified implementations)
    # =========================================================================
    
    async def _render_nephron(self, spec: Dict[str, Any]) -> RenderResult:
        """Render kidney nephron structure."""
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Bowman's capsule
        ax.add_patch(Circle((3, 7), 1.2, facecolor='#FFE4E1', edgecolor='#8B0000', linewidth=2))
        ax.add_patch(Circle((3, 7), 0.6, facecolor='#FF6347', edgecolor='#8B0000', linewidth=1))
        ax.text(3, 7, 'Glomerulus', ha='center', fontsize=8)
        
        # Proximal convoluted tubule
        pct_x = np.linspace(4.2, 7, 30)
        pct_y = 7 + 0.5 * np.sin(pct_x * 4)
        ax.plot(pct_x, pct_y, color='#FFB6C1', linewidth=8, solid_capstyle='round')
        ax.text(5.5, 8.2, 'Proximal Tubule', fontsize=9, ha='center')
        
        # Loop of Henle
        loop_x = [7, 8, 8, 7]
        loop_y = [7, 7, 2, 2]
        ax.plot(loop_x, loop_y, color='#ADD8E6', linewidth=6, solid_capstyle='round')
        ax.text(8.5, 4.5, 'Loop of\nHenle', fontsize=9, ha='left')
        
        # Ascending limb
        ax.plot([7, 6, 5], [2, 2, 4], color='#87CEEB', linewidth=6, solid_capstyle='round')
        
        # Distal convoluted tubule
        dct_x = np.linspace(5, 3, 30)
        dct_y = 4 + 0.4 * np.sin(dct_x * 5)
        ax.plot(dct_x, dct_y, color='#DDA0DD', linewidth=6, solid_capstyle='round')
        ax.text(4, 5, 'Distal Tubule', fontsize=9, ha='center')
        
        # Collecting duct
        ax.plot([3, 3], [3.5, 1], color='#D2691E', linewidth=8, solid_capstyle='round')
        ax.text(3, 0.5, 'Collecting\nDuct', fontsize=9, ha='center')
        
        ax.set_title('Nephron Structure', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
    
    async def _render_eye(self, spec: Dict[str, Any]) -> RenderResult:
        """Render human eye structure."""
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        center = (6, 5)
        
        # Sclera (white of eye)
        ax.add_patch(Ellipse(center, 8, 6, facecolor='#FFFAFA', edgecolor='#8B4513', linewidth=2))
        
        # Choroid
        ax.add_patch(Ellipse(center, 7.5, 5.5, facecolor='#8B0000', edgecolor='none', alpha=0.3))
        
        # Retina
        ax.add_patch(Ellipse(center, 7, 5, facecolor='#FFE4B5', edgecolor='none', alpha=0.5))
        
        # Vitreous humor
        ax.add_patch(Ellipse((6.5, 5), 5, 4, facecolor='#E0FFFF', edgecolor='none', alpha=0.5))
        
        # Lens
        ax.add_patch(Ellipse((4, 5), 1.2, 2, facecolor='#87CEEB', edgecolor='#4682B4', linewidth=1.5, alpha=0.7))
        
        # Cornea
        ax.add_patch(Arc((2.5, 5), 2, 3, angle=0, theta1=-60, theta2=60, color='#ADD8E6', linewidth=3))
        
        # Iris
        ax.add_patch(Wedge((3.2, 5), 1, 70, 110, facecolor='#6B8E23', edgecolor='#556B2F', linewidth=1))
        ax.add_patch(Wedge((3.2, 5), 1, 250, 290, facecolor='#6B8E23', edgecolor='#556B2F', linewidth=1))
        
        # Pupil
        ax.add_patch(Circle((3.2, 5), 0.4, facecolor='black'))
        
        # Optic nerve
        ax.add_patch(Rectangle((9.5, 4.5), 2, 1, facecolor='#FFD700', edgecolor='#B8860B', linewidth=1.5))
        
        # Labels
        ax.text(6, 2.5, 'Retina', fontsize=9, ha='center')
        ax.text(4, 7.5, 'Lens', fontsize=9, ha='center')
        ax.text(1.5, 5, 'Cornea', fontsize=9, ha='center')
        ax.text(3.2, 6.5, 'Iris', fontsize=9, ha='center')
        ax.text(11, 5, 'Optic\nNerve', fontsize=9, ha='center')
        
        ax.set_title('Human Eye Structure', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
    
    async def _render_ear(self, spec: Dict[str, Any]) -> RenderResult:
        """Render human ear structure."""
        fig, ax = self._create_figure(spec, figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Outer ear (pinna)
        pinna = Ellipse((2, 5), 2, 4, facecolor='#FFDAB9', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(pinna)
        ax.text(2, 8, 'Outer Ear\n(Pinna)', ha='center', fontsize=9)
        
        # Ear canal
        ax.add_patch(Rectangle((3, 4.5), 2, 1, facecolor='#FFE4B5', edgecolor='#8B4513', linewidth=1.5))
        ax.text(4, 6, 'Ear Canal', ha='center', fontsize=8)
        
        # Eardrum (tympanic membrane)
        ax.add_patch(Ellipse((5.2, 5), 0.2, 1.2, facecolor='#DEB887', edgecolor='#8B4513', linewidth=2))
        ax.text(5.2, 3.5, 'Eardrum', ha='center', fontsize=8)
        
        # Middle ear (ossicles)
        # Malleus, Incus, Stapes
        ax.plot([5.3, 6], [5.3, 5.5], color='#D2691E', linewidth=3)  # Malleus
        ax.add_patch(Circle((6, 5.5), 0.15, facecolor='#D2691E'))
        ax.plot([6, 7], [5.5, 5.3], color='#D2691E', linewidth=2)  # Incus
        ax.add_patch(Ellipse((7.2, 5.2), 0.3, 0.5, facecolor='#D2691E', edgecolor='#8B4513'))  # Stapes
        ax.text(6.5, 6.5, 'Ossicles', ha='center', fontsize=9)
        
        # Oval window
        ax.add_patch(Ellipse((7.5, 5), 0.2, 0.4, facecolor='#FFB6C1', edgecolor='#8B0000'))
        
        # Cochlea
        theta = np.linspace(0, 4 * np.pi, 100)
        r = 0.8 - theta / 20
        cx = 9 + r * np.cos(theta)
        cy = 5 + r * np.sin(theta)
        ax.plot(cx, cy, color='#FF69B4', linewidth=4)
        ax.text(9, 3, 'Cochlea', ha='center', fontsize=9)
        
        # Semicircular canals
        for angle, offset in [(0, 0), (45, 0.5), (-45, -0.5)]:
            arc = Arc((10, 6 + offset), 1.5, 1.5, angle=angle, theta1=0, theta2=180,
                     color='#9370DB', linewidth=3)
            ax.add_patch(arc)
        ax.text(10, 8, 'Semicircular\nCanals', ha='center', fontsize=9)
        
        # Auditory nerve
        ax.add_patch(FancyArrowPatch((10, 4), (12, 3), arrowstyle='->', mutation_scale=20,
                                    color='#FFD700', linewidth=3))
        ax.text(12, 2.5, 'Auditory\nNerve', ha='center', fontsize=9)
        
        ax.set_title('Human Ear Structure', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
    
    async def _render_digestive_system(self, spec: Dict[str, Any]) -> RenderResult:
        """Render human digestive system."""
        fig, ax = self._create_figure(spec, figsize=(10, 14))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Mouth
        ax.add_patch(Ellipse((5, 13), 1.5, 0.8, facecolor='#FFB6C1', edgecolor='#8B4513', linewidth=2))
        ax.text(5, 13, 'Mouth', ha='center', fontsize=9)
        
        # Esophagus
        ax.add_patch(Rectangle((4.7, 10), 0.6, 2.5, facecolor='#FFA07A', edgecolor='#8B4513'))
        ax.text(6, 11, 'Esophagus', fontsize=9)
        
        # Stomach
        stomach = Ellipse((5, 8.5), 2.5, 2, angle=-20, facecolor='#FF6347', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(stomach)
        ax.text(5, 8.5, 'Stomach', ha='center', fontsize=9, color='white')
        
        # Small intestine (coiled)
        for i in range(8):
            y = 6.5 - i * 0.4
            x_offset = 0.3 * (i % 2)
            ax.plot([3.5 + x_offset, 6.5 - x_offset], [y, y], color='#FFD700', linewidth=6)
        ax.text(7.5, 5, 'Small\nIntestine', fontsize=9)
        
        # Large intestine
        li_x = [2, 2, 3, 5, 7, 8, 8, 7, 5, 5]
        li_y = [2, 4, 5.5, 5.5, 5.5, 4, 2, 1.5, 1.5, 1]
        ax.plot(li_x, li_y, color='#8B4513', linewidth=10, solid_capstyle='round')
        ax.text(1, 3, 'Large\nIntestine', fontsize=9)
        
        # Liver
        ax.add_patch(Polygon([(2, 9), (4, 10), (4.5, 9), (3.5, 8)], closed=True,
                            facecolor='#8B0000', edgecolor='#4B0000', linewidth=2))
        ax.text(2.5, 8.5, 'Liver', fontsize=9, color='white')
        
        # Rectum
        ax.add_patch(Rectangle((4.7, 0.5), 0.6, 0.8, facecolor='#8B4513', edgecolor='#5C4033'))
        ax.text(5, 0.3, 'Rectum', ha='center', fontsize=8)
        
        ax.set_title('Human Digestive System', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
    
    async def _render_respiratory_system(self, spec: Dict[str, Any]) -> RenderResult:
        """Render human respiratory system."""
        fig, ax = self._create_figure(spec, figsize=(10, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Nasal cavity
        ax.add_patch(FancyBboxPatch((4, 10), 2, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#FFB6C1', edgecolor='#8B4513', linewidth=2))
        ax.text(5, 10.75, 'Nasal\nCavity', ha='center', fontsize=8)
        
        # Pharynx
        ax.add_patch(Rectangle((4.5, 8.5), 1, 1.5, facecolor='#FFA07A', edgecolor='#8B4513'))
        ax.text(6, 9, 'Pharynx', fontsize=8)
        
        # Larynx
        ax.add_patch(Polygon([(4.5, 8.5), (5.5, 8.5), (5.3, 7.5), (4.7, 7.5)],
                            facecolor='#DEB887', edgecolor='#8B4513', linewidth=2))
        ax.text(6, 8, 'Larynx', fontsize=8)
        
        # Trachea
        ax.add_patch(Rectangle((4.6, 5), 0.8, 2.5, facecolor='#87CEEB', edgecolor='#4682B4', linewidth=2))
        # Tracheal rings
        for y in np.arange(5.2, 7.3, 0.3):
            ax.plot([4.6, 5.4], [y, y], color='#4682B4', linewidth=1)
        ax.text(6, 6, 'Trachea', fontsize=8)
        
        # Bronchi
        ax.plot([5, 3.5], [5, 4], color='#87CEEB', linewidth=6)
        ax.plot([5, 6.5], [5, 4], color='#87CEEB', linewidth=6)
        
        # Left lung
        left_lung = Polygon([(1, 1), (1, 4.5), (3.5, 5), (4.2, 4), (4.2, 1.5), (2, 1)],
                           closed=True, facecolor='#FFB6C1', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(left_lung)
        ax.text(2.5, 3, 'Left\nLung', ha='center', fontsize=10)
        
        # Right lung
        right_lung = Polygon([(9, 1), (9, 4.5), (6.5, 5), (5.8, 4), (5.8, 1.5), (8, 1)],
                            closed=True, facecolor='#FFB6C1', edgecolor='#8B4513', linewidth=2)
        ax.add_patch(right_lung)
        ax.text(7.5, 3, 'Right\nLung', ha='center', fontsize=10)
        
        # Bronchioles in lungs
        for lung_x in [2.5, 7.5]:
            for _ in range(5):
                x = lung_x + np.random.uniform(-1, 1)
                y = 2.5 + np.random.uniform(-0.5, 0.5)
                ax.add_patch(Circle((x, y), 0.15, facecolor='#FF69B4', edgecolor='#8B4513'))
        
        # Diaphragm
        diaphragm_x = np.linspace(1, 9, 50)
        diaphragm_y = 1 + 0.3 * np.sin((diaphragm_x - 5) * 0.8)
        ax.fill_between(diaphragm_x, 0, diaphragm_y, facecolor='#DEB887', edgecolor='#8B4513', linewidth=2)
        ax.text(5, 0.5, 'Diaphragm', ha='center', fontsize=10)
        
        ax.set_title('Human Respiratory System', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
    
    async def _render_flower(self, spec: Dict[str, Any]) -> RenderResult:
        """Render flower structure."""
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        center = (6, 5)
        
        # Petals
        petal_colors = ['#FF69B4', '#FF1493', '#FF69B4', '#FF1493', '#FF69B4']
        for i, color in enumerate(petal_colors):
            angle = i * 72 + 36
            px = center[0] + 2 * np.cos(np.radians(angle))
            py = center[1] + 2 * np.sin(np.radians(angle))
            petal = Ellipse((px, py), 1.5, 3, angle=angle - 90,
                           facecolor=color, edgecolor='#C71585', linewidth=1.5, alpha=0.8)
            ax.add_patch(petal)
        ax.text(8.5, 7.5, 'Petals', fontsize=10)
        ax.annotate('', xy=(7.5, 6.5), xytext=(8.3, 7.3),
                   arrowprops=dict(arrowstyle='-', color='gray'))
        
        # Sepals
        for i in range(5):
            angle = i * 72
            sx = center[0] + 1.8 * np.cos(np.radians(angle))
            sy = center[1] + 1.8 * np.sin(np.radians(angle))
            sepal = Ellipse((sx, sy), 0.8, 2, angle=angle - 90,
                           facecolor='#228B22', edgecolor='#006400', linewidth=1)
            ax.add_patch(sepal)
        ax.text(9, 4, 'Sepals', fontsize=10)
        
        # Center of flower
        ax.add_patch(Circle(center, 1, facecolor='#FFD700', edgecolor='#B8860B', linewidth=2))
        
        # Stamens (male parts)
        for i in range(6):
            angle = i * 60
            fx = center[0] + 0.6 * np.cos(np.radians(angle))
            fy = center[1] + 0.6 * np.sin(np.radians(angle))
            # Filament
            ax.plot([fx, fx + 0.3 * np.cos(np.radians(angle))], 
                   [fy, fy + 0.3 * np.sin(np.radians(angle))], 
                   color='#FFD700', linewidth=2)
            # Anther
            ax.add_patch(Ellipse((fx + 0.4 * np.cos(np.radians(angle)), 
                                 fy + 0.4 * np.sin(np.radians(angle))),
                                0.2, 0.3, angle=angle, facecolor='#FFA500'))
        ax.text(1, 6, 'Stamens\n(Anther +\nFilament)', fontsize=9)
        ax.annotate('', xy=(5.5, 5.5), xytext=(2, 6),
                   arrowprops=dict(arrowstyle='-', color='gray'))
        
        # Pistil (female part) in center
        ax.add_patch(Circle(center, 0.25, facecolor='#90EE90', edgecolor='#006400', linewidth=2))
        ax.plot([center[0], center[0]], [center[1] + 0.25, center[1] + 0.8], color='#90EE90', linewidth=3)
        ax.add_patch(Circle((center[0], center[1] + 0.9), 0.15, facecolor='#32CD32'))
        ax.text(1, 4, 'Pistil\n(Stigma,\nStyle,\nOvary)', fontsize=9)
        ax.annotate('', xy=(5.8, 5.3), xytext=(2, 4.5),
                   arrowprops=dict(arrowstyle='-', color='gray'))
        
        # Stem
        ax.add_patch(Rectangle((5.8, 0), 0.4, 3.5, facecolor='#228B22', edgecolor='#006400'))
        ax.text(6, -0.3, 'Stem', ha='center', fontsize=10)
        
        # Receptacle
        ax.add_patch(Ellipse((6, 3.5), 1.5, 0.8, facecolor='#90EE90', edgecolor='#228B22', linewidth=2))
        ax.text(8, 3.5, 'Receptacle', fontsize=9)
        
        ax.set_title('Flower Structure', fontsize=14, fontweight='bold', pad=15)
        return self._save_figure(fig, spec)
