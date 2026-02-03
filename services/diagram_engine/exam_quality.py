"""
Exam Quality Standards for Diagrams

This module defines quality settings for JEE/NEET/CBSE exam-ready diagrams.

Key requirements:
1. High contrast (black on white)
2. Clean vector lines
3. No overlapping labels
4. Readable font sizes
5. Proper spacing
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExamQualitySettings:
    """
    Quality settings for exam-ready diagrams.

    These settings ensure diagrams are:
    - Readable when printed
    - High contrast for clarity
    - Properly sized for exam papers
    """

    # Colors - High contrast
    background_color: str = "#ffffff"  # Pure white
    primary_color: str = "#000000"     # Pure black for lines
    secondary_color: str = "#333333"   # Dark gray for less emphasis
    accent_color: str = "#0066CC"      # Blue for highlights
    error_color: str = "#CC0000"       # Red for warnings

    # Standard colors for physics diagrams
    force_color: str = "#CC0000"       # Red for forces
    velocity_color: str = "#0066CC"    # Blue for velocity
    normal_color: str = "#009900"      # Green for normal force
    friction_color: str = "#FF6600"    # Orange for friction

    # Font settings
    min_font_size: int = 12
    default_font_size: int = 14
    label_font_size: int = 14
    title_font_size: int = 16
    font_family: str = "DejaVu Sans"  # Good unicode support
    font_weight: str = "normal"

    # Line settings
    min_line_width: float = 1.5
    default_line_width: float = 2.0
    axis_line_width: float = 1.0
    arrow_line_width: float = 2.0
    arrow_head_width: float = 0.3
    arrow_head_length: float = 0.15

    # Spacing
    label_padding: float = 0.3        # Padding around labels
    element_spacing: float = 1.0      # Min spacing between elements
    margin: float = 0.5               # Margin from edges

    # DPI settings
    print_dpi: int = 300              # For print quality
    screen_dpi: int = 150             # For screen display

    # Size settings (in inches at print_dpi)
    default_width: float = 6.0
    default_height: float = 4.5
    max_width: float = 8.0
    max_height: float = 6.0

    def get_matplotlib_rcparams(self) -> Dict[str, Any]:
        """
        Get matplotlib rcParams for exam quality.

        Apply these at the start of rendering:
            import matplotlib.pyplot as plt
            plt.rcParams.update(settings.get_matplotlib_rcparams())
        """
        return {
            # Font settings
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_family, 'Arial', 'Helvetica'],
            'font.size': self.default_font_size,
            'font.weight': self.font_weight,

            # Axes settings
            'axes.linewidth': self.axis_line_width,
            'axes.edgecolor': self.primary_color,
            'axes.facecolor': self.background_color,
            'axes.labelsize': self.label_font_size,
            'axes.titlesize': self.title_font_size,
            'axes.labelcolor': self.primary_color,
            'axes.titleweight': 'bold',

            # Line settings
            'lines.linewidth': self.default_line_width,
            'lines.color': self.primary_color,
            'lines.antialiased': True,

            # Tick settings
            'xtick.labelsize': self.min_font_size,
            'ytick.labelsize': self.min_font_size,
            'xtick.color': self.secondary_color,
            'ytick.color': self.secondary_color,

            # Figure settings
            'figure.facecolor': self.background_color,
            'figure.edgecolor': self.background_color,
            'figure.dpi': self.print_dpi,
            'figure.figsize': [self.default_width, self.default_height],

            # Save settings
            'savefig.dpi': self.print_dpi,
            'savefig.facecolor': self.background_color,
            'savefig.edgecolor': self.background_color,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,

            # Grid (off by default for exam diagrams)
            'axes.grid': False,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,

            # Legend
            'legend.fontsize': self.min_font_size,
            'legend.frameon': False,

            # Text
            'text.color': self.primary_color,
            'text.antialiased': True,
        }

    def get_schemdraw_settings(self) -> Dict[str, Any]:
        """Get settings for schemdraw circuit diagrams."""
        return {
            'unit': 3,
            'fontsize': self.default_font_size,
            'font': self.font_family,
            'color': self.primary_color,
            'fill': self.background_color,
            'lw': self.default_line_width,
        }

    def get_rdkit_settings(self) -> Dict[str, Any]:
        """Get settings for RDKit molecular diagrams."""
        return {
            'atomLabelFontSize': self.default_font_size,
            'bondLineWidth': self.default_line_width,
            'padding': 0.1,
            'backgroundColour': (1, 1, 1),  # White
        }


@dataclass
class LabelPlacement:
    """
    Helper for non-overlapping label placement.

    Tracks placed labels and finds non-overlapping positions.
    """
    placed_labels: List[Dict[str, Any]] = field(default_factory=list)
    min_distance: float = 0.5

    def add_label(
        self,
        x: float,
        y: float,
        text: str,
        width: float = 1.0,
        height: float = 0.3
    ) -> Dict[str, float]:
        """
        Add a label, adjusting position to avoid overlaps.

        Returns adjusted position.
        """
        original_pos = {'x': x, 'y': y}

        # Check for overlaps and adjust
        adjusted = self._find_non_overlapping(x, y, width, height)

        self.placed_labels.append({
            'x': adjusted['x'],
            'y': adjusted['y'],
            'width': width,
            'height': height,
            'text': text,
        })

        if adjusted != original_pos:
            logger.debug(f"Label '{text}' adjusted from {original_pos} to {adjusted}")

        return adjusted

    def _find_non_overlapping(
        self,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> Dict[str, float]:
        """Find a non-overlapping position near (x, y)."""
        # Try original position first
        if not self._overlaps(x, y, width, height):
            return {'x': x, 'y': y}

        # Try positions in expanding circles
        for radius in [0.3, 0.5, 0.8, 1.0, 1.5]:
            for angle in range(0, 360, 45):
                import math
                dx = radius * math.cos(math.radians(angle))
                dy = radius * math.sin(math.radians(angle))

                new_x = x + dx
                new_y = y + dy

                if not self._overlaps(new_x, new_y, width, height):
                    return {'x': new_x, 'y': new_y}

        # Fall back to offset below
        return {'x': x, 'y': y - height - self.min_distance}

    def _overlaps(
        self,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> bool:
        """Check if a box at (x, y) overlaps any placed labels."""
        for label in self.placed_labels:
            # Simple AABB overlap check
            if (abs(x - label['x']) < (width + label['width']) / 2 + self.min_distance and
                abs(y - label['y']) < (height + label['height']) / 2 + self.min_distance):
                return True
        return False

    def clear(self):
        """Clear all placed labels."""
        self.placed_labels = []


def apply_exam_quality(ax, settings: Optional[ExamQualitySettings] = None):
    """
    Apply exam quality settings to a matplotlib axes.

    Args:
        ax: matplotlib Axes object
        settings: ExamQualitySettings (uses defaults if None)
    """
    if settings is None:
        settings = ExamQualitySettings()

    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color(settings.primary_color)
        spine.set_linewidth(settings.axis_line_width)

    # Set tick colors
    ax.tick_params(
        colors=settings.secondary_color,
        labelsize=settings.min_font_size
    )

    # Set background
    ax.set_facecolor(settings.background_color)


def get_exam_style_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add exam-quality style settings to a diagram spec.

    Args:
        spec: Original diagram specification

    Returns:
        Spec with exam quality settings added
    """
    settings = ExamQualitySettings()

    # Merge style settings
    style = spec.get('style', {})
    exam_style = {
        'background_color': settings.background_color,
        'line_color': settings.primary_color,
        'font_family': settings.font_family,
        'font_size': settings.default_font_size,
        'line_width': settings.default_line_width,
    }

    # User settings override defaults
    merged_style = {**exam_style, **style}

    result = spec.copy()
    result['style'] = merged_style
    result['quality'] = 'high'

    # Ensure dimensions are reasonable
    dimensions = result.get('dimensions', {})
    if 'width' not in dimensions:
        dimensions['width'] = int(settings.default_width * settings.print_dpi)
    if 'height' not in dimensions:
        dimensions['height'] = int(settings.default_height * settings.print_dpi)

    result['dimensions'] = dimensions

    return result


# Default singleton
_default_settings: Optional[ExamQualitySettings] = None


def get_exam_settings() -> ExamQualitySettings:
    """Get the default exam quality settings."""
    global _default_settings
    if _default_settings is None:
        _default_settings = ExamQualitySettings()
    return _default_settings


# Preset configurations for different exam types
EXAM_PRESETS = {
    'jee': ExamQualitySettings(
        default_font_size=14,
        label_font_size=14,
        min_line_width=1.5,
        default_line_width=2.0,
    ),
    'neet': ExamQualitySettings(
        default_font_size=16,  # Larger for biology diagrams
        label_font_size=16,
        min_line_width=2.0,
        default_line_width=2.5,
    ),
    'cbse': ExamQualitySettings(
        default_font_size=14,
        label_font_size=14,
        min_line_width=1.5,
        default_line_width=2.0,
    ),
}


def get_preset(exam_type: str) -> ExamQualitySettings:
    """Get exam-specific quality settings."""
    return EXAM_PRESETS.get(exam_type.lower(), get_exam_settings())
