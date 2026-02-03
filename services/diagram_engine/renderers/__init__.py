"""
Diagram Renderers Package

Subject-specific renderers for generating accurate educational diagrams.

Available Renderers:
- MathRenderer: Graphs, geometry, Venn diagrams, charts (matplotlib)
- PhysicsRenderer: Circuits, ray diagrams, FBDs, wave diagrams (schemdraw/matplotlib)
- ChemistryRenderer: Molecules, reactions, lab apparatus (rdkit/matplotlib)
- BiologyRenderer: Anatomical diagrams, cell structures (matplotlib/SVG)
"""

from .math_renderer import MathRenderer
from .physics_renderer import PhysicsRenderer
from .chemistry_renderer import ChemistryRenderer
from .biology_renderer import BiologyRenderer

__all__ = [
    "MathRenderer",
    "PhysicsRenderer",
    "ChemistryRenderer",
    "BiologyRenderer",
]
