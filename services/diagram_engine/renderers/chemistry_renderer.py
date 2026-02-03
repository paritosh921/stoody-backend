"""
Chemistry Diagram Renderer

Renders chemistry diagrams using RDKit for molecular structures
and matplotlib for lab setups, orbital diagrams, and other diagrams.

Supported diagram types:
- molecule_2d: 2D molecular structure
- molecule_3d: 3D molecular structure (with rotation)
- reaction_scheme: Chemical reaction with reactants and products
- lab_setup_titration: Titration apparatus setup
- lab_setup_distillation: Distillation apparatus setup
- lab_setup_electrolysis: Electrolysis apparatus setup
- periodic_table_section: Highlighted section of periodic table
- orbital_diagram: Electron orbital diagrams
- crystal_structure: Crystal lattice structures
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
        Circle, Rectangle, Polygon, FancyArrowPatch, Arc,
        Ellipse, PathPatch, FancyBboxPatch, Wedge
    )
    from matplotlib.collections import PatchCollection
    from matplotlib.path import Path
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# RDKit for molecular structures
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from ..base_renderer import BaseRenderer, RenderResult, RenderError
from ..specs.base_spec import (
    DiagramSubject,
    OutputFormat,
    SUPPORTED_DIAGRAM_TYPES,
)

logger = logging.getLogger(__name__)


# Periodic table data for element rendering
PERIODIC_TABLE = {
    1: {'symbol': 'H', 'name': 'Hydrogen', 'group': 1, 'period': 1, 'category': 'nonmetal'},
    2: {'symbol': 'He', 'name': 'Helium', 'group': 18, 'period': 1, 'category': 'noble_gas'},
    3: {'symbol': 'Li', 'name': 'Lithium', 'group': 1, 'period': 2, 'category': 'alkali_metal'},
    4: {'symbol': 'Be', 'name': 'Beryllium', 'group': 2, 'period': 2, 'category': 'alkaline_earth'},
    5: {'symbol': 'B', 'name': 'Boron', 'group': 13, 'period': 2, 'category': 'metalloid'},
    6: {'symbol': 'C', 'name': 'Carbon', 'group': 14, 'period': 2, 'category': 'nonmetal'},
    7: {'symbol': 'N', 'name': 'Nitrogen', 'group': 15, 'period': 2, 'category': 'nonmetal'},
    8: {'symbol': 'O', 'name': 'Oxygen', 'group': 16, 'period': 2, 'category': 'nonmetal'},
    9: {'symbol': 'F', 'name': 'Fluorine', 'group': 17, 'period': 2, 'category': 'halogen'},
    10: {'symbol': 'Ne', 'name': 'Neon', 'group': 18, 'period': 2, 'category': 'noble_gas'},
    11: {'symbol': 'Na', 'name': 'Sodium', 'group': 1, 'period': 3, 'category': 'alkali_metal'},
    12: {'symbol': 'Mg', 'name': 'Magnesium', 'group': 2, 'period': 3, 'category': 'alkaline_earth'},
    13: {'symbol': 'Al', 'name': 'Aluminium', 'group': 13, 'period': 3, 'category': 'post_transition'},
    14: {'symbol': 'Si', 'name': 'Silicon', 'group': 14, 'period': 3, 'category': 'metalloid'},
    15: {'symbol': 'P', 'name': 'Phosphorus', 'group': 15, 'period': 3, 'category': 'nonmetal'},
    16: {'symbol': 'S', 'name': 'Sulfur', 'group': 16, 'period': 3, 'category': 'nonmetal'},
    17: {'symbol': 'Cl', 'name': 'Chlorine', 'group': 17, 'period': 3, 'category': 'halogen'},
    18: {'symbol': 'Ar', 'name': 'Argon', 'group': 18, 'period': 3, 'category': 'noble_gas'},
    19: {'symbol': 'K', 'name': 'Potassium', 'group': 1, 'period': 4, 'category': 'alkali_metal'},
    20: {'symbol': 'Ca', 'name': 'Calcium', 'group': 2, 'period': 4, 'category': 'alkaline_earth'},
    26: {'symbol': 'Fe', 'name': 'Iron', 'group': 8, 'period': 4, 'category': 'transition_metal'},
    29: {'symbol': 'Cu', 'name': 'Copper', 'group': 11, 'period': 4, 'category': 'transition_metal'},
    30: {'symbol': 'Zn', 'name': 'Zinc', 'group': 12, 'period': 4, 'category': 'transition_metal'},
    35: {'symbol': 'Br', 'name': 'Bromine', 'group': 17, 'period': 4, 'category': 'halogen'},
    47: {'symbol': 'Ag', 'name': 'Silver', 'group': 11, 'period': 5, 'category': 'transition_metal'},
    53: {'symbol': 'I', 'name': 'Iodine', 'group': 17, 'period': 5, 'category': 'halogen'},
    79: {'symbol': 'Au', 'name': 'Gold', 'group': 11, 'period': 6, 'category': 'transition_metal'},
}

# Category colors for periodic table
CATEGORY_COLORS = {
    'alkali_metal': '#FF6B6B',
    'alkaline_earth': '#FFA94D',
    'transition_metal': '#FFE066',
    'post_transition': '#69DB7C',
    'metalloid': '#4DABF7',
    'nonmetal': '#B197FC',
    'halogen': '#FF8CC8',
    'noble_gas': '#99E9F2',
    'lanthanide': '#F8B4D9',
    'actinide': '#FDB4C8',
}


class ChemistryRenderer(BaseRenderer):
    """
    Renderer for chemistry diagrams.
    
    Uses RDKit for molecular structures when available,
    with matplotlib fallback for all diagram types.
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for ChemistryRenderer. "
                "Install with: pip install matplotlib"
            )
        
        self._has_rdkit = HAS_RDKIT
        if not HAS_RDKIT:
            logger.warning(
                "RDKit not installed. Molecular diagrams will use matplotlib fallback. "
                "For better molecular rendering, install RDKit: pip install rdkit"
            )
    
    @property
    def subject(self) -> DiagramSubject:
        return DiagramSubject.CHEMISTRY
    
    def get_supported_types(self) -> List[str]:
        return SUPPORTED_DIAGRAM_TYPES.get(DiagramSubject.CHEMISTRY, [])
    
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render a chemistry diagram based on specification.
        """
        # Validate spec
        is_valid, error = self.validate_spec(spec)
        if not is_valid:
            raise RenderError(error, spec.get('diagram_type', 'unknown'))
        
        # Apply default styles
        spec = self._apply_style(spec)
        
        diagram_type = spec['diagram_type']
        
        # Route to appropriate renderer
        render_methods = {
            'molecule_2d': self._render_molecule_2d,
            'molecule_3d': self._render_molecule_3d,
            'reaction_scheme': self._render_reaction_scheme,
            'lab_setup_titration': self._render_lab_titration,
            'lab_setup_distillation': self._render_lab_distillation,
            'lab_setup_electrolysis': self._render_lab_electrolysis,
            'lab_setup_manometer': self._render_lab_manometer,
            'periodic_table_section': self._render_periodic_table,
            'orbital_diagram': self._render_orbital_diagram,
            'crystal_structure': self._render_crystal_structure,
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
    
    def _extract_reaction_components(self, spec: Dict[str, Any], component_type: str) -> List[str]:
        """
        Extract reactants or products from spec, checking multiple possible sources.

        The plan may store these under various keys:
        - 'reactants'/'products' directly
        - 'parameters.reactants'/'parameters.products'
        - 'extracted_values.reactants'/'extracted_values.products'
        - 'equation' or 'reaction' that needs parsing
        - 'formula' for simple reactions
        - Individual keys like 'reactant_1', 'product_a', etc.

        Args:
            spec: The diagram specification
            component_type: Either 'reactants' or 'products'

        Returns:
            List of component formulas (e.g., ['H2', 'O2'] for reactants)
        """
        import re

        # 1. Check direct key
        direct = spec.get(component_type)
        if direct and isinstance(direct, list) and len(direct) > 0:
            # Filter out placeholder values
            filtered = [r for r in direct if r not in ['A', 'B', 'C', '?', '']]
            if filtered:
                return filtered

        # 2. Check under 'parameters'
        params = spec.get('parameters', {})
        if isinstance(params, dict):
            param_val = params.get(component_type)
            if param_val and isinstance(param_val, list):
                filtered = [r for r in param_val if r not in ['A', 'B', 'C', '?', '']]
                if filtered:
                    return filtered

        # 3. Check under 'extracted_values'
        extracted = spec.get('extracted_values', {})
        if isinstance(extracted, dict):
            ext_val = extracted.get(component_type)
            if ext_val:
                if isinstance(ext_val, list):
                    filtered = [r for r in ext_val if r not in ['A', 'B', 'C', '?', '']]
                    if filtered:
                        return filtered
                elif isinstance(ext_val, str):
                    # Might be comma-separated or plus-separated
                    parts = re.split(r'[,+]', ext_val)
                    parts = [p.strip() for p in parts if p.strip() and p.strip() not in ['A', 'B', 'C', '?']]
                    if parts:
                        return parts

        # 4. Look for numbered keys like 'reactant_1', 'reactant_2', 'product_1', etc.
        singular = component_type.rstrip('s')  # 'reactants' -> 'reactant', 'products' -> 'product'
        numbered = []
        for key in list(spec.keys()) + list(params.keys()) + list(extracted.keys()):
            if singular in key.lower():
                val = spec.get(key) or params.get(key) or extracted.get(key)
                if val and isinstance(val, str) and val not in ['A', 'B', 'C', '?', '']:
                    numbered.append(val)
        if numbered:
            return numbered

        # 5. Try to parse from 'equation' or 'reaction' string
        equation = spec.get('equation') or spec.get('reaction') or params.get('equation') or extracted.get('equation')
        if equation and isinstance(equation, str):
            # Split by arrow patterns: ->, →, ⟶, =, ⇌
            arrow_pattern = r'(?:->|→|⟶|=|⇌|-->)'
            parts = re.split(arrow_pattern, equation)
            if len(parts) >= 2:
                left_side = parts[0].strip()
                right_side = parts[-1].strip()

                # Split each side by + and clean up
                if component_type == 'reactants':
                    components = re.split(r'\s*\+\s*', left_side)
                else:
                    components = re.split(r'\s*\+\s*', right_side)

                # Clean up: remove coefficients, whitespace
                cleaned = []
                for comp in components:
                    # Remove leading coefficient (e.g., "2H2O" -> "H2O")
                    cleaned_comp = re.sub(r'^\d+\s*', '', comp.strip())
                    if cleaned_comp and cleaned_comp not in ['A', 'B', 'C', '?', '']:
                        cleaned.append(cleaned_comp)
                if cleaned:
                    return cleaned

        # 6. Check for 'formula' key (for simple single-component reactions)
        formula = spec.get('formula') or params.get('formula') or extracted.get('formula')
        if formula and isinstance(formula, str) and component_type == 'products':
            # If only formula provided, assume it's the product
            if formula not in ['A', 'B', 'C', '?', '']:
                return [formula]

        # 7. Look through labels for chemical formulas
        labels = spec.get('labels', [])
        if isinstance(labels, list):
            formulas_found = []
            for label in labels:
                text = label.get('text', '') if isinstance(label, dict) else str(label)
                # Match chemical formula patterns (e.g., H2O, NaCl, C6H12O6)
                formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'
                matches = re.findall(formula_pattern, text)
                for match in matches:
                    # Filter out single letters and common non-formula words
                    if len(match) > 1 and match not in ['A', 'B', 'C', 'The', 'For', 'And']:
                        formulas_found.append(match)
            if formulas_found:
                # Return first half as reactants, second half as products (rough heuristic)
                mid = len(formulas_found) // 2 or 1
                if component_type == 'reactants':
                    return formulas_found[:mid]
                else:
                    return formulas_found[mid:]

        # 8. Check question_text for chemical formulas as last resort
        question = spec.get('question_text', '')
        if question:
            # Look for patterns like "reaction of X with Y" or "X + Y -> Z"
            formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b'
            all_formulas = re.findall(formula_pattern, question)
            # Filter duplicates while preserving order
            seen = set()
            unique_formulas = []
            for f in all_formulas:
                if f not in seen and f not in ['A', 'B', 'C', '?', '']:
                    seen.add(f)
                    unique_formulas.append(f)

            if len(unique_formulas) >= 2:
                # Assume first ones are reactants, last ones are products
                mid = len(unique_formulas) // 2 or 1
                if component_type == 'reactants':
                    return unique_formulas[:mid]
                else:
                    return unique_formulas[mid:]

        # Return empty list if nothing found
        return []

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
        """Save figure to bytes and return RenderResult."""
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
    
    # =========================================================================
    # Molecule Renderers
    # =========================================================================
    
    async def _render_molecule_2d(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render 2D molecular structure.
        
        Spec parameters:
            - smiles: SMILES string for the molecule (e.g., "CCO" for ethanol)
            - molecule_name: Optional name to display
            - show_hydrogens: Whether to show hydrogen atoms (default: False)
            - atom_labels: Whether to show atom labels (default: True)
            - highlight_atoms: List of atom indices to highlight
            - highlight_color: Color for highlighted atoms
        """
        smiles = spec.get('smiles', 'C')
        molecule_name = spec.get('molecule_name', '')
        show_hydrogens = spec.get('show_hydrogens', False)
        highlight_atoms = spec.get('highlight_atoms', [])
        highlight_color = spec.get('highlight_color', '#FFD700')
        
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 400)
        height = dimensions.get('height', 400)
        
        if self._has_rdkit:
            return await self._render_molecule_rdkit(
                smiles, molecule_name, show_hydrogens,
                highlight_atoms, highlight_color,
                width, height, spec
            )
        else:
            return await self._render_molecule_matplotlib(
                smiles, molecule_name, spec
            )
    
    async def _render_molecule_rdkit(
        self,
        smiles: str,
        molecule_name: str,
        show_hydrogens: bool,
        highlight_atoms: List[int],
        highlight_color: str,
        width: int,
        height: int,
        spec: Dict[str, Any]
    ) -> RenderResult:
        """Render molecule using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RenderError(f"Invalid SMILES: {smiles}", 'molecule_2d')
        
        if show_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        
        # Draw with highlights if specified
        if highlight_atoms:
            highlight_color_rgb = mcolors.to_rgb(highlight_color)
            highlight_colors = {i: highlight_color_rgb for i in highlight_atoms}
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        image_data = drawer.GetDrawingText()
        
        # If molecule name provided, add it
        if molecule_name:
            # We'd need to overlay text - for simplicity, include in metadata
            pass
        
        return RenderResult(
            image_data=image_data,
            format=OutputFormat.PNG,
            width=width,
            height=height,
            metadata={
                'diagram_type': 'molecule_2d',
                'smiles': smiles,
                'molecule_name': molecule_name,
                'renderer': 'rdkit'
            }
        )
    
    async def _render_molecule_matplotlib(
        self,
        smiles: str,
        molecule_name: str,
        spec: Dict[str, Any]
    ) -> RenderResult:
        """Fallback molecule rendering using matplotlib (simplified)."""
        fig, ax = self._create_figure(spec, figsize=(6, 6))
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Simple parsing for common molecules
        # This is a very simplified fallback
        atom_positions = self._parse_simple_smiles(smiles)
        
        # Draw bonds
        for i, (atom1, pos1) in enumerate(atom_positions):
            for j, (atom2, pos2) in enumerate(atom_positions[i+1:], i+1):
                dist = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
                if dist < 1.5:  # Assume bonded if close
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           'k-', linewidth=2, zorder=1)
        
        # Draw atoms
        atom_colors = {
            'C': '#404040', 'H': '#FFFFFF', 'O': '#FF0000',
            'N': '#0000FF', 'S': '#FFFF00', 'Cl': '#00FF00',
            'Br': '#A52A2A', 'F': '#90EE90', 'P': '#FFA500'
        }
        
        for atom, pos in atom_positions:
            color = atom_colors.get(atom, '#808080')
            circle = Circle(pos, 0.3, facecolor=color, edgecolor='black', 
                          linewidth=1, zorder=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], atom, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white' if atom == 'C' else 'black',
                   zorder=3)
        
        # Add molecule name
        if molecule_name:
            ax.set_title(molecule_name, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"SMILES: {smiles}", fontsize=10)
        
        return self._save_figure(fig, spec)
    
    def _parse_simple_smiles(self, smiles: str) -> List[Tuple[str, Tuple[float, float]]]:
        """
        Very simple SMILES parser for common molecules.
        Returns list of (atom, position) tuples.
        """
        atoms = []
        x, y = 0.0, 0.0
        angle = 0
        
        i = 0
        while i < len(smiles):
            char = smiles[i]
            
            if char.isupper():
                # Check for two-letter elements
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    atom = smiles[i:i+2]
                    i += 2
                else:
                    atom = char
                    i += 1
                
                atoms.append((atom, (x, y)))
                
                # Move to next position
                angle += 60
                x += math.cos(math.radians(angle))
                y += math.sin(math.radians(angle))
            else:
                i += 1
        
        return atoms if atoms else [('C', (0, 0))]
    
    async def _render_molecule_3d(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render 3D molecular structure.
        
        Spec parameters:
            - smiles: SMILES string for the molecule
            - molecule_name: Optional name to display
            - rotation: Tuple of (x, y, z) rotation angles
            - show_bonds: Whether to show bonds (default: True)
        """
        smiles = spec.get('smiles', 'C')
        molecule_name = spec.get('molecule_name', '')
        rotation = spec.get('rotation', (30, 45, 0))
        
        fig = plt.figure(figsize=(8, 8), dpi=self._get_dpi(spec))
        ax = fig.add_subplot(111, projection='3d')
        
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Get 3D coordinates
        if self._has_rdkit:
            coords = self._get_3d_coords_rdkit(smiles)
        else:
            coords = self._get_3d_coords_simple(smiles)
        
        if not coords:
            raise RenderError(f"Could not generate 3D coordinates for: {smiles}", 'molecule_3d')
        
        # Atom colors and sizes
        atom_colors = {
            'C': '#404040', 'H': '#FFFFFF', 'O': '#FF0000',
            'N': '#0000FF', 'S': '#FFFF00', 'Cl': '#00FF00',
            'Br': '#A52A2A', 'F': '#90EE90', 'P': '#FFA500'
        }
        
        # Draw atoms
        for atom, (x, y, z) in coords:
            color = atom_colors.get(atom, '#808080')
            ax.scatter([x], [y], [z], c=[color], s=500, edgecolors='black', linewidths=1)
        
        # Draw bonds (connect nearby atoms)
        for i, (atom1, pos1) in enumerate(coords):
            for j, (atom2, pos2) in enumerate(coords[i+1:], i+1):
                dist = math.sqrt(
                    (pos1[0]-pos2[0])**2 + 
                    (pos1[1]-pos2[1])**2 + 
                    (pos1[2]-pos2[2])**2
                )
                if dist < 2.0:  # Bond threshold
                    ax.plot3D(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        [pos1[2], pos2[2]],
                        'k-', linewidth=2
                    )
        
        # Set view angle
        ax.view_init(elev=rotation[0], azim=rotation[1])
        
        # Remove axes
        ax.set_axis_off()
        
        if molecule_name:
            ax.set_title(molecule_name, fontsize=14, fontweight='bold')
        
        return self._save_figure(fig, spec)
    
    def _get_3d_coords_rdkit(self, smiles: str) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Get 3D coordinates using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        coords = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
        
        return coords
    
    def _get_3d_coords_simple(self, smiles: str) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Simple 3D coordinate generation fallback."""
        atoms_2d = self._parse_simple_smiles(smiles)
        coords_3d = []
        for atom, (x, y) in atoms_2d:
            z = np.random.uniform(-0.5, 0.5)
            coords_3d.append((atom, (x, y, z)))
        return coords_3d
    
    # =========================================================================
    # Reaction Scheme Renderer
    # =========================================================================
    
    async def _render_reaction_scheme(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render chemical reaction scheme.

        Spec parameters:
            - reactants: List of reactant formulas
            - products: List of product formulas
            - conditions: Reaction conditions (catalyst, temp, etc.)
            - arrow_type: Type of arrow (single, double, equilibrium)
            - show_coefficients: Whether to show stoichiometric coefficients
        """
        # ENHANCED: Extract reactants/products from multiple possible sources
        reactants = self._extract_reaction_components(spec, 'reactants')
        products = self._extract_reaction_components(spec, 'products')
        conditions = spec.get('conditions', [])
        arrow_type = spec.get('arrow_type', 'single')

        # If still empty after extraction, use placeholder with warning
        if not reactants:
            logger.warning("No reactants found in spec - using placeholder")
            reactants = ['?']
        if not products:
            logger.warning("No products found in spec - using placeholder")
            products = ['?']
        
        fig, ax = self._create_figure(spec, figsize=(12, 4))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        font_size = style.get('font_size', 14)
        
        # Position reactants on left
        reactant_text = ' + '.join(reactants)
        ax.text(2, 2, reactant_text, ha='center', va='center',
               fontsize=font_size + 4, fontweight='bold')
        
        # Draw arrow
        arrow_y = 2
        if arrow_type == 'equilibrium':
            # Double-headed arrow for equilibrium
            ax.annotate('', xy=(8.5, arrow_y + 0.1), xytext=(3.5, arrow_y + 0.1),
                       arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
            ax.annotate('', xy=(3.5, arrow_y - 0.1), xytext=(8.5, arrow_y - 0.1),
                       arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
        else:
            # Single arrow
            ax.annotate('', xy=(8.5, arrow_y), xytext=(3.5, arrow_y),
                       arrowprops=dict(arrowstyle='->', color=line_color, lw=2))
        
        # Add conditions above arrow
        if conditions:
            condition_text = ', '.join(conditions)
            ax.text(6, 2.5, condition_text, ha='center', va='bottom',
                   fontsize=font_size - 2, style='italic')
        
        # Position products on right
        product_text = ' + '.join(products)
        ax.text(10, 2, product_text, ha='center', va='center',
               fontsize=font_size + 4, fontweight='bold')
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Lab Setup Renderers
    # =========================================================================
    
    async def _render_lab_titration(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render titration apparatus setup.
        
        Spec parameters:
            - burette_label: Label for burette contents
            - flask_label: Label for flask contents
            - indicator: Indicator being used
            - show_clamp_stand: Whether to show clamp stand
        """
        burette_label = spec.get('burette_label', 'NaOH (aq)')
        flask_label = spec.get('flask_label', 'HCl (aq)')
        indicator = spec.get('indicator', 'Phenolphthalein')
        
        fig, ax = self._create_figure(spec, figsize=(10, 12))
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Draw clamp stand
        # Base
        ax.add_patch(Rectangle((1, 0.5), 3, 0.3, facecolor='#8B4513', edgecolor=line_color))
        # Pole
        ax.add_patch(Rectangle((2.4, 0.8), 0.2, 10, facecolor='#8B4513', edgecolor=line_color))
        # Clamp arm
        ax.add_patch(Rectangle((2.6, 8), 2, 0.2, facecolor='#696969', edgecolor=line_color))
        
        # Draw burette
        burette_x = 4.5
        # Burette body
        ax.add_patch(Rectangle((burette_x - 0.15, 3), 0.3, 6, 
                               facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5))
        # Burette graduations
        for i in range(12):
            y_pos = 3.5 + i * 0.45
            ax.plot([burette_x - 0.15, burette_x - 0.05], [y_pos, y_pos], 
                   color=line_color, linewidth=0.5)
        
        # Stopcock
        ax.add_patch(Rectangle((burette_x - 0.2, 2.8), 0.4, 0.3, 
                               facecolor='#FFD700', edgecolor=line_color))
        
        # Burette tip
        ax.plot([burette_x, burette_x], [2.8, 2.2], color=line_color, linewidth=1.5)
        
        # Liquid in burette
        ax.add_patch(Rectangle((burette_x - 0.12, 3.2), 0.24, 5.5, 
                               facecolor='#ADD8E6', edgecolor='none', alpha=0.7))
        
        # Conical flask
        flask_points = [
            (3.5, 1), (4, 2.5), (5.5, 2.5), (6, 1), (3.5, 1)
        ]
        flask_path = Polygon(flask_points, closed=True, 
                            facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5)
        ax.add_patch(flask_path)
        
        # Flask neck
        ax.add_patch(Rectangle((4.3, 2.5), 0.9, 0.8, 
                               facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5))
        
        # Liquid in flask
        flask_liquid = Polygon([(3.7, 1.1), (4.1, 2.2), (5.4, 2.2), (5.8, 1.1)],
                              closed=True, facecolor='#FFB6C1', edgecolor='none', alpha=0.7)
        ax.add_patch(flask_liquid)
        
        # White tile
        ax.add_patch(Rectangle((3.2, 0.5), 3.1, 0.3, 
                               facecolor='white', edgecolor=line_color, linewidth=1))
        
        # Labels
        ax.annotate(burette_label, xy=(burette_x + 0.3, 6), fontsize=10, 
                   ha='left', fontweight='bold')
        ax.annotate(flask_label, xy=(4.75, 1.5), fontsize=10, 
                   ha='center', fontweight='bold')
        ax.annotate(f'Indicator: {indicator}', xy=(7, 2), fontsize=9,
                   ha='left', style='italic')
        
        # Title
        ax.set_title('Titration Apparatus', fontsize=14, fontweight='bold', pad=10)
        
        return self._save_figure(fig, spec)
    
    async def _render_lab_distillation(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render distillation apparatus setup.
        
        Spec parameters:
            - mixture_label: Label for mixture being distilled
            - distillate_label: Label for collected distillate
            - show_thermometer: Whether to show thermometer
        """
        mixture_label = spec.get('mixture_label', 'Salt water')
        distillate_label = spec.get('distillate_label', 'Pure water')
        
        fig, ax = self._create_figure(spec, figsize=(14, 10))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Round bottom flask
        flask_center = (3, 3)
        flask = Circle(flask_center, 1.5, facecolor='#E8E8E8', 
                      edgecolor=line_color, linewidth=1.5)
        ax.add_patch(flask)
        
        # Flask neck
        ax.add_patch(Rectangle((2.85, 4.5), 0.3, 1.5, 
                               facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5))
        
        # Liquid in flask
        liquid = Circle(flask_center, 1.3, facecolor='#87CEEB', 
                       edgecolor='none', alpha=0.6)
        ax.add_patch(liquid)
        
        # Bunsen burner
        ax.add_patch(Rectangle((2.5, 0.5), 1, 0.3, facecolor='#696969', edgecolor=line_color))
        ax.add_patch(Rectangle((2.85, 0.8), 0.3, 0.5, facecolor='#696969', edgecolor=line_color))
        # Flame
        flame_points = [(3, 1.3), (2.7, 1.8), (3, 2.3), (3.3, 1.8)]
        flame = Polygon(flame_points, closed=True, facecolor='#FFA500', edgecolor='#FF4500')
        ax.add_patch(flame)
        
        # Thermometer
        ax.plot([3, 3], [5.5, 7.5], color='#696969', linewidth=2)
        ax.add_patch(Circle((3, 5.5), 0.15, facecolor='red', edgecolor=line_color))
        
        # Condenser (Liebig condenser)
        # Outer tube
        condenser_points = [(3.3, 6), (9, 4), (9.2, 4.3), (3.5, 6.3)]
        condenser = Polygon(condenser_points, closed=True, 
                           facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5)
        ax.add_patch(condenser)
        
        # Inner tube
        inner_points = [(3.4, 6.1), (8.9, 4.1), (8.95, 4.2), (3.45, 6.2)]
        inner = Polygon(inner_points, closed=True, 
                       facecolor='#D3D3D3', edgecolor=line_color, linewidth=0.5)
        ax.add_patch(inner)
        
        # Water inlet/outlet
        ax.annotate('Cold water in', xy=(7, 3.5), fontsize=8, ha='center')
        ax.plot([7, 7], [3.7, 4.05], color='#4169E1', linewidth=1.5)
        ax.annotate('Water out', xy=(5, 6.8), fontsize=8, ha='center')
        ax.plot([5, 5.2], [6.5, 6.15], color='#4169E1', linewidth=1.5)
        
        # Collection flask
        collection_center = (10.5, 2.5)
        collection = Circle(collection_center, 1, facecolor='#E8E8E8', 
                           edgecolor=line_color, linewidth=1.5)
        ax.add_patch(collection)
        ax.add_patch(Rectangle((10.35, 3.5), 0.3, 1, 
                               facecolor='#E8E8E8', edgecolor=line_color, linewidth=1.5))
        
        # Adapter from condenser to collection flask
        ax.plot([9.15, 10.5], [4.1, 4.5], color=line_color, linewidth=1.5)
        
        # Collected liquid
        collected = Wedge(collection_center, 0.8, 180, 360, 
                         facecolor='#ADD8E6', edgecolor='none', alpha=0.7)
        ax.add_patch(collected)
        
        # Labels
        ax.annotate(mixture_label, xy=(3, 0.2), fontsize=10, ha='center', fontweight='bold')
        ax.annotate(distillate_label, xy=(10.5, 1), fontsize=10, ha='center', fontweight='bold')
        ax.annotate('Condenser', xy=(6, 5.5), fontsize=9, ha='center')
        ax.annotate('Thermometer', xy=(3.5, 7), fontsize=9, ha='left')
        
        # Title
        ax.set_title('Distillation Apparatus', fontsize=14, fontweight='bold', pad=10)
        
        return self._save_figure(fig, spec)
    
    async def _render_lab_electrolysis(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render electrolysis apparatus setup.
        
        Spec parameters:
            - electrolyte: Electrolyte solution label
            - anode_material: Anode material
            - cathode_material: Cathode material
            - show_bubbles: Whether to show gas bubbles
        """
        electrolyte = spec.get('electrolyte', 'Dilute H₂SO₄')
        anode_material = spec.get('anode_material', 'Platinum')
        cathode_material = spec.get('cathode_material', 'Platinum')
        show_bubbles = spec.get('show_bubbles', True)
        
        fig, ax = self._create_figure(spec, figsize=(12, 10))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        
        # Beaker
        beaker_points = [(2, 1), (2, 6), (10, 6), (10, 1)]
        beaker = Polygon(beaker_points, closed=False, fill=False, 
                        edgecolor=line_color, linewidth=2)
        ax.add_patch(beaker)
        
        # Electrolyte solution
        solution = Polygon([(2.1, 1.1), (2.1, 5), (9.9, 5), (9.9, 1.1)],
                          closed=True, facecolor='#E6F3FF', edgecolor='none', alpha=0.7)
        ax.add_patch(solution)
        
        # Electrodes
        # Anode (positive, right)
        ax.add_patch(Rectangle((8, 2), 0.3, 3.5, facecolor='#C0C0C0', edgecolor=line_color))
        return self._save_figure(fig, spec)

    async def _render_lab_manometer(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render manometer diagram (U-tube).

        Spec parameters:
            - fluid_color: Color of the manometer fluid (default: '#C0C0C0' for mercury)
            - fluid_label: Label for fluid (e.g. "Mercury", "Water")
            - left_arm_connection: Label for what left arm connects to (e.g. "Gas Supply")
            - right_arm_connection: Label for right arm (e.g. "Atmosphere")
            - h_diff: Height difference (positive means right side higher)
            - h_diff_label: Label for height difference (e.g. "Δh = 45 mm")
            - p_atm: Atmospheric pressure (if applicable)
            - p_gas: Gas pressure (if applicable)
        """
        fluid_color = spec.get('fluid_color', '#C0C0C0')  # Mercury default
        fluid_label = spec.get('fluid_label', 'Mercury')
        left_connect = spec.get('left_arm_connection', 'Gas')
        right_connect = spec.get('right_arm_connection', 'Atmosphere')
        h_diff = spec.get('h_diff', 0)
        h_diff_label = spec.get('h_diff_label', f'h = {abs(h_diff)}')
        
        # Parse h_diff to float if string
        if isinstance(h_diff, str):
            import re
            match = re.search(r'(-?\d+(?:\.\d+)?)', h_diff)
            if match:
                h_diff = float(match.group(1))
            else:
                h_diff = 0
        
        fig, ax = self._create_figure(spec, figsize=(8, 10))
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        style = spec.get('style', {})
        line_color = style.get('line_color', '#000000')
        line_width = 2
        
        # U-tube dimensions
        tube_width = 1.0
        tube_spacing = 2.0
        left_x = 3.5
        right_x = left_x + tube_spacing + tube_width
        bottom_y = 2.0
        top_y = 10.0
        arc_radius = (right_x - (left_x + tube_width)) / 2 + tube_width
        
        # Draw U-tube path
        # Left vertical outer
        ax.plot([left_x, left_x], [top_y, bottom_y], color=line_color, linewidth=line_width)
        # Right vertical outer
        ax.plot([right_x + tube_width, right_x + tube_width], [top_y, bottom_y], color=line_color, linewidth=line_width)
        
        # Bottom outer arc
        theta = np.linspace(np.pi, 2*np.pi, 50)
        center_x = (left_x + right_x + tube_width) / 2
        r_outer = (right_x + tube_width - left_x) / 2
        x_arc_outer = center_x + r_outer * np.cos(theta)
        y_arc_outer = bottom_y + r_outer * np.sin(theta)
        ax.plot(x_arc_outer, y_arc_outer, color=line_color, linewidth=line_width)
        
        # Inner vertical lines
        ax.plot([left_x + tube_width, left_x + tube_width], [top_y, bottom_y], color=line_color, linewidth=line_width)
        ax.plot([right_x, right_x], [top_y, bottom_y], color=line_color, linewidth=line_width)
        
        # Bottom inner arc
        r_inner = r_outer - tube_width
        x_arc_inner = center_x + r_inner * np.cos(theta)
        y_arc_inner = bottom_y + r_inner * np.sin(theta)
        ax.plot(x_arc_inner, y_arc_inner, color=line_color, linewidth=line_width)
        
        # Fill Fluid
        # Base fluid level
        fluid_base = 5.0
        left_level = fluid_base
        right_level = fluid_base
        
        # Adjust levels based on h_diff (scale factor for visual)
        # Limit visual displacement to avoid going out of tube
        visual_diff = max(min(h_diff * 0.05, 3), -3) # Scale down large numbers
        if h_diff != 0:
             # If right is higher (h_diff > 0), right level up, left level down
             right_level += visual_diff
             left_level -= visual_diff
        
        # Draw fluid in U-bend
        # Create polygon for fluid
        fluid_poly_points = []
        # Left top surface
        fluid_poly_points.extend([(left_x, left_level), (left_x + tube_width, left_level)])
        # Inner right vertical down
        fluid_poly_points.append((left_x + tube_width, bottom_y))
        # Inner arc
        fluid_poly_points.extend(list(zip(x_arc_inner[::-1], y_arc_inner[::-1]))) # Reverse for polygon order
        # Inner right vertical up
        fluid_poly_points.append((right_x, bottom_y))
        # Right top surface
        fluid_poly_points.extend([(right_x, right_level), (right_x + tube_width, right_level)])
        # Outer right vertical down
        fluid_poly_points.append((right_x + tube_width, bottom_y))
        # Outer arc
        fluid_poly_points.extend(list(zip(x_arc_outer, y_arc_outer)))
        # Outer left vertical up
        fluid_poly_points.append((left_x, bottom_y))
        
        fluid_patch = Polygon(fluid_poly_points, closed=True, facecolor=fluid_color, edgecolor='none', alpha=0.8)
        ax.add_patch(fluid_patch)
        
        # Draw connection/bulb on left
        if "gas" in left_connect.lower() or "bulb" in left_connect.lower():
            # Draw a bulb
            bulb_center = (left_x - 1.5, top_y - 1)
            bulb_radius = 1.2
            bulb = Circle(bulb_center, bulb_radius, facecolor='white', edgecolor=line_color, linewidth=line_width)
            ax.add_patch(bulb)
            # Connecting tube
            ax.plot([left_x, left_x], [top_y, top_y + 0.5], color=line_color, linewidth=line_width) # Up cap
            ax.plot([left_x + tube_width, left_x + tube_width], [top_y, top_y + 0.5], color=line_color, linewidth=line_width)
            
            # Connector to bulb
            connector_y = top_y - 1
            ax.plot([bulb_center[0] + bulb_radius, left_x], [connector_y + 0.3, top_y], color=line_color, linewidth=line_width)
            ax.plot([bulb_center[0] + bulb_radius, left_x + tube_width], [connector_y - 0.3, top_y], color=line_color, linewidth=line_width)
            
            ax.text(bulb_center[0], bulb_center[1], left_connect, ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            # Open tube
             ax.text(left_x + tube_width/2, top_y + 0.5, left_connect, ha='center', fontsize=10)
             
        # Draw right connection (usually atmosphere)
        if "atm" in right_connect.lower() or "open" in right_connect.lower():
             ax.text(right_x + tube_width/2, top_y + 0.5, "Open to\nAtmosphere", ha='center', fontsize=9)
             # Arrows indicating P_atm
             ax.arrow(right_x + tube_width/2, top_y + 1.5, 0, -0.8, head_width=0.2, head_length=0.2, fc='k', ec='k')
             ax.text(right_x + tube_width/2 + 0.8, top_y + 1, "P_atm", fontsize=10)
        else:
             ax.text(right_x + tube_width/2, top_y + 0.5, right_connect, ha='center', fontsize=10)

        # Draw Height Difference
        if h_diff != 0:
            mid_x = (left_x + tube_width + right_x) / 2
            # Horizontal lines at levels
            ax.plot([left_x + tube_width, right_x], [left_level, left_level], 'k--', linewidth=1, alpha=0.5)
            ax.plot([right_x, right_x + tube_width*2], [right_level, right_level], 'k--', linewidth=1, alpha=0.5)
            
            # Vertical dimension line
            arrow_x = mid_x
            if right_level > left_level:
                # Right is higher
                extension_line_x = right_x + tube_width + 0.5
                ax.annotate('', xy=(extension_line_x, right_level), xytext=(extension_line_x, left_level),
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text(extension_line_x + 0.2, (right_level + left_level)/2, h_diff_label, va='center', fontsize=10, fontweight='bold')
                # Dashed extension from left level across
                ax.plot([left_x + tube_width, extension_line_x], [left_level, left_level], 'k--', linewidth=0.8, alpha=0.5)
                # Dashed extension from right level
                ax.plot([right_x + tube_width, extension_line_x], [right_level, right_level], 'k--', linewidth=0.8, alpha=0.5)
                
            else:
                 # Left is higher
                 pass # Similar logic if needed, but assuming right usually measures
        
        # Add labels
        ax.text(center_x, bottom_y - 1.5, fluid_label, ha='center', fontsize=10, fontstyle='italic')
        
        if spec.get('title'):
            ax.set_title(spec['title'], fontsize=14, fontweight='bold', pad=15)
            
        return self._save_figure(fig, spec)
        ax.text(3.85, 5.8, '−', fontsize=20, ha='center', va='center', color='blue')
        ax.text(3.85, 1.5, f'Cathode\n({cathode_material})', fontsize=8, ha='center', va='top')
        
        # Power supply
        ax.add_patch(Rectangle((4.5, 7.5), 3, 1.5, facecolor='#D3D3D3', edgecolor=line_color))
        ax.text(6, 8.25, 'DC Power\nSupply', fontsize=9, ha='center', va='center')
        
        # Wires
        ax.plot([3.85, 3.85, 4.5], [5.5, 8.25, 8.25], color='blue', linewidth=2)
        ax.plot([8.15, 8.15, 7.5], [5.5, 8.25, 8.25], color='red', linewidth=2)
        
        # Bubbles if enabled
        if show_bubbles:
            # Bubbles at cathode (H2)
            for _ in range(5):
                bx = 3.85 + np.random.uniform(-0.2, 0.2)
                by = 2.5 + np.random.uniform(0, 2)
                ax.add_patch(Circle((bx, by), 0.1, facecolor='white', 
                                   edgecolor='#4169E1', alpha=0.7))
            ax.text(3.85, 6.3, 'H₂', fontsize=10, ha='center', color='blue')
            
            # Bubbles at anode (O2)
            for _ in range(5):
                bx = 8.15 + np.random.uniform(-0.2, 0.2)
                by = 2.5 + np.random.uniform(0, 2)
                ax.add_patch(Circle((bx, by), 0.1, facecolor='white', 
                                   edgecolor='#4169E1', alpha=0.7))
            ax.text(8.15, 6.3, 'O₂', fontsize=10, ha='center', color='red')
        
        # Labels
        ax.annotate(f'Electrolyte: {electrolyte}', xy=(6, 3), 
                   fontsize=11, ha='center', fontweight='bold')
        
        # Title
        ax.set_title('Electrolysis Apparatus', fontsize=14, fontweight='bold', pad=10)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Periodic Table Renderer
    # =========================================================================
    
    async def _render_periodic_table(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render a section of the periodic table.
        
        Spec parameters:
            - elements: List of atomic numbers to highlight
            - show_all: Show entire periodic table (partial support)
            - highlight_groups: List of groups to highlight
            - highlight_periods: List of periods to highlight
        """
        elements = spec.get('elements', [1, 6, 7, 8])
        highlight_groups = spec.get('highlight_groups', [])
        highlight_periods = spec.get('highlight_periods', [])
        
        fig, ax = self._create_figure(spec, figsize=(16, 10))
        
        ax.set_xlim(0, 19)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        cell_width = 0.9
        cell_height = 0.9
        
        for atomic_num, elem_data in PERIODIC_TABLE.items():
            group = elem_data['group']
            period = elem_data['period']
            
            x = group
            y = 8 - period
            
            # Determine color
            if atomic_num in elements:
                color = '#FFD700'  # Gold for highlighted
                edge_width = 2
            elif group in highlight_groups or period in highlight_periods:
                color = '#90EE90'  # Light green for group/period highlight
                edge_width = 1.5
            else:
                category = elem_data.get('category', 'unknown')
                color = CATEGORY_COLORS.get(category, '#E0E0E0')
                edge_width = 1
            
            # Draw element cell
            cell = FancyBboxPatch(
                (x - cell_width/2, y - cell_height/2),
                cell_width, cell_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=edge_width
            )
            ax.add_patch(cell)
            
            # Element symbol
            ax.text(x, y + 0.1, elem_data['symbol'], 
                   ha='center', va='center',
                   fontsize=12, fontweight='bold')
            
            # Atomic number
            ax.text(x, y - 0.25, str(atomic_num), 
                   ha='center', va='center',
                   fontsize=7)
        
        # Title
        if elements:
            elem_symbols = [PERIODIC_TABLE[n]['symbol'] for n in elements if n in PERIODIC_TABLE]
            ax.set_title(f'Periodic Table - Highlighting: {", ".join(elem_symbols)}', 
                        fontsize=14, fontweight='bold', pad=10)
        else:
            ax.set_title('Periodic Table Section', fontsize=14, fontweight='bold', pad=10)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Orbital Diagram Renderer
    # =========================================================================
    
    async def _render_orbital_diagram(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render electron orbital diagram.
        
        Spec parameters:
            - element: Element symbol or atomic number
            - config: Electron configuration string (optional, auto-generated if not provided)
            - show_boxes: Show orbital boxes with arrows
            - orbital_type: Type to highlight (s, p, d, f)
        """
        element = spec.get('element', 'C')
        show_boxes = spec.get('show_boxes', True)
        
        # Get atomic number
        if isinstance(element, int):
            atomic_num = element
            symbol = PERIODIC_TABLE.get(atomic_num, {}).get('symbol', str(atomic_num))
        else:
            # Find by symbol
            atomic_num = None
            symbol = element
            for num, data in PERIODIC_TABLE.items():
                if data['symbol'] == element:
                    atomic_num = num
                    break
            if atomic_num is None:
                atomic_num = 6  # Default to carbon
                symbol = 'C'
        
        fig, ax = self._create_figure(spec, figsize=(14, 8))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Define orbital order and capacities
        orbitals = [
            ('1s', 2), ('2s', 2), ('2p', 6), ('3s', 2), ('3p', 6),
            ('4s', 2), ('3d', 10), ('4p', 6)
        ]
        
        # Fill orbitals based on atomic number
        electrons_remaining = atomic_num
        orbital_electrons = {}
        
        for orbital, capacity in orbitals:
            if electrons_remaining <= 0:
                break
            electrons = min(electrons_remaining, capacity)
            orbital_electrons[orbital] = electrons
            electrons_remaining -= electrons
        
        # Draw orbitals
        y_pos = 6.5
        x_pos = 1
        
        ax.text(7, 7.5, f'{symbol} (Z = {atomic_num}) Electron Configuration',
               ha='center', fontsize=14, fontweight='bold')
        
        for orbital, capacity in orbitals:
            electrons = orbital_electrons.get(orbital, 0)
            
            # Orbital label
            ax.text(x_pos, y_pos, orbital, ha='center', fontsize=11, fontweight='bold')
            
            if show_boxes:
                # Draw boxes for orbital
                num_boxes = capacity // 2 if orbital[1] == 's' else capacity // 2
                box_width = 0.5
                start_x = x_pos + 0.3
                
                for i in range(num_boxes):
                    bx = start_x + i * (box_width + 0.1)
                    
                    # Draw box
                    ax.add_patch(Rectangle((bx, y_pos - 0.3), box_width, 0.6,
                                          facecolor='white', edgecolor='black'))
                    
                    # Draw electron arrows
                    electrons_in_box = min(2, electrons - i * 2)
                    if electrons_in_box >= 1:
                        # Up arrow
                        ax.annotate('', xy=(bx + 0.15, y_pos + 0.15),
                                   xytext=(bx + 0.15, y_pos - 0.15),
                                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
                    if electrons_in_box >= 2:
                        # Down arrow
                        ax.annotate('', xy=(bx + 0.35, y_pos - 0.15),
                                   xytext=(bx + 0.35, y_pos + 0.15),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            
            # Update position
            x_pos += 2.5 + (capacity // 4)
            if x_pos > 12:
                x_pos = 1
                y_pos -= 1.5
        
        # Write electron configuration
        config_parts = [f'{orb}^{e}' for orb, e in orbital_electrons.items()]
        config_text = '  '.join(config_parts)
        ax.text(7, 1, f'Configuration: {config_text}', ha='center', fontsize=12)
        
        return self._save_figure(fig, spec)
    
    # =========================================================================
    # Crystal Structure Renderer
    # =========================================================================
    
    async def _render_crystal_structure(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render crystal lattice structure.
        
        Spec parameters:
            - structure_type: Type of crystal (cubic, bcc, fcc, hcp)
            - element: Element symbol for labeling
            - show_unit_cell: Highlight the unit cell
            - show_atoms: Show atoms at lattice points
        """
        structure_type = spec.get('structure_type', 'cubic')
        element = spec.get('element', 'Na')
        show_unit_cell = spec.get('show_unit_cell', True)
        
        fig = plt.figure(figsize=(10, 10), dpi=self._get_dpi(spec))
        ax = fig.add_subplot(111, projection='3d')
        
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        fig.patch.set_facecolor(bg_color)
        
        # Define lattice points based on structure type
        if structure_type == 'bcc':
            # Body-centered cubic
            points = [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                (0.5, 0.5, 0.5)  # Body center
            ]
            title = f'{element} - Body-Centered Cubic (BCC)'
        elif structure_type == 'fcc':
            # Face-centered cubic
            points = [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 1), (0.5, 1, 0.5), (1, 0.5, 0.5)
            ]
            title = f'{element} - Face-Centered Cubic (FCC)'
        else:
            # Simple cubic
            points = [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)
            ]
            title = f'{element} - Simple Cubic'
        
        # Draw atoms
        xs, ys, zs = zip(*points)
        ax.scatter(xs, ys, zs, c='#4169E1', s=300, edgecolors='black', linewidths=1)
        
        # Draw unit cell edges
        if show_unit_cell:
            # Bottom face
            ax.plot3D([0, 1], [0, 0], [0, 0], 'k-', linewidth=1)
            ax.plot3D([1, 1], [0, 1], [0, 0], 'k-', linewidth=1)
            ax.plot3D([1, 0], [1, 1], [0, 0], 'k-', linewidth=1)
            ax.plot3D([0, 0], [1, 0], [0, 0], 'k-', linewidth=1)
            
            # Top face
            ax.plot3D([0, 1], [0, 0], [1, 1], 'k-', linewidth=1)
            ax.plot3D([1, 1], [0, 1], [1, 1], 'k-', linewidth=1)
            ax.plot3D([1, 0], [1, 1], [1, 1], 'k-', linewidth=1)
            ax.plot3D([0, 0], [1, 0], [1, 1], 'k-', linewidth=1)
            
            # Vertical edges
            ax.plot3D([0, 0], [0, 0], [0, 1], 'k-', linewidth=1)
            ax.plot3D([1, 1], [0, 0], [0, 1], 'k-', linewidth=1)
            ax.plot3D([0, 0], [1, 1], [0, 1], 'k-', linewidth=1)
            ax.plot3D([1, 1], [1, 1], [0, 1], 'k-', linewidth=1)
        
        # Set axis properties
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('c')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        return self._save_figure(fig, spec)
