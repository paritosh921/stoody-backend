"""
Academic Standards for JEE/NEET/CBSE Diagram Generation

Defines standard conventions, colors, symbols, and styles used in Indian competitive exams.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ExamBoard(str, Enum):
    """Supported exam boards/standards"""
    JEE = "jee"           # Joint Entrance Examination
    NEET = "neet"         # National Eligibility cum Entrance Test
    CBSE = "cbse"         # Central Board of Secondary Education
    ICSE = "icse"         # Indian Certificate of Secondary Education
    GENERIC = "generic"   # Generic academic standard


@dataclass
class ColorScheme:
    """Standard color scheme for diagram elements"""
    primary: str = "#000000"      # Black - main lines/shapes
    secondary: str = "#333333"    # Dark gray - secondary elements
    highlight: str = "#FF0000"    # Red - important/highlighted
    background: str = "#FFFFFF"   # White - background
    
    # Force-specific colors
    weight: str = "#FF0000"       # Red - weight/gravity
    normal: str = "#0000FF"       # Blue - normal force
    friction: str = "#FFA500"     # Orange - friction
    applied: str = "#00FF00"      # Green - applied force
    tension: str = "#800080"      # Purple - tension
    
    # Circuit colors
    wire: str = "#000000"         # Black - wires
    component: str = "#333333"    # Dark gray - components
    current: str = "#FF0000"      # Red - current direction
    
    # Optics colors
    real_ray: str = "#000000"     # Black - real rays
    virtual_ray: str = "#0000FF"  # Blue - virtual rays (often dashed)
    object_color: str = "#00FF00" # Green - object
    image_color: str = "#FF0000"  # Red - image


@dataclass
class FontSettings:
    """Standard font settings for academic diagrams"""
    family: str = "Arial"
    label_size: int = 14          # Labels on shapes
    title_size: int = 16          # Diagram title
    annotation_size: int = 12     # Small annotations
    min_readable: int = 12        # Minimum readable size
    max_size: int = 20            # Maximum for any text
    weight: str = "normal"        # normal or bold


@dataclass
class LineSettings:
    """Standard line settings"""
    width: float = 1.5            # Default line width
    arrow_width: float = 2.0      # Arrow lines
    thin: float = 1.0             # Thin lines (grid, axes)
    thick: float = 2.5            # Thick/emphasized lines
    dashed_pattern: str = "--"    # Dashed line pattern
    dotted_pattern: str = ":"     # Dotted line pattern


class AcademicStandards:
    """
    JEE/NEET/CBSE diagram conventions and standards.
    
    These conventions ensure diagrams match what students expect from
    textbooks and coaching materials used in Indian competitive exams.
    """
    
    # Default settings
    COLORS = ColorScheme()
    FONTS = FontSettings()
    LINES = LineSettings()
    
    # =========================================================================
    # PHYSICS CONVENTIONS
    # =========================================================================
    
    PHYSICS_CONVENTIONS = {
        # Force diagram conventions
        "force_diagram": {
            "arrow_origin": "center",           # Arrows from center of object
            "arrow_style": "filled_head",       # Filled arrow heads
            "proportional_lengths": True,       # Length ~ magnitude
            "weight_direction": "down",         # Always straight down
            "normal_direction": "perpendicular", # Perpendicular to surface
            "colors": {
                "weight": "#FF0000",            # Red
                "normal": "#0000FF",            # Blue
                "friction": "#FFA500",          # Orange
                "applied": "#00FF00",           # Green
                "tension": "#800080",           # Purple
                "net_force": "#000000",         # Black
            },
            "label_format": "{name} = {value} {unit}",  # e.g., "F = 50 N"
            "label_position": "outside_arrow",  # Labels outside arrows
        },
        
        # Circuit conventions
        "circuit": {
            "symbol_standard": "IEC",           # International standard
            "wire_color": "#000000",            # Black wires
            "flow_direction": "conventional",   # + to - current flow
            "component_spacing": "even",        # Even spacing
            "label_format": "{symbol} = {value}{unit}",  # R = 10Ω
            "show_polarity": True,              # Show + and - on sources
            "ground_symbol": "standard",        # Standard ground symbol
        },
        
        # Optics conventions
        "optics": {
            "object_position": "left",          # Object on left side
            "ray_direction": "left_to_right",   # Light travels L to R
            "real_line_style": "solid",         # Solid for real rays
            "virtual_line_style": "dashed",     # Dashed for virtual
            "principal_axis_style": "dotted",   # Dotted horizontal line
            "focal_point_label": "F",           # Use 'F' for focal point
            "object_representation": "upward_arrow",
            "image_representation": "arrow",     # Inverted if real
            "standard_rays": [
                "parallel_to_axis",             # Ray 1
                "through_optical_center",       # Ray 2
                "through_focus",                # Ray 3
            ],
        },
        
        # Wave conventions
        "wave": {
            "direction": "left_to_right",       # Wave propagation direction
            "equilibrium_style": "dashed",      # Dashed center line
            "wavelength_marker": "double_arrow", # ←λ→ style
            "amplitude_marker": "vertical_line", # Vertical line A
            "label_wavelength": "λ",
            "label_amplitude": "A",
        },
        
        # Inclined plane conventions
        "inclined_plane": {
            "angle_position": "base",           # Angle at base
            "coordinate_alignment": "along_plane",  # x along plane, y perpendicular
            "component_colors": {
                "mg_sin": "#FF0000",            # Red - along plane
                "mg_cos": "#0000FF",            # Blue - perpendicular
            },
            "show_components": True,
            "right_angle_marker": True,         # Small square for 90°
        },
        
        # Projectile motion
        "projectile": {
            "trajectory_style": "solid_curve",
            "velocity_components": True,        # Show vx, vy
            "angle_at_origin": True,            # Show launch angle
            "max_height_marker": True,          # Mark maximum height
            "range_marker": True,               # Mark horizontal range
        },
        
        # Electric/Magnetic fields
        "fields": {
            "field_line_arrows": True,          # Arrows on field lines
            "positive_outward": True,           # Field lines from + charge
            "equal_spacing": True,              # Even field line spacing
            "charge_symbols": {
                "positive": "+",
                "negative": "-",
            },
        },
    }
    
    # =========================================================================
    # MATHEMATICS CONVENTIONS
    # =========================================================================
    
    MATHS_CONVENTIONS = {
        # Geometry conventions
        "geometry": {
            "vertex_labels": "uppercase",       # A, B, C for vertices
            "side_labels": "lowercase",         # a, b, c for sides
            "angle_notation": "greek_or_degrees",  # θ, α or 30°
            "right_angle_marker": "small_square",  # □ at 90°
            "equal_sides_marker": "tick_marks", # / or // for equal sides
            "parallel_marker": "arrows",        # >> for parallel lines
            "perpendicular_marker": "square",   # ⊥ symbol
            "construction_lines": "dashed",     # Dashed for construction
            "given_lines": "solid",             # Solid for given
        },
        
        # Number line conventions
        "number_line": {
            "direction": "left_to_right",       # Negative to positive
            "positive_arrow": "right",          # Arrow at right end
            "scale_spacing": "even",            # Equal intervals
            "included_endpoint": "filled_circle",   # ● for ≤, ≥
            "excluded_endpoint": "empty_circle",    # ○ for <, >
            "solution_highlight": "bold_line",  # Thick line for solution
            "critical_point_labeled": True,     # Show the boundary value
            "zero_marked": True,                # Show origin if in range
        },
        
        # Coordinate graph conventions
        "graph": {
            "axes_arrows": "positive_ends",     # Arrows at + ends
            "origin_label": "O",                # Label origin as O
            "x_axis_label": "x",                # Or variable name
            "y_axis_label": "y",                # Or variable name
            "grid_style": "light_gray",         # Light grid lines
            "function_style": "solid",          # Solid curve for functions
            "asymptote_style": "dashed",        # Dashed for asymptotes
            "intercept_markers": "filled_dot",  # Points at intercepts
            "key_points_labeled": True,         # Label important points
        },
        
        # Venn diagram conventions
        "venn": {
            "set_labels": "uppercase",          # A, B, C for sets
            "circle_overlap": True,             # Overlapping circles
            "universal_set": "rectangle",       # Rectangle around all
            "shading_for_operations": True,     # Shade intersection/union
        },
        
        # Trigonometry conventions
        "trigonometry": {
            "unit_circle_radius": 1,            # Standard unit circle
            "angle_from_positive_x": True,      # Measure from +x axis
            "counterclockwise_positive": True,  # Standard convention
            "quadrant_labels": True,            # I, II, III, IV
        },
    }
    
    # =========================================================================
    # CHEMISTRY CONVENTIONS
    # =========================================================================
    
    CHEMISTRY_CONVENTIONS = {
        # Molecular structure conventions
        "molecule": {
            "bond_notation": "lines",           # Single -, double =, triple ≡
            "atom_symbols": "periodic_table",   # Standard symbols
            "lone_pairs": "dots",               # Dot pairs for lone electrons
            "formal_charges": "superscript",    # Show +/- charges
            "3d_wedge_dash": True,              # Wedge for toward, dash for away
        },
        
        # Reaction conventions
        "reaction": {
            "arrow_style": "single",            # → for irreversible
            "equilibrium_arrows": "double",     # ⇌ for equilibrium
            "catalyst_above_arrow": True,       # Write catalyst above
            "conditions_below_arrow": True,     # Temperature, pressure below
            "state_symbols": True,              # (s), (l), (g), (aq)
        },
        
        # Lab setup conventions
        "lab_setup": {
            "standard_apparatus": True,         # Use standard symbols
            "labels_required": True,            # All parts labeled
            "flow_direction": "indicated",      # Show liquid/gas flow
        },
        
        # Orbital conventions
        "orbitals": {
            "s_orbital": "sphere",
            "p_orbital": "dumbbell",
            "d_orbital": "cloverleaf",
            "color_coding": True,               # Different phases
        },
    }
    
    # =========================================================================
    # BIOLOGY CONVENTIONS
    # =========================================================================
    
    BIOLOGY_CONVENTIONS = {
        # Cell diagram conventions
        "cell": {
            "membrane_style": "double_line",    # Double line for membrane
            "organelle_labels": "leader_lines", # Lines pointing to parts
            "nucleus_prominent": True,          # Nucleus clearly visible
            "scale_indication": False,          # Diagrams not to scale
        },
        
        # Anatomy conventions
        "anatomy": {
            "orientation": "standard",          # Anatomical position
            "labels": "leader_lines",           # Lines to parts
            "cross_section_hatching": True,     # Hatching for cut surfaces
            "blood_vessels": {
                "artery": "#FF0000",            # Red
                "vein": "#0000FF",              # Blue
            },
        },
        
        # Process diagrams (mitosis, meiosis, etc.)
        "process": {
            "stages_numbered": True,            # Number each stage
            "arrows_between_stages": True,      # Show progression
            "key_events_labeled": True,         # Label important events
        },
    }
    
    @classmethod
    def get_conventions(cls, subject: str) -> Dict[str, Any]:
        """Get conventions for a specific subject"""
        subject_lower = subject.lower()
        
        if subject_lower in ["physics", "phy"]:
            return cls.PHYSICS_CONVENTIONS
        elif subject_lower in ["maths", "math", "mathematics"]:
            return cls.MATHS_CONVENTIONS
        elif subject_lower in ["chemistry", "chem"]:
            return cls.CHEMISTRY_CONVENTIONS
        elif subject_lower in ["biology", "bio"]:
            return cls.BIOLOGY_CONVENTIONS
        else:
            return {}
    
    @classmethod
    def get_diagram_style(cls, subject: str, diagram_type: str) -> Dict[str, Any]:
        """Get specific style guide for a diagram type"""
        conventions = cls.get_conventions(subject)
        
        # Find matching diagram type
        for key, value in conventions.items():
            if key in diagram_type.lower() or diagram_type.lower() in key:
                return value
        
        # Return general conventions if no specific match
        return conventions.get("general", {})
    
    @classmethod
    def get_color_for_element(cls, element_type: str, subject: str = "physics") -> str:
        """Get standard color for a specific element type"""
        colors = cls.COLORS
        
        # Force colors
        force_colors = {
            "weight": colors.weight,
            "gravity": colors.weight,
            "mg": colors.weight,
            "normal": colors.normal,
            "friction": colors.friction,
            "applied": colors.applied,
            "tension": colors.tension,
        }
        
        element_lower = element_type.lower()
        for key, color in force_colors.items():
            if key in element_lower:
                return color
        
        return colors.primary
    
    @classmethod
    def get_label_format(cls, element_type: str, subject: str = "physics") -> str:
        """Get standard label format for an element"""
        if subject.lower() in ["physics", "phy"]:
            if "force" in element_type.lower():
                return "{name} = {value} N"
            elif "resistance" in element_type.lower():
                return "R = {value} Ω"
            elif "voltage" in element_type.lower():
                return "V = {value} V"
            elif "current" in element_type.lower():
                return "I = {value} A"
            elif "angle" in element_type.lower():
                return "θ = {value}°"
            elif "distance" in element_type.lower() or "length" in element_type.lower():
                return "{value} m"
        
        return "{value}"
    
    @classmethod
    def validate_compliance(cls, diagram_type: str, subject: str, plan: Dict[str, Any]) -> List[str]:
        """
        Check if a diagram plan complies with academic standards.
        Returns list of non-compliance warnings.
        """
        warnings = []
        conventions = cls.get_diagram_style(subject, diagram_type)
        
        if not conventions:
            return warnings
        
        # Check specific conventions based on diagram type
        if subject.lower() in ["physics", "phy"]:
            warnings.extend(cls._check_physics_compliance(diagram_type, plan, conventions))
        elif subject.lower() in ["maths", "math"]:
            warnings.extend(cls._check_maths_compliance(diagram_type, plan, conventions))
        
        return warnings
    
    @classmethod
    def _check_physics_compliance(cls, diagram_type: str, plan: Dict[str, Any], conventions: Dict) -> List[str]:
        """Check physics diagram compliance"""
        warnings = []
        
        if "force" in diagram_type.lower():
            # Check force arrow colors
            objects = plan.get("objects", [])
            for obj in objects:
                if isinstance(obj, dict) and "force" in obj.get("name", "").lower():
                    props = obj.get("properties", {})
                    color = props.get("color", "")
                    expected = conventions.get("colors", {})
                    # Could add specific color checking here
        
        return warnings
    
    @classmethod
    def _check_maths_compliance(cls, diagram_type: str, plan: Dict[str, Any], conventions: Dict) -> List[str]:
        """Check maths diagram compliance"""
        warnings = []
        
        if "number_line" in diagram_type.lower():
            # Check endpoint markers
            pass
        
        return warnings
    
    @classmethod
    def get_style_instructions(cls, subject: str, diagram_type: str) -> str:
        """
        Get human-readable style instructions for LLM prompt.
        Used in diagram generation prompts.
        """
        instructions = []
        conventions = cls.get_diagram_style(subject, diagram_type)
        
        instructions.append(f"=== {subject.upper()} DIAGRAM STANDARDS (JEE/NEET) ===")
        instructions.append("")
        
        if subject.lower() in ["physics", "phy"]:
            if "force" in diagram_type.lower():
                instructions.extend([
                    "FORCE DIAGRAM CONVENTIONS:",
                    "- Draw arrows FROM the center of the object",
                    "- Arrow length proportional to force magnitude",
                    "- Weight (mg): RED arrow pointing DOWN",
                    "- Normal force: BLUE arrow perpendicular to surface",
                    "- Friction: ORANGE arrow opposing motion",
                    "- Applied force: GREEN arrow",
                    "- Label format: 'F = 50 N' (not just '50')",
                    "- Place labels OUTSIDE the arrows, not on them",
                ])
            elif "circuit" in diagram_type.lower():
                instructions.extend([
                    "CIRCUIT DIAGRAM CONVENTIONS:",
                    "- Use standard IEC symbols (zigzag for resistor)",
                    "- Black wires, evenly spaced components",
                    "- Label all values: 'R = 10Ω', 'V = 12V'",
                    "- Show + and - on voltage sources",
                    "- Conventional current: + to - direction",
                ])
            elif "ray" in diagram_type.lower() or "lens" in diagram_type.lower() or "mirror" in diagram_type.lower():
                instructions.extend([
                    "RAY DIAGRAM CONVENTIONS:",
                    "- Object on LEFT, light travels LEFT to RIGHT",
                    "- Object as upward arrow on principal axis",
                    "- Mark focal points clearly with 'F'",
                    "- Principal axis: dashed horizontal line",
                    "- Real rays/images: SOLID lines",
                    "- Virtual rays/images: DASHED lines",
                    "- Draw at least 2 standard rays to locate image",
                ])
        
        elif subject.lower() in ["maths", "math"]:
            if "number" in diagram_type.lower() and "line" in diagram_type.lower():
                instructions.extend([
                    "NUMBER LINE CONVENTIONS:",
                    "- Arrow at RIGHT end (positive direction)",
                    "- Even spacing between scale marks",
                    "- Filled circle (●) for INCLUDED endpoints (≤, ≥)",
                    "- Empty circle (○) for EXCLUDED endpoints (<, >)",
                    "- Bold/thick line for solution region",
                    "- Label the critical point value",
                ])
            elif "geometry" in diagram_type.lower() or "triangle" in diagram_type.lower():
                instructions.extend([
                    "GEOMETRY CONVENTIONS:",
                    "- Vertices: UPPERCASE letters (A, B, C)",
                    "- Sides: lowercase letters (a, b, c) or measurements",
                    "- Right angles: small SQUARE marker (not arc)",
                    "- Equal sides: matching TICK marks",
                    "- Angles: arc with degree value or θ, α, β",
                    "- Parallel lines: matching ARROWS (>>)",
                ])
            elif "graph" in diagram_type.lower() or "coordinate" in diagram_type.lower():
                instructions.extend([
                    "GRAPH CONVENTIONS:",
                    "- Arrows at positive ends of both axes",
                    "- Label axes: 'x' at right, 'y' at top",
                    "- Mark origin as 'O'",
                    "- Show scale numbers on axes",
                    "- Label key points: intercepts, vertices",
                    "- Smooth curves for functions",
                ])
        
        instructions.extend([
            "",
            "GENERAL STANDARDS:",
            "- Font size: 14-16 for labels (must be readable)",
            "- Keep diagram CLEAN and UNCLUTTERED",
            "- SIMPLICITY over visual complexity",
            "- Correctness > Clarity > Beauty",
        ])
        
        return "\n".join(instructions)
