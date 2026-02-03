"""
Question Generation Constants

Contains diagram type guidance, valid diagram types, and section configuration.
"""

from .models.config import QuestionType


# ============================================================================
# DIAGRAM TYPE GUIDANCE FOR LLM
# ============================================================================

DIAGRAM_TYPE_GUIDANCE = {
    "maths": """
MATHS DIAGRAM TYPES - Choose carefully based on the concept:

1. coordinate_graph - USE FOR:
   ✓ Vectors (position vectors, displacement vectors, velocity vectors)
   ✓ Plotting functions like y = x², y = sin(x), linear equations  
   ✓ Points and coordinates (e.g., "Plot point A(3,4)")
   ✓ Line segments, rays, arrows between points
   ✓ Inequalities on 2D plane
   PARAMETERS: functions=[{expression, label, color}], points=[{x, y, label}], lines=[{from, to}]

2. geometry_construction - USE FOR:
   ✓ Triangles, circles, quadrilaterals
   ✓ Angle bisectors, perpendicular bisectors
   ✓ Geometric proofs and constructions
   ✓ Congruence and similarity diagrams
   PARAMETERS: shapes=[{type, vertices}], angles=[{vertex, value}]

3. number_line - USE FOR:
   ✓ Representing integers, fractions, decimals
   ✓ Inequalities like x > 3 or -2 ≤ x ≤ 5
   ✓ Absolute value concepts
   PARAMETERS: range, points, intervals

4. venn_diagram - USE FOR:
   ✓ Set operations (union, intersection)
   ✓ Probability problems with overlapping events
   PARAMETERS: sets=[{label, elements}]

5. bar_chart / pie_chart - USE FOR:
   ✓ Statistics and data representation
   ✓ Frequency distributions
   PARAMETERS: data=[{label, value}]

6. trigonometric_circle - USE FOR:
   ✓ Unit circle with angles
   ✓ Sin, cos values at specific angles
   ✓ Trigonometric identities visualization
   PARAMETERS: angles, show_values

7. 3d_plot - USE ONLY FOR:
   ✓ 3D mathematical surfaces like z = x² + y²
   ✓ 3D coordinate geometry (NOT for simple vectors!)
   ⚠️ DO NOT use for 2D vectors or position vectors - use coordinate_graph instead!
   PARAMETERS: expression (z as function of x,y)

⚠️ IMPORTANT FOR VECTORS:
- Position vectors → coordinate_graph with lines/points
- Displacement vectors → coordinate_graph with arrows
- DO NOT use 3d_plot for vectors unless explicitly about 3D vector calculus
""",

    "physics": """
PHYSICS DIAGRAM TYPES - Choose based on the topic:

⚠️ DO NOT USE DIAGRAMS FOR:
- Conceptual/theoretical questions (e.g., "Explain the Law of Inertia")
- Historical questions (e.g., "Describe Galileo's inclined plane experiment")
- Definition questions (e.g., "Define momentum")
- Questions without specific numerical data
For these, set has_diagram: false and diagram_spec: null

✓ USE DIAGRAMS ONLY FOR NUMERICAL/VISUAL PROBLEMS:

1. series_circuit / parallel_circuit / mixed_circuit - USE FOR:
   ✓ Electrical circuit problems WITH specific resistance/voltage values
   ✓ Resistance, current, voltage calculations
   REQUIRED: components=[{type: resistor/battery/lamp, value, label}]

2. ray_diagram_lens / ray_diagram_mirror - USE FOR:
   ✓ Optics - image formation WITH given distances
   ✓ Convex/concave lens and mirror problems WITH numerical data
   REQUIRED: lens/mirror type, focal_length (number), object_distance (number)

3. free_body_diagram - USE FOR:
   ✓ Force analysis on objects WITH specific mass/force values
   ✓ Newton's laws NUMERICAL problems
   ✓ Equilibrium problems WITH given forces
   REQUIRED: mass (number), forces=[{label, direction, magnitude (NUMBER not string!)}]
   ⚠️ forces array must NOT be empty!
   ⚠️ magnitude must be a number like 98, not a string like "98N"

4. inclined_plane - USE FOR:
   ✓ Block/cylinder/sphere on slope problems WITH specific angle > 0
   ✓ Rolling motion problems (cylinders, spheres, discs)
   ✓ Friction problems on inclines WITH numerical data
   REQUIRED: angle (number > 0), mass (number)
   OPTIONAL but IMPORTANT:
   - object_type: "box", "cylinder", "sphere", "ball", "disc" (default: "box")
   - radius: for cylinders/spheres (in meters)
   - plane_length: length of incline (in meters)
   ⚠️ If question mentions CYLINDER/SPHERE/DISC, you MUST set object_type!
   ⚠️ angle must be > 0! Use free_body_diagram for horizontal surfaces

5. projectile_motion - USE FOR:
   ✓ Trajectory problems WITH initial velocity and angle
   ✓ Range, height calculations WITH numerical data
   REQUIRED: initial_velocity (number > 0), angle (number)

6. wave_diagram - USE FOR:
   ✓ Wave properties WITH specific wavelength, amplitude values
   ✓ Transverse and longitudinal waves
   REQUIRED: wavelength (number > 0), amplitude (number > 0)

7. electric_field / magnetic_field - USE FOR:
   ✓ Field line diagrams
   ✓ Charges and magnets
   PARAMETERS: sources, field_lines
""",

    "chemistry": """
CHEMISTRY DIAGRAM TYPES:

1. molecule_2d - USE FOR:
   ✓ Structural formulas of organic compounds
   ✓ Bonding diagrams
   PARAMETERS: smiles (SMILES notation like "CCO" for ethanol)

2. molecule_3d - USE FOR:
   ✓ Stereochemistry
   ✓ 3D molecular geometry
   PARAMETERS: smiles, show_labels

3. reaction_scheme - USE FOR:
   ✓ Chemical equations with structures
   ✓ Reaction mechanisms
   PARAMETERS: reactants, products, conditions

4. lab_setup_* - USE FOR:
   ✓ Practical/experimental setups
   ✓ Titration, distillation, electrolysis

5. orbital_diagram - USE FOR:
   ✓ Electron configurations
   ✓ Atomic orbitals
   PARAMETERS: element, electrons
""",

    "biology": """
BIOLOGY DIAGRAM TYPES - Anatomical and cellular:

1. human_heart / human_brain / nephron / neuron - USE FOR:
   ✓ Organ structure and function questions
   ✓ Label the parts questions
   PARAMETERS: labels=[parts to show], highlight=[parts to emphasize]

2. plant_cell / animal_cell - USE FOR:
   ✓ Cell structure questions
   ✓ Organelle identification
   PARAMETERS: labels, highlight

3. dna_replication / mitosis_stages / meiosis_stages - USE FOR:
   ✓ Molecular biology
   ✓ Cell division processes
   PARAMETERS: stage, labels

4. digestive_system / respiratory_system - USE FOR:
   ✓ Human body systems
   PARAMETERS: labels, highlight

5. eye_structure / ear_structure / flower_structure - USE FOR:
   ✓ Sense organs
   ✓ Plant reproduction
   PARAMETERS: labels, highlight
"""
}


# ============================================================================
# VALID DIAGRAM TYPES BY SUBJECT
# ============================================================================

VALID_DIAGRAM_TYPES = {
    "maths": [
        "coordinate_graph",
        "geometry_construction",
        "number_line",
        "venn_diagram",
        "bar_chart",
        "pie_chart",
        "trigonometric_circle",
        "3d_plot",
    ],
    "physics": [
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
    ],
    "chemistry": [
        "molecule_2d",
        "molecule_3d",
        "reaction_scheme",
        "lab_setup_titration",
        "lab_setup_distillation",
        "lab_setup_electrolysis",
        "periodic_table_section",
        "orbital_diagram",
        "crystal_structure",
    ],
    "biology": [
        "human_heart",
        "human_brain",
        "nephron",
        "neuron",
        "plant_cell",
        "animal_cell",
        "dna_replication",
        "mitosis_stages",
        "meiosis_stages",
        "digestive_system",
        "respiratory_system",
        "eye_structure",
        "ear_structure",
        "flower_structure",
    ],
}


# ============================================================================
# SECTION CONFIGURATION
# ============================================================================

SECTION_CONFIG = {
    QuestionType.MCQ: {
        "name": "Section A - Multiple Choice Questions",
        "instructions": "Choose the correct option for each question. Each question carries {marks} mark(s).",
    },
    QuestionType.TRUE_FALSE: {
        "name": "Section A - True/False Questions",
        "instructions": "State whether the following statements are True or False. Each question carries {marks} mark(s).",
    },
    QuestionType.FILL_IN_BLANKS: {
        "name": "Section B - Fill in the Blanks",
        "instructions": "Fill in the blanks with appropriate words/terms. Each question carries {marks} mark(s).",
    },
    QuestionType.SHORT_ANSWER: {
        "name": "Section B - Short Answer Questions",
        "instructions": "Answer the following questions briefly (2-4 sentences). Each question carries {marks} mark(s).",
    },
    QuestionType.LONG_ANSWER: {
        "name": "Section C - Long Answer Questions",
        "instructions": "Answer the following questions in detail. Each question carries {marks} mark(s).",
    },
    QuestionType.NUMERICAL: {
        "name": "Section D - Numerical Problems",
        "instructions": "Solve the following problems showing all steps. Each question carries {marks} mark(s).",
    },
    QuestionType.MATCH_THE_FOLLOWING: {
        "name": "Section E - Match the Following",
        "instructions": "Match items in Column A with items in Column B. Each question carries {marks} mark(s).",
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_diagram_types_text(subject: str) -> str:
    """Get formatted text of valid diagram types with detailed guidance for a subject."""
    subject_lower = subject.lower()
    if subject_lower in DIAGRAM_TYPE_GUIDANCE:
        return DIAGRAM_TYPE_GUIDANCE[subject_lower]
    elif subject_lower in VALID_DIAGRAM_TYPES:
        types = VALID_DIAGRAM_TYPES[subject_lower]
        return "\n".join([f"  - {t}" for t in types])
    return "  (No specific diagram types available for this subject)"


def get_valid_diagram_types(subject: str) -> list:
    """Get list of valid diagram types for a subject."""
    subject_lower = subject.lower()
    return VALID_DIAGRAM_TYPES.get(subject_lower, VALID_DIAGRAM_TYPES.get("maths", []))


def get_section_config(question_type: QuestionType) -> dict:
    """Get section configuration for a question type."""
    return SECTION_CONFIG.get(question_type, {
        "name": f"Section - {question_type.value}",
        "instructions": "Answer the following questions. Each question carries {marks} mark(s).",
    })
