"""
Single-Question Focused Prompts

These prompts are designed for generating ONE question at a time,
with specific concept context retrieved via RAG.
"""

from typing import Dict, Optional
from .models.config import QuestionType

# System prompt for all question generation
SYSTEM_PROMPT = """You are an expert educational content creator specializing in exam question generation.
You create high-quality, pedagogically sound questions that:
- Test specific learning objectives
- Are clear, unambiguous, and age-appropriate
- Follow Bloom's taxonomy principles
- Include accurate answers and helpful explanations

When creating diagrams, you ONLY use the EXACT diagram types provided - never invent new types."""


# Critical rules for diagram specifications - ensures unique diagrams per question
DIAGRAM_CRITICAL_RULES = """
CRITICAL DIAGRAM RULES (READ CAREFULLY):

1. For coordinate_graph: Points MUST have BOTH "x" AND "y" coordinates!
   ❌ WRONG: {"x": 1, "label": "A(1,2)"}  <-- Missing "y", will plot on x-axis at y=0!
   ✅ CORRECT: {"x": 1, "y": 2, "label": "A(1,2)"}  <-- Has both x AND y

2. For lines/vectors: "from" and "to" MUST be [x, y] arrays with BOTH coordinates!
   ❌ WRONG: {"from": [1], "to": [4]}  <-- Missing y values!
   ✅ CORRECT: {"from": [1, 2], "to": [4, 6]}  <-- Has [x, y] for both

3. diagram_spec parameters MUST contain EXACT values from THIS question:
   - If question asks about y = 2x + 3, diagram MUST show y = 2x + 3 (not a generic line)
   - If question mentions angle of 45°, diagram MUST show exactly 45°
   - If question has points A(2,3) and B(5,7), diagram MUST plot exactly at (2,3) and (5,7)
   - If question involves inequality x > 3, number_line MUST highlight x > 3

4. NEVER use placeholder or generic values like:
   - "equation_here" or "your_equation"
   - "value" or "x" when specific numbers are given in the question
   - Generic ranges like [-10, 10] when the question has specific values
   - Empty or default parameters

5. The diagram MUST be UNIQUE to THIS specific question:
   - Another question about a different equation should produce a DIFFERENT diagram
   - The diagram should NOT be reusable for other questions
   - Include question-specific labels, values, and markers

6. CORRECT coordinate_graph Example:
   Question: "Show the directed line segment from A(1,2) to B(4,6)"
   diagram_spec: {
       "diagram_type": "coordinate_graph",
       "x_range": [-1, 6],
       "y_range": [-1, 8],
       "points": [
           {"x": 1, "y": 2, "label": "A(1,2)", "color": "#e74c3c"},
           {"x": 4, "y": 6, "label": "B(4,6)", "color": "#3498db"}
       ],
       "lines": [{"from": [1, 2], "to": [4, 6], "label": "AB", "arrow": true}],
       "title": "Directed Line Segment AB"
   }
"""


# Single-question prompts by type
SINGLE_MCQ_PROMPT = """Generate ONE Multiple Choice Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

DIAGRAM REQUIREMENT: {diagram_instruction}

VALID DIAGRAM TYPES (use ONLY these):
{valid_diagram_types}

REQUIREMENTS:
1. The question MUST test the specific concept: "{concept_name}"
2. Include exactly 4 options (A, B, C, D) with only ONE correct answer
3. Distractors should be plausible but clearly incorrect
4. Difficulty level: {difficulty}
5. Marks: {marks}

{diagram_spec_guidance}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Clear question text testing {concept_name}",
    "question_type": "mcq",
    "options": [
        {{"label": "A", "content": "Option text", "is_correct": false}},
        {{"label": "B", "content": "Option text", "is_correct": true}},
        {{"label": "C", "content": "Option text", "is_correct": false}},
        {{"label": "D", "content": "Option text", "is_correct": false}}
    ],
    "correct_answer": "B",
    "solution": "Step-by-step explanation of why the answer is correct",
    "difficulty": "{difficulty}",
    "bloom_level": "remember|understand|apply|analyze",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false,
    "diagram_spec": null
}}

IF has_diagram is true, diagram_spec MUST include:
{{
    "subject": "{subject}",
    "diagram_type": "REQUIRED - must be one of the VALID DIAGRAM TYPES listed above",
    "title": "Descriptive title for THIS specific question",
    ... parameters SPECIFIC TO THIS QUESTION's values (not generic)
}}

CRITICAL: The diagram MUST show the EXACT values from THIS question. 
For example, if the question asks about inequality 2x-3<5 (solution: x<4), the number_line must show x<4 specifically, NOT a generic number line."""


SINGLE_SHORT_ANSWER_PROMPT = """Generate ONE Short Answer Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

DIAGRAM REQUIREMENT: {diagram_instruction}

VALID DIAGRAM TYPES (use ONLY these):
{valid_diagram_types}

REQUIREMENTS:
1. The question MUST test the specific concept: "{concept_name}"
2. Expected answer length: 2-4 sentences
3. Question should be direct and clear
4. Difficulty level: {difficulty}
5. Marks: {marks}

{diagram_spec_guidance}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Clear question testing {concept_name}",
    "question_type": "short_answer",
    "correct_answer": "Expected answer in 2-4 sentences",
    "solution": "Detailed explanation with key points",
    "difficulty": "{difficulty}",
    "bloom_level": "understand|apply|analyze",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false,
    "diagram_spec": null
}}

IF has_diagram is true, diagram_spec MUST include subject, diagram_type, and parameters SPECIFIC TO THIS QUESTION's values (not generic examples)."""


SINGLE_LONG_ANSWER_PROMPT = """Generate ONE Long Answer Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

DIAGRAM REQUIREMENT: {diagram_instruction}

VALID DIAGRAM TYPES (use ONLY these):
{valid_diagram_types}

REQUIREMENTS:
1. The question MUST comprehensively test: "{concept_name}"
2. Should require detailed explanation (1-2 paragraphs)
3. May include sub-parts (a), (b), (c) if appropriate
4. Difficulty level: {difficulty}
5. Marks: {marks}

{diagram_spec_guidance}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Detailed question testing {concept_name}",
    "question_type": "long_answer",
    "correct_answer": "Comprehensive expected answer with all key points",
    "solution": "Detailed solution with marking scheme breakdown",
    "difficulty": "{difficulty}",
    "bloom_level": "analyze|evaluate|create",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false,
    "diagram_spec": null
}}

IF has_diagram is true, diagram_spec MUST include subject, diagram_type, and parameters SPECIFIC TO THIS QUESTION's values."""


SINGLE_NUMERICAL_PROMPT = """Generate ONE Numerical/Problem-Solving Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

DIAGRAM REQUIREMENT: {diagram_instruction}

VALID DIAGRAM TYPES (use ONLY these):
{valid_diagram_types}

REQUIREMENTS:
1. The problem MUST apply the concept: "{concept_name}"
2. Use realistic values and scenarios
3. Include all necessary data in the question
4. Show step-by-step solution
5. Difficulty level: {difficulty}
6. Marks: {marks}

{diagram_spec_guidance}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Numerical problem applying {concept_name}. Given: [data]. Find: [what to calculate]",
    "question_type": "numerical",
    "correct_answer": "Final numerical answer with units",
    "solution": "Step 1: [explanation]\\nStep 2: [calculation]\\n...\\nFinal Answer: [value with units]",
    "difficulty": "{difficulty}",
    "bloom_level": "apply|analyze",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false,
    "diagram_spec": null
}}

IF has_diagram is true, diagram_spec MUST include subject, diagram_type, and parameters SPECIFIC TO THIS QUESTION's values."""


SINGLE_TRUE_FALSE_PROMPT = """Generate ONE True/False Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

REQUIREMENTS:
1. Statement MUST be clearly true OR clearly false
2. Should test understanding of: "{concept_name}"
3. Avoid ambiguous statements
4. Difficulty level: {difficulty}
5. Marks: {marks}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Statement about {concept_name}",
    "question_type": "true_false",
    "options": [
        {{"label": "A", "content": "True", "is_correct": true/false}},
        {{"label": "B", "content": "False", "is_correct": false/true}}
    ],
    "correct_answer": "True" or "False",
    "solution": "Explanation of why the statement is true/false",
    "difficulty": "{difficulty}",
    "bloom_level": "remember|understand",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false
}}"""


SINGLE_FILL_BLANKS_PROMPT = """Generate ONE Fill in the Blanks Question for the following concept.

SUBJECT: {subject}
GRADE: {grade}
CONCEPT TO TEST: {concept_name}
CONCEPT DESCRIPTION: {concept_description}
TARGET DIFFICULTY: {difficulty}

RELEVANT CONTEXT (from study material):
{context}

REQUIREMENTS:
1. Use _______ to indicate blank(s)
2. Should test key terms/concepts related to: "{concept_name}"
3. Blank should be for important keywords, not trivial words
4. Difficulty level: {difficulty}
5. Marks: {marks}

RESPOND WITH JSON (no markdown):
{{
    "question_text": "Statement with _______ for blank related to {concept_name}",
    "question_type": "fill_in_blanks",
    "correct_answer": "The word/phrase that fills the blank",
    "solution": "The complete statement with answer and brief explanation",
    "difficulty": "{difficulty}",
    "bloom_level": "remember|understand",
    "topic": "{concept_name}",
    "marks": {marks},
    "has_diagram": false
}}"""


# Valid diagram types by subject (for prompt inclusion)
VALID_DIAGRAM_TYPES_BY_SUBJECT = {
    "maths": [
        "coordinate_graph - For plotting points, vectors, lines, functions on 2D plane",
        "geometry_construction - For triangles, circles, angles, geometric shapes",
        "venn_diagram - For set theory problems",
        "number_line - For inequalities, intervals",
        "bar_chart - For statistics/data representation",
        "pie_chart - For percentage/proportion problems",
        "trigonometric_circle - For unit circle, trig functions",
        "3d_plot - ONLY for 3D surfaces like z=f(x,y), NOT for vectors",
    ],
    "physics": [
        "series_circuit - For series electrical circuits",
        "parallel_circuit - For parallel electrical circuits", 
        "mixed_circuit - For combined series-parallel circuits",
        "ray_diagram_lens - For convex/concave lens optics",
        "ray_diagram_mirror - For mirror reflection problems",
        "free_body_diagram - For force analysis problems",
        "inclined_plane - For problems involving slopes",
        "projectile_motion - For projectile trajectory problems",
        "wave_diagram - For wave properties (amplitude, wavelength)",
        "electric_field - For field line diagrams",
        "magnetic_field - For magnetic field patterns",
    ],
    "chemistry": [
        "molecule_2d - For 2D structural formulas",
        "molecule_3d - For 3D molecular geometry",
        "reaction_scheme - For chemical reaction equations",
        "orbital_diagram - For electron configurations",
        "lab_setup_titration - For titration apparatus",
        "lab_setup_distillation - For distillation setup",
        "lab_setup_electrolysis - For electrolysis apparatus",
    ],
    "biology": [
        "human_heart - For heart anatomy",
        "human_brain - For brain structure",
        "nephron - For kidney/nephron structure",
        "neuron - For nerve cell diagram",
        "plant_cell - For plant cell structure",
        "animal_cell - For animal cell structure",
        "dna_replication - For DNA processes",
        "mitosis_stages - For cell division stages",
        "meiosis_stages - For meiosis process",
        "digestive_system - For digestive tract",
        "respiratory_system - For respiratory anatomy",
        "eye_structure - For eye diagram",
        "ear_structure - For ear anatomy",
        "flower_structure - For flower parts",
    ],
}


# Diagram specification guidance by subject
DIAGRAM_SPEC_GUIDANCE = {
    "maths": """
DIAGRAM GUIDANCE for Mathematics:

The diagram generator can draw ANY mathematical diagram using basic shapes, lines, and labels.
Simply describe WHAT the diagram should show - the system will figure out HOW to draw it.

When specifying a diagram, provide:
{
    "subject": "maths",
    "title": "Descriptive title of what the diagram shows",
    "description": "Detailed description of what should be visualized"
}

EXAMPLES of good descriptions:
- "Number line from -5 to 10 showing the solution x < 4 with an open circle at 4 and shading to the left"
- "Coordinate plane with points A(1,2) and B(4,6) connected by a directed line segment (arrow from A to B)"
- "Right triangle ABC with right angle at C, where AC=3cm and BC=4cm, labeled sides and angles"
- "Circle with center O, radius r, showing tangent line at point P"

The system can draw:
- Number lines with points, intervals, open/closed circles
- Coordinate graphs with points, lines, arrows, curves
- Geometric shapes: triangles, circles, rectangles, polygons
- Angles with arc markers and labels
- Any combination of the above""",

    "physics": """
DIAGRAM GUIDANCE for Physics:

WHEN NOT TO USE DIAGRAMS:
- Conceptual/theoretical questions (e.g., "Explain Newton's first law")
- Historical questions (e.g., "Describe Galileo's experiment")
- Definition-based questions (e.g., "Define inertia")
- Questions that ask for written explanations without numerical data
For these, set has_diagram: false

WHEN TO USE DIAGRAMS:
- Numerical problems with specific values (mass, angle, velocity, etc.)
- Circuit analysis problems
- Ray optics problems with given distances
- Force analysis with given masses and forces

The diagram generator can draw ANY physics diagram using shapes, arrows, and labels.
Simply describe WHAT the diagram should show - include ALL numerical values from the question!

When specifying a diagram, provide:
{
    "subject": "physics",
    "title": "Descriptive title with key values",
    "description": "Detailed description of what should be visualized, including all numerical values"
}

EXAMPLES of good descriptions:
- "Inclined plane at 30° with a 5kg cylinder (radius 0.2m) on it. Show the weight force (W=49N pointing down), normal force (N perpendicular to surface), and friction force (f parallel to surface, opposing motion). Label the angle as θ=30°."
- "Free body diagram of a 10kg block on a horizontal surface. Forces: Weight W=98N pointing down, Normal force N=98N pointing up, Applied force F=50N pointing right, Friction f=20N pointing left."
- "Series circuit with a 12V battery connected to two resistors R1=4Ω and R2=6Ω. Show current direction with arrows."
- "Ray diagram for a convex lens (f=10cm). Object is 15cm from lens. Show three principal rays and the image formation."
- "Projectile motion showing a ball launched at 45 degrees with initial velocity 20 m/s. Show the parabolic path, initial velocity vector, and key points (max height, landing point)."

The system can draw:
- Force diagrams with labeled arrows (vectors)
- Inclined planes with objects and force vectors
- Circuit diagrams with components
- Ray diagrams for optics
- Motion diagrams with trajectories
- Any physics scenario with shapes, arrows, and labels""",

    "chemistry": """
DIAGRAM GUIDANCE for Chemistry:

The diagram generator can draw chemistry diagrams including molecular structures, 
lab setups, reaction diagrams, and more. Describe WHAT should be shown.

When specifying a diagram, provide:
{
    "subject": "chemistry",
    "title": "Descriptive title",
    "description": "Detailed description of what should be visualized"
}

EXAMPLES of good descriptions:
- "Structural formula of ethanol (C2H5OH) showing all carbon, hydrogen, and oxygen atoms with bonds"
- "Titration setup showing burette with NaOH solution above a conical flask containing HCl with indicator"
- "Energy diagram showing exothermic reaction with reactants at higher energy, products at lower energy, and activation energy barrier labeled"
- "Electron dot structure (Lewis structure) for water molecule H2O showing lone pairs on oxygen"

The system can draw:
- Molecular structures (two-dimensional representations)
- Lab equipment setups
- Energy diagrams
- Reaction mechanisms with arrows
- Periodic table elements with electron configurations""",

    "biology": """
DIAGRAM GUIDANCE for Biology:

The diagram generator can draw biology diagrams including cell structures, 
organ systems, and biological processes. Describe WHAT should be shown.

When specifying a diagram, provide:
{
    "subject": "biology",
    "title": "Descriptive title",
    "description": "Detailed description of what should be visualized"
}

EXAMPLES of good descriptions:
- "Plant cell showing cell wall, cell membrane, nucleus, chloroplasts, vacuole, and mitochondria. Label each organelle."
- "Human heart cross-section showing four chambers (left/right atrium, left/right ventricle), valves, and major blood vessels (aorta, pulmonary artery/vein, vena cava)"
- "Neuron showing cell body, dendrites, axon, myelin sheath, and axon terminals. Include arrows showing direction of nerve impulse."
- "DNA double helix structure showing sugar-phosphate backbone and base pairs (A-T, G-C)"

The system can draw:
- Cell diagrams (plant, animal, bacterial)
- Organ structures and systems
- Biological processes (mitosis, photosynthesis)
- Molecular structures (DNA, proteins)
- Ecological diagrams (food webs, cycles)""",
}


def get_single_question_prompt(
    question_type: QuestionType,
    subject: str,
    grade: str,
    concept_name: str,
    concept_description: str,
    difficulty: str,
    marks: int,
    context: str,
    requires_diagram: bool = False,
    diagram_type_hint: Optional[str] = None,
) -> str:
    """
    Get a formatted prompt for generating a single question.
    
    Args:
        question_type: Type of question to generate
        subject: Subject name
        grade: Grade level
        concept_name: Name of the concept to test
        concept_description: Description of the concept
        difficulty: Target difficulty (easy/medium/hard)
        marks: Marks for this question
        context: Relevant context retrieved via RAG
        requires_diagram: Whether this question should include a diagram
        diagram_type_hint: Suggested diagram type
        
    Returns:
        Formatted prompt string
    """
    # Select appropriate template
    template_map = {
        QuestionType.MCQ: SINGLE_MCQ_PROMPT,
        QuestionType.TRUE_FALSE: SINGLE_TRUE_FALSE_PROMPT,
        QuestionType.FILL_IN_BLANKS: SINGLE_FILL_BLANKS_PROMPT,
        QuestionType.SHORT_ANSWER: SINGLE_SHORT_ANSWER_PROMPT,
        QuestionType.LONG_ANSWER: SINGLE_LONG_ANSWER_PROMPT,
        QuestionType.NUMERICAL: SINGLE_NUMERICAL_PROMPT,
        QuestionType.MATCH_THE_FOLLOWING: SINGLE_SHORT_ANSWER_PROMPT,
    }
    
    template = template_map.get(question_type, SINGLE_MCQ_PROMPT)
    
    # Get valid diagram types for subject
    subject_lower = subject.lower()
    valid_types = VALID_DIAGRAM_TYPES_BY_SUBJECT.get(subject_lower, [])
    valid_diagram_types = "\n".join(f"- {t}" for t in valid_types)
    
    # Get diagram spec guidance
    diagram_guidance = DIAGRAM_SPEC_GUIDANCE.get(subject_lower, "")
    
    # Diagram instruction
    if requires_diagram:
        if diagram_type_hint:
            diagram_instruction = f"YES - Include a diagram. Suggested type: {diagram_type_hint}"
        else:
            diagram_instruction = "YES - Include an appropriate diagram for this concept"
    else:
        diagram_instruction = "NO - No diagram needed for this question"
    
    # Build diagram spec guidance section with critical rules for uniqueness
    if requires_diagram and question_type in [
        QuestionType.MCQ, QuestionType.SHORT_ANSWER, 
        QuestionType.LONG_ANSWER, QuestionType.NUMERICAL
    ]:
        diagram_spec_guidance = f"""
DIAGRAM SPECIFICATION (if has_diagram is true):
{diagram_guidance}

{DIAGRAM_CRITICAL_RULES}

IMPORTANT: 
- diagram_type MUST be one of the valid types listed above
- For vectors/position vectors in maths, use "coordinate_graph" with "lines" having "arrow": true
- Do NOT invent new diagram types
- The diagram MUST be SPECIFIC to THIS question values"""
    else:
        diagram_spec_guidance = ""
    
    # Format the prompt
    formatted_prompt = template.format(
        subject=subject,
        grade=grade,
        concept_name=concept_name,
        concept_description=concept_description,
        difficulty=difficulty,
        marks=marks,
        context=context[:6000],  # Limit context per question
        diagram_instruction=diagram_instruction,
        valid_diagram_types=valid_diagram_types,
        diagram_spec_guidance=diagram_spec_guidance,
    )
    
    # Add explicit diagram instruction when required (overrides the example JSON)
    if requires_diagram:
        diagram_override = f"""

**IMPORTANT - DIAGRAM IS REQUIRED FOR THIS QUESTION**
The example JSON above shows "has_diagram": false, but you MUST set:
- "has_diagram": true
- "diagram_spec": {{ a complete diagram specification with all required fields }}

Your diagram_spec MUST include:
- "subject": "{subject_lower}"
- "diagram_type": one of the valid types listed above (e.g., "coordinate_graph", "number_line", "series_circuit", etc.)
- All other parameters needed for that diagram type with SPECIFIC values from the question

DO NOT return "has_diagram": false or "diagram_spec": null - a diagram IS required."""
        formatted_prompt += diagram_override
    
    return formatted_prompt
