"""
Two-LLM Loop Prompts for RAG-based Question Generation.

Implements the Generator (LLM1) and Reviewer (LLM2) prompt system
as specified in the system architecture.

Key principles:
- Force strict JSON output (no extra text)
- Generator must cite source chunk IDs
- Reviewer must flag hallucination if content not grounded in context
- Revision prompt preserves valid parts
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# System Prompts
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are an expert educational content creator (LLM1 - Generator).

Your role is to generate high-quality exam questions based on provided study material context.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations, no extra text
2. ALL content must be grounded in the provided context
3. Include source_chunk_ids for traceability
4. Follow the exact JSON schema provided
5. When diagram is required, output diagram_spec (NOT images)
6. Generate ONE question per request

You are being evaluated by a Reviewer AI. Your question will be rejected if:
- Content is not supported by the provided context (hallucination)
- Question is ambiguous or unclear
- Format does not match requirements
- Diagram spec is incomplete or invalid"""


REVIEWER_SYSTEM_PROMPT = """You are an expert exam question quality reviewer (LLM2 - Reviewer).

Your role is to evaluate questions for:
1. ACCURACY: Is the content correct and grounded in the provided context?
2. CLARITY: Is the question unambiguous and clearly worded?
3. FORMAT: Does it follow exam standards and the required format?
4. DIFFICULTY: Does it match the target difficulty level?
5. DIAGRAM: If included, is the diagram_spec complete and valid?

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations
2. Flag ANY content not supported by the context as hallucination
3. Provide SPECIFIC, ACTIONABLE fix instructions
4. Be strict but fair - approve good questions
5. Do not suggest unnecessary improvements to valid questions"""


# =============================================================================
# LLM1 Generator Prompts
# =============================================================================

GENERATOR_PROMPT_TEMPLATE = """Generate ONE {question_type} question for the following requirements.

=== PAPER CONTEXT ===
{memory_context}

=== CURRENT QUESTION REQUIREMENTS ===
Subject: {subject}
Grade/Class: {grade}
Question Type: {question_type}
Target Difficulty: {difficulty}
Marks: {marks}
Topic Focus: {topic_focus}
Diagram Required: {diagram_required}

=== RETRIEVED STUDY MATERIAL (RAG Context) ===
The following chunks are from the uploaded study material. Your question MUST be based on this content.

{rag_context}

=== CHUNK IDs FOR CITATION ===
Available chunk IDs: {chunk_ids}

=== OUTPUT REQUIREMENTS ===
Respond with a single JSON object (no markdown, no extra text):

{{
    "question_text": "The complete question text",
    "question_type": "{question_type}",
    "options": {options_requirement},
    "correct_answer": "The correct answer",
    "explanation": "Detailed explanation of why this is correct",
    "marks": {marks},
    "difficulty": "{difficulty}",
    "topic": "Specific topic this question tests",
    "bloom_level": "remember|understand|apply|analyze|evaluate|create",
    "source_chunk_ids": ["id1", "id2"],
    "diagram_required": {diagram_required_bool},
    "diagram_tool": "matplotlib|schemdraw|rdkit|none",
    "diagram_type": "specific_type_from_tool",
    "diagram_spec": {diagram_spec_requirement},
    "diagram_rendering_notes": "print-friendly, clear labels, no clutter",
    "validation_checks": [
        "diagram matches question values",
        "labels are clear and unambiguous",
        "units and scales are correct",
        "no overlapping elements"
    ]
}}

{additional_instructions}"""



GENERATOR_MCQ_OPTIONS_REQUIREMENT = """[
        {{"label": "A", "content": "option text", "is_correct": false}},
        {{"label": "B", "content": "option text", "is_correct": true}},
        {{"label": "C", "content": "option text", "is_correct": false}},
        {{"label": "D", "content": "option text", "is_correct": false}}
    ] (exactly 4 options with ONE correct)"""


GENERATOR_NON_MCQ_OPTIONS_REQUIREMENT = "null"


GENERATOR_DIAGRAM_SPEC_REQUIREMENT = """{{
        "subject": "{subject}",
        "diagram_type": "MUST be one of the SUPPORTED TYPES listed below",
        "tool": "matplotlib|schemdraw|rdkit",
        "title": "Descriptive title matching the question",
        "description": "What the diagram shows",
        "params": {{
            // Diagram-specific parameters with EXACT values from the question
            // Physics circuits: components, voltages, resistances
            // Chemistry molecules: SMILES string (valid!)
            // Math graphs: functions, ranges, points
        }},
        "output": {{"format": "svg", "width": 600, "height": 400}},
        "rendering_notes": "print-friendly, clear labels, no clutter"
    }}

CRITICAL - SUPPORTED DIAGRAM TYPES (use ONLY these exact values):

PHYSICS (subject: "physics"):
  Circuits: "series_circuit", "parallel_circuit", "mixed_circuit"
  Optics: "ray_diagram_lens", "ray_diagram_mirror"
  Mechanics: "free_body_diagram", "inclined_plane", "projectile_motion"
  Waves: "wave_diagram"
  Electromagnetism: "electric_field", "magnetic_field"

CHEMISTRY (subject: "chemistry"):
  Molecules: "molecule_2d", "molecule_3d"
  Reactions: "reaction_scheme"
  Lab: "lab_setup_titration", "lab_setup_distillation", "lab_setup_electrolysis"
  Other: "orbital_diagram", "crystal_structure", "periodic_table_section"

MATH (subject: "math"):
  Graphs: "coordinate_graph", "number_line", "bar_chart", "pie_chart", "3d_plot"
  Geometry: "geometry_construction", "trigonometric_circle", "venn_diagram"

DO NOT use any diagram_type not listed above! If a diagram type isn't supported, set diagram_required=false.

TOOL SELECTION:
- Physics circuits → tool: "schemdraw", diagram_type: series_circuit/parallel_circuit/mixed_circuit
- Physics optics/mechanics/waves → tool: "matplotlib"
- Chemistry molecules (2D/3D structures) → tool: "rdkit" + valid SMILES
- Chemistry lab setups/orbitals → tool: "matplotlib"
- Math graphs/geometry → tool: "matplotlib"

COMMON SMILES (use these, do NOT hallucinate):
ethanol=CCO, benzene=c1ccccc1, phenol=Oc1ccccc1, acetone=CC(=O)C
acetic_acid=CC(=O)O, aniline=Nc1ccccc1, methanol=CO"""


GENERATOR_NO_DIAGRAM_REQUIREMENT = "null"


GENERATOR_ADDITIONAL_MCQ = """
MCQ-SPECIFIC RULES:
- Exactly 4 options (A, B, C, D)
- Only ONE correct answer
- Distractors should be plausible but clearly incorrect
- Options should be roughly equal in length
- Avoid "all of the above" or "none of the above" unless necessary"""


GENERATOR_ADDITIONAL_SHORT = """
SHORT ANSWER RULES:
- Question should require 2-4 sentences to answer
- Be specific about what you're asking
- Include any necessary context in the question"""


GENERATOR_ADDITIONAL_LONG = """
LONG ANSWER RULES:
- Question should require detailed explanation (1-2 paragraphs)
- May include sub-parts (a), (b), (c) if appropriate
- Provide comprehensive marking points in the solution"""


GENERATOR_ADDITIONAL_NUMERICAL = """
NUMERICAL PROBLEM RULES:
- Include all necessary data in the question
- Use realistic values
- Specify units required in the answer
- Show step-by-step solution in the explanation"""


# =============================================================================
# LLM2 Reviewer Prompts
# =============================================================================

REVIEWER_PROMPT_TEMPLATE = """Review the following question for quality and accuracy.

=== QUESTION TO REVIEW ===
```json
{question_draft}
```

=== SOURCE CONTEXT (Ground Truth) ===
The question should be based ONLY on this content:

{rag_context}

=== AVAILABLE SOURCE CHUNK IDs ===
{chunk_ids}

=== REVIEW CRITERIA ===
1. ACCURACY (concept_error, hallucination)
   - Is all content factually correct?
   - Is all content supported by the source context?
   - Are source_chunk_ids valid and relevant?

2. CLARITY (ambiguity)
   - Is the question clearly worded?
   - Is there only one correct interpretation?
   - Are options (if MCQ) unambiguous?

3. FORMAT (format)
   - Does it match the required question type: {question_type}?
   - Is the JSON structure correct?
   - For MCQ: exactly 4 options with 1 correct?

4. DIFFICULTY (difficulty_mismatch)
   - Does it match target difficulty: {target_difficulty}?
   - Is it appropriate for grade: {grade}?

5. DIAGRAM (diagram) - CRITICAL CHECK
   - If diagram_required, is diagram_spec complete?
   - **MOST IMPORTANT**: Does the diagram MATCH the question content?
     * If question asks about circuits, diagram must show circuits
     * If question mentions specific values (resistance, focal length, etc.), diagram params must use SAME values
     * The diagram should help answer the question, not confuse the student
   - Are diagram parameters specific to this question?
   - Is diagram_type valid for the subject?
   - REJECT if diagram shows something different from what question asks about

=== OUTPUT FORMAT ===
Respond with a single JSON object (no markdown, no extra text):

{{
    "approved": true/false,
    "issues": [
        {{
            "type": "concept_error|ambiguity|format|diagram|difficulty_mismatch|hallucination|source_chunk_ids|bloom_level|accuracy|completeness",
            "message": "Description of the issue",
            "fix_instructions": "SPECIFIC instructions on how to fix this"
        }}
    ],
    "required_changes": {{
        // Structured hints for revision, e.g.:
        // "question_text": "Needs rewording to...",
        // "difficulty": "Should be easier/harder",
        // "diagram_spec": {{"missing": "x_range", "issue": "..."}}
    }},
    "quality_score": 0.0-1.0
}}

If approved=true, issues should be empty or contain only minor suggestions.
If approved=false, issues MUST contain at least one item with fix_instructions."""


# =============================================================================
# LLM1 Revision Prompts
# =============================================================================

REVISION_PROMPT_TEMPLATE = """**CRITICAL: YOU MUST REVISE THE QUESTION - DO NOT REPEAT THE SAME OUTPUT**

Your previous question was REJECTED by the reviewer. You MUST fix the issues listed below.
If you output the same content again, this question will FAIL permanently.

=== YOUR REJECTED QUESTION ===
```json
{original_draft}
```

=== REVIEWER FEEDBACK (MUST BE ADDRESSED) ===
Approved: {approved}

**ISSUES TO FIX (MANDATORY):**
{issues_formatted}

**REQUIRED CHANGES:**
{required_changes_formatted}
{diagram_feedback}

=== SOURCE CONTEXT (Ground Truth) ===
{rag_context}

=== REVISION REQUIREMENTS (ALL MANDATORY) ===
**YOU MUST:**
1. ✓ Address EVERY issue listed above - do not skip any
2. ✓ Make VISIBLE changes to the flagged content
3. ✓ Ensure diagram_spec values MATCH the question text exactly
4. ✓ Use ONLY information from the source context (no hallucination)
5. ✓ Keep question type, difficulty, and marks the same

**COMMON MISTAKES TO AVOID:**
- DON'T just rephrase - actually FIX the conceptual issues
- DON'T change the diagram_type if it was correct - only fix params
- DON'T output the same diagram_spec params with different question text
- The diagram MUST illustrate what the question asks about

**DIAGRAM-QUESTION CONSISTENCY CHECK:**
- If question mentions resistance values → diagram params must have SAME values
- If question asks about a lens → diagram must be ray_diagram_lens, not circuits
- If question mentions angles/distances → diagram params must have SAME values

=== DIAGRAM SPEC REQUIREMENTS (if diagram_required=true) ===
Your diagram_spec MUST include:
- "subject": "{subject}" (lowercase)
- "diagram_type": One of the EXACT supported types below
- "params": Object with CORRECT values matching the question

{diagram_types_reference}

=== OUTPUT FORMAT ===
Output ONLY the REVISED JSON object (no markdown, no explanations):

{{
    "question_text": "REVISED text addressing feedback",
    "question_type": "{question_type}",
    "options": [...],
    "correct_answer": "...",
    "explanation": "...",
    "marks": {marks},
    "difficulty": "{difficulty}",
    "topic": "...",
    "bloom_level": "...",
    "source_chunk_ids": [...],
    "diagram_required": {diagram_required},
    "diagram_spec": {{
        "subject": "{subject}",
        "diagram_type": "type_matching_question",
        "params": {{ /* VALUES MUST MATCH QUESTION */ }},
        "title": "Title matching question",
        "description": "Description of what diagram shows"
    }}
}}"""


# =============================================================================
# Helper Functions
# =============================================================================

def build_generator_prompt(
    question_type: str,
    subject: str,
    grade: str,
    difficulty: str,
    marks: int,
    topic_focus: str,
    diagram_required: bool,
    rag_context: str,
    chunk_ids: List[str],
    memory_context: str,
) -> str:
    """
    Build the generator prompt for LLM1.
    
    Args:
        question_type: Type of question (mcq, short, long, numerical)
        subject: Subject name
        grade: Grade/class level
        difficulty: Target difficulty (easy, medium, hard)
        marks: Marks for this question
        topic_focus: Specific topic to focus on
        diagram_required: Whether diagram is required
        rag_context: Retrieved context from Qdrant
        chunk_ids: List of available chunk IDs for citation
        memory_context: Paper generation state context
        
    Returns:
        Formatted prompt string
    """
    # Determine options requirement
    if question_type.lower() == "mcq":
        options_req = GENERATOR_MCQ_OPTIONS_REQUIREMENT
        additional = GENERATOR_ADDITIONAL_MCQ
    elif question_type.lower() in ["short", "short_answer"]:
        options_req = GENERATOR_NON_MCQ_OPTIONS_REQUIREMENT
        additional = GENERATOR_ADDITIONAL_SHORT
    elif question_type.lower() in ["long", "long_answer"]:
        options_req = GENERATOR_NON_MCQ_OPTIONS_REQUIREMENT
        additional = GENERATOR_ADDITIONAL_LONG
    elif question_type.lower() == "numerical":
        options_req = GENERATOR_NON_MCQ_OPTIONS_REQUIREMENT
        additional = GENERATOR_ADDITIONAL_NUMERICAL
    else:
        options_req = GENERATOR_NON_MCQ_OPTIONS_REQUIREMENT
        additional = ""
    
    # Determine diagram spec requirement
    if diagram_required:
        diagram_spec_req = GENERATOR_DIAGRAM_SPEC_REQUIREMENT.format(subject=subject.lower())
    else:
        diagram_spec_req = GENERATOR_NO_DIAGRAM_REQUIREMENT
    
    result = GENERATOR_PROMPT_TEMPLATE.format(
        question_type=question_type,
        subject=subject,
        grade=grade,
        difficulty=difficulty,
        marks=marks,
        topic_focus=topic_focus,
        diagram_required="Yes - include a diagram" if diagram_required else "No",
        diagram_required_bool=str(diagram_required).lower(),
        rag_context=rag_context,
        chunk_ids=", ".join(chunk_ids),
        memory_context=memory_context,
        options_requirement=options_req,
        diagram_spec_requirement=diagram_spec_req,
        additional_instructions=additional,
    )
    
    return result


def build_reviewer_prompt(
    question_draft: Dict[str, Any],
    rag_context: str,
    chunk_ids: List[str],
    question_type: str,
    target_difficulty: str,
    grade: str,
    is_fallback_mode: bool = False,
) -> str:
    """
    Build the reviewer prompt for LLM2.
    
    Args:
        question_draft: The question draft from LLM1
        rag_context: Retrieved context from Qdrant
        chunk_ids: List of valid chunk IDs
        question_type: Expected question type
        target_difficulty: Target difficulty level
        grade: Grade/class level
        is_fallback_mode: Whether LLM1 generated without RAG context
        
    Returns:
        Formatted prompt string
    """
    import json
    
    base_prompt = REVIEWER_PROMPT_TEMPLATE.format(
        question_draft=json.dumps(question_draft, indent=2),
        rag_context=rag_context if rag_context else "No RAG context available (fallback mode)",
        chunk_ids=", ".join(chunk_ids) if chunk_ids else "None (fallback mode)",
        question_type=question_type,
        target_difficulty=target_difficulty,
        grade=grade,
    )
    
    if is_fallback_mode:
        fallback_instructions = """

**FALLBACK MODE ACTIVE - SPECIAL INSTRUCTIONS:**
This question was generated WITHOUT RAG context (LLM knowledge only).

DO NOT reject for:
- hallucination (there's no source to verify against)
- source_chunk_ids (no chunks were available)

STILL CHECK ALL OTHER CRITERIA:
- Factual accuracy based on your knowledge
- Clarity and proper formatting
- Difficulty appropriateness

**CRITICAL - DIAGRAM-QUESTION CONSISTENCY (ALWAYS CHECK THIS):**
If diagram_required=true, you MUST verify:
- The diagram_spec diagram_type matches what the question is asking about
- If question asks about circuits, diagram must be a circuit type
- If question asks about optics/lenses, diagram must be optics type
- Values in diagram params must match values mentioned in question text
- REJECT with issue type "diagram" if there's ANY mismatch between question and diagram

This is the MOST IMPORTANT check - a diagram that doesn't match the question is worse than no diagram!
"""
        base_prompt += fallback_instructions
    
    return base_prompt



def build_reviewer_prompt(
    question_draft: Dict[str, Any],
    rag_context: str,
    chunk_ids: List[str],
    question_type: str,
    target_difficulty: str,
    grade: str,
    is_fallback_mode: bool = False,
) -> str:
    """
    Build the reviewer prompt for LLM2.
    
    Args:
        question_draft: The question draft from LLM1
        rag_context: Retrieved context from Qdrant
        chunk_ids: List of valid chunk IDs
        question_type: Expected question type
        target_difficulty: Target difficulty level
        grade: Grade/class level
        is_fallback_mode: Whether using LLM knowledge fallback
        
    Returns:
        Formatted prompt string
    """
    import json
    
    base_prompt = REVIEWER_PROMPT_TEMPLATE.format(
        question_draft=json.dumps(question_draft, indent=2),
        rag_context=rag_context,
        chunk_ids=", ".join(chunk_ids) if chunk_ids else "None (fallback mode)",
        question_type=question_type,
        target_difficulty=target_difficulty,
        grade=grade,
    )
    
    if is_fallback_mode:
        fallback_instructions = """

=== IMPORTANT: FALLBACK MODE ACTIVE ===
No source documents are available. The question was generated using LLM knowledge.

CRITICAL REVIEW RULES FOR FALLBACK MODE:
1. DO NOT reject for "hallucination" - there are no source documents to check against
2. DO NOT reject for "source_chunk_ids" issues - chunk IDs are not applicable
3. DO NOT reject for "accuracy" related to source documents

FOCUS YOUR REVIEW ON:
- Question clarity and unambiguous wording
- Correct JSON format and structure
- Appropriate difficulty level for the grade
- Conceptual correctness based on general knowledge
- Bloom's taxonomy level appropriateness

**CRITICAL - DIAGRAM-QUESTION CONSISTENCY (ALWAYS CHECK THIS):**
If diagram_required=true, you MUST verify:
- The diagram_spec diagram_type matches what the question is asking about
- If question asks about circuits, diagram must be a circuit type
- If question asks about optics/lenses, diagram must be optics type
- Values in diagram params must match values mentioned in question text
- REJECT with issue type "diagram" if there's ANY mismatch between question and diagram

If the question is well-formed, clear, conceptually correct, AND diagram matches question, APPROVE IT.
Reject for formatting errors, ambiguity, OR diagram-question mismatch.
"""
        return fallback_instructions + "\n" + base_prompt
    
    return base_prompt


def build_revision_prompt(
    original_draft: Dict[str, Any],
    review_result: Dict[str, Any],
    rag_context: str,
    question_type: str,
    marks: int,
    difficulty: str,
    diagram_required: bool,
    subject: str = "",
    diagram_feedback: str = "",
) -> str:
    """
    Build the revision prompt for LLM1.
    
    Args:
        original_draft: The original question draft
        review_result: The review result from LLM2
        rag_context: Retrieved context from Qdrant
        question_type: Question type
        marks: Marks for this question
        difficulty: Target difficulty
        diagram_required: Whether diagram is required
        subject: Subject for diagram type reference
        diagram_feedback: Specific feedback for diagram corrections
        
    Returns:
        Formatted prompt string
    """
    import json
    
    # Format issues
    issues = review_result.get("issues", [])
    if issues:
        issues_formatted = "\n".join([
            f"- [{issue.get('type', 'unknown')}] {issue.get('message', '')}\n"
            f"  Fix: {issue.get('fix_instructions', 'No instructions provided')}"
            for issue in issues
        ])
    else:
        issues_formatted = "No specific issues identified"
    
    # Format required changes
    required_changes = review_result.get("required_changes", {})
    if required_changes:
        required_changes_formatted = json.dumps(required_changes, indent=2)
    else:
        required_changes_formatted = "None specified"
    
    # Format diagram feedback section
    if diagram_feedback:
        diagram_feedback_section = f"\n\n=== DIAGRAM SPECIFIC ISSUES ===\n{diagram_feedback}\n"
    else:
        diagram_feedback_section = ""
    
    # Build diagram types reference based on subject
    diagram_types_reference = _get_diagram_types_reference(subject.lower() if subject else "")
    
    return REVISION_PROMPT_TEMPLATE.format(
        original_draft=json.dumps(original_draft, indent=2),
        approved=review_result.get("approved", False),
        issues_formatted=issues_formatted,
        required_changes_formatted=required_changes_formatted,
        diagram_feedback=diagram_feedback_section,
        rag_context=rag_context,
        question_type=question_type,
        marks=marks,
        difficulty=difficulty,
        diagram_required=str(diagram_required).lower(),
        subject=subject.lower() if subject else "physics",
        diagram_types_reference=diagram_types_reference,
    )


def _get_diagram_types_reference(subject: str) -> str:
    """Get supported diagram types reference for a subject."""
    references = {
        "physics": """PHYSICS SUPPORTED TYPES:
  Circuits (uses schemdraw): series_circuit, parallel_circuit, mixed_circuit
    Required params: components (list), voltage (number)
  
  Ray Optics: ray_diagram_lens, ray_diagram_mirror
    Required params: lens_type/mirror_type, focal_length, object_distance
  
  Wave Optics: interference_pattern, double_slit_experiment, diffraction_pattern, single_slit_diffraction
    Required params: slit_separation (mm), wavelength (nm), screen_distance (m)
    For Young's double-slit: use "interference_pattern" or "double_slit_experiment"
  
  Mechanics: free_body_diagram, inclined_plane, projectile_motion
    Required params: forces (list) / angle, mass / initial_velocity, angle
  
  Waves: wave_diagram, standing_wave, wave_superposition
    Required params: amplitude, wavelength
  
  Thermodynamics: pv_diagram, carnot_cycle
    Required params: process_type / t_hot, t_cold
  
  Modern Physics: photoelectric_effect, atomic_spectrum
    Required params: metal, wavelength / element
  
  Fields: electric_field, magnetic_field""",
        
        "chemistry": """CHEMISTRY SUPPORTED TYPES:
  Molecules (uses RDKit): molecule_2d, molecule_3d
    REQUIRED params: smiles (SMILES string like "CCO" for ethanol)
    Common SMILES: water="O", methane="C", ethanol="CCO", benzene="c1ccccc1"
  Reactions: reaction_scheme
    Required params: reactants (list), products (list)
  Lab setups: lab_setup_titration, lab_setup_distillation, lab_setup_electrolysis
  Other: orbital_diagram, crystal_structure, periodic_table_section""",
        
        "math": """MATH SUPPORTED TYPES:
  Graphs: coordinate_graph, number_line, bar_chart, pie_chart, 3d_plot
    For coordinate_graph: functions (list with expression, label)
    For charts: data (list), labels (list)
  Geometry: geometry_construction, trigonometric_circle, venn_diagram
    For geometry: shapes (list with type, coordinates)
    For trigonometric_circle: angle (degrees)
    For venn_diagram: sets (list with name, elements)""",
        
        "maths": """MATH SUPPORTED TYPES:
  Graphs: coordinate_graph, number_line, bar_chart, pie_chart, 3d_plot
    For coordinate_graph: functions (list with expression, label)
    For charts: data (list), labels (list)
  Geometry: geometry_construction, trigonometric_circle, venn_diagram
    For geometry: shapes (list with type, coordinates)
    For trigonometric_circle: angle (degrees)
    For venn_diagram: sets (list with name, elements)""",
        
        "biology": """BIOLOGY SUPPORTED TYPES:
  Cells: plant_cell, animal_cell
  Anatomy: human_heart, human_brain, nephron, neuron, eye_structure, ear_structure
  Systems: digestive_system, respiratory_system
  Processes: dna_replication, mitosis_stages, meiosis_stages
  Botany: flower_structure""",
    }
    
    return references.get(subject, "No specific diagram types for this subject.")


def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for prompts.
    
    Args:
        chunks: List of chunks from Qdrant search
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("id", f"chunk_{i}")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Build metadata string
        meta_parts = []
        if metadata.get("topic"):
            meta_parts.append(f"Topic: {metadata['topic']}")
        if metadata.get("page_no"):
            meta_parts.append(f"Page: {metadata['page_no']}")
        
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
        
        context_parts.append(
            f"[Chunk {chunk_id}{meta_str}]\n{text}\n"
        )
    
    return "\n---\n".join(context_parts)


# =============================================================================
# JSON Schema Validation
# =============================================================================

QUESTION_DRAFT_SCHEMA = {
    "type": "object",
    "required": [
        "question_text",
        "question_type",
        "correct_answer",
        "explanation",
        "marks",
        "difficulty",
        "source_chunk_ids",
        "diagram_required",
    ],
    "properties": {
        "question_text": {"type": "string", "minLength": 10},
        "question_type": {"type": "string", "enum": ["mcq", "short", "short_answer", "long", "long_answer", "numerical"]},
        "options": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "content", "is_correct"],
                        "properties": {
                            "label": {"type": "string"},
                            "content": {"type": "string"},
                            "is_correct": {"type": "boolean"},
                        },
                    },
                },
            ],
        },
        "correct_answer": {"type": "string", "minLength": 1},
        "explanation": {"type": "string", "minLength": 10},
        "marks": {"type": "integer", "minimum": 1},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "topic": {"type": "string"},
        "bloom_level": {"type": "string"},
        "source_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "diagram_required": {"type": "boolean"},
        "diagram_spec": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": ["subject", "diagram_type"],
                    "properties": {
                        "subject": {"type": "string"},
                        "diagram_type": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "params": {"type": "object"},
                        "output": {"type": "object"},
                    },
                },
            ],
        },
    },
}


REVIEW_RESULT_SCHEMA = {
    "type": "object",
    "required": ["approved", "issues"],
    "properties": {
        "approved": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "message", "fix_instructions"],
                "properties": {
                    "type": {"type": "string", "enum": [
                        "concept_error", "ambiguity", "format",
                        "diagram", "difficulty_mismatch", "hallucination",
                        "source_chunk_ids", "bloom_level", "accuracy", "completeness",
                    ]},
                    "message": {"type": "string"},
                    "fix_instructions": {"type": "string"},
                },
            },
        },
        "required_changes": {"type": "object"},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def validate_question_draft(draft: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a question draft against the schema.
    
    Args:
        draft: Question draft to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required fields
    required = ["question_text", "question_type", "correct_answer", "marks", "difficulty"]
    for field in required:
        if field not in draft:
            errors.append(f"Missing required field: {field}")
        elif not draft[field]:
            errors.append(f"Empty required field: {field}")
    
    if errors:
        return False, errors
    
    # Validate question_text length
    if len(draft.get("question_text", "")) < 10:
        errors.append("question_text is too short (minimum 10 characters)")
    
    # Validate question_type
    valid_types = ["mcq", "short", "short_answer", "long", "long_answer", "numerical"]
    if draft.get("question_type", "").lower() not in valid_types:
        errors.append(f"Invalid question_type: {draft.get('question_type')}")
    
    # Validate MCQ options
    if draft.get("question_type", "").lower() == "mcq":
        options = draft.get("options", [])
        if not options or len(options) != 4:
            errors.append("MCQ must have exactly 4 options")
        else:
            correct_count = sum(1 for opt in options if opt.get("is_correct"))
            if correct_count != 1:
                errors.append(f"MCQ must have exactly 1 correct option, found {correct_count}")
    
    # Validate difficulty
    valid_difficulties = ["easy", "medium", "hard"]
    if draft.get("difficulty", "").lower() not in valid_difficulties:
        errors.append(f"Invalid difficulty: {draft.get('difficulty')}")
    
    # Validate marks
    marks = draft.get("marks")
    if not isinstance(marks, int) or marks < 1:
        errors.append(f"Invalid marks: {marks}")
    
    # Validate diagram_spec if diagram_required
    if draft.get("diagram_required"):
        diagram_spec = draft.get("diagram_spec")
        if not diagram_spec:
            errors.append("diagram_spec is required when diagram_required is true")
        elif not isinstance(diagram_spec, dict):
            errors.append("diagram_spec must be an object")
        else:
            if "subject" not in diagram_spec:
                errors.append("diagram_spec missing required field: subject")
            if "diagram_type" not in diagram_spec:
                errors.append("diagram_spec missing required field: diagram_type")
    
    # Validate source_chunk_ids (handle null from LLM)
    chunk_ids = draft.get("source_chunk_ids") or []
    if not isinstance(chunk_ids, list):
        errors.append("source_chunk_ids must be an array")
    
    return len(errors) == 0, errors


def validate_review_result(result: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a review result against the schema.
    
    Args:
        result: Review result to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required fields
    if "approved" not in result:
        errors.append("Missing required field: approved")
    elif not isinstance(result["approved"], bool):
        errors.append("approved must be a boolean")
    
    if "issues" not in result:
        errors.append("Missing required field: issues")
    elif result["issues"] is not None and not isinstance(result["issues"], list):
        errors.append("issues must be an array")
    
    if errors:
        return False, errors
    
    # If not approved, must have at least one issue
    if not result["approved"] and len(result.get("issues") or []) == 0:
        errors.append("If not approved, at least one issue must be provided")
    
    # Validate each issue
    valid_issue_types = [
        "concept_error", "ambiguity", "format",
        "diagram", "difficulty_mismatch", "hallucination",
        "source_chunk_ids",  # Issues with source attribution/chunk references
        "bloom_level",  # Issues with Bloom's taxonomy level
        "accuracy",  # General accuracy issues
        "completeness",  # Missing required information
    ]
    for i, issue in enumerate(result.get("issues") or []):
        if not isinstance(issue, dict):
            errors.append(f"Issue {i} must be an object")
            continue
        
        if "type" not in issue:
            errors.append(f"Issue {i} missing required field: type")
        elif issue["type"] not in valid_issue_types:
            errors.append(f"Issue {i} has invalid type: {issue['type']}")
        
        if "message" not in issue or not issue["message"]:
            errors.append(f"Issue {i} missing or empty field: message")
        
        if "fix_instructions" not in issue or not issue["fix_instructions"]:
            errors.append(f"Issue {i} missing or empty field: fix_instructions")
    
    # Validate quality_score if present
    quality_score = result.get("quality_score")
    if quality_score is not None:
        if not isinstance(quality_score, (int, float)):
            errors.append("quality_score must be a number")
        elif quality_score < 0 or quality_score > 1:
            errors.append("quality_score must be between 0 and 1")
    
    return len(errors) == 0, errors
