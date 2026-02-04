# Diagram-Based Question Generation System

## Overview

This document describes the enhanced question generation system that supports **exam-style diagram-based questions** (JEE/NEET level) using a two-LLM agentic loop.

## Architecture

### Two-LLM Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Question Generation Loop                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LLM1 (Generator)                    LLM2 (Reviewer)                 │
│  ┌─────────────────┐                 ┌─────────────────┐             │
│  │ Generate        │ ───────────────→│ Review          │             │
│  │ question +      │                 │ - Accuracy      │             │
│  │ diagram_spec    │ ←───────────────│ - Clarity       │             │
│  │                 │   feedback      │ - Diagram       │             │
│  └─────────────────┘                 │ - Difficulty    │             │
│         │                            └─────────────────┘             │
│         │ approved                                                    │
│         ↓                                                             │
│  ┌─────────────────┐                                                 │
│  │ Diagram         │                                                 │
│  │ Renderer        │ ──→ SVG/PNG image                               │
│  │ (matplotlib/    │                                                 │
│  │  schemdraw/     │                                                 │
│  │  RDKit)         │                                                 │
│  └─────────────────┘                                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagram Tool Selection

| Diagram Type | Tool | Use Case |
|--------------|------|----------|
| series_circuit, parallel_circuit, RC/RL/RLC | **schemdraw** | Physics circuit diagrams |
| ray_diagram_lens, ray_diagram_mirror | **matplotlib** | Optics diagrams |
| free_body_diagram, inclined_plane, projectile_motion | **matplotlib** | Mechanics diagrams |
| molecule_2d, molecule_3d, reaction_mechanism | **RDKit** | Chemistry molecular structures |
| orbital_diagram, titration_curve, energy_level | **matplotlib** | Chemistry other |
| coordinate_graph, function_graph, geometry | **matplotlib** | Math diagrams |

## Output Format

### QuestionDraft Model

```python
class QuestionDraft(BaseModel):
    question_text: str
    question_type: str  # mcq, short, long, numerical
    options: Optional[List[Dict[str, Any]]] = None
    correct_answer: str
    explanation: str
    marks: int
    difficulty: str  # easy, medium, hard
    source_chunk_ids: List[str]
    
    # Enhanced diagram fields
    diagram_required: bool = False
    diagram_tool: Optional[str] = None  # matplotlib, schemdraw, rdkit
    diagram_type: Optional[str] = None  # specific type
    diagram_spec: Optional[Dict[str, Any]] = None
    diagram_rendering_notes: Optional[str] = None
    
    # Validation
    validation_checks: List[str] = []  # Quality checklist
    
    topic: Optional[str] = None
    bloom_level: Optional[str] = None
```

### Diagram Spec Structure

```json
{
    "subject": "physics",
    "diagram_type": "series_circuit",
    "tool": "schemdraw",
    "title": "Series Circuit with Two Resistors",
    "description": "Circuit showing battery and resistors in series",
    "params": {
        "circuit_type": "series",
        "components": [
            {"type": "battery", "voltage": "12V", "label": "V"},
            {"type": "resistor", "value": "100Ω", "label": "R1"},
            {"type": "resistor", "value": "200Ω", "label": "R2"}
        ],
        "show_current_arrows": true,
        "show_labels": true
    },
    "output": {"format": "svg", "width": 600, "height": 400},
    "rendering_notes": "print-friendly, clear labels, no clutter"
}
```

### Chemistry Molecular Structure (RDKit)

```json
{
    "subject": "chemistry",
    "diagram_type": "molecule_2d",
    "tool": "rdkit",
    "title": "Benzene Structure",
    "params": {
        "smiles": "c1ccccc1",
        "molecule_name": "Benzene",
        "show_hydrogens": true
    },
    "output": {"format": "svg", "width": 400, "height": 300}
}
```

## Key Files

| File | Description |
|------|-------------|
| `diagram_spec_helpers.py` | Tool mapping, templates, validation |
| `diagram_renderer_service.py` | Renders specs to images |
| `rag_prompts.py` | Enhanced prompts with diagram instructions |
| `llm_orchestrator.py` | Two-LLM loop implementation |
| `paper_generation_worker.py` | Integrates diagram rendering |
| `models/job.py` | Enhanced QuestionDraft model |

## Common SMILES for Chemistry

| Molecule | SMILES |
|----------|--------|
| Ethanol | CCO |
| Methanol | CO |
| Benzene | c1ccccc1 |
| Phenol | Oc1ccccc1 |
| Acetic acid | CC(=O)O |
| Acetone | CC(=O)C |
| Aniline | Nc1ccccc1 |
| Ethene | C=C |
| Ethyne | C#C |

## Iteration Rules

1. **LLM1** generates question + diagram_spec (if required)
2. **LLM2** reviews for accuracy, clarity, diagram validity
3. If rejected, LLM1 revises (preserving valid parts)
4. Repeat until approved or max 3 iterations
5. After approval, diagram is rendered via appropriate tool
6. Rendered image stored as base64 data URL

## Quality Constraints

- Diagrams must use **EXACT values from the question**
- Labels must be **crisp and readable** when printed
- **No placeholder values** (e.g., default 30° when question says 45°)
- **Valid SMILES only** for RDKit (no hallucination)
- **Exam-style simplicity** - avoid artistic styling
