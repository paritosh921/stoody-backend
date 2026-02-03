# Diagram Generation System Refactor

## Overview

This document describes the refactored diagram generation system that addresses the issues outlined in `diagram_refactor_spec.md`.

## Key Changes

### 1. Structured DiagramSpec (instead of raw drawing code)

**Before:** LLMs wrote matplotlib/schemdraw drawing code directly.

**After:** LLMs fill a structured JSON `DiagramSpec` with parameters:

```python
spec = DiagramSpec(
    subject=DiagramSubject.PHYSICS,
    diagram_type="inclined_plane",
    parameters={
        "angle": 30,       # Extracted from question
        "mass": 10,        # kg
        "object_type": "box",
    },
    labels=[...],
)
```

### 2. Strict Library Selection via SpecRouter

**Before:** Library selection relied on fuzzy type mapping that could fail.

**After:** `SpecRouter` enforces correct library selection:

| Diagram Type | Required Library |
|--------------|------------------|
| `series_circuit`, `parallel_circuit`, `mixed_circuit` | SchemDraw |
| `molecule_2d`, `molecule_3d` | RDKit |
| All others | matplotlib |

```python
router = get_spec_router()
decision = router.route(spec)
# decision.renderer_type tells you exactly which library to use
```

### 3. Feedback Updates Spec (not prompts)

**Before:** Feedback was included in regeneration prompts but often ignored.

**After:** `FeedbackHandler` parses feedback into specific corrections:

```python
handler = get_feedback_handler()
state = handler.create_state(spec)

# After verification fails:
corrections = handler.parse_feedback(feedback_text, state)
# Returns: [Correction(parameter="angle", old_value=30, new_value=45)]

state = handler.apply_corrections(state, corrections)
updated_spec = state.get_updated_spec()
```

### 4. Persisted State Across Iterations

**Before:** Each regeneration started fresh, losing context.

**After:** `FeedbackState` persists across iterations:

```python
state.corrections_applied  # All corrections made
state.feedback_texts       # History of feedback
state.iteration           # Current iteration number
```

## New Files

| File | Purpose |
|------|---------|
| `specs/diagram_spec.py` | DiagramSpec model, parameter models, type enums |
| `spec_router.py` | Routes specs to correct renderers |
| `feedback_handler.py` | Parses feedback, applies corrections |
| `spec_based_service.py` | New service using spec-based approach |
| `exam_quality.py` | Exam-ready quality settings |

## Usage

### Option 1: Use SpecBasedDiagramService (recommended for new code)

```python
from services.diagram_engine.spec_based_service import get_spec_based_service

service = get_spec_based_service(
    kimi_service=kimi,
    use_kimi=True,
)

result = await service.generate_diagram(
    question_text="A 5 kg block on a 30° incline...",
    subject="physics",
)

if result.success:
    image_bytes = result.image_bytes
    # result.spec contains the final spec
    # result.corrections_applied shows what was fixed
```

### Option 2: Integrate with existing VerifiedDiagramService

The existing service can use the new spec system by:

1. Creating a `DiagramSpec` from the plan
2. Using `SpecRouter` for library selection
3. Using `FeedbackHandler` for corrections

```python
from services.diagram_engine.specs.diagram_spec import create_spec_from_plan

# In generate_verified_diagram:
spec = create_spec_from_plan(plan)
router = get_spec_router()
decision = router.route(spec)

if decision.renderer_type == RendererType.SCHEMDRAW:
    # Use schemdraw
elif decision.renderer_type == RendererType.RDKIT:
    # Use rdkit
else:
    # Use matplotlib
```

## Parameter Models

Each diagram type has a typed parameter model:

### Physics
- `CircuitParameters` - components, voltage, show_values
- `InclinedPlaneParameters` - angle, mass, object_type, show_forces
- `ProjectileParameters` - initial_velocity, angle, show_trajectory
- `OpticsParameters` - lens_type/mirror_type, focal_length, object_distance
- `WaveParameters` - amplitude, wavelength, wave_type
- `FreeBodyParameters` - forces, body_shape

### Chemistry
- `MoleculeParameters` - smiles (REQUIRED), molecule_name
- `ReactionParameters` - reactants, products, conditions
- `LabSetupParameters` - various setup-specific fields

### Maths
- `CoordinateGraphParameters` - functions, x_range, y_range
- `GeometryParameters` - shapes, angles, labels
- `NumberLineParameters` - min_value, max_value, points

## Common SMILES Reference

For molecule diagrams, use standard SMILES:

| Molecule | SMILES |
|----------|--------|
| Methane | `C` |
| Ethane | `CC` |
| Ethanol | `CCO` |
| Water | `O` |
| Ammonia | `N` |
| Carbon tetrachloride | `ClC(Cl)(Cl)Cl` |
| Benzene | `c1ccccc1` |
| Chloroform | `ClC(Cl)Cl` |

## Exam Quality Settings

Use `ExamQualitySettings` for exam-ready output:

```python
from services.diagram_engine.exam_quality import get_preset

# JEE preset (smaller fonts, standard sizes)
settings = get_preset('jee')

# NEET preset (larger fonts for biology)
settings = get_preset('neet')

# Apply to matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update(settings.get_matplotlib_rcparams())
```

## Testing

Run the test suite:

```bash
pytest tests/test_diagram_spec_refactor.py -v
```

## Migration Guide

### For existing code using VerifiedDiagramService:

1. The existing service continues to work as before
2. Optionally switch to `SpecBasedDiagramService` for better spec handling
3. Or integrate the new components incrementally:
   - Use `SpecRouter` for routing decisions
   - Use `FeedbackHandler` for correction parsing

### For new diagram types:

1. Add to appropriate `*DiagramType` enum in `diagram_spec.py`
2. Create parameter model if needed
3. Add to `PARAMETER_MODELS` dict
4. Add to `SpecRouter._type_to_renderer`
5. Implement renderer method if not using existing renderer

## Troubleshooting

### Wrong library being used

Check the routing decision:
```python
decision = router.route(spec)
print(f"Renderer: {decision.renderer_type}")
print(f"Warnings: {decision.warnings}")
```

### Feedback not being applied

Check parsed corrections:
```python
corrections = handler.parse_feedback(feedback, state)
print(f"Corrections: {corrections}")
```

### SMILES not rendering

Verify SMILES is valid:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
print(f"Valid: {mol is not None}")
```

## Architecture Diagram

```
Question Text
     │
     ▼
┌─────────────────┐
│ Spec Generation │  LLM fills DiagramSpec JSON
│    (LLM 1)      │  NOT drawing code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SpecRouter    │  Enforces library selection
│                 │  SchemDraw / RDKit / matplotlib
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Renderer     │  Uses correct library
│  (Specialized)  │  to generate image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Verification  │  LLM 2 checks diagram
│    (LLM 2)      │  against question
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 Accepted  Rejected
    │         │
    │         ▼
    │  ┌──────────────┐
    │  │ Feedback     │  Parses issues into
    │  │ Handler      │  spec corrections
    │  └──────┬───────┘
    │         │
    │         ▼
    │  ┌──────────────┐
    │  │ Update Spec  │  Apply corrections
    │  │ (not prompt) │  to parameters
    │  └──────┬───────┘
    │         │
    │         └──────► Loop back to Renderer
    │
    ▼
 Final Diagram
```
