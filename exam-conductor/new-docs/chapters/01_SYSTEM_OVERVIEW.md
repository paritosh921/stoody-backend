# Chapter 01: System Overview

## Status
- **Build status:** DRAFT
- **Authority source:** `architecture/DUAL_MODE_ARCHITECTURE.md`

## Overview

ExamPen is a modular assessment subsystem of Stoody composed of:

- a shared ingest substrate
- a DCR engine
- a PCR engine
- a shared LLM gate

```text
Stoody identity / roster / tutor visibility
                 │
                 ▼
      Shared ingest substrate
        │               │
        ▼               ▼
    DCR engine      PCR engine
        │               │
        └───────┬───────┘
                ▼
             LLM gate
```

## Architecture Rules

1. The ingest substrate owns conducted-exam artifact persistence.
2. DCR and PCR are independent evaluators.
3. Practice persistence remains outside ExamPen.
4. All LLM-mediated calls go through the gate.

## Related Docs

- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- `architecture/LLM_GATE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`
