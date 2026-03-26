# Chapter 03: Evaluation Pipelines

## Status
- **Build status:** DRAFT
- **Authority sources:** `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`

## Overview

ExamPen has two evaluation pipelines:

- **DCR** for structured, template-bound conducted exams
- **PCR** for paginated subjective evaluation

```text
Canonical conducted-exam artifacts
            │
   ┌────────┴────────┐
   ▼                 ▼
 DCR engine       PCR engine
   │                 │
   │ fallback only   │ all LLM work
   └────────┬────────┘
            ▼
         LLM gate
```

## DCR

- deterministic default path
- ONNX HWR + template match
- gate only for fallback/assisted modes

## PCR

- PageOCR normalization
- segmentation and classification
- deep evaluation through gate
- stateless practice evaluation path

## Related Docs

- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- `architecture/LLM_GATE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`
