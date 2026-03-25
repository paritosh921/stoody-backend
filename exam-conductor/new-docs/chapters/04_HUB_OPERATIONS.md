# Chapter 04: Hub Operations

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/HUB_DEPLOYMENT_SPEC.md`

## Overview

The hub is part of the shared ingest substrate. Its job is to collect conducted-exam artifacts, keep them locally durable during unstable connectivity, and upload them into canonical tenant/admin storage.

It does not:

- perform DCR evaluation
- perform PCR segmentation or scoring
- own token budgets
- redesign practice persistence

```text
Pens / camera relay
        │
        ▼
   Hub local durability
  (SD + removable media)
        │
        ▼
   upload / reconcile
        │
        ▼
canonical conducted-exam artifacts
        │
        ├─ exam_type=dcr -> DCR engine
        └─ exam_type=pcr -> PCR engine
```

## Operating Rules

1. Local durability comes before evaluator handoff.
2. Student, exam, page, pen, and timestamp provenance must remain attached to every uploaded artifact.
3. The hub must never mutate engine-owned state.
4. Evaluator failures do not retroactively change the hub upload record.

## Related Docs

- `integration/HUB_DEPLOYMENT_SPEC.md`
- `hub/ble-gatt-spec.md`
- `hub/ipc-protocol.md`
- `chapters/02_STROKE_PIPELINE.md`
