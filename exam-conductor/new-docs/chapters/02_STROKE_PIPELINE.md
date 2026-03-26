# Chapter 02: Shared Ingest Substrate

## Status
- **Build status:** DRAFT
- **Authority source:** `architecture/DUAL_MODE_ARCHITECTURE.md`

## Overview

This chapter describes the shared ingest substrate. It is not a scoring pipeline.

Its job is:

- collect BLE pen artifacts
- accept camera/scan artifacts where needed
- persist canonical conducted-exam artifacts with provenance
- route artifacts to DCR or PCR by `exam_type`

```text
BLE Pen -> Hub -> Upload -> Canonical artifact store
Camera ---------------> Canonical artifact store
                               │
                               ├─ exam_type=dcr -> DCR engine
                               └─ exam_type=pcr -> PCR engine
```

## Key Rule

The substrate never owns DCR or PCR evaluation semantics.

## Related Docs

- `integration/HUB_DEPLOYMENT_SPEC.md`
- `references/P05_pen_SDK.md`
- `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md`
- `governance/STATE_OWNERSHIP_MAP.md`
