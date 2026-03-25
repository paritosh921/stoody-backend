# Chapter 09: Teacher BFF and Dashboard

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

Teacher-facing surfaces read across the shared ingest substrate, DCR outputs, PCR outputs, review state, and analytics. They aggregate; they do not become a second evaluator.

```text
Stoody teacher session
        │
        ▼
   teacher BFF / dashboard
        │
        ├-> exam orchestration
        ├-> ingest status
        ├-> DCR results
        ├-> PCR results / flags
        └-> review + analytics
```

## Alignment Rules

1. Teacher surfaces do not write canonical raw artifacts.
2. Teacher surfaces do not bypass the review or gate rules.
3. Tutor visibility follows the existing admin-owned student visibility model.

## Related Docs

- `integration/STOODY_INTEGRATION_SPEC.md`
- `api/teacher-bff.openapi.yaml`
- `api/review.openapi.yaml`
- `chapters/14_OBJECTION_REVIEW.md`
