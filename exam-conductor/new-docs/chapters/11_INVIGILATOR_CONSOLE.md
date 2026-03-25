# Chapter 11: Invigilator Console

## Status
- **Build status:** DRAFT
- **Authority source:** `api/invig-console.openapi.yaml`

## Overview

The invigilator console watches the shared ingest substrate during conducted exams. It is concerned with exam lifecycle, hub connectivity, upload completeness, and operator alerts.

It is not an evaluator surface.

## Key Views

- active exam session state
- hub and pen connectivity
- upload and reconciliation progress
- alerts requiring manual intervention

## Alignment Rules

1. Invigilator actions affect exam operations, not engine scoring rules.
2. Console state is eventually consistent with backend ingest status.
3. Review or override actions belong to teacher/review flows, not invigilation.

## Related Docs

- `api/invig-console.openapi.yaml`
- `chapters/04_HUB_OPERATIONS.md`
- `chapters/06_EXAM_LIFECYCLE.md`
