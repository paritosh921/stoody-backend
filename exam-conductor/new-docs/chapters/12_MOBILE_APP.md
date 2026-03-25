# Chapter 12: Mobile App

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

The mobile app participates in ExamPen through two kinds of flows:

- invigilation and hub-adjacent operations during conducted exams
- Stoody-facing teacher or student views outside the exam hall

Where practice evaluation is exposed, the mobile app reaches it through existing Stoody backend integration. The mobile app does not create a second practice persistence model.

## Alignment Rules

1. Conducted-exam collection still routes through the shared ingest substrate.
2. Mobile practice calls must respect the stateless PCR practice endpoint boundary.
3. Mobile flows must not bypass Stoody identity or tutor visibility rules.

## Related Docs

- `integration/STOODY_INTEGRATION_SPEC.md`
- `api/student-bff.openapi.yaml`
- `api/teacher-bff.openapi.yaml`
- `api/invig-console.openapi.yaml`
