# Chapter 15: Analytics and Reporting

## Status
- **Build status:** DRAFT
- **Authority source:** `api/analytics.openapi.yaml`

## Overview

Analytics is downstream of finalized evaluation and publication state. It aggregates across DCR and PCR results without redefining evaluator logic.

## Alignment Rules

1. Analytics consumes published or review-approved result state.
2. It may segment by engine or exam type, but it must not change engine-owned scores.
3. Tutor and student visibility rules still come from Stoody.

## Related Docs

- `api/analytics.openapi.yaml`
- `contracts/events/score.updated.schema.json`
- `integration/STOODY_INTEGRATION_SPEC.md`
