# Chapter 10: Student BFF and Portal

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

Student-facing ExamPen behavior is read-oriented. Students view published outcomes, objection status, and supporting feedback through Stoody-facing surfaces after DCR or PCR results are finalized.

## Alignment Rules

1. Students do not read draft evaluator internals unless the review policy allows it.
2. Published results come from finalized engine outputs, not directly from raw artifacts.
3. Practice persistence stays in the existing Stoody backend path.

## Related Docs

- `api/student-bff.openapi.yaml`
- `api/review.openapi.yaml`
- `integration/STOODY_INTEGRATION_SPEC.md`
