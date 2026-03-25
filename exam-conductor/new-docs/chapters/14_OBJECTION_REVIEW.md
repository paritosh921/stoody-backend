# Chapter 14: Objection and Review

## Status
- **Build status:** DRAFT
- **Authority sources:** `architecture/TAMPER_PROOF_SPEC.md`, `api/review.openapi.yaml`

## Overview

Review sits above DCR and PCR outputs. It lets authorized actors inspect evidence, resolve flags, and apply audited overrides without mutating canonical raw artifacts.

## Review Inputs

- DCR recognized text and match output
- PCR detected responses, flags, and evaluations
- gate-linked usage or reasoning references where relevant
- append-only history of prior actions

## Alignment Rules

1. Review requests identify server-side artifacts or evaluation records; client-submitted corrected text is not authoritative.
2. Overrides and flag resolutions are append-only audit events.
3. Review may change score state, but not raw conducted-exam artifacts.

## Related Docs

- `architecture/TAMPER_PROOF_SPEC.md`
- `api/review.openapi.yaml`
- `contracts/events/objection.schema.json`
