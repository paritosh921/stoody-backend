# Chapter 16: Notification System

## Status
- **Build status:** DRAFT

## Overview

Notifications reflect state changes in exam lifecycle, evaluation publication, review, and operational alerts. They do not own any engine or ingest state.

## Alignment Rules

1. Notification triggers originate from authoritative exam, review, or result state.
2. Notifications must not expose restricted draft content to unauthorized actors.
3. Delivery failures do not roll back canonical evaluator state.

## Related Docs

- `contracts/events/exam.lifecycle.schema.json`
- `contracts/events/score.updated.schema.json`
- `integration/STOODY_INTEGRATION_SPEC.md`
