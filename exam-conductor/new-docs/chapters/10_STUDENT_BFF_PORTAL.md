# Chapter 10: Student BFF & Portal

## Status
- **Phase:** P10
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Read-only aggregation layer (svc-student-bff) and the React student portal.
Students view published scores, file objections, track objection status, and
communicate with teachers via chat. Parent access is scoped to linked children.

## Architecture Context

<!-- TODO: Diagram showing student-portal -> svc-student-bff -> backing
     services. Reference Chapter 01. BFF has zero write access to any DB.
     Mutations route through backing service APIs (svc-review, svc-chat). -->

## Detailed Design

### BFF Aggregation Patterns
<!-- TODO: Score retrieval, objection status, chat history. -->

### Portal Screens
<!-- TODO: Score view, objection filing, chat, exam history. -->

### Parent Access Scoping
<!-- TODO: Parent sees only linked children's data via Stoody API resolution. -->

### Mobile Student View
<!-- TODO: Flutter student mode, shared BFF endpoints. -->

## Interfaces
<!-- TODO: REST endpoints from api/student-bff.openapi.yaml. -->

## Configuration
<!-- TODO: Environment variables, backing service URLs. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries relevant to
     student-facing data delivery. -->

## Testing
<!-- TODO: Reference test IDs E2E-11 (student BFF objection lifecycle),
     E2E-13 (full integration). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
