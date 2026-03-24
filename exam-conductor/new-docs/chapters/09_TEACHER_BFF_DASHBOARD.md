# Chapter 09: Teacher BFF & Dashboard

## Status
- **Phase:** P9
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Read-only aggregation layer (svc-teacher-bff) and the React teacher dashboard.
The BFF aggregates data from scoring, analytics, review, and exam orchestration
services into dashboard-ready payloads. The frontend consumes these via REST.

## Architecture Context

<!-- TODO: Diagram showing teacher-dashboard -> svc-teacher-bff -> backing
     services. Reference Chapter 01. BFF has zero write access to any DB. -->

## Detailed Design

### BFF Aggregation Patterns
<!-- TODO: Fan-out queries, response composition, caching strategy. -->

### Dashboard Screens
<!-- TODO: Exam list, score review, analytics, plagiarism review, chat. -->

### Score Review Workflow
<!-- TODO: AI draft -> teacher review -> override -> finalize flow in UI. -->

### Mobile Teacher View
<!-- TODO: Flutter teacher mode, shared BFF endpoints. -->

## Interfaces
<!-- TODO: REST endpoints from api/teacher-bff.openapi.yaml. -->

## Configuration
<!-- TODO: Environment variables, backing service URLs. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A4.6
     (AI misrecognition flagging in UI), A5.5 (rubric version display). -->

## Testing
<!-- TODO: Reference test IDs E2E-10 (teacher BFF aggregation),
     E2E-13 (full integration). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
