# Chapter 15: Analytics & Reporting

## Status
- **Phase:** P12
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Analytics service (svc-analytics): percentile calculation, leaderboard generation,
class/section/subject aggregations, score distribution histograms, and data export.
Consumes score.updated events to maintain materialized analytics views.

## Architecture Context

<!-- TODO: Diagram showing svc-analytics consuming score.updated events and
     serving aggregated data via REST to BFF services. Reference Chapter 01. -->

## Detailed Design

### Percentile Calculation
<!-- TODO: Algorithm, tie-breaking, incremental recalculation on override. -->

### Leaderboard Generation
<!-- TODO: Sorting rules, tie-breaking by name, per-exam and cumulative. -->

### Aggregation Views
<!-- TODO: Class-level, section-level, subject-level, question-level stats. -->

### Data Export
<!-- TODO: CSV/Excel export, report generation. -->

## Interfaces
<!-- TODO: REST endpoints from api/analytics.openapi.yaml,
     NATS event consumption of score.updated. -->

## Configuration
<!-- TODO: Recalculation triggers, export format options. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries relevant to
     analytics consistency. -->

## Testing
<!-- TODO: Reference test IDs U-ANLY-01, U-ANLY-02,
     E2E-04 (override -> analytics recalculation). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
