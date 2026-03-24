# Chapter 23: Testing Strategy

## Status
- **Phase:** Cross-cutting
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Testing strategy across all validation evidence levels (L1-L7). Covers unit
testing (L3) with pure domain logic isolation, integration testing (L4) with
real infrastructure in Docker, E2E pipeline testing (L5), hardware-in-loop
testing (L6), and field trial protocols (L7). 1256+ tests across the suite.

## Architecture Context

<!-- TODO: Testing pyramid diagram showing L1-L7 layers and test counts per
     level. Reference Chapter 01 and TEST_SUITE_SPEC.md. -->

## Detailed Design

### Validation Evidence Hierarchy
<!-- TODO: L1 through L7 definitions, what each level proves. -->

### Unit Testing (L3)
<!-- TODO: Domain layer isolation, no I/O imports, factory fixtures. -->

### Integration Testing (L4)
<!-- TODO: Docker Compose test stack, real DB/NATS/S3, per-service tests. -->

### E2E Pipeline Testing (L5)
<!-- TODO: Multi-service pipeline tests, NATS event coordination,
     test IDs E2E-01 through E2E-13. -->

### Hardware-in-Loop Testing (L6)
<!-- TODO: Hub + BLE dongle + pen simulator test setup. -->

### Field Trial Protocol (L7)
<!-- TODO: Real exam protocol, success criteria, data collection. -->

### Test ID Convention
<!-- TODO: U-{SVC}-{NN} for unit, I-{SVC}-{NN} for integration,
     E2E-{NN} for pipeline. -->

## Interfaces
<!-- TODO: pytest markers, test runner commands, CI integration. -->

## Configuration
<!-- TODO: docker-compose.test.yml, fixture data paths, environment variables. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md — testing catches mitigations
     for all entries. -->

## Testing
<!-- TODO: Meta — this chapter IS the testing reference.
     Full test ID catalog in TEST_SUITE_SPEC.md. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
