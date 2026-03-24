# Chapter 06: Exam Lifecycle

## Status
- **Phase:** P1
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Exam orchestration from creation through scoring lock. Covers the exam FSM
(created -> armed -> timer_running -> collecting -> processing -> scored -> locked),
pen-student binding, timer logic, and Stoody integration points for exam metadata.

## Architecture Context

<!-- TODO: Diagram showing svc-exam-orch relationships to hub-supervisor,
     svc-stroke-ingest, and Stoody webhook endpoints. Reference Chapter 01. -->

## Detailed Design

### Exam FSM States and Transitions
<!-- TODO: State machine diagram and transition rules. -->

### Pen-Student Binding
<!-- TODO: Provisional (hub) vs authoritative (server) binding lifecycle. -->

### Timer Logic
<!-- TODO: CLOCK_MONOTONIC, SQLite persistence, reboot recovery. -->

### Stoody Integration for Exam Metadata
<!-- TODO: Webhook delivery on exam created/completed. -->

## Interfaces
<!-- TODO: REST endpoints from api/exam-orch.openapi.yaml, NATS events exam.lifecycle. -->

## Configuration
<!-- TODO: Environment variables for svc-exam-orch. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries F1, F4. -->

## Testing
<!-- TODO: Reference test IDs U-ORCH-01, U-ORCH-02, U-ORCH-03,
     I-ORCH-01, I-ORCH-02, I-ORCH-03. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
