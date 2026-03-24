# Chapter 13: Plagiarism Detection

## Status
- **Phase:** P8b
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Multi-layer plagiarism detection (svc-plagiarism): TF-IDF cosine similarity,
structural similarity via edit distance, temporal/proximity signals from stroke
metadata, and question-type-aware threshold adjustment. All detections require
teacher review before any penalty is applied.

## Architecture Context

<!-- TODO: Diagram showing svc-plagiarism consuming ai.result events and
     producing plagiarism.result events. Reference Chapter 01. -->

## Detailed Design

### Detection Layers
<!-- TODO: TF-IDF text similarity, structural edit distance,
     temporal + proximity signals. -->

### Threshold Configuration
<!-- TODO: Composite >0.75 for "review", >0.90 for "strong match".
     Question-type adjustments (MCQ exclusion). -->

### Teacher Review Workflow
<!-- TODO: Flag presentation, verdict recording, appeal handling. -->

### Scoring Integration
<!-- TODO: How plagiarism verdicts affect score FSM. -->

## Interfaces
<!-- TODO: REST endpoints from api/plagiarism.openapi.yaml,
     NATS events plagiarism.check and plagiarism.result. -->

## Configuration
<!-- TODO: Similarity thresholds, question-type overrides, batch size. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entry PL5
     (false positive mitigation). -->

## Testing
<!-- TODO: Reference test IDs U-PLAG-01, U-PLAG-02, U-PLAG-03,
     E2E-06 (plagiarism detection). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
