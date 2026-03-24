# Chapter 18: Copy Image Fallback

## Status
- **Phase:** P2i-ext
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Fallback answer capture via photographed answer sheets (svc-copy-upload). When
pen data is unavailable (pen failure, data loss), the invigilator or student
photographs answer pages. The service handles image upload, OCR processing,
and feeds results into the scoring pipeline via the copy.ready event.

## Architecture Context

<!-- TODO: Diagram showing mobile camera -> svc-copy-upload -> MinIO storage
     -> OCR -> score pipeline entry. Reference Chapter 01 and
     FAILURE_MITIGATION_REGISTER.md A1.5 (pen battery death mitigation). -->

## Detailed Design

### Upload Flow
<!-- TODO: Image capture, compression, upload to svc-copy-upload REST API. -->

### OCR Processing
<!-- TODO: Image preprocessing, OCR engine, text extraction. -->

### Pipeline Integration
<!-- TODO: copy.ready event emission, score engine consumption. -->

### Quality Validation
<!-- TODO: Image quality checks, blur detection, re-upload prompting. -->

## Interfaces
<!-- TODO: REST endpoints from api/copy-upload.openapi.yaml,
     NATS event copy.ready. -->

## Configuration
<!-- TODO: Image size limits, OCR engine settings, MinIO bucket. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A1.5
     (pen death -> copy fallback), S4 (dual storage failure -> copy fallback). -->

## Testing
<!-- TODO: Reference test IDs E2E-07 (copy image upload -> OCR -> score). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
