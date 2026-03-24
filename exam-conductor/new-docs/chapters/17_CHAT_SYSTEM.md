# Chapter 17: Chat System

## Status
- **Phase:** P9d
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Append-only messaging service (svc-chat): student-teacher communication scoped
to exam/question context. Enforces append-only contract (no UPDATE, no DELETE),
provides read receipts, and integrates with both teacher and student BFF services.

## Architecture Context

<!-- TODO: Diagram showing svc-chat accessed via svc-teacher-bff and
     svc-student-bff. Append-only PostgreSQL storage. Reference Chapter 01. -->

## Detailed Design

### Append-Only Contract
<!-- TODO: INSERT only, no UPDATE/DELETE. Immutability guarantees. -->

### Message Model
<!-- TODO: Schema: exam_id, question_id, sender, body, timestamp. -->

### Read Receipts
<!-- TODO: Mark-read semantics, timestamp recording. -->

### Conversation Scoping
<!-- TODO: Per-exam, per-question, per-student threading. -->

## Interfaces
<!-- TODO: REST endpoints from api/chat.openapi.yaml. -->

## Configuration
<!-- TODO: Message size limits, rate limits, retention policy. -->

## Failure Modes & Mitigations
<!-- TODO: Reference any relevant FAILURE_MITIGATION_REGISTER.md entries. -->

## Testing
<!-- TODO: Reference test IDs U-CHAT-01 (append-only contract),
     U-CHAT-02 (read receipts). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
