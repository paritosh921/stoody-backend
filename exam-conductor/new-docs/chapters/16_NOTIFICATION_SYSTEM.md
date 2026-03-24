# Chapter 16: Notification System

## Status
- **Phase:** P14
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Notification service (svc-notify): trigger-based delivery of email, push, and
SMS notifications. Fires on score publication, objection status changes, exam
lifecycle events, and plagiarism verdicts. Template-driven with per-tenant
customization.

## Architecture Context

<!-- TODO: Diagram showing svc-notify consuming NATS events (score.updated,
     objection.resolved, exam.lifecycle) and dispatching via email/push/SMS
     providers. Reference Chapter 01. -->

## Detailed Design

### Trigger Catalog
<!-- TODO: Which events trigger which notification types. -->

### Template System
<!-- TODO: Template structure, variable interpolation, per-tenant overrides. -->

### Delivery Channels
<!-- TODO: Email (SMTP/SES), push (FCM/APNs), SMS (provider TBD). -->

### Delivery Guarantees
<!-- TODO: At-least-once delivery, retry logic, dead letter handling. -->

## Interfaces
<!-- TODO: NATS event consumption, internal REST endpoints if any. -->

## Configuration
<!-- TODO: Provider credentials, template paths, retry policies, rate limits. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries relevant to
     notification delivery failures. -->

## Testing
<!-- TODO: Reference test IDs E2E-05 (objection -> notification),
     E2E-12 (Stoody webhook). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
