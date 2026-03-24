# Chapter 11: Invigilator Console

## Status
- **Phase:** P13
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Real-time WebSocket-based invigilator dashboard (svc-invig-console) and
its React frontend. Provides live exam monitoring: hub status, pen connectivity,
student progress, timer display, and alert management during an active exam.

## Architecture Context

<!-- TODO: Diagram showing invigilator-console (frontend) <-> svc-invig-console
     <-> hub-uplink (WebSocket relay). Reference Chapter 01. -->

## Detailed Design

### WebSocket Protocol
<!-- TODO: Connection lifecycle, message types, heartbeat/reconnect. -->

### Live Monitoring Dashboard
<!-- TODO: Hub status grid, pen connectivity indicators, student progress. -->

### Alert System
<!-- TODO: Alert types (pen disconnect, dongle failure, timer anomaly),
     escalation rules. -->

### Hub Status Relay
<!-- TODO: How hub state is relayed via hub-uplink to svc-invig-console. -->

## Interfaces
<!-- TODO: WebSocket endpoints and REST endpoints from
     api/invig-console.openapi.yaml. Hub status relay contract
     from hub/ipc-protocol.md. -->

## Configuration
<!-- TODO: Environment variables, WebSocket settings, alert thresholds. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries H1 (USB brownout
     visible in console), H3 (dongle failure alert), F1 (timer drift warning). -->

## Testing
<!-- TODO: Reference test IDs for svc-invig-console integration tests. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
