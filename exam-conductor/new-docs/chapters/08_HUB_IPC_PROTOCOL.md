# Chapter 08: Hub IPC Protocol

## Status
- **Phase:** P2a
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Inter-process communication between hub modules via Unix domain sockets with
JSON-lines encoding. Defines message types, routing, error handling, and the
supervisor's role as the central hub orchestrator.

## Architecture Context

<!-- TODO: Diagram showing hub-supervisor as central IPC router connecting
     hub-ble-mgr, hub-pen-sync, hub-timer, hub-store, hub-uplink,
     hub-invig-ble, hub-tui. Reference Chapter 01 and hub/ipc-protocol.md. -->

## Detailed Design

### Socket Topology
<!-- TODO: Unix domain socket paths, connection lifecycle. -->

### Message Format
<!-- TODO: JSON-lines encoding, message types, request/response correlation. -->

### Supervisor Orchestration
<!-- TODO: Process management, FSM coordination, health monitoring. -->

### Error Handling
<!-- TODO: Reconnect logic, dead module detection, graceful degradation. -->

## Interfaces
<!-- TODO: Full IPC message catalog from hub/ipc-protocol.md. -->

## Configuration
<!-- TODO: Socket paths, timeouts, retry policies. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries F4 (hub reboot),
     S4 (SD failure). -->

## Testing
<!-- TODO: Reference hub integration test IDs. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
