# Chapter 12: Mobile App Architecture

## Status
- **Phase:** P2i / P9a / P10a
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Single Flutter app (exampen-mobile) with dual-mode operation: hub control mode
(BLE-based invigilator interactions with the RPi hub) and portal mode (teacher
and student views consuming BFF APIs). Covers architecture, navigation, BLE
integration, camera for copy image fallback, and offline considerations.

## Architecture Context

<!-- TODO: Diagram showing mobile app modes: hub-control (BLE <-> hub-invig-ble)
     and portal (REST <-> svc-teacher-bff / svc-student-bff).
     Reference Chapter 01. -->

## Detailed Design

### Dual-Mode Architecture
<!-- TODO: Mode switching, shared auth, role-based navigation. -->

### Hub Control Mode
<!-- TODO: BLE connection to hub-invig-ble, command dispatch, status display. -->

### Teacher Portal Mode
<!-- TODO: Score review, analytics, same BFF as web dashboard. -->

### Student Portal Mode
<!-- TODO: Score view, objection filing, chat. -->

### Camera Integration
<!-- TODO: Copy image capture for svc-copy-upload fallback path. -->

## Interfaces
<!-- TODO: BLE characteristics from hub/ble-gatt-spec.md, REST from
     api/teacher-bff.openapi.yaml and api/student-bff.openapi.yaml. -->

## Configuration
<!-- TODO: App configuration, BLE scan parameters, API base URLs. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries S3 (BLE MITM),
     U1 (BLE relay speed). -->

## Testing
<!-- TODO: Reference mobile-specific test IDs, E2E-13 for portal flows. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
