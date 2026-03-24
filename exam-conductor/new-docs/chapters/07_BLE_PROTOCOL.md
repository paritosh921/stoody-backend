# Chapter 07: BLE Protocol & Pen Communication

## Status
- **Phase:** P2b/P2f
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

BLE GATT services for pen communication and invigilator relay. Covers the P05
pen protocol (service 0000ae30), frame format, CRC-16/XMODEM validation,
coordinate extraction, chunk transfer, and the invigilator BLE relay channel.

## Architecture Context

<!-- TODO: Diagram showing pen -> hub-ble-mgr -> hub-pen-sync data flow.
     Reference Chapter 01 and hub/ble-gatt-spec.md. -->

## Detailed Design

### GATT Service Structure
<!-- TODO: Characteristics AE10 (write), AE02 (notify), command set. -->

### Frame Format
<!-- TODO: Head(2) + SerialNum(4) + ID(4) + Cmd(1) + DataFormat(1) +
     DataLen(2) + Data(N) + CRC16(2). -->

### Coordinate Frame
<!-- TODO: 14-byte coordinate format, scale (10 units/mm), Y-inversion. -->

### Chunk Transfer Protocol
<!-- TODO: Per-chunk checkpointing, resume on disconnect. -->

### Invigilator BLE Relay
<!-- TODO: hub-invig-ble rotating auth codes, command channel. -->

## Interfaces
<!-- TODO: Reference hub/ble-gatt-spec.md characteristics and commands. -->

## Configuration
<!-- TODO: Dongle count, scan intervals, connection parameters. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A1.1, A1.5, A1.7,
     H1, H3, H5, S3. -->

## Testing
<!-- TODO: Reference test IDs from L6 hardware-in-loop tests. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
