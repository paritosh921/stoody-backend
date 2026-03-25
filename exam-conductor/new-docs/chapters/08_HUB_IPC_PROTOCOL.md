# Chapter 08: Hub IPC Protocol

## Status
- **Build status:** DRAFT
- **Authority source:** `hub/ipc-protocol.md`

## Overview

Hub IPC is an internal coordination contract inside the shared ingest substrate. It exists so collection, buffering, timer, invigilator relay, and upload modules can coordinate without leaking evaluator concerns into hub code.

## Architecture Context

```text
hub BLE mgr
     │
     ├-> hub store
     ├-> hub timer
     ├-> hub uplink
     ├-> hub invigilator relay
     └-> hub TUI
          │
          ▼
   canonical artifact upload
```

## Alignment Rules

1. IPC messages may describe collection state, upload state, and local failures.
2. IPC messages must not embed DCR or PCR scoring logic.
3. The engine boundary starts after canonical artifact persistence.

## Related Docs

- `hub/ipc-protocol.md`
- `integration/HUB_DEPLOYMENT_SPEC.md`
- `chapters/04_HUB_OPERATIONS.md`
