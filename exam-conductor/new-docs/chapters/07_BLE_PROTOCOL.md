# Chapter 07: BLE Protocol

## Status
- **Build status:** DRAFT
- **Authority source:** `hub/ble-gatt-spec.md`

## Overview

BLE is part of the shared ingest substrate, not part of either evaluation engine.

This chapter exists to explain where BLE fits in the active architecture:

- pens expose stroke buffers and pen metadata
- the hub reads and acknowledges those buffers
- the shared ingest substrate persists uploaded artifacts
- DCR and PCR consume canonical artifacts later

```text
BLE pen <-> hub BLE manager -> local durable store -> canonical artifact store
                                                 │
                                                 └-> route by exam_type
```

## Alignment Rules

1. Use `hub/ble-gatt-spec.md` for UUIDs, frame structure, retries, and error codes.
2. Do not infer DCR or PCR scoring behavior from BLE messages.
3. BLE transport success only proves collection success, not evaluation success.

## Related Docs

- `hub/ble-gatt-spec.md`
- `integration/HUB_DEPLOYMENT_SPEC.md`
- `chapters/04_HUB_OPERATIONS.md`
