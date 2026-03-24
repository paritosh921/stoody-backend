# hub-pen-sync

BLE GATT pen data sync module for ExamPen hub. Reads stroke data from P05 pens via BLE notifications, verifies integrity, and passes chunks to `hub-store` for durable dual-write.

## Architecture

```
Pen (GATT) --BLE--> hub-pen-sync --IPC--> hub-store --fsync--> SD + USB
                          |
                          +--> IPC events --> hub-supervisor, hub-tui, hub-invig-ble
```

## Sync Protocol

1. Connect to pen via BLE (through `hub-ble-mgr`).
2. Read `Buffer Status` characteristic (total bytes, CRC-32).
3. Write `Sync Control = 0x01` (start transfer).
4. Receive chunk notifications on `Stroke Buffer` characteristic.
5. For each chunk: verify per-chunk CRC-32, send to `hub-store` via IPC.
6. After all chunks: verify whole-buffer CRC-32 against `Buffer Status`.
7. **Only after hub-store confirms dual-write for ALL chunks AND checksum matches**: write `Sync Control = 0x03` (clear pen buffer).

## Critical Data Safety

Pen stroke data is **irreplaceable** once the pen buffer is cleared. The `0x03` clear command is NEVER sent until:
- Every chunk has been durably written to both SD and USB (or SD-only in degraded mode).
- The whole-buffer CRC-32 matches the value from the pen's `Buffer Status` characteristic.

## Retry Semantics (FAILURE_MITIGATION_REGISTER.md A1.7)

- 3 reconnect attempts per pen on BLE disconnect.
- 30 second timeout per attempt.
- Resume from last confirmed chunk index on reconnect.

## Module Layout

| File | Layer | I/O? | Purpose |
|------|-------|------|---------|
| `src/sync_state.py` | Domain | ZERO | Per-pen state machine, retry tracking |
| `src/chunk_manager.py` | Domain | ZERO | Chunk assembly, CRC verification |
| `src/config.py` | Config | ZERO | GATT UUIDs, sync parameters |
| `src/gatt_reader.py` | Adapter | BLE | GATT characteristic reads/writes |
| `src/sync_orchestrator.py` | Orchestrator | IPC+BLE | Full sync flow coordination |
| `src/ipc_handlers.py` | Handler | IPC | Message dispatch |
| `src/main.py` | Entry | IPC | Event loop, server startup |

## Ownership Declaration

- **Writes:** BLE GATT characteristics on pen (read buffer, sync control)
- **Reads from:** Pen GATT characteristics, hub-store IPC replies
- **Never writes to:** Local filesystem (owned by `hub-store`), SQLite (owned by `hub-supervisor`)
- **Transactional boundaries:** Pen buffer clear ONLY after hub-store confirms dual-write + checksum match
