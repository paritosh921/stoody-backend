# hub-uplink

WiFi/mobile upload module with resume ledger for ExamPen hub. Reads stored chunks from `hub-store` via IPC and uploads them to `svc-stroke-ingest` on the backend.

## Architecture

```
hub-supervisor ──IPC──> hub-uplink ──HTTP──> svc-stroke-ingest
                            │
                            ├──IPC──> hub-store (store.read.request)
                            ├──SQLite──> upload_ledger (ack tracking)
                            └──nmcli──> NetworkManager (read-only WiFi status)
```

## Upload Protocol

1. Receive `uplink.upload.request` from supervisor (or invigilator BLE relay).
2. Select upload path: WiFi (primary) or mobile BLE relay (last resort).
3. For each pen with synced chunks:
   a. Read chunk from `hub-store` via IPC.
   b. POST to `/api/v1/strokes/ingest` with idempotency key `{exam_id}:{pen_mac}:{chunk_index}`.
   c. On backend 202 ACK: update `upload_ledger.acked_chunks` JSON array.
   d. Retry indefinitely on failure (backend deduplicates).
4. Mark pen as "uploaded" ONLY after ALL chunks ACKd.

## Key Invariants

- **Ledger updated ONLY after backend ACK** (STATE_OWNERSHIP_MAP.md Section 3.1).
- **WiFi status check is read-only** (STATE_OWNERSHIP_MAP.md Section 2.1).
- **Backend reachability check is read-only** (no mutations).
- **Resume**: on restart, `get_pending_chunks()` returns only unacked indices.

## Ownership Declaration

- **Writes:** `upload_ledger` table (ack tracking only, after backend ACK)
- **Reads from:** `hub-store` (chunk data via IPC), NetworkManager (WiFi status)
- **Never writes to:** Pen data files, WiFi config, hub FSM state
- **Transactional boundaries:** Ledger update ONLY after HTTP 202 from backend
