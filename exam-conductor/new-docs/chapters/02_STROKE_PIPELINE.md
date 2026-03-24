# Chapter 02: Stroke Pipeline

## Status
- **Phase:** W6 — Documentation
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6.A6.1)
- **Build status:** DRAFT

## Overview

The stroke pipeline is the data lifeline of ExamPen. It moves raw handwriting data from pen hardware through hub collection, cloud ingestion, processing, and storage. Every stage enforces idempotency and data integrity because pen stroke data is irreplaceable once the pen buffer is cleared.

## Architecture Context

The stroke pipeline spans four physical boundaries: pen firmware, hub (RPi), cloud services, and TimescaleDB. It feeds into `svc-doc-assembly` (Chapter 03) which renders page images for AI processing.

## Pipeline Stages

```
+------+    +------------+    +----------+    +-----------+    +------------+    +----------+
| Pen  |--->| hub-pen-   |--->| hub-     |--->| hub-      |--->| svc-stroke-|--->| svc-     |
| GATT |    | sync       |    | store    |    | uplink    |    | ingest     |    | stroke-  |
|      |    | (chunk     |    | (dual-   |    | (WiFi/    |    | (validate  |    | proc     |
|      |    |  read)     |    |  write)  |    |  mobile)  |    |  + NATS)   |    | (dedup + |
+------+    +------------+    +----------+    +-----------+    +------------+    | Timescale|
                                                                                +----------+
```

### Stage 1: Pen -> hub-pen-sync (BLE GATT Read)

**Input:** Raw stroke bytes in pen flash buffer.

**Processing:**
- Hub discovers pen via BLE scan (registration or post-exam sync mode).
- `hub-pen-sync` reads `Buffer status` characteristic: `{total_bytes, bytes_remaining, checksum_crc32}` (12 bytes, LE).
- Sequential chunk reads from `Stroke buffer` characteristic (`6f5f1001-...`).
- Chunk wire format: version(1) + flags(1) + header_len(2) + chunk_index(4) + total_chunks(4) + payload_len(4) + payload_crc32(4) + payload(N).
- CRC32 verified per chunk before proceeding.

**Output:** Raw chunk bytes passed to `hub-store` via IPC (`store.write.request`).

**Error handling:**
- BLE disconnect: reconnect and resume from last confirmed chunk (mitigation A1.7). 3 retries, 30s timeout each.
- Checksum mismatch: mark `pen_sync_status.status = 'failed'`, alert invigilator.
- Pen does NOT clear buffer until hub sends `Sync control = 0x03` after all chunks verified.

**Source files:** `hub/hub-pen-sync/src/gatt_reader.py`, `hub/hub-pen-sync/src/chunk_manager.py`

### Stage 2: hub-pen-sync -> hub-store (Dual-Write)

**Input:** Chunk bytes from `hub-pen-sync`.

**Processing:**
1. Write chunk to SD: `/var/lib/exampen/data/{exam_id}/{pen_mac}/strokes_raw.bin` (append).
2. `fsync()` SD file descriptor.
3. Write identical chunk to USB: `/mnt/exampen-backup/data/{exam_id}/{pen_mac}/strokes_raw.bin` (append).
4. `fsync()` USB file descriptor.
5. Only after both `fsync()` succeed: ACK pen to advance.

**Output:** Durable stroke data on two independent storage media.

**Error handling:**
- USB write failure: degrade to SD-only mode, set degraded flag, TUI amber warning (mitigation S4).
- SD failure: critical alert, no fallback (mitigation S4).
- After all chunks + checksum match: `hub-store` sends `Sync control = 0x03` to pen, pen clears buffer.

**Source files:** `hub/hub-store/src/`, IPC contract: `hub/ipc-protocol.md` message `store.write.request`

### Stage 3: hub-store -> hub-uplink (Upload)

**Input:** Stored chunks from SD, chunked for upload.

**Processing:**
- Pre-chunk files into `chunks/chunk_NNN.bin` for upload.
- Per-pen upload ledger in SQLite tracks `acked_chunks[]` (JSON array of indices).
- HTTP POST each chunk to `svc-stroke-ingest`: `POST /api/v1/strokes/upload`.
- Wait for backend ACK per chunk before updating ledger.

**Output:** Stroke data delivered to cloud, ledger updated.

**Error handling:**
- Network failure: retry indefinitely. Hub retains all data locally.
- Partial upload resume: only unacked chunks sent on retry (mitigation U4).
- Reconciliation endpoint: `GET /api/v1/exams/{exam_id}/upload-status`.
- WiFi unavailable: fall back to mobile BLE relay path (~12 min for 40 pens, mitigation U1).

**Source files:** `hub/hub-uplink/src/`, upload ledger: `upload_ledger` SQLite table

### Stage 4: svc-stroke-ingest (Validation + NATS Publish)

**Input:** HTTP chunk upload from hub.

**Processing:**
1. Schema validation of stroke packet (field presence, coordinate ranges).
2. Publish to NATS JetStream subject `stroke.raw`.
3. Return HTTP 200 ACK to hub.

**Output:** `stroke.raw` event on NATS JetStream.

**Error handling:**
- Invalid packet: HTTP 400, hub logs and skips chunk.
- NATS publish failure: HTTP 503, hub retries.
- Backpressure: NATS slow consumer handled via JetStream buffering (mitigation A8.4).

**Event schema:** `contracts/events/stroke.raw.schema.json`
- Key fields: `exam_id`, `pen_mac`, `chunk_index`, `total_chunks`, `payload_base64`, `checksum_crc32`, `upload_path`

**Source files:** `services/svc-stroke-ingest/src/routes/`, `services/svc-stroke-ingest/src/domain/`

### Stage 5: svc-stroke-proc (Dedup + Normalize + TimescaleDB)

**Input:** `stroke.raw` events from NATS.

**Processing:**
1. Dedup by idempotency key: `{exam_id, pen_mac, chunk_index}` — SELECT FOR UPDATE within transaction.
2. Coordinate normalization: raw pen units (10 units/mm) to normalized mm.
3. Page assignment: strokes mapped to question regions based on spatial coordinates.
4. Atomic batch write per pen per exam to TimescaleDB.
5. Publish `stroke.processed` event on NATS.

**Output:** Normalized strokes in TimescaleDB, `stroke.processed` event.

**Error handling:**
- Duplicate NATS event: idempotency key ensures single DB commit (mitigation A8.6).
- Transaction failure: NATS message NACKed, redelivered.
- No partial pen data: all chunks for one pen committed in single transaction.

**Event schema:** `contracts/events/stroke.processed.schema.json`

**Source files:** `services/svc-stroke-proc/src/domain/`, `services/svc-stroke-proc/src/storage/`

## Idempotency Keys and Dedup Strategy

| Stage | Idempotency Key | Mechanism |
|---|---|---|
| Hub upload -> ingest | `{exam_id, pen_mac, chunk_index}` | Backend rejects duplicate chunks |
| Ingest -> NATS | Event `event_id` (unique per publish) | JetStream dedup window |
| NATS -> stroke-proc | `{exam_id, pen_mac, chunk_index}` | SELECT FOR UPDATE in TimescaleDB transaction |
| Score engine events | Event sequence number | Duplicate events detected by sequence |

## Data Custody Chain

```
Pen flash (irreplaceable)
  |  0x03 clear ONLY after dual-write confirmed
  v
Hub SD + USB (dual-write, fsync)
  |  ledger tracks acked chunks
  v
Backend svc-stroke-ingest (NATS publish)
  |  idempotent ingestion
  v
TimescaleDB (durable, deduped)
```

**Critical rule:** Pen clears buffer only after hub confirms dual-write. Hub marks upload complete only after backend ACK per chunk. This is a two-phase commit with the pen as transaction coordinator.

## Testing

- **Unit:** U-STRK-01 (packet validation valid), U-STRK-02 (invalid packets rejected), U-PROC-01 (coordinate normalization), U-PROC-02 (dedup), U-PROC-03 (page assignment)
- **Integration:** I-STRK-01 (WebSocket -> NATS), I-STRK-02 (duplicate rejection), I-STRK-03 (backpressure), I-PROC-01 (NATS -> TimescaleDB), I-PROC-02 (duplicate NATS -> single commit)
- **E2E:** E2E-01 (ingestion -> processing -> storage), E2E-08 (40-student simulation)
- **Hardware:** HW-B3 (dual-write integrity), HW-B2 (multi-pen sync)

## Failure Modes & Mitigations

| ID | Failure | Mitigation |
|---|---|---|
| A1.7 | BLE disconnect during sync | Per-chunk checkpoint, 3 retries x 30s |
| S4 | SD card failure | Dual-write to USB, degrade with warning |
| U1 | WiFi unavailable for upload | Mobile BLE relay fallback |
| U4 | Partial upload resume | Per-pen upload ledger, backend idempotency |
| A8.4 | Peak load (10K students) | Stateless ingestion, NATS buffering, rate limiting |
| A8.6 | Duplicate processing | Idempotency keys at every stage |

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Initial draft: full pipeline stages, idempotency, data custody, test references | Claude Agent (W6.A6.1) |
