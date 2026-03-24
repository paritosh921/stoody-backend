# hub-store

Dual-write (SD + USB) storage module for ExamPen hub. This module is the **single writable owner** of local stroke data on the hub (see STATE_OWNERSHIP_MAP.md).

## Architecture

```
hub-pen-sync  ──IPC──>  hub-store  ──fsync──>  SD (/var/lib/exampen)
                                    ──fsync──>  USB (/mnt/exampen-backup)
hub-uplink    ──IPC──>  hub-store  (read-only: get_chunk)
```

## Dual-Write Protocol (HUB_DEPLOYMENT_SPEC.md Section 3.3)

1. Receive chunk via IPC (`store.write.request`).
2. CRC-32 verify decoded bytes against `checksum_crc32`.
3. Write to SD `strokes_raw.bin` (append) + `os.fsync()`.
4. Write to USB `strokes_raw.bin` (append) + `os.fsync()`.
5. Write pre-chunked upload file (`chunks/chunk_NNN.bin`) on both paths + `os.fsync()`.
6. If USB write fails: log warning, set `degraded` flag, continue SD-only.

## Ownership Declaration

- **Writes:** SD file data (`/var/lib/exampen/data/`), USB file data (`/mnt/exampen-backup/data/`)
- **Reads from:** Same paths (for `store.read.request` and integrity checks)
- **Never writes to:** SQLite database (owned by `hub-supervisor`), BLE (owned by `hub-ble-mgr`), network (owned by `hub-uplink`)
- **Transactional boundaries:** Dual fsync protocol — caller blocked until SD write confirmed. USB write is best-effort.
