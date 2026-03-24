# hub-timer

Exam countdown timer module for the ExamPen Raspberry Pi hub.

## Ownership Declaration

- **Writes:** `active_timer` table in hub SQLite database (persist/clear).
- **Reads from:** `hub-supervisor` (receives `timer.arm.request`, `timer.cancel.request`).
- **Never writes to:** `exam_sessions`, `pen_bindings`, stroke files, or any server-side state.
- **Transactional boundaries:** Timer state persisted to SQLite every 10 s. Up to 10 s accuracy loss on crash (acceptable per `FAILURE_MITIGATION_REGISTER` F4).

**Timer is a LOCAL projection of the exam duration set by `svc-exam-orch`.** It does NOT own exam state.

## Architecture

```
src/
  main.py          Entry point: boot recovery, event loop, IPC registration
  config.py        Tunables: SQLite path, persist interval (10 s), tick interval (1 s)
  countdown.py     Pure countdown engine using CLOCK_MONOTONIC
  persistence.py   SQLite WAL read/write for reboot recovery
  ipc_handlers.py  IPC message dispatch and outbound events
tests/
  test_countdown.py    U-TMR-01..06 — countdown logic
  test_persistence.py  U-TMR-P01..P03 — persist, recover, clear
```

## Running

```bash
# From the hub-timer directory
python -m src.main
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Dependencies

- Python 3.12 (stdlib only — no third-party runtime deps)
- `pytest`, `pytest-asyncio` for dev/test
- `hub-common` (not yet packaged — local IPC constants defined in `ipc_handlers.py` with a TODO to switch)

## Key Design Decisions

1. **CLOCK_MONOTONIC** — never `time.time()` or `datetime.now()` for countdown computation. Immune to NTP adjustments mid-exam (F1).
2. **SQLite WAL mode** — crash-safe persistence shared with other hub modules.
3. **10 s persist interval** — balances write load vs. recovery accuracy (F4).
4. **Fire-and-forget ticks** — `timer.tick` events are informational; missing one is harmless.
5. **Immediate expiry on recovery** — if elapsed gap exceeds remaining, fire `timer.expired.event` on boot.
