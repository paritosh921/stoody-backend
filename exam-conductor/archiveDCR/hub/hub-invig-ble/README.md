# hub-invig-ble

Invigilator BLE peripheral module for the ExamPen Raspberry Pi hub.

## Ownership Declaration

- **Reads from:** `invig_codes` table in hub SQLite database (read-only -- codes are generated server-side and cached locally during provisioning).
- **Publishes:** `invig.auth.state.event`, `invig.command.event` via IPC to `hub-supervisor` and `hub-tui`.
- **Requests:** `fsm.snapshot.request` from `hub-supervisor` for status feed data.
- **Never writes to:** `invig_codes`, `exam_sessions`, `pen_bindings` (authoritative), stroke files, or any server-side state.
- **Provisional bindings:** `manual_register` commands create local provisional pen bindings for display only. `svc-exam-orch` is the single writable owner of authoritative bindings.

## Architecture

```
src/
  main.py            Entry point: component wiring, 1 Hz status feed loop
  config.py          Tunables: BLE name, socket paths, lockout policy, GATT UUIDs
  auth_handler.py    Rotating 24h code validation, per-address lockout (5 failures / 5 min)
  command_handler.py Command parsing (wire format), auth-gate, per-command validation
  status_feed.py     IPC snapshot -> BLE status JSON formatter (ble-gatt-spec.md Section 5)
  ipc_handlers.py    Outbound IPC events: auth state, command relay, snapshot requests
  peripheral.py      BLE GATT server abstraction (backend protocol for bless/bluez-peripheral)
tests/
  test_auth_handler.py   U-INVIG-AUTH-01..06 -- code validation, lockout, expiry
  test_command_handler.py U-INVIG-CMD-01..06 -- parsing, auth gate, manual register
  test_status_feed.py    U-INVIG-SF-01..05  -- JSON schema, IPC update, compact output
```

## Running

```bash
# From the hub-invig-ble directory
python -m src.main
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Dependencies

- Python 3.12 (stdlib only -- no third-party runtime deps)
- `pytest`, `pytest-asyncio` for dev/test
- `hub-common` (not yet packaged -- local IPC envelope defined in `ipc_handlers.py`)
- BLE backend (`bless` or `bluez-peripheral`) injected at runtime; all tests use mocks

## Key Design Decisions

1. **Backend protocol pattern** -- `BlePeripheralBackend` is a Protocol class that the real BLE library implements. All domain logic is testable without BLE hardware.
2. **Read-only code store** -- Auth codes are generated server-side, cached in hub SQLite during provisioning. This module never writes to `invig_codes`.
3. **Per-address lockout** -- 5 consecutive failures lock out a BLE address for 5 minutes (S3 mitigation from FAILURE_MITIGATION_REGISTER).
4. **Provisional bindings only** -- `manual_register` creates local provisional records; `svc-exam-orch` is the authoritative owner (HUB_DEPLOYMENT_SPEC Section 4.3).
5. **1 Hz status feed** -- Matches the BLE notify cadence in ble-gatt-spec.md Section 5.
