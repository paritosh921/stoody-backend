# hub-ble-mgr

BLE dongle management module for the ExamPen hub.

**Owner:** hub-ble-mgr team (Agent 2 build track)

## Responsibility

Manages up to 5 USB BLE dongles, each supporting 8 concurrent pen connections (40 pens total per hub). Provides pen discovery, connection lifecycle management, dongle health monitoring, and IPC integration with hub-supervisor and hub-pen-sync.

## Architecture

```
src/
  config.py             # Constants: capacity limits, timeouts, GATT UUID
  dongle_manager.py     # Dongle state tracking, health FSM, pen allocation
  pen_discovery.py      # BLE scan orchestration with staggered activation
  connection_manager.py # Pen connect/disconnect lifecycle, failure re-queue
  health_monitor.py     # Periodic health checks, hot-unplug detection
  ipc_handlers.py       # IPC message handlers (ble.scan.*, ble.connect.*, health)
  main.py               # Entry point: wires components, starts IPC server
tests/
  test_dongle_manager.py
  test_pen_discovery.py
  test_connection_manager.py
  test_health_monitor.py
```

## Key Design Decisions

- **Domain/adapter separation:** All domain logic (dongle state machine, pen tracking, health FSM) is testable without BLE hardware. BLE operations are injected via Protocol classes (`BleAdapter`, `BleScanner`, `BleConnector`, `HealthProbe`).
- **Staggered scan activation (H5):** 500ms delay between dongle scan starts to reduce RF collision.
- **Overflow redirection (A1.1):** When a dongle is full (8 pens), new connections are automatically redirected to the next available dongle.
- **Dongle failure re-queue (H3):** When a dongle fails, its pens are re-queued to remaining dongles.
- **Health FSM:** `unknown -> healthy <-> degraded -> failed`, with recovery path from `failed -> healthy` on re-plug.

## IPC Messages

| Message | Direction | Handler |
|---------|-----------|---------|
| `ble.scan.start.request` | Inbound | `handle_scan_start` |
| `ble.scan.stop.request` | Inbound | `handle_scan_stop` |
| `ble.connect.request` | Inbound | `handle_connect` |
| `supervisor.health.request` | Inbound | `handle_health` |
| `ble.scan.result.event` | Outbound | Emitted per discovered pen |
| `ble.dongle.health.event` | Outbound | Emitted on status change |
| `ble.pen.connected` | Outbound | Emitted on successful connection |
| `ble.pen.disconnected` | Outbound | Emitted on disconnect |

## Testing

```bash
# From hub-ble-mgr directory
pip install -e ".[dev]"
pytest tests/ -v
```

All tests are L3 (unit, no I/O). BLE operations are fully mocked.

## Dependencies

- `bleak>=0.22` (BLE adapter layer, mocked in tests)
- `hub-common` (IPC protocol, with local fallback for standalone development)
