# hub-supervisor

Process manager and FSM orchestrator for the ExamPen hub (Raspberry Pi).

## Ownership Declaration

- **Writes:** Hub FSM state (`exam_sessions.state` in SQLite), `interaction_log` (append-only audit trail)
- **Reads from:** All child modules via IPC (health, status), `hub_config` table, `active_timer` table
- **Never writes to:** Stroke data (owned by `hub-store`), pen sync status (owned by `hub-pen-sync`), pen bindings (authoritative owner is `svc-exam-orch`)
- **Transactional boundaries:** FSM state persisted to SQLite BEFORE side effects execute (crash-safe)

## Architecture

```
hub-supervisor
├── src/
│   ├── main.py              # Entry point — config, IPC server, spawn children, FSM loop
│   ├── config.py            # Paths, socket paths, tunables
│   ├── hub_fsm.py           # ZERO I/O — pure FSM logic
│   ├── process_manager.py   # Child module lifecycle (spawn, stop, crash restart, watchdog)
│   ├── orchestrator.py      # FSM side-effect executor (IPC to children)
│   ├── interaction_log.py   # Forensic audit logger (append-only SQLite)
│   ├── first_boot.py        # First-boot detection and provisioning
│   └── ipc_handlers.py      # IPC message handlers (transition, snapshot, shutdown)
└── tests/
    ├── test_hub_fsm.py          # All valid/invalid transitions, reachability
    ├── test_process_manager.py  # Spawn, stop, crash restart, watchdog (mock subprocesses)
    ├── test_orchestrator.py     # Side effects trigger correct IPC (mock IPC)
    └── test_interaction_log.py  # Append-only log entries, schema validation
```

## Hub FSM States

```
created -> armed -> timer_running -> dongle_activation -> pen_sync
  -> sync_complete | sync_partial -> uploading -> upload_complete
Any non-terminal state -> cancelled
```

Matches `HUB_DEPLOYMENT_SPEC.md` Section 3.1 `exam_sessions.state` CHECK constraint.

## Child Modules Managed

| Module | Socket | Required |
|--------|--------|----------|
| hub-ble-mgr | `/run/exampen/ble-mgr.sock` | Yes |
| hub-pen-sync | `/run/exampen/pen-sync.sock` | Yes |
| hub-timer | `/run/exampen/timer.sock` | Yes |
| hub-store | `/run/exampen/store.sock` | Yes |
| hub-uplink | `/run/exampen/uplink.sock` | Yes |
| hub-invig-ble | `/run/exampen/invig-ble.sock` | Yes |
| hub-tui | `/run/exampen/tui.sock` | No (optional) |

## Running Tests

```bash
cd hub/hub-supervisor
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ../hub-common
pip install -e ".[dev]"
pytest
```

## Validation Levels

- **L3**: Unit tests verified (domain logic, no I/O) — `test_hub_fsm.py`
- **L3**: Unit tests verified (mocked I/O) — `test_process_manager.py`, `test_orchestrator.py`, `test_interaction_log.py`
