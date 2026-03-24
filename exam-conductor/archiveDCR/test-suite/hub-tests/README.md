# Hub Hardware-in-Loop Tests (W6.A4 / L6)

Hardware-in-loop test suite for the ExamPen RPi hub, verifying BLE dongle
management, pen sync, dual-write integrity, timer accuracy, WiFi connectivity,
invigilator BLE relay, and power failure recovery.

## Validation Level

All tests in this directory are **L6 (hardware-in-loop)** per
`TEST_SUITE_SPEC.md` section 2.4. They require physical hardware or a hardware
simulator to pass.

## Required Equipment

| Item | Qty | Purpose |
|---|---|---|
| Raspberry Pi 4B or 5 | 1 | Hub under test (running ExamPen golden image) |
| USB BLE dongles | 5 | BLE adapter pool for pen connections |
| nRF52840-DK (or real P05 pens) | 1-8 | BLE pen simulator / real pens |
| USB flash drive | 1 | Secondary write target for dual-write |
| SD card (16GB+) | 1 | Primary storage |
| WiFi access point | 1 | For WiFi connectivity tests |
| Mobile phone with ExamPen app | 1 | For invigilator BLE relay test (HW-I1) |
| NTP server (or internet) | 1 | For timer accuracy test (HW-T1) |
| Switchable power supply | 1 | For power failure recovery test (HW-P1) |

## Test Procedure

### 1. Prepare the Hub

```bash
# Flash golden image onto SD card
sudo dd if=exampen-hub.img of=/dev/sdX bs=4M status=progress

# Boot the RPi, connect via SSH
ssh exampen@<hub-ip>

# Verify all services are running
sudo systemctl status exampen-supervisor
```

### 2. Prepare Pen Simulators

```bash
# Option A: nRF52840-DK with pen-simulator firmware
# Flash the pen-sim firmware and configure MAC addresses

# Option B: Software BLE simulator (requires a second BLE adapter)
python test-suite/hub-tests/ble_pen_sim.py --pens 8 --adapter hci1
```

### 3. Run Tests

```bash
# Run all hardware tests (from the test host, NOT the hub)
pytest test-suite/hub-tests/ -m hardware -v

# Run only dongle tests
pytest test-suite/hub-tests/test_hw_h1_dongle_enum.py -v

# Run only BLE pen tests
pytest test-suite/hub-tests/test_hw_b1_ble_scan.py -v

# Skip tests when hardware is not detected (auto-skip)
pytest test-suite/hub-tests/ -m hardware -v
# Tests will auto-skip if fixtures detect missing hardware.
```

### 4. Export Results

Test results are written to `test-suite/hub-tests/results/` as JSON files
compatible with the Hub TUI diagnostics export format (TEST_SUITE_SPEC section
3.3).

## Test Matrix

| File | Test ID | What It Proves |
|---|---|---|
| `test_hw_h1_dongle_enum.py` | HW-H1 | All 5 dongles detected, stable MAC |
| `test_hw_h2_dongle_hotplug.py` | HW-H2 | Graceful degradation + recovery on unplug/replug |
| `test_hw_b1_ble_scan.py` | HW-B1 | Pen discovered via BLE, GATT service readable |
| `test_hw_b2_multi_pen_sync.py` | HW-B2 | Concurrent sync of 8 pens per dongle |
| `test_hw_b3_dual_write.py` | HW-B3 | SD + USB copies byte-identical |
| `test_hw_t1_timer_accuracy.py` | HW-T1 | 90-min timer drift < 1 second |
| `test_hw_w1_wifi.py` | HW-W1 | WiFi connect, band, backend reachability |
| `test_hw_i1_invigilator_ble.py` | HW-I1 | Mobile auth flow, command relay, status feed |
| `test_hw_p1_power_recovery.py` | HW-P1 | Timer resumes, partial data preserved, no corruption |

## Configuration

Environment variables for hub connection:

| Variable | Default | Description |
|---|---|---|
| `HUB_SSH_HOST` | `exampen-hub.local` | Hub SSH hostname or IP |
| `HUB_SSH_USER` | `exampen` | Hub SSH username |
| `HUB_SSH_KEY` | `~/.ssh/id_rsa` | SSH private key path |
| `HUB_DB_PATH` | `/var/lib/exampen/hub.db` | Hub SQLite database path |
| `PEN_SIM_ADAPTER` | `hci1` | BLE adapter for pen simulator |
| `PEN_SIM_COUNT` | `8` | Number of simulated pens |

## Architecture

Tests run on a **test host** (laptop/CI runner) and communicate with the hub
via SSH. The test host:

1. SSHs into the hub to run diagnostics commands and read state
2. Optionally controls the BLE pen simulator via a local BLE adapter
3. Uses `subprocess` to invoke hub CLI tools and read SQLite
4. Monitors IPC messages by tailing the hub interaction log

The `conftest.py` provides fixtures for SSH connections, hub database access,
BLE dongle detection, and pen simulator setup.
